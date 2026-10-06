# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimized TPU Pallas/Mosaic SparseCore kernel for Ragged & Indexed Gather."""

import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


@functools.lru_cache(maxsize=64)
def _build_vector_indexed_gather_kernel(
    l_work: int,
    d_padded: int,
    b_size: int,
    num_workers: int,
    num_subcores: int,
    dtype: jnp.dtype,
):
  """Builds a zero-copy vector indirect DMA SparseCore kernel for 32-bit rows."""
  mesh = plsc.VectorSubcoreMesh(
      num_cores=num_workers // num_subcores,
      num_subcores=num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  items_per_subcore = l_work // num_workers
  num_steps = items_per_subcore // b_size

  def sc_kernel(table_hbm, index_hbm, out_hbm, idx_vmem, rows_vmem):
    wid = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    subcore_start = wid * items_per_subcore

    if num_steps <= 4:
      for step in range(num_steps):
        offset = subcore_start + step * b_size
        pltpu.sync_copy(index_hbm.at[pl.ds(offset, b_size)], idx_vmem)
        pltpu.sync_copy(table_hbm.at[idx_vmem], rows_vmem)
        pltpu.sync_copy(rows_vmem, out_hbm.at[pl.ds(offset, b_size), :])
    else:

      def step_fn(step, _):
        offset = subcore_start + step * b_size
        pltpu.sync_copy(index_hbm.at[pl.ds(offset, b_size)], idx_vmem)
        pltpu.sync_copy(table_hbm.at[idx_vmem], rows_vmem)
        pltpu.sync_copy(rows_vmem, out_hbm.at[pl.ds(offset, b_size), :])
        return None

      lax.fori_loop(0, num_steps, step_fn, None)

  return pl.kernel(
      sc_kernel,
      out_type=jax.ShapeDtypeStruct((l_work, d_padded), dtype),
      mesh=mesh,
      scratch_types=[
          pltpu.VMEM((b_size,), jnp.int32),
          pltpu.VMEM((b_size, d_padded), dtype),
      ],
      compiler_params=pltpu.CompilerParams(
          needs_layout_passes=False,
          disable_bounds_checks=True,
      ),
      name="sc_vector_indexed_gather",
  )


def indexed_gather_pallas(
    x: jax.Array,
    indices: jax.Array,
    *,
    block_rows: int | None = None,
) -> jax.Array:
  """Full-range SparseCore vector indirect gather with ~128 KiB VMEM tiles."""
  assert x.ndim == 2, "Indexed gather only supports 2d inputs."
  assert indices.ndim == 1, "Indexed gather only supports 1d indices."

  orig_l = indices.shape[0]
  orig_d = x.shape[1]
  dtype = x.dtype
  dtype_bits = jax.dtypes.itemsize_bits(dtype)

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None or (orig_l <= 4096 and orig_d <= 128):
    d_num = lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    return lax.gather(
        x,
        indices[:, None],
        dimension_numbers=d_num,
        slice_sizes=(1, orig_d),
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

  if dtype_bits != 32:
    return ragged_gather_pallas(
        x,
        indices,
        jnp.array([0], jnp.int32),
        jnp.array([orig_l], jnp.int32),
        use_vector_dma=False,
    )

  num_cores = int(sc_info.num_cores)
  num_subcores = int(sc_info.num_subcores)
  num_workers = num_cores * num_subcores

  pad_d = (-orig_d) % 128
  if pad_d > 0:
    x_in = jnp.pad(x, ((0, 0), (0, pad_d)))
    d_padded = orig_d + pad_d
  else:
    x_in = x
    d_padded = orig_d

  # Size rows_vmem to ~128 KiB (50% of 256 KiB subcore VMEM) to maximize DMA burst
  b_cfg = block_rows if block_rows is not None else (128 if d_padded >= 256 else 256)
  max_b_vmem = (128 * 1024) // (d_padded * 4)
  b_size = max(8, min(b_cfg, max_b_vmem, max(32, orig_l // num_workers)))

  tile_quantum = num_workers * b_size
  pad_l = (-orig_l) % tile_quantum
  if pad_l > 0:
    index_in = jnp.pad(indices, (0, pad_l), constant_values=0)
    l_work = orig_l + pad_l
  else:
    index_in = indices
    l_work = orig_l

  items_per_subcore = l_work // num_workers
  while items_per_subcore % b_size != 0 and b_size > 8:
    b_size //= 2

  kernel = _build_vector_indexed_gather_kernel(
      l_work, d_padded, b_size, num_workers, num_subcores, dtype
  )
  out = kernel(x_in, index_in)
  if pad_l > 0 or pad_d > 0:
    out = out[:orig_l, :orig_d]
  return out


def main_kernel(
    # Inputs.
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    indices_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
):
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  hidden_size = in_hbm_ref.shape[-1]
  col_size = out_vmem_ref.shape[-1]

  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  block_size = num_simd_lanes * num_cores

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  # Read start and end tensor values.
  dma_list = []
  dma = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)
  dma = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)

  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)

  # Calculate number of tiles to visit using start and end arrays.
  start = start_vmem_ref[...][0]
  end = end_vmem_ref[...][0]

  block_start = start // block_size
  block_end = pl.cdiv(end, block_size)
  num_blocks = block_end - block_start
  num_blocks = jnp.where(end == start, 0, num_blocks)
  aligned_start = block_start * block_size

  num_cols = pl.cdiv(hidden_size, col_size)

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_blocks, num_cores, num_cols),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
  )
  def inner_kernel():
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)
    col_id = pl.program_id(2)

    row_tile_start = (
        aligned_start + block_id * block_size + core_id * num_simd_lanes
    )
    col_tile_start = col_id * col_size

    @pl.when(col_id == 0)
    def _():
      pltpu.sync_copy(
          indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
          indices_vmem_ref,
      )

    # HBM to VMEM transfer.
    indices = indices_vmem_ref[...]

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)  # pyrefly: ignore[missing-attribute]
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)  # pyrefly: ignore[missing-attribute]

    for col_vmem_start in range(0, col_size, num_lanes):
      col_hbm_start = pl.multiple_of(col_tile_start + col_vmem_start, num_lanes)
      for row_vmem in range(num_simd_lanes):
        row_hbm = indices[row_vmem] // packing
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            recv_sem,
        ).start()

    # VMEM to HBM transfer.
    @pl.loop(0, col_size, step=num_lanes)
    @jax.named_scope("dma_write_loop")
    def dma_write_loop(col_vmem_start):
      col_hbm_start = col_tile_start + col_vmem_start

      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[0, :num_lanes],
            out_vmem_ref.at[0, :num_lanes],
            recv_sem,
        ).wait()

      if packing > 1:
        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)

          out = None
          for row_src in range(num_simd_lanes):
            row_src_pack = indices[row_src] % packing
            row_dst_pack = row_src % packing

            rightshift_bits = row_src_pack * dtype_bits
            leftshift_bits = row_dst_pack * dtype_bits

            data = out_vmem_ref[row_src, col_slice]
            data = jnp.bitwise_right_shift(data, rightshift_bits)
            data = jnp.bitwise_and(data, 2**dtype_bits - 1)
            data = jnp.bitwise_left_shift(data, leftshift_bits)

            if row_dst_pack == 0:
              out = data
            else:
              assert out is not None
              out = jnp.bitwise_or(out, data)

            if row_dst_pack == packing - 1:
              row_dst = row_src // packing
              out_vmem_ref[row_dst, col_slice] = out
              out = None

      for row_vmem in range(num_simd_lanes // packing):
        row_hbm = row_tile_start // packing + row_vmem
        pltpu.make_async_copy(
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            out_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            send_sem,
        ).start()

    for _ in range(0, col_size, num_lanes):
      for _ in range(num_simd_lanes // packing):
        pltpu.make_async_copy(
            out_vmem_ref.at[0, :num_lanes],
            out_32b_hbm_ref.at[0, :num_lanes],
            send_sem,
        ).wait()

  inner_kernel()


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.generation:
    case 6:
      target_bytes = (256 * 1024) * 0.8
    case 7:
      target_bytes = (512 * 1024) * 0.8
    case _:
      target_bytes = (128 * 1024) * 0.8

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


@functools.partial(
    jax.jit,
    static_argnames=("col_size", "block_rows", "use_vector_dma"),
)
def ragged_gather_pallas(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | None = None,
    end: jax.Array | None = None,
    *,
    col_size: int | None = None,
    block_rows: int | None = None,
    use_vector_dma: bool = True,
) -> jax.Array:
  """Perform gather on indices within dynamic array start and end."""

  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  dtype = x.dtype
  dtype_bits = jax.dtypes.itemsize_bits(dtype)

  if use_vector_dma and dtype_bits == 32 and (start is None or end is None):
    return indexed_gather_pallas(x, indices, block_rows=block_rows)

  if start is None:
    start = jnp.array([0], jnp.int32)
  elif jnp.isscalar(start):
    start = start[None]
  if end is None:
    end = jnp.array([indices.size], jnp.int32)
  elif jnp.isscalar(end):
    end = end[None]

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return x[indices]

  hidden_size = x.shape[-1]
  out_size = indices.size

  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  max_col_size = calculate_col_size(hidden_size)
  if col_size is None:
    col_size = max_col_size
  else:
    col_size = max(128, (min(col_size, hidden_size, max_col_size) // 128) * 128)

  out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
  if out_pad_size > 0:
    indices = jnp.pad(indices, ((0, out_pad_size)))

  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  return pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
      ),
      out_type=jax.ShapeDtypeStruct(
          (out_size + out_pad_size, aligned_hidden_size), dtype
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      scratch_types=dict(
          start_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          end_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
          indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
          sem_ref=pltpu.SemaphoreType.DMA((2,)),
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather",
  )(start, end, x, indices)[:out_size, :hidden_size]
