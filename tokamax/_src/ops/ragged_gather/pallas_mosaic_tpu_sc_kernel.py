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
"""Optimized TPU Pallas/Mosaic kernel for Ragged Gather and Paged Block Gather."""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


def _tc_paged_block_gather_kernel(
    pt_ref,
    len_ref,
    *args,
    block: int,
    p: int,
    K: int,
    has_lengths: bool,
):
  c_refs = args[:K]
  out_ref = args[K]

  i = pl.program_id(0)
  j = pl.program_id(1)
  len_i = len_ref[i]
  rem = len_i - j * (K * block)

  token_idx = jnp.arange(block, dtype=jnp.int32).reshape(block, 1)

  for k in range(K):
    page_val = c_refs[k][0]
    if has_lengths:
      val = jnp.clip(rem - k * block, 0, block)
      mask = token_idx < val
      page_val = jnp.where(mask, page_val, 0).astype(out_ref.dtype)
    out_ref[0, pl.ds(k * block, block), :] = page_val


def _make_tc_cache_index_map(k: int, pages: int, p: int, K: int):
  def index_map(i, j, pt_ref, len_ref):
    idx = i * p + K * j + k
    safe_page = jnp.clip(pt_ref[idx], 0, pages - 1)
    return (safe_page, 0, 0)

  return index_map


def _tc_out_index_map(i, j, pt_ref, len_ref):
  return (i, j, 0)


@functools.partial(
    jax.jit,
    static_argnames=("page_size", "pages_per_Step"),
)
def paged_block_gather_tc(
    x: jax.Array,
    page_table: jax.Array,
    lengths: jax.Array | None = None,
    *,
    page_size: int = 16,
    pages_per_Step: int = 16,
) -> jax.Array:
  """TensorCore DMA block-gather fast path for contiguous (page_size, head_dim) pages."""
  orig_x_shape = x.shape
  pages = orig_x_shape[0]
  if x.ndim == 2:
    assert x.shape[1] % page_size == 0
    hd = x.shape[1] // page_size
    cache_3d = x.reshape(pages, page_size, hd)
  else:
    page_size = orig_x_shape[1]
    hd = int(functools.reduce(lambda a, b: a * b, orig_x_shape[2:], 1))
    cache_3d = x.reshape(pages, page_size, hd)

  if page_table.ndim == 1:
    b, p = 1, page_table.shape[0]
  else:
    b, p = page_table.shape

  K = min(pages_per_Step, p)
  while p % K != 0 and K > 1:
    K //= 2

  flat_pt = page_table.reshape(-1).astype(jnp.int32)
  has_lengths = lengths is not None
  if lengths is None:
    lengths_arr = jnp.full((b,), p * page_size, dtype=jnp.int32)
  else:
    lengths_arr = lengths.reshape(b).astype(jnp.int32)

  grid = (b, p // K)
  in_specs = [
      pl.BlockSpec(
          (1, page_size, hd), _make_tc_cache_index_map(k, pages, p, K)
      )
      for k in range(K)
  ]
  out_specs = pl.BlockSpec((1, K * page_size, hd), _tc_out_index_map)

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
  )

  kernel_fn = functools.partial(
      _tc_paged_block_gather_kernel,
      block=page_size,
      p=p,
      K=K,
      has_lengths=has_lengths,
  )

  out = pl.pallas_call(
      kernel_fn,
      grid_spec=grid_spec,
      out_shape=jax.ShapeDtypeStruct((b, p * page_size, hd), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
  )(flat_pt, lengths_arr, *([cache_3d] * K))

  if x.ndim == 2 and page_table.ndim == 1:
    return out.reshape(p, page_size * hd)
  return out.reshape(b, p * page_size, *orig_x_shape[2:])


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

    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

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


@functools.partial(jax.jit, static_argnames=("col_size", "page_size"))
def ragged_gather_pallas(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array | None = None,
    end: jax.Array | None = None,
    *,
    col_size: int | None = None,
    page_size: int | None = None,
    lengths: jax.Array | None = None,
) -> jax.Array:
  """Perform gather on indices, using TensorCore DMA block gather for contiguous page blocks."""
  if x.ndim >= 3 or page_size is not None or lengths is not None:
    return paged_block_gather_tc(
        x,
        indices,
        lengths=lengths,
        page_size=page_size or (x.shape[1] if x.ndim >= 3 else 16),
    )

  # Fast path for flattened contiguous page blocks (e.g., (pages, block * h * d))
  # where row width is a multiple of (16 * 128) = 2048 and TensorCore DMA block gather
  # avoids SparseCore 2x bitcast read amplification and bitwise unpacking overhead.
  if (
      x.ndim == 2
      and indices.ndim == 1
      and start is None
      and end is None
      and x.shape[1] >= 2048
      and x.shape[1] % (16 * 128) == 0
  ):
    return paged_block_gather_tc(x, indices, page_size=16)

  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  if start is None:
    start = jnp.array([0], jnp.int32)
  if end is None:
    end = jnp.array([indices.size], jnp.int32)
  if jnp.isscalar(start):
    start = start[None]
  if jnp.isscalar(end):
    end = end[None]

  dtype = x.dtype

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
