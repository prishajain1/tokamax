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
"""optimized TPU Pallas/Mosaic kernel for Ragged Gather and Fused Embedding Bag Reduction."""

import functools
import math
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


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


def ragged_gather_reduce_pallas(
    x: jax.Array,
    indices: jax.Array,
    mask: jax.Array | None = None,
    *,
    reduce_group_size: int,
    col_size: int | None = None,
    row_chunk_size: int = 128,
) -> jax.Array:
  """Fused SparseCore gather + masked bag reduction in SC_VECTOR_SUBCORE VMEM.

  Avoids materializing the full `(B, J, D)` gathered tensor in HBM by
  accumulating `sub_group = gcd(reduce_group_size, sc_info.num_lanes // packing)`
  rows inside SparseCore vector registers via double-buffered `pltpu.emit_pipeline`.
  """
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    raise RuntimeError("This kernel requires TPU SparseCore.")

  flat_indices = indices.reshape(-1)
  m_orig = flat_indices.shape[0]
  vocab_size, hidden_size = x.shape
  if m_orig % reduce_group_size != 0:
    raise ValueError(
        f"indices.size={m_orig} must be divisible by reduce_group_size={reduce_group_size}"
    )
  m_final_out = m_orig // reduce_group_size

  packing = 32 // jax.dtypes.itemsize_bits(x.dtype)
  max_sc_group = sc_info.num_lanes // packing
  sub_group = math.gcd(reduce_group_size, max_sc_group)
  outer_group = reduce_group_size // sub_group

  num_subcores_total = sc_info.num_cores * sc_info.num_subcores
  num_lanes = sc_info.num_lanes

  if col_size is None:
    col_chunk = min(hidden_size, 2048)
  else:
    col_chunk = min(int(col_size), hidden_size, 2048)
  col_chunk = max(128, (col_chunk // 128) * 128)
  while col_chunk > 128 and hidden_size % col_chunk != 0:
    col_chunk -= 128

  min_rc = max(
      num_lanes,
      sub_group,
      ((m_orig + (num_subcores_total * 64) - 1) // (num_subcores_total * 64)),
  )
  min_rc = ((min_rc + num_lanes - 1) // num_lanes) * num_lanes
  rc = max(int(row_chunk_size), min_rc)
  rc = ((rc + num_lanes - 1) // num_lanes) * num_lanes
  max_rc = max(num_lanes, ((m_orig // num_subcores_total) // num_lanes) * num_lanes)
  rc = min(rc, max_rc)
  while rc > num_lanes and (m_orig // num_subcores_total) % rc != 0:
    rc -= num_lanes

  row_wave_size = rc * num_subcores_total
  weights = None if mask is None else mask.astype(jnp.float32).reshape(-1)

  pad_m = (row_wave_size - (m_orig % row_wave_size)) % row_wave_size
  if pad_m > 0:
    flat_indices = jnp.pad(flat_indices, ((0, pad_m),), constant_values=0)
    if weights is not None:
      weights = jnp.pad(weights, ((0, pad_m),), constant_values=0.0)

  flat_indices = jnp.clip(flat_indices, 0, vocab_size - 1)
  m_padded = flat_indices.shape[0]
  m_sc_out = m_padded // sub_group
  m_sc_out_unpadded = m_orig // sub_group

  @jax.jit
  @pl.kernel(
      out_type=jax.ShapeDtypeStruct((m_sc_out, hidden_size), x.dtype),
      mesh=plsc.VectorSubcoreMesh(
          core_axis_name="core",
          subcore_axis_name="subcore",
          num_cores=sc_info.num_cores,
          num_subcores=sc_info.num_subcores,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          needs_layout_passes=True,
      ),
      name="sc_ragged_gather_reduce",
  )
  def reduce_kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref):
    num_row_chunks = m_padded // row_wave_size
    num_col_chunks = hidden_size // col_chunk
    subcore_first_row_chunk = lax.axis_index(("core", "subcore")) * num_row_chunks

    in_spec = pl.BlockSpec((rc,), lambda i: (subcore_first_row_chunk + i,))
    in_specs = (in_spec,) * (1 + (weights_hbm_ref is not None))

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(num_row_chunks,),
        in_specs=in_specs,
    )
    def idx_pipeline(idx_ref, weights_ref=None):
      row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)
      row_subchunk_size = sc_info.num_lanes
      out_rows_per_step = row_subchunk_size // sub_group
      num_row_subchunks = rc // row_subchunk_size

      @functools.partial(
          pltpu.emit_pipeline,
          grid=(num_row_subchunks, num_col_chunks),
          in_specs=pl.BlockSpec(
              (pl.Indirect(row_subchunk_size), col_chunk),
              lambda r, c: (
                  lax.div(
                      idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                      packing,
                  ),
                  c,
              ),
          ),
          out_specs=pl.BlockSpec(
              (out_rows_per_step // packing, col_chunk),
              lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
          ),
      )
      def data_pipeline(gather_ref, out_ref):
        gather_ref = gather_ref.bitcast(x.dtype)
        out_ref = out_ref.bitcast(x.dtype)
        row_slice = pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        w_slice = (
            None
            if weights_ref is None
            else weights_ref[row_slice].astype(jnp.float32)
        )
        unpack_col_chunk = 32

        @plsc.parallel_loop(0, col_chunk, step=unpack_col_chunk)
        def _(col_base):
          accs = []
          for rg in range(out_rows_per_step):
            row_datas = []
            for r_in in range(sub_group):
              row = rg * sub_group + r_in
              row_data = gather_ref[
                  pl.ds(row * packing, packing),
                  pl.ds(col_base, unpack_col_chunk),
              ].astype(jnp.float32)
              if packing == 1:
                row_data = row_data[0]
              else:
                row_data = jnp.where(
                    lax.bitwise_and(subchunk_idxs[row], 1) == 0,
                    row_data[0],
                    row_data[1],
                )
              if w_slice is not None:
                row_data *= w_slice[row]
              row_datas.append(row_data)
            while len(row_datas) > 1:
              next_level = []
              for i in range(0, len(row_datas), 2):
                if i + 1 < len(row_datas):
                  next_level.append(row_datas[i] + row_datas[i + 1])
                else:
                  next_level.append(row_datas[i])
              row_datas = next_level
            accs.append(row_datas[0])
          out_ref[:, pl.ds(col_base, unpack_col_chunk)] = jnp.stack(
              accs, axis=0
          ).astype(x.dtype)

      data_pipeline(
          in_hbm_ref.bitcast(jnp.int32), out_hbm_ref.bitcast(jnp.int32)
      )

    idx_pipeline(
        idx_hbm_ref, *([weights_hbm_ref] if weights_hbm_ref is not None else [])
    )

  out_sc = reduce_kernel(x, flat_indices, weights)
  if pad_m > 0:
    out_sc = out_sc[:m_sc_out_unpadded, :]
  if outer_group > 1:
    out_sc = (
        out_sc.astype(jnp.float32)
        .reshape(m_final_out, outer_group, hidden_size)
        .sum(axis=1)
        .astype(x.dtype)
    )
  return out_sc


@functools.partial(
    jax.jit, static_argnames=("col_size", "reduce_group_size", "row_chunk_size")
)
def ragged_gather_pallas(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    *,
    col_size: int | None = None,
    mask: jax.Array | None = None,
    reduce_group_size: int | None = None,
    row_chunk_size: int = 128,
) -> jax.Array:
  """Perform gather on indices within dynamic array start and end, with optional fused bag reduction."""

  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  if reduce_group_size is not None:
    return ragged_gather_reduce_pallas(
        x,
        indices,
        mask=mask,
        reduce_group_size=reduce_group_size,
        col_size=col_size,
        row_chunk_size=row_chunk_size,
    )

  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  if jnp.isscalar(start):
    start = start[None]
  if jnp.isscalar(end):
    end = end[None]

  dtype = x.dtype

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    raise RuntimeError("This imported reference requires TPU SparseCore.")

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

  # Pad to align to the block size.
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
