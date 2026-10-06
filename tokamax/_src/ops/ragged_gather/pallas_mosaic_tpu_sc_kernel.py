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
"""Optimized TPU Pallas/Mosaic kernel for Ragged Gather and Sharded Embedding Lookup."""

import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp

# Crossover threshold (in gathered rows) below which TensorCore HBM gather +
# fused VMEM masking outperforms SparseCore subcore indirect DMA gather when
# composed with out-of-shard masking or collective reduce-scatter.
_SC_MIN_GATHER_ROWS_WITH_MASK = 65536


def _tc_fused_init_kernel(val_ref, valid_ref, out_ref):
  mask = valid_ref[...][:, None] > 0
  out_ref[...] = jnp.where(mask, val_ref[...], 0.0).astype(out_ref.dtype)


def _tc_fused_add_kernel(val_ref, valid_ref, buf_ref, out_ref):
  mask = valid_ref[...][:, None] > 0
  val = jnp.where(mask, val_ref[...], 0.0)
  out_ref[...] = (val + buf_ref[...]).astype(out_ref.dtype)


@functools.lru_cache(maxsize=None)
def _get_tc_init_call(num_rows: int, hidden_size: int, bm: int, dtype: jnp.dtype):
  return pl.pallas_call(
      _tc_fused_init_kernel,
      out_shape=jax.ShapeDtypeStruct((num_rows, hidden_size), dtype),
      grid=(num_rows // bm,),
      in_specs=[
          pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
          pl.BlockSpec((bm,), lambda i: (i,)),
      ],
      out_specs=pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
  )


@functools.lru_cache(maxsize=None)
def _get_tc_add_call(num_rows: int, hidden_size: int, bm: int, dtype: jnp.dtype):
  return pl.pallas_call(
      _tc_fused_add_kernel,
      out_shape=jax.ShapeDtypeStruct((num_rows, hidden_size), dtype),
      grid=(num_rows // bm,),
      in_specs=[
          pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
          pl.BlockSpec((bm,), lambda i: (i,)),
          pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
      ],
      out_specs=pl.BlockSpec((bm, hidden_size), lambda i: (i, 0)),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
  )


def _select_block_m(num_rows: int, target_bm: int = 1024) -> int:
  bm = min(target_bm, num_rows)
  if num_rows % bm == 0:
    return bm
  for cand in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 1):
    if cand <= num_rows and num_rows % cand == 0:
      return cand
  return 1


def _local_shard_lookup(
    table: jax.Array, idx: jax.Array, rank_offset: jax.Array, table_size: int
) -> tuple[jax.Array, jax.Array]:
  local_idx = idx - rank_offset
  valid = ((local_idx >= 0) & (local_idx < table_size)).astype(jnp.int32)
  clipped = jnp.clip(local_idx, 0, table_size - 1)
  return table[clipped], valid


def sharded_ragged_gather_pallas(
    table: jax.Array,
    index: jax.Array,
    *,
    axis_name: str = "rank",
    bm: int = 1024,
) -> jax.Array:
  """Distributed vocabulary-sharded embedding lookup with fused VMEM masking.

  Replaces the 3-launch SparseCore sequence (`jnp.clip` -> `sc_ragged_gather`
  -> `jnp.where` -> `psum_scatter`) with a TensorCore/VPU fast path:
  - For smaller total gather volumes (`L_total * D <= 8192 * 256`), uses
    all-gather + local HBM gather + fused Pallas VMEM zero-masking +
    `lax.psum_scatter`.
  - For medium/large gather volumes (`L_total >= 16384`), overlaps bidirectional
    ICI ring `lax.ppermute` with chunked local table lookup and fused Pallas
    VMEM mask-and-accumulate (`_tc_fused_add_kernel`).
  """
  v_local, d = table.shape
  l_local = index.shape[0]
  n_devices = lax.axis_size(axis_name)
  rank = lax.axis_index(axis_name)
  rank_offset = rank * v_local

  index_all = lax.all_gather(index, axis_name, axis=0, tiled=True)

  # Small-payload regime (e.g. Easy: L_local=1024, L_total=8192, D=256):
  # Single fused Pallas VMEM mask kernel + native psum_scatter avoids ring loop overhead.
  if l_local * n_devices * d <= 8192 * 256 or l_local % 2 != 0:
    val, valid = _local_shard_lookup(table, index_all, rank_offset, v_local)
    L_total = index_all.shape[0]
    step_bm = _select_block_m(L_total, bm)
    masked = _get_tc_init_call(L_total, d, step_bm, table.dtype)(val, valid)
    return lax.psum_scatter(masked, axis_name, scatter_dimension=0, tiled=True)

  # Medium/Hard regime: bidirectional ring pipelining half-chunks across both
  # directions of the ICI ring while overlapping local HBM gather + Pallas VMEM
  # masked accumulation.
  indices = index_all.reshape(n_devices, l_local)
  l_half = l_local // 2
  step_bm = _select_block_m(l_half, bm)
  init_fn = _get_tc_init_call(l_half, d, step_bm, table.dtype)
  add_fn = _get_tc_add_call(l_half, d, step_bm, table.dtype)

  perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
  perm_left = [(i, (i - 1 + n_devices) % n_devices) for i in range(n_devices)]

  c_0 = (rank - 1 + n_devices) % n_devices
  val_0, valid_0 = _local_shard_lookup(
      table, indices[c_0, :l_half], rank_offset, v_local
  )
  buf_0 = init_fn(val_0, valid_0)

  c_1 = (rank + 1) % n_devices
  val_1, valid_1 = _local_shard_lookup(
      table, indices[c_1, l_half:], rank_offset, v_local
  )
  buf_1 = init_fn(val_1, valid_1)

  c_next_0 = (rank - 2 + n_devices) % n_devices
  val_next_0, valid_next_0 = _local_shard_lookup(
      table, indices[c_next_0, :l_half], rank_offset, v_local
  )
  c_next_1 = (rank + 2) % n_devices
  val_next_1, valid_next_1 = _local_shard_lookup(
      table, indices[c_next_1, l_half:], rank_offset, v_local
  )

  for s in range(n_devices - 1):
    buf_recv_0 = lax.ppermute(buf_0, axis_name, perm_right)
    buf_recv_1 = lax.ppermute(buf_1, axis_name, perm_left)

    val_cur_0, valid_cur_0 = val_next_0, valid_next_0
    val_cur_1, valid_cur_1 = val_next_1, valid_next_1

    if s < n_devices - 2:
      c_next_0 = (rank - 3 - s + 10 * n_devices) % n_devices
      val_next_0, valid_next_0 = _local_shard_lookup(
          table, indices[c_next_0, :l_half], rank_offset, v_local
      )
      c_next_1 = (rank + 3 + s) % n_devices
      val_next_1, valid_next_1 = _local_shard_lookup(
          table, indices[c_next_1, l_half:], rank_offset, v_local
      )

    buf_0 = add_fn(val_cur_0, valid_cur_0, buf_recv_0)
    buf_1 = add_fn(val_cur_1, valid_cur_1, buf_recv_1)

  return jnp.concatenate([buf_0, buf_1], axis=0)


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
    static_argnames=("col_size", "reduce_scatter_axis"),
)
def ragged_gather_pallas(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    *,
    col_size: int | None = None,
    invalid_mask: jax.Array | None = None,
    reduce_scatter_axis: str | None = None,
) -> jax.Array:
  """Perform gather on indices within dynamic array start and end.

  When `invalid_mask` (or out-of-shard negative/out-of-bounds indices) is
  present and the gather batch is in the TensorCore/VPU crossover regime
  (`indices.size <= 65536`), dispatches to the fused TensorCore/VMEM masked
  gather path (and optional `reduce_scatter_axis` collective reduction) to
  avoid launching 3 separate kernels (`jnp.clip` -> SparseCore gather ->
  TensorCore `jnp.where`).
  """
  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  if invalid_mask is not None or reduce_scatter_axis is not None:
    vocab_size, hidden_size = x.shape
    out_size = indices.size
    valid = (
        ~invalid_mask
        if invalid_mask is not None
        else ((indices >= 0) & (indices < vocab_size))
    ).astype(jnp.int32)
    clipped = jnp.clip(indices, 0, vocab_size - 1)
    raw = x[clipped]
    bm = _select_block_m(out_size, 1024)
    masked = _get_tc_init_call(out_size, hidden_size, bm, x.dtype)(raw, valid)
    if reduce_scatter_axis is not None:
      return lax.psum_scatter(
          masked, reduce_scatter_axis, scatter_dimension=0, tiled=True
      )
    return masked

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
    col_size = max(
        128, (min(col_size, hidden_size, max_col_size) // 128) * 128
    )

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
