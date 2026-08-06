# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面到 Ascend 自定义算子的薄适配层。

本模块只固定 LIDU、KSC、SFA-Offload 和满块复制的 tensor ABI。请求阶段、
resident 行分配、DRAM block 预留和 attention 时序由上层 runtime 负责；
这里不隐式创建输出 tensor，也不执行 D2H 或 list-to-tensor 转换。
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class DSALightningIndexerOutputs(NamedTuple):
    """LIDU 的四个 caller-owned 原址输出。"""

    topk_index: torch.Tensor
    topk_slots: torch.Tensor
    miss_count: torch.Tensor
    tail_info: torch.Tensor


class DSAOffloadSelectionOutput(NamedTuple):
    """LIDU/KSC 完成后交给 SFA-Offload 的逐行 resident 视图。"""

    sparse_indices: torch.Tensor
    tail_info: torch.Tensor


_REQUIRED_OPS = (
    "npu_lightning_indexer_decode_update_out",
    "npu_kvcache_scatter_copy",
    "npu_sparse_flash_attention_for_offload",
    "kv_cache_full_block_dump",
)


def require_dsa_offload_ops() -> None:
    """在 worker 初始化期确认四个设备算子均已部署。"""

    missing = [
        op_name
        for op_name in _REQUIRED_OPS
        if not hasattr(torch.ops._C_ascend, op_name)
    ]
    if missing:
        raise RuntimeError(
            "DSA sparse offload custom operators are not installed: "
            f"{tuple(missing)}"
        )


def _squeeze_cache_head_dim(
    cache: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if cache.ndim == 4 and cache.shape[2] == 1:
        return cache.squeeze(2)
    if cache.ndim == 3:
        return cache
    raise ValueError(
        f"{name} must be [blocks, block, 1, dim] or "
        f"[blocks, block, dim], got {tuple(cache.shape)}"
    )


def _normalize_lidu_weights_layout(weights: torch.Tensor) -> torch.Tensor:
    """将 v0.23 合并投影产生的 weights view 收敛到 LIDU 连续布局。

    v0.23 的 SFA 通过 ``wk_weights_proj`` 一次生成 key 与 weights，
    ``kw[:, head_dim:]`` 在 batch size 大于 1 时是带较大行 stride 的列后缀
    view。原生 lightning-indexer 能接收该 view，但 LIDU AscendC kernel 按
    紧凑 ``[B, N_idx]`` 布局寻址。这里只规范化这一个小输入；cache 和固定
    容量元数据仍必须由各自 owner 提供连续 tensor，避免在逐层热路径隐式
    复制大张量。
    """

    return weights.contiguous()


def lightning_indexer_decode_update(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    weights = _normalize_lidu_weights_layout(weights)
    torch.ops._C_ascend.npu_lightning_indexer_decode_update_out(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        row_modes,
        actual_seq_lengths_key,
        block_table,
        outputs.topk_index,
        outputs.topk_slots,
        outputs.miss_count,
        outputs.tail_info,
    )


def lightning_indexer_decode_update_quant(
    *,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    """C8 量化 Indexer cache 的 LIDU 变体（int8 K + fp16 scale，query 亦量化）。

    TODO(遗留事项): 接受 int8 key + fp16 key_scale + int8 query + query_scale
    的 AscendC quant LIDU 算子（目标名
    ``npu_lightning_indexer_decode_update_quant_out``）尚未实现，本接口仅为
    后续接入预留。当前 dense 路径的 ``npu_lightning_indexer_quant`` 不维护
    DSA 逐层 resident slot 状态，不能直接替代。
    """
    raise NotImplementedError(
        "DSA C8 Indexer offload requires a quantized LIDU AscendC operator "
        "(npu_lightning_indexer_decode_update_quant_out) that is not yet "
        "implemented. See DSA-offload-GLM5.2适配.md 遗留事项."
    )


def kvcache_scatter_copy(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    resident_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    src_token_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    torch.ops._C_ascend.npu_kvcache_scatter_copy(
        _squeeze_cache_head_dim(
            resident_rope_cache,
            name="resident_rope_cache",
        ),
        _squeeze_cache_head_dim(
            resident_nope_cache,
            name="resident_nope_cache",
        ),
        _squeeze_cache_head_dim(
            dram_rope_arena,
            name="dram_rope_arena",
        ),
        _squeeze_cache_head_dim(
            dram_nope_arena,
            name="dram_nope_arena",
        ),
        resident_block_table,
        dram_block_table,
        src_token_ids,
        dst_slots,
        copy_counts,
    )


def sparse_flash_attention_for_offload(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    tail_info: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_sparse_flash_attention_for_offload(
        query,
        key,
        key,
        sparse_indices,
        tail_info,
        float(scale_value),
        1,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        "TND",
        "PA_BSND",
        3,
    )


def dump_full_kv_cache_blocks(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    if src_block_ids.numel() == 0:
        return
    torch.ops._C_ascend.kv_cache_full_block_dump(
        _squeeze_cache_head_dim(
            resident_nope_cache,
            name="resident_nope_cache",
        ),
        _squeeze_cache_head_dim(
            resident_rope_cache,
            name="resident_rope_cache",
        ),
        _squeeze_cache_head_dim(
            dram_nope_arena,
            name="dram_nope_arena",
        ),
        _squeeze_cache_head_dim(
            dram_rope_arena,
            name="dram_rope_arena",
        ),
        src_block_ids,
        dst_block_ids,
    )
