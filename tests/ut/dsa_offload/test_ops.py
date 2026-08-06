# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    _normalize_lidu_weights_layout,
    lightning_indexer_decode_update_quant,
)

import pytest


def test_lidu_quant_variant_raises_until_operator_lands() -> None:
    outputs = DSALightningIndexerOutputs(
        topk_index=torch.zeros((1, 1, 16), dtype=torch.int32),
        topk_slots=torch.zeros((1, 1, 16), dtype=torch.int32),
        miss_count=torch.zeros((1,), dtype=torch.int32),
        tail_info=torch.zeros((1, 2), dtype=torch.int32),
    )
    with pytest.raises(NotImplementedError, match="quantized LIDU"):
        lightning_indexer_decode_update_quant(
            query=torch.zeros((1, 32, 128), dtype=torch.int8),
            query_scale=torch.zeros((1,), dtype=torch.float16),
            key=torch.zeros((2, 4, 128), dtype=torch.int8),
            key_scale=torch.zeros((2, 4, 1), dtype=torch.float16),
            weights=torch.zeros((1, 32), dtype=torch.bfloat16),
            req_pool_entries=torch.zeros((1,), dtype=torch.int32),
            cache_slots=torch.zeros((2, 16), dtype=torch.int32),
            row_modes=torch.zeros((1,), dtype=torch.int32),
            actual_seq_lengths_key=torch.ones((1,), dtype=torch.int32),
            block_table=torch.zeros((1, 4), dtype=torch.int32),
            outputs=outputs,
        )


def test_lidu_weights_normalizes_fused_projection_suffix_view() -> None:
    fused_projection = torch.arange(
        4 * 192,
        dtype=torch.bfloat16,
    ).view(4, 192)
    weights = fused_projection[:, 128:]

    assert weights.shape == (4, 64)
    assert weights.stride() == (192, 1)
    assert not weights.is_contiguous()

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.shape == weights.shape
    assert normalized.stride() == (64, 1)
    assert normalized.is_contiguous()
    torch.testing.assert_close(normalized, weights)


def test_lidu_weights_keeps_already_contiguous_storage() -> None:
    weights = torch.empty((1, 64), dtype=torch.bfloat16)

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.is_contiguous()
    assert normalized.data_ptr() == weights.data_ptr()
