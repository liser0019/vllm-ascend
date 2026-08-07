# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.dsa_offload.ops import _normalize_lidu_weights_layout


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
