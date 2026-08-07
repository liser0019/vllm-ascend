# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_offload.model_support import (
    get_dsa_offload_model_capabilities,
    require_dsa_offload_model_support,
)


def _model_config(
    architecture: str,
    *,
    use_mla: bool = True,
    index_topk: int = 2048,
    compress_ratios=None,
):
    hf_text_config = SimpleNamespace(
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    hf_config = SimpleNamespace()
    if compress_ratios is not None:
        hf_text_config.compress_ratios = compress_ratios
        hf_config.compress_ratios = compress_ratios
    return SimpleNamespace(
        architecture=architecture,
        use_mla=use_mla,
        hf_text_config=hf_text_config,
        hf_config=hf_config,
    )


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"],
)
def test_model_support_is_capability_based(architecture: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config(architecture))

    assert capabilities.supported
    assert capabilities.architecture == architecture


def test_unknown_architecture_with_required_capabilities_is_supported() -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config("FutureMlaIndexerForCausalLM"))

    assert capabilities.supported


@pytest.mark.parametrize(
    ("model_config", "missing"),
    [
        (_model_config("NoMla", use_mla=False), "MLA attention"),
        (_model_config("WrongTopK", index_topk=1024), "index_topk=2048"),
        (
            _model_config("CompressedDSA", compress_ratios=[2]),
            "sparse indexer",
        ),
    ],
)
def test_missing_capability_is_reported(model_config, missing: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(model_config)

    assert not capabilities.supported
    assert any(missing in item for item in capabilities.missing_requirements)
    with pytest.raises(ValueError, match=missing):
        require_dsa_offload_model_support(model_config)


def _glm52_model_config() -> SimpleNamespace:
    # 真实 GLM-5.2 拓扑（zai-org/GLM-5.2 config.json）：78 层 = 21 full + 57
    # shared，full 在 0,1,2 后每隔 4 层一个（6,10,...,74）。
    full_layers = {0, 1, 2, *range(6, 78, 4)}
    assert len(full_layers) == 21
    indexer_types = ["full" if i in full_layers else "shared" for i in range(78)]
    hf_text_config = SimpleNamespace(
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        indexer_types=indexer_types,
        index_topk_freq=4,
        num_hidden_layers=78,
    )
    return SimpleNamespace(
        architecture="GlmMoeDsaForCausalLM",
        use_mla=True,
        hf_text_config=hf_text_config,
        hf_config=SimpleNamespace(),
    )


def test_shared_indexer_topology_is_parsed() -> None:
    capabilities = get_dsa_offload_model_capabilities(_glm52_model_config())

    assert capabilities.supported  # 拓扑不影响 supported 判定
    assert capabilities.has_shared_indexer_layers
    assert capabilities.index_topk_freq == 4
    full = capabilities.full_indexer_layer_indices
    shared = capabilities.shared_indexer_layer_indices
    assert full is not None and shared is not None
    assert len(full) == 21
    assert len(shared) == 57
    assert full[:3] == (0, 1, 2)
    assert 6 in full and 74 in full and 3 not in full
    assert len(full) + len(shared) == 78


def test_all_full_topology_is_default_and_not_shared() -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config("DeepseekV32ForCausalLM"))

    assert capabilities.supported
    assert not capabilities.has_shared_indexer_layers
    assert capabilities.full_indexer_layer_indices is None
    assert capabilities.shared_indexer_layer_indices is None
    assert capabilities.index_topk_freq is None


def test_malformed_indexer_types_is_rejected() -> None:
    config = _model_config("GlmMoeDsaForCausalLM")
    config.hf_text_config.indexer_types = ["full", 128, "shared"]  # 非全字符串

    with pytest.raises(ValueError, match="entries must be strings"):
        get_dsa_offload_model_capabilities(config)


def test_unknown_indexer_type_is_rejected() -> None:
    config = _model_config("GlmMoeDsaForCausalLM")
    config.hf_text_config.indexer_types = ["full", "reuse"]

    with pytest.raises(ValueError, match="must be 'full' or 'shared'"):
        get_dsa_offload_model_capabilities(config)


def test_undeclared_topk_freq_topology_is_not_treated_as_shared() -> None:
    # DeepSeek IndexCache 风格：用 index_topk_freq（无 indexer_types）造 skip_topk
    # 层。未声明 indexer_types → 不识别为 shared 拓扑，spec/bind 门会拒绝其
    # 缺 indexer 的层（不进入共享复用路径）。
    config = _model_config("DeepseekV32ForCausalLM")
    config.hf_text_config.index_topk_freq = 4

    capabilities = get_dsa_offload_model_capabilities(config)

    assert capabilities.index_topk_freq == 4
    assert capabilities.indexer_types is None
    assert not capabilities.has_shared_indexer_layers
    assert capabilities.full_indexer_layer_indices is None
