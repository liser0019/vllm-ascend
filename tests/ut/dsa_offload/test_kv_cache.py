# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.dsa_offload import kv_cache as dsa_kv_cache
from vllm_ascend.dsa_offload.kv_cache import (
    DSAIndexerC8KVSpec,
    DSAIndexerKVSpec,
    DSAResidentMLAAttentionSpec,
    build_dsa_kv_cache_config,
    build_dsa_kv_cache_groups,
    dsa_max_memory_usage_bytes,
    dsa_pool_bytes_per_base_block,
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_binding_order,
    get_dsa_kv_cache_group_ids,
    is_dsa_indexer_spec,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSABlockPoolView,
    DSAKVCacheCoordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_groups,
)


def _make_specs(
    num_layers: int = 2,
) -> dict[str, DSAIndexerKVSpec | DSAResidentMLAAttentionSpec]:
    specs: dict[
        str,
        DSAIndexerKVSpec | DSAResidentMLAAttentionSpec,
    ] = {}
    for layer_idx in range(num_layers):
        specs[f"model.layers.{layer_idx}.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        specs[f"model.layers.{layer_idx}.self_attn.attn"] = DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
    return specs


def _make_vllm_config(
    *,
    max_model_len: int = 32768,
    override: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def test_indexer_spec_accounts_for_one_vector_per_token() -> None:
    spec = next(spec for spec in _make_specs(num_layers=1).values() if isinstance(spec, DSAIndexerKVSpec))

    assert spec.page_size_bytes == 128 * 1 * 128 * 2


def test_indexer_c8_spec_splits_k_and_scale_bytes() -> None:
    spec = DSAIndexerC8KVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    # int8 K（128*1*128*1）+ fp16 scale（128*1*1*2）。
    assert spec.real_page_size_bytes == 128 * 1 * 128 * 1 + 128 * 1 * 1 * 2
    # C8 spec 仍归入 Indexer plane（isinstance 匹配）。
    assert is_dsa_indexer_spec(spec)


def test_indexer_c8_spec_group_merge_preserves_type_and_size() -> None:
    # 全 C8 indexer 层 + 全 resident 层：组 merge 应保持 C8 类型与拆分容量。
    specs: dict[str, DSAIndexerC8KVSpec | DSAResidentMLAAttentionSpec] = {}
    for layer_idx in range(2):
        specs[f"model.layers.{layer_idx}.self_attn.indexer.k_cache"] = DSAIndexerC8KVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        specs[f"model.layers.{layer_idx}.self_attn.attn"] = DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )

    groups = build_dsa_kv_cache_groups(specs)

    assert isinstance(groups[0].kv_cache_spec, DSAIndexerC8KVSpec)
    assert groups[0].kv_cache_spec.real_page_size_bytes == (
        128 * 1 * 128 * 1 + 128 * 1 * 1 * 2
    )


def test_split_groups_keep_stable_plane_order() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())

    assert len(groups) == 2
    assert isinstance(groups[0].kv_cache_spec, DSAIndexerKVSpec)
    assert isinstance(
        groups[1].kv_cache_spec,
        DSAResidentMLAAttentionSpec,
    )
    assert len(groups[0].layer_names) == len(groups[1].layer_names) == 2


def test_engine_core_grouping_initializes_process_local_ascend_config(
    monkeypatch,
) -> None:
    initialized_with: list[object] = []
    vllm_config = object()
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.init_ascend_config",
        initialized_with.append,
    )

    groups = _ascend_get_kv_cache_groups(  # type: ignore[arg-type]
        vllm_config,
        _make_specs(),
    )

    assert initialized_with == [vllm_config]
    assert len(groups) == 2


def test_split_group_ids_follow_spec_identity_not_position() -> None:
    groups = list(reversed(build_dsa_kv_cache_groups(_make_specs())))
    config = SimpleNamespace(kv_cache_groups=groups)

    group_ids = get_dsa_kv_cache_group_ids(config)  # type: ignore[arg-type]

    assert group_ids.indexer == 1
    assert group_ids.resident_mla == 0


def test_split_cache_binding_orders_each_layer_resident_then_indexer() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.attn",
        "model.layers.1.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_rejects_duplicate_layer_name() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs(num_layers=1))
    groups[1].layer_names.append(groups[0].layer_names[0])
    config = SimpleNamespace(kv_cache_groups=groups)

    with pytest.raises(RuntimeError, match="more than one group"):
        get_dsa_kv_cache_binding_order(config)


def test_ratio_is_expressed_by_final_tensor_sizes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert not hasattr(groups[0], "dsa_num_blocks")
    assert not hasattr(groups[1], "dsa_num_blocks")


def test_num_gpu_blocks_override_keeps_base_block_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(override=320),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    assert config.num_blocks == 320
    assert get_dsa_group_num_blocks(config, groups[0]) == 960
    assert get_dsa_group_num_blocks(config, groups[1]) == 320


def test_cross_rank_base_capacity_shrink_preserves_plane_ratio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    old_base_blocks = config.num_blocks
    config.num_blocks = 128
    for tensor in config.kv_cache_tensors:
        tensor.size = tensor.size // old_base_blocks * config.num_blocks

    validate_dsa_kv_cache_config(config)
    assert get_dsa_group_num_blocks(config, groups[0]) == 384
    assert get_dsa_group_num_blocks(config, groups[1]) == 128


def test_prefill_admission_uses_weighted_base_block_cost(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    vllm_config = _make_vllm_config(max_model_len=2048)

    assert dsa_max_memory_usage_bytes(
        vllm_config,
        groups,
    ) == 16 * dsa_pool_bytes_per_base_block(groups)


def test_split_groups_reject_missing_per_layer_indexer() -> None:
    specs = _make_specs(num_layers=2)
    del specs["model.layers.1.self_attn.indexer.k_cache"]

    # GLM-5.2 共享 indexer 拓扑下 indexer 层是 resident 层的真子集，
    # 不再要求每层都有独立 indexer，分组应通过。
    groups = build_dsa_kv_cache_groups(specs)
    assert len(groups) == 2
    assert len(groups[0].layer_names) == 1  # 仅 layer 0 有 indexer
    assert len(groups[1].layer_names) == 2


def test_split_groups_reject_orphan_indexer_without_resident() -> None:
    # indexer 指向一个不存在 resident 层的 transformer 下标仍属硬错误。
    specs = _make_specs(num_layers=1)
    orphan = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    specs["model.layers.7.self_attn.indexer.k_cache"] = orphan

    with pytest.raises(RuntimeError, match="no matching resident MLA layer"):
        build_dsa_kv_cache_groups(specs)


def test_split_groups_reject_more_indexer_than_resident() -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.1.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    specs["model.layers.1.self_attn.attn"] = DSAResidentMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        sparse_head_dim=(512, 64, 0),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    # 移除唯一 resident 层，使 indexer 数 > resident 数。
    del specs["model.layers.0.self_attn.attn"]
    del specs["model.layers.1.self_attn.attn"]

    with pytest.raises(RuntimeError, match="more Indexer caches than resident"):
        build_dsa_kv_cache_groups(specs)


def test_block_pool_view_resets_every_physical_pool() -> None:
    class _Pool:
        num_gpu_blocks = 4

        def __init__(self, result: bool) -> None:
            self.result = result
            self.reset_calls = 0

        def reset_prefix_cache(self) -> bool:
            self.reset_calls += 1
            return self.result

    first = _Pool(False)
    second = _Pool(True)
    view = DSABlockPoolView([first, second])  # type: ignore[list-item]

    assert view.reset_prefix_cache() is False
    assert first.reset_calls == 1
    assert second.reset_calls == 1


def test_component_admission_checks_each_physical_pool() -> None:
    class _Pool:
        def __init__(self, free_blocks: int) -> None:
            self.free_blocks = free_blocks

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.physical_block_pools = (  # type: ignore[attr-defined]
        _Pool(6),
        _Pool(2),
    )

    assert coordinator.can_allocate((6, 2))
    assert not coordinator.can_allocate((7, 1))
    assert not coordinator.can_allocate((1, 3))
    with pytest.raises(RuntimeError, match="reservations"):
        coordinator.can_allocate((1, 1), reserved_blocks=1)
