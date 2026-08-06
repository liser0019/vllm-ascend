# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA Indexer/MLA 解耦后的 KV-cache 规格与物理容量规划。

vLLM v0.23 将 KV-cache 的职责拆成三层：

* ``KVCacheSpecRegistry`` 决定一种 spec 使用哪个 manager、能否与其他
  spec 归入同一组；
* ``KVCacheGroupSpec`` 表达共享同一张 block table 的逻辑层集合；
* ``KVCacheTensor`` 表达 worker 最终实际分配的字节数。

DSA 的 Indexer dense plane 与 MLA resident plane 在这三层都必须显式分离。
本模块只负责静态规格、分组、容量和最终 tensor 布局，不处理请求阶段转换，
也不在对象上动态追加 ``dsa_num_blocks`` 一类旁路字段。每个 plane 的真实
block 数统一由最终 ``KVCacheTensor.size / spec.page_size_bytes`` 推导。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

# 沿用 v0.19 DSA 原型的物理块数对齐。该对齐只作用于自动计算出来的
# base block 数；显式 num_gpu_blocks_override 仍由 vLLM 的标准入口接管。
DSA_KV_BLOCK_COUNT_ALIGNMENT = 128


def _may_override_num_blocks(
    vllm_config: VllmConfig,
    num_blocks: int,
) -> int:
    override = vllm_config.cache_config.num_gpu_blocks_override
    return int(override) if override is not None else num_blocks


@dataclass(frozen=True, kw_only=True)
class DSAIndexerKVSpec(FullAttentionSpec):
    """完整驻留 HBM 的 Indexer dense plane。

    ``FullAttentionSpec`` 默认按 K+V 两个向量计算 page 大小，而 Indexer
    cache 每个 token 只有一个 key 向量，因此这里覆盖真实 page 字节数。
    继承 ``FullAttentionSpec`` 是为了复用 full-attention 的块生命周期
    语义；独立的 class identity 则阻止 registry 将它和 resident MLA
    自动合并成同一 KV-cache group。
    """

    @property
    def real_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)


@dataclass(frozen=True, kw_only=True)
class DSAIndexerC8KVSpec(DSAIndexerKVSpec):
    """C8 量化 Indexer plane（int8 K + 每 token fp16 scale）。

    仅描述容量拆分；scale 的 dtype/维度用类级常量而非实例字段，因为
    ``FullAttentionSpec.merge`` 只重建基类字段，新增实例字段会在组 merge
    后悄悄丢回默认值。独立的 class identity 让 isinstance / registry MRO
    解析仍归入 Indexer plane（见 ``is_dsa_indexer_spec``）。

    TODO(遗留事项): 对应的 AscendC quant LIDU 算子尚未实现；当前 C8
    indexer 在 bind/spec 阶段被显式拒绝（见 model_runner / sfa_v1），
    本 spec 仅为后续接入预留容量契约。
    """

    C8_K_CACHE_DTYPE = torch.int8
    C8_K_SCALE_DTYPE = torch.float16
    C8_K_SCALE_DIM = 1

    @property
    def real_page_size_bytes(self) -> int:
        k_bytes = (
            self.block_size * self.num_kv_heads * self.head_size
            * get_dtype_size(self.C8_K_CACHE_DTYPE)
        )
        scale_bytes = (
            self.block_size * self.num_kv_heads * self.C8_K_SCALE_DIM
            * get_dtype_size(self.C8_K_SCALE_DTYPE)
        )
        return k_bytes + scale_bytes


@dataclass(frozen=True, kw_only=True)
class DSAResidentMLAAttentionSpec(AscendMLAAttentionSpec):
    """仅保存 attention resident working set 的 MLA plane。

    该 spec 的 ``sparse_head_dim`` 必须把 Indexer 维度设为 0；Indexer
    cache 由 ``DSAIndexerKVSpec`` 独立描述。请求何时从 dense 布局收缩到
    resident budget 属于 scheduler/manager 生命周期语义，不放在 spec 中。
    """


@dataclass(frozen=True)
class DSAKVCacheGroupIds:
    """finalized KVCacheConfig 中两个物理 plane 的稳定 group id。"""

    indexer: int
    resident_mla: int


def is_dsa_indexer_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAIndexerKVSpec)


def is_dsa_resident_mla_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAResidentMLAAttentionSpec)


def has_dsa_split_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> bool:
    return any(is_dsa_indexer_spec(spec) or is_dsa_resident_mla_spec(spec) for spec in kv_cache_specs.values())


def has_dsa_split_kv_cache_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec) or is_dsa_resident_mla_spec(group.kv_cache_spec)
        for group in kv_cache_groups
    )


def get_dsa_kv_cache_group_ids(
    kv_cache_config: KVCacheConfig,
) -> DSAKVCacheGroupIds:
    """解析并校验 Indexer/resident MLA 的最终 group id。

    group id 是 ``MultiGroupBlockTable``、scheduler block IDs 与 worker
    cache tensor 之间的共同索引。运行期只解析一次并保存在 model runner，
    避免热路径依赖“Indexer 固定是 group 0”之类的隐式顺序假设。
    """

    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_indexer_spec(group.kv_cache_spec)
    ]
    resident_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    if len(indexer_group_ids) != 1 or len(resident_group_ids) != 1:
        raise RuntimeError(
            "DSA KV-cache config must contain exactly one Indexer group and "
            "one resident MLA group: "
            f"indexer_group_ids={indexer_group_ids}, "
            f"resident_group_ids={resident_group_ids}"
        )
    return DSAKVCacheGroupIds(
        indexer=indexer_group_ids[0],
        resident_mla=resident_group_ids[0],
    )


def _merge_group_specs(
    layer_specs: dict[str, KVCacheSpec],
) -> KVCacheSpec:
    specs = list(layer_specs.values())
    assert specs
    merged = type(specs[0]).merge(specs)
    assert type(merged) is type(specs[0])
    return merged


def build_dsa_kv_cache_groups(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """按 plane 构造两个稳定有序的 KV-cache group。

    Indexer group 固定在前、resident MLA group 固定在后。稳定顺序不仅便于
    容量报告，也为 ``DSAKVCacheCoordinator`` 的 component-wise block
    table 语义提供确定的 group id。
    """

    indexer_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_indexer_spec(spec)}
    resident_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_resident_mla_spec(spec)}
    foreign_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if not is_dsa_indexer_spec(spec) and not is_dsa_resident_mla_spec(spec)
    }

    if not indexer_specs or not resident_specs or foreign_specs:
        raise RuntimeError(
            "DSA split KV-cache requires exactly one Indexer plane and one "
            "resident MLA plane: "
            f"indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}, "
            f"foreign_specs={tuple(sorted(type(spec).__name__ for spec in foreign_specs.values()))}"
        )
    if len(indexer_specs) > len(resident_specs):
        raise RuntimeError(
            "DSA split KV-cache cannot host more Indexer caches than resident "
            f"MLA layers: indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}"
        )

    # GLM-5.2 共享 indexer 拓扑下 indexer 层是 resident 层的真子集（仅 full
    # 层建 indexer）。校验每个 indexer 层都有同下标的 resident 层兜底，
    # 使「indexer 指向不存在的 resident 层」仍然响亮失败。
    if len(indexer_specs) != len(resident_specs):
        from vllm.model_executor.models.utils import extract_layer_index

        resident_indices = {extract_layer_index(name) for name in resident_specs}
        orphan_indexer = sorted(
            name for name in indexer_specs
            if extract_layer_index(name) not in resident_indices
        )
        if orphan_indexer:
            raise RuntimeError(
                "DSA Indexer cache has no matching resident MLA layer: "
                f"orphan_indexer_layers={tuple(orphan_indexer)}"
            )

    return [
        KVCacheGroupSpec(
            layer_names=list(indexer_specs),
            kv_cache_spec=_merge_group_specs(indexer_specs),
        ),
        KVCacheGroupSpec(
            layer_names=list(resident_specs),
            kv_cache_spec=_merge_group_specs(resident_specs),
        ),
    ]


def _get_dsa_ratio() -> int:
    from vllm_ascend.ascend_config import get_ascend_config

    ratio = int(get_ascend_config().dsa_offload_config.indexer_mla_block_ratio)
    if ratio <= 0:
        raise ValueError(f"DSA indexer_mla_block_ratio must be positive, got {ratio}")
    return ratio


def dsa_pool_bytes_per_base_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """返回一个 MLA base block 对应的总物理字节数。"""

    ratio = _get_dsa_ratio()
    total = 0
    for group in kv_cache_groups:
        if is_dsa_indexer_spec(group.kv_cache_spec):
            weight = ratio
        elif is_dsa_resident_mla_spec(group.kv_cache_spec):
            weight = 1
        else:
            raise TypeError(f"Unexpected KV-cache spec in DSA split layout: {type(group.kv_cache_spec).__name__}")
        total += group.kv_cache_spec.page_size_bytes * len(group.layer_names) * weight
    if total <= 0:
        raise ValueError("DSA KV-cache groups contain no physical layers")
    return total


def build_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """为 worker 构造 ratio 解耦的最终物理 tensor 布局。"""

    ratio = _get_dsa_ratio()
    bytes_per_base_block = dsa_pool_bytes_per_base_block(kv_cache_groups)
    num_base_blocks = available_memory // bytes_per_base_block
    num_base_blocks = num_base_blocks // DSA_KV_BLOCK_COUNT_ALIGNMENT * DSA_KV_BLOCK_COUNT_ALIGNMENT
    num_base_blocks = _may_override_num_blocks(
        vllm_config,
        num_base_blocks,
    )
    if num_base_blocks <= 0:
        raise ValueError(
            "No KV-cache blocks fit in the DSA split layout: "
            f"available_memory={available_memory}, "
            f"bytes_per_base_block={bytes_per_base_block}"
        )

    tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        group_num_blocks = num_base_blocks * ratio if is_dsa_indexer_spec(group.kv_cache_spec) else num_base_blocks
        for layer_name in group.layer_names:
            tensors.append(
                KVCacheTensor(
                    size=(group.kv_cache_spec.page_size_bytes * group_num_blocks),
                    shared_by=[layer_name],
                )
            )

    config = KVCacheConfig(
        # vLLM v0.23 仍保留一个标量 num_blocks。DSA 将它定义为 MLA
        # base capacity；Indexer 的 ratio 容量只由最终 tensor size 表达。
        num_blocks=num_base_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
    )
    validate_dsa_kv_cache_config(config)
    return config


def dsa_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """计算至少容纳一个完整 prefill 请求所需的物理字节数。"""

    resident_groups = [group for group in kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)]
    if len(resident_groups) != 1:
        raise RuntimeError("DSA memory admission requires exactly one resident MLA group")
    resident_group = resident_groups[0]
    resident_page_size = resident_group.kv_cache_spec.page_size_bytes
    resident_bytes_per_layer = resident_group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
    if resident_bytes_per_layer % resident_page_size != 0:
        raise RuntimeError("DSA resident max-memory requirement is not page aligned")

    # 一个完整 prefill 请求要求 MLA base plane 至少容纳完整上下文。
    # Indexer plane 虽有 ratio 倍容量，但不能拿多出来的 Indexer blocks
    # 抵偿缺失的 MLA blocks，因此 admission 必须按加权 base-block 成本
    # 检查，而不是简单相加两个 plane 各自的一份 max-memory。
    required_base_blocks = resident_bytes_per_layer // resident_page_size
    return required_base_blocks * dsa_pool_bytes_per_base_block(kv_cache_groups)


def _layer_tensor_sizes(
    kv_cache_config: KVCacheConfig,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in tensor.shared_by:
            if layer_name in sizes:
                raise RuntimeError(f"DSA layer is backed by more than one KVCacheTensor: {layer_name}")
            sizes[layer_name] = int(tensor.size)
    return sizes


def get_dsa_group_num_blocks(
    kv_cache_config: KVCacheConfig,
    group: KVCacheGroupSpec,
) -> int:
    """从最终 tensor 大小反推 group 的真实物理 block 数。"""

    if not group.layer_names:
        return 0
    tensor_sizes = _layer_tensor_sizes(kv_cache_config)
    page_size = int(group.kv_cache_spec.page_size_bytes)
    block_counts: set[int] = set()
    for layer_name in group.layer_names:
        try:
            tensor_size = tensor_sizes[layer_name]
        except KeyError as exc:
            raise RuntimeError(f"DSA KV-cache tensor is missing for layer {layer_name}") from exc
        if tensor_size % page_size != 0:
            raise RuntimeError(
                "DSA KV-cache tensor is not page aligned: "
                f"layer={layer_name}, tensor_size={tensor_size}, "
                f"page_size={page_size}"
            )
        block_counts.add(tensor_size // page_size)
    if len(block_counts) != 1:
        raise RuntimeError(
            "Layers in one DSA KV-cache group have different capacities: "
            f"group_spec={type(group.kv_cache_spec).__name__}, "
            f"block_counts={sorted(block_counts)}"
        )
    return block_counts.pop()


def validate_dsa_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> None:
    """校验最终物理 tensor 是否仍满足两 plane 的容量契约。"""

    indexer_groups = [group for group in kv_cache_config.kv_cache_groups if is_dsa_indexer_spec(group.kv_cache_spec)]
    resident_groups = [
        group for group in kv_cache_config.kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    if len(indexer_groups) != 1 or len(resident_groups) != 1:
        raise RuntimeError(
            "DSA KV-cache config must contain exactly one Indexer group and "
            f"one resident MLA group, got indexer={len(indexer_groups)}, "
            f"resident={len(resident_groups)}"
        )

    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_groups[0])
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_groups[0])
    ratio = _get_dsa_ratio()
    if resident_blocks != int(kv_cache_config.num_blocks):
        raise RuntimeError(
            "DSA resident MLA capacity must equal KVCacheConfig.num_blocks: "
            f"resident={resident_blocks}, "
            f"num_blocks={kv_cache_config.num_blocks}"
        )
    if indexer_blocks != resident_blocks * ratio:
        raise RuntimeError(
            "DSA Indexer/MLA capacity ratio mismatch: "
            f"indexer={indexer_blocks}, resident={resident_blocks}, "
            f"ratio={ratio}"
        )


def get_dsa_kv_cache_binding_order(
    kv_cache_config: KVCacheConfig,
) -> list[str]:
    """返回同层双 plane cache 的稳定绑定顺序。

    vLLM 的通用 ``bind_kv_cache`` 会把 cache 名先折叠为 transformer layer
    index；非 CUDA/XPU/CPU 平台若同一 layer index 对应多个 cache 名会直接
    拒绝。DSA 的 attention resident cache 与 Indexer cache 恰好必须共享
    transformer layer index，因此由 NPU model runner 使用此顺序逐个绑定。

    顺序定义为 layer index 优先、同层 resident MLA 在前、Indexer 在后。
    ``runner_kv_caches`` 当前只承担引用持有与清理职责，但仍固定顺序，避免
    后续消费者把字典插入顺序误当作 ABI。
    """

    # 延迟导入，避免 cache spec 注册阶段为了一个纯绑定辅助函数提前载入
    # 整个 model-executor utilities 模块。
    from vllm.model_executor.models.utils import extract_layer_index

    ordered_layers: list[tuple[int, int, str]] = []
    seen_layers: set[str] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_resident_mla_spec(spec):
            plane_order = 0
        elif is_dsa_indexer_spec(spec):
            plane_order = 1
        else:
            raise RuntimeError(f"Unexpected KV-cache group in DSA binding: {type(spec).__name__}")
        for layer_name in group.layer_names:
            if layer_name in seen_layers:
                raise RuntimeError(f"DSA KV-cache layer appears in more than one group: {layer_name}")
            seen_layers.add(layer_name)
            ordered_layers.append(
                (
                    extract_layer_index(layer_name),
                    plane_order,
                    layer_name,
                )
            )

    if not ordered_layers:
        raise RuntimeError("DSA KV-cache binding received no cache layers")
    return [layer_name for _, _, layer_name in sorted(ordered_layers)]


def report_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """按最终 tensor 大小输出一次确定性的 DSA HBM 容量报告。"""

    validate_dsa_kv_cache_config(kv_cache_config)
    indexer_group = next(group for group in kv_cache_config.kv_cache_groups if is_dsa_indexer_spec(group.kv_cache_spec))
    resident_group = next(
        group for group in kv_cache_config.kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)
    )
    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_group)
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_group)
    indexer_tokens = indexer_blocks * indexer_group.kv_cache_spec.block_size
    resident_tokens = resident_blocks * resident_group.kv_cache_spec.block_size
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    block_size = int(resident_group.kv_cache_spec.block_size)

    from vllm_ascend.ascend_config import get_ascend_config

    dsa_config = get_ascend_config().dsa_offload_config
    resident_slots_per_request = cdiv(dsa_config.max_resident_budget_tokens, block_size) * block_size + block_size
    decode_by_mla = resident_tokens // resident_slots_per_request
    indexer_blocks_per_request = cdiv(max_model_len, indexer_group.kv_cache_spec.block_size)
    decode_by_indexer = indexer_blocks // indexer_blocks_per_request
    configured_decode_limit = min(
        decode_by_mla,
        decode_by_indexer,
        max_num_seqs,
    )
    total_bytes = sum(int(tensor.size) for tensor in kv_cache_config.kv_cache_tensors)

    logger.info_once(
        "\n"
        "================ DSA HBM CACHE CAPACITY REPORT ================\n"
        "  Split ratio             : indexer:mla = %d:1; base blocks = %s\n"
        "  Allocated HBM KV bytes  : %s bytes (%s GiB)\n"
        "  MLA resident plane      : %s tokens (%s blocks x %s tokens)\n"
        "  Indexer dense plane     : %s tokens (%s blocks x %s tokens)\n"
        "  Batched prefill limit   : %s tokens (dense cache required in both planes)\n"
        "  Sparse decode MLA limit : %s requests (%s resident slots/request)\n"
        "  Dense Indexer limit     : %s requests at max_model_len=%s\n"
        "  Configured decode limit : %s requests (max_num_seqs=%s)\n"
        "=================================================================",
        _get_dsa_ratio(),
        f"{resident_blocks:,}",
        f"{total_bytes:,}",
        format_gib(total_bytes),
        f"{resident_tokens:,}",
        f"{resident_blocks:,}",
        f"{block_size:,}",
        f"{indexer_tokens:,}",
        f"{indexer_blocks:,}",
        f"{indexer_group.kv_cache_spec.block_size:,}",
        f"{min(resident_tokens, indexer_tokens):,}",
        f"{decode_by_mla:,}",
        f"{resident_slots_per_request:,}",
        f"{decode_by_indexer:,}",
        f"{max_model_len:,}",
        f"{configured_decode_limit:,}",
        f"{max_num_seqs:,}",
    )
