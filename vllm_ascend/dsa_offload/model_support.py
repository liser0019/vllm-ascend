# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA offload 的模型能力识别。

模型架构名称只用于诊断输出，不作为使能白名单。是否支持 DSA offload
由 vLLM 已解析的 MLA 和 sparse-indexer 能力决定，因此 DeepSeek-V3.2 与
GLM-5.1 可以自然共享同一条判断路径，后续兼容模型也无需追加名称分支。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_ascend.dsa_offload.contracts import DSA_SFA_COMPUTE_TOPK


def _model_uses_sparse_indexer(model_config: Any) -> bool:
    """复用原生 SFA 的能力语义，同时避免配置初始化阶段的循环导入。

    ``vllm_ascend.utils`` 会反向导入 ``ascend_config``，因此这里不能直接
    调用其中的 ``model_uses_sfa_sparse``。两处判断保持同一份字段契约，
    后续若原生 SFA 能力协议发生变化，应同步更新并由单测约束。
    """
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return (
        hf_text_config is not None
        and hasattr(hf_text_config, "index_topk")
        and not hasattr(hf_text_config, "compress_ratios")
        and not hasattr(hf_config, "compress_ratios")
    )


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _indexer_types_or_none(value: Any) -> tuple[str, ...] | None:
    """归一化 HF config 的 indexer_types（如 GLM-5.2 的 full/shared 列表）。

    ``None`` 表示模型未声明拓扑，沿用每层独立 Indexer。只要模型显式提供
    该字段，就必须完整且只包含 ``full``/``shared``，避免异常配置被静默
    降级成 all-full 后在权重加载或绑定阶段才失败。
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"DSA indexer_types must be a sequence containing only 'full'/'shared', got {type(value).__name__}"
        )
    lowered: list[str] = []
    for layer_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"DSA indexer_types entries must be strings: layer={layer_index}, value={item!r}")
        normalized = item.lower()
        if normalized not in {"full", "shared"}:
            raise ValueError(
                f"DSA indexer_types entries must be 'full' or 'shared': layer={layer_index}, value={item!r}"
            )
        lowered.append(normalized)
    return tuple(lowered)


@dataclass(frozen=True)
class DSAOffloadModelCapabilities:
    """DSA offload 从已解析模型配置中提取出的稳定能力描述。"""

    architecture: str | None
    uses_mla: bool
    has_sparse_indexer: bool
    index_topk: int | None
    index_head_dim: int | None
    index_num_heads: int | None
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    # 共享 indexer 拓扑（GLM-5.2）。None 表示未声明 → 每层独立 indexer。
    indexer_types: tuple[str, ...] | None = None
    index_topk_freq: int | None = None

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.uses_mla:
            missing.append("MLA attention")
        if not self.has_sparse_indexer:
            missing.append("vLLM-Ascend sparse indexer")
        if self.index_topk != DSA_SFA_COMPUTE_TOPK:
            missing.append(f"index_topk={DSA_SFA_COMPUTE_TOPK} (got {self.index_topk!r})")
        for name, value in (
            ("index_head_dim", self.index_head_dim),
            ("index_n_heads", self.index_num_heads),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
        ):
            if value is None:
                missing.append(f"positive {name}")
        return tuple(missing)

    @property
    def supported(self) -> bool:
        return not self.missing_requirements

    @property
    def has_shared_indexer_layers(self) -> bool:
        """是否存在复用他层 top-K 的 shared indexer 层（GLM-5.2 拓扑）。"""
        return self.indexer_types is not None and any(t == "shared" for t in self.indexer_types)

    @property
    def full_indexer_layer_indices(self) -> tuple[int, ...] | None:
        """声明为 full（建独立 indexer）的 transformer 层下标；未声明拓扑时为 None。"""
        if self.indexer_types is None:
            return None
        return tuple(i for i, t in enumerate(self.indexer_types) if t == "full")

    @property
    def shared_indexer_layer_indices(self) -> tuple[int, ...] | None:
        """声明为 shared（复用所属 full 层 top-K）的层下标；未声明拓扑时为 None。"""
        if self.indexer_types is None:
            return None
        return tuple(i for i, t in enumerate(self.indexer_types) if t == "shared")


def get_dsa_offload_model_capabilities(
    model_config: Any | None,
) -> DSAOffloadModelCapabilities:
    """Extract DSA offload capabilities from vLLM's resolved ModelConfig."""
    if model_config is None:
        return DSAOffloadModelCapabilities(
            architecture=None,
            uses_mla=False,
            has_sparse_indexer=False,
            index_topk=None,
            index_head_dim=None,
            index_num_heads=None,
            kv_lora_rank=None,
            qk_rope_head_dim=None,
        )

    hf_text_config = getattr(model_config, "hf_text_config", None)
    return DSAOffloadModelCapabilities(
        architecture=getattr(model_config, "architecture", None),
        uses_mla=bool(getattr(model_config, "use_mla", False)),
        has_sparse_indexer=_model_uses_sparse_indexer(model_config),
        index_topk=_positive_int_or_none(getattr(hf_text_config, "index_topk", None)),
        index_head_dim=_positive_int_or_none(getattr(hf_text_config, "index_head_dim", None)),
        index_num_heads=_positive_int_or_none(getattr(hf_text_config, "index_n_heads", None)),
        kv_lora_rank=_positive_int_or_none(getattr(hf_text_config, "kv_lora_rank", None)),
        qk_rope_head_dim=_positive_int_or_none(getattr(hf_text_config, "qk_rope_head_dim", None)),
        indexer_types=_indexer_types_or_none(getattr(hf_text_config, "indexer_types", None)),
        index_topk_freq=_positive_int_or_none(getattr(hf_text_config, "index_topk_freq", None)),
    )


def require_dsa_offload_model_support(
    model_config: Any | None,
) -> DSAOffloadModelCapabilities:
    """Return capabilities or fail early with an actionable diagnostic."""
    capabilities = get_dsa_offload_model_capabilities(model_config)
    if capabilities.supported:
        return capabilities

    architecture = capabilities.architecture or "<unresolved>"
    raise ValueError(
        "DSA sparse offload requires an MLA model with the vLLM-Ascend "
        "sparse indexer and the current operator ABI. "
        f"architecture={architecture}, missing="
        f"{list(capabilities.missing_requirements)}"
    )
