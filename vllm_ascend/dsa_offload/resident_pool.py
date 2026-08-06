# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 请求行与逐层 token-position→resident-slot 持久状态池。

LIDU 会在每层、每个 decode step 原址刷新 ``cache_slots``。前 ``W-1`` 列
保存完整序列 token position 到 resident 逻辑 slot 的映射，最后一列保存
该行的初始化状态：

* ``0``：尚未进入 sparse resident；
* ``-budget``：本层下一次 LIDU 需要执行 first-fill；
* ``+budget``：本层已经建立稳定 resident 映射。

pool row 独立于 ``InputBatch`` 行号，因此基线对请求行做 condense/reorder 时
不需要搬运一整行 ``max_model_len`` 状态；每轮只需把最终 batch row 映射为
一个稳定 pool index。最后额外保留一行给图模式 PAD 使用。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from contextlib import suppress

import torch

from vllm_ascend.dsa_offload.contracts import (
    DSA_LIDU_CACHE_ROW_ALIGNMENT,
    DSA_LIDU_TOKEN_CAPACITY,
)


class DSAResidentTokenPool:
    """管理 worker 内活跃请求的稳定 resident 行与逐层 LIDU 状态。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        num_layers: int,
        max_model_len: int,
        max_resident_budget_tokens: int,
        device: torch.device,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("DSA resident pool capacity must be positive")
        if num_layers <= 0:
            raise ValueError("DSA resident pool requires at least one layer")
        if max_model_len <= 0:
            raise ValueError("DSA resident pool max_model_len must be positive")
        if max_model_len > DSA_LIDU_TOKEN_CAPACITY:
            raise ValueError(
                "DSA max_model_len exceeds LIDU token-position capacity: "
                f"max_model_len={max_model_len}, "
                f"capacity={DSA_LIDU_TOKEN_CAPACITY}"
            )

        self.max_num_reqs = int(max_num_reqs)
        self.padding_pool_index = self.max_num_reqs
        self.storage_rows = self.max_num_reqs + 1
        self.num_layers = int(num_layers)
        self.max_model_len = int(max_model_len)
        self.max_resident_budget_tokens = int(
            max_resident_budget_tokens
        )
        self.device = torch.device(device)

        raw_width = self.max_model_len + 1
        aligned_width = (
            (
                raw_width
                + DSA_LIDU_CACHE_ROW_ALIGNMENT
                - 1
            )
            // DSA_LIDU_CACHE_ROW_ALIGNMENT
            * DSA_LIDU_CACHE_ROW_ALIGNMENT
        )
        self.cache_row_width = (
            aligned_width
            if aligned_width - 1 <= DSA_LIDU_TOKEN_CAPACITY
            else raw_width
        )
        self.cache_metadata_index = self.cache_row_width - 1

        self._free_indices = deque(range(self.max_num_reqs))
        self._request_to_index: dict[Hashable, int] = {}
        self._request_target_budgets: dict[Hashable, int] = {}
        # ``_cache_slots`` 按 resident 层数建满。GLM-5.2 的 shared indexer
        # 层不跑 LIDU，其对应行从不被读写（死行），仅为保持层稠密编号与
        # all-full 模型逐字节一致；后续可在 shared 拓扑下按 full 层数收缩。
        self._cache_slots = torch.full(
            (
                self.num_layers,
                self.storage_rows,
                self.cache_row_width,
            ),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._cache_slots[
            :,
            :,
            self.cache_metadata_index,
        ].zero_()
        self._graph_capture_row_count = 0

    def acquire(self, request_id: Hashable) -> int:
        """返回请求稳定行；首次出现时分配并清空一行。"""

        current = self._request_to_index.get(request_id)
        if current is not None:
            return current
        if not self._free_indices:
            raise RuntimeError(
                "No free DSA resident metadata row is available"
            )
        pool_index = self._free_indices.popleft()
        self._request_to_index[request_id] = pool_index
        self._clear_index(pool_index)
        return pool_index

    def prepare_sparse_request(
        self,
        request_id: Hashable,
        *,
        target_budget_tokens: int,
    ) -> None:
        """在请求首次进入 sparse 时为所有层写入 first-fill 负预算。"""

        pool_index = self._require_index(request_id)
        target_budget_tokens = int(target_budget_tokens)
        if not (
            0
            < target_budget_tokens
            <= self.max_resident_budget_tokens
        ):
            raise ValueError(
                "DSA resident budget is outside pool capacity: "
                f"budget={target_budget_tokens}, "
                f"capacity={self.max_resident_budget_tokens}"
            )
        existing = self._request_target_budgets.get(request_id)
        if existing is not None:
            if existing != target_budget_tokens:
                raise RuntimeError(
                    "DSA request resident budget changed after binding: "
                    f"request={request_id!r}, old={existing}, "
                    f"new={target_budget_tokens}"
                )
            return
        self._cache_slots[
            :,
            pool_index,
            self.cache_metadata_index,
        ].fill_(-target_budget_tokens)
        self._request_target_budgets[request_id] = (
            target_budget_tokens
        )

    def release(self, request_id: Hashable) -> int | None:
        """释放请求行；下一次分配该行时统一清除所有层状态。

        未分配行不会进入 ``req_pool_entries``，因此在 release 和 acquire
        各清一次没有额外安全收益。把唯一一次整行清理放在 acquire，可避免
        请求结束时对 ``num_layers * cache_row_width`` 再发起一次设备写。
        """

        pool_index = self._request_to_index.pop(request_id, None)
        if pool_index is None:
            return None
        self._request_target_budgets.pop(request_id, None)
        self._free_indices.appendleft(pool_index)
        return pool_index

    def get_index(self, request_id: Hashable) -> int | None:
        return self._request_to_index.get(request_id)

    def get_cache_slots(self, layer_id: int) -> torch.Tensor:
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"DSA layer_id={layer_id} is outside "
                f"[0, {self.num_layers})"
            )
        return self._cache_slots[layer_id]

    @property
    def graph_capture_row_count(self) -> int:
        """返回当前临时安装的 graph-capture 行数。"""

        return self._graph_capture_row_count

    def prepare_graph_capture(
        self,
        *,
        row_count: int,
        target_budget_tokens: int,
    ) -> None:
        """为原生 FULL-graph dummy-run 安装逐层 first-fill 状态。

        dummy 行直接复用真实 ``cache_slots`` 的前缀地址。捕获结束后统一
        清空；实际 replay 仍通过 ``req_pool_entries`` 选择请求自己的稳定
        pool row。
        """

        row_count = int(row_count)
        target_budget_tokens = int(target_budget_tokens)
        if self._graph_capture_row_count:
            raise RuntimeError(
                "DSA resident graph-capture rows were installed twice"
            )
        if self._request_to_index:
            raise RuntimeError(
                "DSA graph capture must complete before serving requests"
            )
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA resident graph-capture row count is outside pool "
                f"capacity: rows={row_count}, capacity={self.max_num_reqs}"
            )
        if not (
            0
            < target_budget_tokens
            <= self.max_resident_budget_tokens
        ):
            raise ValueError(
                "DSA graph-capture budget is outside resident capacity: "
                f"budget={target_budget_tokens}, "
                f"capacity={self.max_resident_budget_tokens}"
            )

        self._graph_capture_row_count = row_count
        try:
            self._cache_slots[:, :row_count].fill_(-1)
            self._cache_slots[
                :,
                :row_count,
                self.cache_metadata_index,
            ].fill_(-target_budget_tokens)
        except Exception:
            # 保留首次安装失败作为根因；后续 capture 会重写完整前缀。
            with suppress(Exception):
                self._cache_slots[:, :row_count].fill_(-1)
                self._cache_slots[
                    :,
                    :row_count,
                    self.cache_metadata_index,
                ].zero_()
            self._graph_capture_row_count = 0
            raise

    def restore_after_graph_capture(self) -> None:
        """清除 dummy 对逐层 tokenwise 状态的原址修改。"""

        row_count = self._graph_capture_row_count
        if not row_count:
            return
        self._cache_slots[:, :row_count].fill_(-1)
        self._cache_slots[
            :,
            :row_count,
            self.cache_metadata_index,
        ].zero_()
        self._graph_capture_row_count = 0

    def _require_index(self, request_id: Hashable) -> int:
        pool_index = self._request_to_index.get(request_id)
        if pool_index is None:
            raise KeyError(
                f"DSA request {request_id!r} has no resident pool row"
            )
        return pool_index

    def _clear_index(self, pool_index: int) -> None:
        self._cache_slots[:, pool_index].fill_(-1)
        self._cache_slots[
            :,
            pool_index,
            self.cache_metadata_index,
        ].zero_()
