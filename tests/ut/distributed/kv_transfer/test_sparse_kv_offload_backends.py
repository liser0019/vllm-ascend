from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    COPY_BACKEND_CPU,
    COPY_BACKEND_SPARSE_COPY,
    SparseKVOffloadManager,
)


def _manager_without_init() -> SparseKVOffloadManager:
    return SparseKVOffloadManager.__new__(SparseKVOffloadManager)


def test_cpu_copy_rejects_graph_runtime() -> None:
    manager = _manager_without_init()
    manager.copy_backend = COPY_BACKEND_CPU

    with pytest.raises(RuntimeError, match="--enforce-eager"):
        manager._check_cpu_copy_runtime(capturing=True)


def test_sparse_copy_allows_graph_runtime() -> None:
    manager = _manager_without_init()
    manager.copy_backend = COPY_BACKEND_SPARSE_COPY

    manager._check_cpu_copy_runtime(capturing=True)


def test_npu_descriptors_are_copied_to_cpu() -> None:
    manager = _manager_without_init()
    manager.num_tokens_buffer_npu = torch.tensor([2], dtype=torch.int32)
    manager.num_tokens_buffer_cpu = torch.zeros(1, dtype=torch.int32)
    manager.gvas_buffer_npu = torch.tensor([11, 12, 0, 0], dtype=torch.int64)
    manager.addr_buffer_npu = torch.tensor([21, 22, 0, 0], dtype=torch.int64)
    manager.size_buffer_npu = torch.tensor([31, 32, 0, 0], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.zeros(4, dtype=torch.int64)
    manager.addr_buffer_cpu = torch.zeros(4, dtype=torch.int64)
    manager.size_buffer_cpu = torch.zeros(4, dtype=torch.int32)

    descriptor_count = manager._copy_h2d_descriptors_to_cpu()

    assert descriptor_count == 2
    assert manager.gvas_buffer_cpu[:2].tolist() == [11, 12]
    assert manager.addr_buffer_cpu[:2].tolist() == [21, 22]
    assert manager.size_buffer_cpu[:2].tolist() == [31, 32]


def test_invalid_npu_descriptor_count_is_rejected() -> None:
    manager = _manager_without_init()
    manager.num_tokens_buffer_npu = torch.tensor([5], dtype=torch.int32)
    manager.num_tokens_buffer_cpu = torch.zeros(1, dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.zeros(4, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="invalid descriptor count"):
        manager._copy_h2d_descriptors_to_cpu()


def _runtime_routing_manager() -> SparseKVOffloadManager:
    manager = _manager_without_init()
    manager.use_sparse_kv_runtime = True
    manager.max_num_topk_rows = 8
    manager.max_num_blocks = 2
    manager.topk_buffers_k = [torch.empty(1) for _ in range(4)]
    manager.runtime_block_table_npu = torch.empty(
        (8, 2), dtype=torch.int32
    )
    manager.runtime_row_to_req_npu = torch.empty(8, dtype=torch.int64)
    manager._get_offload_layer_id = lambda _layer_name: 3
    manager._check_cpu_copy_runtime = lambda _capturing: None
    manager._onload_topk_kv_runtime = Mock()
    return manager


def _runtime_inputs(num_tokens: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(num_tokens * 4, dtype=torch.int32).view(num_tokens, 4),
        torch.empty((num_tokens, 4), dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int64),
        torch.zeros(num_tokens, dtype=torch.int32),
    )


def test_runtime_path_bypasses_legacy_descriptor_pipeline() -> None:
    manager = _runtime_routing_manager()
    topk, current_slots, req_ids, stable_prefix = _runtime_inputs(2)
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

    manager.onload_topk_kv(
        "layer.3",
        num_tokens=2,
        num_reqs=2,
        block_table=block_table,
        topk_indices_npu=topk,
        current_slots_npu=current_slots,
        req_ids_npu=req_ids,
        stable_prefix_lens_npu=stable_prefix,
    )

    manager._onload_topk_kv_runtime.assert_called_once()
    call = manager._onload_topk_kv_runtime.call_args.args
    assert call[0:2] == (3, 2)
    assert torch.equal(call[2], block_table)
    assert call[2].data_ptr() == manager.runtime_block_table_npu.data_ptr()
    assert call[3] is topk
    assert call[4] is current_slots


def test_runtime_path_expands_block_table_for_speculative_rows() -> None:
    manager = _runtime_routing_manager()
    topk, current_slots, req_ids, stable_prefix = _runtime_inputs(3)
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    token_to_req = torch.tensor([1, 0, 1], dtype=torch.int32)

    manager.onload_topk_kv(
        "layer.3",
        num_tokens=3,
        num_reqs=2,
        block_table=block_table,
        topk_indices_npu=topk,
        current_slots_npu=current_slots,
        req_ids_npu=req_ids,
        stable_prefix_lens_npu=stable_prefix,
        token_to_req_npu=token_to_req,
    )

    expanded = manager._onload_topk_kv_runtime.call_args.args[2]
    expected = torch.tensor(
        [[20, 21], [10, 11], [20, 21]], dtype=torch.int32
    )
    assert torch.equal(expanded, expected)
    assert expanded.data_ptr() == manager.runtime_block_table_npu.data_ptr()


def test_runtime_raw_pointer_inputs_use_persistent_staging() -> None:
    manager = _manager_without_init()
    rows = 2
    topk = 4
    capacity = 5
    layer_id = 0
    manager.topk_buffers_k = [torch.empty(1)]
    manager.runtime_req_ids_npu = torch.empty(8, dtype=torch.int64)
    manager.runtime_topk_indices_npu = torch.empty((8, topk), dtype=torch.int32)
    manager.runtime_stable_prefix_lens_npu = torch.empty(8, dtype=torch.int32)
    manager.lru_last_req_ids_npu_list = [torch.full((8,), -1, dtype=torch.int64)]
    manager.lru_slot_to_token_npu_list = [
        torch.full((8, capacity), -1, dtype=torch.int32)
    ]
    manager.lru_slots_npu_list = [
        torch.arange(capacity, dtype=torch.int32).repeat(8, 1)
    ]
    manager.lru_miss_count_npu_list = [torch.empty(8, dtype=torch.int32)]
    manager.lru_miss_tokens_npu_list = [
        torch.empty((8, topk), dtype=torch.int32)
    ]
    manager.lru_miss_slots_npu_list = [
        torch.empty((8, topk), dtype=torch.int32)
    ]
    manager.sparse_kv_plan_workspace_npu = torch.empty(128, dtype=torch.uint8)
    manager.gvas_k_bases = [100]
    manager.gvas_v_bases = [200]
    manager.addr_k_bases = [300]
    manager.addr_v_bases = [400]
    manager.block_size = 4
    manager.token_size_bytes_k = 8
    manager.token_size_bytes_v = 4
    manager.max_model_len = 128
    manager.num_physical_blocks = 32

    req_ids = torch.tensor([10, 11], dtype=torch.int64)
    topk_storage = torch.arange(rows * topk * 2, dtype=torch.int32).view(
        rows, topk * 2
    )
    topk_input = topk_storage[:, ::2]
    stable_prefix = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.arange(rows * 3, dtype=torch.int32).view(rows, 3)
    current_slots = torch.empty((rows, topk), dtype=torch.int32)

    target = (
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload."
        "sparse_kv_offload_manager.offload.sparse_kv_load_runtime"
    )
    with patch(target, return_value=0) as runtime:
        manager._onload_topk_kv_runtime(
            layer_id,
            rows,
            block_table,
            topk_input,
            current_slots,
            req_ids,
            stable_prefix,
        )

    args = runtime.call_args.args
    assert args[0].data_ptr() == manager.runtime_req_ids_npu.data_ptr()
    assert args[2].data_ptr() == manager.runtime_topk_indices_npu.data_ptr()
    assert args[3].data_ptr() == (
        manager.runtime_stable_prefix_lens_npu.data_ptr()
    )
    assert torch.equal(args[0], req_ids)
    assert torch.equal(args[2], topk_input)
    assert torch.equal(args[3], stable_prefix)
    assert args[-2] == manager.num_physical_blocks
