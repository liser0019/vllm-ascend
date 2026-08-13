import pytest
import torch

import vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager as manager_module
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


def test_sparse_copy_uses_memfabric_device_address(monkeypatch) -> None:
    manager = _manager_without_init()
    manager.copy_backend = COPY_BACKEND_SPARSE_COPY
    manager._host_copy_addresses = {}
    tensor = torch.zeros(16, dtype=torch.uint8)
    device_address = 0x12340000

    monkeypatch.setattr(
        manager_module.offload,
        "get_device_address",
        lambda candidate: device_address if candidate is tensor else 0,
    )

    assert manager._register_host_copy_address(tensor) == device_address
    assert manager._get_host_copy_address(tensor) == device_address
    assert manager._host_copy_addresses[tensor.data_ptr()] != tensor.data_ptr()


def test_cpu_copy_uses_cpu_address() -> None:
    manager = _manager_without_init()
    manager.copy_backend = COPY_BACKEND_CPU
    manager._host_copy_addresses = {}
    tensor = torch.zeros(16, dtype=torch.uint8)

    assert manager._register_host_copy_address(tensor) == tensor.data_ptr()
    assert manager._get_host_copy_address(tensor) == tensor.data_ptr()


def test_unregistered_host_tensor_is_rejected() -> None:
    manager = _manager_without_init()
    manager._host_copy_addresses = {}

    with pytest.raises(RuntimeError, match="was not registered"):
        manager._get_host_copy_address(torch.zeros(4, dtype=torch.uint8))
