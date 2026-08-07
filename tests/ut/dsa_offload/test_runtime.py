# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.dsa_offload.dram_store import DSAHotDRAMStore
from vllm_ascend.dsa_offload.input_batch import DSAInputBatchCacheLayout
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_offload.runtime import (
    DSALayerOffloadContext,
    DSAOffloadRuntime,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSARequestCacheLayoutProjection,
    DSAResidentBlockTableReplacement,
)


class _ResidentBlockTable:
    def __init__(self) -> None:
        self._rows = np.array([[10, 11, 12, 0]], dtype=np.int32)
        self.num_blocks_per_row = np.array([3], dtype=np.int32)

    def get_numpy_array(self) -> np.ndarray:
        return self._rows


def _make_runtime(
    max_num_reqs: int = 1,
) -> tuple[
    DSAResidentTokenPool,
    DSAOffloadRuntime,
    DSAHotDRAMStore,
]:
    resident_pool = DSAResidentTokenPool(
        max_num_reqs=max_num_reqs,
        num_layers=2,
        max_model_len=512,
        max_resident_budget_tokens=256,
        device=torch.device("cpu"),
    )
    runtime = DSAOffloadRuntime(
        max_num_reqs=max_num_reqs,
        max_num_tokens=512,
        num_layers=2,
        max_model_len=512,
        block_size=128,
        resident_token_pool=resident_pool,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    store = DSAHotDRAMStore(
        usable_blocks=8,
        storage_rows=resident_pool.storage_rows,
        max_logical_blocks=runtime.max_logical_blocks,
        device=torch.device("cpu"),
        arena_factory=lambda shape, dtype, capacity, device: torch.zeros(
            (capacity, *shape),
            dtype=dtype,
            device=device,
        ),
    )
    runtime.bind_dram_store(store)
    return resident_pool, runtime, store


def test_lidu_scratch_is_shared_but_cache_slots_remain_per_layer() -> None:
    resident_pool, runtime, _ = _make_runtime()

    first = runtime.get_lidu_outputs(num_reqs=1)
    second = runtime.get_lidu_outputs(num_reqs=1)

    assert first is second
    assert first.topk_index.data_ptr() == runtime._lidu_topk_index.data_ptr()
    assert first.topk_slots.data_ptr() == runtime._lidu_topk_slots.data_ptr()
    assert resident_pool.get_cache_slots(0).data_ptr() != resident_pool.get_cache_slots(1).data_ptr()


def _register_layer_arenas(store: DSAHotDRAMStore, num_layers: int = 2) -> None:
    for layer_id in range(num_layers):
        store.add_layer(
            layer_id=layer_id,
            resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        )


def _full_context(layer_id: int, runtime: DSAOffloadRuntime) -> DSALayerOffloadContext:
    return DSALayerOffloadContext(
        layer_id=layer_id,
        indexer_cache=torch.zeros((2, 4, 128), dtype=torch.bfloat16),
        runtime=runtime,
        selection_source_layer_id=None,
    )


def _shared_context(layer_id: int, source_id: int, runtime: DSAOffloadRuntime) -> DSALayerOffloadContext:
    return DSALayerOffloadContext(
        layer_id=layer_id,
        indexer_cache=None,
        runtime=runtime,
        selection_source_layer_id=source_id,
    )


def test_full_layer_records_selection_source(monkeypatch) -> None:
    resident_pool, runtime, store = _make_runtime()
    _register_layer_arenas(store)
    lidu_calls: list[dict] = []
    monkeypatch.setattr(
        "vllm_ascend.dsa_offload.runtime.lightning_indexer_decode_update",
        lambda **kw: lidu_calls.append(kw),
    )
    monkeypatch.setattr(
        "vllm_ascend.dsa_offload.runtime.kvcache_scatter_copy",
        lambda **kw: None,
    )

    ctx = _full_context(layer_id=0, runtime=runtime)
    ctx.execute_decode_selection(
        query=torch.zeros((1, 32, 128), dtype=torch.bfloat16),
        weights=torch.zeros((1, 32), dtype=torch.bfloat16),
        row_modes=torch.zeros((1,), dtype=torch.int32),
        resident_pool_indices=torch.zeros((1,), dtype=torch.int32),
        actual_seq_lengths_key=torch.ones((1,), dtype=torch.int32),
        indexer_block_table=torch.zeros((1, 4), dtype=torch.int32),
        resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
        dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
    )

    assert lidu_calls, "full 层应跑 LIDU"
    assert runtime._selection_source_layer == 0


def test_shared_layer_reuses_source_outputs_with_own_arenas(monkeypatch) -> None:
    resident_pool, runtime, store = _make_runtime()
    _register_layer_arenas(store)
    monkeypatch.setattr(
        "vllm_ascend.dsa_offload.runtime.lightning_indexer_decode_update",
        lambda **kw: None,
    )
    ksc_calls: list[dict] = []
    monkeypatch.setattr(
        "vllm_ascend.dsa_offload.runtime.kvcache_scatter_copy",
        lambda **kw: ksc_calls.append(kw),
    )

    full = _full_context(layer_id=0, runtime=runtime)
    full.execute_decode_selection(
        query=torch.zeros((1, 32, 128), dtype=torch.bfloat16),
        weights=torch.zeros((1, 32), dtype=torch.bfloat16),
        row_modes=torch.zeros((1,), dtype=torch.int32),
        resident_pool_indices=torch.zeros((1,), dtype=torch.int32),
        actual_seq_lengths_key=torch.ones((1,), dtype=torch.int32),
        indexer_block_table=torch.zeros((1, 4), dtype=torch.int32),
        resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
        dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
    )
    assert len(ksc_calls) == 1  # full 层的 KSC

    shared = _shared_context(layer_id=1, source_id=0, runtime=runtime)
    result = shared.execute_shared_decode_selection(
        resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
        dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
        num_reqs=1,
    )

    assert len(ksc_calls) == 2  # shared 层又跑一次 KSC
    shared_ksc = ksc_calls[1]
    # KSC 用的是本层（layer 1）的 arena，不是源 full 层的。
    own_arenas = store.get_layer_arenas(1)
    assert shared_ksc["dram_nope_arena"] is own_arenas.nope
    assert shared_ksc["dram_rope_arena"] is own_arenas.rope
    # 复用源 full 层的 LIDU 输出作为 KSC 与 SFA 输入。
    outputs = runtime.get_lidu_outputs(num_reqs=1)
    assert shared_ksc["dst_slots"] is outputs.topk_slots
    assert result.sparse_indices is outputs.topk_slots


def test_shared_layer_rejects_stale_or_missing_source(monkeypatch) -> None:
    resident_pool, runtime, store = _make_runtime()
    _register_layer_arenas(store)
    monkeypatch.setattr(
        "vllm_ascend.dsa_offload.runtime.kvcache_scatter_copy",
        lambda **kw: None,
    )

    shared = _shared_context(layer_id=1, source_id=0, runtime=runtime)
    # 本步尚未有任何 full 层跑 LIDU → 守卫应响。
    runtime._begin_selection_epoch()
    with pytest.raises(RuntimeError, match="selection source is stale"):
        shared.execute_shared_decode_selection(
            resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
            dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
            num_reqs=1,
        )
    # 来源 id 不匹配同样应响。
    runtime._selection_source_layer = 5
    with pytest.raises(RuntimeError, match="selection source is stale"):
        shared.execute_shared_decode_selection(
            resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
            dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
            num_reqs=1,
        )


def test_shared_layer_rejects_lidu_entry() -> None:
    resident_pool, runtime, store = _make_runtime()
    _register_layer_arenas(store)
    shared = _shared_context(layer_id=1, source_id=0, runtime=runtime)
    with pytest.raises(RuntimeError, match="must not run LIDU"):
        shared.execute_decode_selection(
            query=torch.zeros((1, 32, 128), dtype=torch.bfloat16),
            weights=torch.zeros((1, 32), dtype=torch.bfloat16),
            row_modes=torch.zeros((1,), dtype=torch.int32),
            resident_pool_indices=torch.zeros((1,), dtype=torch.int32),
            actual_seq_lengths_key=torch.ones((1,), dtype=torch.int32),
            indexer_block_table=torch.zeros((1, 4), dtype=torch.int32),
            resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
            dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
        )


def test_full_layer_rejects_shared_path() -> None:
    resident_pool, runtime, store = _make_runtime()
    _register_layer_arenas(store)
    full = _full_context(layer_id=0, runtime=runtime)
    with pytest.raises(RuntimeError, match="must use execute_decode_selection"):
        full.execute_shared_decode_selection(
            resident_nope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_rope_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
            resident_block_table=torch.zeros((1, 4), dtype=torch.int32),
            dram_block_table=torch.zeros((1, 4), dtype=torch.int32),
            num_reqs=1,
        )


def test_dump_plan_is_compact_and_idempotent() -> None:
    resident_pool, runtime, store = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.PREFILL),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
        ),
    )
    positions = torch.arange(257, dtype=torch.int64)
    req_indices = torch.zeros(257, dtype=torch.int64)
    scheduled = np.array([257], dtype=np.int32)

    returned_positions = runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=scheduled,
        req_indices=req_indices,
        positions=positions,
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )

    assert returned_positions.data_ptr() == positions.data_ptr()
    assert runtime.dump_job_count == 2
    assert runtime.dump_src_block_ids.np[:2].tolist() == [10, 11]
    assert store.logical_block_table[0, :2].tolist() != [0, 0]

    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=scheduled,
        req_indices=req_indices,
        positions=positions,
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )
    assert runtime.dump_job_count == 0


def test_consecutive_prefill_chunks_dump_only_newly_completed_blocks() -> None:
    resident_pool, runtime, store = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.PREFILL),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
        ),
    )

    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=np.array([257], dtype=np.int32),
        req_indices=torch.zeros(257, dtype=torch.int64),
        positions=torch.arange(257, dtype=torch.int64),
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )
    assert runtime.dump_job_count == 2
    first_two_dram_blocks = store.logical_block_table[0, :2].copy()

    input_batch.num_computed_tokens_cpu[0] = 257
    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=np.array([128], dtype=np.int32),
        req_indices=torch.zeros(128, dtype=torch.int64),
        positions=torch.arange(257, 385, dtype=torch.int64),
        num_reqs=1,
        num_tokens=128,
        resident_group_id=0,
    )

    assert runtime.dump_job_count == 1
    assert runtime.dump_src_block_ids.np[0] == 12
    assert store.logical_block_table[0, :2].tolist() == (first_two_dram_blocks.tolist())
    assert store.logical_block_table[0, 2] != 0


def test_enter_rejects_missing_dram_source_blocks() -> None:
    resident_pool, runtime, _ = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([256], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.ENTER_SPARSE_DECODE),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(256,),
            resident_valid_tokens=(257,),
            resident_block_table_replacements=(
                DSAResidentBlockTableReplacement(
                    request_id="req-0",
                    block_ids=(10, 11, 12),
                ),
            ),
        ),
    )

    try:
        runtime.prepare_forward(
            input_batch=input_batch,
            state=state,
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            req_indices=torch.zeros(1, dtype=torch.int64),
            positions=torch.tensor([256], dtype=torch.int64),
            num_reqs=1,
            num_tokens=1,
            resident_group_id=0,
        )
    except RuntimeError as error:
        assert "incomplete DRAM block table" in str(error)
        assert "first_missing_logical_block=0" in str(error)
    else:
        raise AssertionError("ENTER must reject a null DRAM source mapping")


def test_graph_execution_view_pads_dump_jobs_with_noop_rows() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)
    runtime.active_num_reqs = 2
    runtime.dump_job_count = 1
    runtime.dump_src_block_ids.np[0] = 7
    runtime.dump_dst_block_ids.np[0] = 9

    execution_rows = runtime.prepare_execution_view(
        active_num_reqs=2,
        graph_row_count=4,
    )

    assert execution_rows == 4
    assert runtime.execution_num_reqs == 4
    assert runtime.dump_launch_count == 4
    assert runtime.dump_src_block_ids.gpu[:4].tolist() == [7, 0, 0, 0]
    assert runtime.dump_dst_block_ids.gpu[:4].tolist() == [9, -1, -1, -1]


def test_eager_execution_view_keeps_compact_dump_jobs() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)
    runtime.active_num_reqs = 2
    runtime.dump_job_count = 1
    runtime.dump_src_block_ids.np[0] = 7
    runtime.dump_dst_block_ids.np[0] = 9

    execution_rows = runtime.prepare_execution_view(
        active_num_reqs=2,
        graph_row_count=None,
    )

    assert execution_rows == 2
    assert runtime.execution_num_reqs == 2
    assert runtime.dump_launch_count == 1
    assert runtime.dump_src_block_ids.gpu[:1].tolist() == [7]
    assert runtime.dump_dst_block_ids.gpu[:1].tolist() == [9]


def test_graph_capture_runtime_can_be_reused_for_multiple_sizes() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)

    for row_count in (4, 2, 1):
        runtime.prepare_graph_capture(row_count=row_count)

        assert runtime.graph_capture_row_count == row_count
        assert runtime.active_num_reqs == row_count
        assert runtime.execution_num_reqs == row_count
        assert runtime.dump_launch_count == row_count
        assert runtime.active_dram_block_table.gpu[:row_count].eq(0).all()
        assert runtime.dump_dst_block_ids.gpu[:row_count].eq(-1).all()

        runtime.restore_after_graph_capture()

        assert runtime.graph_capture_row_count == 0
        assert runtime.active_num_reqs == 0
        assert runtime.execution_num_reqs == 0
        assert runtime.dump_job_count == 0
        assert runtime.dump_launch_count == 0
