import os

import torch
import torch_npu
from memfabric_hybrid import offload
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import SparseKVOffloadConfig


# Main BF16 cache: [k_cache, v_cache, k_cache_cpu, v_cache_cpu,
# topk_buffer_k, topk_buffer_v]. Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5

LRU_BACKEND_CPU = "cpu"
LRU_BACKEND_NPU = "npu"
COPY_BACKEND_CPU = "cpu"
COPY_BACKEND_SPARSE_COPY = "sparse_copy"
COPY_DIRECTION_H2D = 0
COPY_DIRECTION_D2H = 1


_SUBSCRIBED_COMPUTE_STREAMS: set[object] = set()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


def get_host_device_memory_usage_ratio(kv_cache_spec: dict[str, KVCacheSpec]) -> float:
    page_size_bytes_host = 0
    page_size_bytes_device = 0
    for kv_cache_spec in kv_cache_spec.values():
        assert isinstance(kv_cache_spec, KVCacheSpec)
        if getattr(kv_cache_spec, 'store_on_host', False):
            page_size_bytes_host += kv_cache_spec.page_size_bytes
        else:
            page_size_bytes_device += kv_cache_spec.page_size_bytes

    assert page_size_bytes_device > 0, "Case of no device kv cache is not considered."
    return page_size_bytes_host / page_size_bytes_device


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1 # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    topk_buffer_k_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * k_dim * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * v_dim * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device='npu')
    topk_buffer_k = (
        topk_buffer_raw[:topk_buffer_k_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim])
    )
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes:topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim])
    )
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(
        bool(getattr(spec, "store_on_host", False))
        for spec in kv_cache_spec.values()
    )
    if num_offload_layers == 0:
        raise RuntimeError(
            "Sparse KV offload profile did not find any host-resident SFA "
            "KV cache layers."
        )

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
        for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size()
        for buffer_pair in buffers
        for tensor in buffer_pair
    )
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d "
        "layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    This is used for GLM-5.2 indexer reuse optimize:
    make sure that delta_k_cache_addrs and delta_v_cache_addrs
    between each two layers are the same,
    so we only need to add one delta_addr to the addr tensor of sparse_copy
    without need to mask k and v separately.
    Same reason that we allocate topk_buffer_k & topk_buffer_v together.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset:base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: callable,
) -> tuple[torch.Tensor | int]:
    if tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size],
            alignment,
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple[torch.Tensor | int],
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor]:
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0:
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = (
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    )
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the availble memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        self.lru_backend = envs_ascend.VLLM_ASCEND_SPARSE_KV_LRU_BACKEND
        self.copy_backend = envs_ascend.VLLM_ASCEND_SPARSE_KV_COPY_BACKEND
        if self.lru_backend not in (LRU_BACKEND_CPU, LRU_BACKEND_NPU):
            raise ValueError(
                "VLLM_ASCEND_SPARSE_KV_LRU_BACKEND must be 'cpu' or 'npu', "
                f"got {self.lru_backend!r}"
            )
        if self.copy_backend not in (
            COPY_BACKEND_CPU,
            COPY_BACKEND_SPARSE_COPY,
        ):
            raise ValueError(
                "VLLM_ASCEND_SPARSE_KV_COPY_BACKEND must be 'cpu' or "
                f"'sparse_copy', got {self.copy_backend!r}"
            )
        runtime_compatible = (
            self.lru_backend == LRU_BACKEND_NPU
            and self.copy_backend == COPY_BACKEND_SPARSE_COPY
        )
        self.use_sparse_kv_runtime = (
            runtime_compatible
            and envs_ascend.VLLM_ASCEND_SPARSE_KV_RUNTIME
            and hasattr(offload, "get_sparse_kv_plan_workspace_size")
            and hasattr(offload, "sparse_kv_load_runtime")
        )
        if (
            runtime_compatible
            and envs_ascend.VLLM_ASCEND_SPARSE_KV_RUNTIME
            and not self.use_sparse_kv_runtime
        ):
            logger.warning(
                "The installed MemFabric package does not provide "
                "SparseKvLoadRuntime; falling back to the legacy "
                "Compact -> ResidentAddrs -> SparseCopy path."
            )
        if self.tp_size != 1 and (
            self.lru_backend == LRU_BACKEND_NPU
            or self.copy_backend == COPY_BACKEND_CPU
        ):
            raise RuntimeError(
                "The experimental NPU LRU and CPU copy backends currently "
                "support only tensor_parallel_size=1"
            )
        if self.lru_backend == LRU_BACKEND_NPU and not self.use_sparse_kv_runtime:
            missing_ops = [
                name
                for name in (
                    "lru_resident_compact",
                    "compute_lru_resident_addrs",
                )
                if not hasattr(offload, name)
            ]
            if missing_ops:
                raise RuntimeError(
                    "The installed MemFabric package does not provide the "
                    f"required NPU LRU operators: {', '.join(missing_ops)}"
                )
        if self.copy_backend == COPY_BACKEND_CPU and not hasattr(
            torch.ops._C_ascend, "swap_blocks_batch"
        ):
            raise RuntimeError(
                "The CPU copy backend requires torch.ops._C_ascend."
                "swap_blocks_batch"
            )
        if self.copy_backend == COPY_BACKEND_SPARSE_COPY and not hasattr(
            offload, "get_device_address"
        ):
            raise RuntimeError(
                "The sparse_copy path requires MemFabric registered-Host-DVA "
                "support through offload.get_device_address()."
            )
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.max_num_blocks = max_block_num
        self.block_table_cpu = torch.zeros(
            [self.max_num_reqs, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.block_table_expanded_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self._npu_runtime = torch_npu.npu

        if self.lru_backend == LRU_BACKEND_CPU:
            self._build_cpp()
        else:
            self.sparse_kv_offload_cpp = None

        logger.warning(
            "Sparse KV offload experimental backends: lru=%s, copy=%s, "
            "runtime=%s. "
            "The NPU LRU and CPU copy paths require single-rank validation; "
            "CPU copy also requires eager execution.",
            self.lru_backend,
            self.copy_backend,
            self.use_sparse_kv_runtime,
        )

        logger.info(
            f"SparseKVOffloadManager start init CPU KV pool with {sparse_kv_offload_config.dram_size_per_dp_GB} "
            "GB dram per dp group, it might be time consuming, please wait."
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30)
        config.alloc_size = sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30) if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = offload.Scene.SHARED
        if self.copy_backend == COPY_BACKEND_SPARSE_COPY:
            if not hasattr(config, "register_host_memory"):
                raise RuntimeError(
                    "The installed MemFabric package cannot register its Host "
                    "pool for NPU access; update MemFabric before selecting "
                    "the sparse_copy or SparseKvLoadRuntime path."
                )
            config.register_host_memory = True
        assert offload.initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

    def _build_cpp(self):
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        npu_include_path = os.path.join(ascend_home, "include")
        npu_lib_path = os.path.join(ascend_home, "lib64")
        if not os.path.exists(npu_lib_path):
            npu_lib_path = os.path.join(ascend_home, "lib")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        os.environ['CXX'] = 'clang++'
        os.environ['CC'] = 'clang'
        abs_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(abs_path, "sparse_kv_offload.cpp")
        logger.info(f'Sparse KV offload build cpp utils from src: {src_path}')
        self.sparse_kv_offload_cpp = torch.utils.cpp_extension.load(
            name="sparse_kv_offload",
            sources=[src_path],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=True,
        )

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig | None,
    ) -> int:
        assert len(kv_cache_config.kv_cache_groups) == 1, "Hybrid KV is not supported."
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        return kv_cache_spec.block_size

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [
            layer_name for layer_name in kv_caches
            if 'indexer' not in layer_name
        ]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id
            for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info(
            "Sparse KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        self._register_offload_layers(kv_caches)

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H uses a separate descriptor set from the shared H2D buffers below.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_dst_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_lengths_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int32, device=device
        )
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        if self.copy_backend == COPY_BACKEND_CPU:
            self.d2h_src_ptrs_cpu = torch.empty(
                d2h_descriptor_rows,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self.d2h_dst_ptrs_cpu = torch.empty(
                d2h_descriptor_rows,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self.d2h_lengths_cpu = torch.empty(
                d2h_descriptor_rows,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )

        pages_per_row = self.topk_buffer_size // self.block_size
        self.current_slots_npu = torch.empty(
            (self.max_num_topk_rows, self.topk),
            dtype=torch.int32,
            device=device,
        )
        self.resident_block_table_npu = torch.arange(
            self.max_num_topk_rows * pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(self.max_num_topk_rows, pages_per_row)
        self.resident_query_lens_npu = torch.arange(
            1, self.max_num_topk_rows + 1, dtype=torch.int32, device=device
        )
        self.resident_seq_lens_npu = torch.full(
            (self.max_num_topk_rows,),
            self.topk_buffer_size,
            dtype=torch.int32,
            device=device,
        )

        # sparse_copy related addrs and buffers
        self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
        self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        self.cpu_block_lens: list[tuple[int, int]] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device='npu')
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device='npu')
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device='npu')
        self.host_dva_by_data_ptr: dict[int, int] = {}
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                if self.copy_backend == COPY_BACKEND_SPARSE_COPY:
                    k_dva = int(offload.get_device_address(k_cpu))
                    v_dva = int(offload.get_device_address(v_cpu))
                    self.host_dva_by_data_ptr[k_cpu.data_ptr()] = k_dva
                    self.host_dva_by_data_ptr[v_cpu.data_ptr()] = v_dva
                    gvas_k_tensor[layer_id] = k_dva
                    gvas_v_tensor[layer_id] = v_dva
                else:
                    gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                    gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())
            self.cpu_block_lens.append((
                cpu_block_lens_tensor[layer_id, 0].item(),
                cpu_block_lens_tensor[layer_id, 1].item(),
            ))

        # SparseKvLoadRuntime consumes the compact miss plan directly.  Only
        # legacy CPU/three-op modes need the intermediate descriptor buffers.
        if not self.use_sparse_kv_runtime:
            gvas_buffer_offset = 0
            gvas_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
            addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
            addr_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
            size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
            size_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 4
            num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
            num_tokens_buffer_size_bytes = 4
            descriptor_bytes = (
                gvas_buffer_size_bytes + addr_buffer_size_bytes
                + size_buffer_size_bytes + num_tokens_buffer_size_bytes
            )
            self.sparse_copy_args_buffer_cpu = torch.zeros(
                [descriptor_bytes], dtype=torch.int8, device='cpu',
                pin_memory=True,
            )
            self.sparse_copy_args_buffer_npu = torch.zeros(
                [descriptor_bytes], dtype=torch.int8, device=device,
            )
            self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[
                gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes
            ].view(torch.int64)
            self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[
                addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes
            ].view(torch.int64)
            self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[
                size_buffer_offset:size_buffer_offset + size_buffer_size_bytes
            ].view(torch.int32)
            self.num_tokens_buffer_cpu = self.sparse_copy_args_buffer_cpu[
                num_tokens_buffer_offset:
                num_tokens_buffer_offset + num_tokens_buffer_size_bytes
            ].view(torch.int32)
            self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[
                gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes
            ].view(torch.int64)
            self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[
                addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes
            ].view(torch.int64)
            self.size_buffer_npu = self.sparse_copy_args_buffer_npu[
                size_buffer_offset:size_buffer_offset + size_buffer_size_bytes
            ].view(torch.int32)
            self.num_tokens_buffer_npu = self.sparse_copy_args_buffer_npu[
                num_tokens_buffer_offset:
                num_tokens_buffer_offset + num_tokens_buffer_size_bytes
            ].view(torch.int32)
            if self.copy_backend == COPY_BACKEND_CPU:
                self.copy_sizes_cpu = torch.empty(
                    self.max_num_topk_rows * self.topk * 2,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )

        # topk cache reuse related
        self.lru_workspace_threads = 8
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_to_req_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [torch.full(
            [self.max_num_topk_rows, self.topk_buffer_size],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_slots_cpu_list = [torch.arange(
            self.topk_buffer_size,
            dtype=torch.int32,
            device='cpu',
        ).view(1, -1).repeat(self.max_num_topk_rows, 1).pin_memory() for _ in range(self.num_layers)]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_tokens_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_slots_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device='cpu', pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [torch.full(
            [self.max_num_topk_rows],
            -1,
            dtype=torch.int64,
            device='cpu',
            pin_memory=True,
        ) for _  in range(self.num_layers)]
        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list]
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list]
        self.lru_miss_tokens_ptrs = [lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list]
        self.lru_miss_slots_ptrs = [lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()

        if self.lru_backend == LRU_BACKEND_NPU:
            self.lru_last_req_ids_npu_list = [
                torch.full(
                    [self.max_num_topk_rows],
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
            self.lru_slot_to_token_npu_list = [
                torch.full(
                    [self.max_num_topk_rows, self.topk_buffer_size],
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
            initial_lru_slots = torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device=device,
            ).view(1, -1).repeat(self.max_num_topk_rows, 1)
            self.lru_slots_npu_list = [
                initial_lru_slots.clone() for _ in range(self.num_layers)
            ]
            self.lru_miss_count_npu_list = [
                torch.empty(
                    [self.max_num_topk_rows],
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
            self.lru_current_slots_npu = torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device=device,
            )
            self.lru_miss_tokens_npu_list = [
                torch.empty(
                    [self.max_num_topk_rows, self.topk],
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
            self.lru_miss_slots_npu_list = [
                torch.empty(
                    [self.max_num_topk_rows, self.topk],
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
            if self.use_sparse_kv_runtime:
                plan_workspace_bytes = int(
                    offload.get_sparse_kv_plan_workspace_size(
                        self.max_num_topk_rows,
                        self.topk,
                        self.topk_buffer_size,
                    )
                )
                if plan_workspace_bytes <= 0:
                    raise RuntimeError(
                        "MemFabric returned an invalid SparseKvPlanRuntime "
                        f"workspace size: {plan_workspace_bytes}"
                    )
                self.sparse_kv_plan_workspace_npu = torch.empty(
                    [plan_workspace_bytes], dtype=torch.uint8, device=device
                )
                # MemFabric receives raw data_ptr values and submits Plan and
                # Transfer asynchronously.  Keep every runtime input whose
                # layout may require copying/expansion in manager-owned NPU
                # storage so Python temporaries cannot be reclaimed while the
                # external kernels are still queued on the stream.
                self.runtime_req_ids_npu = torch.empty(
                    [self.max_num_topk_rows],
                    dtype=torch.int64,
                    device=device,
                )
                self.runtime_topk_indices_npu = torch.empty(
                    [self.max_num_topk_rows, self.topk],
                    dtype=torch.int32,
                    device=device,
                )
                self.runtime_stable_prefix_lens_npu = torch.empty(
                    [self.max_num_topk_rows],
                    dtype=torch.int32,
                    device=device,
                )
                self.runtime_block_table_npu = torch.empty(
                    [self.max_num_topk_rows, self.max_num_blocks],
                    dtype=torch.int32,
                    device=device,
                )
                self.runtime_row_to_req_npu = torch.empty(
                    [self.max_num_topk_rows],
                    dtype=torch.int64,
                    device=device,
                )
            else:
                self.lru_token_mark_workspace_npu = torch.zeros(
                    [self.lru_workspace_threads, self.max_model_len],
                    dtype=torch.int32,
                    device=device,
                )
                self.lru_token_pos_workspace_npu = torch.full(
                    [self.lru_workspace_threads, self.max_model_len],
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
                self.lru_epochs_npu = torch.zeros(
                    [self.lru_workspace_threads],
                    dtype=torch.int32,
                    device=device,
                )

    def _check_cpu_copy_runtime(self, capturing: bool) -> None:
        if self.copy_backend == COPY_BACKEND_CPU and capturing:
            raise RuntimeError(
                "The sparse KV CPU copy backend cannot run during ACL graph "
                "capture/replay. Start vLLM with --enforce-eager or select "
                "VLLM_ASCEND_SPARSE_KV_COPY_BACKEND=sparse_copy."
            )

    @staticmethod
    def _submit_cpu_copy(
        src_ptrs_cpu: torch.Tensor,
        dst_ptrs_cpu: torch.Tensor,
        sizes_cpu: torch.Tensor,
        direction: int,
    ) -> None:
        if src_ptrs_cpu.numel() == 0:
            return
        positive_sizes = sizes_cpu > 0
        if not bool(positive_sizes.all().item()):
            src_ptrs_cpu = src_ptrs_cpu[positive_sizes]
            dst_ptrs_cpu = dst_ptrs_cpu[positive_sizes]
            sizes_cpu = sizes_cpu[positive_sizes]
        if src_ptrs_cpu.numel() == 0:
            return
        torch.ops._C_ascend.swap_blocks_batch(
            src_ptrs_cpu,
            dst_ptrs_cpu,
            sizes_cpu,
            direction,
        )

    def _copy_h2d_descriptors_to_cpu(self) -> int:
        self.num_tokens_buffer_cpu.copy_(
            self.num_tokens_buffer_npu,
            non_blocking=False,
        )
        descriptor_count = int(self.num_tokens_buffer_cpu[0].item())
        descriptor_capacity = self.gvas_buffer_cpu.numel()
        if descriptor_count < 0 or descriptor_count > descriptor_capacity:
            raise RuntimeError(
                "MemFabric compute_lru_resident_addrs returned an invalid "
                f"descriptor count {descriptor_count}; capacity={descriptor_capacity}"
            )
        if descriptor_count == 0:
            return 0
        self.gvas_buffer_cpu[:descriptor_count].copy_(
            self.gvas_buffer_npu[:descriptor_count],
            non_blocking=False,
        )
        self.addr_buffer_cpu[:descriptor_count].copy_(
            self.addr_buffer_npu[:descriptor_count],
            non_blocking=False,
        )
        self.size_buffer_cpu[:descriptor_count].copy_(
            self.size_buffer_npu[:descriptor_count],
            non_blocking=False,
        )
        return descriptor_count

    def _submit_h2d_cpu_copy(self, descriptor_count: int) -> None:
        if descriptor_count <= 0:
            return
        self.copy_sizes_cpu[:descriptor_count].copy_(
            self.size_buffer_cpu[:descriptor_count]
        )
        self._submit_cpu_copy(
            self.gvas_buffer_cpu[:descriptor_count],
            self.addr_buffer_cpu[:descriptor_count],
            self.copy_sizes_cpu[:descriptor_count],
            COPY_DIRECTION_H2D,
        )

    def offload_new_kv(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        self._check_cpu_copy_runtime(capturing)
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = (
            k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        )
        num_v_slots = (
            v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        )
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                "Sparse KV offload CPU K/V pools have incompatible token capacities: "
                f"k={num_k_slots}, v={num_v_slots}"
            )
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        if self.copy_backend == COPY_BACKEND_SPARSE_COPY:
            try:
                dst_k_base = self.host_dva_by_data_ptr[k_cache_cpu.data_ptr()]
                dst_v_base = self.host_dva_by_data_ptr[v_cache_cpu.data_ptr()]
            except KeyError as error:
                raise RuntimeError(
                    "Sparse KV Host cache is not present in the registered-DVA "
                    "mapping created by register_kv_caches()."
                ) from error
        else:
            dst_k_base = int(k_cache_cpu.data_ptr())
            dst_v_base = int(v_cache_cpu.data_ptr())
        dst_k = dst_k_base + safe_slots * self.token_size_bytes_k
        dst_v = dst_v_base + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(
            self.token_size_bytes_v
        )
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        if self.copy_backend == COPY_BACKEND_SPARSE_COPY:
            result = offload.sparse_copy(
                self.d2h_src_ptrs_npu,
                self.d2h_dst_ptrs_npu,
                self.d2h_lengths_npu,
                self.d2h_size_npu,
                device,
            )
            if result not in (None, 0):
                raise RuntimeError(
                    f"memfabric D2H sparse_copy failed with result={result}"
                )
            return

        descriptor_count = 2 * token_count
        self.d2h_src_ptrs_cpu[:descriptor_count].copy_(
            self.d2h_src_ptrs_npu[:descriptor_count],
            non_blocking=False,
        )
        self.d2h_dst_ptrs_cpu[:descriptor_count].copy_(
            self.d2h_dst_ptrs_npu[:descriptor_count],
            non_blocking=False,
        )
        self.d2h_lengths_cpu[:descriptor_count].copy_(
            self.d2h_lengths_npu[:descriptor_count],
            non_blocking=False,
        )
        self._submit_cpu_copy(
            self.d2h_src_ptrs_cpu[:descriptor_count],
            self.d2h_dst_ptrs_cpu[:descriptor_count],
            self.d2h_lengths_cpu[:descriptor_count],
            COPY_DIRECTION_D2H,
        )
        # swap_blocks_batch receives raw pointer arrays, so PyTorch cannot
        # associate the async DMA with temporary contiguous K/V tensors. Keep
        # their storage alive until D2H completes in this correctness-first
        # experimental backend.
        torch_npu.npu.current_stream().synchronize()

    def onload_topk_kv(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        block_table: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
        skip_topk: bool = False,
    ):
        layer_id = self._get_offload_layer_id(layer_name)
        self._check_cpu_copy_runtime(capturing)
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "Sparse KV offload topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        if self.use_sparse_kv_runtime:
            runtime_device = self.topk_buffers_k[layer_id].device
            if (
                block_table.device != runtime_device
                or block_table.dtype != torch.int32
            ):
                raise ValueError(
                    "SparseKvLoadRuntime block_table must be an int32 tensor "
                    f"on {runtime_device}, got device={block_table.device}, "
                    f"dtype={block_table.dtype}"
                )
            if (
                block_table.ndim != 2
                or block_table.shape[1] != self.max_num_blocks
            ):
                raise ValueError(
                    "SparseKvLoadRuntime block_table must have shape "
                    f"[rows, {self.max_num_blocks}], got "
                    f"{tuple(block_table.shape)}"
                )
            block_table_npu = self.runtime_block_table_npu[:num_tokens]
            if token_to_req_npu is not None:
                row_to_req = self.runtime_row_to_req_npu[:num_tokens]
                row_to_req.copy_(token_to_req_npu[:num_tokens])
                torch.index_select(
                    block_table,
                    0,
                    row_to_req,
                    out=block_table_npu,
                )
            else:
                if num_tokens != num_reqs:
                    raise ValueError(
                        "SparseKvLoadRuntime requires token_to_req_npu when "
                        f"num_tokens ({num_tokens}) differs from num_reqs "
                        f"({num_reqs})"
                    )
                block_table_npu.copy_(block_table[:num_tokens])
            # skip_topk means that the attention layer reused cached TopK
            # indices.  The descriptor-free path deliberately re-runs Plan for
            # this layer so its per-layer LRU state is updated; there is no
            # global descriptor array to offset/reuse as in the legacy path.
            self._onload_topk_kv_runtime(
                layer_id,
                num_tokens,
                block_table_npu,
                topk_indices_npu,
                current_slots_npu,
                req_ids_npu,
                stable_prefix_lens_npu,
            )
            return
        if (
            self.lru_backend == LRU_BACKEND_CPU
            and layer_id in [0, self.mtp_layer_id]
        ):
            # metadata which are same across all layers, only compute/copy once in first layer.
            # last layer (mtp layer) may have different metadata, do not skip.
            if token_to_req_npu is not None:
                # spec decode case, expand block_table to actual num decode tokens.
                token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
                token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
                block_table_expanded = torch.index_select(
                    block_table, 0, token_to_req_npu[:num_tokens].to(torch.int64))
                self.block_table_expanded_cpu[:num_tokens].copy_(block_table_expanded, non_blocking=capturing)
            else:
                self.block_table_cpu[:num_reqs].copy_(block_table, non_blocking=capturing)
            self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
            self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                stable_prefix_lens_npu[:num_tokens],
                non_blocking=capturing,
            )

        if skip_topk:
            assert layer_id > 0, "No previous layer to reuse."
            gvas_offset = self.gvas_k_bases[layer_id] - self.gvas_k_bases[layer_id - 1]
            addr_offset = self.addr_k_bases[layer_id] - self.addr_k_bases[layer_id - 1]
            assert self.gvas_v_bases[layer_id] - self.gvas_v_bases[layer_id - 1] == gvas_offset, (
                "k/v gvas base delta mismatch."
            )
            assert self.addr_v_bases[layer_id] - self.addr_v_bases[layer_id - 1] == addr_offset, (
                "k/v addr base delta mismatch."
            )
            if self.lru_backend == LRU_BACKEND_CPU and (
                self.copy_backend == COPY_BACKEND_CPU
            ):
                self.gvas_buffer_cpu += gvas_offset
                self.addr_buffer_cpu += addr_offset
            else:
                self.gvas_buffer_npu += gvas_offset
                self.addr_buffer_npu += addr_offset
        else:
            if self.lru_backend == LRU_BACKEND_CPU:
                if token_to_req_npu is not None:
                    block_table_cpu = self.block_table_expanded_cpu[:num_tokens]
                else:
                    block_table_cpu = self.block_table_cpu[:num_reqs]
                topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
                topk_indices_cpu.copy_(
                    topk_indices_npu[:num_tokens],
                    non_blocking=capturing,
                )

                args = (
                    num_tokens,
                    self.lru_miss_count_cpu_list[layer_id][:num_tokens],
                    self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
                    self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
                    self.lru_req_ids_ptr,
                    self.lru_last_req_ids_ptrs[layer_id],
                    self.lru_topk_indices_ptr,
                    self.lru_stable_prefix_lens_ptr,
                    self.lru_slot_to_token_ptrs[layer_id],
                    self.lru_slots_ptrs[layer_id],
                    self.lru_current_slots_ptr,
                    self.lru_miss_count_ptrs[layer_id],
                    self.lru_miss_tokens_ptrs[layer_id],
                    self.lru_miss_slots_ptrs[layer_id],
                    block_table_cpu,
                    self.block_size,
                    self.token_size_bytes_k,
                    self.token_size_bytes_v,
                    self.gvas_k_bases[layer_id],
                    self.gvas_v_bases[layer_id],
                    self.addr_k_bases[layer_id],
                    self.addr_v_bases[layer_id],
                    self.lru_token_mark_workspace_ptr,
                    self.lru_token_pos_workspace_ptr,
                    self.lru_slot_workspace_ptr,
                    self.lru_miss_position_workspace_ptr,
                    self.lru_epochs_ptr,
                    self.gvas_buffer_cpu,
                    self.addr_buffer_cpu,
                    self.size_buffer_cpu,
                    self.num_tokens_buffer_cpu,
                    layer_id,
                )

                if capturing:
                    current_compute_stream = torch_npu.npu.current_stream()
                    subscribed_compute_streams = get_subscribed_compute_streams()
                    if current_compute_stream not in subscribed_compute_streams:
                        torch_npu.npu._subscribe_report(current_compute_stream)
                        subscribed_compute_streams.add(current_compute_stream)
                    torch_npu.npu._launch_host_func(
                        current_compute_stream,
                        self._onload_topk_kv_cpu,
                        args,
                    )
                else:
                    self._onload_topk_kv_cpu(args)
            else:
                if token_to_req_npu is not None:
                    block_table_npu = torch.index_select(
                        block_table,
                        0,
                        token_to_req_npu[:num_tokens].to(torch.int64),
                    )
                else:
                    block_table_npu = block_table[:num_reqs]
                self._onload_topk_kv_npu(
                    layer_id,
                    num_tokens,
                    block_table_npu,
                    topk_indices_npu,
                    req_ids_npu,
                    stable_prefix_lens_npu,
                )

        if self.copy_backend == COPY_BACKEND_SPARSE_COPY:
            if self.lru_backend == LRU_BACKEND_CPU and not skip_topk:
                self.sparse_copy_args_buffer_npu.copy_(
                    self.sparse_copy_args_buffer_cpu,
                    non_blocking=capturing,
                )
            result = offload.sparse_copy(
                self.gvas_buffer_npu,
                self.addr_buffer_npu,
                self.size_buffer_npu,
                self.num_tokens_buffer_npu,
                self.topk_buffers_k[0].device,
            )
            if result not in (None, 0):
                raise RuntimeError(
                    f"memfabric H2D sparse_copy failed with result={result}"
                )
        else:
            if self.lru_backend == LRU_BACKEND_NPU:
                descriptor_count = self._copy_h2d_descriptors_to_cpu()
            else:
                descriptor_count = int(self.num_tokens_buffer_cpu[0].item())
            self._submit_h2d_cpu_copy(descriptor_count)

        if self.lru_backend == LRU_BACKEND_CPU:
            current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
            current_slots_npu[:num_tokens].copy_(
                current_slots_cpu,
                non_blocking=capturing,
            )
        else:
            current_slots_npu[:num_tokens].copy_(
                self.lru_current_slots_npu[:num_tokens]
            )

    def _onload_topk_kv_runtime(
        self,
        layer_id: int,
        num_tokens: int,
        block_table_npu: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
    ) -> None:
        device = self.topk_buffers_k[layer_id].device
        npu_inputs = (
            ("req_ids", req_ids_npu, torch.int64),
            ("topk_indices", topk_indices_npu, torch.int32),
            ("stable_prefix_lens", stable_prefix_lens_npu, torch.int32),
            ("block_table", block_table_npu, torch.int32),
            ("current_slots", current_slots_npu, torch.int32),
        )
        for name, tensor, expected_dtype in npu_inputs:
            if tensor.device != device or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"SparseKvLoadRuntime input {name} must be a {device} "
                    f"{expected_dtype} tensor, got device={tensor.device}, "
                    f"dtype={tensor.dtype}"
                )
        if block_table_npu.ndim != 2 or block_table_npu.shape[0] < num_tokens:
            raise ValueError(
                "SparseKvLoadRuntime requires one block-table row per TopK "
                f"row, got rows={block_table_npu.shape[0]}, "
                f"required={num_tokens}"
            )
        # Copy into persistent manager-owned tensors before exposing raw
        # pointers to MemFabric.  These copies, Plan and Transfer are all
        # queued on the current stream; no Host synchronization is required.
        runtime_req_ids = self.runtime_req_ids_npu[:num_tokens]
        runtime_topk_indices = self.runtime_topk_indices_npu[:num_tokens]
        runtime_stable_prefix_lens = (
            self.runtime_stable_prefix_lens_npu[:num_tokens]
        )
        runtime_req_ids.copy_(req_ids_npu[:num_tokens])
        runtime_topk_indices.copy_(topk_indices_npu[:num_tokens])
        runtime_stable_prefix_lens.copy_(
            stable_prefix_lens_npu[:num_tokens]
        )
        block_table_npu = block_table_npu[:num_tokens]
        current_slots_output = current_slots_npu[:num_tokens]
        if not current_slots_output.is_contiguous():
            raise ValueError(
                "SparseKvLoadRuntime current_slots output must be contiguous"
            )

        result = offload.sparse_kv_load_runtime(
            runtime_req_ids,
            self.lru_last_req_ids_npu_list[layer_id][:num_tokens],
            runtime_topk_indices,
            runtime_stable_prefix_lens,
            self.lru_slot_to_token_npu_list[layer_id][:num_tokens],
            self.lru_slots_npu_list[layer_id][:num_tokens],
            current_slots_output,
            self.lru_miss_count_npu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_npu_list[layer_id][:num_tokens],
            self.lru_miss_slots_npu_list[layer_id][:num_tokens],
            self.sparse_kv_plan_workspace_npu,
            block_table_npu,
            self.gvas_k_bases[layer_id],
            self.gvas_v_bases[layer_id],
            self.addr_k_bases[layer_id],
            self.addr_v_bases[layer_id],
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.max_model_len,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(
                "memfabric sparse_kv_load_runtime failed with "
                f"result={result}"
            )

    def _onload_topk_kv_npu(
        self,
        layer_id: int,
        num_tokens: int,
        block_table_npu: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
    ) -> None:
        device = self.topk_buffers_k[0].device
        npu_inputs = (
            ("req_ids", req_ids_npu, torch.int64),
            ("topk_indices", topk_indices_npu, torch.int32),
            ("stable_prefix_lens", stable_prefix_lens_npu, torch.int32),
            ("block_table", block_table_npu, torch.int32),
        )
        for name, tensor, expected_dtype in npu_inputs:
            if tensor.device.type != "npu" or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"MemFabric NPU LRU input {name} must be an NPU "
                    f"{expected_dtype} tensor, got device={tensor.device}, "
                    f"dtype={tensor.dtype}"
                )
        req_ids_npu = req_ids_npu[:num_tokens].contiguous()
        topk_indices_npu = topk_indices_npu[:num_tokens].contiguous()
        stable_prefix_lens_npu = stable_prefix_lens_npu[:num_tokens].contiguous()
        block_table_npu = block_table_npu.contiguous()
        result = offload.lru_resident_compact(
            req_ids_npu,
            self.lru_last_req_ids_npu_list[layer_id][:num_tokens],
            topk_indices_npu,
            stable_prefix_lens_npu,
            self.lru_slot_to_token_npu_list[layer_id][:num_tokens],
            self.lru_slots_npu_list[layer_id][:num_tokens],
            self.lru_current_slots_npu[:num_tokens],
            self.lru_miss_count_npu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_npu_list[layer_id][:num_tokens],
            self.lru_miss_slots_npu_list[layer_id][:num_tokens],
            self.lru_token_mark_workspace_npu,
            self.lru_token_pos_workspace_npu,
            self.lru_epochs_npu,
            num_tokens,
            self.topk,
            self.topk_buffer_size,
            self.max_model_len,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(
                "memfabric lru_resident_compact failed with "
                f"result={result}"
            )

        result = offload.compute_lru_resident_addrs(
            self.lru_miss_count_npu_list[layer_id][:num_tokens],
            self.lru_miss_tokens_npu_list[layer_id][:num_tokens],
            self.lru_miss_slots_npu_list[layer_id][:num_tokens],
            block_table_npu,
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.block_size,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
            self.gvas_k_bases[layer_id],
            self.gvas_v_bases[layer_id],
            self.addr_k_bases[layer_id],
            self.addr_v_bases[layer_id],
            self.topk_buffer_size,
            num_tokens,
            self.topk,
            block_table_npu.shape[1],
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(
                "memfabric compute_lru_resident_addrs failed with "
                f"result={result}"
            )

    def _onload_topk_kv_cpu(self, args):
        # code that is incompatible with graph mode, compute here outside graph
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
        ) = args
        if self.tp_size > 1:
            # Graph callbacks are stream-ordered after TP0's D2H. In eager mode,
            # the blocking metadata copies above wait for that same stream first.
            self.tp_group.barrier()
        self.sparse_kv_offload_cpp.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.topk,
            self.topk_buffer_size,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        self.sparse_kv_offload_cpp.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.topk_buffer_size,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER
