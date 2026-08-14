# Sparse KV Cache Offload Guide

## Overview
This guide shows how to use Sparse KV Cache Offload, a long sequence inference optimization technique.

In long sequence inference scenario, KV cache size has become one of the inference bottlenecks. Although Layerwise KV Cache Offload proposed in RFC ([#33398](https://github.com/vllm-project/vllm/issues/33398)) can save KV cache NPU memory usage during prefill, we found out that Layerwise KV Cache Offload is not suitable for decode: time consumption of loading full KV Cache is too long to be overlapped by decoding, leads to a significant increase in TPOT.

To solve the problems above, we propose sparse KV cache offload, a sparse attention based long sequence inference optimization technique. Take [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) as an example, although we still need to store full KV cache, only topk (2048) of them are needed in sparse attention. By offloading the full KV cache to host, we are able to save most of the KV Cache NPU memory usage. And since we only need to onload the needed topk KV cache in each decode step, the h2d data transmission size is small (about ~2MB for each batch), thus h2d loading time is relatively low and make it available in decode phase.

For for information about Layerwise KV Cache Offload and Sparse KV Cache Offload, please refer to our RFC ([#48203](https://github.com/vllm-project/vllm/issues/48203)).

## Supported Scenarios

- Sparse KV Cache Offload only support sparse attention based model, such as [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) and [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).
- Currently Sparse KV Cache Offload only support PD disagregate scenario, and it can be only used in D node.
- Currently Sparse KV Cache Offload only support Tensor Parallel (TP).

Some other features that can be used together with Sparse KV Cache Offload are as follows:

|         | Eager | Graph | SpecDecode <br> (MTP) | Li C8 |
| ------- | ----- | ------| -----                 | ----- |
| **PCP** | ✅    | ✅   | ✅                   | ✅      |
| **DCP** | ✅    | ✅   | ✅                   | ✅      |

## How to use KV Cache offload

### Experimental LRU and copy backends

The sparse KV manager has three experimental backend switches:

| Environment variable | Values | Default | Meaning |
| --- | --- | --- | --- |
| `VLLM_ASCEND_SPARSE_KV_LRU_BACKEND` | `cpu`, `npu` | `cpu` | Run resident LRU compaction and address generation in the original OpenMP helper or the MemFabric NPU operators. |
| `VLLM_ASCEND_SPARSE_KV_COPY_BACKEND` | `sparse_copy`, `cpu` | `sparse_copy` | Run the MemFabric AIV `sparse_copy` kernel or submit H2D/D2H copies through the Ascend ACL runtime from the CPU. |
| `VLLM_ASCEND_SPARSE_KV_RUNTIME` | `0`, `1` | `0` | With the `npu` LRU and `sparse_copy` backends, opt into MemFabric's descriptor-free `SparseKvLoadRuntime` Plan -> Transfer path. It remains disabled by default until A5 validation is complete. |

The `npu` LRU and `cpu` copy implementations are currently limited to a
single TP rank. CPU copy is not graph-capturable and requires eager mode. To
test NPU LRU management without executing the MemFabric `sparse_copy` kernel:

```bash
export VLLM_ASCEND_SPARSE_KV_LRU_BACKEND=npu
export VLLM_ASCEND_SPARSE_KV_COPY_BACKEND=cpu

vllm serve ... --tensor-parallel-size 1 --enforce-eager
```

To select the descriptor-free runtime path, use the NPU LRU and SparseCopy
backends together. MemFabric then submits Plan and Transfer on the current NPU
stream without an intermediate host synchronization:

```bash
export VLLM_ASCEND_SPARSE_KV_LRU_BACKEND=npu
export VLLM_ASCEND_SPARSE_KV_COPY_BACKEND=sparse_copy
export VLLM_ASCEND_SPARSE_KV_RUNTIME=1
```

The CPU copy backend still requires MemFabric for the host pool and the two NPU
LRU operators. It only replaces the actual H2D/D2H copy operation. NPU-generated
copy descriptors are synchronized to the host before ACL memcpy submission, so
this mode is intended for correctness and compatibility experiments rather than
performance measurement.

You can enable Sparse KV Cache Offload by setting `sparse_kv_offload_config` in `additional-config`. You also need to specify `SFAPDCpuOffloadConnector` in `kv-transfer-config` for PD KV transfer. Refer to the following example:

```bash
vllm serve zai-org/GLM-5.2 \
    --host 0.0.0.0 \
    --port 8005 \
    --served-model-name model \
    --tensor-parallel-size 16 \
    --max-num-seqs 4 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.7 \
    --quantization ascend \
    --no-enable-prefix-caching \
    --additional-config '{"sparse_kv_offload_config": {"enabled": true, "topk_buffer_size": 4096, "dram_size_per_dp_GB": 128}}' \
    --kv-transfer-config "{
        \"kv_connector\": \"SFAPDCpuOffloadConnector\",
        \"kv_buffer_device\": \"npu\",
        \"kv_role\": \"kv_consumer\",
        \"kv_parallel_size\": 1,
        \"kv_port\": ${KV_PORT},
        \"kv_rank\": ${KV_RANK},
        \"kv_connector_extra_config\": {\"use_layerwise\": true}
    }"
```
