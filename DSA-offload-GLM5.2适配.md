# DSA 稀疏卸载 GLM5.2 适配计划

> - 最后更新：2026-08-06
> - 当前阶段：**Phase 1 共享 indexer 适配完成；Phase 2 C8 仅保留 fail-fast（完整数据面为遗留事项）**
> - 适配目标：只修改 vLLM-Ascend，不修改 vLLM

## 1. 文档职责

本文档是将 DSA 稀疏卸载功能适配GLM 5.2模型
基线的持续维护记录，负责固定以下信息：

- 已确认的功能语义和首版支持边界；
- 新旧基线的关键差异；
- 各迁移阶段的优先级、设计约束、验收门槛和当前状态；

每完成一个阶段，都必须同步更新本文档的状态表、验收结果和变更记录。

## 2. 基线与目录

| 角色 | 本地仓库 | 基线提交 |
|---|---|---|
| DSA 特性（GLM5.1 版） | 本仓库 `vllm-ascend-v0.23.0-custom` 分支 | `6af99b372`（DSA 稀疏卸载落地末端） |
| 目标基线 | 本仓库 `official/main`（detached HEAD） | `9a52ca5fc`（已含 layerwise prefill offload / MTP / sparse C8） |
| 目标模型 | GLM-5.2 W4A4-mxfp4（bf16/fp16 Indexer） | HF `zai-org/GLM-5.2`（实测 config 见 §3；Indexer C8 尚未支持） |

> 说明：`dsa_offload/` 模块当前不在工作区（HEAD 已切到 `official/main`），
> 下文对 DSA 侧的引用均按 `git show 6af99b372:<path>` 读取的行号。

## 代码的疑问

1. 分成两个group 之后，tensor buffer物理空间是否会共享
2. mla中如何使用两个group
3. PREFILL/DENSE/ENTER/SPARSE 分别是什么？
4. 新的scheduler和worker分别做了什么事？怎么进行配合？
5. 数据投影添加了什么数据？为什么要添加？
6. 各个算子是什么作用
7. LIDU是做什么的？
8. 图模式？
9. chunked prefill

> Q1–Q9 的权威解答已在分析阶段完成（见会话梳理 / `dsa_offload_design.md`）。
> 其中受 GLM5.2 适配影响的条目在 §7 标注。

---

## 3. GLM5.2 实测 config（`zai-org/GLM-5.2`）

| 字段 | GLM5.2 实测 | DSA ABI 要求 | 判定 |
|---|---|---|---|
| `model_type` | `glm_moe_dsa` | — | ✅ |
| `architectures` | `GlmMoeDsaForCausalLM` | — | ✅（复用上游模型，无独立文件） |
| `index_topk` | 2048 | 2048 | ✅ |
| `index_head_dim` | 128 | 128 | ✅ |
| `kv_lora_rank` | 512 | 512 | ✅ |
| `qk_rope_head_dim` | 64 | 64 | ✅ |
| `index_n_heads` | **32** | ∈{32, 64} | ✅（用 32 档；GLM5.1 为 64） |
| `compress_ratios` | 无 | 必须无 | ✅ |
| `num_hidden_layers` | 78 | 未假设 | — |
| `first_k_dense_replace` | 3 | — | — |
| **`indexer_types`** | **78 层 = 21 `full` + 57 `shared`** | 假设每层独立 | ❌ **结构性冲突** |

`indexer_types` 精确分布（决定共享拓扑，是 ① 改造的输入）：

```text
full×3, 然后 [full + shared×3] 循环
full 层：0,1,2, 6,10,14,...,74   （共 21 层，建独立 Indexer cache）
shared 层：其余 57 层             （不建独立 Indexer，复用最近 full 层 top-K）
相关字段：index_topk_freq=4, index_skip_topk_offset=3, index_share_for_mtp_iteration=true
```

**两个「先按假设走 / 不确定」的确定答案：**

- 维度 → **全部落在 DSA ABI 内**，LIDU/KSC/SFA 算子**不用重调**。
- 共享 indexer → **确实存在且激进（57/78 层 shared）**，是适配的**核心结构性改动**。

---

## 4. 改动分级（按工作量排序）

### ① 共享 indexer 适配 —— 核心、结构性（阻塞项）

DSA 数据面假设「每层都跑 LIDU、每层有独立 indexer_cache」。GLM5.2 的 57 个
`shared` 层没有本层 indexer，需复用所属 `full` 层的 top-K。打在 DSA 三处
「每层独立 indexer」假设上（行号按 `6af99b372`）：

| DSA 现有逻辑 | 假设 | GLM5.2 现实 | 后果 |
|---|---|---|---|
| `build_dsa_kv_cache_groups`（kv_cache.py:176-181） | indexer 数 == resident 数，一一对应 | 78 resident 层 vs 20 indexer cache | 强校验直接报错 |
| `_bind_dsa_split_kv_caches` + `DSALayerOffloadContext.indexer_cache` | 每个 SFA 层绑一个 indexer_cache | 57 层无自己的 indexer_cache | 绑定失败 / LIDU 无输入 |
| `runtime` LIDU scratch + `resident_pool.cache_slots`（每层） | 每层跑 LIDU、每层一套 cache_slots | shared 层不跑 LIDU、复用 full 层 top-K | 78 层全建属浪费且语义错 |

改造方向：

- **plane 规划**：只给 21 个 `full` 层建 Indexer plane；校验放宽为「indexer 数 == full 层数」。
- **绑定**：shared 层的 `DSALayerOffloadContext.indexer_cache` 指向所属 full 层 cache。
- **数据面**：shared 层跳过 LIDU，复用所属 full 层的 `topk_index/topk_slots`，只跑 KSC + SFA-Offload；LIDU scratch 与 `cache_slots` 只对 full 层建。
- **检测**：`model_support.py` 新增读取 `indexer_types` / `index_topk_freq` 的能力字段（当前完全未读），识别 full/shared 拓扑。
- 参考：原生 SFA 已用 `_has_shared_indexer_layers`（`sfa_v1.py:101-105`）与
  `patch_deepseek_v2.py:40-56` 处理共享 indexer 前向，但那是 eager 非 offload 路径；
  DSA 的 plane 规划 / 绑定 / 算子链需自行对齐。

### ② Indexer C8 int8 —— 仅 fail-fast（完整数据面为遗留事项）

- 互斥性 ✅：与主 cache `enable_sparse_sfa_c8` 互斥，与 offload 的 `enable_sparse_li_c8`
  方向不冲突（indexer 驻留设备、不被卸载）。
- **关键事实**：DSA 的 LIDU（`npu_lightning_indexer_decode_update_out`）只收
  **fp16/bf16** 单 dense key（C++ dtype 门 `lightning_indexer_decode_update_torch_adpt.h:78-82`），
  **无 int8/fp8、无 scale 入参**；dense 路径虽有 `npu_lightning_indexer_quant`，但它不维护
  DSA 逐层 resident slot 状态，不能替代。→ 需要**新的 quant LIDU AscendC 算子**。
- **当前策略**：遇 C8 **启动期报错不进入**。不保留无法贯通 allocator/reshape/bind/write/decode
  的半成品 spec 或 runtime dispatch，避免未来误解除单个拦截后把 K+scale 字节错误 reshape 成
  单个 dense tensor。
- **三处拦截**：`bind_dsa_offload_context`、`get_kv_cache_spec` spec 发射门、
  `indexer_select_post_process` 均对 `enable_sparse_li_c8` 抛 `NotImplementedError`；
  `enable_sparse_sfa_c8` 继续维持 resident MLA bf16-only 拒绝。
- 待完整接入：实现 quant LIDU AscendC 算子，同时补齐设备相关 K/scale dtype、容量 spec、
  独立分配与 reshape、bind tuple、indexer 写入量化、prefill quant LI、decode query scale 和真机测试。
  这些必须作为一个可运行合同一次性落地，范围仍只覆盖 21 个 full 层 Indexer plane。

### ③ 量化 W4A4-mxfp4 —— 几乎零改动

- mxfp4 只量化 linear/MoE 的权重+激活（FP4 e2m1 packed uint8 + e8m0 group scale），
  **不碰 KV cache dtype、不碰 attention 结构**，对 offload 数据面透明。
- 无 attention scheme（`w4a4_mxfp4.py:34,117` 仅注册 linear/moe）；
  `glm_moe_dsa` 的 `packed_modules_model_mapping` 已映射（`modelslim_config.py:126-130`）。
- **唯一硬约束**：量化配置里 `kv_b_proj` 必须标 **FLOAT**
  （`AscendSFAImpl.process_weights_after_loading` 断言其不量化，`sfa_v1.py:682-683`）。

### ④ 零改动确认

调度 / 双 pool / plan-commit / scheduler→worker 投影 / 图模式 / chunked prefill /
DRAM 账本 / 模型能力识别（`index_topk=2048` 直接通过）—— 与精度、代次无关。
`block_size=128` 与 SFA kernel 上限一致；async / prefix cache / MTP / KV transfer /
PP / DCP-PCP 继续保持拒绝。

---

## 5. 量化方式 × 搬运数据（为什么量化对 offload 透明）

DSA 卸载/加载搬的**只有 MLA resident cache（nope+rope）**，Indexer plane 留在 HBM
从不被搬（`dump`/`KSC` 入参只有 resident nope/rope 与 DRAM nope/rope，见
`runtime.py:786-807`、`sfa_v1.py:2186-2189`）。而 DRAM arena 的 dtype **无条件照搬
resident cache dtype**（`dram_store.py:160-167`）。因此「量化是否改变搬运数据」
等价于「该量化是否改变 MLA resident cache 的 dtype」。

cache dtype 的真正决定逻辑（`model_runner_v1.py:5379-5423`）：

```text
k_cache_dtype = v_cache_dtype = spec.dtype            # 默认：模型 dtype（bf16）
if enable_fa_quant:                                   # FAKQuant
    k/v_cache_dtype = quant_config.get_kv_quant_dtype(...)   # 可改 K cache dtype
if use_sparse and current_sparse_sfa_c8:              # SFA C8
    k_cache_dtype = c8_k_cache_dtype                  # 主 MLA cache → int8/fp8
if use_sparse and has_indexer_cache and current_sparse_li_c8:
    dsa_k_cache = ...view(c8_k_cache_dtype)           # indexer cache → int8（+scale）
```

| 量化方式 | 改哪块 cache | MLA resident dtype | 搬运数据变吗 | 需要适配吗 |
|---|---|---|---|---|
| **W4A4 / mxfp4**（权重+激活） | 不碰任何 cache | bf16 | ❌ 不变 | 否 |
| **W8A8 / W4A8 / FP8**（权重+激活） | 不碰任何 cache | bf16 | ❌ 不变 | 否 |
| **Indexer C8**（`enable_sparse_li_c8`） | 仅 indexer → int8 | bf16 | ❌ 不变 | 否（indexer 不被搬） |
| **FAKQuant KV**（`fa_quant_type`） | MLA **K cache** → int8（V 不变） | **int8（K 半）** | ⚠️ 变 | 是 |
| **SFA C8**（`enable_sparse_sfa_c8`） | 主 MLA → int8/fp8 packed | **int8/fp8** | ⚠️ 变 | 是（且与 offload 互斥） |

结论：

- **权重量化（W4A4/W4A8/W8A8/FP8/mxfp4）只压权重+激活，KV cache 全程 bf16**，
  被搬的 MLA 数据字节布局一字节不变，对卸载/加载**完全透明**。
- **只有 KV cache 量化那一小类**（FAKQuant 的 `fa_quant_type`、SFA 的
  `enable_sparse_sfa_c8`）才改 MLA cache dtype、才需动搬运数据面（DRAM arena 加
  scale、dump/KSC 带 scale、容量重算）。
- 当前可运行目标是 **W4A4-mxfp4 权重量化 + bf16/fp16 Indexer**；权重量化不改变被搬的
  MLA resident cache，故卸载/加载通路零适配。Indexer C8 仍为显式未支持组合。
- 注：`enable_sparse_sfa_c8`（MLA int8）与 sparse offload **本就互斥**
  （`ascend_config.py:299-306`），「MLA int8 + offload」组合当前被显式拒绝，
  不在本适配范围。

---

## 6. 推进顺序（建议）

1. **确认量化配置 `kv_b_proj` 为 FLOAT**（否则 SFA 起不来）。
2. **做 ① 共享 indexer 适配**（结构性，决定能否绑定/起服务）：
   plane 规划 → 绑定 → LIDU 分组复用 → `model_support` 检测。
3. **做 ② Indexer C8**（在 ① 的 21 个 full 层上）。
4. 按 DSA 既有验收矩阵回归：UT → disabled 回归 → cache-init → eager →
   FULL decode graph → QA 精度对照（对照模型：GLM5.1；回归模型：DeepSeek-V3.2）。

---

## 7. 受影响的 Q1–Q9 条目

- **Q1（两 group tensor 是否共享物理空间）**：结论不变（物理独立），但 ① 后
  Indexer plane 只覆盖 full 层，group 内 layer 列表 ≠ resident group。
- **Q2（MLA 如何用两个 group）**：① 后 shared 层的「indexer」指向所属 full 层 cache，
  不再是每层一个。
- **Q6（各算子作用）/ Q7（LIDU）**：① 后 shared 层不跑 LIDU；② 后 LIDU 输入支持 int8。
- **Q3/Q4/Q5/Q8/Q9**（阶段机 / 调度 / 投影 / 图模式 / chunked prefill）：不受 GLM5.2 影响，结论原样成立。

---

## 8. 变更记录

- 2026-08-06：代码分析阶段完成。确认 GLM5.2 维度落在 DSA ABI 内（含 index_n_heads=32），
  识别出唯一结构性改动为「21 full + 57 shared」共享 indexer 拓扑；C8 定为 Indexer cache int8，
  与 offload 兼容但需改 LIDU 读取；W4A4-mxfp4 对 offload 透明（仅 kv_b_proj 需 FLOAT）。
- 2026-08-06：补充 §5「量化方式 × 搬运数据」对照表。确认卸载/加载只搬 MLA resident cache，
  权重量化（含 mxfp4）不改 cache dtype、对搬运透明；仅 KV cache 量化（FAKQuant/SFA C8）才需
  动搬运数据面，且 SFA C8 与 offload 互斥。本方案（W4A4 + Indexer C8）搬运通路零适配。
- 2026-08-06：**Phase 1 共享 indexer 结构性适配落地**（基线 `6af99b372`，DSA 特性分支）。
  核心设计：shared 层原位复用所属 full 层的 LIDU 输出 + host 侧 epoch/source 新鲜度守卫；
  不别名 cache_slots 行（`resident_pool` 78 行字节不变，57 个 shared 行为死行）、不复制副本；
  `runtime_layer_id` 仍按 78 个 resident 层稠密编号，仅 LIDU 输入（indexer_cache / cache_slots
  行）为 full 层独有。改动点：
    - `model_support.py`：加 `indexer_types/index_topk_freq` 能力字段 + `has_shared_indexer_layers`/
    `full|shared_indexer_layer_indices` 派生（`supported` 判定不变）。
    - `config.py`：`_validate_runtime_contract` 加 shared 拓扑交叉检查（indexer_types 长度 ==
    num_hidden_layers 且至少一个 full 层）。
    - `kv_cache.py`：`build_dsa_kv_cache_groups` 把「indexer 数==resident 数」放宽为「indexer 数
    ≤ resident 数 + transformer 下标子集检查」（orphan indexer 仍硬失败）。
    - `model_runner_v1.py`：`get_kv_cache_spec` 对 `has_indexer=False 且 skip_topk=True` 的 shared
    层放行发射 resident spec（其余缺 indexer 仍报错，C8 拒绝保留）；`_bind_dsa_split_kv_caches`
    plane 集合改子集语义、循环跟踪 `last_full_layer` 为 shared 层设 `selection_source_layer_id`、
    加 indexer 数与声明拓扑一致性交叉检查。
    - `runtime.py`：`DSALayerOffloadContext.indexer_cache` 改可选 + 新增 `selection_source_layer_id`；
    新增 `execute_shared_decode_selection`（不跑 LIDU，用本层 arena + 复用源 full 层 LIDU 输出跑
    KSC）；epoch 守卫在 `prepare_forward`/`prepare_execution_view`/`prepare_graph_capture` 三处
    每步入口重置。
    - `sfa_v1.py`：`bind_dsa_offload_context` 删 `use_index_cache` raise、改 context/impl 拓扑一致性
    校验；`forward` skip_topk 分支加 DSA decode 分流（shared 走 `execute_shared_decode_selection`，
    其余仍走原生 buffer 路径）；stash 门加 `isinstance(topk_indices, torch.Tensor)`；
    `record_attention_compute_start()` 从 `indexer_select_post_process` 移到 forward（覆盖无本层
    indexer 的 shared 层）。
    - `resident_pool.py`：仅补 shared 死行不变量注释，行为不变。
  UT：改写 `test_kv_cache.py` 组校验测试（21/78 混合通过、orphan/more-indexer 失败）、
  `test_model_support.py` 补 GLM-5.2 拓扑派生、`test_runtime.py` 补 shared 选择/守卫/误路由测试。
  本地无 torch/vllm，AST 全过；pytest 与 E2E 待 Ascend 机器执行。
  **对未启用 runtime IndexCache 的 all-full 模型（DeepSeek-V3.2/GLM-5.1）每步均为 no-op**；
  保留本层 Indexer 且 `skip_topk=True` 的 runtime IndexCache 组合继续启动期拒绝，避免 decode 误入
  checkpoint-shared 路径。
- 2026-08-07：对抗审查后将 Phase 2 收敛为纯 fail-fast。移除未接入 allocator/reshape/bind/write
  的 C8 spec、quant LIDU 占位函数和 runtime 虚假 dispatch；保留三处启动期拦截。Indexer C8
  后续必须以完整数据面和真机证据一次性接入，不能通过解除单个 `NotImplementedError` 启用。
- 2026-08-07：修复 runtime IndexCache 回归。只有 `indexer_types` 对当前层明确声明 `shared`、
  `skip_topk=True` 且无本层 Indexer 时才进入共享 LIDU 复用；保留本层 Indexer 的 skip_topk
  配置在 spec/bind 阶段明确拒绝。KV group 无条件校验 Indexer 层下标为 resident 子集。
- 2026-08-06：**提交前对抗评审（5 维度 + 3 视角核实）修复 3 项**：
  1. **GLM-5.2 拓扑更正**：以 HF `zai-org/GLM-5.2` config.json 为准，实为 **21 full / 57 shared**
     （full 在 0,1,2 后每隔 4 层一个，6..74），全文 20/58 → 21/57 更正；`test_model_support.py`
     fixture 断言同步改为 21/57。
  2. **declared-vs-emitted 拓扑校验收紧**（model_runner 绑定循环）：由「只比数量」改为「比较
     层下标集合」（`emitted_indexer != declared_full` 报缺失/多余），零成本抓住保持数量但错位的漂移。
  3. **skip_topk 门收紧**（spec 发射门 + `bind_dsa_offload_context`）：shared 层放行除
     `skip_topk and not has_indexer` 外，新增要求模型**声明了 shared 拓扑**
     （`capabilities.has_shared_indexer_layers`），否则 DeepSeek 等用 `index_topk_freq/pattern`
     （无 `indexer_types`）造出的 skip_topk 层会绕过全部 shared 校验、静默进入共享复用路径——
     恢复基线的响亮拒绝。
