# Phase 2 Spec —— vLLM #50484 Direct-DCP 路径的忠实 port

> 分支：`dsymm_dcp`（PR #33364）。阶段 1（rebase 到 main + 把 `symm_a2a`
> 迁回 `arg_groups` 命名空间模型）已完成。本文档覆盖阶段 2：从**已合入的**
> vLLM PR #50484 移植剩余的 direct-DCP 组件。

## 0. 目标

让 sglang 的 `symm_a2a` DCP 后端成为**已合入上游** vLLM #50484 direct-DCP
路径的忠实 port，而不是停在已废弃的 #48897 基线上。具体两件事：

1. **empty-shard LSE masking** —— 一个*正确性*修复（零 KV 的 rank / CUDA-graph
   padding 行会污染 LSE 加权合并）。
2. **direct q-gather** —— 一个*性能*收益（用一次 NVLS `multimem` 多播替换
   每层的 NCCL Q AllGather）。

两者都来自 #50484，它是 canonical 已合入路径，用的是与我们现有 `symm_a2a`
output/LSE kernel **同一套 symmetric-memory 族**的原语。foraxe 本人还在
#50484 里贡献了 q-gather 的 commit。

## 1. 上游谱系（搞清楚这个——它就是 review 论据）

| vLLM PR | 原语 | 状态 | 族 |
|---|---|---|---|
| #48897 | PyTorch symmetric-mem，P2P output/LSE | **closed**（被吸收） | symm-mem（我们的线）|
| **#50484** | symm-mem output/LSE **+ NVLS `multimem` q-gather/kv-gather + empty-shard masking** | **MERGED** 2026-08-10，WoosukKwon 合入 | symm-mem（canonical）|
| #50009 / sglang #32851 | CUDA **VMM peer mapping**，Shared-DCP | **停滞**（最后代码 2026-07-29，dirty）| VMM（另一条，更重）|

两个能堵住 reviewer 顾虑的事实：
- #48897 没被拒——它被**吸收**进了已合入的 #50484，而 #50484 用的就是我们已经
  移植的同一个 `dcp_direct_a2a_lse_reduce` kernel。
- foraxe 的相关贡献（q-gather「publish query into final NVLS buffer」）**落在
  #50484 里**。停滞的 #50009/#32851 是一条*不同的*、基于 VMM 的线，7 月底后
  就没动过。

**一句话定位：**「我们是 vLLM #50484 direct-DCP 路径（output + q-gather +
empty-shard masking）的忠实 sglang port；#50009/#32851 是一条正交的 VMM 线，
停滞 7 周，不在范围内。」

## 2. 背景：DCP 合并到底在算什么（面试核心）

**Decode Context Parallelism（DCP）** 对 MLA 把 KV cache 沿 context 维度切分到
`dcp_size` 张卡。每张卡只在自己那段 KV 上算 attention，产出（对它负责的 head）：

- `out_i` —— V 在它那段上的加权求和 partial
- `lse_i = log Σ exp(q·k)` —— 那段的 log-partition function（log-sum-exp）

完整的、数值正确的 attention 输出是跨 rank 的 **LSE 加权合并**：

```
lse_max = max_i lse_i
w_i     = exp(lse_i - lse_max)            # 每个 rank 的权重
out     = Σ_i w_i · out_i  /  Σ_i w_i
```

这是**精确**的（不是近似），且数值稳定。每层的**代价就是跨 rank 交换
`(out_i, lse_i)` partial + 合并**。四个后端只在「怎么交换」上不同：

| backend | 交换方式 | 依赖 |
|---|---|---|
| `ag_rs` | AllGather partials + ReduceScatter（2 次 NCCL 集合通信）| NCCL |
| `a2a` | 打包 (out+LSE) 的 fused All-to-All + 本地 Triton 合并 | NCCL |
| `fi_a2a` | FlashInfer MNNVL kernel | NVL72 fabric（稀有，GB200）|
| `symm_a2a`（我们）| **直接 P2P symmetric-memory 写 + fused CUDA 合并** | 全 NVLink 单机（常见）|

`a2a` 把 fp32 LSE 重解释成 output-dtype 沿 D 的列（`_lse_pack_dim`），让一个 A2A
同时携带 output 和 LSE → 「1 次 NCCL 调用/层 代替 2 次」。`symm_a2a` 连那一次
NCCL 都省了：每个 rank 把 `(out_i, lse_i)` 直接写进 peer 的 symmetric-memory
buffer，用 epoch 计数器发信号，一个 CUDA kernel spin-wait 后就地做 LSE 加权合并。

## 3. 组件 A —— empty-shard LSE masking（调查后：当前路径不触发，延后）

### 调查结论（2026-09-15）
实地查了 sglang DCP combine 的作用域，**A 在当前 MLA DCP decode 路径不触发**：

1. **combine 是 per-position 隔离的**。`dcp_lse_combine_triton`
   （`kernels/ops/attention/dcp_kernels.py:586`）grid = `(B, H_local)`，一个
   program 处理一个 (token, head) 位置，`lse_max = max over N ranks` 只在该位置
   内算。`symm_a2a` 的 CUDA kernel 同样 per-(token,head) block。所以 padding 行
   的垃圾 LSE 只影响它自己那个 program 的输出，而 padding 行输出下游丢弃 →
   **不污染真实行**。
2. **sglang 当前没有 sparse MLA**。`forward_mla.py` 是标准 MLA，每个 DCP rank 对
   每个真实 query 都持有非零 KV 段 → `seq_lens == 0` 在 decode 不发生。
3. **sglang 已有事后防御**：`dcp_a2a_lse_reduce` 里 `nan_to_num`
   （`comm.py:96-98`）清理 NaN/inf。

vLLM 的 `mask_dcp_empty_shards_` 针对的是它自己的 sparse-MLA 场景（top-k 跳 KV →
某 rank 对某 query 零 KV）。sglang 还没这条路径，且 sglang decode 的 query 是
1-token-per-request，没有 vLLM 那种 multi-token `query_start_loc` 映射。

### 处置
**延后**，不移植死代码。当 sglang 引入 sparse MLA（或任何 rank-per-query 零 KV
的路径）时再 port，那时 `seq_lens`/`query_start_loc` 语义才存在。**直接进入组件 B。**

### 原始分析（保留作参考）

当某个 DCP rank 对某个 query **没有 KV**（sparse-MLA top-k 跳过，或
**CUDA-graph 的 padding 行**——超出真实 query 数量的行）时，它的 `lse_i` 是
未定义的——通常是某个 finite 垃圾值（比如零初始化 buffer 的 `1.0`）。一个
finite 的假 LSE 会让那个 rank 在合并里拿到非零权重，**静默污染**输出。vLLM
bug #54305（GLM sparse-MLA strided output 崩溃）就是这类问题。

### #50484 怎么做（`vllm/v1/attention/ops/dcp.py`，commit `76b2e9d4`）
`mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)` —— 原地、约 15 行：
- 用 `torch.searchsorted(query_start_loc, ...)` 把每行 LSE 映射到所属 sequence。
- 把 `seq_lens == 0`（空 KV shard）的行 **以及** padding 行
  （`row_idx >= query_start_loc[-1]`）置 `-inf`。

`searchsorted` 形式（相对原来的 `repeat_interleave`）才是真正的修复点：它还能
抓到 graph-captured batch 里的 padding 行。

### 移植
- 纯 Python，**无 CUDA 依赖**，和 kernel 独立。
- 把 `mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)` 加到
  `python/sglang/srt/layers/dcp/comm.py`（或新开一个 `dcp_utils.py`）。
- 在 `dcp_a2a_lse_reduce(...)` **正前方**调用，对 `a2a`/`fi_a2a`/`symm_a2a`
  **全部生效**（这个 bug 和 backend 无关——任何 LSE 合并都受影响）。在
  `forward_mla.py` dispatch 前面接线：
  ```python
  # correct attn_output w.r.t. lse from other ranks
  if is_dcp_mla_decode_phase(forward_batch):
      attn_output = attn_output.view(-1, num_local_heads * attn_dcp_size, kv_lora_rank)
      mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)   # <-- 新增
      if get_in_autotune_dummy_run(): ...
      else: dcp_comm_backend = ...; if dcp_comm_backend in (...): dcp_a2a_lse_reduce(...)
  ```
- 要确认 `seq_lens` / `query_start_loc` 在该 call site 是否可见（来自
  `forward_batch` / attn metadata）。如果不可见，要传进来。

### 测试
- 单元测试：构造一批 真实 + 空 + padding 行的 `lse`，断言空/padding 行被置
  `-inf`、真实行不变。新风格 test 放 `test/registered/unit/layers/`。

## 4. 组件 B —— direct q-gather（性能，decode 路径）

### 为什么
DCP decode 里，**query 必须复制到所有 DCP rank**（每个 rank 都需要完整 Q 去
对自己的 KV 段算 attention）。sglang 现在有两个选项：

1. **NCCL AllGather** Q —— `forward_mla.py:638` 的
   `all_gather_q_for_mla_decode(q_nope, q_pe)`。每层一次集合通信延迟。
2. **`dcp_replicate_q_proj`** —— 每个 rank 冗余本地重算全头 Q 来跳过 AllGather。
   用计算换延迟。

#50484 的 q-gather 是有 NVLS 多播时的**第三条、严格更优**选项：每个 rank 把
自己的 Q-shard **写一次**，NVSwitch 的 `multimem` 指令一次复制到所有 peer 的
final buffer。无 NCCL、无冗余 GEMM。

### #50484 怎么做（`dcp_direct_q_gather.cu` + `dcp_direct_common.cuh`）
- `direct_dcp_q_gather_multimem_kernel`：一个 block 处理一个 (token, head-slice)。
  `multimem_store_16(query_mc_ptr + offset, local_q_shard)` 用一条指令把 16 字节
  payload 复制到所有 symmetric-buffer 副本。
- epoch 双缓冲（低位选 slot 0/1）+ `multimem_store_release_system` 信号 +
  对 peer 的 spin-wait。CUDA-graph 安全。
- host 入口 `direct_dcp_q_gather(...)` 校验 `query_mc_ptr != 0`（「requires
  multicast pointers」）。

### sglang 可复用现成 multimem 基建（大幅降低成本）
sglang **已经包装了** PyTorch `_SymmetricMemory` 的多播，在
`custom_all_reduce_v2.py`：
```python
multicast_ptr = int(symm_mem.multicast_ptr)   # NVLS multimem 基地址
self.has_multicast = multicast_ptr != 0
set_pull_multicast_blocks(...)                # NVLS pull-multicast 配置
TWO_SHOT_PULL algo, use_multicast=True
```
所以**不用从零**搬 `dcp_direct_common.cuh` 的 PTX wrapper——我们可以通过 sglang
已有的 `symm_mem` accessor 拿到多播指针，kernel 直接对着它写。（可能仍需
`multimem_store_release_system` / `wait_for_epoch` 这些小 helper，但我们 output
kernel 里已有等价的 `st_flag_release_u64` / `ld_flag_acquire_u64`。）

### 硬件门槛（关键诚实点）
| 原语 | 需要 | 硬件 |
|---|---|---|
| `symm_a2a` output/LSE（P2P 写）| 全单机 NVLink | H20 / H100 / A100 ×8 |
| q-gather（`multimem` 多播）| **NVSwitch + SM90+** | H20 / H100（DGX/HGX），**不含 A100** |

`multimem.st` 是 Hopper（SM90）指令，需要 NVSwitch fabric。所以：
- q-gather **gate on `has_multicast`**（就是 `custom_all_reduce_v2` 已经算了的那个
  标志）。不可用（如 A100）时回退到现成的 `all_gather_q_for_mla_decode` /
  `dcp_replicate_q_proj`。
- `symm_a2a` output 在 A100 上仍然可用（P2P）。无硬件退化。

### 移植
1. **Kernel**（`csrc/attention/dcp_direct_q_gather.cu`）：移植多播 kernel，但多播
   指针从 sglang 的 `symm_mem` 拿，不用 vLLM 专用 allocator。保留 epoch 双缓冲
   + spin-wait 的形状。
2. **Workspace**（`layers/dcp/comm.py`）：一个 `DirectDCPQGatherWorkspace`，分配
   symmetric query 接收 buffer + signal slot，key 风格同 `_SYMM_A2A_WORKSPACE_KEY`。
   在 `base_runner` 里、`_pre_initialize_symm_a2a_workspace` 旁边预 init，gate on
   `has_multicast && backend == symm_a2a`。
3. **Dispatch**（`forward_mla.py`）：在 Q-gather 点（~line 638）分支：
   ```python
   if symm_a2a_active and qgather_workspace.has_multicast:
       q_nope, q_pe = direct_dcp_q_gather(q_nope_local, q_pe_local, ...)  # 新增
   else:
       q_nope, q_pe = all_gather_q_for_mla_decode(...)                    # 回退
   ```
4. **Env/flag**：仿 #50484 的 tri-state `VLLM_USE_DIRECT_DCP_Q_GATHER`，做成 sglang
   的 `--dcp-direct-q-gather` arg（auto/force/off），遵循 `env-var-conventions` +
   命名空间模型（field 在 `arg_groups/fields/parallel.py`，校验在
   `parallel_hook.py`）。

### 测试
- 分布式 test 放 `test/registered/dcp/`：TP/DCP decode 用 `symm_a2a` + q-gather，
  断言与 `a2a` 后端 logit 一致（tolerance 内）。
- 一个 `has_multicast=False` 路径 test（mock）确认 NCCL 回退生效。

## 5. 组件 C —— direct kv-gather（延后）

`dcp_direct_kv_gather.cu` 把 chunked-context **prefill** 的 KV AllGather 换成
multimem 多播 + materialize。它只对 chunked-context prefill 路径有用（不对
decode），而且是单个体量最大的 kernel。**延后**到 B 落地、profiling 显示 prefill
KV-gather 是瓶颈后再做。不在本阶段。

## 6. 集成：阶段 2 后的完整 decode 调用路径

```
base_runner.__init__（CUDA-graph capture 之前）：
  ├─ _pre_initialize_fi_a2a_workspace()        # 若 backend == fi_a2a
  └─ _pre_initialize_symm_a2a_workspace()      # 若 backend == symm_a2a
       └─（新增）若 has_multicast，同时分配 q-gather 接收 buffer

forward_mla.py :: MLA decode，DCP 阶段：
  1. Q 准备：
     ├─（新增）若 symm_a2a + has_multicast：
     │      q_nope, q_pe = direct_dcp_q_gather(...)        # multimem 多播
     ├─ elif dcp_replicate_q_proj：冗余本地重算 Q
     └─ else：q_nope, q_pe = all_gather_q_for_mla_decode() # NCCL 回退
  2. 本地 MLA attention 在本 rank 的 KV 段上算 → attn_output, lse
  3. attn_output.view(-1, num_local_heads * dcp_size, kv_lora_rank)
  4.（新增）mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)   # 正确性
  5. if get_in_autotune_dummy_run()：短路（autotune guard）
     else：
       dcp_comm_backend = get_parallel().dcp_comm_backend
       if backend in (a2a, fi_a2a, symm_a2a)：
           attn_output = dcp_a2a_lse_reduce(          # 打包 (out+LSE) 交换
               out, lse, dcp_group,
               is_lse_base_on_e=is_lse_base_on_e,     # base-2 FlashInfer / base-e FA
               comm_backend=backend,
               ubatch_id=forward_batch.ubatch_id)     # epoch 双缓冲 slot
       else：cp_lse_ag_out_rs_mla(...)                # ag_rs 回退
```

`dcp_a2a_lse_reduce` 内部，`comm_backend == "symm_a2a"` dispatch 到
`_dcp_symm_a2a_lse_reduce`：
```
_dcp_symm_a2a_lse_reduce(out, lse, group, is_lse_base_on_e, ubatch_id)：
  1. 解析 workspace slot = epoch & 1（双缓冲；ubatch_id 选 stage）
  2. P2P dispatch kernel：每个 rank 把 (out_i, lse_i) 写进每个 peer 的
     symmetric 接收 buffer；release 信号（epoch）
  3. wait_lse_combine kernel：spin-acquire 所有 peer 的 epoch 信号，
     算 lse_max → softmax 权重 → 加权求和 out_i，就地。
     （WIP 优化：把 per-rank wait 并行化 + warp-shuffle LSE max）
  4. 返回合并后的 out（本 rank 拥有的 head）
```

## 7. CUDA-graph 安全（面试官爱钻的点）

- **workspace 在 capture 前预分配**：P2P/多播 buffer 分配是 driver call
  （不可捕获）。`base_runner._pre_initialize_symm_a2a_workspace` 在 init/warmup
  跑，早于任何 `cudaGraph` capture。（阶段 1 已把 gate 修成读
  `parallel.dcp_enabled`，不再读 `server_args`。）
- **epoch 双缓冲**：单调递增计数器，bit 0 选 staging slot 0 或 1。第 N 次写
  slot (N&1)，第 N-1 次的合并读 ((N-1)&1)——一张 graph 内两次执行不互相踩。
- **`ubatch_id` slot**：TBO / piecewise graph 在一个线程上交错多个 ubatch；
  `ubatch_id` 扩展 slot 空间，两个在飞的 ubatch 不写同一块 staging。
- **bounded spin + trap**：spin `kSpinLimit`（100M）次还不通就
  `asm("trap")`——在 CUDA graph 里把 hang 变成可定位 crash，而不是无声冻结。

## 8. 上线 & 风险

| 步骤 | 风险 | 缓解 |
|---|---|---|
| A. empty-shard masking | 低（Python，additive）| 单元测试 + 与 `a2a` parity |
| B. q-gather | 中（新 CUDA kernel，NVLS）| gate on `has_multicast`；NCCL 回退；与 `a2a` parity test |
| C. kv-gather | 延后 | — |
| 正确性 oracle | — | 真实 DCP decode 上 `symm_a2a` vs `a2a` 的 logit parity |

顺序：**A → B**。A 是正确性且不引入风险；B 依赖 A 的 masking 存在（q-gather
不改合并，但整条路径应该带上这个修复）。先推 A、CI 绿，再做 B。

## 9. 实现 B 之前要确认的 open question
- `seq_lens` / `query_start_loc` 在 `forward_mla.py` 合并 call site 是否可用，
  还是要从 attn metadata 传进来？（A 也会受影响。）
- q-gather 要 **NVLS 多播**路径（一次写 → 全员，需要 NVSwitch）还是也要
  **P2P fan-out** 回退（每个 rank 写 N-1 次，非 NVSwitch 全 mesh 也能用）？
  #50484 只有多播形式；P2P 回退能扩 A100 覆盖但额外工作。建议：只做多播 +
  NCCL 回退，与上游完全对齐。
- 复用 sglang `symm_mem.multicast_ptr` vs 搬 #50484 的 allocator？优先复用
  （保持单一 symmetric-memory 抽象）。
