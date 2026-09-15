# 阶段 2 direct q-gather 验证方案（分支 `verify/dcp-qgather`）

> 目标：验证 sglang port 的 vLLM #50484 direct q-gather（NVLS `multimem` 多播）
> 编译通过、调度正确、与 NCCL 路径数值一致。验证通过后再把改动并回
> `dsymm_dcp`。

## 0. 改动清单（验证对象）

| 文件 | 改动 |
|---|---|
| `kernels/aot/csrc/attention/dcp_direct_q_gather.cu` | 新增：multimem 多播 kernel + epoch 双缓冲 + spin-wait |
| `kernels/aot/csrc/common_extension.cc` | `direct_dcp_q_gather` op def+impl |
| `kernels/aot/include/sgl_kernel_ops.h` | 函数声明 |
| `kernels/aot/CMakeLists.txt` | 源文件注册 |
| `kernels/aot/python/sgl_kernel/attention.py` / `__init__.py` | Python wrapper + 导出 |
| `srt/layers/dcp/comm.py` | `DirectDCPQGatherWorkspace` + `init`/`get`/`estimate` |
| `srt/layers/dcp/__init__.py` | 导出 |
| `srt/model_executor/runner/base_runner.py` | `_pre_initialize_dcp_q_gather_workspace` |
| `srt/models/.../forward_mla.py` | Q-gather dispatch 分支（cat→gather→split）|

## 1. 静态验证（本机可跑，无需 GPU）

- [x] 所有改动 Python 文件 `ast.parse` 通过（comm / base_runner / forward_mla /
      sgl_kernel.attention / dcp.__init__）。
- [x] ruff 无新错（已有 F722 是 parallel.py 旧问题，非本次引入）。
- [ ] `python -c "import sglang.srt.layers.dcp"` 不报 import error（验证导出链）。
- [ ] `python -c "from sgl_kernel import direct_dcp_q_gather"` —— 需先构建
      sgl-kernel（见 §2），构建前会 `AttributeError`，属预期。

## 2. 构建验证（需 GPU 机器 + nvcc）

在 H20/H100(NVSwitch) 8 卡机器上：
```bash
cd /path/to/sglang && pip install -e .  # 触发 sgl-kernel AOT 构建
```
- [ ] `dcp_direct_q_gather.cu` 编译通过（关注 `multimem.st` PTX 在 SM90 的
      汇编、`at::cuda::OptionalCUDAGuard` / `getCurrentCUDAStream` 链接）。
- [ ] `python -c "import torch; torch.ops.sgl_kernel.direct_dcp_q_gather"` 不报
      undefined op。

## 3. 多播可用性探针（H20 实测）

`has_multicast` 是设计里的硬 gate。先确认 H20 上它为 True：
```python
import torch
from torch._C._distributed_c10d import _SymmetricMemory
# 在一个已 rendezvous 的 symm_mem 上：
print("multicast_ptr =", int(symm_mem.multicast_ptr))  # 非 0 = H20 支持 NVLS 多播
```
- [ ] H20 HGX：`multicast_ptr != 0`（预期 True，SM90+NVSwitch）。
- 若为 0：q-gather 自动回退 NCCL，不崩；但本阶段 q-gather 等于没启用，需换
  NVSwitch 机型。

## 4. 调度路由单元测试（单进程，mock，无需多卡）

验证 `forward_mla.py` 的 dispatch 分支选择正确：
- mock `get_dcp_q_gather_workspace()` 返回一个 `has_multicast=True` 的假
  workspace → 断言走 `gather` 路径。
- mock 返回 `None` 或 `has_multicast=False` → 断言走
  `all_gather_q_for_mla_decode`（NCCL）路径。
- 这隔离验证「cat→gather→split」的 packing 与 `all_gather_q_for_mla_decode`
  输出 layout 一致（`[B, ws*H, d_pe+d_nope]` split 回 q_pe/q_nope）。

> 注：此测试需能 import forward_mla 的模型路径，可能要在 meta device 上构造
> MLA 模块。若成本高，降级为对 `all_gather_q_for_mla_decode` 与
> `gather` 的 reference 一致性测试（见 §5）。

## 5. kernel 数值 parity（≥2 GPU + NVSwitch）

q-gather kernel 输出 vs `torch.distributed.all_gather` reference：
- 构造 `local_q [T, H_per_rank, D]`（bf16，随机），每 rank 一份。
- `direct_dcp_q_gather` 后 `final_q [T, ws*H, D]`。
- reference：`torch.cat([all ranks' local_q], dim=1)`（head 维拼接）。
- 断言 `torch.allclose(gathered, reference, atol=1e-3, rtol=1e-3)`。
- 覆盖：T=1（decode 常见）、T>1、不同 head_dim、CUDA-graph 捕获前后。

## 6. 端到端 parity（2-GPU，复用 SymmA2ATestBase）

`test/registered/dcp/test_dcp_q_gather.py`：TP2/DCP2，`symm_a2a`（含 q-gather）
vs `ag_rs`，CUDA-graph on/off，bf16，断言 logit parity。在 NVSwitch 2 卡上 q-gather
真实启用；在非 NVSwitch 2 卡上回退 NCCL（仍 parity 通过，但未测 kernel）。

- [ ] graph on：parity 通过。
- [ ] graph off（eager）：parity 通过。
- [ ] 服务器 stderr 出现 q-gather 启用/回退日志之一（证明路由到了新代码）。

## 7. 回归 & 边界

- [ ] `symm_a2a` 无 q-gather（mock `has_multicast=False`）仍 parity 通过
      （证明回退路径不退化）。
- [ ] A100（SM80，无 multimem）：`has_multicast=False`，回退 NCCL，不崩、parity 通过。
- [ ] `dcp_comm_backend != symm_a2a`（如 `a2a`）：q-gather workspace 不 init，
      forward_mla 走原 `all_gather_q_for_mla_decode`，零行为变化。

## 8. 通过判据

§1 全过 + §2 编译过 + §5 kernel parity + §6 端到端 parity（NVSwitch 机）全过 →
q-gather 验证通过，可把 `verify/dcp-qgather` 的 commit cherry-pick/fast-forward
回 `dsymm_dcp`。非 NVSwitch 机器只能验证到 §1+§2+§4+§7（回退路径），kernel
parity（§5）和端到端 q-gather 启用（§6）必须等 NVSwitch 机。
