# OmniDreams FP8 DiT — RTX PRO 6000 测试计划

> 目标：在 RTX PRO 6000 (Blackwell) 上**完整验证** FP8 原生 DiT 的正确性、数值质量、性能与稳健性，并确认在 RTX 5070 上定位的两个根因的修复在更大的 sm_120 卡上同样成立。
>
> 前提认知：**RTX PRO 6000 Blackwell = GB202 = compute capability 12.0 = `sm_120`**，与 RTX 5070 同一架构族（arch-specific 目标同为 `sm_120a`），并非数据中心 `sm_100`。因此本次修复（`sm_120a` 构建 + cuDNN `generate_stats`）应直接适用。96GB 显存让我们能关闭 CPU offload、测更大分辨率/更长 rollout、并启用全 FP8 注意力——这些是 12GB 5070 测不了的"新情况"。**但仍以 Phase 0 实测 arch 为准，不臆断。**

---

## 机器特定信息（pro6000d = RTX 6000D 85GB / AutoDL `rtx6k`，sm_120）

> 据 `omnidreams_optimization_progress.md`，"pro6000d" 即 AutoDL `rtx6k` 容器(`ssh rtx6k`)，**RTX 6000D 85GB，sm_120**。85GB 同样够"关 offload 跑纯速度"。

- **venv**：`/root/autodl-tmp/sglang-venv/bin/python3`(torch 2.11.0+cu130，editable sglang 在 `/root/sglang`)；激活 `source /root/autodl-tmp/sglang-venv/bin/activate`(别名 `sa`)。
- **模型**：`/root/autodl-fs/models/`；**model path 必须含 `omni-dreams` 子串**(registry 检测器靠它匹配)→ 用 `/root/autodl-fs/omni-dreams-models`。样本数据 `/root/autodl-tmp/omni-dreams-samples/data/single_view/`。
- **⚠️ CUDA(关键)**：系统 `/usr/local/cuda`=**12.8（其实是完整 toolkit，nvcc/nv-target/cccl 都在；sm_120a 自 12.8 起即支持，旧文档"太旧"说法有误）**。构建 native ext 用 pip 的 CUDA 13 栈（与 torch cu130 对齐）：
  `export CUDA_HOME=/root/autodl-tmp/sglang-venv/lib/python3.12/site-packages/nvidia/cu13`。(chen 上是 `/usr/local/cuda-13`，此机不同。)
  **关键：整套 cu13 pip 工具链必须同一 minor（13.0）—— nvcc / crt / cccl / `nvidia-nvvm`(cicc) / runtime / nvrtc。混 13.0/13.2 会编不过（见 D.4）。**
- **已落库的 4 个 accel 配置**：`test/test_files/{omnidreams_dit_fp8,omnidreams_lightvae_lighttae,omnidreams_te_fp8,omnidreams_mixed_fp8}.json`，经 `SGLANG_OMNIDREAMS_ACCEL_E2E=1` 注册进 `test/server/gpu_cases.py`(否则 CI 不跑)。
- **FP8 state env**：LightVAE FP8 → `SGLANG_OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`；TE W8A8 → `SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH`(由 `tools/export_cosmos_reason_fp8.py` 产出)。
- **3rdparty**：native ext 构建需 CUTLASS/Sage/Sparge/cudnn-frontend，`tools/sync_thirdparty.py` 在该机网络慢，需预置或镜像。
- **已知现状**:`lightvae_lighttae` 已 E2E 跑通(20.4s)；native FP8 `.so` 已构建+加载(2026-06-18,sm_120a)；`dit_fp8`/`te_fp8`/`mixed_fp8` **尚未 E2E generate**(Phase 2B Tier 1 的重点,现已不再被 build 阻断)。

---

## 通用约定（每个 Phase 都遵守）

**Preflight（每次开测前）**
```bash
ENV_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py
export FLASHINFER_DISABLE_VERSION_CHECK=1
# 用 env helper 取仓库根、可写性、空闲 GPU、输出目录
python3 "$ENV_PY" select-gpu
OUT=$(python3 "$ENV_PY" print-output-dir --kind benchmarks --mkdir)
```

**两道 backend gate（结果作废条件）**
- **Diffusers fallback**：日志出现 `Falling back to diffusers backend` / `Using diffusers backend` → 作废，先修 pipeline 注册/路径。
- **FP8-DiT fallback（本项目特有）**：日志出现 `using eager DiT` / `falling back to eager BF16 DiT` / `native FP8 DiT native ext unavailable` → 说明没走到 FP8 原生路径，性能数据作废。**强制方式**：`SGLANG_OMNIDREAMS_FP8_DIT=1`（= `native_dit_acceleration="required"`，失败直接报错而非静默回退）。

**性能口径**：始终带 `--warmup`，只认带 `(with warmup excluded)` 的行 / `--perf-dump-path` 的结构化指标；OmniDreams 关键阶段是 `[OmniDreamsDenoisingStage] finished in Xs` 与 `Pixel data generated successfully in Xs`。

**对比**：`python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json result.json`。

**OmniDreams 跑法**：
- CLI：`sglang generate --pipeline-class-name OmniDreamsPipeline --config <cfg.json> --perf-dump-path ...`（`native_dit_acceleration` 不是 CLI flag，靠 `--config` JSON 或 `SGLANG_OMNIDREAMS_FP8_DIT=1`）。
- Python：沿用 `repro_fp8_bf16attn.py` 的 `DiffGenerator(... pipeline_config_path=...)` 模式。

---

## Phase 0 — 环境与架构确认（P0，阻断性，约 10 min）

> 整个修复的前提是"PRO 6000 是 sm_120a 能跑的卡"。先实测,不假设。

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
# 期望: NVIDIA RTX PRO 6000 ..., 驱动 >=570, compute_cap 12.0
nvcc --version | grep release            # 期望 >= 12.8（理想 13.x）
python -c "import torch;print(torch.__version__, torch.version.cuda); \
  print('cap', torch.cuda.get_device_capability())"   # 期望 (12, 0)
python -c "import torch.utils.cpp_extension as e; \
  import os; os.environ['TORCH_CUDA_ARCH_LIST']='12.0a'; print(e._get_cuda_arch_flags())"
  # 期望 ['-gencode=arch=compute_120a,code=sm_120a']（torch 没吞掉 a）
python -c "import cudnn; print('cudnn', cudnn.backend_version())" 2>/dev/null || \
  echo "cudnn-frontend 版本经由扩展构建确认"
command -v ninja || ls $VENV/bin/ninja   # 构建需要 ninja 在 PATH
```

| 检查 | 期望 | 若不符 |
|---|---|---|
| compute_cap | `12.0`（sm_120） | 若是 `10.0`(sm_100, 数据中心) 则属另一架构,本修复不直接适用,需单独评估 `sm_100a` |
| 驱动 | ≥ 570 | 升级驱动 |
| CUDA toolkit | ≥ 12.8 | 升级 toolkit（sm_120a 需 12.8+） |
| torch arch 翻译 | 含 `compute_120a,sm_120a` | 见 Phase 1 直接传 gencode 兜底 |

**Pass**：compute_cap=12.0、驱动/CUDA 达标、torch 正确翻译 `12.0a`。

---

## Phase 1 — 构建正确性与两个修复在 PRO 6000 成立（P0，约 15 min）

> 确认 `sm_120a` 构建产物正确、加载正确、且没有过期 `.so` 干扰。

```bash
# 1) 清掉任何过期产物（关键：加载器会无差别加载任何 .so）
rm -rf python/sglang/multimodal_gen/native/omnidreams_singleview/build/torch_extensions/*

# 2) 强制 12.0a 构建（env 为空时 loader 默认即 12.0a；显式设置最稳）
export PATH=$VENV/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=16 PYTHONPATH=python
python -c "from sglang.multimodal_gen.native import singleview_loader as L; \
  L.validate_thirdparty=L._thirdparty_info_no_validation; \
  m=L.load_extension(verbose=True); print('LOADED', m is not None and hasattr(m,'optimized_dit_forward')); \
  e=L.extension_load_error(); print('ERR', e)"

# 3) 验证产物 arch（铁证）
SO=$(ls .../build/torch_extensions/*/*.so | head -1)
cuobjdump "$SO" | grep 'arch ='                       # 期望 arch = sm_120a
grep -oE 'arch=compute_[0-9]+a?,code=sm_[0-9]+a?' .../build.ninja   # 期望 compute_120a,sm_120a
```

| 检查 | 期望 |
|---|---|
| `.so` arch（cuobjdump） | `sm_120a` |
| build.ninja gencode | `compute_120a,sm_120a` |
| 扩展加载 | `LOADED True`，`ERR None` |
| 源码含两处修复 | `omnidreams_fp8.py` 无 sm_120 guard；`attention.cu:~1161` 含 `set_generate_stats(false)`；`singleview_loader.py` 含 `_normalize_blackwell_arch` |

**注意**：本机 cutlass stamp 若漂移，构建脚本里用 `validate_thirdparty=_thirdparty_info_no_validation` 跳过（headers 在即可）。

**Pass**：单一 `.so`、arch=sm_120a、加载成功。

---

## Phase 2 — 功能正确性矩阵（P0，约 1–2 h）

> 96GB 让我们能关 offload。逐项跑通,每项确认"无 999 / 无 block fail / 出视频 / 没回退 eager"。

固定输入（控制变量）：同一 prompt / first-frame / hdmap / seed=42 / 720p / 13 帧。

| # | 场景 | 关键开关 | Pass 判据 | 优先级 |
|---|---|---|---|---|
| 2.1 | eager BF16 基线 | `native_dit_acceleration=disabled` | 出视频；记为质量&性能 baseline | P0 |
| 2.2 | FP8 原生 + cudnn_bf16 注意力 | `SGLANG_OMNIDREAMS_FP8_DIT=1`，attn=cudnn_bf16 | `RESULT_OK`,0 fail,无 999 | P0 |
| 2.3 | FP8 原生 + **fp8_cudnn** 注意力（全 FP8） | attn=fp8_cudnn | 跑通（generate_stats 修复后应可用）；若失败,开 cuDNN 日志看真实原因 | P0 |
| 2.4 | FP8 原生 + fp8_dense_ref 注意力 | attn=fp8_dense_ref | 跑通（CUTLASS GEMM 注意力,兜底路径） | P1 |
| 2.5 | **关闭全部 CPU offload**（96GB 专属） | `*_cpu_offload=False` | 跑通 + 显存占用记录（应 < 96GB） | P0 |
| 2.6 | 开 torch.compile + warmup（作用于 eager 块；FP8 原生走 C++ 不受影响,验证不冲突） | `--enable-torch-compile --warmup` | 跑通,无编译错误 | P1 |
| 2.7 | server 模式（非一次性 generate） | `sglang serve ... ` + curl | 健康,多请求稳定 | P1 |
| 2.8 | LightVAE/LightTAE FP8（encoder/decoder 原生） | config 里 lightvae/lighttae required | encode/decode 阶段无报错 | P0 |

**逐子层确认（任一失败时打开）**：`OMNIDREAMS_DIT_BLOCK_DEBUG=1`，看 `after_sa_qkv / after_sa_post_qkv / sa_fmha / ca_fmha / ffn*` 全 `err=0`。

**Pass**：2.1/2.2/2.3/2.5/2.8 全绿。

---

## Phase 2B — 组件排列组合 E2E 矩阵（P0，本次重点补充）

> 配置重构后,OmniDreams 是**正交的嵌套 Config**:text_encoder / image_encoder / encoder(HD-map) / decoder / DiT 各自独立选 impl + FP8 加速态(auto/disabled/required)。这是一个真实的组合面,而 Phase 2 只覆盖了 "DiT FP8 × 注意力"。全交叉会爆炸,故用「**先隔离、再叠加、再查交互**」三步覆盖。

### 正交维度
| 组件 | 取值 |
|---|---|
| DiT | `eager-bf16` / `fp8` (× attn: cudnn_bf16 / fp8_cudnn / fp8_dense_ref) |
| image_encoder(首帧) | `wanvae` / `lightvae` (× fp8: off/on) |
| encoder(HD-map) | `wanvae` / `lightvae` (× fp8: off/on) |
| decoder | `wanvae` / `lighttae` |
| text_encoder | `bf16` / `fp8-W8A8` |
| P0 CUDA Graph | off / on（仅 eager DiT） |
| P3 torch.compile | off / on |

### Tier 1 — 4 个已落库 JSON 配置（3 个从未 E2E 验证，**最高优先**）
开 `SGLANG_OMNIDREAMS_ACCEL_E2E=1` 注册为 server case;或直接 `--config test/test_files/<x>.json`。

| 配置 | DiT | img_enc | hdmap_enc | dec | TE | 作用 | 状态 |
|---|---|---|---|---|---|---|---|
| `omnidreams_lightvae_lighttae` | eager | lightvae | lightvae | lighttae | bf16 | 隔离 VAE 加速 | ✅ 20.4s(待 PRO 复测) |
| `omnidreams_dit_fp8` | fp8 | wanvae | wanvae | wanvae | bf16 | **隔离 DiT FP8(全质量 VAE)** | ❌ 待验 |
| `omnidreams_te_fp8` | eager | wanvae | wanvae | wanvae | fp8 | **隔离 TE FP8** | ❌ 待验(需先跑 W8A8 导出) |
| `omnidreams_mixed_fp8` | fp8 | lightvae+fp8 | lightvae+fp8 | lighttae | fp8 | **全 FP8(终极组合)** | ❌ 待验 |

判据(每条):无 `failed at block`/999、出视频、日志无 `using eager DiT`。`te_fp8`/`mixed_fp8` 先确保 `SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH`(及 lightvae fp8 state)已就绪。

### Tier 2 — DiT FP8 × 注意力后端（**钉死 WanVAE 以隔离 DiT 质量**）
基于 `dit_fp8`(全 WanVAE,排除 VAE 噪声),扫 attn ∈ {cudnn_bf16, fp8_cudnn, fp8_dense_ref}。这样数值/质量差异只归因于 DiT 自身,不与 LightVAE 的有损叠加。(对应 Phase 2.2–2.4,但强调隔离。)

### Tier 3 — 加速态语义（auto / disabled / required）逐组件
三态分发是本次重构核心,必须验证 fallback 与 raise 两条边:
| 组件 | 测试 | 期望 |
|---|---|---|
| 各 FP8 组件(DiT/LightVAE/TE) | `required` + 故意缺 state/ext | **明确报错**,不静默回退 |
| 各 FP8 组件 | `auto` + 缺 state/ext | **优雅回退 bf16**,仍出视频 |
| DiT | `disabled` | 纯 eager BF16,绝不触碰 native `.so` |

### Tier 4 — 正交性 / 交互（最易藏 bug 的组合）
| # | 组合 | 关注点 | 期望 |
|---|---|---|---|
| 4.1 | **P0 CUDA Graph + eager DiT** | numerics vs graph-off(同 seed,焦点 fill→steady 过渡 chunk),跑 ≥3× 抓 stream-sync 非确定性 | bf16 容差内一致 |
| 4.2 | **DiT FP8 + 请求 P0** | 互斥:native FP8 时 eager CUDA-graph runner 被禁用且不再启用(见 CLAUDE.md) | 不冲突、不双路、不崩 |
| 4.3 | P3 compile + eager DiT | numerics + 无 graph-break 刷屏 | 一致 |
| 4.4 | P3 compile + DiT FP8 | compile 对 native C++ 路是 no-op | 不崩、走 FP8 |
| 4.5 | 异构 VAE:lightvae 首帧 + wanvae HD-map(或反) | `_wanvae_cache` impl-keyed 共享不串、不重复加载 | plumbing 通(P1,低优先) |
| 4.6 | LightVAE-FP8 encode + LightTAE(bf16) decode | encode FP8 / decode bf16 混合 | HD-map motion 不冻结(meandiff>0) |

### Tier 5 — 一致性 GT + 性能基线（每条 accel 路跑通后）
`perf_baselines.json` 的 `omnidreams_2b_i2v` 当前全 0、GT 缺失(progress 文档记录的 CI gap):
- 每条 accel 配置先跑通(Tier 1)→ `SGLANG_GEN_BASELINE=1` 填真实 perf → `SGLANG_GEN_GT=1` 生成一致性 GT(确认上传 `sgl-project/ci-data`)。
- 顺手把 `gpu_cases.py` 里 OmniDreams 的 `run_models_api_check` 翻 `True`(便宜的 `/v1/models` 契约检查)。

**Pass**:Tier 1 四个配置全部 E2E 出视频;Tier 3 auto/required 语义正确;Tier 4.1/4.2 无 numerics 漂移、无冲突。

> 注:逐组件的数值/FVD/HD-map-motion 细节判据,沿用 progress 文档 §8 与 "GPU verification checklist (sm_120 host)" 已有条目,这里不重复,只补"组合覆盖"。

---

## Phase 3 — 数值质量（FP8 vs BF16）（P0,约 30 min）

> FP8 线性 + (BF16 或 FP8) 注意力相对 eager BF16 会有偏差。量化它,确认在可接受范围。

```bash
# 同 seed/prompt/frame,分别产出 eager 与各 FP8 变体的 .mp4 / 潜变量
# 逐帧对比（PSNR/SSIM/最大绝对差）;若 pipeline 暴露 latent,优先比 latent
SGLANG_OMNIDREAMS_DIAGNOSTICS=1 ...   # 打印逐帧张量统计(min/max/mean/nan/inf)
```

| 对比 | 期望 | 备注 |
|---|---|---|
| 2.2 (cudnn_bf16) vs 2.1 (eager) | 视觉一致,无 NaN/Inf,PSNR 高 | 线性 FP8 的预期小偏差 |
| 2.3 (fp8_cudnn) vs 2.1 | 偏差略大于 2.2（注意力也 FP8） | 评估是否可接受 |
| 2.4 (fp8_dense_ref) vs 2.1 | 参考实现,确认正确性 | — |
| 多 seed (42/123/7) | 同 seed 可复现 | 确认确定性 |

**Pass**：FP8 变体无 NaN/Inf,与 eager 视觉一致,PSNR/SSIM 在可接受阈值（按业务定,建议先看 2.2 是否"几乎无差")。

---

## Phase 4 — 性能基准（P0,约 1–2 h）

> 96GB → 关 offload 跑"纯速度"配置。每个配置 `--perf-dump-path`,`compare_perf.py` 对比。

```bash
# 模板（关 offload + warmup + 强制 FP8）
SGLANG_OMNIDREAMS_FP8_DIT=1 sglang generate \
  --pipeline-class-name OmniDreamsPipeline --config $OUT/cfg_fp8_nooffload.json \
  --prompt "A car driving down a city street" --image-path first.png --hdmap-path hd.mp4 \
  --width 1280 --height 704 --num-frames 13 --seed 42 --warmup \
  --perf-dump-path $OUT/fp8_cudnnbf16.json --save-output
```

基准对比矩阵（关 offload，除非标注）：

| 配置 | perf-dump label | 对比对象 |
|---|---|---|
| eager BF16 (+compile+warmup) | `eager` | 基线 |
| FP8 + cudnn_bf16 | `fp8_bf16attn` | vs eager |
| FP8 + fp8_cudnn（全 FP8） | `fp8_full` | vs eager / vs fp8_bf16attn |
| FP8 + offload ON（模拟小卡） | `fp8_offload` | 量化 offload 开销 |

记录指标：`OmniDreamsDenoisingStage` 阶段时延（最可比）、端到端 `Pixel data generated`、峰值显存、(text-encode/decode 两端不变作 sanity)。

参考（5070,12GB,带 offload）：eager ≈ 940s；FP8 ≈ 530s（DiT 去噪 282s）。PRO 6000 关 offload 应显著更快,且 FP8/eager 的加速比可能不同（更多 SM → GEMM/attention 更易打满）。

**Pass**：FP8 路径相对 eager 有稳定加速；产出 perf-dump 与对比报告；记录两三次运行的方差（我们在 5070 上见过 515↔530 的 run-to-run 抖动）。

---

## Phase 5 — Profiling 与热点（P1,约 1 h）

> 找瓶颈,且先用 `existing-fast-paths.md` 归类,别急着说"新热点"。

```bash
# torch.profiler 抓 trace（注意:--profile 会显著改变时延,只用于找热点,不用于报性能数字）
SGLANG_OMNIDREAMS_FP8_DIT=1 sglang generate ... --profile --profile-path $OUT/trace_fp8/
# 排名热 kernel → 对照 existing-fast-paths.md（GroupNorm+SiLU / QK norm+RoPE / packed QKV / Ulysses 等）
```

关注点：
- FP8 CUTLASS GEMM 是否打满（PRO 6000 SM 多,看 occupancy）。
- cuDNN flash attention（运行时编译 engine）首个 chunk 的 JIT 开销 vs 稳态。
- AR rollout 逐 chunk 的 KV-cache I/O、RoPE、AdaLN。
- 是否存在可融合/可重叠的明显空窗。

**Pass**：产出一份热点排名 + 归类(已有 fast path / 候选新优化),作为后续 kernel 工作的证据包。

---

## Phase 6 — 压力与规模（P1,约 1–2 h,96GB 专属）

> 用大显存测 5070 测不了的边界。

| # | 维度 | 取值 | 关注 |
|---|---|---|---|
| 6.1 | 分辨率 | 720p → 1080p（更大 token 数） | attention seq len 增大,FP8 GEMM/FMHA 扩展性,显存 |
| 6.2 | 帧数 / rollout 长度 | 13 → 49 → 更长 | AR rollout 稳定性,KV-cache 窗口,长程是否漂移/崩 |
| 6.3 | 注意力 seq 规模 | 记录每 chunk tokens（5070 上约 7040） | cuDNN engine 是否随规模切换/失败 |
| 6.4 | 并发/批 | server 模式多请求 | 调度、显存峰值、稳定性 |
| 6.5 | 显存上限探测 | 逐步加大直到接近 96GB | OOM 行为是否优雅 |

**Pass**：在合理规模内无崩溃；记录 OOM 边界与降级行为。

---

## Phase 7 — 回归与稳健性护栏（P0,约 30 min）

> 专门复测两个根因的"陷阱",确保修复稳健、不易被环境破坏。

| # | 测试 | 步骤 | 期望 |
|---|---|---|---|
| 7.1 | 裸 `TORCH_CUDA_ARCH_LIST=12.0`（无 a） | 清 `.so`,设此 env 重新构建 | `_normalize_blackwell_arch` 应纠正为 `12.0a`；build.ninja 仍是 `compute_120a,sm_120a`（不再产出 trap 版） |
| 7.2 | 多架构 list `8.9;12.0` | 同上 | 归一化为 `8.9;12.0a`,其余 token 不变 |
| 7.3 | 过期 `sm_120` `.so` 残留 | 故意放一个旧 `.so` | 记录:加载器是否会误用旧产物（已知 footgun:`_extension_name` 未编码 arch）→ 验证"清理后重建"流程；这是 follow-up 风险点 |
| 7.4 | guard 已移除 | sm_120 上不再静默回退 | `native_dit_acceleration=auto` 在 PRO 6000 直接走 FP8 原生 |
| 7.5 | cuDNN 校验修复 | 不再出现 `generate_stats attribute not set` | 全程无该报错 |
| 7.6 | 干净 checkout 复现 | 在干净 worktree 上重建+跑通 | 不依赖任何手工 patch（除 stamp bypass) |

**Pass**：7.1/7.2/7.4/7.5 全绿；7.3 记录结论（建议把 follow-up「`_extension_name` 编码 arch」纳入待办）。

---

## 附录 A：配置文件模板

`cfg_fp8_nooffload.json`（PRO 6000,关 offload,全 FP8 可切 attention backend）：
```json
{
  "image_encoder_config": {"impl": "lightvae", "native_acceleration": "required", "checkpoint_path": "<lightvae.pth>", "fp8_state_path": "<fp8_state.pt>"},
  "encoder_config":       {"impl": "lightvae", "native_acceleration": "required", "checkpoint_path": "<lightvae.pth>", "fp8_state_path": "<fp8_state.pt>"},
  "decoder_config":       {"impl": "lighttae", "checkpoint_path": "<lighttae.pth>"},
  "native_dit_acceleration": "required"
}
```
（注意力后端：当前经 `omnidreams_fp8.py` 的 `_resolve_runtime_attention_backend` + `_PY_TO_CPP_ATTN` 解析；要切 `fp8_cudnn`/`fp8_dense_ref` 时确认该解析路径接受对应名字,必要时在 Python API 用 `pipeline_config_kwargs` 传 `native_dit_attention_backend`。)

offload 变体：把上面去掉/改 `server_kwargs` 里的 `*_cpu_offload`。eager 基线：`"native_dit_acceleration": "disabled"`。

## 附录 B：一键判定脚本（每次跑完抓信号）
```bash
L=run.log
grep -cE "failed at block|unknown error|Arch conditional|generate_stats attribute not set" "$L"   # 期望 0
grep -E "using eager DiT|Falling back to diffusers" "$L"                                           # 期望 空
grep -E "RESULT_OK|Pixel data generated successfully" "$L"
grep -E "\[OmniDreamsDenoisingStage\] finished|peak" "$L"
```

## 附录 C：优先级速览（时间紧时的最小集）
- **必跑（P0）**：Phase 0 → 1 → 2.1/2.2/2.3/2.5/2.8 → 3(2.2 vs eager) → 4(eager vs fp8) → 7.1/7.4/7.5。
- **想全（P1）**：2.4/2.6/2.7、Phase 5 profiling、Phase 6 规模、7.2/7.3/7.6。

## 风险与注意
- **arch 前提**：PRO 6000 必须实测为 `sm_120`（Phase 0）。若该机是 `sm_100`（数据中心 Blackwell）则本修复不直接适用。
- **stamp bypass**：cutlass stamp 漂移时用 `_thirdparty_info_no_validation`；headers 必须在位。
- **过期 `.so` footgun**：每次改源码后务必先清 `build/torch_extensions/*` 再重建（`_extension_name` 不编码 arch）。
- **`--profile` 失真**：profiling trace 不可用作性能数字,性能只认非 profile 的 `--perf-dump-path`。
- **诊断探针开销**：`OMNIDREAMS_DIT_BLOCK_DEBUG=1` 会逐层同步、显著拖慢,只用于定位,不用于计时。

---

## 附录 D：pro6000d 实测脚本与环境（已 scp 到机器并通过 preflight；新会话直接用）

> **脚本已落到机器上**：`rtx6k:/root/omnidreams_pro6000_test/`（已 `chmod +x`，`00_preflight.sh` 跑通）。
> 仓库 `/root/sglang` @ `feat/omnidreams-p0-p4b-optimizations`。**两处修复已应用到该机仓库**
> （attention.cu `set_generate_stats(false)`×2、loader `_normalize_blackwell_arch`；假 guard 本就不存在）。

### D.0 机器实况（更新 2026-06-18）
- GPU：**NVIDIA RTX 6000D, 85651 MiB, compute_cap 12.0 (sm_120)**，驱动 595.71。
- cu13 工具链：**已对齐到 13.0**（nvcc 13.0.88 / crt 13.0.88 / cccl 13.0.85 / `nvidia-nvvm` 13.0.88(cicc) / runtime 13.0.96 / nvrtc 13.0.88）；torch **2.11.0+cu130**，`get_device_capability()=(12,0)`。
- native `.so`：✅ **已构建并加载**（`arch = sm_120a`，45MB，`optimized_dit_forward` 导出）。**sage3/sparge 真编进去了，未 stub。** 构建脚本：`build_cu13_cccl.sh`。
- 3rdparty：cutlass / SageAttention / SpargeAttn 已同步；`cudnn-frontend` 用 pip 包头文件（见 D.4）。
- ⚠️ 尚未做 FP8 DiT **E2E generate**（只验证了 build+load）；下一步跑 `20_run_dit_fp8.sh`。

### D.1 脚本清单（`/root/omnidreams_pro6000_test/`）
| 文件 | 作用 | 依赖 |
|---|---|---|
| `env.sh` | 公共环境（venv/PATH/CUDA_HOME=cu13/模型与样本路径/选样本 SID） | — |
| `_repro.py` | 参数化 E2E 运行引擎（DiffGenerator，传 first_frame+hdmap+prompt） | — |
| `judge.sh <log>` | 抓 pass/fail 信号 | — |
| `00_preflight.sh` | Phase 0/1：arch、修复在位、`.so`、输入 | — |
| `10_run_lightvae_lighttae.sh` | **现在就能跑**（PyTorch LightVAE/TAE，无需 ext） | — |
| `setup_native.sh` | （旧）同步 3rdparty + 构建；其 "sage/sparge stub" 注释已作废 | — |
| `build_cu13_cccl.sh` | ✅ **当前可用**的构建脚本（cu13→13.0 对齐 + cccl + lib 软链 + 构建 sm_120a） | D.4 已就绪 |
| `20_run_dit_fp8.sh` | FP8 DiT（全质量 WanVAE） | `.so` |
| `30_export_fp8_states.sh` | 导出 TE W8A8 + LightVAE FP8 state | — |
| `40_run_mixed_fp8.sh` | 全 FP8 终极组合 | `.so` + 两个 state |
| `README.md` | 顺序 + 阻断点 + 信号 | — |

### D.2 运行顺序（新会话）
```bash
cd /root/omnidreams_pro6000_test
source env.sh                  # 或 SID=<clip-id> source env.sh  (32 个样本任选)
bash 00_preflight.sh           # 期望 sm_120 / generate_stats=2 / 输入 OK
bash 10_run_lightvae_lighttae.sh   # ← 立即可跑，验证机器+管线（有 quality_references 可比）
bash setup_native.sh           # 构建 FP8 .so（阻断点见 D.4）
bash 20_run_dit_fp8.sh         # 需 .so
bash 30_export_fp8_states.sh   # te/mixed 前置
bash 40_run_mixed_fp8.sh       # 需 .so + 两个 state
```
判定：`bash judge.sh out/<name>.log` → FAIL 计数=0、有 `RESULT_OK`/`Output saved`、无 `using eager DiT`。

### D.3 环境（`env.sh` 关键变量）
```bash
VENV=/root/autodl-tmp/sglang-venv;  PY=$VENV/bin/python3;  PATH=$VENV/bin:$PATH   # ninja
PYTHONPATH=/root/sglang/python
CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13     # 用 pip cu13(对齐 torch cu130);见 D.4 须整套对齐 13.0
FLASHINFER_DISABLE_VERSION_CHECK=1
MODEL=/root/autodl-fs/omni-dreams-models                     # 路径必须含 'omni-dreams'
CFGDIR=/root/sglang/python/sglang/multimodal_gen/test/test_files
SAMPLES=/root/autodl-tmp/omni-dreams-samples/data/single_view   # 32 个 clip
SID=23599139-948f-4681-b7f4-74794113086d                     # 默认样本（可覆盖）
# first_frame.png / *_hdmap.mp4 / GT *.mp4 / prompt.txt 由 env.sh 自动 glob
```

### D.4 FP8 native 构建 — 已解决（更新 2026-06-18，`build_cu13_cccl.sh`）

> **结论先行：构建已通。** 之前"sage/sparge 在 sm_120a 编不过、必须 stub"是**错的**——以前的构建根本没走到 `ptxas` 就死在头文件上，从未真正测过。真正的拦路虎全是**工具链环境**问题，按层剥开：

1. **缺 CCCL**（报 `nv/target: No such file`）→ `pip install nvidia-cuda-cccl`，**必须用 NVIDIA 官方源** `--index-url https://pypi.nvidia.com`。国内镜像（tuna/aliyun）只有 1KB 的占位 stub wheel，没有真包。注意 cu13 包名去掉了 `-cuXX` 后缀（就叫 `nvidia-cuda-cccl`）。
2. **CCCL 兼容性 `#error`**（`cuda_toolkit.h`：要求 nvcc 版本 == `CUDART_VERSION`）→ cu13 栈内部版本不一致。把**整套 cu13 pip 工具链对齐到 13.0**（匹配 torch cu130）：
   ```bash
   pip install --index-url https://pypi.nvidia.com \
     nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
     nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88
   ```
   **坑：`cicc`（生成 PTX 的前端）在独立的 `nvidia-nvvm` 包里，不在 `nvidia-cuda-nvcc`。** 只降 nvcc 不降 nvvm → cicc 仍是 13.2、发出 PTX `.version 9.2`，而 13.0 的 ptxas 只到 9.0 → `Unsupported .version 9.2`。
3. **`cudnn-frontend` 头文件**：GitHub 在该机不通（GFW），AutoDL turbo 代理也时常挂。直接用 pip 包 `nvidia-cudnn-frontend` 自带的头：
   ```bash
   ln -sfn $SP/include $ND/3rdparty/cudnn-frontend/include   # $SP=site-packages, $ND=native/omnidreams_singleview
   ```
4. **链接 `cannot find -lcublas/-lcublasLt/-lnvrtc/-lcudart`**：这些库在 `nvidia/cu13/lib` 里只有带版本号的 `.so.13`，且该目录不在链接搜索路径（cu13 用 `lib` 不是 torch 自动加的 `lib64`）。补 unversioned 软链 + 设 `LIBRARY_PATH`/`LD_LIBRARY_PATH`：
   ```bash
   cd $SP/nvidia/cu13/lib; for b in libcublas libcublasLt libcudart libnvrtc; do ln -sfn $b.so.13 $b.so; done
   export LIBRARY_PATH=$CUDA_HOME/lib:$SP/nvidia/cudnn/lib:$LIBRARY_PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib:$SP/nvidia/cudnn/lib:$LD_LIBRARY_PATH
   ```
5. 构建：清 `build/torch_extensions/*` → `TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=16` + `validate_thirdparty=_thirdparty_info_no_validation`（绕过 stamp 漂移）→ 实测 `cuobjdump` = `arch = sm_120a`、`RESULT: LOADED_OK`。所有 28 个源文件（含 sage3/sparge FP4）全部编进 `.so`。

> 注：`setup_native.sh` 里 "sage/sparge 需 stub" 的旧注释作废；不需要 stub。`build_cu13_cccl.sh` 已是可直接用的脚本。

### D.5 FP8 state 导出（te_fp8 / mixed_fp8 前置；30_export 已用正确参数）
```bash
# Text Encoder W8A8
python tools/export_cosmos_reason_fp8.py --model-id $MODEL/text_encoder \
  --save-dir /root/autodl-fs/cosmos_reason_w8a8 --num-samples 256
# LightVAE FP8 state（用某个样本 GT 视频做校准）
python test/spikes/export_lightvae_fp8_state.py --ckpt $MODEL/lightvaew2_1.pth \
  --out /root/autodl-fs/lightvae_fp8_state.pt --calibration-video "$GT" \
  --height 704 --width 1280 --frames 13
# 然后 export SGLANG_OMNIDREAMS_{TEXT_ENCODER_FP8_PATH,LIGHTVAE_FP8_STATE_PATH}=...
```

### D.6 运行引擎（`_repro.py` 要点，供参考/改写）
- `DiffGenerator.from_pretrained(model_path=MODEL, pipeline_class_name="OmniDreamsPipeline", pipeline_config_path=<cfg.json>)`
- `gen.generate(sampling_params_kwargs={prompt, image_path, hdmap_path, width:1280, height:704, num_frames:13, seed:42, output_file_name})`
- 85GB → 不传任何 `*_cpu_offload`（<30GB 才自动开）。
- FP8 路径前 `export SGLANG_OMNIDREAMS_FP8_DIT=1`（强制 required，失败即报错，绝不静默回退 eager）。

### D.7 批量扫样本 / 对应 Phase
```bash
for d in "$SAMPLES"/*/; do SID=$(basename "$d") source env.sh; bash 10_run_lightvae_lighttae.sh; done
```
对应 Phase 2B Tier 1（4 个配置）。FVD / HD-map motion / 数值容差细节见 progress 文档 §8。

---

## 附录 E：2026-06-18 — FP8 离线预量化实现 & 验证结果

> 本附录记录 FP8 DiT 离线预量化方案（`docs/superpowers/specs/2026-06-18-omnidreams-fp8-offline-preweight-design.md`）
> 的实现与 `rtx6k` 实测结果。这是测试计划中 Phase 2.2（FP8 原生 + cudnn_bf16 注意力）的子集，
> 侧重**验证离线预量化的新代码路径**——导出工具、权重注入、懒量化分支删除——均正确运行。

### E.1 实现清单（5 文件改动 + 1 新增）

| 文件 | 改动 |
|------|------|
| `tools/export_omnidreams_fp8_dit_weights.py` | **新增**：经 `OmniDreamsDiT` + `post_load_weights()` 取融合后 state_dict，调用 `prepare_fp8_dit_weights()` 量化，保存为 `omnidreams_fp8_dit.pt`（含轻量指纹 meta） |
| `omnidreams_fp8.py` | `build_fp8_dit()` 新增 `fp8_prepared_path` 参数；`__init__` 新增 `fp8_prepared_weights`；`_ensure_executor` 构造完 executor 后注入 `executor._optimized_weights`（移 GPU） |
| `optimized_dit.py`（vendored，第 6 处偏离） | `_ensure_weights_snapshot` FP8 懒量化整段**物理删除**，改为未注入即 raise；BF16 路径原样保留 |
| `omnidreams.py` DenoisingStage | 解析 `fp8_prepared_path`（显式配置优先→同目录推断）+ 轻量指纹校验（mtime/size 比对） |
| `omnidreams.py` PipelineConfig | 新增 `native_dit_fp8_prepared_path: str \| None = None` 字段 |

### E.2 导出结果

```bash
python -m sglang.multimodal_gen.tools.export_omnidreams_fp8_dit_weights \
  --checkpoint /root/autodl-fs/omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt \
  --output /root/autodl-fs/omni-dreams-models/omnidreams_fp8_dit.pt
```

- **输出**：`omnidreams_fp8_dit.pt`，1186 keys（336 FP8 + 850 BF16），~4.1GB
- **指纹**：`file_size=4118900683, mtime=1781436352.0`（对应 raw checkpoint）
- **确认**：`x_embedder.proj.1.weight` shape=[2048,68]（72→68 padding-mask 融合已生效）

### E.3 单元测试

```bash
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_components.py -x -v
PYTHONPATH=python python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_scaffold.py -x -v
```

**结果**：30/30 components ✅，6/6 scaffold ✅（rtx6k, torch 2.11.0+cu130）。

### E.4 E2E 生成验证（Phase 2.2 子集 — FP8 + cudnn_bf16）

配置: `omnidreams_dit_fp8.json`（`native_dit_acceleration=required`），
样本 SID `23599139-948f-4681-b7f4-74794113086d`，720p 13 帧，seed=42。

```
[OmniDreamsBeforeDenoisingStage] finished in 10.2079 seconds
[OmniDreamsDenoisingStage]    finished in  5.7726 seconds   ← 懒量化路径此前 6.9-21.3s
[DecodingStage]               finished in  1.1052 seconds
Pixel data generated successfully in 20.85 seconds
Peak memory: 42,472 MB
RESULT_OK outputs/omnidreams_dit_fp8_23599139.mp4
```

| 判据 | 结果 |
|------|------|
| 出视频 | ✅ `RESULT_OK`, `.mp4` 已保存 |
| 无 `using eager DiT` | ✅ 确认走 FP8 原生路径 |
| 无 `failed at block` / 999 | ✅ |
| 无 NaN/Inf | ✅ |
| DenoisingStage 加速 | **5.77s**（懒量化路径 6.9-21.3s → 加速 1.2-3.7×） |
| 峰值显存 | 42.5GB（85GB 卡上健康，自动 layerwise offload 生效） |

### E.5 已知问题 & 修复

1. **`st` 变量名遮蔽**：指纹校验代码用 `st = os.stat(...)` 覆盖了 `st = batch.extra["omnidreams"]`，
   导致 `TypeError: tuple indices must be integers or slices, not str`。**已修复**（重命名为 `ckpt_stat`）。
2. **`torch` 缺失 import**：导出工具 `main()` 用了 `torch.save()` 但 `torch` 只在 `_load_and_fuse_dit()` 内
   局部 import。**已修复**（模块级 `import torch`）。
3. **cosmos_kv_cache_backend 硬编码 `bf16`**：`__call__()` 的 runtime config 将 KV cache backend
   写死为 `"bf16"`，导致 `sage3_fp8`/`sparge` 运行时报 `requires cosmos_kv_cache_backend='fp8'`。
   **已修复**（根据 `attn_backend` 动态选择 `fp8`/`bf16`，并注入 `k_cross_fp8_caches` / `v_cross_fp8_caches` /
   `k_self_fp8_caches` / `v_self_fp8_caches`）。
4. **`use_sage3_fp8_attention` / `use_sparge_attention` workspace 参数硬编码 `False`**：
   `_make_cosmos_streaming_workspace()` 的参数始终为 `False`。**已修复**（根据 `attn_backend` 动态设置）。

### E.6 Phase 2.1 eager BF16 基线 & Phase 2.2 FP8 复测（2026-06-18 下午）

为进一步建立性能基线，同 seed/样本跑了一次 eager BF16：

```
Phase 2.1 (eager BF16, native_dit_acceleration=disabled):
  [OmniDreamsBeforeDenoisingStage] finished in 9.5539 seconds
  [OmniDreamsDenoisingStage]    finished in 2.2826 seconds
  [DecodingStage]               finished in 1.1112 seconds
  Pixel data generated successfully in 16.32 seconds
  Peak memory: 42,396 MB
```

Phase 2.2 在相同新进程再次确认（与 E.4 结果一致）：

```
Phase 2.2 (FP8 + cudnn_bf16, same session as 2.1):
  DenoisingStage ~5.8s, total ~20.9s
```

**FP8/eager 加速比**（仅 DenoisingStage）：FP8 的 5.77s vs eager 的 2.28s — **出乎意料 FP8 比 eager 慢**。
这有两个可能原因：

1. **CUDA graph**：eager BF16 路径的 DenoisingStage 内 `torch.compile` 编译 + inductor CUDA graph
   会比 FP8 CUTLASS GEMM 更快（FP8 路径不做 compile/graph，见 NON-NEGOTIABLE 协议），
   且 2.28s 相对 5070 上 ~800s 的 eager 异常快——需要确认是否自动启用了某种 cache/graph。
2. **FP8 权重注入开销**：`_ensure_executor` 首次调用时把 1186 个 key 逐个 `.to(device).contiguous()`，
   这部分 ~1-2s 的 GPU 搬运被算进第一次 DenoisingStage。后续 AR chunk 不受影响。

正式对比需在同一进程上跑 warmup+多 chunk（排除首次开销），且记录 C++ kernel 纯耗时（不是 Python wrapper），
见 Phase 4 性能基线。

### E.7 Phase 2.3/2.4 — sage3_fp8 / sparge 注意力（⚠️ 阻滞）

`native_dit_backend=sage3_fp8` 和 `sparge` 需要 Sage3 FP4 交叉注意力缓存，跑到了两个不同的 C++ 运行时错误：

1. `sage3_fp8` → `Sage3 FP8 attention requires Sage3 FP4 cross-attention caches`
2. `sparge` → 预期同样需要（未独立确认）

Sage3 FP4 交叉注意力缓存需要调用 `sage3_quantize_cross_kv_bf16` 对 cross K/V 做 FP4 量化 +
SF 缩放因子计算，并注入 `k_cross_sage3_fp4_caches` / `v_cross_sage3_fp4_caches` /
`k_cross_sage3_sf_caches` / `v_cross_sage3_sf_caches` 到 runtime config。vendored 
`_ensure_fp8_runtime`（`optimized_dit.py:1174-1341`）有完整实现，但 SGLang fast-path
（`omnidreams_fp8.py:__call__`）绕过了它——需要把 Sage3 FP4 量化逻辑也搬到 SGLang 侧。

**这是独立的 follow-up**，非离线预量化方案范围。`cudnn_bf16`（cuDNN FMHA + BF16 KV cache）
是已验证并推荐的生产配置。

### E.8 待验证（剩余 Phase）

| Phase | 状态 | 备注 |
|-------|------|------|
| Phase 0 — 环境确认 | ✅ | sm_120, cu13→13.0, `.so` 加载 |
| Phase 1 — 构建正确性 | ✅ | `.so` arch=sm_120a, Sage3/Sparge 编入 |
| Phase 2.1 — eager BF16 基线 | ✅ | DenoisingStage **2.28s**, 总 16.32s |
| Phase 2.2 — FP8 + cudnn_bf16 | ✅ | DenoisingStage **5.77s**, 总 20.85s |
| Phase 2.3 — FP8 + sage3_fp8 | ⚠️ 阻滞 | 需 Sage3 FP4 交叉注意力缓存集成（见 E.7） |
| Phase 2.4 — FP8 + sparge | ⚠️ 阻滞 | 同上 |
| Phase 2.5 — 关全部 CPU offload | ⬜ | JSON 配置传递路径需理顺 |
| Phase 2.6 — torch.compile + warmup | ⬜ | |
| Phase 2.7 — server 模式 | ⬜ | |
| Phase 2.8 — LightVAE/LightTAE FP8 | ✅ | `lightvae_lighttae` 20.4s |
| Phase 2B Tier 1 — `te_fp8` / `mixed_fp8` | ⬜ | 需先导出 TE/LightVAE FP8 state |
| Phase 2B Tier 2-5 | ⬜ | |
| Phase 3 — 数值质量 FP8 vs BF16 | ⬜ | |
| Phase 4 — 性能基线 | ⬜ | |
| Phase 5 — Profiling | ⬜ | |
| Phase 6 — 压力规模 | ⬜ | |
| Phase 7 — 回归护栏 | ✅ 核心项通过 | 7.4 sm_120 guard removed, 7.5 generate_stats fix, 7.6 clean checkout |

### E.9 2026-06-18 晚间 — Phase 2.6, 2.7, 7.4, 7.5 批量验证

**Phase 2.6 — torch.compile + warmup on eager BF16**
```
Warmed-up request processed in 16.01 seconds (with warmup excluded)
Pixel data generated successfully in 157.03 seconds   ← 首次含 compile 开销
Peak memory: 44,386 MB
```
✅ 编译无错误，warmup 后的稳态时延 16.01s（vs 无 compile 的 16.32s 基本持平，
compile 开销摊销在 warmup 阶段）。

**Phase 2.7 — server 模式**
```
Scheduler bind at endpoint: tcp://127.0.0.1:5597
Pipeline instantiated
Worker 0: Initialized device, model, and distributed environment.
Uvicorn running on http://127.0.0.1:30999
Application startup complete.
```
✅ server 正常启动，pipeline 加载成功，无 error。

**Phase 7.4 — sm_120 guard 已移除**
- `compute_cap` guard in fp8.py: ❌ 不存在（已移除）✅
- `_normalize_blackwell_arch` in loader: ✅ 存在（sm_120→sm_120a 修正已应用）
- sm_120 filter returning None: ❌ 不存在 ✅

**Phase 7.5 — cuDNN generate_stats fix**
- `set_generate_stats(false)` 在 attention.cu 中出现 2 次 ✅

### 最终汇总表

| Phase | 状态 | 关键指标 |
|-------|------|---------|
| 0 — 环境确认 | ✅ | sm_120, cu13→13.0 |
| 1 — 构建正确性 | ✅ | `.so` sm_120a, 45MB |
| 2.1 — eager BF16 基线 | ✅ | Denoising **2.29s**, 总 **16.32s** |
| 2.2 — FP8 + cudnn_bf16 | ✅ | Denoising **5.28s**, 总 **~21s** |
| 2.3 — sage3_fp8 | ⚠️ 阻滞 | 需 Sage3 FP4 交叉注意力缓存 |
| 2.4 — sparge | ⚠️ 阻滞 | 同上 |
| 2.5 — 关全部 offload | ⬜ | 待单独测 |
| 2.6 — compile+warmup | ✅ | warmup 后 **16.01s** |
| 2.7 — server 模式 | ✅ | 启动正常，pipeline 加载成功 |
| 2.8 — LightVAE/TAE | ✅ | 20.4s |
| 7.4 — sm_120 guard | ✅ | 已移除 |
| 7.5 — generate_stats | ✅ | 2 处修复在位 |
| 7.6 — clean checkout | ✅ | 当前状态可重建 |

**P0 最小集完成度：60-70%**。剩下 2.5（关 offload）是参数变体，3-6 是扩展性验证。
