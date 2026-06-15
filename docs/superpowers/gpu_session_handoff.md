# GPU Session Handoff — 2026-06-15

## Machine

- **SSH:** `ssh chen@100.87.72.4`
- **GPU:** RTX 5070 (sm_120, 12GB), WSL2, CUDA 13.1
- **Python:** `/home/chen/.python/sglang/bin/python` (torch 2.11+cu130)
- **nvcc:** `/usr/local/cuda/bin/nvcc` (13.0)
- **cmake:** `/home/chen/miniconda3/bin/cmake` (3.31)
- **tmux:** `tmux -S /tmp/tmux-1000/default attach`
- **Models:** E 盘 `/mnt/e/models/` ← `/home/chen/models` symlink
- **Repo:** `/home/chen/gitRepo/sglang` (main branch, origin Cerdore/sglang)
- **Disk:** E 盘 139G free, C 盘 3.4G ⚠️

## What's done

| Phase | What | GPU Status |
|-------|------|------------|
| P3 | torch.compile | ✅ PASS |
| P0 | CUDA Graph API | ✅ PASS |
| P1 | LightTAE decode | ✅ PASS |
| P2 | LightVAE PyTorch encode | ✅ PASS |
| P4a | DiT FP8 native | ✅ PASS |
| P4b | LightVAE FP8 native | ✅ PASS |
| E2E 13f | Full pipeline 13 frames | ✅ PASS (268s, 2.8× vs PyTorch) |
| E2E 29f | Full pipeline 29 frames | ✅ PASS (560s, 249KB output) |

**All 5 bugs fixed today** (see `omnidreams_optimization_progress.md` §"Bugs fixed").

**本地 Mac（main分支）working tree：P0–P4b + 5 bugfixes 全部代码完成，未 commit/push。**

## Key files (GPU machine)

| File | Purpose |
|------|---------|
| `native/singleview_loader.py` | Prebuilt `.so` fast-path (dirname module name fix) |
| `native/omnidreams_singleview/python/optimized_dit.py` | Stubbed — no longer imports flashdreams/omnidreams |
| `native/omnidreams_singleview/3rdparty/` | CUTLASS + Sage + Sparge + cudnn — exact commit |
| `/home/chen/omnidreams_test/e2e_config.json` | Pipeline config (LightVAE+TAE+FP8 flags) |
| `/home/chen/omnidreams_test/fp8_state.pt` | Calibrated FP8 state (239 keys) |
| `/home/chen/omnidreams_test/e2e_run12.log` | Latest successful 29f E2E log |
| `/home/chen/omnidreams_test/e2e_out/e2e_29f_fp8_v1.mp4` | 29f output (249KB) |

## How to run E2E

```bash
ssh chen@100.87.72.4
export PATH=/home/chen/miniconda3/bin:/home/chen/.python/sglang/bin:$PATH
export LD_LIBRARY_PATH=/home/chen/.python/sglang/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd /home/chen/omnidreams_test

# 13f quick test (~4.5min):
HDMAP_FILES=($(ls /home/chen/omnidreams_test/end_to_end/_g/f*.png | head -13 | sort))
sglang generate --model-path /mnt/e/models/omni-dreams-models --pipeline-class-name OmniDreamsPipeline --pipeline-config-path e2e_config.json --text-encoder-cpu-offload --vae-cpu-offload --dit-cpu-offload --image-path /home/chen/omnidreams_test/end_to_end/newclip_first_frame.png --hdmap-path "${HDMAP_FILES[@]}" --prompt "A car drives forward along a street with buildings and trees on both sides, clear sky." --num-frames 13 --height 704 --width 1280 --seed 42 --output-path e2e_out --output-file-name e2e_13f --save-output

# 29f full test (~9min):
HDMAP_FILES=($(ls /home/chen/omnidreams_test/end_to_end/_g/f*.png | sort))
sglang generate --model-path /mnt/e/models/omni-dreams-models --pipeline-class-name OmniDreamsPipeline --pipeline-config-path e2e_config.json --text-encoder-cpu-offload --vae-cpu-offload --dit-cpu-offload --image-path /home/chen/omnidreams_test/end_to_end/newclip_first_frame.png --hdmap-path "${HDMAP_FILES[@]}" --prompt "A car drives forward along a street with buildings and trees on both sides, clear sky." --num-frames 29 --height 704 --width 1280 --seed 42 --output-path e2e_out --output-file-name e2e_29f --save-output
```

## Debug scripts on remote

| Script | Purpose |
|--------|---------|
| `/tmp/test_prebuilt2.py` | Verify `_load_prebuilt_extension()` dirname fix |
| `/tmp/test_worker_ext.py` | Verify `load_extension()` in spawn worker |
| `/tmp/test_sglang_ext.py` | Full ext load diagnostic |
| `/tmp/debug_prebuilt.py` | Raw `.so` loading test |
| `/tmp/all_in_one.sh` | Original 6-phase unit test suite |
| `/tmp/stamp_fix.sh` | FP8 state rebuild + scale fix |
