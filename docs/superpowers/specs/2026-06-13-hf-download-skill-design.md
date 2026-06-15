# Design: HuggingFace Mirror Download Skill

## Overview

A reference skill that teaches Claude how to download models/datasets from HuggingFace
via the `hf-mirror.com` mirror and the `hfd.sh` script on cloud GPU environments.

## Skill Type

Reference skill (with procedural steps for the download flow).

## Name

`downloading-from-huggingface`

## Trigger Conditions

- User provides a HuggingFace repo link (huggingface.co or hf-mirror.com)
- User mentions downloading models/datasets from HF on a cloud GPU
- User asks about `hfd.sh`, `HF_ENDPOINT`, or HF mirror configuration

## Core Flow

Downloads typically run on a **remote GPU test machine** with no Claude Code —
commands are sent via SSH or tmux. Claude runs locally and issues commands
through the remote session.

1. **Parse** user input to extract `org_name/repo_name`
2. **Identify target machine** — ask user which GPU host to download to (e.g. `gpu`, `gpu2`)
3. **Deploy hfd.sh** — check if `hfd.sh` exists on the remote machine; if not, `scp`/`rsync` it over
4. **Check/set environment variables** on the remote machine (`HF_ENDPOINT`, optionally `HF_USERNAME`/`HF_TOKEN`)
5. **Ask user** for target download directory (with a sensible default like `~/models/<repo_name>`)
6. **Build** the `hfd.sh` command with appropriate flags
7. **Execute** the download on the remote machine via SSH/tmux
8. **Report** result

## Environment Variables (one-time setup)

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_ENDPOINT` | Yes | Mirror URL, default `https://hf-mirror.com` |
| `HF_USERNAME` | No (only gated repos) | HF account username |
| `HF_TOKEN` | No (only gated repos) | HF access token from https://huggingface.co/settings/tokens |

These should be set as shell environment variables (e.g. in `~/.zshrc` or `~/.bashrc`),
never hardcoded in commands and never stored in the skill directory.
Credentials live in the user's shell profile, not in any project or skill file.

Claude should check if they are set before each download and prompt user to set them if missing.
Example setup in `~/.bashrc` or `~/.zshrc`:
```
export HF_ENDPOINT=https://hf-mirror.com
export HF_USERNAME=your_username   # only if you use gated repos
export HF_TOKEN=hf_xxxxxxxxxxxx    # only if you use gated repos
```

## hfd.sh Quick Reference

### Basic download
```
hfd org_name/repo_name --local-dir /path/to/dir
```

### Gated repo (authentication required)
```
hfd org_name/repo_name --local-dir /path/to/dir --hf_username $HF_USERNAME --hf_token $HF_TOKEN
```

### Dataset download
```
hfd org_name/dataset_name --dataset --local-dir /path/to/dir
```

### Specific revision
```
hfd org_name/repo_name --revision v1.0 --local-dir /path/to/dir
```

### File filtering
```
hfd org_name/repo_name --include "*.safetensors" --exclude "*.md" --local-dir /path/to/dir
```

## Prerequisites

- `hfd.sh` located locally (e.g. `~/gitRepo/models/hfd.sh`) for deployment to remote machines
- On remote:
  - `curl` — **required**, used for metadata fetching
  - Download tool — **one of** `aria2c` (default, multi-threaded) or `wget` (fallback, `--tool wget`)
  - `jq` — recommended for fast JSON parsing (slow grep/awk fallback built-in)

## Remote Machine Setup

Before the first download on a new machine:
1. `scp` the `hfd.sh` script to the remote machine (e.g. `~/bin/hfd.sh`)
2. Ensure `hfd.sh` is executable and on `PATH` (or use full path in commands)
3. Set up `HF_ENDPOINT` in remote's `~/.bashrc` / `~/.zshrc`
4. For gated repos: also set `HF_USERNAME` and `HF_TOKEN` on remote

Claude should check for `hfd.sh` on the remote before each download session.
If missing, prompt the user to deploy it first via scp.

## Common Issues

| Issue | Fix |
|-------|-----|
| `aria2c not installed` | `apt install aria2` or use `--tool wget` |
| Gated repo auth error | Set `HF_USERNAME` and `HF_TOKEN`, ensure token has repo access |
| Download interrupted | Re-run same command — `hfd.sh` supports resume |
| Token not passed properly | Use env vars (`$HF_TOKEN`) not literal token in command |

## Scope

This skill covers only `hfd.sh` + mirror downloads. Python-based downloads
(`huggingface_hub`, `snapshot_download`) are out of scope.
