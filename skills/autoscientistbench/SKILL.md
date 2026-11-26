---
name: autoscientistbench
description: >-
  Operate, evaluate, and score AI agents on AutoScientistBench (104 real-paper
  scientific research tasks, 001_InertialGenome through 104_M3SVM). Use this
  skill when asked to understand the repository structure, fetch datasets or
  snapshot runtimes from Hugging Face, generate 052_AnisoBivFunc data locally,
  configure harnesses or model API endpoints, run single-task or multi-task
  evaluations (`bench run`, `bench suite`, `bench judge`), resume interrupted
  runs, or compute final leaderboard scores (`Passed` and `Improvement Score %`
  via `bench report`).
---

# AutoScientistBench Operator Skill

This skill guides an **operator agent** in setting up environments, fetching task assets, launching sandboxed benchmark evaluations, and computing official leaderboard scores for **AutoScientistBench**.

> **Safety & Isolation Rules (MUST FOLLOW)**
> 1. **Never modify frozen task files**: Do NOT edit `tasks/*/{agent,judge}/bench/*`, `tasks/*/judge/reference/*`, or `tasks/*/runtime/runtime.json`. Judge integrity hashes and private-eval wire protocols depend on exact bytes.
> 2. **Two-agent boundary**: This skill is for the **outer operator agent** managing evaluations. The **evaluated research agent** runs inside an isolated `bwrap` sandbox (`/ws`) containing only `/ws/{README.md, code/, data/, bench/}` and never sees outer repository files.

---

## 1. Repository Map

| Path | Purpose |
|---|---|
| `benchmark_tasks.json` | Catalog of all 104 tasks (`001_InertialGenome` .. `104_M3SVM`), 9 scientific disciplines (`domains`), baselines, pass thresholds, and `target_ceiling` anchors for leaderboard scoring. |
| `assets_manifest.json` | Exact byte sizes and SHA-256 digests for all HF dataset/test/runtime archives + `052_AnisoBivFunc` local generator spec. |
| `bench/` | Pure-Python stdlib CLI (`python3 -m bench`): `tasks`, `assets-status`, `fetch`, `run`, `judge`, `suite`, `report`, `smoke`, `lint`, `fixture`. |
| `harnesses/<name>/config.json` | Agent adapter configs (`codex-api`, `codex`, `claude-code`, `kimi`, `agy`, `zcode`, `scripted`), including `upstream_base_url` / proxy `allow_hosts`. |
| `scripts/generate_052_AnisoBivFunc.py` | Deterministic local R generator for `052_AnisoBivFunc`'s 153.6 GB synthetic surfaces (`val` + `test`). |
| `tasks/<task_id>/` | Per-task `task.json`, `agent/{README.md, code/, bench/}`, `judge/{bench/, reference/}`, and (for 29 snapshot tasks) `runtime/runtime.json`. |
| `runs/<run_id>/` | Output of `bench run` + `bench judge`: `run.json`, `final.patch`, `agent_events.jsonl`, `proxy.log`, `result.json`. |
| `results/<suite_id>/` | Output of `bench suite`: `per_run/*.json` and `summary.json`. |

---

## 2. Pre-flight Host Check

Before running evaluations, verify host prerequisites in one command:

```bash
which git uv bwrap taskset && python3 -c "import huggingface_hub; print('hf_hub ok')"
```

- **UID isolation**: By default `bench` uses `sudo` to run each sandbox under an ephemeral UID. On machines **without passwordless `sudo`**, always export:
  ```bash
  export ARB_SANDBOX_UID=0
  ```
- **Scratch directory**: Point `ARB_LOCAL_ROOT` to fast local NVMe/SSD with >= 50 GB free (default `/tmp/arb`):
  ```bash
  export ARB_LOCAL_ROOT=/tmp/arb
  ```
- **Quick harness sanity check** (no LLM calls, no HF download needed, ~45s):
  ```bash
  ARB_SANDBOX_UID=0 bash tests/e2e_scripted.sh
  ```

---

## 3. Fetching Datasets, Test Splits, and Snapshot Runtimes

Task code and metadata ship in Git, while large `agent/data`, `judge/test`, and the 29 pre-built `runtime/*.tar` snapshots (~186 GB total) are fetched on demand from [`allencbzhang/AutoScientistBench`](https://huggingface.co/datasets/allencbzhang/AutoScientistBench) (configured by default in `assets_manifest.json`):

```bash
# Inspect which tasks are locally ready
python3 -m bench assets-status

# Fetch specific task(s) (downloads data + hidden test + runtime if snapshot mode)
python3 -m bench fetch 001_InertialGenome,014_SNooPy,100_xPatch

# Fetch all 104 tasks (downloads 103 tasks + 29 runtimes, and generates 052_AnisoBivFunc locally)
python3 -m bench fetch all --cores 12
```

- **Special case — `052_AnisoBivFunc`**: Uses 153.6 GB of synthetic surfaces (76.8 GB `val` + 76.8 GB `test`). `bench fetch 052_AnisoBivFunc --cores 12` downloads only its 196 MB R runtime (`runtime/r.tar`) and runs `scripts/generate_052_AnisoBivFunc.py` inside `bwrap` (~5–10 min on 12 CPUs, bit-exact across x86_64 CPUs).
- **Integrity check without re-downloading**: `python3 -m bench fetch <tasks|all> --verify-only`.

---

## 4. Configuring Harnesses & Model Endpoints

Inside `bwrap`, direct network access is disabled (`--unshare-net`). All traffic goes through `bench/proxy.py`.

1. **Standard API & CLI harnesses**:
   - `codex-api`: OpenAI-compatible API routed via the operator proxy (`OPENAI_API_KEY` is injected outside `/ws`, never exposed inside the sandbox).
   - `claude-code`: Anthropic-compatible API routed via the operator proxy (`ANTHROPIC_API_KEY` injected outside `/ws`).
   - `codex`: ChatGPT subscription login (`~/.codex/auth.json`, HTTPS `CONNECT` via `allow_hosts`).
   - `kimi`: Kimi Code CLI (`~/.kimi-code/config.toml`).
   - `agy`: Antigravity CLI adapter (`harnesses/agy/config.json`).
   - `zcode`: ZCode CLI adapter (`harnesses/zcode/config.json`, `BIGMODEL_API_KEY`).
2. **Using a custom OpenAI/Anthropic-compatible gateway**:
   Override `upstream_base_url`, `upstream_api_key_env`, and (for `/v1/chat/completions` endpoints instead of `/v1/responses`) `wire_api`:
   ```bash
   MY_API_KEY=... python3 -m bench run \
     --task 001_InertialGenome \
     --harness codex-api \
     --model my-model-name \
     --set upstream_base_url='"https://my-gateway.example.com/v1"' \
     --set upstream_api_key_env='"MY_API_KEY"' \
     --set wire_api='"chat"' \
     --budget-hours 2 \
     --judge-budget-hours 5 \
     --judge
   ```
   *(For direct HTTPS `CONNECT` harnesses like `codex` subscription or `zcode`, add custom domains via `--set 'allow_hosts=["my-gateway.example.com"]'`. For `venv` tasks, PyPI and PyTorch CPU wheel hosts are automatically added by `bench/run.py`.)*

---

## 5. Running Evaluations & Judging

### Budget & CPU Sizing Rules
Inspect `tasks/<task_id>/task.json` before launching large runs:
- `agent_budget_hours`: recommended agent wall-clock budget (typically `2` or `3` hours).
- `budget_hours_default`: formal judge replay budget across all evaluation seeds (typically `5` hours — **always pass `--judge-budget-hours 5`** so multi-seed retraining in `bench judge` never starves on a shorter agent budget).
- `cpus`: task CPU allocation (typically `4`, `8`, `12`, `16`, or `128`). When using `bench suite --parallel P`, ensure `P * cpus <= available_host_cores` (or pass `--cpus C` to cap per-job CPUs).

### Commands

```bash
# 1) Single task: run agent + immediately judge held-out test split
ARB_SANDBOX_UID=0 OPENAI_API_KEY=... python3 -m bench run \
  --task 001_InertialGenome \
  --harness codex-api \
  --model gpt-5.4 \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --run-id exp1__001_InertialGenome \
  --judge

# 2) Multi-task suite with automatic disjoint CPU pinning
ARB_SANDBOX_UID=0 OPENAI_API_KEY=... python3 -m bench suite \
  --tasks all \
  --harnesses codex-api \
  --set model='"gpt-5.4"' \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --parallel 4 \
  --suite gpt54-full

# 3) Re-judge an existing run or judge an external unified diff patch
ARB_SANDBOX_UID=0 python3 -m bench judge --run runs/exp1__001_InertialGenome --budget-hours 5
ARB_SANDBOX_UID=0 python3 -m bench judge --task 001_InertialGenome --patch /path/to/candidate.patch --budget-hours 5

# 4) Resume an interrupted resumable run with its remaining wall-clock budget
ARB_SANDBOX_UID=0 python3 -m bench run \
  --task 001_InertialGenome --harness claude-code --budget-hours 2 \
  --run-id exp1__001_InertialGenome --resume --judge
```

---

## 6. Computing Official Final Scores (`bench report`)

AutoScientistBench publishes **two official leaderboard metrics**:
1. **`Passed` (`X / 104`)**: Count of tasks where `status == "scored"`, all seeds & scientific guards pass, and relative gain $\ge \max(\delta_{\text{task}}, 2\sigma_{\text{ref}}/|\mu_{\text{ref}}|)$.
2. **`Improvement Score (%)`**: Normalized progress from reproduced `baseline` ($b_i \mapsto 0\%$) to calibrated `target_ceiling` ($c_i \mapsto 100\%$):
   $$\text{score}_i = \max\!\left(0,\; \frac{x_i - b_i}{c_i - b_i} \times 100\right), \qquad \text{Improvement Score} = \frac{1}{104}\sum_{i=1}^{104} \text{score}_i$$
   - Negative scores (regressions) are clipped to `0%`.
   - Scores exceeding `target_ceiling` are **uncapped above `100%`**.
   - Unattempted, `no_submission`, timeout, build-error, or guard-violating tasks contribute `0%` to the 104-task sum.

Run `bench report` to generate the official table, per-discipline breakdown (9 disciplines), JSON summary, and per-task CSV:

```bash
# Score all runs under ./runs (automatically grouped by harness:model, latest run per task)
python3 -m bench report

# Score a specific suite and export JSON + per-task CSV
python3 -m bench report --suite gpt54-full --json results/gpt54-full/leaderboard.json --csv results/gpt54-full/per_task.csv

# Filter by model or by scientific discipline (e.g., bioinformatics, physics, computer_science)
python3 -m bench report --model gpt-5.4 --domain bioinformatics
```

---

## 7. Troubleshooting Checklist

- **`connect=403` in `runs/<run_id>/proxy.log`**: A direct-TLS CLI tried to reach an API host not listed in `allow_hosts`. Add the hostname via `--set 'allow_hosts=["api.example.com"]'` (or use `codex-api` / `claude-code` with `--set upstream_base_url=...`).
- **`sudo: a password is required` / `sandbox_user` failure**: Export `ARB_SANDBOX_UID=0`.
- **`status: "no_submission"` in `run.json`**: The evaluated agent edited `/ws/code` or ran `bench eval-dev` but never called `bench submit --final` before the deadline. Explicit submission is mandatory by protocol.
- **`status: "budget_exhausted"` during `bench judge`**: Increase `--judge-budget-hours` (e.g. `5`) or re-run `python3 -m bench judge --run runs/<run_id> --budget-hours 5`.
- **`agent/data missing` or `snapshot source missing`**: Run `python3 -m bench fetch <task_id>`.
