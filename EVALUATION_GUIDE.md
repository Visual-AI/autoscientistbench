# AutoScientistBench — Evaluation & Scoring Guide

This guide provides a complete reference for **human researchers** and **AI operator agents** running evaluations, integrating custom models or harnesses, and computing official leaderboard scores on **AutoScientistBench**.

> **Using an AI Coding Agent to Run Evaluations?**
> Point your agent (Claude Code, Codex, Antigravity, Cursor, Kimi, etc.) directly to [`skills/autoscientistbench/SKILL.md`](skills/autoscientistbench/SKILL.md), or copy `skills/autoscientistbench` into your tool's skills folder (e.g., `.agents/skills/` or `.claude/skills/`).
> *(Note: Evaluated benchmark agents run inside an isolated `/ws` bubblewrap sandbox and never have access to outer repository files.)*

---

## 1. End-to-End Evaluation Workflow

```text
┌─────────────────────────┐     ┌──────────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 1. Fetch Assets (HF)    │ ──▶ │ 2. Sandboxed Agent Run       │ ──▶ │ 3. Clean-Room Judge     │ ──▶ │ 4. Leaderboard Report   │
│ python3 -m bench fetch  │     │ python3 -m bench run / suite │     │ python3 -m bench judge  │     │ python3 -m bench report │
└─────────────────────────┘     └──────────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
  • data/ (103 tasks)             • /ws/{README,code,data,bench}       • Fresh baseline checkout       • Passed (X / 104)
  • test/ (103 tasks)             • --unshare-net + proxy              • Rebuild env / snapshot        • Improvement Score (%)
  • runtime/*.tar (29 tasks)      • Agent runs `bench eval-dev`        • Multi-seed train+predict      • 9-discipline breakdown
  • 052_AnisoBivFunc R gen        • Agent runs `bench submit`          • Frozen validate + scorer      • JSON & CSV export
```

---

## 2. Host Setup & Sanity Check

1. **System packages**: Linux `x86_64` with `git`, `uv`, `bwrap` (bubblewrap), `taskset` (`util-linux`), and Python >= 3.10.
2. **Python dependency** (for downloading assets from Hugging Face):
   ```bash
   pip install huggingface_hub
   ```
3. **Recommended environment variables**:
   ```bash
   export ARB_SANDBOX_UID=0            # Run bwrap as current user (no passwordless sudo needed)
   export ARB_LOCAL_ROOT=/tmp/arb      # Fast local SSD/NVMe scratch (>= 50 GB free recommended)
   # Optional mirror override (defaults to allencbzhang/AutoScientistBench in assets_manifest.json):
   # export ASB_HF_REPO="allencbzhang/AutoScientistBench"
   ```
4. **45-second end-to-end self-test** (verifies `bwrap`, network isolation, allowlist proxy, `uv` venv build, submission broker, and judge):
   ```bash
   ARB_SANDBOX_UID=0 bash tests/e2e_scripted.sh
   ```

---

## 3. Preparing Task Data & Runtimes

All 104 tasks (`001_InertialGenome` through `104_M3SVM`) follow the exact numbering on the [AutoScientistBench website](https://visual-ai.github.io/autoscientistbench/):

- **75 Python `venv` tasks**: `bench` automatically builds an isolated Python environment from `code/env.lock` via `uv`.
- **29 `snapshot` runtime tasks** (R, Julia, Bioconda, Go, Octave, C++/Boost): `bench fetch` downloads pre-built `runtime/*.tar` archives and mounts them read-only inside `bwrap`.
- **`052_AnisoBivFunc`**: Instead of downloading 153.6 GB of synthetic float64 surfaces, `bench fetch 052_AnisoBivFunc` downloads its 196 MB `runtime/r.tar` and deterministically generates all 240 `.bin` shards (`val` + `test`) locally via [`scripts/generate_052_AnisoBivFunc.py`](scripts/generate_052_AnisoBivFunc.py).

```bash
# Check local readiness across all 104 tasks
python3 -m bench assets-status

# Fetch a few tasks to start
python3 -m bench fetch 001_InertialGenome,004_GradNormSmooth,014_SNooPy

# Fetch everything (103 HF datasets + 29 runtimes + local generation of 052_AnisoBivFunc)
python3 -m bench fetch all --cores 12
```

---

## 4. Launching Evaluations

### 4.1 Single-Task Run + Judge

```bash
ARB_SANDBOX_UID=0 OPENAI_API_KEY=... python3 -m bench run \
  --task 001_InertialGenome \
  --harness codex-api \
  --model gpt-5.4 \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --run-id gpt54__001_InertialGenome \
  --judge
```

### 4.2 Custom OpenAI / Anthropic Compatible Endpoints

`codex-api` and `claude-code` route model requests through the operator-side proxy (`bench/proxy.py`), which injects the API key outside the sandbox so credentials never enter `/ws`. Override `upstream_base_url`, `upstream_api_key_env`, and (for Chat Completions endpoints) `wire_api`:

```bash
ARB_SANDBOX_UID=0 DEEPSEEK_API_KEY=... python3 -m bench run \
  --task 004_GradNormSmooth \
  --harness codex-api \
  --model deepseek-v4 \
  --set upstream_base_url='"https://api.deepseek.com/v1"' \
  --set upstream_api_key_env='"DEEPSEEK_API_KEY"' \
  --set wire_api='"chat"' \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --judge
```

*(For harnesses that connect directly via HTTPS `CONNECT` — such as `codex` subscription mode or `zcode` — extend `allow_hosts` via `--set 'allow_hosts=["your-api-host.example.com"]'`.)*

### 4.3 Full or Subset Multi-Task Suite

`bench suite` automatically pins concurrent workers to disjoint CPU sets:

```bash
ARB_SANDBOX_UID=0 OPENAI_API_KEY=... python3 -m bench suite \
  --tasks all \
  --harnesses codex-api \
  --set model='"gpt-5.4"' \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --parallel 4 \
  --suite gpt54-full
```

> **Budget Tip**: Each task's `task.json` specifies `agent_budget_hours` (default agent runtime, usually `2`–`3` hours) and `budget_hours_default` (formal judge budget, usually `5` hours across all seeds). Passing `--judge-budget-hours 5` ensures multi-seed clean-room retraining during `bench judge` has its full intended budget.

### 4.4 Adding a Custom Agent Harness

To evaluate your own research agent scaffold:
1. Subclass `Harness` in `bench/harness/<your_harness>.py` (see [`bench/harness/scripted.py`](bench/harness/scripted.py) and [`bench/harness/codex.py`](bench/harness/codex.py) for minimal examples) and register it in [`bench/harness/__init__.py`](bench/harness/__init__.py).
2. Add `harnesses/<your_harness>/config.json` specifying `"type": "<your_harness>"` and `"allow_hosts": ["your-api-host.example.com"]` (or implement `model_upstream()` for key-injected proxy routing).
3. Inside `/ws`, your agent reads `/ws/README.md` and `/ws/bench/io_spec.md`, edits `/ws/code/`, tests on train/val via `bench eval-dev`, and submits its best patch via `bench submit --final -m "summary"`.

---

## 5. Computing Official Leaderboard Scores

AutoScientistBench evaluates models on **two official metrics** (stored and documented in [`benchmark_tasks.json`](benchmark_tasks.json) and [`PROTOCOL.md`](PROTOCOL.md)):

### 5.1 Metric 1 — `Passed` (`X / 104`)

A task counts as **Passed** (`result.json["pass"] == true`) when:
1. All evaluation seeds finish within budget and pass `validate.py` + `scorer.py` guards (finite metric), and
2. Relative gain over the reproduced baseline $\mu_{\text{ref}}$ meets or exceeds the task's noise-aware threshold:
   $$\text{gain}_i = s_i \cdot \frac{x_i - \mu_{\text{ref}, i}}{|\mu_{\text{ref}, i}|} \ge \max\!\left(\delta_i,\; \frac{2\,\sigma_{\text{ref}, i}}{|\mu_{\text{ref}, i}|}\right)$$

### 5.2 Metric 2 — `Improvement Score (%)`

Each task $i \in \{1, \dots, 104\}$ has a reproduced baseline $b_i = \mu_{\text{ref}, i}$ (`0%`) and a calibrated target ceiling $c_i$ (`100%`, `tasks_meta.<task_id>.target_ceiling` in `benchmark_tasks.json`):

$$\text{score}_i = \begin{cases} \max\!\left(0,\; \dfrac{x_i - b_i}{c_i - b_i} \times 100\right) & \text{if } \texttt{status == "scored"} \text{ and } x_i \text{ is finite} \\[8pt] 0 & \text{otherwise (no submission, timeout, build/guard failure)} \end{cases}$$

- Regressions below baseline are clipped to `0%`.
- Breakthroughs surpassing the target ceiling remain **uncapped above `100%`**.
- The overall **Improvement Score** averages $\text{score}_i$ across all **104 tasks** (or across the $N_d$ tasks of discipline $d$):
  $$\text{Improvement Score} = \frac{1}{104} \sum_{i=1}^{104} \text{score}_i$$

### 5.3 Generating Score Reports

```bash
# Summarize all runs in ./runs (grouped by harness:model, deduplicated to latest run per task)
python3 -m bench report

# Summarize a suite and export structured JSON + per-task CSV
python3 -m bench report \
  --suite gpt54-full \
  --json results/gpt54-full/leaderboard.json \
  --csv results/gpt54-full/per_task.csv

# Filter to a single scientific discipline (e.g., computer_science, bioinformatics, physics, mathematics,
# medicine, economics, chemistry, statistics, environmental_science)
python3 -m bench report --domain bioinformatics
```
