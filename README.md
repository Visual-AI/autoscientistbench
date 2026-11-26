# AutoScientistBench

**Benchmarking Long-Horizon AI Agents with 104 Real-Paper AutoResearch Tasks across Nine Scientific Disciplines**

[![Project Page](https://img.shields.io/badge/Project%20Page-Visual--AI-2a78d6)](https://visual-ai.github.io/autoscientistbench/)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-allencbzhang%2FAutoScientistBench-ffd21e)](https://huggingface.co/datasets/allencbzhang/AutoScientistBench)
[![Tasks](https://img.shields.io/badge/Tasks-104%20Published%20Papers-008300)](#benchmark-overview)
[![Hardware](https://img.shields.io/badge/Hardware-CPU--Only-6250d6)](#requirements)

**AutoScientistBench** evaluates whether autonomous AI agents can take real methods published in 2026 as baselines and genuinely advance them through open-ended scientific research. Unlike synthetic coding benchmarks, every task ships an adapted official baseline implementation, frozen inputs/splits, domain-specific validity guards, and an authoritative held-out evaluation pipeline — all runnable on **CPU-only** hardware.

---

## Benchmark Overview

- **104 active tasks** defined in [`benchmark_tasks.json`](benchmark_tasks.json) (`001_InertialGenome` through `104_M3SVM`, matching the `001 · InertialGenome` .. `104 · M3SVM` numbering on the [project website](https://visual-ai.github.io/autoscientistbench/)) spanning nine scientific disciplines.
- **Submission unit = Code patch**: agents edit `code/` inside a network-isolated Linux sandbox (`bwrap`), validate locally on train/val via `bench eval-dev`, and submit a patch via `bench submit --final`.
- **Independent held-out judge**: `bench judge` applies the candidate patch onto a clean checkout, rebuilds the environment from `code/env.lock` (or mounts the frozen runtime snapshot), retrains from scratch across the task's evaluation seeds, predicts on hidden test inputs, runs the frozen `validate.py` + `scorer.py`, and compares against the archived baseline reference in `tasks/<task_id>/judge/reference/result.json`.
- **Two reproducible runtime modes**:
  - **75 Python `venv` tasks**: rebuilt automatically via `uv venv` + `uv pip sync code/env.lock` from PyPI and PyTorch CPU wheel indexes.
  - **29 `snapshot` runtime tasks** (R, Julia, Bioconda, Go, Octave, C++/Boost): pre-packaged read-only `.tar` runtime snapshots hosted on Hugging Face and mounted inside `bwrap`.
- **Large asset hosting**: datasets and pre-built runtimes (~186 GB total) are hosted on Hugging Face and fetched on demand with SHA-256 verification via `python3 -m bench fetch`. For `052_AnisoBivFunc` (153.6 GB of pure synthetic bivariate functional surfaces), only its R runtime is downloaded and the exact observation shards are generated deterministically locally via [`scripts/generate_052_AnisoBivFunc.py`](scripts/generate_052_AnisoBivFunc.py).

See [`PROTOCOL.md`](PROTOCOL.md) for the formal evaluation contract, sandboxing architecture, and scoring formulas, [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) for the complete step-by-step evaluation and custom-harness guide, and [`skills/autoscientistbench/SKILL.md`](skills/autoscientistbench/SKILL.md) if you are using an AI coding agent to operate benchmark runs.

---

## Repository Structure

```text
autoscientistbench/
├── benchmark_tasks.json       # Official catalog of 104 active tasks, 9 disciplines, baselines & target ceilings
├── assets_manifest.json       # SHA-256 catalog of HF dataset/test/runtime archives & 052_AnisoBivFunc generator
├── PROTOCOL.md                # Formal benchmark contract, sandbox isolation, and scoring rules
├── EVALUATION_GUIDE.md        # Step-by-step operator guide (custom endpoints, suites, leaderboard scoring)
├── skills/autoscientistbench/ # Ready-to-use SKILL.md for AI coding agents operating benchmark runs
├── bench/                     # Pure Python (>=3.10) stdlib harness CLI (`python3 -m bench`)
│   ├── assets.py              # On-demand HF fetcher, SHA-256 verifier, multi-part tar stream extractor
│   ├── run.py                 # Sandboxed agent runner (bwrap + allowlist proxy + submission broker)
│   ├── judge.py               # Independent clean-checkout evaluator & reference scorer
│   ├── report.py              # Official leaderboard calculator (`Passed` & `Improvement Score %`)
│   ├── envlock.py             # uv-based Python environment builder & validator
│   ├── runtime.py             # Read-only snapshot runtime materializer for the 29 non-pip tasks
│   └── harness/               # Adapters for codex, claude-code, kimi, agy, zcode, scripted
├── harnesses/                 # Harness configuration templates
├── scripts/
│   └── generate_052_AnisoBivFunc.py  # Deterministic local shard generator for 052_AnisoBivFunc (val + test)
├── tasks/<task_id>/           # All 104 benchmark tasks
│   ├── task.json              # Task metadata, metric, research_contract, validation_policy, env spec
│   ├── agent/
│   │   ├── README.md          # Task statement, data schema, baseline, rules, research contract
│   │   ├── code/              # Baseline implementation (train.sh, predict.sh, env.lock)
│   │   └── bench/             # Frozen validator, scorer, io_spec.md
│   ├── judge/
│   │   ├── bench/             # Authoritative mirror of agent/bench/
│   │   └── reference/         # Archived baseline result.json and env.lock
│   └── runtime/               # (29 snapshot tasks) runtime.json manifest; *.tar fetched from HF
└── tests/                     # Unit tests & scripted end-to-end verification
```

---

## Requirements

- **OS**: Linux (x86_64)
- **Python**: >= 3.10 (standard library only for `bench/`; `huggingface_hub` for `bench fetch`)
- **System tools**:
  - `git`
  - `uv` (for rebuilding the 75 Python `venv` tasks)
  - `bwrap` (bubblewrap) and `taskset` (for sandbox isolation and CPU pinning)
  - Passwordless `sudo` if using per-sandbox UID isolation (set `ARB_SANDBOX_UID=0` to run under the current UID instead)
- **Optional agent CLIs**: `codex`, `claude`, `kimi`, `agy`, or `zcode` depending on the harness you evaluate.

### Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `ASB_HF_REPO` | `allencbzhang/AutoScientistBench` | Hugging Face Dataset repo ID hosting task data & runtimes |
| `ARB_LOCAL_ROOT` | `/tmp/arb` | Fast node-local scratch for sandboxes, venvs, and extracted snapshots |
| `ARB_UV_CACHE` | `$ARB_LOCAL_ROOT/uv-cache` | Shared `uv` wheel cache used by operator/judge builds |
| `ARB_SANDBOX_UID` | `1` | Set to `0` to disable per-sandbox UID isolation (no `sudo` required) |
| `ARB_TASKS_DIR` / `ARB_RUNS_DIR` / `ARB_RESULTS_DIR` | `./tasks` / `./runs` / `./results` | Path overrides |

---

## Quickstart

### 1. Clone and install `huggingface_hub`

```bash
git clone https://github.com/Visual-AI/autoscientistbench.git
cd autoscientistbench
pip install huggingface_hub
```

### 2. Verify the harness with the built-in toy task

```bash
# Run unit tests (no model calls, no HF download required)
python3 -m unittest discover -s tests -v

# Run end-to-end scripted agent + sandbox closure + judge test
ARB_SANDBOX_UID=0 bash tests/e2e_scripted.sh
```

### 3. Fetch datasets, test splits, and runtimes on demand

`bench fetch` downloads task assets directly from [`allencbzhang/AutoScientistBench`](https://huggingface.co/datasets/allencbzhang/AutoScientistBench) on Hugging Face (configured by default in [`assets_manifest.json`](assets_manifest.json); override with `ASB_HF_REPO` or `--hf-repo` if using a mirror):

```bash
# Check which tasks have local data / test / runtime assets ready
python3 -m bench assets-status

# Fetch a single Python venv task (~small download)
python3 -m bench fetch 001_InertialGenome

# Fetch a snapshot runtime task (downloads data + test + frozen runtime/*.tar)
python3 -m bench fetch 014_SNooPy

# Fetch only validation/training data (skip hidden test split)
python3 -m bench fetch 100_xPatch --only data

# Fetch all 104 tasks (downloads 103 tasks + 29 runtimes, and generates 052_AnisoBivFunc locally)
python3 -m bench fetch all --cores 12
```

#### Local deterministic generation for `052_AnisoBivFunc`

`052_AnisoBivFunc` requires 153.6 GB of pure synthetic bivariate functional surfaces (76.8 GB `val` + 76.8 GB `test`, 120 shards each). Rather than downloading 153.6 GB, `python3 -m bench fetch 052_AnisoBivFunc` fetches only its 196 MB R runtime snapshot (`runtime/r.tar`) and invokes [`scripts/generate_052_AnisoBivFunc.py`](scripts/generate_052_AnisoBivFunc.py) inside `bwrap` to reproduce the bit-exact `.bin` shards locally (~5–10 minutes on 12 CPUs):

```bash
# Fetch 052_AnisoBivFunc R runtime and generate both val and test splits locally
python3 -m bench fetch 052_AnisoBivFunc --cores 12

# Or invoke the generator directly (supports --split val|test|all, --verify-only, --limit-shards K)
python3 scripts/generate_052_AnisoBivFunc.py --split val --cores 12
python3 scripts/generate_052_AnisoBivFunc.py --verify-only
```

### 4. Run an agent and judge its submission

```bash
# Run Codex against an OpenAI-compatible API endpoint, then judge automatically
OPENAI_API_KEY=... python3 -m bench run \
  --task 001_InertialGenome \
  --harness codex-api \
  --model gpt-5.4 \
  --budget-hours 2 \
  --judge

# Or run with Claude Code / Kimi / Antigravity / ZCode
ANTHROPIC_API_KEY=... python3 -m bench run --task 100_xPatch --harness claude-code --budget-hours 2 --judge
python3 -m bench run --task 100_xPatch --harness kimi --budget-hours 2 --judge

# Judge an existing run or a standalone git patch file
python3 -m bench judge --run <run_id>
python3 -m bench judge --task 001_InertialGenome --patch path/to/candidate.patch

# Multi-task suite execution
python3 -m bench suite \
  --tasks 001_InertialGenome,100_xPatch,014_SNooPy \
  --harnesses codex-api \
  --set model='"gpt-5.4"' \
  --budget-hours 2 \
  --judge-budget-hours 5 \
  --parallel 2 \
  --suite demo-suite
```

### 5. Compute official leaderboard scores

AutoScientistBench reports two official headline metrics across all 104 tasks (and across each of the 9 scientific disciplines):
1. **`Passed` (`X / 104`)**: tasks where all evaluation seeds succeed, all scientific validity guards pass, and relative gain over the reproduced baseline meets $\max(\delta_{\text{task}},\, 2\sigma_{\text{ref}}/|\mu_{\text{ref}}|)$.
2. **`Improvement Score (%)`**: mean over all 104 tasks of $\max\!\big(0,\, \frac{x_i - b_i}{c_i - b_i}\times 100\big)$, mapping each task's reproduced baseline $b_i \mapsto 0\%$ and calibrated target ceiling $c_i \mapsto 100\%$ (`tasks_meta.<task_id>.target_ceiling` in [`benchmark_tasks.json`](benchmark_tasks.json)), clipped at `0%` on regressions/failures and uncapped above `100%`.

```bash
# Summarize all runs under ./runs (grouped by harness:model, latest run per task)
python3 -m bench report

# Summarize a suite and export structured JSON + per-task CSV
python3 -m bench report \
  --suite demo-suite \
  --json results/demo-suite/leaderboard.json \
  --csv results/demo-suite/per_task.csv

# Filter to one of the 9 disciplines (e.g., bioinformatics, physics, computer_science)
python3 -m bench report --domain bioinformatics
```

For custom API gateways (`upstream_base_url`), multi-seed budget sizing, resuming interrupted runs, and adding new agent harnesses, see [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) and [`skills/autoscientistbench/SKILL.md`](skills/autoscientistbench/SKILL.md).

---

## Understanding the 29 Snapshot Runtime Tasks

The 29 `snapshot` tasks (`014_SNooPy`, `015_LexicMap`, `017_krepp`, `018_ETfinder`, `019_MultiContextSeeds`, `020_ALPINE`, `021_TARPON`, `022_WEPP`, `023_LI-DTI`, `025_QUED`, `038_SuperIsing`, `041_ClusterSMART`, `042_FrontDoor-NP`, `043_Geodesic-DiD`, `046_Chatterjee-BC`, `047_CoupledPoisson`, `048_ALSE-Treat`, `049_DiPMInd`, `050_Lasso-FDR`, `051_SpectralSpatial`, `052_AnisoBivFunc`, `053_DMFM-IGau`, `054_StirlingGamma`, `057_BT-SBM`, `058_AnytimeMC`, `061_DP-FRT`, `062_SkewPosterior`, `063_MartingalePost`, `098_BalancedActiveInf`) rely on complex R, Julia, Bioconda, Go, Octave, or C++/Boost toolchains.

Each snapshot task ships a `runtime/runtime.json` manifest describing its `.tar` archive(s) and SHA-256 digests. When running inside `bwrap`, `bench/runtime.py` extracts the archives to `$ARB_LOCAL_ROOT/snapshots/<task_id>/` and bind-mounts them **read-only** at their compiled prefix path (plus `/opt/arb/runtime/<name>`). Because `bwrap` creates virtual mount targets inside the isolated mount namespace, those prefix paths work on any x86_64 Linux host regardless of the host's directory layout and never access the host filesystem outside the extracted snapshot.

---

## Run Artifacts

Each `bench run` writes a self-contained record under `runs/<run_id>/`:

```text
runs/<run_id>/
├── run.json            # Budget, cpuset, termination reason, token usage, submission stats
├── prompt.md           # Exact prompt delivered to the agent
├── agent_events.jsonl  # Structured agent event stream
├── harness.log         # Harness stderr
├── events.jsonl        # Agent bench eval-dev / bench submit log
├── submissions/        # NNN.patch + NNN.json for every explicit submission
├── final.patch / .json # Selected patch evaluated by the judge
├── proxy.log           # Allowlist proxy CONNECT and model-route audit log
├── judge/              # Clean-room rebuild logs, per-seed train/predict logs, predictions
└── result.json         # Final judge verdict (status, gain, threshold, pass, diagnostics)
```

---

## License

This repository's harness code and benchmark metadata are released under the license in [`LICENSE`](LICENSE). Individual task baselines under `tasks/<task_id>/agent/code/` retain their respective upstream paper repository licenses (documented in each task's `agent/README.md` and `code/` directory).
