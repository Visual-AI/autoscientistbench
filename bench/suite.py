"""bench-suite (single node): fan out tasks x harnesses x repeats over disjoint cpusets, judge each, summarise."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import RESULTS_DIR, RUNS_DIR
from .judge import bench_judge
from .run import bench_run
from .task import list_tasks, load_task
from .util import ensure_dir, free_cpus, log, now_iso, ts_compact, write_json


def summarize(results):
    by = {}
    for r in results:
        h = r.get("harness") or "?"
        d = by.setdefault(h, {"n": 0, "scored": 0, "pass": 0, "harm": 0, "gains": [], "status": {}})
        d["n"] += 1
        d["status"][r.get("status")] = d["status"].get(r.get("status"), 0) + 1
        if r.get("status") == "scored":
            d["scored"] += 1
            d["pass"] += int(bool(r.get("pass")))
            g = r.get("gain")
            if g is not None:
                d["gains"].append(g)
                d["harm"] += int(g < 0)
    for h, d in by.items():
        n_valid = sum(v for k, v in d["status"].items() if k != "infra_error")
        d["pass_rate"] = d["pass"] / n_valid if n_valid else None
        d["mean_gain"] = sum(d["gains"]) / len(d["gains"]) if d["gains"] else None
        d["harm_rate"] = d["harm"] / d["scored"] if d["scored"] else None
    return by


def run_suite(tasks, harnesses, budget_hours, *, repeats=1, parallel=1, cpus=None, suite=None, judge=True, keep=False,
              overrides=None, judge_budget_hours=None):
    suite = suite or f"suite-{ts_compact()}"
    pids = list_tasks() if tasks == "all" else [t for t in tasks.split(",") if t]
    hs = [h for h in harnesses.split(",") if h]
    jobs = [(p, h, r) for p in pids for h in hs for r in range(1, repeats + 1)]
    free = free_cpus()
    out_dir = ensure_dir(RESULTS_DIR / suite)
    ensure_dir(out_dir / "per_run")
    log(f"suite {suite}: {len(jobs)} jobs, parallel={parallel}, cpus free={len(free)}")
    results = []

    def one(idx, job):
        pid, h, r = job
        slot = idx % parallel
        c = int(cpus or load_task(pid).cpus)
        cpuset = free[slot * c:(slot + 1) * c] or free[:c]
        rec = {"task": pid, "harness": h, "repeat": r, "status": "infra_error"}
        try:
            meta = bench_run(pid, h, budget_hours, cpus=c, cpuset=",".join(map(str, cpuset)), repeat=r, suite=suite, keep=keep,
                             overrides=overrides, judge_budget_hours=judge_budget_hours)
            rec.update(run_id=meta["run_id"], ended_reason=meta.get("ended_reason"), agent=meta.get("agent"))
            if judge:
                res = bench_judge(run_dir=RUNS_DIR / meta["run_id"], cpus=c, cpuset=",".join(map(str, cpuset)), keep=keep,
                                  budget_hours=judge_budget_hours)
                rec.update({k: res.get(k) for k in ("status", "gain", "threshold", "pass", "failure", "diagnostics")})
                rec["candidate_test_mean"] = ((res.get("candidate") or {}).get("test") or {}).get("mean")
            else:
                rec["status"] = "no_submission" if meta.get("status") == "no_submission" else "run_only"
                if rec["status"] == "no_submission": rec["pass"] = False
        except Exception as e:  # noqa: BLE001
            rec["failure"] = {"stage": "suite", "reason": str(e)[-1000:]}
            log(f"suite job {job} failed: {e}")
        write_json(out_dir / "per_run" / f"{pid}__{h}__r{r}.json", rec)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = [ex.submit(one, i, j) for i, j in enumerate(jobs)]
        for f in futs:
            results.append(f.result())
    summary = {"suite": suite, "created": now_iso(), "budget_hours": budget_hours,
               "judge_budget_hours": budget_hours if judge_budget_hours is None else judge_budget_hours,
               "tasks": pids, "harnesses": hs, "repeats": repeats,
               "by_harness": summarize(results), "runs": results}
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary["by_harness"], indent=2))
    return summary
