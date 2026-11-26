"""Operator CLI: python -m bench <subcommand>."""
import argparse
import json
import sys
from pathlib import Path

from . import RESULTS_DIR, RUNS_DIR, TASKS_DIR, VERSION


def _kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _seeds(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def cmd_tasks(a):
    from .task import list_tasks, load_task
    for pid in list_tasks():
        t = load_task(pid)
        ref = (t.reference_dir / "result.json").exists()
        probs = t.problems()
        print(f"{pid:32s} class={t.d['class']} metric={t.metric_name} cpus={t.cpus} adapt={'yes' if t.adapt_commit else 'no '} "
              f"reference={'yes' if ref else 'no '} {'OK' if not probs else 'problems: ' + '; '.join(probs[:2])}")


def cmd_init(a):
    from .convert import init
    init(a.pid, title=a.title, metric=a.metric, higher=a.higher, python=a.python, cpus=a.cpus, force=a.force, split=a.split)


def cmd_adapt(a):
    from .convert import adapt
    kw = {"allow_fresh": a.allow_fresh}
    adapt(a.pid, message=a.message, **kw) if a.message else adapt(a.pid, **kw)


def cmd_snapshot(a):
    from .convert import snapshot
    snapshot(a.pid, a.src, excludes=a.exclude or ())


def cmd_lint(a):
    from .convert import lint
    return 1 if lint(a.pid, full=a.full) else 0


def cmd_reference(a):
    from .judge import compute_reference
    compute_reference(a.pid, seeds=_seeds(a.seeds), cpus=a.cpus, cpuset=a.cpuset, keep=a.keep,
                      sandbox=not a.no_sandbox, budget_hours=a.budget_hours)


def cmd_smoke(a):
    from .smoke import smoke
    result = smoke(a.pid, input_path=a.input, labels_path=a.labels, cpus=a.cpus,
                   cap_s=a.cap_s, sandbox=not a.no_sandbox, seed=a.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


def cmd_run(a):
    from .run import bench_run
    meta = bench_run(a.task, a.harness, a.budget_hours, cpus=a.cpus, cpuset=a.cpuset, repeat=a.repeat, suite=a.suite,
                     keep=a.keep, overrides=_kv(a.set), model=a.model, grace_s=a.grace_s, run_id=a.run_id, dry=a.dry,
                     judge_budget_hours=a.judge_budget_hours, resume=a.resume)
    print(json.dumps({k: meta.get(k) for k in ("run_id", "status", "ended_reason", "elapsed_s", "patch", "submissions", "agent")},
                     indent=2, ensure_ascii=False))
    if a.judge and not a.dry and meta.get('status') != 'paused':
        from .judge import bench_judge
        res = bench_judge(run_dir=RUNS_DIR / meta["run_id"], cpus=a.cpus, cpuset=a.cpuset, keep=a.keep,
                          budget_hours=a.judge_budget_hours)
        print(json.dumps({k: res.get(k) for k in ("status", "gain", "threshold", "pass", "failure")}, indent=2))
        return 0 if res.get("status") == "scored" else 2
    return 2 if meta.get("status") == "no_submission" else 0


def cmd_judge(a):
    from .judge import bench_judge
    res = bench_judge(run_dir=a.run, pid=a.task, patch_path=a.patch, seeds=_seeds(a.seeds),
                      reference_check=(False if a.no_reference_check else (True if a.reference_check else None)),
                      reference_now=a.reference_now, cpus=a.cpus, cpuset=a.cpuset, keep=a.keep,
                      val_judge=(None if a.val_judge is None else a.val_judge), label=a.label, budget_hours=a.budget_hours)
    print(json.dumps({k: res.get(k) for k in ("judge_id", "status", "gain", "threshold", "pass", "candidate", "reference", "failure")},
                     indent=2, ensure_ascii=False, default=str))
    return 0 if res.get("status") in ("scored",) else 2


def cmd_suite(a):
    from .suite import run_suite
    run_suite(a.tasks, a.harnesses, a.budget_hours, repeats=a.repeats, parallel=a.parallel, cpus=a.cpus, suite=a.suite,
              judge=not a.no_judge, keep=a.keep, overrides=_kv(a.set), judge_budget_hours=a.judge_budget_hours)


def cmd_fixture(a):
    if a.name == "toy-ridge":
        from .fixtures.toy_ridge import build
        build(force=a.force)
    else:
        raise SystemExit(f"unknown fixture {a.name}")


def cmd_proxy(a):
    from . import proxy
    sys.argv = ["bench proxy", *a.rest]
    proxy.main()


def cmd_fetch(a):
    from .assets import fetch_assets
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    fetch_assets(
        a.tasks,
        hf_repo=a.hf_repo,
        only=only,
        local_dir=a.local_dir,
        cores=a.cores,
        force=a.force,
        verify_only=a.verify_only,
        no_sandbox=a.no_sandbox,
    )


def cmd_assets_status(a):
    from .assets import assets_status
    assets_status(a.tasks)


def cmd_report(a):
    from .report import build_report, format_report_text, write_report_csv
    run_ids = [x.strip() for x in a.runs.split(",") if x.strip()] if a.runs else None
    rep = build_report(
        suite=a.suite,
        runs_dir=a.runs_dir,
        run_ids=run_ids,
        harness_filter=a.harness,
        model_filter=a.model,
        domain_filter=a.domain,
    )
    print(format_report_text(rep))
    if a.json_out:
        out_p = Path(a.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.csv_out:
        write_report_csv(rep, a.csv_out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench", description=f"AutoScientistBench harness {VERSION}")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("tasks", help="list tasks and their readiness state").set_defaults(f=cmd_tasks)

    p = sp.add_parser("fetch", help="download task datasets, test splits, and snapshot runtimes from Hugging Face (and generate 052_AnisoBivFunc locally)")
    p.add_argument("tasks", help="'all' or comma-separated task IDs (e.g. 001_InertialGenome,014_SNooPy,052_AnisoBivFunc)")
    p.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo ID (or set ASB_HF_REPO)")
    p.add_argument("--only", default=None, help="comma subset of: data,test,runtime (default: all three)")
    p.add_argument("--local-dir", default=None, help="read archives from a local staged dataset directory instead of Hugging Face")
    p.add_argument("--cores", type=int, default=None, help="CPU workers for local deterministic generation (052_AnisoBivFunc)")
    p.add_argument("--force", action="store_true", help="re-download / re-extract even if already verified")
    p.add_argument("--verify-only", action="store_true", help="verify existing local assets without downloading or generating")
    p.add_argument("--no-sandbox", action="store_true", help="run 052_AnisoBivFunc R generator without bwrap")
    p.set_defaults(f=cmd_fetch)

    p = sp.add_parser("assets-status", help="show local readiness of data, test, and runtime assets")
    p.add_argument("tasks", nargs="?", default="all", help="'all' or comma-separated task IDs")
    p.set_defaults(f=cmd_assets_status)

    p = sp.add_parser("report", help="compute official leaderboard Passed count and Improvement Score (%) across runs or a suite")
    p.add_argument("--suite", help="suite ID under results/<suite> or path to suite directory")
    p.add_argument("--runs-dir", help="directory of run records (default: ./runs)")
    p.add_argument("--runs", help="comma-separated run IDs or run paths")
    p.add_argument("--harness", help="filter by harness name")
    p.add_argument("--model", help="filter by model name")
    p.add_argument("--domain", default="all", help="filter by scientific discipline ID (default: all 104 tasks)")
    p.add_argument("--json", dest="json_out", help="write full structured JSON report to FILE")
    p.add_argument("--csv", dest="csv_out", help="write per-task CSV breakdown to FILE")
    p.set_defaults(f=cmd_report)

    p = sp.add_parser("init", help="create tasks/<pid>/ skeleton")
    p.add_argument("pid"); p.add_argument("--title", default=""); p.add_argument("--metric", default="mse")
    p.add_argument("--higher", action="store_true", help="metric: higher is better (default lower)")
    p.add_argument("--python", default="3.11"); p.add_argument("--cpus", type=int, default=4); p.add_argument("--force", action="store_true")
    p.add_argument("--split", default="std3", help="split.type written into task.json (std3, train_test, synthetic, online, training_free, ...)")
    p.set_defaults(f=cmd_init)

    p = sp.add_parser("adapt", help="commit protocol edits of agent/code as bench-adapt, sync judge/bench, write adapt.patch")
    p.add_argument("pid"); p.add_argument("-m", "--message", default=None)
    p.add_argument("--allow-fresh", action="store_true", help="no git history yet: commit everything as the base (adapt.patch empty)")
    p.set_defaults(f=cmd_adapt)

    p = sp.add_parser("snapshot", help="copy the official code into agent/code and make it the git base (run once, before adapt)")
    p.add_argument("pid"); p.add_argument("--from", dest="src", required=True, help="official code directory (keeps its .git if any)")
    p.add_argument("--exclude", action="append", help="extra ignore pattern (repeatable); .venv/__pycache__ etc. are always excluded")
    p.set_defaults(f=cmd_snapshot)

    p = sp.add_parser("lint", help="acceptance checklist for a converted task")
    p.add_argument("pid"); p.add_argument("--full", action="store_true", help="also build the env and exercise validate/scorer")
    p.set_defaults(f=cmd_lint)

    p = sp.add_parser("reference", help="measure the baseline under the bench protocol -> judge/reference/")
    p.add_argument("pid"); p.add_argument("--seeds"); p.add_argument("--cpus", type=int); p.add_argument("--cpuset")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--no-sandbox", action="store_true", help="run train/predict without bwrap (hosts where the sandbox cannot nest)")
    p.add_argument("--budget-hours", type=float, help="overall reference budget; defaults to the task budget")
    p.set_defaults(f=cmd_reference)

    p = sp.add_parser("smoke", help="small train/predict/validate/score check; does not archive a full reference")
    p.add_argument("pid"); p.add_argument("--input"); p.add_argument("--labels")
    p.add_argument("--cpus", type=int, default=2); p.add_argument("--cap-s", type=int, default=300)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--no-sandbox", action="store_true")
    p.set_defaults(f=cmd_smoke)

    p = sp.add_parser("run", help="bench-run: closed workspace -> harness -> deadline -> final.patch")
    p.add_argument("--task", required=True); p.add_argument("--harness", required=True)
    p.add_argument("--budget-hours", type=float, required=True); p.add_argument("--cpus", type=int); p.add_argument("--cpuset")
    p.add_argument("--judge-budget-hours", type=float, help="overall subsequent judge budget; defaults to the agent budget")
    p.add_argument("--repeat", type=int, default=1); p.add_argument("--suite", default="adhoc"); p.add_argument("--keep", action="store_true")
    p.add_argument("--model"); p.add_argument("--set", action="append", help="override harness config key=value (JSON value)")
    p.add_argument("--grace-s", type=int, default=300); p.add_argument("--run-id"); p.add_argument("--dry", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue an unfinished opt-in resumable run with its remaining original budget")
    p.add_argument("--judge", action="store_true", help="judge right after the run"); p.set_defaults(f=cmd_run)

    p = sp.add_parser("judge", help="bench-judge: patch -> rebuild -> k seeds -> score")
    p.add_argument("--run", help="runs/<run_id> (uses its final.patch)"); p.add_argument("--task"); p.add_argument("--patch")
    p.add_argument("--seeds"); p.add_argument("--no-reference-check", action="store_true")
    p.add_argument("--reference-check", action="store_true", help="re-run the baseline for parity (off by default since 2026-11-09)")
    p.add_argument("--reference-now", action="store_true", help="compute the reference on the fly if none is archived")
    p.add_argument("--cpus", type=int); p.add_argument("--cpuset"); p.add_argument("--keep", action="store_true")
    p.add_argument("--budget-hours", type=float, help="overall judge budget; defaults to the saved run budget or task budget")
    p.add_argument("--val-judge", dest="val_judge", action="store_true", default=None)
    p.add_argument("--no-val-judge", dest="val_judge", action="store_false")
    p.add_argument("--label"); p.set_defaults(f=cmd_judge)

    p = sp.add_parser("suite", help="tasks x harnesses x repeats on this node, then judge, then summary")
    p.add_argument("--tasks", required=True, help="all | comma list"); p.add_argument("--harnesses", required=True)
    p.add_argument("--budget-hours", type=float, required=True); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--judge-budget-hours", type=float, help="overall budget for each judge run; defaults to the agent budget")
    p.add_argument("--parallel", type=int, default=1); p.add_argument("--cpus", type=int); p.add_argument("--suite", default=None)
    p.add_argument("--no-judge", action="store_true"); p.add_argument("--keep", action="store_true"); p.add_argument("--set", action="append")
    p.set_defaults(f=cmd_suite)

    p = sp.add_parser("fixture", help="build a test task (toy-ridge)"); p.add_argument("name"); p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_fixture)

    p = sp.add_parser("proxy", help="run the allowlist proxy standalone (debug)"); p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(f=cmd_proxy)

    a = ap.parse_args(argv)
    return a.f(a) or 0
