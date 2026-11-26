"""Official AutoScientistBench leaderboard scoring & aggregation (`python3 -m bench report`).

Computes the two official benchmark headline metrics identically to the public leaderboard:
1. Passed (`passed / total`): tasks where all seeds succeed, all scientific guards pass,
   and relative gain >= max(delta_task, 2 * sigma_ref / |mu_ref|).
2. Improvement score (%):
     task_score_i = max(0.0, (x_i - baseline_i) / (target_ceiling_i - baseline_i) * 100.0)
   for scored tasks with finite held-out test mean `x_i` (0.0 for unattempted, no-submission,
   timeout, guard-violating, or regressing tasks; uncapped above 100%), averaged over the
   full benchmark task denominator (104 overall, or N within a selected discipline).
"""
import csv
import json
import math
from pathlib import Path

from . import RESULTS_DIR, ROOT, RUNS_DIR
from .catalog import read_catalog


def compute_task_improvement_score(candidate_mean, baseline, target_ceiling, status="scored"):
    """Return `(clipped_score_pct, raw_score_pct)` for a single task."""
    if status != "scored" or candidate_mean is None or baseline is None or target_ceiling is None:
        return 0.0, None
    x = float(candidate_mean)
    b = float(baseline)
    c = float(target_ceiling)
    span = c - b
    if not (math.isfinite(x) and math.isfinite(b) and math.isfinite(c)) or abs(span) < 1e-15:
        return 0.0, None
    raw = (x - b) / span * 100.0
    return max(0.0, raw), raw


def _extract_test_mean(res: dict):
    if "candidate_test_mean" in res and res["candidate_test_mean"] is not None:
        return res["candidate_test_mean"]
    cand = res.get("candidate") or {}
    test = cand.get("test") or {}
    return test.get("mean")


def _extract_harness_and_model(rmeta: dict, fallback_harness: str = "unknown") -> tuple[str, str | None]:
    h = rmeta.get("harness")
    if isinstance(h, dict):
        harness_name = str(h.get("name") or fallback_harness)
        cfg_model = h.get("model")
    elif isinstance(h, str) and h:
        harness_name = h
        cfg_model = None
    else:
        harness_name = fallback_harness
        cfg_model = None
    model_name = rmeta.get("requested_model") or rmeta.get("model") or cfg_model
    return harness_name, (str(model_name) if model_name else None)


def _infer_task_from_judge_id(judge_id: str | None, known_tasks: set[str]) -> str | None:
    if not judge_id or not judge_id.startswith("judge__"):
        return None
    parts = judge_id.split("__")
    if len(parts) >= 3 and parts[1] in known_tasks:
        return parts[1]
    return None


def _load_records_from_suite(suite_arg: str, runs_dir: Path, known_tasks: set[str]) -> list[dict]:
    p = Path(suite_arg)
    suite_dir = p if p.is_dir() else (RESULTS_DIR / suite_arg)
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    raw_items = []
    per_run_dir = suite_dir / "per_run"
    if per_run_dir.is_dir():
        for fp in sorted(per_run_dir.glob("*.json")):
            raw_items.append((fp.stat().st_mtime, json.loads(fp.read_text(encoding="utf-8"))))
    elif (suite_dir / "summary.json").is_file():
        s = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
        for idx, rec in enumerate(s.get("runs") or []):
            raw_items.append((float(idx), rec))
    else:
        raise FileNotFoundError(f"No per_run/*.json or summary.json found in {suite_dir}")

    records = []
    for mtime, rec in raw_items:
        tid = rec.get("task")
        if tid not in known_tasks:
            continue
        harness = str(rec.get("harness") or "unknown")
        model = rec.get("model")
        cand_mean = _extract_test_mean(rec)
        status = rec.get("status") or "unknown"
        passed = bool(rec.get("pass"))
        gain = rec.get("gain")
        run_id = rec.get("run_id")
        if run_id:
            rdir = runs_dir / run_id
            if (rdir / "run.json").is_file():
                rmeta = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
                harness, rmodel = _extract_harness_and_model(rmeta, fallback_harness=harness)
                model = rmodel or model
            if cand_mean is None and (rdir / "result.json").is_file():
                jres = json.loads((rdir / "result.json").read_text(encoding="utf-8"))
                cand_mean = _extract_test_mean(jres)
                status = jres.get("status", status)
                passed = bool(jres.get("pass", passed))
                gain = jres.get("gain", gain)
        group = f"{harness}:{model}" if model else harness
        records.append({
            "group": group,
            "harness": harness,
            "model": model,
            "task": tid,
            "run_id": run_id,
            "status": status,
            "passed": passed,
            "gain": gain,
            "candidate_test_mean": cand_mean,
            "mtime": mtime,
        })
    return records


def _load_records_from_runs(run_dirs: list[Path], known_tasks: set[str]) -> list[dict]:
    records = []
    for rdir in run_dirs:
        if not rdir.is_dir():
            continue
        run_json = rdir / "run.json"
        res_json = rdir / "result.json"
        if not run_json.is_file() and not res_json.is_file():
            continue
        rmeta = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else {}
        jres = json.loads(res_json.read_text(encoding="utf-8")) if res_json.is_file() else {}

        tid = (
            rmeta.get("task")
            or jres.get("task")
            or _infer_task_from_judge_id(jres.get("judge_id") or rdir.name, known_tasks)
        )
        if tid not in known_tasks:
            continue
        fallback_h = "standalone" if res_json.is_file() and not run_json.is_file() else "unknown"
        harness, model = _extract_harness_and_model(rmeta, fallback_harness=fallback_h)
        group = f"{harness}:{model}" if model else harness
        status = jres.get("status") or rmeta.get("status") or "unknown"
        passed = bool(jres.get("pass")) if jres else False
        gain = jres.get("gain") if jres else None
        cand_mean = _extract_test_mean(jres) if jres else None
        mtime = (res_json if res_json.is_file() else run_json).stat().st_mtime
        records.append({
            "group": group,
            "harness": harness,
            "model": model,
            "task": tid,
            "run_id": rmeta.get("run_id") or rdir.name,
            "status": status,
            "passed": passed,
            "gain": gain,
            "candidate_test_mean": cand_mean,
            "mtime": mtime,
        })
    return records


def build_report(
    *,
    suite: str | None = None,
    runs_dir: str | Path | None = None,
    run_ids: list[str] | None = None,
    harness_filter: str | None = None,
    model_filter: str | None = None,
    domain_filter: str | None = None,
    catalog_path: Path | None = None,
) -> dict:
    cat = read_catalog(catalog_path or (ROOT / "benchmark_tasks.json"))
    if not cat or "tasks_meta" not in cat:
        raise RuntimeError("benchmark_tasks.json is missing tasks_meta needed for leaderboard scoring.")

    all_task_ids = list(cat["task_ids"])
    tasks_meta = cat["tasks_meta"]
    domains = cat.get("domains") or []
    known_set = set(all_task_ids)

    if domain_filter and domain_filter != "all":
        selected_tasks = [tid for tid in all_task_ids if tasks_meta[tid].get("domain") == domain_filter]
        if not selected_tasks:
            valid = [d["id"] for d in domains]
            raise ValueError(f"Unknown --domain {domain_filter!r}; valid domains: {valid}")
    else:
        selected_tasks = all_task_ids

    rroot = Path(runs_dir) if runs_dir else RUNS_DIR
    if suite:
        raw_records = _load_records_from_suite(suite, rroot, known_set)
    elif run_ids:
        dirs = [Path(rid) if Path(rid).is_dir() else (rroot / rid) for rid in run_ids]
        raw_records = _load_records_from_runs(dirs, known_set)
    else:
        dirs = sorted(rroot.iterdir()) if rroot.is_dir() else []
        raw_records = _load_records_from_runs(dirs, known_set)

    if harness_filter:
        raw_records = [r for r in raw_records if r["harness"] == harness_filter]
    if model_filter:
        raw_records = [r for r in raw_records if r["model"] == model_filter]

    # Group by (harness, model) and keep latest run per task
    by_group_latest: dict[str, dict[str, dict]] = {}
    for rec in raw_records:
        g = rec["group"]
        tid = rec["task"]
        prev = by_group_latest.setdefault(g, {}).get(tid)
        if prev is None or rec["mtime"] >= prev["mtime"]:
            by_group_latest[g][tid] = rec

    leaderboard = []
    for group, task_map in sorted(by_group_latest.items()):
        per_task_rows = []
        evaluated = 0
        scored = 0
        passed = 0
        total_clipped = 0.0
        eval_clipped = 0.0

        for tid in selected_tasks:
            tmeta = tasks_meta[tid]
            rec = task_map.get(tid)
            if rec is None:
                per_task_rows.append({
                    "task": tid,
                    "domain": tmeta["domain"],
                    "metric": tmeta["metric"],
                    "status": "unattempted",
                    "passed": False,
                    "candidate_test_mean": None,
                    "baseline": tmeta["baseline"],
                    "target_ceiling": tmeta["target_ceiling"],
                    "gain": None,
                    "improvement_score": 0.0,
                    "raw_improvement_score": None,
                    "run_id": None,
                })
                continue

            evaluated += 1
            st = rec["status"]
            cmean = rec["candidate_test_mean"]
            clipped, raw = compute_task_improvement_score(cmean, tmeta["baseline"], tmeta["target_ceiling"], status=st)
            if st == "scored" and raw is not None:
                scored += 1
            if rec["passed"]:
                passed += 1
            total_clipped += clipped
            eval_clipped += clipped
            per_task_rows.append({
                "task": tid,
                "domain": tmeta["domain"],
                "metric": tmeta["metric"],
                "status": st,
                "passed": bool(rec["passed"]),
                "candidate_test_mean": cmean,
                "baseline": tmeta["baseline"],
                "target_ceiling": tmeta["target_ceiling"],
                "gain": rec["gain"],
                "improvement_score": clipped,
                "raw_improvement_score": raw,
                "run_id": rec["run_id"],
            })

        denom = len(selected_tasks)
        domain_breakdown = []
        if not domain_filter or domain_filter == "all":
            for dom in domains:
                did = dom["id"]
                drows = [r for r in per_task_rows if r["domain"] == did]
                d_denom = len(drows)
                if d_denom == 0:
                    continue
                domain_breakdown.append({
                    "domain": did,
                    "name": dom["name"],
                    "total": d_denom,
                    "evaluated": sum(1 for r in drows if r["status"] != "unattempted"),
                    "passed": sum(1 for r in drows if r["passed"]),
                    "improvement_score": sum(r["improvement_score"] for r in drows) / d_denom,
                })

        sample_rec = next(iter(task_map.values()))
        leaderboard.append({
            "group": group,
            "harness": sample_rec["harness"],
            "model": sample_rec["model"],
            "denominator": denom,
            "evaluated": evaluated,
            "scored": scored,
            "passed": passed,
            "pass_rate_pct": (passed / denom * 100.0) if denom else 0.0,
            "improvement_score": (total_clipped / denom) if denom else 0.0,
            "evaluated_only_improvement_score": (eval_clipped / evaluated) if evaluated else None,
            "by_domain": domain_breakdown,
            "per_task": per_task_rows,
        })

    leaderboard.sort(key=lambda g: (-g["improvement_score"], -g["passed"], g["group"]))
    return {
        "benchmark": "AutoScientistBench",
        "domain_filter": domain_filter or "all",
        "denominator": len(selected_tasks),
        "groups": leaderboard,
    }


def format_report_text(report: dict) -> str:
    groups = report["groups"]
    if not groups:
        return "No evaluated runs found. Run `python3 -m bench run ... --judge` or `python3 -m bench suite ...` first."

    lines = [
        f"AutoScientistBench Official Leaderboard Report (scope: {report['domain_filter']}, denominator: {report['denominator']} tasks)",
        "=" * 102,
        f"{'Group (Harness[:Model])':34s} {'Evaluated':>11s} {'Scored':>8s} {'Passed':>12s} {'Improvement Score':>20s} {'Eval-Only Mean':>15s}",
        "-" * 102,
    ]
    for g in groups:
        denom = g["denominator"]
        eval_str = f"{g['evaluated']}/{denom}"
        pass_str = f"{g['passed']}/{denom} ({g['pass_rate_pct']:.1f}%)"
        imp_str = f"{g['improvement_score']:.2f}%"
        eo = f"{g['evaluated_only_improvement_score']:.2f}%" if g["evaluated_only_improvement_score"] is not None else "-"
        lines.append(f"{g['group'][:34]:34s} {eval_str:>11s} {g['scored']:>8d} {pass_str:>12s} {imp_str:>20s} {eo:>15s}")

        if g["by_domain"] and g["evaluated"] > 0:
            lines.append("")
            lines.append(f"  Discipline breakdown for {g['group']}:")
            for d in g["by_domain"]:
                lines.append(
                    f"    - {d['name']:24s} ({d['evaluated']:2d}/{d['total']:2d} eval): "
                    f"Passed {d['passed']:2d}/{d['total']:2d} | Improvement Score {d['improvement_score']:6.2f}%"
                )
            lines.append("-" * 102)
    return "\n".join(lines)


def write_report_csv(report: dict, csv_path: str | Path):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group", "harness", "model", "task", "domain", "metric",
        "status", "passed", "candidate_test_mean", "baseline", "target_ceiling",
        "gain", "improvement_score", "raw_improvement_score", "run_id",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for g in report["groups"]:
            for row in g["per_task"]:
                out = {"group": g["group"], "harness": g["harness"], "model": g["model"], **row}
                w.writerow(out)
