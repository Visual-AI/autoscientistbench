#!/usr/bin/env python3
"""`bench` as the agent sees it inside the sandbox: status · eval-dev · validate-output · submit.

Runs under /usr/bin/python3 (stdlib only) with the bench sources bound read-only at /opt/arb/src.
Everything it knows comes from $BENCH_RUN_DIR/state.json written by bench-run.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from bench import envlock, evalrun, patch as patchmod, score as scoremod  # noqa: E402
from bench.budget import remaining_time
from bench.submission import client_submit
from bench.util import append_jsonl, read_json, read_jsonl, sha256_bytes, write_json  # noqa: E402


def run_dir():
    d = os.environ.get("BENCH_RUN_DIR")
    if not d or not Path(d, "state.json").exists():
        sys.exit("bench: BENCH_RUN_DIR/state.json not found; this command only works inside a bench-run sandbox")
    return Path(d)


def load_state():
    return read_json(run_dir() / "state.json")


def out(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_status(a, st):
    now = time.time()
    subs = sorted((run_dir() / "submissions").glob("*.json"))
    out({
        "run_id": st["run_id"], "task": st["pid"],
        "remaining_s": max(0, int(st["deadline_ts"] - now)), "budget_s": st["budget_s"],
        "deadline": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st["deadline_ts"])),
        "cpus": st["cpus"], "memory_gb": st.get("memory_gb"),
        "metric": st["metric"], "baseline_val": st.get("baseline_val"),
        "submissions": len(subs),
        "paths": {"workspace": st["ws"], "code": st["code"], "data": st["data"], "bench": st["bench"], "venv": st["venv"]},
    })


def _exec_env(st, seed):
    return evalrun.stage_env(st["venv"], st["cpus"], seed, st["data"], os.environ.get("HOME", "/home/agent"),
                             os.environ.get("TMPDIR", "/tmp"),
                             extra={**st.get("runtime_env", {}), "BENCH_BENCH_DIR": st["bench"], "BENCH_CODE_DIR": st["code"], **{k: v for k, v in os.environ.items() if k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                                                                                 "http_proxy", "https_proxy", "no_proxy",
                                                                                 "BENCH_RUN_DIR", "UV_CACHE_DIR")}})


def _tool_env(st, seed):
    # Trusted tools may import their frozen sibling modules, as they do in the judge.
    env = _exec_env(st, seed)
    env.pop("PYTHONSAFEPATH", None)
    return env


def cmd_eval_dev(a, st):
    seed = a.seed if a.seed is not None else 0
    dev_root = run_dir() / "dev" / time.strftime("%Y%m%d-%H%M%S")
    io = st["io"]
    caps = {"train": None, "predict": None, "score": None}
    env = _exec_env(st, seed)
    py = str(Path(st["venv"]) / "bin" / "python")
    t0 = time.time()
    res = evalrun.one_seed(seed=seed, code_dir=st["code"], data_dir=st["data"], out_root=dev_root,
                           input_path=io["val_input"], labels_path=io["val_labels"],
                           validator_py=io["validator"], scorer_py=io["scorer"],
                           exec_env=env, tool_env=_tool_env(st, seed), tool_python=py, caps=caps, cwd=st["code"],
                           deadline_ts=st["deadline_ts"])
    m = st["metric"]
    rep = {"split": "val", "seed": seed, "status": res["status"], "metrics": res["metrics"], "timing": res["timing"],
           "failure": res["failure"], "baseline_val": st.get("baseline_val"), "out_dir": str(dev_root),
           "wall_s": round(time.time() - t0, 1)}
    if res["metrics"] and st.get("baseline_val") and m["name"] in res["metrics"] and m["name"] in st["baseline_val"]:
        rep["gain_vs_baseline_val"] = scoremod.gain(res["metrics"][m["name"]], st["baseline_val"][m["name"]], m["higher_is_better"])
        rep["judge_threshold_hint"] = m.get("delta")
    append_jsonl(run_dir() / "events.jsonl", {"kind": "eval-dev", "ts": time.time(), **{k: rep[k] for k in
                 ("seed", "status", "metrics", "timing", "wall_s")}, "gain_vs_baseline_val": rep.get("gain_vs_baseline_val")})
    if not a.keep and res["status"] == "ok":
        # keep logs + prediction, drop the model to save disk
        import shutil
        shutil.rmtree(dev_root / "model", ignore_errors=True)
    out(rep)
    return 0 if res["status"] == "ok" else 1


def cmd_validate(a, st):
    io = st["io"]
    env = _tool_env(st, 0)
    py = str(Path(st["venv"]) / "bin" / "python")
    try:
        ok, msg = evalrun.validate(io["validator"], a.file, a.input or io["val_input"], py, env,
                                   timeout=remaining_time(st["deadline_ts"]))
    except TimeoutError as exc:
        ok, msg = False, str(exc)
    out({"ok": ok, "message": msg})
    return 0 if ok else 1


def json_safe(value):
    """Non-finite floats -> their repr, recursively.

    A scorer signals a violated guard with a non-finite metric, and the submission protocol rejects non-finite JSON
    (bench/submission.py keeps allow_nan=False). Passing such metrics through as self_reported_val made every later
    `bench submit` fail with "Out of range float values are not JSON compliant" until some eval-dev came back finite:
    a guard violation also locked the agent out of submitting its fix, which cost 072_EMP both of its runs. The value
    is informational, so it is reported as "inf" rather than dropped.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def make_submission(st, final=False, msg="", source="submit"):
    """Freeze env -> code/env.lock, diff code/ against the adapt commit, check, store. -> record dict."""
    if time.time() >= st["deadline_ts"]:
        return {"ok": False, "stage": "deadline", "problems": ["submission deadline has passed"]}
    code = Path(st["code"])
    if st.get("env_mode") == "snapshot":
        lock_text = (code / "env.lock").read_text()
        if lock_text != st.get("snapshot_lock"):
            return {"ok": False, "stage": "env.lock", "problems": ["snapshot env.lock is immutable"]}
    else:
        lock_text = envlock.freeze(st["venv"], env=_exec_env(st, 0))
    ok, problems, pins = envlock.validate_lock_text(lock_text)
    if not ok:
        return {"ok": False, "stage": "env.lock", "problems": problems}
    (code / "env.lock").write_text(lock_text)
    pb = patchmod.make_patch(code, st["adapt_commit"])
    ok, problems, files = patchmod.check_patch(pb, st["patch_limits"])
    if not ok:
        return {"ok": False, "stage": "patch", "problems": problems}
    if not st.get("submission_socket"):
        return {"ok": False, "stage": "submit", "problems": ["trusted submission service is unavailable"]}
    evs = [e for e in read_jsonl(run_dir() / "events.jsonl") if e.get("kind") == "eval-dev" and e.get("status") == "ok"]
    try:
        rec = client_submit(st["submission_socket"], pb, final=bool(final), msg=msg,
                            self_reported_val=json_safe(evs[-1].get("metrics")) if evs else None)
    except (OSError, ValueError) as exc:
        return {"ok": False, "stage": "submit", "problems": [str(exc)]}
    append_jsonl(run_dir() / "events.jsonl", {"kind": "submit", "ts": rec.get("ts", time.time()),
                  "n": rec.get("n"), "final": bool(final), "ok": rec.get("ok", False),
                  "bytes": rec.get("bytes", len(pb)), "files": rec.get("files", len(files))})
    return rec


def cmd_submit(a, st):
    rec = make_submission(st, final=a.final, msg=a.message or "")
    out(rec)
    return 0 if rec.get("ok") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench", description="AutoResearch Bench — agent-side commands")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status", help="remaining time, budget, cpus, metric, baseline val score")
    e = sp.add_parser("eval-dev", help="train -> predict(val) -> validate -> score, exactly like the judge does on test")
    e.add_argument("--seed", type=int, default=None)
    e.add_argument("--keep", action="store_true", help="keep the trained model directory")
    v = sp.add_parser("validate-output", help="check a prediction file against the task's output schema")
    v.add_argument("file")
    v.add_argument("--input", default=None)
    s = sp.add_parser("submit", help="freeze env -> code/env.lock, diff code/ vs the baseline commit, store as a submission")
    s.add_argument("--final", action="store_true")
    s.add_argument("-m", "--message", default="")
    a = ap.parse_args(argv)
    st = load_state()
    return {"status": cmd_status, "eval-dev": cmd_eval_dev, "validate-output": cmd_validate, "submit": cmd_submit}[a.cmd](a, st) or 0


if __name__ == "__main__":
    sys.exit(main())
