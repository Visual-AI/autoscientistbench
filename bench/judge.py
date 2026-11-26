"""bench-judge: patch -> clean clone -> rebuild env -> k seeds (train/predict in a no-network sandbox) -> validate ->
scorer -> reference parity -> gain/PASS -> result.json. Also `bench reference` (the archived baseline numbers)."""
import json
import os
import shutil
import time
from pathlib import Path

from . import LOCAL_ROOT, RUNS_DIR, UV_CACHE
from .budget import deadline_env, positive_seconds, remaining_time
from .envlock import build_venv, validate_lock_text, venv_env
from .runtime import bind_runtime, build_environment, host_python, is_snapshot, runtime_env, tool_launcher
from .evalrun import one_seed, predict, score as run_scorer, stage_env, validate as run_validate
from .patch import apply_patch, check_patch, clean_clone, git, materialize_code_repo
from .sandbox import IN_TMP, IN_VENV, IN_WS, Sandbox, bwrap_bin, python_binds_for_venv
from .score import FAIL_GAIN, aggregate, decide, parity_ok
from .task import load_task
from .util import (ensure_dir, free_cpus, log, now_iso, parse_cpuset, read_json, rm_rf, sha256_bytes, sha256_file,
                   thread_env, ts_compact, write_json)

IN_OUT = f"{IN_WS}/out"
IN_INPUT = f"{IN_WS}/input"
IN_HOME_J = "/home/judge"


class Stage:
    """A sandboxed train/predict runner for one code checkout + venv."""

    def __init__(self, task, code_host, venv_host, tmp_host, scratch_host, cpuset, cpus):
        self.task, self.code, self.venv, self.tmp, self.scratch = task, Path(code_host), Path(venv_host), Path(tmp_host), Path(scratch_host)
        self.cpuset, self.cpus = cpuset, cpus
        self.deadline_ts = None
        self.n = 0
        jb = task.judge_bench_dir
        self.bench_host = jb if (jb / "scorer.py").exists() else task.bench_dir   # frozen generators/environments live here

    def prefix(self, seed, out_host, input_host=None):
        self.n += 1
        sb = Sandbox(cpuset=self.cpuset, unshare_net=True, own_uid=True)   # runs submitted code: a uid of its own
        sb.rw_bind(self.code, f"{IN_WS}/code").ro_bind(self.task.data_dir, f"{IN_WS}/data").rw_bind(out_host, IN_OUT)
        sb.ro_bind(self.bench_host, f"{IN_WS}/bench")
        if input_host is not None:
            sb.ro_bind(input_host, IN_INPUT)
        sb.ro_bind(self.venv, IN_VENV)
        for d in ([] if is_snapshot(self.task) else python_binds_for_venv(self.venv)):
            sb.ro_bind(d, d)
        sb.tmpfs(IN_HOME_J).rw_bind(self.tmp, IN_TMP)
        bind_runtime(sb, self.task, deadline_ts=self.deadline_ts)
        sb.setenv(**deadline_env(stage_env(IN_VENV, self.cpus, seed, f"{IN_WS}/data", IN_HOME_J, IN_TMP,
                              extra={"PATH": f"{IN_VENV}/bin:/usr/local/bin:/usr/bin:/bin", **runtime_env(self.task, deadline_ts=self.deadline_ts), "BENCH_JUDGE": "1",
                                     "BENCH_BENCH_DIR": f"{IN_WS}/bench", "BENCH_CODE_DIR": f"{IN_WS}/code"}), self.deadline_ts))
        sb.cwd = f"{IN_WS}/code"
        return sb.argv([], self.scratch / f"sb{self.n}")

    def path_map(self, out_host, input_host=None):
        pairs = [(str(self.code), f"{IN_WS}/code"), (str(out_host), IN_OUT), (str(self.task.data_dir), f"{IN_WS}/data"),
                 (str(self.bench_host), f"{IN_WS}/bench")]
        if input_host is not None:
            pairs.append((str(input_host), IN_INPUT))

        def pm(p):
            p = str(p)
            for h, i in pairs:
                if p == h:
                    return i
                if p.startswith(h.rstrip("/") + "/"):
                    return i + p[len(h.rstrip("/")):]
            return p
        return pm


class DirectStage:
    """No bwrap: train/predict run as plain subprocesses with the judge's environment variables (same BENCH_* vars, thread
    caps, tmp HOME). No network or filesystem isolation -- only for `bench reference --no-sandbox` on hosts where bwrap cannot
    run (e.g. inside another sandbox); the real judge always uses Stage."""

    def __init__(self, task, code_host, venv_host, tmp_host, scratch_host, cpuset, cpus):
        self.task, self.code, self.venv, self.tmp = task, Path(code_host), Path(venv_host), Path(tmp_host)
        self.cpuset, self.cpus = cpuset, cpus
        self.deadline_ts = None
        self.home = ensure_dir(Path(scratch_host) / "home")
        jb = task.judge_bench_dir
        self.bench_host = jb if (jb / "scorer.py").exists() else task.bench_dir

    def prefix(self, seed, out_host, input_host=None):
        return None

    def path_map(self, out_host, input_host=None):
        return lambda p: str(p)

    def exec_env(self, seed):
        return deadline_env(stage_env(self.venv, self.cpus, seed, self.task.data_dir, self.home, self.tmp,
                         extra={"PATH": f"{self.venv}/bin:/usr/local/bin:/usr/bin:/bin", **runtime_env(self.task, host=True, deadline_ts=self.deadline_ts), "BENCH_JUDGE": "1",
                                "BENCH_BENCH_DIR": str(self.bench_host), "BENCH_CODE_DIR": str(self.code)}), self.deadline_ts)


def tool_ctx(venv_host, cpus, task=None):
    env = venv_env(venv_host, base={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(Path(venv_host).parent),
                                    "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(thread_env(cpus))
    if task is not None and is_snapshot(task):
        env["ARB_BWRAP"] = bwrap_bin()
        return env, tool_launcher(task, Path(venv_host).parent, cpus)
    return env, str(Path(venv_host) / "bin" / "python")


def run_seeds(task, stage: Stage, seeds, out_root, tool_env, tool_py, validator, scorer, caps, val_too=True,
              deadline_ts=None):
    """-> list of per-seed dicts (test metrics, val metrics, timing, failures)."""
    res = []
    stage.deadline_ts = deadline_ts
    tool_env = deadline_env(tool_env, deadline_ts)
    for s in seeds:
        sd = ensure_dir(Path(out_root) / f"seed_{s}")
        try:
            pm = stage.path_map(sd, task.test_input)
            exec_env = stage.exec_env(s) if hasattr(stage, "exec_env") else None     # DirectStage (no bwrap) needs the env itself
            prefix_train = stage.prefix(s, sd)
            prefix_predict = stage.prefix(s, sd, task.test_input)
            r = one_seed(seed=s, code_dir=stage.code, data_dir=task.data_dir, out_root=sd, input_path=task.test_input,
                         labels_path=task.test_labels, validator_py=validator, scorer_py=scorer, exec_env=exec_env,
                         tool_env=tool_env, tool_python=tool_py, caps=caps,
                         prefix_train=prefix_train, prefix_predict=prefix_predict,
                         path_map=pm, pred_name="pred_test.out", cwd=(str(stage.code) if exec_env is not None else None),
                         deadline_ts=deadline_ts)
        except TimeoutError as exc:
            r = {"status": "budget_exhausted", "metrics": None, "timing": {}, "pred_path": None,
                 "failure": {"stage": "prepare", "reason": str(exc), "log": None}}
        r["seed"] = s
        r["val"] = None
        if r["status"] == "ok" and val_too:
            pv = sd / "pred_val.out"
            pmv = stage.path_map(sd)
            try:
                prefix_val = stage.prefix(s, sd)
                pr = predict(pmv(stage.code), pmv(sd / "model"), pmv(task.val_input()), pmv(pv), exec_env,
                             remaining_time(deadline_ts, caps.get("predict")), sd / "predict_val.log",
                             prefix=prefix_val, cwd=(str(stage.code) if exec_env is not None else None))
                r["timing"]["predict_val_s"] = pr["elapsed_s"]
                remaining_time(deadline_ts)
                if pr["rc"] == 0 and pv.exists():
                    ok, msg = run_validate(validator, pv, task.val_input(), tool_py, tool_env,
                                           timeout=remaining_time(deadline_ts))
                    if ok:
                        r["val"] = run_scorer(scorer, pv, task.val_labels(), tool_py, tool_env,
                                               timeout=remaining_time(deadline_ts, caps.get("score")))
                    else:
                        r["val_error"] = msg[-500:]
                else:
                    r["val_error"] = f"predict(val) rc={pr['rc']}"
                remaining_time(deadline_ts)
            except Exception as exc:
                r["val_error"] = str(exc)[-500:]
                if deadline_ts is not None and time.time() >= deadline_ts:
                    r.update(status="budget_exhausted", failure={"stage": "validation",
                             "reason": "overall evaluation budget exhausted", "log": None})
        log(f"  seed {s}: {r['status']} test={r['metrics']} val={r['val']} timing={r['timing']}")
        res.append(r)
        if r["status"] != "ok":
            break   # one failed seed = the submission failed; no point burning more compute
    return res


def _authoritative_bench(task):
    jb = task.judge_bench_dir
    b = jb if (jb / "scorer.py").exists() else task.bench_dir
    return b / "validate.py", b / "scorer.py"


def _finalize(result, rec, run_dir, status, failure=None):
    result["status"] = status
    result["failure"] = failure
    if status not in ("scored", "infra_error"):
        result["gain"] = FAIL_GAIN
        result["pass"] = False
    result["finished"] = now_iso()
    write_json(rec / "result.json", result)
    if run_dir:
        write_json(Path(run_dir) / "result.json", result)
    log(f"judge: {status} gain={result.get('gain')} pass={result.get('pass')} -> {rec / 'result.json'}")
    return result


def bench_judge(run_dir=None, pid=None, patch_path=None, *, seeds=None, reference_check=None, reference_now=False,
                cpus=None, cpuset=None, keep=False, val_judge=None, uv_cache=None, local_root=None, label=None,
                budget_hours=None):
    # The parity re-run of the baseline is off by default since 2026-11-09 (user decision): it never entered the score, which is
    # always computed against the archived reference, but it cost ~70% of all scorer time and turned a drifting baseline into a
    # lost candidate result (infra_error/reference_drift). BENCH_REFERENCE_CHECK=1 or --reference-check brings it back.
    if reference_check is None:
        reference_check = os.environ.get("BENCH_REFERENCE_CHECK", "") not in ("", "0")
    # ---- resolve inputs --------------------------------------------------------------
    run_meta, harness_name, run_id = None, None, None
    if run_dir:
        run_dir = Path(run_dir)
        if not run_dir.is_absolute() and not run_dir.exists():
            run_dir = RUNS_DIR / run_dir
        run_meta = read_json(run_dir / "run.json")
        pid = pid or run_meta["task"]
        patch_path = patch_path or (run_dir / "final.patch")
        harness_name = (run_meta.get("harness") or {}).get("name")
        run_id = run_meta["run_id"]
        final_meta = read_json(run_dir / "final.json", default={})
    else:
        final_meta = {"source": "file", "ok": None}
    if not pid or patch_path is None:
        raise ValueError("need --run <run_dir> or --task <pid> --patch <file>")
    task = load_task(pid)
    budget_s = positive_seconds(float(budget_hours) * 3600 if budget_hours is not None else
                                (run_meta or {}).get("judge_budget_s", (run_meta or {}).get(
                                    "budget_s", task.d.get("budget_hours_default", 2) * 3600)))
    judge_started = time.time()
    deadline_ts = judge_started + budget_s
    pb = Path(patch_path).read_bytes() if Path(patch_path).exists() else b""
    seeds = [int(s) for s in seeds] if seeds else task.seeds
    cpus = int(cpus or task.cpus)
    cpuset = parse_cpuset(cpuset) if cpuset else free_cpus()[:cpus]
    val_judge = task.d["judge"].get("val_judge", True) if val_judge is None else bool(val_judge)
    jid = f"judge__{run_id or task.pid}__{label or ts_compact()}"
    local = Path(local_root or LOCAL_ROOT) / "judge" / jid
    rec = (run_dir / "judge") if run_dir else (RUNS_DIR / jid)
    ensure_dir(local / "tmp"), ensure_dir(local / "sandbox"), ensure_dir(rec)
    uv_cache = Path(uv_cache or UV_CACHE)
    caps = {"train": task.train_cap_s, "predict": task.predict_cap_s, "score": task.score_cap_s}
    validator, scorer = _authoritative_bench(task)
    m = task.metric_name
    result = {"task": task.pid, "run": run_id, "harness": harness_name, "status": "running", "judge_id": jid,
              "patch": {"sha256": sha256_bytes(pb), "bytes": len(pb), "files": None, "source": final_meta.get("source"),
                        "submit_ok": final_meta.get("ok"), "submit_problems": final_meta.get("problems")},
              "seeds": seeds, "cpus": cpus, "candidate": {}, "reference": {}, "gain": None, "threshold": None, "pass": None,
              "diagnostics": {}, "timing": {}, "failure": None, "started": now_iso(), "metric": task.metric,
              "budget_s": budget_s, "deadline_ts": deadline_ts}
    write_json(rec / "result.json", result)
    log(f"judge {jid}: task={task.pid} patch={len(pb)}B seeds={seeds} cpus={cpuset}")
    if run_dir and (final_meta.get("source") not in ("final", "last_submission") or not final_meta.get("ok")):
        return _finalize(result, rec, run_dir, "no_submission",
                         {"stage": "submission", "reason": "no valid explicit submission before the agent deadline"})
    probs = task.problems(deep=True)
    if probs:
        return _finalize(result, rec, run_dir, "infra_error", {"stage": "task", "reason": "; ".join(probs)})

    # ---- 1. static patch checks -----------------------------------------------------------
    ok, problems, files = check_patch(pb, task.patch_limits)
    result["patch"]["files"] = len(files)
    result["patch"]["file_list"] = [f["path"] for f in files][:200]
    if not ok:
        return _finalize(result, rec, run_dir, "invalid_patch", {"stage": "apply", "reason": "; ".join(problems)[:2000]})

    # ---- 2. clean clone + apply -------------------------------------------------------------
    cand_code = local / "cand" / "code"
    try:
        materialize_code_repo(task.code_dir, cand_code, task.adapt_commit)
        if pb.strip():
            (local / "cand.patch").write_bytes(pb)
            apply_patch(cand_code, local / "cand.patch")
    except Exception as e:  # noqa: BLE001
        return _finalize(result, rec, run_dir, "invalid_patch", {"stage": "apply", "reason": str(e)[-2000:]})

    # ---- 3. environments ----------------------------------------------------------------------
    lock_path = cand_code / "env.lock"
    if not lock_path.exists():
        return _finalize(result, rec, run_dir, "invalid_patch", {"stage": "apply", "reason": "code/env.lock missing after patch"})
    ok, problems, pins = validate_lock_text(lock_path.read_text())
    if not ok:
        return _finalize(result, rec, run_dir, "invalid_patch", {"stage": "build", "reason": "env.lock: " + "; ".join(problems)[:2000]})
    t0 = time.time()
    log("judge: build candidate venv")
    try:
        build_timeout = remaining_time(deadline_ts, task.build_cap_s)
    except TimeoutError as exc:
        return _finalize(result, rec, run_dir, "budget_exhausted", {"stage": "build", "reason": str(exc)})
    b = build_environment(task, local / "cand" / "venv", lock_path, task.indexes, uv_cache, rec / "build_cand.log", cap_s=build_timeout)
    result["timing"]["build_s"] = b["elapsed_s"]
    if time.time() >= deadline_ts:
        return _finalize(result, rec, run_dir, "budget_exhausted", {"stage": "build", "reason": "overall evaluation budget exhausted"})
    if not b["ok"]:
        return _finalize(result, rec, run_dir, "build_failed",
                         {"stage": "build", "reason": f"uv {b['stage']} failed" + (" (timeout)" if b.get("timed_out") else ""),
                          "log": str(rec / "build_cand.log")})
    ref_lock = task.reference_dir / "env.lock"
    if not ref_lock.exists():
        # baseline lock = env.lock at the adapt commit (or directly from code/env.lock when unbundled)
        ref_lock = local / "ref_env.lock"
        if (task.code_dir / ".git").exists():
            ref_lock.write_text(git(["show", f"{task.adapt_commit}:env.lock"], task.code_dir).stdout)
        else:
            shutil.copy2(task.code_dir / "env.lock", ref_lock)
    log("judge: build reference venv (trusted scorer env)")
    try:
        build_timeout = remaining_time(deadline_ts, task.build_cap_s)
    except TimeoutError as exc:
        return _finalize(result, rec, run_dir, "budget_exhausted", {"stage": "build", "reason": str(exc)})
    b2 = build_environment(task, local / "ref" / "venv", ref_lock, task.indexes, uv_cache, rec / "build_ref.log", cap_s=build_timeout)
    result["timing"]["build_ref_s"] = b2["elapsed_s"]
    if time.time() >= deadline_ts:
        return _finalize(result, rec, run_dir, "budget_exhausted", {"stage": "build", "reason": "overall evaluation budget exhausted"})
    if not b2["ok"]:
        return _finalize(result, rec, run_dir, "infra_error", {"stage": "build", "reason": "reference venv build failed",
                                                                "log": str(rec / "build_ref.log")})
    tool_env, tool_py = tool_ctx(local / "ref" / "venv", cpus, task)

    # ---- 4/5. candidate seeds (+ val) ------------------------------------------------------------
    stage = Stage(task, cand_code, local / "cand" / "venv", local / "tmp", local / "sandbox" / "cand", cpuset, cpus)
    log(f"judge: candidate seeds {seeds}")
    cand = run_seeds(task, stage, seeds, local / "cand" / "runs", tool_env, tool_py, validator, scorer, caps,
                     val_too=val_judge, deadline_ts=deadline_ts)
    result["candidate"] = {
        "per_seed": [{"seed": r["seed"], "status": r["status"], "test": r["metrics"], "val": r["val"], "timing": r["timing"]} for r in cand],
        "test": aggregate([(r["metrics"] or {}).get(m) for r in cand if r["status"] == "ok"]),
        "val": aggregate([(r["val"] or {}).get(m) for r in cand if r["val"]]),
    }
    result["timing"]["train_s"] = [r["timing"].get("train_s") for r in cand]
    result["timing"]["predict_s"] = [r["timing"].get("predict_s") for r in cand]
    for r in cand:
        shutil.copy2(r["pred_path"], rec / f"pred_test_seed{r['seed']}.out") if r.get("pred_path") else None
        for lg in ("train.log", "predict.log", "predict_val.log"):
            src = local / "cand" / "runs" / f"seed_{r['seed']}" / lg
            if src.exists():
                shutil.copy2(src, rec / f"seed{r['seed']}_{lg}")
    bad = [r for r in cand if r["status"] != "ok"]
    if bad:
        r = bad[0]
        return _finalize(result, rec, run_dir, r["status"], dict(r["failure"], seed=r["seed"]))

    # ---- 6. reference ------------------------------------------------------------------------------
    ref_path = task.reference_dir / "result.json"
    ref = read_json(ref_path) if ref_path.exists() else None
    if ref is None and not reference_now:
        return _finalize(result, rec, run_dir, "infra_error",
                         {"stage": "reference", "reason": f"no archived reference at {ref_path}; run `bench reference {task.pid}`"})
    ref_code = local / "ref" / "code"
    materialize_code_repo(task.code_dir, ref_code, task.adapt_commit)
    rstage = Stage(task, ref_code, local / "ref" / "venv", local / "tmp", local / "sandbox" / "ref", cpuset, cpus)
    if ref is None:
        log("judge: computing reference now (not archived)")
        rr = run_seeds(task, rstage, seeds, local / "ref" / "runs", tool_env, tool_py, validator, scorer, caps,
                       val_too=val_judge, deadline_ts=deadline_ts)
        if any(r["status"] == "budget_exhausted" for r in rr):
            return _finalize(result, rec, run_dir, "budget_exhausted",
                             {"stage": "reference", "reason": "overall evaluation budget exhausted"})
        if any(r["status"] != "ok" for r in rr):
            return _finalize(result, rec, run_dir, "infra_error",
                             {"stage": "reference", "reason": "reference execution failed"})
        ref = {"test": aggregate([(r["metrics"] or {}).get(m) for r in rr if r["status"] == "ok"]),
               "val": aggregate([(r["val"] or {}).get(m) for r in rr if r["val"]]),
               "per_seed": {str(r["seed"]): {"test": r["metrics"], "val": r["val"]} for r in rr}, "archived": False}
        result["reference"] = ref
    else:
        result["reference"] = {k: ref.get(k) for k in ("test", "val", "hashes", "created")}
        result["reference"]["archived"] = True
        if not reference_check:
            result["reference"]["parity"] = {"checked": False, "reason": "baseline parity re-run disabled; scored against the archived reference only"}
        if reference_check and not bad:
            s0 = seeds[0]
            log(f"judge: reference parity re-run seed {s0}")
            rr = run_seeds(task, rstage, [s0], local / "ref" / "runs", tool_env, tool_py, validator, scorer, caps,
                           val_too=False, deadline_ts=deadline_ts)
            r0 = rr[0]
            if r0["status"] == "budget_exhausted":
                return _finalize(result, rec, run_dir, "budget_exhausted",
                                 {"stage": "reference", "reason": "overall evaluation budget exhausted"})
            for lg in ("train.log", "predict.log"):      # keep the parity re-run's logs: a failing baseline must be diagnosable
                src = local / "ref" / "runs" / f"seed_{s0}" / lg
                if src.exists():
                    shutil.copy2(src, rec / f"ref_seed{s0}_{lg}")
            mdir = local / "ref" / "runs" / f"seed_{s0}" / "model"
            for extra in sorted(list(mdir.glob("*.log")) + list(mdir.glob("*/*.log")))[:10]:   # e.g. build_ext.log written by train.sh
                shutil.copy2(extra, rec / f"ref_seed{s0}_model_{extra.name}")
            if r0["status"] != "ok":
                log(f"judge: reference parity re-run failed: {r0['failure']}")
                result["reference"]["parity_failure"] = r0["failure"]
            obs = (r0["metrics"] or {}).get(m) if r0["status"] == "ok" else None
            per = (ref.get("per_seed") or {}).get(str(s0), {})
            archived = ((per.get("test") or {}).get(m)) if per else None
            archived_mean = archived if archived is not None else (ref["test"] or {}).get("mean")
            okp, info = parity_ok(obs, archived_mean, (ref["test"] or {}).get("std"), task.d["judge"].get("parity_rel_tol", 0.0),
                                  task.d["judge"].get("parity_abs_tol", 0.0))
            result["reference"]["parity"] = {"seed": s0, "observed": obs, "archived": archived_mean, "ok": okp, **(info or {}),
                                             "status": r0["status"]}
            result["timing"]["reference_s"] = r0["timing"].get("train_s")
            if not okp:
                return _finalize(result, rec, run_dir, "infra_error",
                                 {"stage": "reference", "reason": f"reference_drift: seed {s0} gave {obs} vs archived {archived_mean} (tol {info and info.get('tol')})"})

    # ---- 7. decide ----------------------------------------------------------------------------------
    if time.time() >= deadline_ts:
        return _finalize(result, rec, run_dir, "budget_exhausted",
                         {"stage": "score", "reason": "overall evaluation budget exhausted"})
    cm = result["candidate"]["test"]["mean"]
    rm_, rs = (ref["test"] or {}).get("mean"), (ref["test"] or {}).get("std")
    d = decide(cm, rm_, rs, task.higher_is_better, task.metric["delta"])
    result.update({"gain": d["gain_clipped"], "gain_raw": d["gain"], "threshold": d["threshold"], "pass": d["pass"]})
    vj = result["candidate"]["val"]["mean"]
    rv = (ref.get("val") or {}).get("mean")
    srv = (final_meta.get("self_reported_val") or {}).get(m) if isinstance(final_meta.get("self_reported_val"), dict) else None
    result["diagnostics"] = {
        "val_judge": vj, "val_reference": rv, "val_self_reported": srv,
        "val_gain": decide(vj, rv, 0.0, task.higher_is_better, 0.0)["gain"] if (vj is not None and rv is not None) else None,
        "val_test_gap": (decide(vj, rv, 0.0, task.higher_is_better, 0.0)["gain"] - d["gain"])
        if (vj is not None and rv is not None and d["gain"] is not None) else None,
        "submissions": (run_meta or {}).get("submissions"), "first_improvement_s": (run_meta or {}).get("first_improvement_s"),
        "tokens_in": ((run_meta or {}).get("agent") or {}).get("tokens_in"),
        "tokens_out": ((run_meta or {}).get("agent") or {}).get("tokens_out"),
        "ended": (run_meta or {}).get("ended_reason"),
    }
    result["timing"]["judge_wall_s"] = round(time.time() - judge_started, 1)
    out = _finalize(result, rec, run_dir, "scored")
    if not keep:
        rm_rf(local)
    return out


def compute_reference(pid, *, seeds=None, cpus=None, cpuset=None, keep=False, uv_cache=None, local_root=None,
                      sandbox=True, budget_hours=None):
    """Baseline numbers under the bench protocol -> judge/reference/{result.json, env.lock, pred_*.out}; updates task.json."""
    task = load_task(pid)
    budget_s = positive_seconds(float(budget_hours if budget_hours is not None else
                                     task.d.get("budget_hours_default", 2)) * 3600)
    deadline_ts = time.time() + budget_s
    probs = task.problems(deep=True)
    if probs:
        raise RuntimeError("task not ready: " + "; ".join(probs))
    seeds = [int(s) for s in seeds] if seeds else task.seeds
    cpus = int(cpus or task.cpus)
    cpuset = parse_cpuset(cpuset) if cpuset else free_cpus()[:cpus]
    local = Path(local_root or LOCAL_ROOT) / "reference" / f"{task.pid}__{ts_compact()}"
    ensure_dir(local / "tmp"), ensure_dir(local / "sandbox")
    ref_dir = ensure_dir(task.reference_dir)
    caps = {"train": task.train_cap_s, "predict": task.predict_cap_s, "score": task.score_cap_s}
    validator, scorer = _authoritative_bench(task)
    m = task.metric_name
    code = local / "code"
    materialize_code_repo(task.code_dir, code, task.adapt_commit)
    lock = code / "env.lock"
    log(f"reference {task.pid}: build venv")
    b = build_environment(task, local / "venv", lock, task.indexes, Path(uv_cache or UV_CACHE), ref_dir / "build.log",
                           cap_s=remaining_time(deadline_ts, task.build_cap_s))
    if not b["ok"]:
        raise RuntimeError("venv build failed; see " + str(ref_dir / "build.log"))
    tool_env, tool_py = tool_ctx(local / "venv", cpus, task)
    stage_cls = Stage if sandbox else DirectStage
    if not sandbox:
        log(f"reference {task.pid}: --no-sandbox (no bwrap: no network/filesystem isolation for the baseline run)")
    stage = stage_cls(task, code, local / "venv", local / "tmp", local / "sandbox", cpuset, cpus)
    rr = run_seeds(task, stage, seeds, local / "runs", tool_env, tool_py, validator, scorer, caps,
                   val_too=True, deadline_ts=deadline_ts)
    bad = [r for r in rr if r["status"] != "ok"]
    if bad:
        raise RuntimeError(f"baseline failed under the bench protocol: {bad[0]['status']} {bad[0]['failure']}")
    res = {
        "task": task.pid, "created": now_iso(), "seeds": seeds, "cpus": cpus, "adapt_commit": task.adapt_commit, "sandbox": bool(sandbox),
        "budget_s": budget_s,
        "test": aggregate([r["metrics"][m] for r in rr]), "val": aggregate([(r["val"] or {}).get(m) for r in rr]),
        "per_seed": {str(r["seed"]): {"test": r["metrics"], "val": r["val"], "timing": r["timing"]} for r in rr},
        "timing": {"train_s": [r["timing"]["train_s"] for r in rr], "predict_s": [r["timing"]["predict_s"] for r in rr], "build_s": b["elapsed_s"]},
    }
    shutil.copy2(lock, ref_dir / "env.lock")
    for r in rr:
        shutil.copy2(r["pred_path"], ref_dir / f"pred_test_seed{r['seed']}.out")
        pv = local / "runs" / f"seed_{r['seed']}" / "pred_val.out"
        if pv.exists():
            shutil.copy2(pv, ref_dir / f"pred_val_seed{r['seed']}.out")
    if task.base_commit and (task.code_dir / ".git").exists():
        (ref_dir / "adapt.patch").write_text(git(["diff", task.base_commit, task.adapt_commit], task.code_dir).stdout)
    res["hashes"] = {"env_lock": sha256_file(ref_dir / "env.lock"), "scorer": sha256_file(scorer), "validate": sha256_file(validator),
                     "test_labels": sha256_file(task.test_labels) if task.test_labels.is_file() else None,
                     "adapt_patch": sha256_file(ref_dir / "adapt.patch") if (ref_dir / "adapt.patch").exists() else None}
    write_json(ref_dir / "result.json", res)
    # Record measured baseline costs for diagnostics, not per-stage limits.
    tj = read_json(task.dir / "task.json")
    tj.setdefault("baseline", {})["val"] = {m: res["val"]["mean"]}
    tj["baseline"]["train_min"] = round(max(max(res["timing"]["train_s"]), 60.0) / 60, 2)
    tj["baseline"]["eval_min"] = round(max(max(res["timing"]["predict_s"]), 30.0) / 60, 2)
    tj["baseline"]["measured_s"] = {"train": res["timing"]["train_s"], "predict": res["timing"]["predict_s"]}
    tj.setdefault("judge", {})["reference_hash"] = "sha256:" + sha256_file(ref_dir / "result.json")
    write_json(task.dir / "task.json", tj)
    log(f"reference {task.pid}: test {res['test']} val {res['val']} -> {ref_dir}")
    if not keep:
        rm_rf(local)
    return res
