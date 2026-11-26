"""The four fixed calls into task code, used identically by `bench eval-dev` (agent side, val) and the judge (test):

    bash code/train.sh --out <dir>                                 env: BENCH_SEED BENCH_CPUS BENCH_DATA_DIR
    bash code/predict.sh --model <dir> --input <path> --output <file>
    python bench/validate.py <pred> <input>                        exit 0 = ok
    python bench/scorer.py <pred> <labels>                         prints JSON {metric_name: value, ...}

Paths are passed as the command must see them (inside-sandbox paths when `prefix` is a bwrap/taskset argv).
"""
import json
import os
import time
from pathlib import Path

from .envlock import venv_env
from .budget import deadline_env, remaining_time
from .util import run, capture, thread_env


def stage_env(venv_inside, cpus, seed, data_dir, home, tmp, extra=None):
    e = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
         "HOME": str(home), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
         "XDG_CACHE_HOME": f"{home}/.cache", "XDG_CONFIG_HOME": f"{home}/.config", "XDG_DATA_HOME": f"{home}/.local/share",
         "MPLCONFIGDIR": f"{home}/.config/matplotlib", "HF_HOME": f"{home}/.cache/hf", "TORCH_HOME": f"{home}/.cache/torch",
         "BENCH_SEED": str(seed), "BENCH_CPUS": str(cpus), "BENCH_DATA_DIR": str(data_dir),
         "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": str(seed)}
    e.update(thread_env(cpus))
    e = venv_env(venv_inside, base=e)
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e


def _wrap(prefix, cmd):
    return [*(prefix or []), *cmd]


def train(code_dir, out_dir, env, cap_s, log_path, prefix=None, cwd=None):
    r = run(_wrap(prefix, ["bash", f"{code_dir}/train.sh", "--out", str(out_dir)]), cwd=cwd, env=None if prefix else env,
            timeout=cap_s, log_path=log_path)
    return r


def predict(code_dir, model_dir, input_path, output_path, env, cap_s, log_path, prefix=None, cwd=None):
    r = run(_wrap(prefix, ["bash", f"{code_dir}/predict.sh", "--model", str(model_dir), "--input", str(input_path),
                           "--output", str(output_path)]), cwd=cwd, env=None if prefix else env, timeout=cap_s, log_path=log_path)
    return r


def validate(validator_py, pred_path, input_path, python, env, timeout=None):
    """-> (ok, message)."""
    p = capture([python, str(validator_py), str(pred_path), str(input_path)], env=env, timeout=timeout)
    msg = (p.stdout + p.stderr).strip()[-4000:]
    return p.returncode == 0, msg


def score(scorer_py, pred_path, labels_path, python, env, timeout=None):
    """-> dict of metrics (raises RuntimeError on failure)."""
    p = capture([python, str(scorer_py), str(pred_path), str(labels_path)], env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"scorer failed rc={p.returncode}: {(p.stderr or p.stdout)[-2000:]}")
    text = p.stdout.strip().splitlines()
    for line in reversed(text):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    return d
            except ValueError:
                continue
    raise RuntimeError(f"scorer printed no JSON object: {p.stdout[-500:]!r}")


def one_seed(*, seed, code_dir, data_dir, out_root, input_path, labels_path, validator_py, scorer_py,
             exec_env, tool_env, tool_python, caps, prefix_train=None, prefix_predict=None, cwd=None,
             pred_name="pred.out", path_map=None, deadline_ts=None):
    """Full train -> predict -> validate -> score for one seed.

    exec_env is the env for train/predict (inside the venv); tool_env/tool_python run validator and scorer
    (judge: the trusted reference venv; eval-dev: the agent's own venv).
    path_map(host_path) -> path as the (possibly sandboxed) command sees it; identity by default.
    -> dict(status, metrics, timing, failure, pred_path)
    """
    pm = path_map or (lambda p: str(p))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir = out_root / "model"
    model_dir.mkdir(exist_ok=True)
    pred_path = out_root / pred_name
    timing = {}
    tool_env = deadline_env(tool_env, deadline_ts)
    if exec_env is not None:
        exec_env = deadline_env(exec_env, deadline_ts)

    def failed_budget(stage, error):
        return {"status": "budget_exhausted", "metrics": None, "timing": timing,
                "pred_path": str(pred_path) if pred_path.exists() else None,
                "failure": {"stage": stage, "reason": str(error), "log": None}}

    try:
        train_timeout = remaining_time(deadline_ts, caps.get("train"))
    except TimeoutError as exc:
        return failed_budget("train", exc)
    t = train(pm(code_dir), pm(model_dir), exec_env, train_timeout, out_root / "train.log", prefix=prefix_train, cwd=cwd)
    timing["train_s"] = t["elapsed_s"]
    if t["timed_out"]:
        if deadline_ts is not None and time.time() >= deadline_ts:
            return failed_budget("train", "overall evaluation budget exhausted")
        return {"status": "train_timeout", "metrics": None, "timing": timing, "pred_path": None,
                "failure": {"stage": "train", "reason": f"exceeded {caps['train']} s", "log": t["log"]}}
    if t["rc"] != 0:
        return {"status": "train_failed", "metrics": None, "timing": timing, "pred_path": None,
                "failure": {"stage": "train", "reason": f"train.sh rc={t['rc']}", "log": t["log"]}}
    try:
        predict_timeout = remaining_time(deadline_ts, caps.get("predict"))
    except TimeoutError as exc:
        return failed_budget("predict", exc)
    p = predict(pm(code_dir), pm(model_dir), pm(input_path), pm(pred_path), exec_env, predict_timeout,
                out_root / "predict.log", prefix=prefix_predict, cwd=cwd)
    timing["predict_s"] = p["elapsed_s"]
    if p["timed_out"] or p["rc"] != 0 or not pred_path.exists():
        if deadline_ts is not None and time.time() >= deadline_ts:
            return failed_budget("predict", "overall evaluation budget exhausted")
        why = f"exceeded {caps['predict']} s" if p["timed_out"] else (f"predict.sh rc={p['rc']}" if p["rc"] else "no output file")
        return {"status": "predict_failed", "metrics": None, "timing": timing, "pred_path": None,
                "failure": {"stage": "predict", "reason": why, "log": p["log"]}}
    try:
        ok, msg = validate(validator_py, pred_path, input_path, tool_python, tool_env,
                           timeout=remaining_time(deadline_ts, caps.get("validate")))
    except Exception as exc:
        if deadline_ts is not None and time.time() >= deadline_ts:
            return failed_budget("validate", "overall evaluation budget exhausted")
        return {"status": "format_invalid", "metrics": None, "timing": timing, "pred_path": str(pred_path),
                "failure": {"stage": "validate", "reason": str(exc)[-1000:], "log": None}}
    if not ok:
        return {"status": "format_invalid", "metrics": None, "timing": timing, "pred_path": str(pred_path),
                "failure": {"stage": "validate", "reason": msg[-1000:], "log": None}}
    t0 = time.time()
    try:
        metrics = score(scorer_py, pred_path, labels_path, tool_python, tool_env,
                        timeout=remaining_time(deadline_ts, caps.get("score")))
        remaining_time(deadline_ts)
    except Exception as e:  # noqa: BLE001
        if deadline_ts is not None and time.time() >= deadline_ts:
            return failed_budget("score", "overall evaluation budget exhausted")
        return {"status": "score_failed", "metrics": None, "timing": timing, "pred_path": str(pred_path),
                "failure": {"stage": "score", "reason": str(e)[-1000:], "log": None}}
    timing["score_s"] = round(time.time() - t0, 2)
    return {"status": "ok", "metrics": metrics, "timing": timing, "pred_path": str(pred_path), "failure": None}
