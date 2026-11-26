"""A small functional conversion check, explicitly not a measured reference."""
import json
import tempfile
import time
from pathlib import Path

from . import LOCAL_ROOT, UV_CACHE
from .evalrun import one_seed, validate
from .judge import DirectStage, Stage, tool_ctx
from .patch import materialize_code_repo
from .runtime import build_environment
from .task import load_task
from .util import free_cpus, now_iso, sha256_file, sha256_tree, write_json


def smoke(pid, *, input_path=None, labels_path=None, cpus=2, cap_s=300, sandbox=True, seed=1):
    task = load_task(pid)
    spec = task.d.get("smoke", {})
    if input_path:
        input_path = Path(input_path)
    else:
        cand_in = task.dir / spec["input"] if spec.get("input") else None
        input_path = cand_in if (cand_in and cand_in.exists()) else (task.dir / ("agent/" + task.d["io"]["val_input"]))
    if labels_path:
        labels_path = Path(labels_path)
    else:
        cand_lbl = task.dir / spec["labels"] if spec.get("labels") else None
        labels_path = cand_lbl if (cand_lbl and cand_lbl.exists()) else (task.dir / ("agent/" + task.d["io"]["val_labels"]))
    input_path, labels_path = input_path.resolve(), labels_path.resolve()
    if not input_path.exists() or not labels_path.exists():
        raise ValueError("smoke requires existing small input and matching labels")
    local_root = LOCAL_ROOT / "smoke"
    local_root.mkdir(parents=True, exist_ok=True)
    local = Path(tempfile.mkdtemp(prefix=task.pid + "-", dir=local_root))
    local.chmod(0o755)   # the sandbox runs as a uid of its own and must reach its binds
    record = task.judge_dir / "verification" / "smoke"
    record.mkdir(parents=True, exist_ok=True)
    result = {"task": task.pid, "kind": "smoke_only", "full_reference": False, "sandbox": bool(sandbox),
              "created": now_iso(), "input": str(input_path), "labels": str(labels_path),
              "local_dir": str(local), "cpus": cpus, "seed": seed, "cap_s": cap_s}
    result["adapt_commit"] = task.adapt_commit
    result["bench_sha256"] = sha256_tree(task.judge_bench_dir)
    result["input_sha256"] = sha256_file(input_path) if input_path.is_file() else sha256_tree(input_path)
    result["labels_sha256"] = sha256_file(labels_path) if labels_path.is_file() else sha256_tree(labels_path)
    if task.d["env"].get("mode") == "snapshot":
        result["runtime_manifest_sha256"] = sha256_file(task.dir / task.d["env"].get("snapshot", "runtime/runtime.json"))
    started = time.monotonic()
    try:
        probs = task.problems(deep=True)
        if probs:
            raise ValueError("; ".join(probs))
        code = local / "code"
        materialize_code_repo(task.code_dir, code, task.adapt_commit)
        build = build_environment(task, local / "venv", code / "env.lock", task.indexes, UV_CACHE,
                                  record / "build.log", cap_s=task.build_cap_s)
        result["environment"] = build
        if not build["ok"]:
            raise RuntimeError("environment preparation failed: " + str(record / "build.log"))
        (local / "tmp").mkdir()
        cls = Stage if sandbox else DirectStage
        stage = cls(task, code, local / "venv", local / "tmp", local / "sandbox", free_cpus()[:cpus], cpus)
        tool_env, tool_py = tool_ctx(local / "venv", cpus, task)
        out = local / "out"
        out.mkdir()
        # Smoke flags affect only this explicit check, never the default run/judge.
        extra = {"BENCH_SMOKE": "1", **{k: str(v) for k, v in spec.get("env", {}).items()}}
        if sandbox:
            def prefix(input_=None):
                argv = stage.prefix(seed, out, input_)
                index = len(argv) - 1 - argv[::-1].index("--")   # bwrap's own "--"; a sandbox_user.py prefix has one too
                flags = [x for key, value in extra.items() for x in ("--setenv", key, value)]
                return argv[:index] + flags + argv[index:]
            train_prefix, predict_prefix = prefix(), prefix(input_path)
            exec_env = None
        else:
            train_prefix = predict_prefix = None
            exec_env = stage.exec_env(seed)
            exec_env.update(extra)
        check = one_seed(seed=seed, code_dir=code, data_dir=task.data_dir, out_root=out,
                         input_path=input_path, labels_path=labels_path,
                         validator_py=task.judge_bench_dir / "validate.py", scorer_py=task.judge_bench_dir / "scorer.py",
                         exec_env=exec_env, tool_env=tool_env, tool_python=tool_py,
                         caps={"train": cap_s, "predict": cap_s, "score": cap_s}, prefix_train=train_prefix, prefix_predict=predict_prefix,
                         path_map=stage.path_map(out, input_path), cwd=str(code) if not sandbox else None)
        result.update(check)
        import shutil
        for name in ("train.log", "predict.log", "pred.out"):
            if (out / name).is_file():
                shutil.copy2(out / name, record / name)
        if check["status"] == "ok":
            invalid = local / "invalid.out"
            invalid.write_text("this is deliberately not a prediction\n")
            accepted, message = validate(task.judge_bench_dir / "validate.py", invalid, input_path, tool_py, tool_env)
            result["invalid_prediction_rejected"] = not accepted
            if accepted:
                result.update(status="validator_failed", failure={"stage": "validate", "reason": "accepted invalid prediction"})
    except Exception as exc:
        result.update(status="smoke_failed", failure={"reason": str(exc)})
    result["elapsed_s"] = round(time.monotonic() - started, 2)
    write_json(task.judge_dir / "verification" / "smoke.json", result)
    return result
