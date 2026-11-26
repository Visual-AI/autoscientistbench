"""bench-run: tasks/<pid>/agent -> closed workspace -> launch harness -> deadline -> final.patch -> runs/<run_id>/."""
import json
import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import LOCAL_ROOT, RUNS_DIR, UV_CACHE, ROOT
from .budget import positive_seconds
from .catalog import require_active_task
from .runtime import bind_runtime, build_environment, is_snapshot, runtime_env
from .evalrun import stage_env
from .harness import load_harness
from .harness.base import RunContext
from .patch import head_commit, is_clean, materialize_code_repo
from .proxy import Proxy, ProxyConfig
from .submission import SubmissionBroker
from .sandbox import (IN_BIN, IN_HOME, IN_RUN, IN_TMP, IN_UVCACHE, IN_VENV, IN_WS, Sandbox, python_binds_for_venv,
                      tool_paths, write_entry, write_launcher, write_profile)
from .task import load_task
from .util import (append_jsonl, ensure_dir, free_cpus, kill_group, kill_stragglers, log, now_iso, parse_cpuset, read_json,
                   read_jsonl, rm_rf, sha256_bytes, supervised_argv, ts_compact, write_json)

OPTIONAL_TOOLS = {"traex-code-mode-host"}
BRIDGE_PORT = 18080          # inside the sandbox's own netns, so every run can use the same port
PROMPT_TPL = ROOT / "bench" / "templates" / "prompt_wrapper.md"


def render_prompt(task, harness, budget_s, cpus, readme_text):
    tpl_path = harness.cfg_dir / "prompt_wrapper.md"
    tpl = (tpl_path if tpl_path.exists() else PROMPT_TPL).read_text(encoding="utf-8")
    m = task.metric
    bval = (task.d["baseline"].get("val") or {}).get(m["name"])
    subs = {
        "pid": task.pid, "budget_h": f"{budget_s / 3600:g}", "cpus": str(cpus), "metric": str(m["name"]),
        "direction": "higher is better" if task.higher_is_better else "lower is better",
        "baseline_val": f"{bval:.6g}" if isinstance(bval, (int, float)) else "see README",
        "delta_pct": f"{100 * float(m.get('delta', 0.005)):g}",
        "readme": readme_text,
    }
    for k, v in subs.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    # A harness may append its own closing instruction. The shared brief ends with the task README and then the
    # data-use rules, i.e. it never issues a direct request -- which GLM-5.3-Flash reads as a session preamble:
    # in 278 runs it explored the workspace and then replied "I don't see an actual question or task in your
    # message yet. Could you let me know what you'd like to do?" and the headless session closed as a success
    # (measured 2026-11-10, see harnesses/zcode-glm53local/config.json). Every other model treats the same brief
    # as its task, so this stays opt-in per harness rather than changing the brief for everyone.
    suffix = str(harness.cfg.get("prompt_suffix") or "").strip()
    if suffix:
        tpl = tpl.rstrip("\n") + "\n\n---\n\n" + suffix + "\n"
    return tpl


def short_sock_dir(run_id):
    """AF_UNIX paths are limited to 108 bytes; keep the socket somewhere short."""
    import hashlib
    d = Path("/tmp/arb-s") / hashlib.sha1(run_id.encode()).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_final(sub_dir, deadline_ts=None, *, prefer_final=True):
    """Select an intact valid submission; optionally use strictly the latest one."""
    subs = sorted([read_json(p) for p in Path(sub_dir).glob("*.json")], key=lambda s: s.get("n", 0))
    def intact(s):
        if type(s.get("n")) is not int or s["n"] < 1:
            return False
        path = Path(sub_dir) / f"{s['n']:03d}.patch"
        if not path.is_file():
            return False
        patch = path.read_bytes()
        return sha256_bytes(patch) == s.get("patch_sha256") and len(patch) == s.get("bytes")
    oks = [s for s in subs if s.get("ok") and s.get("source", "submit") == "submit" and
           (deadline_ts is None or isinstance(s.get("ts"), (int, float)) and s["ts"] <= deadline_ts) and intact(s)]
    finals = [s for s in oks if s.get("final")]
    chosen = finals[-1] if prefer_final and finals else (oks[-1] if oks else None)
    return chosen, subs


def bench_run(pid, harness_name, budget_hours, *, cpus=None, cpuset=None, repeat=1, suite="adhoc", keep=False,
              overrides=None, model=None, grace_s=300, memory_gb=None, run_id=None, uv_cache=None, local_root=None,
              dry=False, judge_budget_hours=None, resume=False):
    run_id = run_id or f"{suite}__{load_task(pid).pid}__{harness_name}__r{repeat}__{ts_compact()}"
    # This NAS mount does not implement flock; runner ownership is pod-local.
    locks = Path('/tmp/arb-run-locks')
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (hashlib.sha256((str(RUNS_DIR.resolve())+'/'+run_id).encode()).hexdigest() + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _bench_run(pid, harness_name, budget_hours, cpus=cpus, cpuset=cpuset, repeat=repeat,
                          suite=suite, keep=keep, overrides=overrides, model=model, grace_s=grace_s,
                          memory_gb=memory_gb, run_id=run_id, uv_cache=uv_cache, local_root=local_root,
                          dry=dry, judge_budget_hours=judge_budget_hours, resume=resume)


def _bench_run(pid, harness_name, budget_hours, *, cpus=None, cpuset=None, repeat=1, suite="adhoc", keep=False,
               overrides=None, model=None, grace_s=300, memory_gb=None, run_id=None, uv_cache=None, local_root=None,
               dry=False, judge_budget_hours=None, resume=False):
    task = load_task(pid)
    require_active_task(task.pid)
    probs = task.problems()
    if probs:
        raise RuntimeError(f"task {task.pid} is not runnable: " + "; ".join(probs))
    harness = load_harness(harness_name, overrides)
    selection = harness.cfg.get('submission_selection', 'prefer_final')
    if selection not in ('prefer_final', 'last_valid'):
        raise ValueError('Unknown submission_selection: ' + str(selection))
    cpus = int(cpus or task.cpus)
    if cpuset:
        cpuset = parse_cpuset(cpuset)
    else:
        cpuset = free_cpus()[:cpus]
    budget_s = positive_seconds(float(budget_hours) * 3600)
    judge_budget_s = budget_s if judge_budget_hours is None else positive_seconds(float(judge_budget_hours) * 3600)
    run_id = run_id or f"{suite}__{task.pid}__{harness_name}__r{repeat}__{ts_compact()}"
    local = Path(local_root or LOCAL_ROOT) / run_id
    rec = RUNS_DIR / run_id
    resumable = bool(harness.cfg.get('resumable'))
    previous = read_json(rec/'run.json') if resume else None
    if resume:
        if not resumable or previous.get('status') not in ('paused', 'running', 'interrupted'):
            raise ValueError('Resume requires an unfinished resumable run')
        if (previous['task'] != task.pid or previous['harness']['name'] != harness_name or
                previous['budget_s'] != budget_s or previous.get('requested_model') != model):
            raise ValueError('Resume must preserve task, harness, model and original time budget')
        if not (local/'ws/code').is_dir() or not (local/'venv').is_dir():
            raise RuntimeError('Local resume workspace/environment missing; restore from checkpoint before resuming')
        used_s = float(previous.get('active_elapsed_s', previous.get('elapsed_s', 0)))
        if previous['status'] == 'running':
            # A killed runner can lose at most one heartbeat interval. Conservatively charge it.
            used_s += 5
        if used_s >= budget_s:
            raise ValueError('The original agent time budget is exhausted; no new budget is granted on resume')
    elif local.exists() or rec.exists():
        raise FileExistsError(f"run {run_id} already exists")
    else:
        used_s = 0.0
    for d in ("ws", "run/submissions", "run/dev", "home", "tmp", "uvcache", "sandbox"):
        ensure_dir(local / d)
    ensure_dir(rec)
    durable = Path(os.environ.get('ARB_DURABLE_RUNS_DIR', RUNS_DIR)) / run_id
    if harness.cfg.get('record_solution_versions') and durable.resolve() != rec.resolve() and not (rec/'submissions').exists():
        ensure_dir(durable / 'submissions')
        (rec / 'submissions').symlink_to(durable / 'submissions', target_is_directory=True)
    else:
        ensure_dir(rec / "submissions")
    uv_cache = Path(uv_cache or UV_CACHE)
    meta = previous or {"run_id": run_id, "task": task.pid, "harness": harness.describe(), "suite": suite, "repeat": repeat,
            "budget_s": budget_s, "judge_budget_s": judge_budget_s, "cpus": cpus, "cpuset": cpuset, "memory_gb": memory_gb or task.d["memory_gb"],
            "created": now_iso(), "status": "materializing", "local_dir": str(local)}
    meta.update(resumable=resumable, requested_model=model, runner_pid=os.getpid(),
                active_elapsed_s=used_s, remaining_s=max(0, budget_s-used_s))
    if resume:
        meta['status'] = 'resuming'
        (rec/'PAUSE.json').unlink(missing_ok=True)
    write_json(rec / "run.json", meta)
    log(f"[{run_id}] materialize")

    # 1. workspace ----------------------------------------------------------------
    ws = local / "ws"
    has_git = (task.code_dir / ".git").exists()
    if has_git:
        head = head_commit(task.code_dir)
        if not head.startswith(task.adapt_commit):
            raise RuntimeError(f"agent/code HEAD {head[:12]} != adapt_commit {task.adapt_commit[:12]}; run `bench adapt`")
        if not is_clean(task.code_dir):
            raise RuntimeError("agent/code working tree is not clean; commit it with `bench adapt` first")
        base_commit = task.adapt_commit
        if not resume:
            shutil.copytree(task.code_dir, ws / "code", symlinks=True)
            shutil.copy2(task.agent_dir / "README.md", ws / "README.md")
            if (task.agent_dir / "paper").exists():
                shutil.copytree(task.agent_dir / "paper", ws / "paper", symlinks=True)
    else:
        if not resume:
            base_commit = materialize_code_repo(task.code_dir, ws / "code")
            shutil.copy2(task.agent_dir / "README.md", ws / "README.md")
            if (task.agent_dir / "paper").exists():
                shutil.copytree(task.agent_dir / "paper", ws / "paper", symlinks=True)
        else:
            saved_state = read_json(rec / "state.json", default={}) or {}
            base_commit = saved_state.get("adapt_commit") or head_commit(ws / "code")
    (ws / "data").mkdir(exist_ok=True)
    (ws / "bench").mkdir(exist_ok=True)

    # 2. venv (trusted cache, operator side) --------------------------------------------
    log(f"[{run_id}] build venv from code/env.lock")
    b = ({'ok':True, 'elapsed_s':meta['timing']['build_s']} if resume else
         build_environment(task, local / "venv", ws / "code" / "env.lock", task.indexes, uv_cache, rec / "build.log",
                           cap_s=task.build_cap_s))
    if not b["ok"]:
        meta.update(status="infra_error", failure={"stage": "build", "reason": f"baseline venv build failed ({b['stage']})",
                                                   "log": str(rec / "build.log")})
        write_json(rec / "run.json", meta)
        raise RuntimeError("baseline venv build failed; see " + str(rec / "build.log"))
    meta["timing"] = {"build_s": b["elapsed_s"]}

    # 3. prompt, state, proxy, home -----------------------------------------------------
    readme = (ws / "README.md").read_text(encoding="utf-8")
    prompt = render_prompt(task, harness, budget_s, cpus, readme)
    if not resume:
        (rec / "prompt.md").write_text(prompt, encoding="utf-8")
    io = task.d["io"]
    state = {
        "run_id": run_id, "pid": task.pid, "budget_s": budget_s, "deadline_ts": None, "cpus": cpus,
        "memory_gb": meta["memory_gb"], "ws": IN_WS, "code": f"{IN_WS}/code", "data": f"{IN_WS}/data",
        "bench": f"{IN_WS}/bench", "venv": IN_VENV, "adapt_commit": base_commit, "metric": task.metric,
        "baseline_val": task.d["baseline"].get("val") or {},
        "io": {"validator": f"{IN_WS}/{io['validator']}", "scorer": f"{IN_WS}/{io['scorer']}",
               "val_input": f"{IN_WS}/{io['val_input']}", "val_labels": f"{IN_WS}/{io['val_labels']}"},
        "caps": {"train_s": task.train_cap_s, "predict_s": task.predict_cap_s, "score_s": task.score_cap_s},
        "patch_limits": task.patch_limits,
        "env_mode": task.d["env"].get("mode", "pip"),
        "runtime_env": runtime_env(task),
        "snapshot_lock": (task.code_dir / "env.lock").read_text() if is_snapshot(task) else None,
        "submission_socket": f"{IN_RUN}/submit.sock",
    }
    sock_dir = short_sock_dir(run_id)
    if resume:
        # The pod-local runner lock is held; a previous crashed broker may leave
        # its inode behind even though its process no longer owns the socket.
        (sock_dir/'submit.sock').unlink(missing_ok=True)
    sock = sock_dir / "proxy.sock"
    allow = task.index_hosts() + harness.allow_hosts()
    proxy = Proxy(ProxyConfig(allow, harness.model_upstream(), trace_dir=rec / "trace", log_path=rec / "proxy.log"),
                  unix_path=str(sock)).start()
    ctx = RunContext(run_id=run_id, pid=task.pid, budget_s=budget_s, deadline_ts=None, cpus=cpus, port=BRIDGE_PORT,
                     ws_inside=IN_WS, home_inside=IN_HOME, run_dir_inside=IN_RUN, prompt_text=prompt,
                     last_message_inside=f"{IN_RUN}/last_message.txt", model=model)
    ctx.resuming = resume
    if resume:
        ctx.prompt_text = (f'Continue the same benchmark task from the saved session and workspace. '
                           f'The interruption was operational. You have {int(budget_s-used_s)} seconds of the original '
                           'agent budget remaining; check BENCH_DEADLINE_TS for the updated deadline. '
                           'Previous background processes were stopped; check their files before restarting work. '
                           'All original task rules still apply. Submit the final solution using bench submit.')
    harness.prepare_home(local / "home", ctx)
    write_launcher(local / "run" / "bench")
    write_entry(local / "run" / "entry.sh", BRIDGE_PORT, f"{IN_RUN}/proxy.sock", f"{IN_RUN}/bridge.ready")
    (local / "run" / "events.jsonl").touch()

    # 4. sandbox -------------------------------------------------------------------------
    sb = Sandbox(cpuset=cpuset, unshare_net=True, own_uid=True)
    sb.rw_bind(ws, IN_WS).ro_bind(task.data_dir, f"{IN_WS}/data").ro_bind(task.bench_dir, f"{IN_WS}/bench")
    sb.rw_bind(local / "venv", IN_VENV)
    for d in ([] if is_snapshot(task) else python_binds_for_venv(local / "venv")):
        sb.ro_bind(d, d)
    sb.rw_bind(local / "home", IN_HOME).rw_bind(local / "tmp", IN_TMP).rw_bind(local / "uvcache", IN_UVCACHE)
    bind_runtime(sb, task)
    sb.rw_bind(local / "run", IN_RUN).ro_file(sock, f"{IN_RUN}/proxy.sock")
    sb.ro_bind(rec / "submissions", f"{IN_RUN}/submissions")
    requested_tools = ([] if is_snapshot(task) else ["uv"]) + list(harness.tools())
    tools = tool_paths(requested_tools)
    missing = [t for t in requested_tools if t not in tools and t not in OPTIONAL_TOOLS]
    if missing:
        proxy.stop()
        raise RuntimeError(f"tool binaries not found on this host: {missing}")
    sb.add_tools(tools, launcher=local / "run" / "bench")
    # login shells (`bash -lc`, codex's default) must keep our PATH: replace /etc/profile and seed HOME dotfiles
    write_profile(local / "run" / "profile.sh")
    sb.ro_file(local / "run" / "profile.sh", "/etc/profile")
    for dotfile in (".profile", ".bash_profile", ".bashrc"):
        (local / "home" / dotfile).write_text('[ -n "$BENCH_PATH" ] && export PATH="$BENCH_PATH"\n')
    proxy_url = f"http://127.0.0.1:{BRIDGE_PORT}"
    env = stage_env(IN_VENV, cpus, 0, f"{IN_WS}/data", IN_HOME, IN_TMP, extra={
        "BENCH_RUN_DIR": IN_RUN, "BENCH_BENCH_DIR": f"{IN_WS}/bench", "BENCH_CODE_DIR": f"{IN_WS}/code", "PATH": f"{IN_VENV}/bin:{IN_BIN}:/usr/local/bin:/usr/bin:/bin",
        "BENCH_PATH": f"{IN_VENV}/bin:{IN_BIN}:/usr/local/bin:/usr/bin:/bin",
        "HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url, "http_proxy": proxy_url, "https_proxy": proxy_url,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        "UV_CACHE_DIR": IN_UVCACHE, "UV_PYTHON_DOWNLOADS": "never", "UV_NO_PROGRESS": "1",
        "UV_INDEX_STRATEGY": "unsafe-best-match", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "TERM": "dumb", "USER": "agent", "LOGNAME": "agent", "SHELL": "/bin/bash", "BENCH_TASK": task.pid,
    })
    if task.indexes:
        env["UV_INDEX_URL"] = env["PIP_INDEX_URL"] = task.indexes[0]
        if len(task.indexes) > 1:
            env["UV_EXTRA_INDEX_URL"] = env["PIP_EXTRA_INDEX_URL"] = " ".join(task.indexes[1:])
    env.update(harness.env(ctx))
    if is_snapshot(task):
        env.update(runtime_env(task))
        env["PATH"] = f"{IN_VENV}/bin:{IN_BIN}:" + env["PATH"]
        env["BENCH_PATH"] = env["PATH"]
    if dry:
        write_json(local / "run" / "state.json", state)
        sb.ro_file(local / "run" / "state.json", f"{IN_RUN}/state.json")
        sb.setenv(**env)
        inner = ["/bin/sh", f"{IN_RUN}/entry.sh", *harness.command(ctx)]
        argv = sb.argv(inner, local / "sandbox")
        (rec / "sandbox_argv.json").write_text(json.dumps(argv, indent=1))
        proxy.stop()
        log(f"[{run_id}] dry run: sandbox argv written, not launching")
        return meta

    recorder = None
    if harness.cfg.get('record_solution_versions'):
        from .artifacts import SolutionRecorder
        recorder = SolutionRecorder(ws, local/'home/.kimi-code/sessions', durable).start()
        meta['solution_archive'] = str(durable/'solutions')
    meta['submission_selection'] = selection
    # 5. launch + watchdog ---------------------------------------------------------------
    started = time.time()
    deadline = started + budget_s - used_s
    state["deadline_ts"] = deadline
    ctx.deadline_ts = deadline
    env["BENCH_DEADLINE_TS"] = str(deadline)
    write_json(local / "run" / "state.json", state)
    write_json(rec / "state.json", state)
    sb.ro_file(local / "run" / "state.json", f"{IN_RUN}/state.json")
    broker = SubmissionBroker(sock_dir / "submit.sock", rec / "submissions", deadline, task.patch_limits).start()
    sb.ro_file(sock_dir / "submit.sock", f"{IN_RUN}/submit.sock").own_file(sock_dir / "submit.sock")   # the broker made it 0600
    sb.setenv(**env)
    inner = ["/bin/sh", f"{IN_RUN}/entry.sh", *harness.command(ctx)]
    argv = sb.argv(inner, local / "sandbox")
    (rec / "sandbox_argv.json").write_text(json.dumps(argv, indent=1))
    attempt = {'number':len(meta.get('attempts', []))+1, 'started_ts':started,
               'started_utc':now_iso(), 'remaining_at_start_s':budget_s-used_s, 'resume':resume}
    meta.setdefault('attempts', []).append(attempt)
    meta.setdefault('first_started_ts', started)
    if resume:
        for name in ('ended','ended_reason','pause','rc'):
            meta.pop(name,None)
    meta.update(status="running", started=meta.get('started', now_iso()), deadline_ts=deadline,
                last_attempt_started_ts=started)
    write_json(rec / "run.json", meta)
    log(f"[{run_id}] launch {harness_name} (budget {budget_s}s, cpus {cpuset})")
    out_f = open(rec / "agent_events.jsonl", "a" if resume else "w", encoding="utf-8")
    err_f = open(rec / "harness.log", "a" if resume else "w", encoding="utf-8")
    stdin_text = harness.stdin(ctx)
    if resumable and hasattr(harness, 'begin_attempt'):
        harness.begin_attempt(rec)
    try:
        p = subprocess.Popen(supervised_argv(argv), stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                             stdout=out_f, stderr=err_f, start_new_session=True, cwd=str(local))
    except BaseException:
        broker.stop()
        proxy.stop()
        out_f.close()
        err_f.close()
        raise
    if stdin_text is not None:
        def feed():
            try:
                p.stdin.write(stdin_text.encode("utf-8"))
                p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=feed, daemon=True).start()
    ended_reason = "self"
    pause = {}
    saved_handlers = {}
    if resumable and threading.current_thread() is threading.main_thread():
        def request_pause(signum, _frame):
            pause.update(kind='operator_signal', signal=signum)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            saved_handlers[signum] = signal.signal(signum, request_pause)
    last_beat = 0
    while True:
        if resumable:
            if (rec/'PAUSE.json').exists():
                pause = read_json(rec/'PAUSE.json')
            if not pause and hasattr(harness, 'pause_reason'):
                try:pause = harness.pause_reason(rec, started) or {}
                except Exception as exc:pause = {'kind':'observer_error','error_type':type(exc).__name__,'message':str(exc)[:400]}
            if not pause and recorder is not None and recorder.errors:
                pause = {'kind':'artifact_recording_error','errors':recorder.errors[-3:]}
            if pause:
                ended_reason = 'paused'
                kill_group(p, grace=5)
                break
        rc = p.poll()
        if rc is not None:
            break
        now = time.time()
        if now >= deadline:
            ended_reason = "deadline"
            log(f"[{run_id}] deadline reached: SIGTERM group, {grace_s}s grace")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                kill_group(p, grace=10)
            break
        if now - last_beat > (5 if resumable else 60):
            meta["heartbeat"] = now_iso()
            meta["remaining_s"] = int(deadline - now)
            meta['active_elapsed_s'] = min(budget_s, used_s + now-started)
            write_json(rec / "run.json", meta)
            last_beat = now
        time.sleep(2)
    rc = p.poll()
    for signum, handler in saved_handlers.items():
        signal.signal(signum, handler)
    try:
        os.killpg(p.pid, signal.SIGKILL)   # saved PGID survives leader exit
    except OSError:
        pass
    n_killed = kill_stragglers(local)
    broker.stop()
    out_f.close()
    err_f.close()
    ended = time.time()
    if ended_reason == "self" and rc not in (0, None):
        ended_reason = "crash"
        if resumable and ended-started < budget_s-used_s:
            pause = {'kind':'process_interrupted', 'exit_code':rc}
            ended_reason = 'paused'
    active_s = min(budget_s, used_s + ended-started)
    if ended_reason == 'paused' and active_s >= budget_s:
        ended_reason = 'deadline'
    attempt.update(ended_ts=ended, ended_utc=now_iso(), duration_s=ended-started, reason=ended_reason,
                   exit_code=rc, pause=pause or None)
    log(f"[{run_id}] harness ended rc={rc} reason={ended_reason} after {ended - started:.0f}s (stragglers killed: {n_killed})")

    if recorder is not None:
        meta['solution_recording'] = recorder.stop()
    try:
        harness.after_run(local / "home", ctx)
    except Exception as e:  # noqa: BLE001
        log(f"[{run_id}] after_run hook failed: {e}")

    # 6. final patch ---------------------------------------------------------------------
    chosen, subs = pick_final(rec / "submissions", deadline, prefer_final=selection != 'last_valid')
    if chosen:
        pb = (rec / "submissions" / f"{chosen['n']:03d}.patch").read_bytes()
        final = dict(chosen, source="final" if chosen.get("final") else "last_submission")
    else:
        final = {"source": "no_submission", "ok": False, "problems": ["no valid submission before the deadline"],
                 "patch_sha256": None, "bytes": 0, "files": 0}
        pb = b""
    if ended_reason != 'paused':
        (rec / "final.patch").write_bytes(pb)
    final['selection_policy'] = selection
    final['selected_submission_n'] = chosen.get('n') if chosen else None
    write_json(rec / ("checkpoint_submission.json" if ended_reason == 'paused' else "final.json"), final)

    # 7. collect + record -----------------------------------------------------------------
    for rel in ("events.jsonl", "last_message.txt", "state.json"):
        src = local / "run" / rel
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, rec / rel, dirs_exist_ok=True)
            else:
                shutil.copy2(src, rec / rel)
    for d in sorted((local / "run" / "dev").glob("*")):
        dst = rec / "dev" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("train.log", "predict.log"):
            if (d / f).exists():
                shutil.copy2(d / f, dst / f)
    proxy.stop()
    evs = read_jsonl(local / "run" / "events.jsonl")
    dev_ok = [e for e in evs if e.get("kind") == "eval-dev" and e.get("status") == "ok"]
    first_impr = None
    for e in dev_ok:
        g = e.get("gain_vs_baseline_val")
        if g is not None and g > 0:
            first_impr = int(e["ts"] - started)
            break
    meta.update({
        "status": "paused" if ended_reason == 'paused' else ("finished" if chosen else "no_submission"),
        "ended": now_iso(), "ended_reason": ended_reason, "rc": rc,
        "elapsed_s": int(active_s), "active_elapsed_s":active_s, "remaining_s":max(0, budget_s-active_s),
        "wall_elapsed_s":ended-meta['first_started_ts'], "pause":pause or None, "stragglers_killed": n_killed,
        "patch": {"sha256": final.get("patch_sha256"), "bytes": final.get("bytes"), "files": final.get("files"),
                  "source": final.get("source"), "ok": final.get("ok"), "problems": final.get("problems")},
        "submissions": len(subs), "eval_dev_calls": len([e for e in evs if e.get("kind") == "eval-dev"]),
        "first_improvement_s": first_impr, "agent": harness.collect(rec), "proxy": proxy.stats,
    })
    meta["timing"]["agent_s"] = int(active_s)
    write_json(rec / "run.json", meta)
    if ended_reason == 'paused':
        checkpoint = rec/'checkpoint'
        checkpoint.mkdir(exist_ok=True)
        try:
            result = subprocess.run(['rsync','-a','--delete','--exclude=.git','--exclude=/data',
                                     '--exclude=/bench','--exclude=/paper',str(ws)+'/',str(checkpoint/'workspace')+'/'],
                                    capture_output=True, timeout=300)
            copied = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            copied = False
        meta['checkpoint'] = {'saved_utc':now_iso(), 'workspace_copy_ok':copied,
                              'workspace':str(checkpoint/'workspace'), 'local_workspace':str(ws),
                              'local_environment':str(local/'venv'), 'native_sessions':str(rec/'trace/kimi/sessions'),
                              'restore_scope':'Exact local workspace/venv/session continuation; NAS workspace and session backup for recovery'}
        write_json(checkpoint/'state.json', {**meta['checkpoint'], 'active_elapsed_s':active_s,
                                          'remaining_s':max(0,budget_s-active_s), 'attempt':attempt})
    write_json(rec / "run.json", meta)
    if not keep and ended_reason != 'paused' and (recorder is None or not meta['solution_recording']['errors']):
        rm_rf(local)
        rm_rf(sock_dir)
    log(f"[{run_id}] recorded at {rec}")
    return meta
