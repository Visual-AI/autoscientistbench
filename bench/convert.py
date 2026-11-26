"""Conversion tools: `bench init` (skeleton), `bench adapt` (freeze our protocol edits as the bench-adapt commit),
`bench lint` (the acceptance checklist in PROTOCOL.md)."""
import os
import re
import shutil
from pathlib import Path

from . import LOCAL_ROOT, ROOT, TASKS_DIR, UV_CACHE
from .envlock import build_venv, validate_lock_text, venv_env
from .runtime import build_environment, is_snapshot, load_snapshot
from .evalrun import score as run_scorer, validate as run_validate
from .patch import git, head_commit, is_clean
from .task import load_task
from .util import ensure_dir, log, read_json, rm_rf, sha256_file, sha256_tree, write_json

TPL = ROOT / "bench" / "templates" / "task"
GIT_ENV = {"GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@localhost", "GIT_COMMITTER_NAME": "bench",
           "GIT_COMMITTER_EMAIL": "bench@localhost"}
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv", ".rst", ".ipynb"}


def init(pid, title="", metric="mse", higher=False, python="3.11", cpus=4, force=False, split="std3"):
    d = TASKS_DIR / pid
    if d.exists():
        leftovers = [p.name for p in d.iterdir() if p.name not in (".convert", "BRIEF.md")]
        if leftovers and not force:
            raise FileExistsError(f"{d} already has {leftovers[:5]}")
        if force:
            rm_rf(d)
    for rel in ("agent/code", "agent/data/train", "agent/data/val", "agent/bench", "agent/paper", "judge/test", "judge/reference", "judge/bench"):
        ensure_dir(d / rel)
    subs = {"pid": pid, "title": title or pid, "metric": metric, "higher": "true" if higher else "false",
            "python": python, "cpus": str(cpus), "direction": "higher is better" if higher else "lower is better",
            "split": split}

    def render(src, dst):
        t = (TPL / src).read_text(encoding="utf-8")
        for k, v in subs.items():
            t = t.replace("{{" + k + "}}", v)
        dst.write_text(t, encoding="utf-8")
        if src.endswith((".sh", ".py")):
            dst.chmod(0o755)
    render("task.json", d / "task.json")
    render("README.md", d / "agent" / "README.md")
    render("io_spec.md", d / "agent" / "bench" / "io_spec.md")
    render("scorer.py", d / "agent" / "bench" / "scorer.py")
    render("validate.py", d / "agent" / "bench" / "validate.py")
    render("train.sh", d / "agent" / "code" / "train.sh")
    render("predict.sh", d / "agent" / "code" / "predict.sh")
    (d / "agent" / "code" / "env.lock").write_text("# one `name==version` per line (pip-freeze format); rebuilt by uv pip sync\n")
    (d / "agent" / "code" / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\ncheckpoints/\n*.pt\n*.pth\n*.ckpt\n")
    log(f"init: skeleton at {d}")
    return d


SNAPSHOT_EXCLUDES = (".venv", "venv", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
                     ".ipynb_checkpoints", "*.egg-info", ".DS_Store")
TEMPLATE_FILES = ("train.sh", "predict.sh", "env.lock", ".gitignore")


def snapshot(pid, src, excludes=()):
    """Copy the official code snapshot into agent/code and make it the git base: `bench snapshot <pid> --from DIR`.

    Runs once, right after `bench init` and BEFORE any protocol edit: the template files init put into agent/code are set
    aside, DIR is copied (keeping its .git if it has one, so base_commit can be the upstream commit), a 'baseline snapshot'
    commit is made when there is no history, and the template files are put back uncommitted so that `bench adapt`
    records them -- and every later protocol edit -- in adapt.patch. Files matching SNAPSHOT_EXCLUDES are never copied.
    """
    task = load_task(pid)
    code, src = task.code_dir, Path(src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if (code / ".git").exists():
        raise RuntimeError(f"{code} already has a git history; snapshot must run once, before adapt")
    aside = {}
    for n in TEMPLATE_FILES:
        p = code / n
        if p.exists():
            aside[n] = p.read_bytes()
            p.unlink()
    leftovers = sorted(p.name for p in code.iterdir())
    if leftovers:
        raise RuntimeError(f"agent/code must be empty apart from the template files; found {leftovers[:8]}")
    shutil.copytree(src, code, symlinks=True, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*SNAPSHOT_EXCLUDES, *excludes))
    env = dict(os.environ, **GIT_ENV)
    if (code / ".git").exists():
        dirty = git(["status", "--porcelain"], code).stdout.strip()
        base = head_commit(code)
        note = f"upstream history kept, HEAD {base[:12]}" + ("; working tree has local modifications (they go into adapt.patch)" if dirty else "")
    else:
        git(["init", "-q"], code, env=env)
        git(["add", "-A"], code, env=env)
        git(["commit", "-q", "--allow-empty", "-m", f"baseline snapshot (from {src})"], code, env=env)
        base = head_commit(code)
        note = f"no upstream .git; snapshot commit {base[:12]}"
    for n, b in aside.items():
        if not (code / n).exists():
            (code / n).write_bytes(b)
            if n.endswith(".sh"):
                (code / n).chmod(0o755)
    tj = read_json(task.dir / "task.json")
    tj.setdefault("adapt", {})["base_commit"] = base
    tj["adapt"]["source"] = str(src)
    write_json(task.dir / "task.json", tj)
    log(f"snapshot {pid}: {note}; template files restored uncommitted: {sorted(aside)}")
    return base


def adapt(pid, message="bench-adapt: protocol-only changes (data entry, model selection, save/predict, seeds)", allow_fresh=False):
    task = load_task(pid)
    code = task.code_dir
    env = dict(os.environ, **GIT_ENV)
    if not (code / ".git").exists():
        if not allow_fresh:
            raise RuntimeError(f"{code} has no git history: run `bench snapshot {pid} --from <official code dir>` first "
                               "(then edit, then adapt); `--allow-fresh` commits everything as the base, leaving adapt.patch empty")
        git(["init", "-q"], code, env=env)
        git(["add", "-A"], code, env=env)
        git(["commit", "-q", "-m", "baseline snapshot"], code, env=env)
        base = adapt_commit = head_commit(code)
    else:
        before = head_commit(code)
        base = task.base_commit
        if not is_clean(code):
            git(["add", "-A"], code, env=env)
            git(["commit", "-q", "-m", message], code, env=env)
            adapt_commit = head_commit(code)
            base = base or before
        else:
            adapt_commit = before
            if not base:
                msg = git(["log", "-1", "--format=%s"], code).stdout.strip()
                parents = git(["rev-list", "--parents", "-n", "1", "HEAD"], code).stdout.split()
                base = parents[1] if msg.startswith("bench-adapt") and len(parents) > 1 else before
    # judge gets its own authoritative copy of bench/
    rm_rf(task.judge_bench_dir)
    shutil.copytree(task.bench_dir, task.judge_bench_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ensure_dir(task.reference_dir)
    (task.reference_dir / "adapt.patch").write_text(git(["diff", base, adapt_commit], code).stdout)
    tj = read_json(task.dir / "task.json")
    tj.setdefault("adapt", {}).update({"base_commit": base, "adapt_commit": adapt_commit})
    write_json(task.dir / "task.json", tj)
    log(f"adapt {pid}: base={base[:12]} adapt={adapt_commit[:12]}; judge/bench synced; adapt.patch "
        f"{len((task.reference_dir / 'adapt.patch').read_text().splitlines())} lines")
    return {"base_commit": base, "adapt_commit": adapt_commit}


def _scan_forbidden(agent_dir, patterns, aliases=()):
    hits = []
    for p in Path(agent_dir).rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.suffix.lower() not in TEXT_EXT or p.stat().st_size > 5_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for a in aliases:
            text = text.replace(a, "")
        for pat in patterns:
            if pat.search(text):
                hits.append(f"{p.relative_to(agent_dir)}: /{pat.pattern}/")
                break
    return hits


def lint(pid, full=False):
    task = load_task(pid)
    out = []       # (level, message)

    def err(m):
        out.append(("ERROR", m))

    def warn(m):
        out.append(("WARN", m))

    def ok(m):
        out.append(("ok", m))

    for p in task.problems(deep=True):
        err(p)
    if (task.code_dir / ".git").exists() and task.adapt_commit:
        head = head_commit(task.code_dir)
        if head.startswith(task.adapt_commit):
            ok(f"code HEAD = adapt_commit {head[:12]}")
        else:
            err(f"code HEAD {head[:12]} != adapt_commit {task.adapt_commit[:12]} (run `bench adapt`)")
        if is_clean(task.code_dir):
            ok("code working tree clean")
        else:
            err("code working tree dirty (uncommitted protocol edits? run `bench adapt`)")
    if task.judge_bench_dir.exists():
        if sha256_tree(task.bench_dir) == sha256_tree(task.judge_bench_dir):
            ok("judge/bench == agent/bench")
        else:
            err("judge/bench differs from agent/bench (run `bench adapt` to resync)")
    else:
        err("judge/bench missing (run `bench adapt`)")
    lock = task.agent_dir / task.lock_rel
    if lock.exists():
        okl, probs, pins = validate_lock_text(lock.read_text())
        (ok if okl else err)(f"env.lock: {len(pins)} pins" + ("" if okl else "; " + "; ".join(probs[:3])))
    pats = [re.compile(r"/mnt/" + "bn/"), re.compile(r"\.double-check"), re.compile(r"(^|[\s/\"'])env\.sh\b"),
            re.compile(r"baseline_result\.json"), re.compile(r"\bjudge/")]
    test_names = set()
    agent_names = {p.name for p in task.data_dir.rglob("*") if p.is_file()} if task.data_dir.exists() else set()
    generic = re.compile(r"^(input|inputs|label|labels|truth|x|y|seed|seeds)\.[a-z0-9]+$", re.I)
    if task.judge_dir.exists():
        for p in (task.judge_dir / "test").rglob("*"):
            if p.is_file() and len(p.name) >= 10 and p.name not in agent_names and not generic.match(p.name):
                test_names.add(re.escape(p.name))
    if test_names:
        pats.append(re.compile("|".join(sorted(test_names))))
    # A snapshot's original install prefixes are aliases of the copy inside the sandbox, not NAS reads.
    aliases = [m["target"] for m in load_snapshot(task)["mounts"]] if is_snapshot(task) else []
    hits = _scan_forbidden(task.agent_dir, pats, aliases)
    if hits:
        for h in hits[:20]:
            err(f"forbidden reference in agent/: {h}")
    else:
        ok("no NAS paths / double-check / env.sh / judge references in agent/")
    readme = task.agent_dir / "README.md"
    if readme.exists():
        left = re.findall(r"<(?!https?://)[A-Za-z][^<>\n]{2,80}>", readme.read_text(encoding="utf-8", errors="ignore"))
        if left:
            warn(f"README.md still has {len(left)} template placeholder(s), e.g. {left[0][:60]!r}")
        if "{{" in readme.read_text(encoding="utf-8", errors="ignore"):
            err("README.md still contains unrendered {{...}} fields")
    ref = task.reference_dir / "result.json"
    if ref.exists():
        r = read_json(ref)
        hs = r.get("hashes") or {}
        checks = {"env_lock": task.reference_dir / "env.lock", "scorer": task.judge_bench_dir / "scorer.py",
                  "validate": task.judge_bench_dir / "validate.py"}
        stale = [k for k, p in checks.items() if p.exists() and hs.get(k) and hs[k] != sha256_file(p)]
        if stale:
            err(f"reference is stale for {stale} (re-run `bench reference`)")
        else:
            ok(f"reference archived: test mean {r['test'].get('mean')} std {r['test'].get('std')} (seeds {r.get('seeds')})")
        bval = (task.d["baseline"].get("val") or {}).get(task.metric_name)
        if bval is None:
            warn("baseline.val not set in task.json (bench reference fills it)")
    else:
        warn("no judge/reference/result.json yet (run `bench reference`)")
    if full and not any(l == "ERROR" for l, _ in out):
        out += _lint_full(task)
    for lvl, msg in out:
        print(f"[{lvl:5s}] {msg}")
    n_err = sum(1 for l, _ in out if l == "ERROR")
    print(f"lint {pid}: {n_err} error(s), {sum(1 for l, _ in out if l == 'WARN')} warning(s)")
    return n_err


def _lint_full(task):
    """validator must accept the archived baseline prediction and reject corrupted variants; scorer reproduces the archive."""
    out = []
    ref_dir = task.reference_dir
    preds = sorted(ref_dir.glob("pred_val_seed*.out"))
    if not preds:
        return [("WARN", "full lint skipped: no archived pred_val_seed*.out (run `bench reference`)")]
    local = ensure_dir(LOCAL_ROOT / "lint" / task.pid)
    b = build_environment(task, local / "venv", ref_dir / "env.lock", task.indexes, UV_CACHE, local / "build.log", cap_s=task.build_cap_s)
    if not b["ok"]:
        return [("ERROR", f"full lint: venv build failed, see {local / 'build.log'}")]
    from .judge import tool_ctx
    env, py = tool_ctx(local / "venv", task.cpus, task)
    validator, scorer = task.judge_bench_dir / "validate.py", task.judge_bench_dir / "scorer.py"
    pred = preds[0]
    okv, msg = run_validate(validator, pred, task.val_input(), py, env)
    out.append(("ok" if okv else "ERROR", f"validate accepts baseline val prediction: {msg[:80]}"))
    head = pred.read_bytes()[:8]
    binary = head.startswith(b"PK\x03\x04") or head.startswith(b"\x93NUMPY") or b"\x00" in head
    if binary:
        out.append(("WARN", "prediction file is binary (npz/npy): the text corruption variants are skipped; "
                            "provide rejected samples under judge/tools/negatives/"))
    else:
        text = pred.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(text) >= 2:
            nan = text[:1] + [re.sub(r"[-0-9.eE+]+$", "nan", text[1])] + text[2:]
            # line 2 may be a header or end in a quoted field: put nan in the first numeric field of a later line instead
            for k in range(1, len(text) if nan == text else 0):
                line = re.sub(r"(?<=[,\t ])-?\d[\d.eE+-]*(?=[,\t ]|$)", "nan", text[k], count=1)
                if line != text[k]:
                    nan = text[:k] + [line] + text[k + 1:]
                    break
            variants = {"truncated": text[:-1], "nan": nan, "duplicate": text + [text[-1]]}
        else:  # one line, e.g. a JSON document: cut it, put NaN in the first number, write it twice
            line = text[0] if text else ""
            variants = {"truncated": [line[:len(line) // 2]],
                        "nan": [re.sub(r"(?<=[:\[,])\s*-?\d[\d.eE+-]*", "NaN", line, count=1)],
                        "duplicate": [line, line]}
        if variants["nan"] == text:  # no number anywhere, e.g. a JSON of names: an unchanged file proves nothing
            del variants["nan"]
            out.append(("WARN", "no numeric field to replace with nan: that corruption variant is skipped"))
        for name, lines in variants.items():
            vp = local / f"corrupt_{name}.out"
            vp.write_text("\n".join(lines) + "\n")
            okc, msgc = run_validate(validator, vp, task.val_input(), py, env)
            out.append(("ok" if not okc else "ERROR", f"validate rejects {name} file" + ("" if not okc else " -- it ACCEPTED it")))
    neg_dir = task.judge_dir / "tools" / "negatives"
    negs = sorted(p for p in neg_dir.iterdir() if p.is_file()) if neg_dir.is_dir() else []
    if binary and not negs:
        out.append(("ERROR", "binary prediction format but no judge/tools/negatives/ samples (each must be rejected by validate.py)"))
    for np_ in negs:
        okc, msgc = run_validate(validator, np_, task.val_input(), py, env)
        out.append(("ok" if not okc else "ERROR", f"validate rejects negative sample {np_.name}" + ("" if not okc else " -- it ACCEPTED it")))
    try:
        m = run_scorer(scorer, pred, task.val_labels(), py, env, timeout=task.score_cap_s)
        r = read_json(ref_dir / "result.json")
        seed = re.search(r"seed(\d+)", pred.name).group(1)
        archived = ((r.get("per_seed") or {}).get(seed) or {}).get("val") or {}
        a = archived.get(task.metric_name)
        v = m.get(task.metric_name)
        if a is not None and v is not None and abs(a - v) <= 1e-9 + 1e-6 * abs(a):
            out.append(("ok", f"scorer reproduces archived val {task.metric_name}={v}"))
        else:
            out.append(("ERROR", f"scorer gives {v} but archive says {a}"))
    except Exception as e:  # noqa: BLE001
        out.append(("ERROR", f"scorer failed on baseline prediction: {e}"))
    rm_rf(local)
    return out
