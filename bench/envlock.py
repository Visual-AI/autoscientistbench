"""The environment is part of the patch: code/env.lock (pip-freeze format) is the only source of truth.

sandbox venv  = uv venv + uv pip sync code/env.lock      (built by the operator before launch, trusted cache)
agent         = uv pip install whatever it likes (through the allowlist proxy)
bench submit  = uv pip freeze -> code/env.lock (validated), shipped inside the patch
judge         = uv venv + uv pip sync <patched code/env.lock>, same image / indexes / command
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .budget import remaining_time
from .util import capture, run

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?==([A-Za-z0-9][A-Za-z0-9.+!_-]*)$")
BARE_MINOR_RE = re.compile(r"^\d+\.\d+$")
EXACT_RE = re.compile(r"^\d+\.\d+\.\d+$")


def uv_bin():
    return os.environ.get("ARB_UV", shutil.which("uv") or "uv")


def validate_lock_text(text):
    """-> (ok, problems, pins) ; a pin is (name, version)."""
    problems, pins = [], []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e") or line.startswith("--editable"):
            problems.append(f"line {ln}: editable install cannot be rebuilt: {line}")
            continue
        if "@" in line or "://" in line or line.startswith("-"):
            problems.append(f"line {ln}: only `name==version` pins from an allowed index are reproducible: {line}")
            continue
        m = PIN_RE.match(line)
        if not m:
            problems.append(f"line {ln}: not a `name==version` pin: {line}")
            continue
        pins.append((m.group(1).lower().replace("_", "-"), m.group(3)))
    return (not problems), problems, pins


def freeze(venv_dir, env=None):
    """-> lock text from `uv pip freeze` of venv_dir (sorted, comments stripped)."""
    py = str(Path(venv_dir) / "bin" / "python")
    e = dict(os.environ if env is None else env)
    e.setdefault("UV_NO_PROGRESS", "1")
    e["UV_PYTHON_DOWNLOADS"] = "never"
    p = capture([uv_bin(), "pip", "freeze", "--python", py], env=e, check=True, timeout=300)
    # Packaging tools can also be build/runtime dependencies (e.g. pkg_resources
    # comes from setuptools). Keep their exact versions: seeding a fresh judge
    # venv neither preserves these pins nor guarantees the same imports.
    lines = sorted({l.strip() for l in p.stdout.splitlines() if l.strip() and not l.startswith("#")}, key=str.lower)
    return "\n".join(lines) + "\n"


def sync_args(indexes):
    args = []
    for i, idx in enumerate(indexes or []):
        args += (["--index-url", idx] if i == 0 else ["--extra-index-url", idx])
    if indexes and len(indexes) > 1:
        # torch==x.y.z+cpu lives only on the pytorch index while `torch` itself is on PyPI:
        # uv's default first-index strategy would stop at PyPI and fail.
        args += ["--index-strategy", "unsafe-best-match"]
    return args


def exact_python(python, env, deadline_ts):
    """"3.11" -> the patch release uv would use ("3.11.15"); a more specific request is returned unchanged. -> (version, error)

    From a bare minor, uv links the venv's bin/python through its cpython-3.11-* symlink dir, while sys.base_prefix names the real
    directory. A nested sandbox binding those prefixes (private_eval.py) then sees a dangling /venv/bin/python, and `python` silently
    falls back to the system interpreter and its packages. An exact request links the real directory; it is the same interpreter."""
    python = str(python)
    if not BARE_MINOR_RE.match(python):
        return python, None
    find = [uv_bin(), "python", "find", "--system", "--no-project", python]
    p = capture(find, env=env, timeout=remaining_time(deadline_ts, 600))
    if p.returncode != 0:      # not on this node yet
        p = capture([uv_bin(), "python", "install", "--no-bin", python], env=env, timeout=remaining_time(deadline_ts, 600))
        if p.returncode != 0:
            return None, p.stderr[-1000:]
        p = capture(find, env=env, timeout=remaining_time(deadline_ts, 600))
        if p.returncode != 0:
            return None, p.stderr[-1000:]
    lines = p.stdout.strip().splitlines()
    if not lines:
        return None, "uv python find printed nothing"
    try:
        p = capture([lines[-1], "-c", "import platform; print(platform.python_version())"], env=env,
                    timeout=remaining_time(deadline_ts, 60))
    except OSError as exc:
        return None, str(exc)
    version = p.stdout.strip()
    if p.returncode != 0 or not EXACT_RE.match(version):
        return None, f"{lines[-1]} reported {version!r}: {p.stderr[-500:]}"
    return version, None


def _log_note(log_path, text):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def build_venv(venv_dir, python, lock_path, indexes, uv_cache, log_path, cap_s=1800, env=None, python_install_dir=None):
    """Fresh venv at venv_dir from lock_path. -> dict(ok, elapsed_s, stage, log)."""
    t0 = time.monotonic()
    deadline_ts = time.time() + float(cap_s)
    stage = "venv"

    def result(ok, timed_out):
        return {"ok": ok, "elapsed_s": round(time.monotonic() - t0, 1),
                "stage": "done" if ok else stage, "log": str(log_path), "timed_out": timed_out}

    try:
        remaining_time(deadline_ts)
    except TimeoutError:
        return result(False, True)
    venv_dir = Path(venv_dir)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ if env is None else env)
    # Always a uv-managed CPython (downloaded node-locally when absent): the distro python's headers differ (Debian 3.11
    # lacks a top-level longintrepr.h, so Cython 0.29 extensions fail to build there) and a venv must behave the same on the
    # devbox, the conversion node and the judge node.
    e.update({"UV_CACHE_DIR": str(uv_cache), "UV_NO_PROGRESS": "1", "UV_HTTP_TIMEOUT": "300",
              "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": e.get("UV_PYTHON_DOWNLOADS", "automatic"),
              "UV_PYTHON_PREFERENCE": e.get("UV_PYTHON_PREFERENCE", "only-managed")})
    if python_install_dir:
        e["UV_PYTHON_INSTALL_DIR"] = str(python_install_dir)
    e.pop("VIRTUAL_ENV", None)
    # Operator-side mirror for pods without public PyPI connectivity. Keep
    # every pinned version and any task-specific (e.g. PyTorch) extra index.
    mirror = e.get("ARB_PYPI_MIRROR")
    seed_indexes = []
    if mirror:
        from urllib.parse import urlsplit
        url = urlsplit(mirror)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("ARB_PYPI_MIRROR must be a credential-free HTTPS URL")
        public = {"https://pypi.org/simple", "https://pypi.python.org/simple"}
        indexes = [mirror if idx.rstrip("/") in public else idx for idx in (indexes or [mirror])]
        seed_indexes = ["--index-url", mirror]
        _log_note(log_path, f"### operator PyPI mirror: {mirror}; submission pins unchanged")
    wheelhouse = e.get("ARB_BUILD_WHEELHOUSE")
    find_args = []
    if wheelhouse:
        if not Path(wheelhouse).is_dir():
            raise ValueError("ARB_BUILD_WHEELHOUSE is not an existing directory")
        find_args = ["--find-links", wheelhouse]
        if e.get("ARB_CACHED_PYTORCH_CPU_INDEX") == "1":
            indexes = [idx for idx in (indexes or [])
                       if idx.rstrip("/") != "https://download.pytorch.org/whl/cpu"]
        _log_note(log_path, f"### operator verified wheel cache: {wheelhouse}; submission pins unchanged")
    try:
        exact, error = exact_python(python, e, deadline_ts)
        if exact is None:
            _log_note(log_path, f"### cannot resolve python {python}: {error}")
            return result(False, False)
        if exact != str(python):
            _log_note(log_path, f"### python {python} -> {exact}")
        r = run([uv_bin(), "venv", "--quiet", "--python", exact, "--seed", "--no-project", *seed_indexes, str(venv_dir)],
                env=e, timeout=remaining_time(deadline_ts, 600), log_path=log_path)
        if r["rc"] != 0 or r["timed_out"]:
            return result(False, r["timed_out"])
        stage = "sync"
        r = run([uv_bin(), "pip", "sync", "--quiet", "--python", str(venv_dir / "bin" / "python"), *sync_args(indexes), *find_args,
                 str(lock_path)], env=e, timeout=remaining_time(deadline_ts), log_path=log_path)
        remaining_time(deadline_ts)
        return result(r["rc"] == 0 and not r["timed_out"], r["timed_out"])
    except (TimeoutError, subprocess.TimeoutExpired):
        return result(False, True)


def venv_env(venv_dir, base=None):
    """Environment that makes `python` resolve to the venv (train.sh / predict.sh just call python)."""
    venv_dir = str(venv_dir)
    e = dict(base if base is not None else os.environ)
    e["VIRTUAL_ENV"] = venv_dir
    e["PATH"] = venv_dir + "/bin:" + e.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    e.pop("PYTHONHOME", None)
    e.pop("PYTHONPATH", None)
    return e
