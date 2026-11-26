"""Patch = the submission unit (D1). Making, checking and applying it.

make_patch: working tree of code/ (incl. untracked, minus .gitignore) vs the adapt commit, computed with a
temporary index so the agent's own commits / staging are irrelevant.
check_patch: the frozen-surface locks that can be checked statically (D7): size, no binaries, no new data
files, sane paths. bench/ and data/ live outside the repo, so a patch cannot touch them by construction.
apply_patch: `git apply --check` then `git apply` onto a clean clone at the adapt commit.
"""
import os
import re
import tempfile
from pathlib import Path

from .util import capture

DIFF_HDR = re.compile(r"^diff --git a/(.*?) b/(.*?)$")


def git(args, cwd, env=None, check=True, timeout=600):
    e = dict(os.environ)
    e.update({"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    if env:
        e.update(env)
    return capture(["git", *args], cwd=cwd, env=e, check=check, timeout=timeout)


def head_commit(code_dir):
    return git(["rev-parse", "HEAD"], code_dir).stdout.strip()


def is_clean(code_dir):
    return git(["status", "--porcelain", "--untracked-files=all"], code_dir).stdout.strip() == ""


def make_patch(code_dir, base_commit):
    """-> patch bytes (unified diff with --binary so binaries are at least visible and rejectable)."""
    code_dir = str(code_dir)
    with tempfile.TemporaryDirectory(prefix="arb-idx-") as td:
        idx = os.path.join(td, "index")
        env = {"GIT_INDEX_FILE": idx}
        git(["read-tree", base_commit], code_dir, env=env)
        git(["add", "-A", "--", "."], code_dir, env=env)
        # capture() reads text and normalizes CRLF, including CR bytes in diff
        # context. Let git write raw bytes so strict replay matches the baseline.
        output = Path(td) / "patch"
        capture(["git", "diff", "--cached", "--binary", "--no-color", "--no-ext-diff", "--full-index",
                 "--src-prefix=a/", "--dst-prefix=b/", f"--output={output}", base_commit],
                cwd=code_dir, env={**os.environ, **env, "LC_ALL": "C"}, check=True)
        return output.read_bytes()


def parse_patch_files(patch_bytes):
    """-> list of dicts {path, new, deleted, binary, added, removed} per file in the diff."""
    files = []
    cur = None
    text = patch_bytes.decode("utf-8", "replace")
    for line in text.splitlines():
        m = DIFF_HDR.match(line)
        if m:
            cur = {"path": m.group(2), "old_path": m.group(1), "new": False, "deleted": False, "binary": False,
                   "added": 0, "removed": 0}
            files.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur["new"] = True
        elif line.startswith("deleted file mode"):
            cur["deleted"] = True
        elif line.startswith("GIT binary patch") or line.startswith("Binary files"):
            cur["binary"] = True
        elif line.startswith("+") and not line.startswith("+++"):
            cur["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            cur["removed"] += 1
    return files


def check_patch(patch_bytes, limits, protected_paths=()):
    """-> (ok, problems:list[str], files:list[dict]). Static part of the D7 locks."""
    problems = []
    files = parse_patch_files(patch_bytes)
    max_bytes = int(limits.get("max_bytes", 2 * 1024 * 1024))
    forbidden_ext = tuple(limits.get("forbidden_new_ext", []))
    if len(patch_bytes) > max_bytes:
        problems.append(f"patch is {len(patch_bytes)} bytes > limit {max_bytes}")
    if not files and patch_bytes.strip():
        problems.append("patch has content but no `diff --git` headers")
    for f in files:
        path = f["path"]
        if path.startswith("/") or path.split("/")[0] == ".." or "/../" in path:
            problems.append(f"unsafe path {path}")
        if path == ".git" or path.startswith(".git/"):
            problems.append(f"patch touches .git: {path}")
        for pp in protected_paths:
            if path == pp or path.startswith(pp.rstrip("/") + "/"):
                problems.append(f"patch touches protected path {path}")
        if f["binary"]:
            problems.append(f"binary content not allowed: {path}")
        if f["new"] and path.lower().endswith(forbidden_ext):
            problems.append(f"new data/artifact file not allowed in a patch: {path}")
    return (not problems), problems, files


def apply_patch(clone_dir, patch_path):
    """Apply onto a clean checkout; raises RuntimeError with git's message on failure."""
    r = git(["apply", "--check", "--whitespace=nowarn", str(patch_path)], clone_dir, check=False)
    if r.returncode != 0:
        raise RuntimeError("git apply --check failed:\n" + (r.stderr or r.stdout)[-4000:])
    r = git(["apply", "--whitespace=nowarn", str(patch_path)], clone_dir, check=False)
    if r.returncode != 0:
        raise RuntimeError("git apply failed:\n" + (r.stderr or r.stdout)[-4000:])


def clean_clone(src_repo, dst_dir, commit):
    """Local clone of src_repo at `commit`. Objects are copied, not hardlinked: the clone is a writable bind of a sandbox
    that runs submitted code, and a hardlinked object is the same file as the task's own."""
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        raise FileExistsError(dst_dir)
    git(["clone", "--quiet", "--no-checkout", "--no-hardlinks", str(src_repo), str(dst_dir)], cwd=None)
    git(["checkout", "--quiet", "--detach", commit], dst_dir)
    head = head_commit(dst_dir)
    if not head.startswith(commit) and not commit.startswith(head):
        raise RuntimeError(f"checkout mismatch: HEAD={head} wanted {commit}")
    return head


_DETERMINISTIC_GIT_ENV = {
    "GIT_AUTHOR_NAME": "AutoScientistBench",
    "GIT_AUTHOR_EMAIL": "bench@autoscientistbench.org",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "AutoScientistBench",
    "GIT_COMMITTER_EMAIL": "bench@autoscientistbench.org",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def materialize_code_repo(src_code_dir, dst_dir, commit=None):
    """Create a clean, isolated git checkout of `src_code_dir` at `dst_dir` and return its HEAD commit SHA.

    - When `src_code_dir` is itself a git repository (e.g., local development tasks or `toy-ridge`),
      delegates to `clean_clone(..., commit)` (or clones HEAD if `commit` is omitted).
    - When `src_code_dir` is a plain directory of tracked files (as distributed on GitHub without
      nested `.git` folders), copies the tree to `dst_dir` and creates a single deterministic
      baseline commit so `make_patch` and `apply_patch` work identically.
    """
    import shutil

    src_code_dir = Path(src_code_dir)
    dst_dir = Path(dst_dir)
    if (src_code_dir / ".git").exists():
        target = commit or head_commit(src_code_dir)
        return clean_clone(src_code_dir, dst_dir, target)
    if dst_dir.exists():
        raise FileExistsError(dst_dir)
    shutil.copytree(
        src_code_dir,
        dst_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    git(["init", "--quiet", "-b", "main"], dst_dir, env=_DETERMINISTIC_GIT_ENV)
    git(["add", "-A", "--", "."], dst_dir, env=_DETERMINISTIC_GIT_ENV)
    git(["commit", "--quiet", "--allow-empty", "-m", "bench-adapt"], dst_dir, env=_DETERMINISTIC_GIT_ENV)
    return head_commit(dst_dir)
