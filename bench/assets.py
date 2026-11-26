"""Fetch and verify AutoScientistBench datasets, hidden test splits, and pre-built runtimes.

Assets are described by `assets_manifest.json` at the repository root:
- 103 tasks: `agent/data` and `judge/test` hosted on Hugging Face as single or multi-part `.tar` archives.
- `052_AnisoBivFunc`: 153.6 GB of synthetic surfaces generated deterministically locally via
  `scripts/generate_052_AnisoBivFunc.py` using its R snapshot runtime (`runtime/r.tar`).
- 29 snapshot tasks: frozen non-Python runtimes (`runtime/*.tar`) hosted directly on Hugging Face.
"""
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from . import ROOT, TASKS_DIR

MANIFEST_PATH = ROOT / "assets_manifest.json"
VALID_ONLY = {"data", "test", "runtime"}


class _MultiPartReader(io.RawIOBase):
    """Stream-concatenate ordered archive part files without writing a merged temporary file."""

    def __init__(self, paths):
        self._paths = [Path(p) for p in paths]
        self._idx = 0
        self._cur = None

    def readable(self):
        return True

    def _ensure_open(self):
        while self._cur is None and self._idx < len(self._paths):
            self._cur = self._paths[self._idx].open("rb")
            self._idx += 1
        return self._cur is not None

    def readinto(self, b):
        mv = memoryview(b)
        total = 0
        while total < len(mv):
            if not self._ensure_open():
                break
            n = self._cur.readinto(mv[total:])
            if not n:
                self._cur.close()
                self._cur = None
                continue
            total += n
        return total

    def close(self):
        if self._cur is not None:
            self._cur.close()
            self._cur = None
        super().close()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path=None) -> dict:
    p = Path(path) if path else MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Asset manifest not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_tasks(manifest: dict, task_spec: str) -> list[str]:
    all_ids = list(manifest["tasks"].keys())
    if task_spec == "all":
        return all_ids
    chosen = [t.strip() for t in task_spec.split(",") if t.strip()]
    unknown = [t for t in chosen if t not in manifest["tasks"]]
    if unknown:
        raise ValueError(f"Unknown task ID(s): {unknown}")
    return chosen


def _resolve_hf_repo(manifest: dict, hf_repo: str | None) -> str:
    repo = hf_repo or os.environ.get("ASB_HF_REPO") or manifest.get("hf_repo_default") or ""
    if not repo or repo.startswith("<"):
        raise ValueError(
            "Hugging Face dataset repo ID is not configured. Pass --hf-repo <ORG>/<REPO> "
            "or set the ASB_HF_REPO environment variable (or pass --local-dir for local staged archives)."
        )
    return repo


def _fetch_file(rel_hf_path: str, expected_bytes: int, expected_sha256: str, *, hf_repo: str | None, local_dir: Path | None) -> Path:
    if local_dir is not None:
        src = (Path(local_dir) / rel_hf_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Missing local staged asset: {src}")
        path = src
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Fetching from Hugging Face requires `huggingface_hub` (`pip install huggingface_hub`).") from exc
        path = Path(hf_hub_download(repo_id=hf_repo, repo_type="dataset", filename=rel_hf_path))

    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"Size mismatch for {rel_hf_path}: expected {expected_bytes} bytes, got {actual_bytes}")
    if expected_sha256:
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {rel_hf_path}: expected {expected_sha256}, got {actual_sha}")
    return path


def _safe_extract_tar_parts(part_paths: list[Path], dest_dir: Path):
    """Safely stream-extract tar part(s) into `dest_dir`, blocking traversal and unsafe links."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    reader = _MultiPartReader(part_paths)
    links = []
    try:
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for member in tf:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                target = dest_dir.joinpath(*rel.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                    target.chmod(member.mode & 0o777)
                elif member.issym() or member.islnk():
                    links.append(member)
                else:
                    raise ValueError(f"Unsupported special tar entry: {member.name}")
        for member in links:
            target = dest_dir.joinpath(*PurePosixPath(member.name).parts)
            if not target.parent.resolve().is_relative_to(dest_resolved):
                raise ValueError(f"Symlink parent escapes destination: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            if member.issym():
                link_target = Path(member.linkname)
                if link_target.is_absolute():
                    raise ValueError(f"Absolute symlink not allowed in data archive: {member.name} -> {member.linkname}")
                resolved_link = (target.parent / link_target).resolve()
                if not resolved_link.is_relative_to(dest_resolved):
                    raise ValueError(f"Symlink escapes destination: {member.name} -> {member.linkname}")
                target.symlink_to(member.linkname)
            else:
                rel = PurePosixPath(member.linkname)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe hardlink: {member.linkname}")
                src_target = dest_dir.joinpath(*rel.parts)
                if not src_target.resolve().is_relative_to(dest_resolved):
                    raise ValueError(f"Hardlink escapes destination: {member.linkname}")
                os.link(src_target, target)
    finally:
        reader.close()


def _marker_matches(target_dir: Path, parts: list[dict]) -> bool:
    marker = target_dir / ".asb_fetched.json"
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        return saved.get("parts") == parts
    except (OSError, ValueError):
        return False


def _write_marker(target_dir: Path, parts: list[dict]):
    (target_dir / ".asb_fetched.json").write_text(
        json.dumps({"schema_version": 1, "parts": parts}, indent=2) + "\n",
        encoding="utf-8",
    )


def fetch_assets(
    task_spec: str,
    *,
    hf_repo: str | None = None,
    only: list[str] | None = None,
    local_dir: str | Path | None = None,
    cores: int | None = None,
    force: bool = False,
    verify_only: bool = False,
    no_sandbox: bool = False,
    manifest_path: Path | None = None,
    tasks_dir: Path | None = None,
) -> dict:
    manifest = load_manifest(manifest_path)
    task_ids = _resolve_tasks(manifest, task_spec)
    wanted = set(only) if only else set(VALID_ONLY)
    bad_only = wanted - VALID_ONLY
    if bad_only:
        raise ValueError(f"Invalid --only value(s): {sorted(bad_only)}; expected subset of {sorted(VALID_ONLY)}")

    troot = Path(tasks_dir) if tasks_dir else TASKS_DIR
    ldir = Path(local_dir).resolve() if local_dir else None
    repo = None if (ldir is not None or verify_only) else _resolve_hf_repo(manifest, hf_repo)

    summary = {"tasks": len(task_ids), "fetched": 0, "skipped": 0, "verified": 0, "failed": []}

    for tid in task_ids:
        entry = manifest["tasks"][tid]
        tdir = troot / tid
        if not tdir.is_dir():
            raise FileNotFoundError(f"Task directory missing: {tdir}")

        # 1. Runtimes first (052_AnisoBivFunc local generator depends on runtime/r.tar)
        need_runtime = "runtime" in wanted or (
            entry["data"].get("mode") == "local_generator" and bool(wanted & {"data", "test"})
        )
        if need_runtime and entry.get("runtimes"):
            for rt in entry["runtimes"]:
                dst = tdir / rt["target_path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                already_ok = dst.is_file() and dst.stat().st_size == rt["bytes"] and (
                    not verify_only or _sha256_file(dst) == rt["sha256"]
                )
                if verify_only:
                    if already_ok:
                        summary["verified"] += 1
                    else:
                        summary["failed"].append(f"{tid}:{rt['target_path']}")
                    continue
                if already_ok and not force:
                    summary["skipped"] += 1
                    continue
                print(f"[fetch] {tid} runtime {rt['name']} ({rt['bytes'] / 1e6:.1f} MB)...", flush=True)
                src = _fetch_file(rt["hf_path"], rt["bytes"], rt["sha256"], hf_repo=repo, local_dir=ldir)
                tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copy2(src, tmp_dst)
                os.replace(tmp_dst, dst)
                summary["fetched"] += 1

        # 2. Data & hidden test archives (or local deterministic generator for 052_AnisoBivFunc)
        if entry["data"].get("mode") == "local_generator":
            gen_splits = []
            if "data" in wanted:
                gen_splits.append("val")
            if "test" in wanted:
                gen_splits.append("test")
            if gen_splits:
                split_arg = "all" if len(gen_splits) == 2 else gen_splits[0]
                import sys
                scripts_dir = str(ROOT / "scripts")
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                import generate_052_AnisoBivFunc
                res = generate_052_AnisoBivFunc.generate_shards(
                    tdir,
                    split=split_arg,
                    cores=cores,
                    verify_only=verify_only,
                    no_sandbox=no_sandbox,
                )
                if verify_only:
                    if res["ok"]:
                        summary["verified"] += 1
                    else:
                        summary["failed"].append(f"{tid}:{split_arg}")
                else:
                    summary["fetched"] += 1
        else:
            for kind in ("data", "test"):
                if kind not in wanted:
                    continue
                spec = entry[kind]
                target_dir = tdir / spec["target_dir"]
                parts = spec.get("parts") or []
                if verify_only:
                    if _marker_matches(target_dir, parts):
                        summary["verified"] += 1
                    else:
                        summary["failed"].append(f"{tid}:{kind}")
                    continue
                if not force and _marker_matches(target_dir, parts):
                    summary["skipped"] += 1
                    continue
                tot_mb = sum(p["bytes"] for p in parts) / 1e6
                print(f"[fetch] {tid} {kind} ({tot_mb:.1f} MB, {len(parts)} part(s))...", flush=True)
                part_paths = [
                    _fetch_file(p["path"], p["bytes"], p["sha256"], hf_repo=repo, local_dir=ldir)
                    for p in parts
                ]
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                _safe_extract_tar_parts(part_paths, target_dir)
                _write_marker(target_dir, parts)
                summary["fetched"] += 1

    if verify_only and summary["failed"]:
        raise RuntimeError(f"Asset verification failed for {len(summary['failed'])} item(s): {summary['failed'][:10]}")
    print(f"[fetch] summary: {summary}")
    return summary


def assets_status(task_spec: str = "all", *, manifest_path: Path | None = None, tasks_dir: Path | None = None):
    manifest = load_manifest(manifest_path)
    task_ids = _resolve_tasks(manifest, task_spec)
    troot = Path(tasks_dir) if tasks_dir else TASKS_DIR
    ready = 0
    for tid in task_ids:
        entry = manifest["tasks"][tid]
        tdir = troot / tid
        if entry["data"].get("mode") == "local_generator":
            val_bins = len(list((tdir / "agent/data/val/input").glob("*.bin"))) if (tdir / "agent/data/val/input").is_dir() else 0
            test_bins = len(list((tdir / "judge/test/input").glob("*.bin"))) if (tdir / "judge/test/input").is_dir() else 0
            d_ok = val_bins >= 120
            t_ok = test_bins >= 120
        else:
            d_ok = (tdir / "agent/data").is_dir() and any((tdir / "agent/data").iterdir())
            t_ok = (tdir / "judge/test").is_dir() and any((tdir / "judge/test").iterdir())
        rts = entry.get("runtimes") or []
        r_ok = all((tdir / r["target_path"]).is_file() and (tdir / r["target_path"]).stat().st_size == r["bytes"] for r in rts)
        if d_ok and t_ok and r_ok:
            ready += 1
        rt_lbl = "n/a" if not rts else ("ok" if r_ok else "missing")
        print(f"{tid:44s} env={entry['env_mode']:8s} data={'ok' if d_ok else 'missing':7s} test={'ok' if t_ok else 'missing':7s} runtime={rt_lbl}")
    print(f"\nReady: {ready}/{len(task_ids)} tasks")
