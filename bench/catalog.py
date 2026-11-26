"""Current benchmark membership; retired task artifacts remain readable."""
import json
from pathlib import Path

from . import ROOT


def read_catalog(path=None):
    path = Path(path) if path is not None else ROOT / "benchmark_tasks.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    active = data.get("task_ids")
    retired = data.get("retired", [])
    if (data.get("schema_version") != 1 or not isinstance(active, list) or not active
            or not all(isinstance(pid, str) and pid for pid in active)
            or not isinstance(retired, list)
            or not all(isinstance(row, dict) and isinstance(row.get("task_id"), str) for row in retired)):
        raise ValueError(f"Invalid benchmark task catalog: {path}")
    removed = [row["task_id"] for row in retired]
    if len(set(active)) != len(active) or len(set(removed)) != len(removed) or set(active) & set(removed):
        raise ValueError(f"Duplicate or conflicting benchmark task membership: {path}")
    return data


def retired_task_ids():
    return {row["task_id"] for row in (read_catalog() or {}).get("retired", [])}


def require_active_task(pid):
    if pid in retired_task_ids():
        raise ValueError(f"Task {pid} was removed from the benchmark; see benchmark_tasks.json. Historical results remain archived.")
