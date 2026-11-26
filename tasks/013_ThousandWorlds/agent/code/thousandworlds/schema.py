from pathlib import Path

def support_path(data_dir, subset, *, kind):
    if kind != "stats_dir":
        raise ValueError("Only materialized training statistics are supported")
    return Path(data_dir) / "train" / subset
