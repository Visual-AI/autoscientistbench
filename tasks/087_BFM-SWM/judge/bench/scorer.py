"""Run the sealed policy once per instance against a private supplier oracle and print welfare_ratio."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offer_oracle import OfferOracle, load_graph
from private_eval import Evaluator, remaining_time

TASK_ID = "icml-4454-BFM-SWM"
TASK_CPUS = 4


def read_labels(labels):
    labels = Path(labels)
    manifest = json.loads((labels / "manifest.json").read_text())
    if manifest.get("task_id") != TASK_ID or manifest.get("protocol") != "private-policy-v2":
        raise ValueError("wrong private instance container")
    with (labels / "instances.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("invalid instance list")
    return manifest, rows


def run_policy(pred_path, labels):
    """-> (rows, records, timing): one trusted record per instance, in label order."""
    started = time.monotonic()
    labels = Path(labels)
    manifest, rows = read_labels(labels)
    graph_source = (labels / manifest["graph_dir"]).resolve()
    with np.load(labels / "costs.npz", allow_pickle=False) as z:
        costs = {name: z[name] for name in {r["draw"] for r in rows}}
    with tempfile.TemporaryDirectory(prefix="bfm-score-") as scratch:
        scratch = Path(scratch)
        # One node-local copy of the public graphs: every worker reads them, labels stay out of its sight.
        graphs = {}
        for name in sorted({r["graph"] for r in rows}):
            shutil.copytree(graph_source / name, scratch / "graphs" / name)
            graphs[name] = load_graph(scratch / "graphs" / name)
        inputs = scratch / "inputs"
        inputs.mkdir()
        for index, row in enumerate(rows):
            n, _, _ = graphs[row["graph"]]
            if costs[row["draw"]].shape != (n,):
                raise ValueError("cost draw does not match its graph")
            (inputs / f"{index}.json").write_text(json.dumps(
                {"id": row["id"], "graph": row["graph"], "budget": float(row["budget"]), "algorithm_index": index}))
        limit = int(os.environ.get("OMP_NUM_THREADS") or TASK_CPUS)
        workers = max(1, min(TASK_CPUS, limit, len(rows), len(os.sched_getaffinity(0))))
        preparation = time.monotonic() - started
        with Evaluator(pred_path, TASK_ID, cpus=workers, timeout_s=None, data_dir=scratch / "graphs") as evaluator:
            artifact = time.monotonic() - evaluator.started
            loop_started = time.monotonic()
            lock = threading.Lock()
            held = set()
            invalid = threading.Event()  # one broken rule settles the result, so the queued instances need not run

            def acquire():
                # The node's CPU pool can change during a run, so each worker picks from the live mask.
                with lock:
                    live = os.sched_getaffinity(0)
                    cpu = min(live - held or live)
                    held.add(cpu)
                    return cpu

            def run(index):
                if invalid.is_set():
                    return None
                row = rows[index]
                _, offset, adjacency = graphs[row["graph"]]
                for attempt in range(5):
                    # A retried instance starts over with a fresh oracle state.
                    oracle = OfferOracle(costs[row["draw"]], float(row["budget"]), offset, adjacency)
                    cpu = acquire()
                    try:
                        result = evaluator.run(inputs / f"{index}.json", handler=oracle, affinity=[cpu])
                    except (ValueError, RuntimeError) as exc:
                        moved = cpu not in os.sched_getaffinity(0) or "invalid worker CPU allocation" in str(exc)
                        if not moved or attempt == 4:
                            invalid.set()
                            why = oracle.violations[-1] if oracle.violations else str(exc)
                            raise ValueError(f"instance {row['id']}: {why}") from exc
                        continue
                    finally:
                        with lock:
                            held.discard(cpu)
                    with open(result["output_path"], errors="replace") as f:
                        note = f.read(4096)  # the policy's own note; never scored
                    try:
                        trusted = oracle.accepted_record()
                    except ValueError:
                        invalid.set()
                        raise
                    record = dict(trusted, elapsed_s=result["elapsed_s"], rpc_calls=result["rpc_calls"], policy_output=note)
                    shutil.rmtree(Path(result["output_path"]).parent)
                    return record

            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(run, range(len(rows))))
            remaining_time()
            timing = {"preparation_s": preparation, "artifact_prepare_s": artifact, "workers": workers,
                      "prediction_elapsed_s": time.monotonic() - loop_started}
    return rows, records, timing


def evaluate(pred_path, labels):
    rows, records, timing = run_policy(pred_path, labels)
    ratios = [record["welfare"] / float(row["greedy"]) for row, record in zip(rows, records)]
    groups = {}
    for row, ratio in zip(rows, ratios):
        for key in ("graph:" + row["graph"], "family:" + row["family"]):
            groups.setdefault(key, []).append(ratio)
    elapsed = [record["elapsed_s"] for record in records]
    return {"welfare_ratio": float(np.mean(ratios)), "guards_ok": True, "n_instances": len(rows),
            "mean_welfare": float(np.mean([record["welfare"] for record in records])),
            "mean_greedy": float(np.mean([float(row["greedy"]) for row in rows])),
            "mean_payment_share": float(np.mean([record["payment"] / float(row["budget"]) for row, record in zip(rows, records)])),
            "mean_upper_bound_ratio": float(np.mean([float(row["upper_bound"]) / float(row["greedy"]) for row in rows])),
            "per_group": {key: float(np.mean(value)) for key, value in sorted(groups.items())},
            "offers_total": int(sum(record["offers"] for record in records)),
            "rpc_calls_total": int(sum(record["rpc_calls"] for record in records)),
            "worker_elapsed_mean_s": float(np.mean(elapsed)), "worker_elapsed_max_s": float(np.max(elapsed)),
            "protocol": "private-policy-v2", **timing}


if __name__ == "__main__":
    try:
        output = evaluate(sys.argv[1], sys.argv[2])
    except Exception as exc:
        output = {"welfare_ratio": float("inf"), "guards_ok": False, "reason": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(output))
