"""Python client for the private offer service; holds no private state. C++ policies use offer_client.h."""
import json
import os
from pathlib import Path
from private_rpc import call


def read_instance(path):
    """The worker's --input file -> {"id", "graph", "budget", "algorithm_index"}."""
    return json.loads(Path(path).read_text())


def graph_dir(instance):
    return Path(os.environ["BENCH_DATA_DIR"]) / instance["graph"]


def load_graph(instance):
    """-> (n, offset, adjacency) as numpy arrays; numpy is imported only when this is called."""
    import numpy as np
    directory = graph_dir(instance)
    n = np.fromfile(directory / "renum_node.txt", sep=" ").size // 4
    edges = np.fromfile(directory / "renum_edge.txt", sep=" ", dtype=np.int64)
    src, dst = edges[0::2], edges[1::2]
    offset = np.zeros(n + 1, dtype=np.int64)
    np.add.at(offset, src + 1, 1)
    return n, np.cumsum(offset), dst[np.argsort(src, kind="stable")]


def offer(ids, prices, stop_on_accept=False):
    """Offer prices[i] to ids[i], in order. -> str of '1' (accepted) / '0' (rejected), one per offer actually made.

    With stop_on_accept=True the offers after the first acceptance are not made and the string is shorter."""
    return call("offer", ids, prices, bool(stop_on_accept))


def declare(winners):
    """Commit the winner set (once). -> {"value": v(S), "payment": total payment}."""
    return call("declare", winners)
