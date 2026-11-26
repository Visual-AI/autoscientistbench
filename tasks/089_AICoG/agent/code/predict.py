"""Write the frozen representation in the task's prediction format. The input holds only the node count."""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

with open(args.input, encoding="utf-8") as f:
    n_nodes = int(json.load(f)["n_nodes"])
with np.load(f"{args.model}/embedding.npz", allow_pickle=False) as d:
    z, b, decoder = d["z"], d["b"], d["decoder"]
if len(z) != n_nodes or len(b) != n_nodes:
    raise SystemExit(f"model has {len(z)} nodes, input asks for {n_nodes}")
with open(args.output, "wb") as f:     # through a handle: np.savez would append .npz to a bare path
    np.savez(f, z=z, b=b, decoder=decoder)
