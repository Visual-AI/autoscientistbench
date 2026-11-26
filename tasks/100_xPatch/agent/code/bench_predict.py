"""bench entry point behind predict.sh: windows in -> forecasts out, one trained xPatch per horizon.

    python bench_predict.py --model DIR --input input.npz --output pred.out

DIR is what train.sh wrote: DIR/h<T>/<setting>/{checkpoint.pth, meta.json} for T in {96, 192, 336, 720}.
Input/output formats are specified in bench/io_spec.md. Every window is forecast from its own 96 input steps only.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from models import xPatch

BATCH = 2048


def load_horizon_model(model_dir, T):
    metas = sorted(glob.glob(os.path.join(model_dir, f"h{T}", "*", "meta.json")))
    if not metas:
        raise FileNotFoundError(f"no trained model for horizon {T} under {model_dir}/h{T}/")
    with open(metas[0]) as f:
        meta = json.load(f)
    args = argparse.Namespace(**meta["args"])
    model = xPatch.Model(args).float()
    state = torch.load(os.path.join(os.path.dirname(metas[0]), "checkpoint.pth"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    mean = np.asarray(meta["scaler_mean"], dtype=np.float64)
    scale = np.asarray(meta["scaler_scale"], dtype=np.float64)
    return model, args, mean, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if os.environ.get("BENCH_CPUS"):
        torch.set_num_threads(int(os.environ["BENCH_CPUS"]))

    d = np.load(a.input, allow_pickle=False)
    ids, horizons, x = d["id"], d["horizon"], d["x"]
    out = {}
    for T in sorted(set(int(h) for h in horizons.tolist())):
        model, args, mean, scale = load_horizon_model(a.model, T)
        m = horizons == T
        xs = (x[m].astype(np.float64) - mean) / scale        # the loader's train-split StandardScaler
        preds = []
        with torch.no_grad():
            for i in range(0, len(xs), BATCH):
                b = torch.from_numpy(xs[i:i + BATCH]).float()
                y = model(b)[:, -args.pred_len:, :]              # [batch, T, channels], standardized scale
                preds.append(y.cpu().numpy().astype(np.float64))
        p = np.concatenate(preds, 0) * scale + mean               # back to the raw scale of the input
        out[f"id_{T}"] = ids[m]
        out[f"pred_{T}"] = p.astype(np.float32)
        print(f"horizon {T}: {int(m.sum())} windows -> {out[f'pred_{T}'].shape}", flush=True)
    with open(a.output, "wb") as f:                               # file handle: numpy must not append ".npz"
        np.savez(f, **out)


if __name__ == "__main__":
    main()
