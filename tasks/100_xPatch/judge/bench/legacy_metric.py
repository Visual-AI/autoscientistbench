#!/usr/bin/env python3
"""Mean standardized MSE across four horizons, subject to mean MAE <= 0.4275 on test and <= 0.72 on val.
The frozen input provides 96 history steps per window. Locality and lookback are
checked by the operator's subset probe and patch audit. Shapes and id sets are
checked by validate.py. Guard violations are reported as positive Infinity.
"""
import json
import sys
from pathlib import Path

import numpy as np

HORIZONS = (96, 192, 336, 720)
CHANNELS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
# StandardScaler statistics of the ETTh1 training rows [0, 8640) (population std, as sklearn computes it)
TRAIN_MEAN = np.array([7.937742245549186, 2.0210386567124052, 5.079770601170338, 0.7461858799814872,
                       2.7817623863756924, 0.788453123560763, 17.128261698278603])
TRAIN_STD = np.array([5.812749409078979, 2.0901046503913965, 5.518793579069257, 1.9263792741210177,
                      1.023522659511737, 0.6302366362134583, 9.176491025127563])
MAE_MAX = {"test": 0.4275, "val": 0.72}
WORST_SCORE = float("inf")


def standardize(a):
    return (np.asarray(a, dtype=np.float64) - TRAIN_MEAN) / TRAIN_STD


def split_of(labels_path):
    return "val" if "val" in Path(labels_path).parts[-3:-1] else "test"


def main(pred_path, labels_path):
    pred = np.load(pred_path, allow_pickle=False)
    lab = np.load(labels_path, allow_pickle=False)
    mse, mae = {}, {}
    for T in HORIZONS:
        ids = lab[f"id_{T}"]
        y = standardize(lab[f"y_{T}"])
        pid = pred[f"id_{T}"]
        order = {k: i for i, k in enumerate(pid.tolist())}
        idx = np.array([order[k] for k in ids.tolist()])
        p = standardize(pred[f"pred_{T}"][idx])
        if p.shape != y.shape:
            sys.exit(f"horizon {T}: prediction shape {p.shape} != label shape {y.shape}")
        if not np.isfinite(p).all():
            print(json.dumps({"mse": WORST_SCORE, "guard_mae_ok": False, "reason": "nonfinite predictions"}))
            return
        err = p - y
        mse[str(T)] = float(np.mean(err ** 2))
        mae[str(T)] = float(np.mean(np.abs(err)))
    mean_mse = sum(mse.values()) / len(HORIZONS)
    mean_mae = sum(mae.values()) / len(HORIZONS)
    mae_max = MAE_MAX[split_of(labels_path)]
    guard_ok = mean_mae <= mae_max
    score = mean_mse if guard_ok else WORST_SCORE
    print(json.dumps({"mse": score, "mae": mean_mae, "mse_unguarded": mean_mse, "guard_mae_ok": guard_ok,
                      "mae_max": mae_max, "mse_per_horizon": mse, "mae_per_horizon": mae}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
