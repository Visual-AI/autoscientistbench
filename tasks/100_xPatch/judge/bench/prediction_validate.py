#!/usr/bin/env python3
"""Output-format check (same file the judge runs). Usage: validate.py <pred_file> <input_file>; exit 0 = ok.

The prediction file must be an npz archive (any file name) with, for every horizon T present in the input,
`id_<T>` (the ids of that horizon's windows, each exactly once) and `pred_<T>` of shape (n_T, T, 7), floating
point, finite. See bench/io_spec.md.
"""
import sys
import zipfile

import numpy as np

N_CHANNELS = 7


def main(pred_path, input_path):
    inp = np.load(input_path, allow_pickle=False)
    ids_in, hz_in = inp["id"], inp["horizon"]
    if not zipfile.is_zipfile(pred_path):
        sys.exit("prediction file is not an npz archive (write it with np.savez on an open file handle)")
    try:
        pred = np.load(pred_path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot load prediction file: {e}")
    keys = set(pred.files)
    total = 0
    for T in sorted(set(int(h) for h in hz_in.tolist())):
        want = set(ids_in[hz_in == T].tolist())
        for k in (f"id_{T}", f"pred_{T}"):
            if k not in keys:
                sys.exit(f"missing array {k!r} (horizon {T}); arrays present: {sorted(keys)}")
        try:
            pid = pred[f"id_{T}"]
            p = pred[f"pred_{T}"]
        except Exception as e:  # noqa: BLE001
            sys.exit(f"cannot read arrays for horizon {T}: {e}")
        got = pid.tolist()
        if len(got) != len(set(got)):
            sys.exit(f"horizon {T}: duplicate ids")
        missing, extra = want - set(got), set(got) - want
        if missing or extra:
            sys.exit(f"horizon {T}: id mismatch, {len(missing)} missing, {len(extra)} extra "
                     f"(e.g. {sorted(missing)[:3]} / {sorted(extra)[:3]})")
        if p.shape != (len(got), T, N_CHANNELS):
            sys.exit(f"horizon {T}: pred_{T} has shape {p.shape}, expected {(len(got), T, N_CHANNELS)}")
        if not np.issubdtype(p.dtype, np.floating):
            sys.exit(f"horizon {T}: pred_{T} must be floating point, got {p.dtype}")
        if not np.isfinite(p).all():
            sys.exit(f"horizon {T}: pred_{T} contains NaN/inf")
        total += len(got)
    print(f"ok: {total} windows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
