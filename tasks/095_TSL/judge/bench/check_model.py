#!/usr/bin/env python3
"""Verify a sum of at most four separable products of arbitrary univariate factors.
The original two-stage TSL is one representation of this scientific rank budget.
Constant offsets consume one rank-one term. Predictions must reproduce the frozen
model card with relative error <1e-6. Executable factors see only one scalar input.

Usage: check_model.py MODEL_DIR [--input CSV] [--pred CSV] [--code DIR] [--tol 1e-6] [--report FILE]
  MODEL_DIR   the `train.sh --out` directory (must contain model_card.json)
  --input     feature CSV (default: $BENCH_DATA_DIR/val/input.csv)
  --pred      predictions of predict.sh for --input; if omitted, `bash <code>/predict.sh` is run on --input
              (<code> = --code, else $BENCH_CODE_DIR, else ../code next to this bench/ directory)
Prints a one-line JSON report and PASS / FAIL <reason>. Exit 0 = PASS, 1 = FAIL, 2 = usage / IO error.
The optional executable-factor path uses the task-owned private runtime.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FORMAT = "separable-product-card/v1"
GENERIC_FORMAT = "separable-product-card/v2"
MAX_RANK = 4
FEATURE_NAMES = ['longitude','latitude','housingMedianAge','totalRooms',
                 'totalBedrooms','population','households','medianIncome']
sys.path.insert(0,str(Path(__file__).resolve().parent))
from factor_runtime import FactorRuntime
from private_eval import remaining_time


class Reject(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Reject(msg)


def finite_list(v, what, min_len=1):
    need(isinstance(v, list) and len(v) >= min_len, f"{what}: expected a list of >= {min_len} numbers")
    arr = np.asarray(v, dtype=np.float64)
    need(arr.ndim == 1 and np.all(np.isfinite(arr)), f"{what}: non-finite or non-numeric entries")
    return arr


def eval_factor(fac, x, where, factor_runner=None):
    need(isinstance(fac, dict), f"{where}: factor must be an object")
    kind = fac.get("kind", "step")
    if kind == "step":
        splits = finite_list(fac.get("splits", []), f"{where}.splits", min_len=0)
        values = finite_list(fac.get("values"), f"{where}.values")
        need(len(values) == len(splits) + 1, f"{where}: step factor needs len(values) == len(splits) + 1")
        need(len(splits) == 0 or np.all(np.diff(splits) > 0), f"{where}: splits must be strictly increasing")
        return values[np.searchsorted(splits, x, side="right")]
    if kind == "linear":
        knots = finite_list(fac.get("knots"), f"{where}.knots")
        values = finite_list(fac.get("values"), f"{where}.values")
        need(len(knots) == len(values), f"{where}: linear factor needs len(knots) == len(values)")
        need(len(knots) == 1 or np.all(np.diff(knots) > 0), f"{where}: knots must be strictly increasing")
        return np.interp(x, knots, values)
    if kind == 'executable':
        need(factor_runner is not None, 'executable factors require a frozen factors.zip artifact')
        return factor_runner.evaluate(fac,x)
    raise Reject(f"{where}: unknown factor kind {kind!r} (step | linear | executable)")


def reconstruct(card, X, feature_names, max_stages=None, max_products=None, factor_runner=None):
    """-> (prediction from the card, summary dict); raises Reject on a structural violation."""
    need(isinstance(card, dict) and card.get("format") in (FORMAT,GENERIC_FORMAT), 'unknown separable model card format')
    n = X.shape[1]
    need(n==8 and list(feature_names)==FEATURE_NAMES,'input must preserve the eight raw feature columns')
    need(int(card.get("n_features", n)) == n, f"card n_features={card.get('n_features')} but input has {n} feature columns")
    if card.get("feature_names") is not None:
        need(list(card["feature_names"]) == list(feature_names),
             f"card feature_names {card['feature_names']} != input columns {feature_names}")
    intercept = float(card.get("intercept", 0.0))
    need(math.isfinite(intercept), 'intercept must be finite')
    if card['format']==GENERIC_FORMAT:
        need(isinstance(card.get('terms'),list),'card.terms must be a list')
        stages=[{'products':card['terms']}] if card['terms'] else []
    else:
        stages = card.get("stages")
    need(isinstance(stages, list), "card.stages must be a list")
    total_rank=int(intercept!=0.0)
    for stage in stages:
        need(isinstance(stage,dict) and isinstance(stage.get('products'),list),'stage needs a products list')
        total_rank+=len(stage['products'])
    need(total_rank<=MAX_RANK,f'{total_rank} separable terms exceed the scientific rank budget {MAX_RANK}')
    pred = np.full(X.shape[0], intercept, dtype=np.float64)
    per_stage = []
    for s, stage in enumerate(stages):
        need(isinstance(stage, dict) and isinstance(stage.get("products"), list), f"stage {s}: needs a `products` list")
        prods = stage["products"]
        for p, prod in enumerate(prods):
            where = f"stage {s} product {p}"
            need(isinstance(prod, dict), f"{where}: must be an object")
            coef = prod.get("coef", 1.0)
            need(isinstance(coef, (int, float)) and math.isfinite(float(coef)), f"{where}: coef must be a finite number")
            factors = prod.get("factors")
            need(isinstance(factors, list), f"{where}: needs a `factors` list")
            term = np.full(X.shape[0], float(coef), dtype=np.float64)
            for q, fac in enumerate(factors):
                fw = f"{where} factor {q}"
                j = fac.get("feature") if isinstance(fac, dict) else None
                need(isinstance(j, int) and 0 <= j < n, f"{fw}: `feature` must be a raw feature index in 0..{n - 1}")
                # Repeated univariate factors on a coordinate multiply into one
                # univariate function, so they do not increase separation rank.
                term *= eval_factor(fac, X[:, j], fw, factor_runner)
            need(np.all(np.isfinite(term)), f"{where}: non-finite product values")
            pred += term
        per_stage.append(len(prods))
    need(np.all(np.isfinite(pred)), 'nonfinite reconstructed prediction')
    return pred, {"n_stages": len(stages), "products_per_stage": per_stage,
                  'separation_rank_bound':total_rank,'maximum_separation_rank':MAX_RANK}


def read_input(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = [c for c in (r.fieldnames or []) if c != "id"]
        need("id" in (r.fieldnames or []) and cols, f"{path}: header must be id,<features>")
        ids, rows = [], []
        for row in r:
            ids.append(row["id"])
            rows.append([float(row[c]) for c in cols])
    need(ids and len(ids)==len(set(ids)),'input IDs must be nonempty and unique')
    return ids, cols, np.asarray(rows, dtype=np.float64)


def read_pred(path):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        need(r.fieldnames and "id" in r.fieldnames and "pred" in r.fieldnames, f"{path}: header must contain id,pred")
        for row in r:
            need(row['id'] not in out,'duplicate prediction ID')
            out[row["id"]] = float(row["pred"])
    return out


def run_predict(code_dir, model_dir, input_path, out_path):
    cmd = ["bash", str(Path(code_dir) / "predict.sh"), "--model", str(model_dir), "--input", str(input_path), "--output", str(out_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=remaining_time())
    if p.returncode != 0 or not Path(out_path).exists():
        raise Reject(f"predict.sh failed rc={p.returncode}: {(p.stderr or p.stdout)[-800:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--input", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--code", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    # Accepted for old operator invocations; they cannot alter the fixed rank budget.
    ap.add_argument("--max-stages", type=int, default=None,help=argparse.SUPPRESS)
    ap.add_argument("--max-products", type=int, default=None,help=argparse.SUPPRESS)
    ap.add_argument("--report", default=None, help="also write the JSON report here")
    a = ap.parse_args()
    if not math.isfinite(a.tol) or not 0 < a.tol <= 1e-6:
        print('reconstruction tolerance must be positive and at most 1e-6',file=sys.stderr)
        return 2

    model_dir = Path(a.model_dir)
    input_path = a.input or (os.environ.get("BENCH_DATA_DIR") and str(Path(os.environ["BENCH_DATA_DIR"]) / "val" / "input.csv"))
    if not input_path:
        print("check_model: need --input or BENCH_DATA_DIR", file=sys.stderr)
        return 2
    report = {"model_dir": str(model_dir), "input": str(input_path), "pass": False}
    tmp = None
    try:
        card_path = model_dir / "model_card.json"
        need(card_path.exists(), f"{card_path} missing (train.sh must write the structure certificate)")
        try:
            card = json.loads(card_path.read_text())
        except ValueError as e:
            raise Reject(f"model_card.json is not valid JSON: {e}")
        ids, cols, X = read_input(input_path)
        need(np.all(np.isfinite(X)), "input has non-finite values")
        pred_path = a.pred
        if pred_path is None:
            code_dir = a.code or os.environ.get("BENCH_CODE_DIR") or str(Path(__file__).resolve().parent.parent / "code")
            tmp = tempfile.NamedTemporaryFile(prefix="check_model_", suffix=".csv", delete=False)
            tmp.close()
            pred_path = tmp.name
            run_predict(code_dir, model_dir, input_path, pred_path)
        pred_map = read_pred(pred_path)
        need(set(pred_map)==set(ids),'prediction IDs must match the input exactly')
        pred = np.array([pred_map[i] for i in ids], dtype=np.float64)
        need(np.all(np.isfinite(pred)), "predictions contain NaN/inf")
        with FactorRuntime(model_dir,card) as factor_runner:
            recon, summary = reconstruct(card, X, cols, factor_runner=factor_runner)
            summary['private_factor_calls']=factor_runner.calls
        rel = np.abs(pred - recon) / np.maximum(1.0, np.abs(pred))
        report.update(summary, n_rows=int(len(ids)), max_rel_err=float(rel.max()), max_abs_err=float(np.abs(pred - recon).max()),
                      tol=a.tol, pred=str(pred_path))
        need(report["max_rel_err"] < a.tol, f"reconstruction error {report['max_rel_err']:.3e} >= {a.tol:g}: predictions "
                                            'are not those of the declared separable model')
        report["pass"] = True
        report["reason"] = (f"separation rank bound {summary['separation_rank_bound']} <= {MAX_RANK}, "
                            f"max relative reconstruction error {report['max_rel_err']:.3e} < {a.tol:g}")
    except Reject as e:
        report["reason"] = str(e)
    except (OSError, ValueError, RuntimeError) as e:
        report["reason"] = f"{type(e).__name__}: {e}"
        print(json.dumps(report))
        print(f"ERROR {report['reason']}")
        return 2
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    print(("PASS " if report["pass"] else "FAIL ") + report["reason"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
