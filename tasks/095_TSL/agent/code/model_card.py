"""Export a fitted tensorsl model as `model_card.json` (format `separable-product-card/v1`).

The card is the structure certificate that the judge's probe `bench/check_model.py` verifies (see bench/io_spec.md):
it lists every stage of the model as a pair (<= 2) of rank-1 products of univariate functions of the *raw* input
features, with coefficients, so that the predictions written by predict.sh can be reproduced from the card alone.

For TSL the prediction is (src/stage_predictor.rs: `extract_two_tensor_predictions_unscaled` + the OLS scalings;
src/forest.rs: stages are summed from a zero accumulator, i.e. no intercept)

    stage_s(x) = scaling_plus_s * lambda_plus_s  * prod_j b_sj(x_j) * exp(+d_sj(x_j))
               - scaling_minus_s * lambda_minus_s * prod_j b_sj(x_j) * exp(-d_sj(x_j))
    f(x)       = sum_s stage_s(x)

where b_sj, d_sj are step functions over the bins of feature j (Rust: partition_point(split <= x), i.e.
numpy.searchsorted(splits, x, side="right")). Each factor table therefore has len(splits) + 1 values.

If you replace the method, write your own exporter: any sum of <= 2 pairs of rank-1 products of univariate
functions of the raw features can be expressed in this format (factor kinds: "step" or piecewise-"linear").
"""
import numpy as np

FORMAT = "separable-product-card/v1"


def export_card(model, feature_names):
    """model: fitted tensorsl.sklearn.TSLRegressor (or tensorsl.TSL); feature_names: the input CSV feature columns."""
    n = len(feature_names)
    stages = []
    for s_idx, sp in enumerate(model.stage_predictors):
        g = sp.combined_grid_tensor
        splits, backbone, tilt = g.splits, g.backbone_values, g.tilt_values
        if not (len(splits) == len(backbone) == len(tilt) == n):
            raise ValueError(f"stage {s_idx}: expected factor tables for {n} features, got "
                             f"{len(splits)}/{len(backbone)}/{len(tilt)}")
        scaling_plus = 1.0 if sp.scaling_plus is None else float(sp.scaling_plus)
        scaling_minus = 0.0 if sp.scaling_minus is None else float(sp.scaling_minus)
        products = []
        for sign, coef in ((+1.0, scaling_plus * float(g.lambda_plus)), (-1.0, -scaling_minus * float(g.lambda_minus))):
            factors = []
            for j in range(n):
                cuts = np.asarray(splits[j], dtype=np.float64)
                b, d = np.asarray(backbone[j], dtype=np.float64), np.asarray(tilt[j], dtype=np.float64)
                if not (b.shape == d.shape and len(b) == len(cuts) + 1):
                    raise ValueError(f"stage {s_idx} feature {j}: factor table layout {len(cuts)} splits / {len(b)} values")
                values = b * np.exp(sign * d)
                factors.append({"feature": j, "kind": "step", "splits": cuts.tolist(), "values": values.tolist()})
            products.append({"coef": coef, "factors": factors})
        stages.append({"products": products})
    return {"format": FORMAT, "model": "tensorsl.TSL", "n_features": n, "feature_names": list(feature_names),
            "intercept": 0.0, "stages": stages}
