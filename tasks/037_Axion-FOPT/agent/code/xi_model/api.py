"""XiModel adapter to the certified model, initialized from neutral priors."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.environ.get("BENCH_BENCH_DIR", str(Path(__file__).resolve().parents[2] / "bench")))
from expressions import Model
from baseline import certificate

@dataclass
class PredictResult:
    xi: float
    xi_dm: float
    f_bm: float
    g_bm: float
    rc_mean: float
    kappa: float
    tp: float
    theta0: float
    vw: float
    hstar: float
    beta_over_h: float

    def to_dict(self):
        return asdict(self)

class XiModel:
    def __init__(self, parameters=None):
        self.model = Model(certificate(parameters))

    def predict(self, hstar, vw, theta0, beta_over_h, clip=True):
        values = self.model.predict(dict(hstar=hstar, vw=vw, theta0=theta0, beta_over_h=beta_over_h))
        if not clip:
            p = self.model.parameters
            values["xi"] = values["xi_dm"] * ((1-values["f_bm"])**p[3] + p[4]*values["v_factor"]*values["g_bm"])
        return PredictResult(**{key: values[key] for key in PredictResult.__dataclass_fields__})

def load_default_model():
    return XiModel()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hstar", required=True, type=float)
    parser.add_argument("--vw", required=True, type=float)
    parser.add_argument("--theta0", required=True, type=float)
    parser.add_argument("--beta-over-h", required=True, type=float)
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(XiModel().predict(args.hstar, args.vw, args.theta0, args.beta_over_h, not args.no_clip).to_dict()))
    return 0
