"""Inert scalar expression certificates: never eval or execute candidate Python."""
import ast
import csv
import json
import math
import operator
import re
from pathlib import Path
from physics import Physics

FORMAT = "axion-expression-v1"
INPUT = ["id", "theta0", "hstar", "beta_over_h", "vw"]
LABEL = INPUT + ["xi", "xi_sem"]
FUNCTIONS = {"abs": abs, "min": min, "max": max, "log": math.log, "log1p": math.log1p,
             "exp": math.exp, "expm1": math.expm1, "sqrt": math.sqrt, "tanh": math.tanh,
             "sin": math.sin, "cos": math.cos}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}
FEATURES = {"theta0", "hstar", "beta_over_h", "vw", "tp", "tau", "fanh", "finf", "pilot",
            "gamma_vw", "pi", "e", "tiny", "sum_floor", "power_floor", "a_exp", "xi_floor", "p_mink", "kappa_eps",
            "fanh_eps", "fanh_alpha", "finf_eps", "finf_alpha"}
GEOMETRY = {"f_bm", "g_bm", "rc_mean", "a_ratio", "v_factor"}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite scalar required")
    return float(value)


def _object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key")
        out[k] = v
    return out


def load(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("certificate exceeds size bound")
    return json.loads(path.read_text(), object_pairs_hook=_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_table(path, labels=False):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != (LABEL if labels else INPUT):
            raise ValueError("invalid input columns")
        rows = list(reader)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate row ids")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["id"]):
            raise ValueError("invalid row id")
        for key in (LABEL if labels else INPUT)[1:]:
            row[key] = finite(float(row[key]))
        if not (0 <= row["theta0"] < math.pi and row["hstar"] > 0 and row["beta_over_h"] > 0 and 0 < row["vw"] <= 1):
            raise ValueError("input outside physical domain")
        if labels and not (row["xi"] > 0 and row["xi_sem"] >= 0):
            raise ValueError("invalid target")
    return rows


def parse_expression(source, names):
    if not isinstance(source, str) or not 1 <= len(source) <= 2048:
        raise ValueError("invalid expression size")
    node = ast.parse(source, mode="eval").body
    if sum(1 for _ in ast.walk(node)) > 256:
        raise ValueError("expression too complex")
    def check(n, depth=0):
        if depth > 40:
            raise ValueError("expression nesting too deep")
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or type(n.value) not in (int, float) or n.value not in (0, 1, 2, 3):
                raise ValueError("only structural literals 0,1,2,3 are allowed; use a fitted scalar")
        elif isinstance(n, ast.Name):
            if n.id not in names:
                raise ValueError("unknown expression variable")
        elif isinstance(n, ast.BinOp) and type(n.op) in OPS:
            check(n.left, depth+1); check(n.right, depth+1)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            check(n.operand, depth+1)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
            arity = len(n.args)
            if not (2 <= arity <= 8 if n.func.id in ("min", "max") else arity == 1):
                raise ValueError("invalid function arity")
            for arg in n.args:
                check(arg, depth+1)
        else:
            raise ValueError("unsupported expression syntax")
    check(node)
    return node


def evaluate(node, values):
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        value = values[node.id]
    elif isinstance(node, ast.BinOp):
        value = OPS[type(node.op)](evaluate(node.left, values), evaluate(node.right, values))
    elif isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, values) * (-1 if isinstance(node.op, ast.USub) else 1)
    else:
        value = FUNCTIONS[node.func.id](*(evaluate(arg, values) for arg in node.args))
    return finite(value)


class Model:
    def __init__(self, certificate, physics=None):
        if not isinstance(certificate, dict) or set(certificate) != {"format", "parameters", "pre", "post"} or certificate["format"] != FORMAT:
            raise ValueError("invalid model certificate")
        pars = certificate["parameters"]
        if not isinstance(pars, list) or len(pars) > 7:
            raise ValueError("at most seven global fitted scalars are allowed")
        self.parameters = [finite(p) for p in pars]
        self.certificate = certificate
        names = FEATURES | {f"p{i}" for i in range(len(pars))}
        protected = FEATURES | GEOMETRY | set(FUNCTIONS) | {f"p{i}" for i in range(7)}
        self.pre, self.post = [], []
        for stage in ("pre", "post"):
            entries = certificate[stage]
            if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
                raise ValueError("invalid expression list")
            if stage == "post":
                names |= GEOMETRY
            parsed = []
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("each assignment is [name,expression]")
                name, source = entry
                if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in names or name in protected:
                    raise ValueError("invalid or repeated expression assignment")
                parsed.append((name, parse_expression(source, names)))
                names.add(name)
            setattr(self, stage, parsed)
            if stage == "pre" and "kappa" not in names:
                raise ValueError("pre expressions must define kappa")
        if not {"xi", "xi_dm"} <= names:
            raise ValueError("post expressions must define xi and xi_dm")
        if {"xi", "xi_dm"} & {name for name, _ in self.pre}:
            raise ValueError("xi and xi_dm belong in post expressions")
        self.physics = physics or Physics()

    def predict(self, row, parameters=None):
        pars = self.parameters if parameters is None else parameters
        if len(pars) != len(self.parameters):
            raise ValueError("parameter count changed")
        values = self.physics.features(row)
        values.update({f"p{i}": finite(float(p)) for i, p in enumerate(pars)})
        for name, node in self.pre:
            values[name] = evaluate(node, values)
        values.update(self.physics.complete(values, values["kappa"]))
        for name, node in self.post:
            values[name] = evaluate(node, values)
        if values["xi"] <= 0 or values["xi_dm"] <= 0:
            raise ValueError("positive enhancement and delayed component required")
        return values


def validate_prediction(prediction, rows):
    if not isinstance(prediction, dict) or set(prediction) != {"format", "model", "rows"} or prediction["format"] != FORMAT:
        raise ValueError("invalid prediction envelope")
    Model(prediction["model"], physics=True)
    values = prediction["rows"]
    if not isinstance(values, list) or len(values) != len(rows):
        raise ValueError("prediction row count mismatch")
    for result, row in zip(values, rows):
        if not isinstance(result, dict) or set(result) != {"id", "xi"} or result["id"] != row["id"]:
            raise ValueError("prediction ids must match the input in order")
        if finite(result["xi"]) <= 0:
            raise ValueError("positive prediction required")
    return values


def physical_checks(model):
    def predict(h, vw, theta, beta):
        return model.predict(dict(hstar=h, vw=vw, theta0=theta, beta_over_h=beta))
    result = predict(.2, .5, 1.2, 6.)
    if not (result["xi"] > 0 and result["xi_dm"] > 0 and 0 < result["f_bm"] < 1 and
            result["g_bm"] >= 0 and result["rc_mean"] > 0 and result["tp"] > 0):
        raise ValueError("physical smoke: sensible result")
    high = predict(5., .5, 1.2, 6.)["xi"]
    if not 1 <= high < 2:
        raise ValueError("physical smoke: high-H floor and limit")
    xs = [predict(h, .7, 1., 4.)["xi"] for h in (1e-3, 1e-2, 1e-1, 1.)]
    if not (all(a > b for a, b in zip(xs, xs[1:])) and xs[0] > 10):
        raise ValueError("physical smoke: monotonic H decline and enhancement")
    low1, low2 = (predict(h, .7, .1, 4.)["xi"] for h in (1e-6, 1e-5))
    slope = math.log(low1 / low2) / math.log(1e-6 / 1e-5)
    if not abs(slope + .5) < .05:
        raise ValueError("physical smoke: low-H slope")
    return dict(sensible=True, high_h_floor=True, monotonic_h=True, low_h_slope=slope)
