"""Declarative set membership, interpreted without running candidate Python.

Every expression operates on one proposed mean. Vectorized evaluation merely
adds a leading sample axis; reductions never combine different proposed means.
"""
import json
import numpy as np

_UNARY = {"neg": np.negative, "abs": np.abs, "exp": np.exp, "log": np.log,
          "sqrt": np.sqrt, "tanh": np.tanh, "sin": np.sin, "cos": np.cos,
          "not": np.logical_not}
_BINARY = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
           "div": np.divide, "pow": np.power, "minimum": np.minimum,
           "maximum": np.maximum, "lt": np.less, "le": np.less_equal,
           "gt": np.greater, "ge": np.greater_equal, "eq": np.equal,
           "and": np.logical_and, "or": np.logical_or}
_REDUCE = {"sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max,
           "all": np.all, "any": np.any}


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate predicate key")
        out[key] = value
    return out


class Predicate:
    """A finite numeric DAG, with no file, import, random or private-label op."""
    def __init__(self, text, dimension):
        self.dimension = dimension
        spec = json.loads(str(text), object_pairs_hook=_object)
        if not isinstance(spec, dict) or set(spec) != {"nodes", "output"}:
            raise ValueError("predicate requires nodes and output")
        self.nodes = spec["nodes"]
        self.output = spec["output"]
        if not isinstance(self.nodes, list) or not self.nodes:
            raise ValueError("nonempty predicate nodes required")
        if type(self.output) is not int or not 0 <= self.output < len(self.nodes):
            raise ValueError("invalid output node")
        for i, node in enumerate(self.nodes):
            if not isinstance(node, dict):
                raise ValueError("invalid predicate node")
            op = node.get("op")
            if op == "input":
                fields = {"op"}
            elif op == "const":
                fields = {"op", "value"}
                value = np.asarray(node.get("value"))
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError("constants must be finite numeric arrays")
            else:
                arity = (1 if op in _UNARY or op in _REDUCE or op == "take"
                         else 2 if op in _BINARY or op == "matmul"
                         else 3 if op == "where" else None)
                if arity is None:
                    raise ValueError("unknown predicate operation")
                fields = {"op", "args"}
                args = node.get("args")
                if not isinstance(args, list) or len(args) != arity or any(
                        type(j) is not int or not 0 <= j < i for j in args):
                    raise ValueError("arguments must reference earlier nodes")
                if op in _REDUCE or op == "take":
                    fields.add("axis")
                    if type(node.get("axis")) is not int:
                        raise ValueError("integer feature axis required")
                if op == "take":
                    fields.add("indices")
                    ix = np.asarray(node.get("indices"))
                    if ix.dtype.kind not in "iu":
                        raise ValueError("integer indices required")
            if set(node) != fields:
                raise ValueError("unexpected predicate fields")

    def evaluate_many(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("predicate point dimensions")
        values = []
        # The boolean marks sample-dependent values. A sample axis cannot be
        # addressed by take/reduce or mixed into neural-network feature axes.
        with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
            for node in self.nodes:
                op = node["op"]
                if op == "input":
                    value, batched = points, True
                elif op == "const":
                    value, batched = np.asarray(node["value"]), False
                else:
                    pairs = [values[i] for i in node["args"]]
                    batched = any(b for _, b in pairs)
                    if op in _UNARY:
                        value = _UNARY[op](pairs[0][0])
                    elif op in _REDUCE or op == "take":
                        a, batched = pairs[0]
                        ndim = a.ndim - int(batched)
                        axis = node["axis"]
                        if not -ndim <= axis < ndim or ndim == 0:
                            raise ValueError("invalid feature axis")
                        axis = axis % ndim + int(batched)
                        value = (np.take(a, node["indices"], axis=axis) if op == "take"
                                 else _REDUCE[op](a, axis=axis))
                    elif op == "matmul":
                        a, ba = pairs[0]; b, bb = pairs[1]
                        # Per-point vectors may multiply a constant matrix.
                        # Constant matrices may also multiply each other.
                        if bb or b.ndim not in (1, 2) or (ba and a.ndim < 2):
                            raise ValueError("matmul needs a constant right matrix")
                        value = np.matmul(a, b)
                    else:
                        # Broadcast feature shapes without ever broadcasting
                        # the sample axis against a feature dimension.
                        rank = max(a.ndim - int(b) for a, b in pairs)
                        args = [a.reshape(((len(points),) if b else ((1,) if batched else ()))
                                          + (1,) * (rank - a.ndim + int(b))
                                          + a.shape[int(b):]) for a, b in pairs]
                        value = (np.where(*args) if op == "where" else _BINARY[op](*args))
                value = np.asarray(value)
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError("nonfinite predicate computation")
                values.append((value, batched))
        answer, batched = values[self.output]
        if answer.dtype.kind != "b" or answer.shape != ((len(points),) if batched else ()):
            raise ValueError("predicate output must be one Boolean per point")
        return answer if batched else np.full(len(points), answer, dtype=bool)


def membership(text, points, lower, upper):
    points = np.asarray(points, dtype=float)
    pred = Predicate(text, points.shape[1])
    bounds = np.all((points >= lower) & (points <= upper), axis=1)
    return bounds & pred.evaluate_many(points)


def volume(text, lower, upper, seed):
    """Original bounding-box estimator: five 25,000-point RNG replicates."""
    from geometry import VolumeCalculator
    lower = np.asarray(lower, dtype=float); upper = np.asarray(upper, dtype=float)
    pred = Predicate(text, len(lower))
    domain = VolumeCalculator.safe_positive_product(upper - lower)
    estimates = []
    for repeat in range(5):
        rng = np.random.default_rng(int(seed) + repeat)
        count = 0
        # Chunking preserves exactly the flattened original RNG sequence.
        for offset in range(0, 25000, 256):
            points = rng.random((min(256, 25000 - offset), len(lower)))
            points = points * (upper - lower) + lower
            count += int(np.count_nonzero(pred.evaluate_many(points)))
        estimates.append(domain * (count / 25000))
    return float(np.mean(estimates)), float(np.std(estimates))
