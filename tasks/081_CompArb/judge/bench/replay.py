"""Frozen purchase replay, shared by validate.py and scorer.py. Standard library only.

Formulas are the paper's (Section 3) and the authors' notebooks/utils.py:

  pass@k  = 1 - C(n-m, k) / C(n, k)              unbiased estimate from n attempts with m successes
  k       = floor(dollars / mean_cost)           attempts a dollar budget affords
  spend   = integral_0^d (1 - pass@floor(x/c)) dx   survival identity: expected dollars when stopping at first success
  cascade = segments tried in order, each reached only if every earlier segment failed (Eq. 8)

The integral is computed exactly on the step function, so no budget grid is involved here.
"""
import json
import math
import re

MODELS = ("gpt5mini", "deepseek")
BUDGET_CAP_USD = 1.0
CAP_TOLERANCE_USD = 1e-9
MAX_LEVELS = 250
MAX_FILE_BYTES = 32 * 1024 * 1024
HEADER = "level,instance_id,plan"
MONOTONE_TOLERANCE_USD = 1e-12

_LEVEL = re.compile(r"(?:0|[1-9][0-9]{0,3})\Z")
_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?\Z")


class FamilyError(ValueError):
    """The prediction file is not a valid purchase family."""


class ModelCurve:
    """pass@k and expected spend for one model on one issue."""

    def __init__(self, attempts, mean_cost):
        self.n = len(attempts)
        self.m = int(sum(attempts))
        self.cost = float(mean_cost)
        if self.n < 1 or not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("bad outcome record")
        self.top = self.n - self.m + 1 if self.m > 0 else 0   # pass@top == 1
        self.pass_table = [self._pass_at(t) for t in range(self.top + 1)]
        self.survival_sum = [0.0]                             # survival_sum[t] = sum_{s<t} (1 - pass@s)
        for t in range(self.top):
            self.survival_sum.append(self.survival_sum[-1] + 1.0 - self.pass_table[t])

    def _pass_at(self, k):
        n, m = self.n, self.m
        if k < 1:
            return 0.0
        if k >= n or n - m < k:
            return 1.0 if m > 0 else 0.0
        fail = 1.0
        for j in range(n - m + 1, n + 1):
            fail *= 1.0 - k / j
        return 1.0 - fail

    def pass_and_spend(self, dollars):
        """-> (probability solved within `dollars`, expected dollars spent stopping at first success)."""
        if self.m == 0:
            return 0.0, dollars
        t = int(dollars / self.cost)
        if t >= self.top:
            return 1.0, self.cost * self.survival_sum[self.top]
        p = self.pass_table[t]
        return p, self.cost * self.survival_sum[t] + (dollars - t * self.cost) * (1.0 - p)


def parse_plan(text):
    """'gpt5mini:0.08;deepseek:0.2' -> (('gpt5mini', 0.08), ('deepseek', 0.2)). '' buys nothing."""
    if text == "":
        return ()
    plan, total = [], 0.0
    for token in text.split(";"):
        model, sep, usd = token.partition(":")
        if not sep or model not in MODELS:
            raise FamilyError(f"unknown model in plan segment {token[:40]!r}")
        if any(model == seen for seen, _ in plan):
            raise FamilyError(f"model {model} appears twice in one plan")
        if not _NUMBER.match(usd):
            raise FamilyError(f"bad dollar amount {usd[:40]!r}")
        dollars = float(usd)
        if not math.isfinite(dollars) or dollars <= 0:
            raise FamilyError(f"dollar amount must be finite and positive: {usd[:40]!r}")
        plan.append((model, dollars))
        total += dollars
    if total > BUDGET_CAP_USD + CAP_TOLERANCE_USD:
        raise FamilyError(f"plan spends {total} > cap {BUDGET_CAP_USD} USD")
    return tuple(plan)


def read_issue_ids(input_path):
    """Issue ids of an input.jsonl file."""
    ids = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["instance_id"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("input has no issues or duplicate issues")
    return ids


def read_family(path, issue_ids):
    """Parse strictly. -> list (index = level) of {instance_id: plan}. Anything but the three columns is rejected."""
    with open(path, "rb") as f:
        raw = f.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise FamilyError(f"file larger than {MAX_FILE_BYTES} bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FamilyError("file is not plain ASCII") from exc
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != HEADER:
        raise FamilyError(f"first line must be exactly {HEADER!r}")
    known = set(issue_ids)
    levels = {}
    for number, line in enumerate(lines[1:], 2):
        parts = line.split(",")
        if len(parts) != 3:
            raise FamilyError(f"line {number}: expected exactly 3 columns")
        level, iid, plan_text = parts
        if not _LEVEL.match(level):
            raise FamilyError(f"line {number}: bad level {level[:20]!r}")
        if iid not in known:
            raise FamilyError(f"line {number}: unknown instance_id {iid[:60]!r}")
        bucket = levels.setdefault(int(level), {})
        if iid in bucket:
            raise FamilyError(f"line {number}: duplicate row for level {level}, {iid}")
        bucket[iid] = parse_plan(plan_text)
    count = len(levels)
    if count < 1 or count > MAX_LEVELS:
        raise FamilyError(f"{count} levels; need 1 to {MAX_LEVELS}")
    if set(levels) != set(range(count)):
        raise FamilyError("levels must be the consecutive integers 0..L-1")
    for level in range(count):
        missing = known - set(levels[level])
        if missing:
            raise FamilyError(f"level {level}: {len(missing)} issue(s) missing, e.g. {sorted(missing)[0]}")
    return [levels[level] for level in range(count)]


def check_nested(family):
    """Every level must extend the one before it, issue by issue: per-model dollars never fall and models keep their
    relative order. So the family is one path through budgets, and its solve rate cannot fall as the level rises."""
    for iid in family[0]:
        previous, before = {}, set()
        for level, plans in enumerate(family):
            plan = plans[iid]
            dollars = dict(plan)
            for model, spent in previous.items():
                if dollars.get(model, 0.0) < spent - MONOTONE_TOLERANCE_USD:
                    raise FamilyError(f"level {level}, {iid}: {model} dollars fall below level {level - 1}")
            order = [model for model, _ in plan]
            for a in range(len(order)):
                for b in range(a + 1, len(order)):
                    if (order[b], order[a]) in before:
                        raise FamilyError(f"level {level}, {iid}: model order differs from an earlier level")
                    before.add((order[a], order[b]))
            previous = dollars


def load_family(path, issue_ids):
    family = read_family(path, issue_ids)
    check_nested(family)
    return family


def replay_plan(plan, curves):
    """-> (solve probability, expected dollars) of one issue under one plan (Eq. 8 with the survival identity)."""
    reach, spent = 1.0, 0.0
    for model, dollars in plan:
        p, s = curves[model].pass_and_spend(dollars)
        spent += reach * s
        reach *= 1.0 - p
    return 1.0 - reach, spent


def family_points(family, curves_by_issue):
    """-> per-level mean solve rate and mean dollars per issue."""
    solve, cost = [], []
    for plans in family:
        s, c = 0.0, 0.0
        for iid, plan in plans.items():
            a, b = replay_plan(plan, curves_by_issue[iid])
            s += a
            c += b
        solve.append(s / len(plans))
        cost.append(c / len(plans))
    return solve, cost


def envelope_cost(solve, cost, target):
    """Least mean dollars of a mix of at most two levels (or buying nothing) with mean solve rate >= target.

    This reads the lower convex envelope of the (solve rate, cost) points. A linear objective under one linear constraint
    has an optimum with at most two support points, so checking pairs is exact. -> (cost or inf, (low, high, weight)),
    with low or high = None for the buy-nothing point.
    """
    points = [(None, 0.0, 0.0)] + [(i, s, c) for i, (s, c) in enumerate(zip(solve, cost))]
    above = [p for p in points if p[1] >= target]
    below = [p for p in points if p[1] < target]
    if not above:
        return math.inf, None
    high = min(above, key=lambda p: p[2])
    best, mix = high[2], (high[0], high[0], 1.0)
    for lo in below:
        for hi in above:
            w = (target - lo[1]) / (hi[1] - lo[1])
            value = lo[2] * (1.0 - w) + hi[2] * w
            if value < best:
                best, mix = value, (lo[0], hi[0], w)
    return best, mix


def plain_interpolated_cost(solve, cost, target):
    """The authors' reading without the envelope: log-cost interpolation along the level order (utils.interp_logy)."""
    if not solve or solve[-1] < target:
        return math.inf
    for i, s in enumerate(solve):
        if s >= target:
            if i == 0 or s == target or solve[i - 1] == s or cost[i - 1] <= 0:
                return cost[i]
            w = (target - solve[i - 1]) / (s - solve[i - 1])
            return math.exp(math.log(cost[i - 1]) * (1.0 - w) + math.log(cost[i]) * w)
    return math.inf


def load_labels(path):
    """-> (target solve rate, benchmark size, {instance_id: {model: ModelCurve}})."""
    with open(path, encoding="utf-8") as f:
        labels = json.load(f)
    curves = {}
    for iid, record in labels["issues"].items():
        curves[iid] = {m: ModelCurve(record[m]["attempts"], record[m]["mean_cost"]) for m in MODELS}
    return float(labels["target_solve_rate"]), int(labels["benchmark_size"]), curves
