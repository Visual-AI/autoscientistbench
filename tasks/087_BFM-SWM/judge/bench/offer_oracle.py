"""Trusted supplier oracle for one instance. Only the scorer constructs it; a policy reaches it over RPC.

Rules enforced here (any violation closes the instance and makes the whole evaluation invalid):
  * An offer is (supplier, price). A supplier accepts iff its private cost <= price; the answer is all that comes back.
  * Prices to a supplier never rise; a supplier that rejected is out for good and may not be offered again or declared.
  * `declare` commits the winner set once. A winner must have accepted its most recent offer, which is also its payment.
  * Total payment <= budget and v(S) - payment >= 0 are checked against the public value function.
"""
import numpy as np

BUDGET_TOL = 1e-9
SURPLUS_TOL = 1e-9


def load_graph(directory):
    """renum_node.txt + renum_edge.txt -> (n, offset, adjacency): out-neighbours of u are adjacency[offset[u]:offset[u+1]]."""
    rows = np.fromfile(f"{directory}/renum_node.txt", sep=" ", dtype=np.float64)
    if rows.size % 4 or rows.size == 0:
        raise ValueError("malformed node table")
    n = rows.size // 4
    edges = np.fromfile(f"{directory}/renum_edge.txt", sep=" ", dtype=np.int64)
    if edges.size % 2:
        raise ValueError("malformed edge list")
    src, dst = edges[0::2], edges[1::2]
    if src.size and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n):
        raise ValueError("edge endpoint out of range")
    offset = np.zeros(n + 1, dtype=np.int64)
    np.add.at(offset, src + 1, 1)
    offset = np.cumsum(offset)
    return n, offset, dst[np.argsort(src, kind="stable")]


def coverage(winners, offset, adjacency):
    """v(S) = |N(S)|, the authors' coverage valuation with unit node weights."""
    if len(winners) == 0:
        return 0
    return int(np.unique(np.concatenate([adjacency[offset[u]:offset[u + 1]] for u in winners])).size)


class OfferOracle:
    def __init__(self, cost, budget, offset, adjacency):
        self._cost = cost
        self._budget = float(budget)
        self._offset = offset
        self._adjacency = adjacency
        self._n = int(cost.size)
        self._last = np.full(self._n, np.inf)
        self._offered = np.zeros(self._n, dtype=bool)
        self._rejected = np.zeros(self._n, dtype=bool)
        self._record = None
        self.offers = 0
        self.violations = []

    def __call__(self, method, args, kwargs):
        try:
            if kwargs:
                raise ValueError("keyword arguments are unsupported")
            if self._record is not None:
                raise ValueError("instance is closed")
            if method == "offer" and len(args) == 3:
                return self._offer(*args)
            if method == "declare" and len(args) == 1:
                return self._declare(args[0])
            raise ValueError("unknown or malformed request")
        except Exception as exc:
            self.violations.append(type(exc).__name__ + ": " + str(exc))
            raise

    def _ids(self, value):
        ids = np.asarray(value)
        if ids.ndim != 1:
            raise ValueError("supplier ids must be a flat list")
        if ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        if ids.dtype.kind not in "iu":
            raise ValueError("supplier ids must be integers")
        ids = ids.astype(np.int64)
        if ids.min() < 0 or ids.max() >= self._n:
            raise ValueError("supplier id out of range")
        return ids

    def _offer(self, ids, prices, stop_on_accept):
        if type(stop_on_accept) is not bool:
            raise ValueError("stop_on_accept must be a boolean")
        ids = self._ids(ids)
        prices = np.asarray(prices)
        if prices.ndim != 1 or prices.size != ids.size:
            raise ValueError("ids and prices must have the same length")
        if ids.size == 0:
            return ""
        if prices.dtype.kind not in "iuf":
            raise ValueError("prices must be numbers")
        prices = prices.astype(np.float64)
        accepted = self._cost[ids] <= prices
        if stop_on_accept:
            # Offers after the first acceptance are never made, so they are neither recorded nor checked.
            hits = np.flatnonzero(accepted)
            if hits.size:
                stop = int(hits[0]) + 1
                ids, prices, accepted = ids[:stop], prices[:stop], accepted[:stop]
        if not np.all(np.isfinite(prices)) or np.any(prices < 0):
            raise ValueError("prices must be finite and non-negative")
        if np.unique(ids).size != ids.size:
            raise ValueError("a supplier may appear only once per batch")
        if np.any(self._rejected[ids]):
            raise ValueError("offer to a supplier that already rejected")
        if np.any(prices > self._last[ids]):
            raise ValueError("offered price rose")
        self._last[ids] = prices
        self._offered[ids] = True
        self._rejected[ids[~accepted]] = True
        self.offers += int(ids.size)
        return np.where(accepted, 49, 48).astype(np.uint8).tobytes().decode("ascii")

    def _declare(self, winners):
        winners = self._ids(winners)
        if np.unique(winners).size != winners.size:
            raise ValueError("a winner may be declared only once")
        if np.any(~self._offered[winners]):
            raise ValueError("winner never received an offer")
        if np.any(self._rejected[winners]):
            raise ValueError("winner rejected an offer")
        payments = self._last[winners]
        payment = float(payments.sum())
        if payment > self._budget + BUDGET_TOL:
            raise ValueError("total payment exceeds the budget")
        value = coverage(winners, self._offset, self._adjacency)
        if value - payment < -SURPLUS_TOL:
            raise ValueError("negative auctioneer surplus")
        cost = self._cost[winners]
        if np.any(cost > payments):
            raise RuntimeError("individual rationality broken; oracle bug")
        self._record = {"winners": [int(u) for u in winners], "payments": [float(p) for p in payments],
                        "payment": payment, "value": value, "cost": float(cost.sum()),
                        "welfare": value - float(cost.sum()), "offers": self.offers}
        return {"value": value, "payment": payment}

    def accepted_record(self):
        if self.violations or self._record is None:
            raise ValueError("protocol violation or no winner set declared")
        return self._record
