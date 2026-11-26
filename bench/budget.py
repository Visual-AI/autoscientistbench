"""One deadline shared by every stage and seed of an evaluation."""
import math
import time


def positive_seconds(value, name="budget"):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return seconds


def remaining_time(deadline_ts=None, limit=None):
    """A subprocess timeout, or None when no deadline/operator smoke cap exists."""
    limits = []
    if limit is not None:
        limits.append(positive_seconds(limit, "timeout"))
    if deadline_ts is not None:
        deadline = float(deadline_ts)
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("overall evaluation budget exhausted")
        limits.append(remaining)
    return min(limits) if limits else None


def deadline_env(env, deadline_ts):
    result = dict(env or {})
    if deadline_ts is not None:
        result["BENCH_EVAL_DEADLINE_TS"] = str(deadline_ts)
    return result
