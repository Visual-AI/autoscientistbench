"""RTA estimator and analysis construction, verbatim from the reproduction.
Data loaders are replaced by the benchmark driver; estimator arithmetic is unchanged.
"""
import multiprocessing as mp
import os

import numpy as np
from pdda.rta import DiffusivityAnalysis, DiffusiveSurvivalModel, _histogram_excess_nll, _time_origin_histograms

def wall_allowed_mask(interval_widths, positions, domains) -> np.ndarray:
    """Union of the interval masks of every sampling window."""
    widths = np.asarray(interval_widths, dtype=np.float64)
    centers = np.asarray(positions, dtype=np.float64)
    allowed = np.zeros((widths.size, centers.size), dtype=bool)
    for lower, upper in domains:
        allowed |= (centers[None, :] - widths[:, None] / 2.0 >= lower) & (
            centers[None, :] + widths[:, None] / 2.0 <= upper
        )
    return allowed


def group_boundaries(cell_keys):
    """Split sorted cell keys into contiguous per-interval spans."""
    cell_keys = np.asarray(cell_keys)
    boundaries = np.flatnonzero(np.diff(cell_keys)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [cell_keys.size]))
    return starts, stops


_CELLS = None


def _recompute_cell(group_index):
    """-> (width index, center index, D, excess NLL, self-check result) of one interval; D is None if it has no residence time."""
    widths, centers, sampling_period, cell_keys, n_centers, lengths, counts, starts, stops, checked = _CELLS
    start, stop = starts[group_index], stops[group_index]
    key = int(np.asarray(cell_keys[start]))
    width_index, center_index = divmod(key, n_centers)
    span_lengths = lengths[start:stop]
    span_counts = counts[start:stop]

    # exact equivalent of pdda.rta._time_origin_histograms for a cell whose
    # segments all end in an observed exit (no terminal censoring)
    by_length = np.bincount(span_lengths, weights=span_counts.astype(np.float64))
    event_counts = np.cumsum(by_length[::-1])[::-1]
    event_counts[0] = 0.0

    equivalent = True
    if group_index in checked:
        reference, _ = _time_origin_histograms(
            dict(zip(span_lengths.tolist(), span_counts.tolist())), {}
        )
        equivalent = bool(np.array_equal(event_counts.astype(np.int64), reference))

    observed_total = int(event_counts.sum())
    if observed_total == 0:
        return width_index, center_index, None, None, equivalent
    residence = (
        sampling_period
        * float(np.dot(np.arange(event_counts.size), event_counts))
        / observed_total
    )
    if residence <= 0.0:
        return width_index, center_index, None, None, equivalent
    width = float(widths[width_index])
    center = float(centers[center_index])
    diffusivity = width**2 / (12.0 * residence)

    model = DiffusiveSurvivalModel(center - width / 2.0, center + width / 2.0,
                                   diffusivity)
    predicted = model.survival(
        np.arange(event_counts.size, dtype=np.float64) * sampling_period
    )
    nll = _histogram_excess_nll(
        event_counts.astype(np.int64),
        np.zeros(event_counts.size, dtype=np.int64),
        predicted,
    )
    return width_index, center_index, diffusivity, nll, equivalent


def recompute_from_arrays(
    interval_widths,
    positions,
    sampling_period: float,
    cell_keys,
    n_centers: int,
    observed_lengths,
    observed_counts,
    *,
    allowed=None,
    stored_diffusivities=None,
    stored_excess_nll=None,
    self_check_cells: int = 25,
    seed: int = 0,
):
    """Recompute D and the excess NLL for every analysed interval."""
    widths = np.asarray(interval_widths, dtype=np.float64)
    centers = np.asarray(positions, dtype=np.float64)
    shape = (widths.size, centers.size)
    diffusivities = np.full(shape, np.nan)
    excess_nll = np.full(shape, np.nan)

    lengths = np.asarray(observed_lengths, dtype=np.int64)
    counts = np.asarray(observed_counts, dtype=np.int64)
    starts, stops = group_boundaries(cell_keys)

    rng = np.random.default_rng(seed)
    n_groups = int(starts.size)
    checked = set(
        rng.choice(n_groups, size=min(self_check_cells, n_groups),
                   replace=False).tolist()
    ) if n_groups else set()
    equivalent = True
    n_intervals = 0

    # Cells share no state and draw no random numbers, so they can run in any order;
    # the forked workers read the (memory-mapped) inputs through _CELLS.
    global _CELLS
    _CELLS = (widths, centers, sampling_period, cell_keys, n_centers, lengths, counts, starts, stops, checked)
    cpus = max(1, int(os.environ.get("BENCH_CPUS", "1")))
    with mp.get_context("fork").Pool(min(cpus, max(1, n_groups))) as pool:
        for width_index, center_index, diffusivity, nll, same in pool.imap_unordered(_recompute_cell, range(n_groups)):
            equivalent &= same
            if diffusivity is None:
                continue
            diffusivities[width_index, center_index] = diffusivity
            excess_nll[width_index, center_index] = nll
            n_intervals += 1

    n_masked = 0
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != shape:
            raise ValueError("allowed mask must match the analysis grid")
        n_masked = int((~allowed & np.isfinite(diffusivities)).sum())
        diffusivities = np.where(allowed, diffusivities, np.nan)
        excess_nll = np.where(allowed, excess_nll, np.nan)

    diagnostics = {
        "recomputed_intervals": n_intervals,
        "intervals_inside_walls": int(allowed.sum()) if allowed is not None else None,
        "intervals_masked_by_walls": n_masked,
        "n_interval_widths": int(shape[0]),
        "n_positions": int(shape[1]),
        "sampling_period_ps": float(sampling_period),
        "histogram_expansion_equivalent_to_pdda": equivalent,
        "self_checked_cells": len(checked),
    }
    for label, stored, recomputed in (
        ("diffusivity", stored_diffusivities, diffusivities),
        ("excess_nll", stored_excess_nll, excess_nll),
    ):
        if stored is None:
            continue
        stored = np.asarray(stored, dtype=np.float64)
        finite = np.isfinite(stored) & np.isfinite(recomputed)
        diagnostics[f"max_abs_diff_vs_cache_{label}"] = (
            float(np.max(np.abs(recomputed[finite] - stored[finite])))
            if finite.any() else float("nan")
        )
        diagnostics[f"n_compared_{label}"] = int(finite.sum())
    return diffusivities, excess_nll, diagnostics


def rebuild_analysis(estimates: dict, diffusivities, excess_nll,
                     sampling_period: float) -> DiffusivityAnalysis:
    """Return a DiffusivityAnalysis carrying the recomputed estimates."""
    return DiffusivityAnalysis(
        positions=np.asarray(estimates["positions"], dtype=np.float64),
        interval_widths=np.asarray(estimates["interval_widths"], dtype=np.float64),
        diffusivities=np.asarray(diffusivities, dtype=np.float64),
        excess_nll=np.asarray(excess_nll, dtype=np.float64),
        n_time_origins=np.asarray(estimates["n_time_origins"], dtype=np.int64),
        first_step_exit_fractions=np.asarray(
            estimates["first_step_exit_fractions"], dtype=np.float64
        ),
        terminal_censored_fractions=np.asarray(
            estimates["terminal_censored_fractions"], dtype=np.float64
        ),
        sampling_period=float(sampling_period),
        histograms=None,
        diffusivity_sems=np.asarray(estimates["diffusivity_sems"], dtype=np.float64),
        symmetry_center=0.0,
    )
