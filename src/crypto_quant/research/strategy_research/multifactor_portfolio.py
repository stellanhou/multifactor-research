"""Convert decision-time composite scores into graduated portfolio targets.

These functions operate at the caller's rebalance timestamps. They do not fit
models or submit trades; their signed targets use the existing account engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multifactor_account import _utc_index


def _scores(scores: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        raise ValueError("scores must be a nonempty DataFrame")
    if scores.columns.has_duplicates or not all(isinstance(s, str) and s for s in scores.columns):
        raise ValueError("score columns must be unique nonempty symbols")
    index = _utc_index(scores.index, "scores.index")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("score timestamps must be unique and increasing")
    values = scores.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("scores may be missing but cannot be infinite")
    return pd.DataFrame(values, index=index, columns=scores.columns)


def _limits(gross_exposure: float, max_asset_weight: float) -> None:
    if not np.isfinite(gross_exposure) or not 0 < gross_exposure <= 1:
        raise ValueError("gross_exposure must be in (0, 1]")
    if not np.isfinite(max_asset_weight) or not 0 < max_asset_weight <= 1:
        raise ValueError("max_asset_weight must be in (0, 1]")


def generate_tapered_targets(
    scores: pd.DataFrame,
    *,
    side_count: int,
    gross_exposure: float,
    max_asset_weight: float,
) -> pd.DataFrame:
    """Allocate each side by linear rank weights ``K, ..., 1``.

    Long and short seats are disjoint and equally numerous. When fewer than
    ``2*K`` assets are eligible, unavailable outer seats remain cash; remaining
    seats are not enlarged. Per-asset clipping also leaves the excess in cash.
    """
    frame = _scores(scores)
    _limits(gross_exposure, max_asset_weight)
    if type(side_count) is not int or side_count <= 0:
        raise ValueError("side_count must be a positive integer")
    profile = np.arange(side_count, 0, -1, dtype=float)
    weights = np.minimum(profile / profile.sum() * gross_exposure / 2, max_asset_weight)
    result = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    for timestamp, row in frame.iterrows():
        available = [s for s in row.index if pd.notna(row[s])]
        count = min(side_count, len(available) // 2)
        longs = sorted(available, key=lambda s: (-row[s], s))[:count]
        shorts = sorted([s for s in available if s not in longs], key=lambda s: (row[s], s))[:count]
        result.loc[timestamp, longs] = weights[:count]
        result.loc[timestamp, shorts] = -weights[:count]
    return result


def _capped_proportional(values: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """Redistribute capped allocations among the remaining positive scores."""
    result = np.zeros(len(values), dtype=float)
    remaining = np.arange(len(values))
    while len(remaining):
        proposed = values[remaining] / values[remaining].sum() * budget
        capped = proposed > cap
        if not capped.any():
            result[remaining] = proposed
            break
        result[remaining[capped]] = cap
        budget -= cap * int(capped.sum())
        remaining = remaining[~capped]
    return result


def generate_score_weighted_targets(
    scores: pd.DataFrame,
    *,
    gross_exposure: float,
    max_asset_weight: float,
) -> pd.DataFrame:
    """Center scores and allocate capped, proportional long/short budgets.

    Both legs receive the same budget, bounded by their available asset capacity.
    A flat score cross-section is cash. All eligible nonzero centered scores can
    receive weight; a tiny cross-sectional dispersion does not imply low exposure.
    """
    frame = _scores(scores)
    _limits(gross_exposure, max_asset_weight)
    result = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    for timestamp, row in frame.iterrows():
        available = row.dropna()
        if available.empty:
            continue
        centered = available.to_numpy() - available.mean()
        if not np.isfinite(centered).all():
            raise ValueError("centered scores must be finite")
        positive = np.flatnonzero(centered > 0)
        negative = np.flatnonzero(centered < 0)
        budget = min(gross_exposure / 2, len(positive) * max_asset_weight,
                     len(negative) * max_asset_weight)
        if budget == 0:
            continue
        for indices, sign in ((positive, 1), (negative, -1)):
            weights = _capped_proportional(sign * centered[indices], budget, max_asset_weight)
            result.loc[timestamp, available.index[indices]] = sign * weights
    return result


def smooth_target_weights(
    targets: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Apply a causal target-weight EMA starting from cash.

    Each step is ``(1-alpha)*previous_target + alpha*new_target``. An ordinary
    zero target decays; an ineligible asset exits immediately and resets its
    history. To force a model to cash, mark its entire eligibility row false.
    No renormalization follows the EMA, because that would undo its inertia.
    """
    frame = _scores(targets)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("targets must be finite")
    if not isinstance(eligible, pd.DataFrame) or not eligible.index.equals(frame.index) \
            or not eligible.columns.equals(frame.columns):
        raise ValueError("eligibility must have exactly the target index and columns")
    if not all(dtype == bool for dtype in eligible.dtypes):
        raise ValueError("eligibility must contain booleans without missing values")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    values, mask = frame.to_numpy(), eligible.to_numpy()
    if np.any((values != 0) & ~mask):
        raise ValueError("an ineligible asset cannot have a nonzero desired target")
    result = np.zeros_like(values)
    previous = np.zeros(values.shape[1], dtype=float)
    for i, current in enumerate(values):
        previous = (1 - alpha) * previous + alpha * current
        previous[~mask[i]] = 0.0
        result[i] = previous
    return pd.DataFrame(result, index=frame.index, columns=frame.columns)
