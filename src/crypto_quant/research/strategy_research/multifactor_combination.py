"""Symmetric factor whitening and causal rolling Rank-ICIR combination.

Bar timestamps label opens. A factor at t uses the completed t bar; its
next-open h-hour return ends at t+h+1. Only labels ending at or before the
current signal bar's open enter its weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import validate_universe
from crypto_quant.research.factor_mining.contracts import require


@dataclass(frozen=True)
class FactorCombination:
    orthogonalized: pd.DataFrame
    cross_sections: pd.DataFrame
    rank_ic: pd.DataFrame
    mature_ic_count: pd.DataFrame
    icir: pd.DataFrame
    weights: pd.DataFrame
    directions: pd.DataFrame
    score: pd.Series


def validate_combination(policy: dict) -> None:
    require(isinstance(policy, dict) and set(policy) == {
        "method", "window_hours", "min_periods", "negative_icir",
    }, "combination fields differ from the schema")
    require(policy["method"] == "symmetric_orthogonalized_rolling_icir",
            "unsupported factor combination method")
    require(type(policy["window_hours"]) is int and policy["window_hours"] >= 2,
            "ICIR window_hours must be an integer >= 2")
    require(type(policy["min_periods"]) is int
            and 2 <= policy["min_periods"] <= policy["window_hours"],
            "ICIR min_periods must be in [2, window_hours]")
    require(policy["negative_icir"] == "flip_factor",
            "negative_icir must be flip_factor")


def symmetric_orthogonalize(values: pd.DataFrame, common_mask: pd.Series):
    """Apply X (X'X/n)^{+,-1/2} per common cross-section, without imputation.

    SVD evaluates the symmetric pseudoinverse square root without squaring
    the condition number. In deficient rank, covariance is a projector onto
    the retained factor subspace, not an identity matrix over all columns.
    """
    membership = validate_universe(common_mask)
    require(isinstance(values, pd.DataFrame) and values.index.equals(membership.index)
            and len(values.columns) > 0 and not values.columns.has_duplicates,
            "orthogonalization needs aligned, unique factor columns")
    require(not np.isinf(values.to_numpy(dtype=float)).any(),
            "factor values cannot be infinite")
    output = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    diagnostics = []
    for timestamp, positions in values.groupby(level="timestamp", sort=True).indices.items():
        positions = np.asarray(positions)
        included = positions[membership.iloc[positions].to_numpy()]
        x = values.iloc[included].to_numpy(dtype=float)
        require(np.isfinite(x).all(), "common orthogonalization sample contains missing factors")
        n, k = x.shape
        if n < 3:
            diagnostics.append({"timestamp": timestamp, "common_symbols": n, "factor_count": k,
                                "effective_rank": 0, "status": "insufficient_cross_section"})
            continue
        x = x - x.mean(axis=0)
        u, singular, vt = np.linalg.svd(x / np.sqrt(n), full_matrices=False)
        tolerance = singular[0] * max(n, k) * np.finfo(float).eps
        retained = singular > tolerance
        rank = int(retained.sum())
        # The small, thin product needs no BLAS dispatch. Einsum also avoids
        # spurious floating-point warnings from the local macOS BLAS runtime.
        whitened = np.sqrt(n) * np.einsum("ik,kj->ij", u[:, retained], vt[retained])
        require(np.isfinite(whitened).all(), "symmetric orthogonalization produced nonfinite values")
        output.iloc[included] = whitened
        diagnostics.append({"timestamp": timestamp, "common_symbols": n, "factor_count": k,
                            "effective_rank": rank,
                            "status": "full_rank" if rank == k else "rank_deficient" if rank else "zero_rank"})
    return output, pd.DataFrame(diagnostics).set_index("timestamp")


def _rank_ic(values: pd.DataFrame, opens: pd.Series, horizon: int) -> pd.DataFrame:
    prices = opens.unstack("symbol")
    entry, exit_price = prices.shift(-1), prices.shift(-(horizon + 1))
    returns = (exit_price / entry - 1).where((entry > 0) & (exit_price > 0))
    labels = returns.stack(future_stack=True).reindex(values.index)
    rows = []
    for timestamp, positions in values.groupby(level="timestamp", sort=True).indices.items():
        positions = np.asarray(positions)
        sample = values.iloc[positions]
        target = labels.iloc[positions]
        good = sample.notna().all(axis=1) & target.notna() & np.isfinite(target)
        if int(good.sum()) < 3:
            rows.append(pd.Series(np.nan, index=values.columns, name=timestamp))
            continue
        ranks = sample.loc[good].rank(method="average").to_numpy()
        y = target.loc[good].rank(method="average").to_numpy()
        ranks -= ranks.mean(axis=0)
        y -= y.mean()
        denominator = np.sqrt(np.square(ranks).sum(axis=0) * np.square(y).sum())
        coefficient = np.divide(ranks.T @ y, denominator,
                                out=np.full(len(values.columns), np.nan), where=denominator > 0)
        rows.append(pd.Series(coefficient, index=values.columns, name=timestamp))
    result = pd.DataFrame(rows)
    result.index.name = "timestamp"
    return result


def rolling_icir_combine(orthogonalized: pd.DataFrame, cross_sections: pd.DataFrame,
                         opens: pd.Series, *, horizon_hours: int, policy: dict,
                         history: tuple[pd.DataFrame, pd.Series] | None = None) -> FactorCombination:
    """Weight supplied factor values using only matured hourly ICs.

    Orthogonalization is applied by the caller. The four-arm ablation also
    supplies standardized factors directly to isolate the weighting effect.

    History contains the preceding stage's transformed factors and open
    prices, including pending labels. It takes precedence over repeated
    factor warmup so stage splitting matches one uninterrupted calculation.
    Negative ICIR flips the transformed factor's direction; absolute ICIR
    supplies its positive weight. Zero IC variance, insufficient history,
    and undefined IC leave that factor inactive. With no active weight the
    score is unavailable (cash).
    """
    validate_combination(policy)
    require(type(horizon_hours) is int and horizon_hours in {1, 4, 24}, "unsupported IC horizon")
    require(opens.index.equals(orthogonalized.index), "IC opening prices and factors differ in index")
    values, prices = orthogonalized, opens
    if history is not None:
        previous, previous_prices = history
        require(previous.index.equals(previous_prices.index)
                and list(previous.columns) == list(values.columns), "IC history schema differs")
        overlap = previous_prices.index.intersection(prices.index)
        require(np.allclose(previous_prices.loc[overlap], prices.loc[overlap], rtol=0, atol=0,
                            equal_nan=True), "IC history prices differ from the current raw observations")
        fresh = ~values.index.isin(previous.index)
        values = pd.concat([previous, values.loc[fresh]]).sort_index()
        prices = pd.concat([previous_prices, prices.loc[~prices.index.isin(previous_prices.index)]]).sort_index()
    times = values.index.get_level_values("timestamp").unique().sort_values()
    require(times.equals(pd.date_range(times[0], times[-1], freq="h", name="timestamp")),
            "rolling IC needs a complete hourly grid")
    rank_ic = _rank_ic(values, prices, horizon_hours)
    mature = rank_ic.shift(horizon_hours + 1)
    window = mature.rolling(policy["window_hours"], min_periods=policy["min_periods"])
    std = window.std(ddof=1)
    icir = window.mean().div(std.where(std > 0))
    require(not np.isinf(icir.to_numpy()).any(), "rolling ICIR produced an infinite value")
    active = icir.fillna(0.0)
    directions = np.sign(active)
    total = active.abs().sum(axis=1)
    weights = active.abs().div(total.where(total > 0), axis=0).fillna(0.0)
    current = values.reindex(orthogonalized.index)
    current_times = current.index.get_level_values("timestamp")
    row_weights = (weights * directions).reindex(current_times).to_numpy()
    score = pd.Series((current.to_numpy() * row_weights).sum(axis=1),
                      index=current.index, name="score")
    score = score.where(total.reindex(current_times).to_numpy() > 0)
    # Diagnostics describe the newly computed cross-section. Mark repeated
    # warmup rows whose factor values came from the preceding stage instead.
    cross_sections = cross_sections.copy()
    cross_sections["history_reused"] = False
    if history is not None:
        previous_times = history[0].index.get_level_values("timestamp").unique()
        reused = cross_sections.index.intersection(previous_times)
        cross_sections.loc[reused, "history_reused"] = True
    return FactorCombination(current, cross_sections, rank_ic,
                             mature.rolling(policy["window_hours"], min_periods=1).count(),
                             icir, weights, directions, score)
