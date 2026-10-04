"""Causal combination of already constructed per-horizon target portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HOUR = pd.Timedelta(hours=1)
DIAGONAL_SHRINKAGE = 0.1
RISK_BUDGET_TOLERANCE = 1e-10
RISK_BUDGET_MAX_ITERATIONS = 10_000


@dataclass(frozen=True)
class HorizonCombinationResult:
    """Combined signed targets plus sleeve budgets and decision-level audit tables."""

    targets: pd.DataFrame
    budgets: pd.DataFrame
    audit: pd.DataFrame
    fits: pd.DataFrame


def _utc_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f"{name} must be a timezone-aware UTC DatetimeIndex")
    return index.tz_convert("UTC")


def _validate_panels(
    targets: dict[int, pd.DataFrame], returns: pd.DataFrame,
) -> tuple[list[int], list[str], pd.DatetimeIndex, dict[int, pd.DataFrame], pd.DataFrame]:
    if not isinstance(targets, dict) or not targets:
        raise ValueError("targets must map integer horizons to target DataFrames")
    if any(type(horizon) is not int or horizon <= 0 for horizon in targets):
        raise ValueError("target horizons must be positive integers")
    horizons = sorted(targets)
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a nonempty DataFrame")
    if returns.columns.has_duplicates or set(returns.columns) != set(horizons):
        raise ValueError("returns columns must match the integer target horizons exactly")

    normalized_targets: dict[int, pd.DataFrame] = {}
    reference_index: pd.DatetimeIndex | None = None
    reference_symbols: list[str] | None = None
    for horizon in horizons:
        panel = targets[horizon]
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            raise ValueError(f"targets[{horizon}] must be a nonempty DataFrame")
        if panel.columns.has_duplicates or not all(
            isinstance(symbol, str) and symbol for symbol in panel.columns
        ):
            raise ValueError(f"targets[{horizon}] must have unique nonempty symbol columns")
        index = _utc_index(panel.index, f"targets[{horizon}].index")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"targets[{horizon}].index must be unique and increasing")
        if reference_index is None:
            reference_index = index
            reference_symbols = sorted(panel.columns)
        elif not index.equals(reference_index):
            raise ValueError("all target panels must use the same hourly signal grid")
        elif set(panel.columns) != set(reference_symbols or []):
            raise ValueError("all target panels must use the same symbols")
        normalized = panel.loc[:, reference_symbols].copy()
        normalized.index = index
        values = normalized.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"targets[{horizon}] must contain finite target weights")
        if (np.abs(values) > 1 + 1e-12).any() or (np.abs(values).sum(axis=1) > 1 + 1e-12).any():
            raise ValueError(f"targets[{horizon}] exceed single-name or gross unit exposure")
        normalized_targets[horizon] = normalized

    assert reference_index is not None and reference_symbols is not None
    if len(reference_index) < 2 or not reference_index.equals(
        pd.date_range(reference_index[0], reference_index[-1], freq=HOUR, tz="UTC")
    ):
        raise ValueError("target panels must cover a complete hourly signal grid")
    return_index = _utc_index(returns.index, "returns.index")
    if return_index.has_duplicates or not return_index.is_monotonic_increasing:
        raise ValueError("returns.index must be unique and increasing")
    if not return_index.equals(reference_index):
        raise ValueError("returns and target panels must use the same hourly signal grid")
    normalized_returns = returns.loc[:, horizons].copy()
    normalized_returns.index = return_index
    try:
        return_values = normalized_returns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric realized account returns") from exc
    if np.isinf(return_values).any():
        raise ValueError("returns may contain NaN warmup values but cannot be infinite")
    return horizons, reference_symbols, reference_index, normalized_targets, normalized_returns


def _validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("policy must be a dictionary")
    allowed = {
        "method", "gross_exposure", "max_asset_weight", "mean_gate",
        "lookback_hours", "min_history", "covariance_shrinkage",
    }
    unknown = set(policy) - allowed
    if unknown:
        raise ValueError(f"unknown horizon policy fields: {sorted(unknown)}")
    required = {
        "method", "gross_exposure", "max_asset_weight", "mean_gate",
        "lookback_hours", "min_history", "covariance_shrinkage",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"horizon policy missing fields: {sorted(missing)}")
    method = policy["method"]
    if method not in ("fixed", "risk_budget"):
        raise ValueError("method must be 'fixed' or 'risk_budget'")
    for name in ("gross_exposure", "max_asset_weight"):
        value = policy[name]
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    mean_gate = policy["mean_gate"]
    if type(mean_gate) is not bool:
        raise ValueError("mean_gate must be a boolean")
    if (method == "risk_budget") != mean_gate:
        raise ValueError("positive historical mean gating is enabled only for risk_budget")
    if type(policy["lookback_hours"]) is not int or policy["lookback_hours"] <= 0:
        raise ValueError("lookback_hours must be a positive integer")
    if type(policy["min_history"]) is not int or policy["min_history"] < 2:
        raise ValueError("min_history must be an integer of at least two")
    if policy["min_history"] > policy["lookback_hours"]:
        raise ValueError("min_history cannot exceed lookback_hours")
    shrinkage = policy["covariance_shrinkage"]
    if not np.isfinite(shrinkage) or float(shrinkage) != DIAGONAL_SHRINKAGE:
        raise ValueError(f"covariance_shrinkage is frozen at {DIAGONAL_SHRINKAGE}")
    return {**policy, "mean_gate": mean_gate, "covariance_shrinkage": DIAGONAL_SHRINKAGE}


def _risk_budget_weights(covariance: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Solve equal risk contributions by cyclic coordinate minimization."""
    size = covariance.shape[0]
    if covariance.shape != (size, size) or size == 0 or not np.isfinite(covariance).all():
        raise ValueError("risk-budget covariance must be a finite nonempty square matrix")
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (np.diag(covariance) <= 0).any() or eigenvalues[0] <= 0:
        raise ValueError("risk-budget covariance must be positive definite with positive variances")
    budget = np.full(size, 1.0 / size)
    if size == 1:
        return np.ones(1), 0, 0.0
    allocation = np.full(size, 1.0 / np.sqrt(np.diag(covariance).mean()))
    error = np.inf
    for iteration in range(1, RISK_BUDGET_MAX_ITERATIONS + 1):
        for i in range(size):
            cross = float(covariance[i] @ allocation - covariance[i, i] * allocation[i])
            discriminant = cross * cross + 4.0 * covariance[i, i] * budget[i]
            root = np.sqrt(discriminant)
            allocation[i] = (
                (root - cross) / (2.0 * covariance[i, i])
                if cross < 0
                else 2.0 * budget[i] / (root + cross)
            )
        contributions = allocation * (covariance @ allocation)
        error = float(np.max(np.abs(contributions / contributions.sum() - budget)))
        if error <= RISK_BUDGET_TOLERANCE:
            weights = allocation / allocation.sum()
            return weights, iteration, error
    raise RuntimeError(
        f"risk-budget solver did not converge within {RISK_BUDGET_MAX_ITERATIONS} iterations"
    )


def combine_horizon_targets(
    targets: dict[int, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    policy: dict[str, Any],
) -> HorizonCombinationResult:
    """Combine horizon holdings with causal, shared sleeve budgets.

    Targets and realized account returns share an hourly signal-time index. The
    budget at timestamp ``t`` uses returns strictly before ``t``. Horizon
    positions are signed and netted by symbol before portfolio exposure caps.
    """
    horizons, symbols, index, target_panels, return_panel = _validate_panels(targets, returns)
    config = _validate_policy(policy)
    method = config["method"]
    lookback = config["lookback_hours"]
    min_history = config["min_history"]
    gross_limit = float(config["gross_exposure"])
    asset_limit = float(config["max_asset_weight"])

    budgets = pd.DataFrame(0.0, index=index, columns=horizons)
    combined = pd.DataFrame(0.0, index=index, columns=symbols)
    fit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for position, timestamp in enumerate(index):
        prior = return_panel.iloc[max(0, position - lookback):position]
        latest_returns = {
            horizon: (prior.index[prior[horizon].notna()][-1]
                      if prior[horizon].notna().any() else None)
            for horizon in horizons
        }
        latest_return_timestamp = max(
            (value for value in latest_returns.values() if value is not None),
            default=None,
        )
        prior_means: dict[int, float] = {}
        gate_active: dict[int, bool] = {}
        gate_reason: dict[int, str] = {}
        for horizon in horizons:
            values = prior[horizon].dropna()
            prior_means[horizon] = float(values.mean()) if len(values) else np.nan
            if not config["mean_gate"]:
                gate_active[horizon] = True
                gate_reason[horizon] = "gate_disabled"
            elif len(values) < min_history:
                gate_active[horizon] = False
                gate_reason[horizon] = "insufficient_history"
            elif prior_means[horizon] <= 0.0:
                gate_active[horizon] = False
                gate_reason[horizon] = "nonpositive_mean"
            else:
                gate_active[horizon] = True
                gate_reason[horizon] = "positive_mean"

        active = [horizon for horizon in horizons if gate_active[horizon]]
        solver_iterations = 0
        solver_error = np.nan
        covariance_values: np.ndarray | None = None
        status = "allocated"
        if not active:
            weights: dict[int, float] = {}
            status = "no_active_sleeves"
        elif method == "fixed":
            weights = {horizon: 1.0 / len(active) for horizon in active}
        else:
            complete = prior.loc[:, active].dropna(how="any")
            if len(complete) < min_history:
                weights = {}
                status = "insufficient_covariance_history"
            elif len(active) == 1:
                weights = {active[0]: 1.0}
            else:
                covariance_values = complete.to_numpy(dtype=float).T
                covariance = np.cov(covariance_values, ddof=1)
                covariance = (1.0 - config["covariance_shrinkage"]) * covariance + config["covariance_shrinkage"] * np.diag(
                    np.diag(covariance)
                )
                active_weights, solver_iterations, solver_error = _risk_budget_weights(covariance)
                weights = {horizon: float(weight) for horizon, weight in zip(active, active_weights)}

        for horizon in horizons:
            budget = weights.get(horizon, 0.0)
            budgets.loc[timestamp, horizon] = budget
            observations = int(prior[horizon].count())
            audit_rows.append({
                "timestamp": timestamp,
                "horizon": horizon,
                "method": method,
                "prior_observations": observations,
                "latest_return_timestamp": latest_returns[horizon],
                "prior_mean": prior_means[horizon],
                "gate_enabled": config["mean_gate"],
                "gate_active": gate_active[horizon],
                "gate_reason": gate_reason[horizon],
                "active": horizon in active,
                "budget": budget,
                "allocation_status": status,
            })

        raw = sum(
            budgets.loc[timestamp, horizon] * target_panels[horizon].loc[timestamp]
            for horizon in horizons
        )
        capped = raw.clip(lower=-asset_limit, upper=asset_limit)
        asset_cap_count = int((capped != raw).sum())
        pre_gross = float(capped.abs().sum())
        scale = min(1.0, gross_limit / pre_gross) if pre_gross > 0 else 1.0
        final = capped * scale
        combined.loc[timestamp] = final
        fit = {
            "timestamp": timestamp,
            "method": method,
            "active_horizons": tuple(active),
            "prior_complete_observations": int(
                prior.loc[:, active].dropna(how="any").shape[0]
            ) if active else 0,
            "latest_return_timestamp": latest_return_timestamp,
            "solver_iterations": solver_iterations,
            "solver_error": solver_error,
            "allocation_status": status,
            "pre_gross_after_asset_cap": pre_gross,
            "asset_cap_count": asset_cap_count,
            "gross_cap_scale": scale,
            "final_gross": float(final.abs().sum()),
        }
        if covariance_values is not None:
            sample_covariance = np.cov(covariance_values, ddof=1)
            shrunk_covariance = (
                (1.0 - config["covariance_shrinkage"]) * sample_covariance
                + config["covariance_shrinkage"] * np.diag(np.diag(sample_covariance))
            )
            for i, left in enumerate(active):
                for j, right in enumerate(active):
                    fit[f"cov_{left}_{right}"] = float(shrunk_covariance[i, j])
        fit_rows.append(fit)

    budgets.index.name = combined.index.name = "signal_timestamp"
    return HorizonCombinationResult(
        targets=combined,
        budgets=budgets,
        audit=pd.DataFrame(audit_rows),
        fits=pd.DataFrame(fit_rows),
    )
