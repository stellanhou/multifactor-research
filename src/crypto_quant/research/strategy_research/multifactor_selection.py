"""Causal factor-combination routes for hourly multifactor research.

Signals at bar ``t`` use the completed factor row at ``t``. A training label
for source bar ``s`` enters an update at ``t`` only when its exit open at
``s + horizon_hours + 1`` is at or before ``t``. Regression targets are
demeaned across the common complete-factor cross-section at each source bar.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

HOUR = pd.Timedelta(hours=1)
POLICY_FIELDS = {
    "fit_window_hours",
    "min_train_periods",
    "refit_every_hours",
    "min_cross_section_symbols",
    "icir_window_hours",
    "icir_min_periods",
    "correlation_window_hours",
    "cluster_abs_correlation_threshold",
    "inner_validation_hours",
    "min_inner_validation_periods",
    "alpha_grid",
    "l1_ratio",
    "pool_capacity",
    "pool_search_budget",
    "min_objective_improvement",
}
ROUTES = ("equal", "rolling_icir", "cluster", "ridge", "elastic_net", "pool")
STEPWISE_ROUTE = "stepwise"
SUPPORTED_ROUTES = ROUTES + (STEPWISE_ROUTE,)

# Fixed deterministic solver bound; convergence warnings become errors.
ELASTIC_NET_MAX_ITER = 5_000_000
ELASTIC_NET_TOL = 1e-8


@dataclass(frozen=True)
class SelectionResult:
    """Requested full-panel scores, shared readiness, and model-update audits."""

    scores: dict[str, pd.Series]
    fits: list[dict[str, Any]]
    shared_model_ready: pd.Series


@dataclass(frozen=True)
class _RidgeValidationStats:
    """Full-column sufficient statistics for repeated ridge subset proposals."""

    train_gram: np.ndarray
    train_rhs: np.ndarray
    validation_gram: np.ndarray
    validation_rhs: np.ndarray
    validation_target_mean_square: float
    valid: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_policy(policy: dict[str, Any]) -> None:
    _require(isinstance(policy, dict) and set(policy) == POLICY_FIELDS,
             "selection policy fields differ from the schema")
    for name in ("fit_window_hours", "min_train_periods", "refit_every_hours",
                 "min_cross_section_symbols", "icir_window_hours", "icir_min_periods",
                 "correlation_window_hours", "inner_validation_hours",
                 "min_inner_validation_periods", "pool_capacity", "pool_search_budget"):
        value = policy[name]
        _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    _require(policy["min_cross_section_symbols"] >= 3,
             "min_cross_section_symbols must be at least 3")
    _require(2 <= policy["icir_min_periods"] <= policy["icir_window_hours"],
             "icir_min_periods must be in [2, icir_window_hours]")
    _require(policy["min_inner_validation_periods"] <= policy["inner_validation_hours"],
             "min_inner_validation_periods cannot exceed inner_validation_hours")
    threshold = policy["cluster_abs_correlation_threshold"]
    _require(not isinstance(threshold, bool) and np.isfinite(threshold) and 0 < threshold <= 1,
             "cluster_abs_correlation_threshold must be in (0, 1]")
    ratio = policy["l1_ratio"]
    _require(not isinstance(ratio, bool) and isinstance(ratio, (int, float))
             and np.isfinite(ratio) and 0 < ratio < 1,
             "l1_ratio must be in (0, 1) for a strongly convex elastic-net objective")
    improvement = policy["min_objective_improvement"]
    _require(not isinstance(improvement, bool) and isinstance(improvement, (int, float))
             and np.isfinite(improvement) and improvement >= 0,
             "min_objective_improvement must be finite and non-negative")
    alphas = policy["alpha_grid"]
    _require(isinstance(alphas, list) and len(alphas) > 0,
             "alpha_grid must be a nonempty list")
    _require(all(not isinstance(alpha, bool) and isinstance(alpha, (int, float))
                 and np.isfinite(alpha) and alpha > 0 for alpha in alphas),
             "alpha_grid values must be finite and positive")
    _require(len(set(alphas)) == len(alphas), "alpha_grid values must be unique")


def _validate_inputs(values: pd.DataFrame, opens: pd.Series, horizon_hours: int) -> tuple[pd.DatetimeIndex, list[str]]:
    _require(type(horizon_hours) is int and horizon_hours in {1, 4, 24},
             "horizon_hours must be one of 1, 4, or 24")
    _require(isinstance(values, pd.DataFrame) and isinstance(opens, pd.Series),
             "values must be a DataFrame and opens must be a Series")
    _require(isinstance(values.index, pd.MultiIndex) and values.index.nlevels == 2
             and list(values.index.names) == ["timestamp", "symbol"],
             "values must use a (timestamp, symbol) MultiIndex")
    _require(values.index.is_unique and values.index.is_monotonic_increasing,
             "factor index must be unique and sorted")
    _require(opens.index.equals(values.index), "opening prices and factors differ in index")
    _require(len(values.columns) > 0 and not values.columns.has_duplicates,
             "at least one unique factor column is required")
    _require(all(isinstance(column, str) and column for column in values.columns),
             "factor names must be nonempty strings")
    try:
        matrix = values.to_numpy(dtype=float)
        price_values = opens.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("factor values and opening prices must be numeric") from exc
    _require(not np.isinf(matrix).any(), "factor values cannot be infinite")
    _require(not np.isinf(price_values).any(), "opening prices cannot be infinite")
    finite_prices = price_values[np.isfinite(price_values)]
    _require(np.all(finite_prices > 0), "finite opening prices must be positive")

    timestamps = values.index.get_level_values("timestamp")
    symbols = values.index.get_level_values("symbol")
    _require(isinstance(timestamps, pd.DatetimeIndex) and timestamps.tz is not None,
             "timestamps must be timezone-aware")
    _require(str(timestamps.tz) == "UTC", "timestamps must be timezone-aware UTC")
    timestamps = timestamps.tz_convert("UTC")
    unique_times = timestamps.unique().sort_values()
    unique_symbols = list(pd.Index(symbols).unique().sort_values())
    _require(len(unique_times) > 0 and len(unique_symbols) > 0, "factor panel cannot be empty")
    _require(unique_times.equals(pd.date_range(unique_times[0], unique_times[-1], freq="h", name="timestamp")),
             "factor panel timestamps must form a complete hourly grid")
    expected = pd.MultiIndex.from_product([unique_times, unique_symbols], names=["timestamp", "symbol"])
    _require(values.index.equals(expected), "factor panel must contain every symbol at every timestamp")
    return unique_times, unique_symbols


def _make_labels(values: pd.DataFrame, opens: pd.Series, timestamps: pd.DatetimeIndex,
                 symbols: list[str], horizon_hours: int, min_symbols: int) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    prices = opens.unstack("symbol").reindex(index=timestamps, columns=symbols)
    entry = prices.shift(-1)
    exit_price = prices.shift(-(horizon_hours + 1))
    returns = exit_price.div(entry).sub(1.0).where((entry > 0) & (exit_price > 0))
    result: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    for timestamp in timestamps:
        factor_row = values.xs(timestamp, level="timestamp").to_numpy(dtype=float)
        target = returns.loc[timestamp].to_numpy(dtype=float)
        common = np.isfinite(factor_row).all(axis=1) & np.isfinite(target)
        if int(common.sum()) < min_symbols:
            continue
        x = factor_row[common]
        y = target[common]
        y = y - y.mean()
        result[pd.Timestamp(timestamp)] = (x, y)
    return result


def _rolling_icir_weights(dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                          timestamps: pd.DatetimeIndex, factor_names: list[str],
                          horizon: int, policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_ic = pd.DataFrame(np.nan, index=timestamps, columns=factor_names, dtype=float)
    for timestamp, (x, y) in dataset.items():
        ranks = pd.DataFrame(x, columns=factor_names).rank(method="average").to_numpy()
        target_rank = pd.Series(y).rank(method="average").to_numpy()
        ranks -= ranks.mean(axis=0, keepdims=True)
        target_rank -= target_rank.mean()
        denominator = np.sqrt(np.square(ranks).sum(axis=0) * np.square(target_rank).sum())
        coefficients = np.divide(
            np.einsum("ni,n->i", ranks, target_rank), denominator,
            out=np.full(len(factor_names), np.nan), where=denominator > 0,
        )
        rank_ic.loc[timestamp] = coefficients
    matured = rank_ic.shift(horizon + 1)
    rolling = matured.rolling(policy["icir_window_hours"],
                              min_periods=policy["icir_min_periods"])
    standard_deviation = rolling.std(ddof=1)
    icir = rolling.mean().div(standard_deviation.where(standard_deviation > 0))
    active = icir.fillna(0.0)
    directions = np.sign(active)
    absolute = active.abs()
    total = absolute.sum(axis=1)
    weights = absolute.div(total.where(total > 0), axis=0).fillna(0.0)
    return weights, directions


def _matured_window(data_times: pd.DatetimeIndex, fit_time: pd.Timestamp, horizon: int,
                    policy: dict[str, Any]) -> list[pd.Timestamp]:
    mature_cutoff = fit_time - (horizon + 1) * HOUR
    window_start = fit_time - policy["fit_window_hours"] * HOUR
    return [pd.Timestamp(t) for t in data_times[(data_times >= window_start) & (data_times <= mature_cutoff)]]


def _window_split(eligible: list[pd.Timestamp], fit_time: pd.Timestamp, horizon: int,
                  policy: dict[str, Any]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    mature_cutoff = fit_time - (horizon + 1) * HOUR
    validation_start = mature_cutoff - (policy["inner_validation_hours"] - 1) * HOUR
    validation = [t for t in eligible if t >= validation_start]
    # A training label must exit before the first validation source bar, leaving
    # a full horizon-plus-one-hour purge at the chronological split.
    train_cutoff = validation_start - (horizon + 1) * HOUR
    training = [t for t in eligible if t < train_cutoff]
    return training, validation


def _stack_sample(dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                  timestamps: list[pd.Timestamp], columns: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for timestamp in timestamps:
        x, y = dataset[timestamp]
        if columns is not None:
            x = x[:, columns]
        # Target and predictors are centered within each source cross-section;
        # this removes any residual mean caused by unavailable labels.
        x = x - x.mean(axis=0, keepdims=True)
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, 0 if columns is None else len(columns))), np.empty(0)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = np.einsum("ni,nj->ij", x, x) / len(y)
    rhs = np.einsum("ni,n->i", x, y) / len(y)
    return np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)


def _elastic_net_coefficients(x: np.ndarray, y: np.ndarray, alpha: float,
                              l1_ratio: float) -> np.ndarray:
    """Solve elastic net through its exact augmented-data Lasso equivalent.

    For ``n`` samples and ``p`` factors, append ``sqrt(n*alpha*(1-l1_ratio))*I``
    to the design and zero targets to y. Lasso uses ``n+p`` rows, so its L1
    penalty is ``alpha*l1_ratio*n/(n+p)``; multiplying that objective by
    ``(n+p)/n`` gives exactly the original elastic-net objective. Its seeded
    random coordinate order handles singular panels reproducibly. The reported
    per-row dual gap must meet ``tol * ||y_aug||^2 / (n+p)``; scaling back by
    ``(n+p)/n`` gives the same relative certificate for the original objective.
    """
    n, p = x.shape
    augmented_x = np.zeros((n + p, p), dtype=float)
    augmented_x[:n] = x
    augmented_x[n:] = np.sqrt(n * alpha * (1.0 - l1_ratio)) * np.eye(p)
    augmented_y = np.concatenate([y, np.zeros(p, dtype=float)])
    estimator = Lasso(
        alpha=alpha * l1_ratio * n / (n + p),
        fit_intercept=False,
        precompute=True,
        max_iter=ELASTIC_NET_MAX_ITER,
        tol=ELASTIC_NET_TOL,
        selection="random",
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(augmented_x, augmented_y)
    coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    _require(np.isfinite(coefficients).all(), "elastic-net produced nonfinite coefficients")
    dual_gap = float(estimator.dual_gap_)
    dual_gap_bound = ELASTIC_NET_TOL * float(np.square(augmented_y).sum()) / len(augmented_y)
    # Roundoff can make a converged gap slightly negative; sklearn's stopping
    # contract compares its upper bound, not its sign.
    _require(np.isfinite(dual_gap) and dual_gap <= dual_gap_bound,
             "elastic-net dual gap does not satisfy the declared tolerance")
    return coefficients


def _fit_coefficients(method: str, x: np.ndarray, y: np.ndarray, alpha: float,
                      l1_ratio: float) -> np.ndarray:
    if method == "ridge":
        return _ridge_coefficients(x, y, alpha)
    return _elastic_net_coefficients(x, y, alpha, l1_ratio)


def _validation_r2(method: str, dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                   train_times: list[pd.Timestamp], validation_times: list[pd.Timestamp],
                   columns: list[int], alpha: float, l1_ratio: float) -> tuple[float, np.ndarray]:
    x_train, y_train = _stack_sample(dataset, train_times, columns)
    x_validation, y_validation = _stack_sample(dataset, validation_times, columns)
    if len(y_train) == 0 or len(y_validation) == 0:
        return np.nan, np.empty(0)
    target_scale = float(np.sqrt(np.mean(np.square(y_train))))
    if not np.isfinite(target_scale) or target_scale <= 0:
        return np.nan, np.empty(0)
    y_train = y_train / target_scale
    y_validation = y_validation / target_scale
    coefficient = _fit_coefficients(method, x_train, y_train, alpha, l1_ratio)
    baseline = float(np.mean(np.square(y_validation)))
    if not np.isfinite(baseline) or baseline <= 0:
        return np.nan, coefficient
    validation_prediction = np.einsum("ni,i->n", x_validation, coefficient)
    mse = float(np.mean(np.square(y_validation - validation_prediction)))
    return 1.0 - mse / baseline, coefficient


def _select_alpha(method: str, dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                  train_times: list[pd.Timestamp], validation_times: list[pd.Timestamp],
                  columns: list[int], policy: dict[str, Any]) -> tuple[float | None, float, list[dict[str, Any]]]:
    trials = []
    for alpha in sorted(policy["alpha_grid"]):
        objective, _ = _validation_r2(method, dataset, train_times, validation_times,
                                      columns, float(alpha), float(policy["l1_ratio"]))
        trials.append({"alpha": float(alpha), "validation_r2": float(objective) if np.isfinite(objective) else None})
    valid = [(trial["validation_r2"], trial["alpha"]) for trial in trials
             if trial["validation_r2"] is not None]
    if not valid:
        return None, np.nan, trials
    # Lowest validation MSE wins; the smaller alpha resolves exact ties.
    objective, alpha = max(valid, key=lambda item: (item[0], -item[1]))
    return alpha, float(objective), trials


def _ridge_validation_stats(dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                            train_times: list[pd.Timestamp],
                            validation_times: list[pd.Timestamp],
                            factor_count: int) -> _RidgeValidationStats:
    """Build full-column sufficient statistics once for stepwise proposals.

    The resulting quadratic form gives the same ridge validation MSE as
    predicting each validation row directly, while avoiding a full panel
    traversal for every feature subset and alpha.
    """
    x_train, y_train = _stack_sample(dataset, train_times)
    x_validation, y_validation = _stack_sample(dataset, validation_times)
    empty = np.empty((factor_count, factor_count), dtype=float)
    if len(y_train) == 0 or len(y_validation) == 0:
        return _RidgeValidationStats(empty, np.empty(factor_count), empty,
                                     np.empty(factor_count), np.nan, False)
    target_scale = float(np.sqrt(np.mean(np.square(y_train))))
    if not np.isfinite(target_scale) or target_scale <= 0:
        return _RidgeValidationStats(empty, np.empty(factor_count), empty,
                                     np.empty(factor_count), np.nan, False)
    y_train = y_train / target_scale
    y_validation = y_validation / target_scale
    validation_target_mean_square = float(np.mean(np.square(y_validation)))
    if not np.isfinite(validation_target_mean_square) or validation_target_mean_square <= 0:
        return _RidgeValidationStats(empty, np.empty(factor_count), empty,
                                     np.empty(factor_count), validation_target_mean_square, False)
    return _RidgeValidationStats(
        train_gram=(x_train.T @ x_train) / len(y_train),
        train_rhs=(x_train.T @ y_train) / len(y_train),
        validation_gram=(x_validation.T @ x_validation) / len(y_validation),
        validation_rhs=(x_validation.T @ y_validation) / len(y_validation),
        validation_target_mean_square=validation_target_mean_square,
        valid=True,
    )


def _select_alpha_ridge_stats(stats: _RidgeValidationStats, columns: list[int],
                              policy: dict[str, Any]) -> tuple[float | None, float,
                                                               list[dict[str, Any]]]:
    """Select ridge alpha from cached sufficient statistics for one subset."""
    trials = []
    for alpha_value in sorted(policy["alpha_grid"]):
        alpha = float(alpha_value)
        objective = np.nan
        if stats.valid:
            gram = stats.train_gram[np.ix_(columns, columns)]
            rhs = stats.train_rhs[columns]
            coefficient = np.linalg.solve(gram + alpha * np.eye(len(columns)), rhs)
            validation_gram = stats.validation_gram[np.ix_(columns, columns)]
            validation_rhs = stats.validation_rhs[columns]
            mse = (stats.validation_target_mean_square
                   - 2.0 * float(coefficient @ validation_rhs)
                   + float(coefficient @ validation_gram @ coefficient))
            objective = 1.0 - mse / stats.validation_target_mean_square
        trials.append({"alpha": alpha,
                       "validation_r2": float(objective) if np.isfinite(objective) else None})
    valid = [(trial["validation_r2"], trial["alpha"]) for trial in trials
             if trial["validation_r2"] is not None]
    if not valid:
        return None, np.nan, trials
    objective, alpha = max(valid, key=lambda item: (item[0], -item[1]))
    return alpha, float(objective), trials


def _stepwise_select_with_evaluator(
        factor_names: list[str], policy: dict[str, Any],
        subset_evaluator: Callable[[tuple[int, ...]],
                                   tuple[float | None, float, list[dict[str, Any]]]] | None,
        *, objective_name: str,
        batch_subset_evaluator: Callable[
            [list[tuple[int, ...]]],
            dict[tuple[int, ...], tuple[float | None, float, list[dict[str, Any]]]],
        ] | None = None,
        budget_unit: str = "feature_subset_proposal",
        initial_objective: float = 0.0) -> dict[str, Any]:
    """Run complete-round forward selection with repeated backward pruning.

    Legacy budgets count proposals; matched-search budgets count previously
    unseen nonempty subsets. Either mode preflights the complete next round.
    The evaluator receives sorted factor indices and returns selected alpha,
    objective and per-alpha trials. An unscored empty baseline may be -inf.
    """
    _require(subset_evaluator is not None or batch_subset_evaluator is not None,
             "a subset evaluator is required")
    _require(budget_unit in {"feature_subset_proposal", "unique_nonempty_subset"},
             "unknown stepwise budget unit")
    factor_count = len(factor_names)
    cache: dict[tuple[int, ...], tuple[float | None, float, list[dict[str, Any]]]] = {}
    selected: list[int] = []
    selection_order: list[str] = []
    current_objective = initial_objective
    selected_alpha: float | None = None
    rounds: list[dict[str, Any]] = []
    budget_total = policy["pool_search_budget"]
    remaining_budget = budget_total
    proposal_evaluations = 0
    alpha_trial_evaluations = 0
    stop_reason = ""
    round_number = 0

    def round_cost(column_sets: list[list[int]]) -> int:
        if budget_unit == "feature_subset_proposal":
            return len(column_sets)
        return len({tuple(sorted(columns)) for columns in column_sets
                    if columns and tuple(sorted(columns)) not in cache})

    def evaluate_round(column_sets: list[list[int]]) -> list[
            tuple[float | None, float, list[dict[str, Any]], bool]]:
        nonlocal remaining_budget, proposal_evaluations, alpha_trial_evaluations
        canonical_sets = [tuple(sorted(columns)) for columns in column_sets]
        cost = round_cost(column_sets)
        _require(remaining_budget >= cost,
                 "stepwise round exceeded its preflight proposal budget")
        proposal_evaluations += len(canonical_sets)
        remaining_budget -= cost
        cache_hits = [canonical in cache for canonical in canonical_sets]
        uncached = list(dict.fromkeys(canonical for canonical in canonical_sets
                                      if canonical not in cache))
        for canonical in uncached:
            if not canonical:
                cache[canonical] = (None, initial_objective, [])
        uncached_nonempty = [canonical for canonical in uncached if canonical]
        if uncached_nonempty:
            if batch_subset_evaluator is not None:
                evaluated = batch_subset_evaluator(uncached_nonempty)
                _require(isinstance(evaluated, dict)
                         and set(evaluated) == set(uncached_nonempty),
                         "batch subset evaluator must return every requested subset exactly once")
                for canonical in uncached_nonempty:
                    result = evaluated[canonical]
                    cache[canonical] = result
                    alpha_trial_evaluations += len(result[2])
            else:
                for canonical in uncached_nonempty:
                    assert subset_evaluator is not None
                    result = subset_evaluator(canonical)
                    cache[canonical] = result
                    alpha_trial_evaluations += len(result[2])
        return [
            (cache[canonical][0], cache[canonical][1],
             [dict(trial) for trial in cache[canonical][2]], cache_hit)
            for canonical, cache_hit in zip(canonical_sets, cache_hits)
        ]

    def record_incomplete_round(phase: str, before: float, required: int,
                                chosen: list[int]) -> None:
        nonlocal round_number
        round_number += 1
        rounds.append({
            "round": round_number,
            "phase": phase,
            "objective_name": objective_name,
            "objective_before": float(before),
            "objective_after": float(before),
            "selected_before": [factor_names[index] for index in chosen],
            "selected_after": [factor_names[index] for index in chosen],
            "required_proposals": required,
            "available_budget": remaining_budget,
            "proposals": [],
            "action": "stop_budget_before_round",
            "status": "budget_insufficient_for_complete_round",
        })

    while True:
        if len(selected) >= policy["pool_capacity"]:
            stop_reason = "capacity_reached"
            break
        excluded = [index for index in range(factor_count) if index not in selected]
        if not excluded:
            stop_reason = "all_factors_selected"
            break
        proposed_subsets = [sorted([*selected, candidate]) for candidate in excluded]
        required = round_cost(proposed_subsets)
        if remaining_budget < required:
            record_incomplete_round("forward", current_objective, required, selected)
            stop_reason = "budget_insufficient_forward_round"
            break

        round_number += 1
        forward_round = round_number
        before_selected = selected.copy()
        before_objective = current_objective
        proposals = []
        evaluations = evaluate_round(proposed_subsets)
        for candidate, (alpha, objective, trials, cache_hit) in zip(excluded, evaluations):
            delta = objective - current_objective if np.isfinite(objective) else np.nan
            proposals.append({
                "proposal": proposal_evaluations,
                "phase": "forward",
                "objective_name": objective_name,
                "candidate": factor_names[candidate],
                "candidate_index": candidate,
                "selected_before": [factor_names[index] for index in selected],
                "selected_alpha": alpha,
                "validation_objective": float(objective) if np.isfinite(objective) else None,
                "validation_r2": (float(objective) if objective_name == "validation_r2"
                                  and np.isfinite(objective) else None),
                "delta": float(delta) if np.isfinite(delta) else None,
                "alpha_trials": trials,
                "cache_hit": cache_hit,
                "action": "pending",
            })
        eligible = [proposal for proposal in proposals
                    if proposal["validation_objective"] is not None
                    and (not np.isfinite(current_objective)
                         or proposal["delta"] > policy["min_objective_improvement"])]
        chosen_proposal = (max(eligible, key=lambda item: (item["validation_objective"],
                                                           -item["candidate_index"]))
                           if eligible else None)
        for proposal in proposals:
            proposal["action"] = ("add" if proposal is chosen_proposal else
                                  "rejected_below_minimum_improvement"
                                  if proposal["validation_objective"] is None
                                  or (np.isfinite(current_objective) and proposal["delta"]
                                      <= policy["min_objective_improvement"])
                                  else "rejected_not_best_forward_gain")
        if chosen_proposal is None:
            action = "stop_no_positive_forward_gain"
            after_objective = current_objective
        else:
            chosen_index = chosen_proposal["candidate_index"]
            selected = sorted([*selected, chosen_index])
            selection_order.append(factor_names[chosen_index])
            current_objective = float(chosen_proposal["validation_objective"])
            selected_alpha = chosen_proposal["selected_alpha"]
            action = "add_best_forward_gain"
            after_objective = current_objective
        rounds.append({
            "round": forward_round,
            "phase": "forward",
            "objective_name": objective_name,
            "objective_before": float(before_objective),
            "objective_after": float(after_objective),
            "selected_before": [factor_names[index] for index in before_selected],
            "selected_after": [factor_names[index] for index in selected],
            "proposals": proposals,
            "action": action,
            "status": "complete",
        })
        if chosen_proposal is None:
            stop_reason = "no_positive_forward_gain"
            break

        while selected:
            removable = selected.copy()
            proposed_subsets = [[index for index in selected if index != candidate]
                                for candidate in removable]
            required = round_cost(proposed_subsets)
            if remaining_budget < required:
                record_incomplete_round("backward_prune", current_objective,
                                        required, selected)
                stop_reason = "budget_insufficient_backward_round"
                break
            round_number += 1
            before_selected = selected.copy()
            before_objective = current_objective
            proposals = []
            evaluations = evaluate_round(proposed_subsets)
            for candidate, (alpha, objective, trials, cache_hit) in zip(removable, evaluations):
                delta = objective - current_objective if np.isfinite(objective) else np.nan
                proposals.append({
                    "proposal": proposal_evaluations,
                    "phase": "backward_prune",
                    "objective_name": objective_name,
                    "candidate": factor_names[candidate],
                    "candidate_index": candidate,
                    "selected_before": [factor_names[index] for index in selected],
                    "selected_alpha": alpha,
                    "validation_objective": float(objective) if np.isfinite(objective) else None,
                    "validation_r2": (float(objective) if objective_name == "validation_r2"
                                      and np.isfinite(objective) else None),
                    "delta": float(delta) if np.isfinite(delta) else None,
                    "alpha_trials": trials,
                    "cache_hit": cache_hit,
                    "action": "pending",
                })
            eligible = [proposal for proposal in proposals
                        if proposal["validation_objective"] is not None
                        and proposal["delta"] > policy["min_objective_improvement"]]
            chosen_proposal = (max(eligible, key=lambda item: (item["delta"],
                                                               -item["candidate_index"]))
                               if eligible else None)
            for proposal in proposals:
                proposal["action"] = ("remove" if proposal is chosen_proposal else
                                      "rejected_below_minimum_improvement"
                                      if proposal["delta"] is None or proposal["delta"]
                                      <= policy["min_objective_improvement"]
                                      else "rejected_not_best_pruning_gain")
            if chosen_proposal is None:
                action = "keep_no_positive_pruning_gain"
                after_objective = current_objective
            else:
                removed_index = chosen_proposal["candidate_index"]
                selected = [index for index in selected if index != removed_index]
                current_objective = float(chosen_proposal["validation_objective"])
                selected_alpha = chosen_proposal["selected_alpha"]
                action = "remove_best_pruning_gain"
                after_objective = current_objective
            rounds.append({
                "round": round_number,
                "phase": "backward_prune",
                "objective_name": objective_name,
                "objective_before": float(before_objective),
                "objective_after": float(after_objective),
                "selected_before": [factor_names[index] for index in before_selected],
                "selected_after": [factor_names[index] for index in selected],
                "proposals": proposals,
                "action": action,
                "status": "complete",
            })
            if chosen_proposal is None:
                break
        if stop_reason == "budget_insufficient_backward_round":
            break

    return {
        "selected": selected,
        "selected_alpha": selected_alpha,
        "objective_name": objective_name,
        "objective": current_objective,
        "validation_r2": current_objective if objective_name == "validation_r2" else None,
        "selection_order": selection_order,
        "rounds": rounds,
        "proposal_evaluations": proposal_evaluations,
        "proposal_search_budget_total": budget_total,
        "remaining_proposal_budget": remaining_budget,
        "proposal_budget_unit": budget_unit,
        "alpha_trial_evaluations": alpha_trial_evaluations,
        "unique_subset_evaluations": (sum(bool(key) for key in cache)
                                       if budget_unit == "unique_nonempty_subset" else len(cache)),
        "stop_reason": stop_reason,
    }


def _stepwise_select(dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
                     train_times: list[pd.Timestamp], validation_times: list[pd.Timestamp],
                     factor_names: list[str], policy: dict[str, Any]) -> dict[str, Any]:
    """Run the R² route using one set of cached full-column statistics."""
    stats = _ridge_validation_stats(dataset, train_times, validation_times, len(factor_names))

    def evaluate(columns: tuple[int, ...]) -> tuple[float | None, float, list[dict[str, Any]]]:
        return _select_alpha_ridge_stats(stats, list(columns), policy)

    return _stepwise_select_with_evaluator(factor_names, policy, evaluate,
                                           objective_name="validation_r2")


def _cross_section_correlations(values: pd.DataFrame, timestamp: pd.Timestamp,
                                min_symbols: int) -> np.ndarray:
    section = values.xs(timestamp, level="timestamp").to_numpy(dtype=float)
    complete = np.isfinite(section).all(axis=1)
    section = section[complete]
    if len(section) < min_symbols:
        return np.full((values.shape[1], values.shape[1]), np.nan)
    return np.corrcoef(section, rowvar=False)


def _cluster_members(values: pd.DataFrame, past_times: list[pd.Timestamp],
                     min_symbols: int, threshold: float) -> list[list[tuple[int, int]]]:
    factor_count = values.shape[1]
    correlations = []
    for timestamp in past_times:
        correlation = _cross_section_correlations(values, timestamp, min_symbols)
        correlations.append(correlation)
    mean_corr = np.full((factor_count, factor_count), np.nan)
    if correlations:
        stack = np.asarray(correlations)
        for left in range(factor_count):
            for right in range(factor_count):
                observations = stack[:, left, right]
                observations = observations[np.isfinite(observations)]
                if len(observations):
                    mean_corr[left, right] = float(observations.mean())
    np.fill_diagonal(mean_corr, 1.0)
    # Absolute-correlation graph components group duplicates and inverses.
    # Sorted breadth-first traversal chooses a deterministic spanning tree;
    # signs align each member with the component's lowest-column representative.
    unseen = set(range(factor_count))
    groups: list[list[tuple[int, int]]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = [(root, 1)]
        orientations = {root: 1}
        frontier = [root]
        while frontier:
            current = frontier.pop(0)
            neighbors = [candidate for candidate in sorted(unseen)
                         if np.isfinite(mean_corr[current, candidate])
                         and abs(mean_corr[current, candidate]) >= threshold]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                orientation = orientations[current] * (1 if mean_corr[current, neighbor] >= 0 else -1)
                orientations[neighbor] = orientation
                component.append((neighbor, orientation))
                frontier.append(neighbor)
        groups.append(sorted(component, key=lambda item: item[0]))
    return groups


def _score_row(row: np.ndarray, groups: list[list[tuple[int, int]]]) -> float:
    if not np.isfinite(row).all():
        return np.nan
    group_scores = [float(np.mean([row[index] * direction for index, direction in members]))
                    for members in groups]
    return float(np.mean(group_scores)) if group_scores else np.nan


def _audit_base(route: str, timestamp: pd.Timestamp, matured_times: list[pd.Timestamp],
                inner_train_times: list[pd.Timestamp], validation_times: list[pd.Timestamp],
                horizon: int) -> dict[str, Any]:
    train_start = matured_times[0] if matured_times else None
    train_end = matured_times[-1] if matured_times else None
    validation_start = validation_times[0] if validation_times else None
    latest_train_maturity = train_end + (horizon + 1) * HOUR if train_end is not None else None
    latest_used_source = max(matured_times + validation_times) if matured_times or validation_times else None
    latest_maturity = (latest_used_source + (horizon + 1) * HOUR
                       if latest_used_source is not None else None)
    return {
        "route": route,
        "timestamp": timestamp,
        "train_source_start": train_start,
        "train_source_end": train_end,
        "inner_train_source_start": inner_train_times[0] if inner_train_times else None,
        "inner_train_source_end": inner_train_times[-1] if inner_train_times else None,
        "validation_source_start": validation_start,
        "validation_source_end": validation_times[-1] if validation_times else None,
        "latest_label_maturity": latest_maturity,
        "latest_train_label_maturity": latest_train_maturity,
        "train_periods": len(matured_times),
        "inner_train_periods": len(inner_train_times),
        "validation_periods": len(validation_times),
        "horizon_hours": horizon,
    }


def generate_selection_scores(values: pd.DataFrame, opens: pd.Series, *,
                              horizon_hours: int, policy: dict[str, Any],
                              methods: tuple[str, ...] | list[str] | None = None) -> SelectionResult:
    """Generate equal, ICIR, cluster, ridge, elastic-net, and finite-pool scores.

    The default method set preserves the original six routes. Requested
    regression and selection routes use the latest matured source bars as an
    inner validation block, with a horizon-plus-one-hour purge before that
    block. The chosen model is then refit on every matured source bar in the
    rolling fit window.
    """
    _validate_policy(policy)
    if methods is None:
        requested_routes = ROUTES
    else:
        _require(isinstance(methods, (tuple, list)), "methods must be a tuple or list")
        requested_routes = tuple(methods)
    _require(bool(requested_routes), "methods must contain at least one supported route")
    _require(len(set(requested_routes)) == len(requested_routes),
             "methods must not contain duplicate routes")
    _require(all(route in SUPPORTED_ROUTES for route in requested_routes),
             "methods contains an unsupported route")
    legacy_routes_requested = set(requested_routes).intersection(ROUTES)
    stepwise_requested = STEPWISE_ROUTE in requested_routes
    timestamps, symbols = _validate_inputs(values, opens, horizon_hours)
    factor_names = list(values.columns)
    dataset = _make_labels(values, opens, timestamps, symbols, horizon_hours,
                           policy["min_cross_section_symbols"])
    data_times = pd.DatetimeIndex(sorted(dataset), name="timestamp")

    if "rolling_icir" in requested_routes:
        icir_weights, icir_directions = _rolling_icir_weights(
            dataset, timestamps, factor_names, horizon_hours, policy,
        )
    else:
        icir_weights = icir_directions = None

    scores = {route: pd.Series(np.nan, index=values.index, name="score", dtype=float)
              for route in requested_routes}
    fits: list[dict[str, Any]] = []
    states: dict[str, Any] = {route: None for route in requested_routes}
    first_timestamp = timestamps[0]
    shared_legacy_active = False
    legacy_active = False
    stepwise_active = False
    shared_ready_values = np.zeros(len(timestamps), dtype=bool)

    for position, timestamp_value in enumerate(timestamps):
        timestamp = pd.Timestamp(timestamp_value)
        if legacy_routes_requested:
            matured_times = _matured_window(data_times, timestamp, horizon_hours, policy)
            train_times, validation_times = _window_split(matured_times, timestamp, horizon_hours, policy)
            base_ready = (len(train_times) >= policy["min_train_periods"]
                          and len(validation_times) >= policy["min_inner_validation_periods"])
            correlation_ready = position >= policy["correlation_window_hours"]
            icir_latest_source = timestamp - (horizon_hours + 1) * HOUR
            icir_earliest_source = icir_latest_source - (policy["icir_window_hours"] - 1) * HOUR
            icir_ready = int(((data_times >= icir_earliest_source)
                              & (data_times <= icir_latest_source)).sum()) >= policy["icir_min_periods"]
            common_ready = base_ready and correlation_ready and icir_ready
            due = int((timestamp - first_timestamp) / HOUR) % policy["refit_every_hours"] == 0
        else:
            # A complete hourly grid makes this exactly equivalent to the
            # timestamp modulo used by the legacy routes. Stepwise only needs
            # its matured split when it is about to refit.
            due = position % policy["refit_every_hours"] == 0
            if due:
                matured_times = _matured_window(data_times, timestamp, horizon_hours, policy)
                train_times, validation_times = _window_split(
                    matured_times, timestamp, horizon_hours, policy,
                )
                base_ready = (len(train_times) >= policy["min_train_periods"]
                              and len(validation_times) >= policy["min_inner_validation_periods"])
                correlation_ready = position >= policy["correlation_window_hours"]
                icir_latest_source = timestamp - (horizon_hours + 1) * HOUR
                icir_earliest_source = icir_latest_source - (policy["icir_window_hours"] - 1) * HOUR
                icir_ready = int(((data_times >= icir_earliest_source)
                                  & (data_times <= icir_latest_source)).sum()) >= policy["icir_min_periods"]
                common_ready = base_ready and correlation_ready and icir_ready
            else:
                matured_times = train_times = validation_times = []
                base_ready = correlation_ready = icir_ready = common_ready = False

        # Preserve the original shared readiness boundary even when the caller
        # requests only stepwise, so downstream account runs use the same mask.
        if due:
            shared_legacy_active = common_ready

        if due and legacy_routes_requested:
            if not common_ready:
                legacy_active = False
                for route in legacy_routes_requested:
                    states[route] = None
            else:
                legacy_active = True

                if "cluster" in requested_routes:
                    groups = _cluster_members(
                        values, [pd.Timestamp(t) for t in timestamps[max(0, position - policy["correlation_window_hours"]):position]],
                        policy["min_cross_section_symbols"],
                        float(policy["cluster_abs_correlation_threshold"]),
                    )
                    states["cluster"] = groups
                    fits.append({**_audit_base("cluster", timestamp, matured_times, train_times,
                                               validation_times, horizon_hours),
                                 "correlation_window_hours": policy["correlation_window_hours"],
                                 "grouping": "absolute_correlation_connected_components",
                                 "groups": [{"representative": factor_names[group[0][0]],
                                             "members": [factor_names[index] for index, _ in group],
                                             "orientations": {factor_names[index]: direction
                                                              for index, direction in group}}
                                            for group in groups]})

                for method in ("ridge", "elastic_net"):
                    if method not in requested_routes:
                        continue
                    alpha, objective, trials = _select_alpha(method, dataset, train_times,
                                                             validation_times, list(range(len(factor_names))), policy)
                    coefficient = None
                    status = "no_valid_inner_objective"
                    if alpha is not None:
                        x_train, y_train = _stack_sample(dataset, matured_times)
                        target_scale = float(np.sqrt(np.mean(np.square(y_train))))
                        _require(np.isfinite(target_scale) and target_scale > 0,
                                 "matured regression targets have zero or invalid scale")
                        y_train = y_train / target_scale
                        coefficient = _fit_coefficients(method, x_train, y_train, alpha,
                                                        float(policy["l1_ratio"]))
                        if method == "elastic_net" and not np.any(coefficient != 0):
                            coefficient = None
                            status = "cash_zero_coefficients"
                        else:
                            status = "active"
                    states[method] = coefficient
                    audit = {**_audit_base(method, timestamp, matured_times, train_times,
                                            validation_times, horizon_hours),
                             "selected_alpha": alpha,
                             "inner_validation_r2": objective if np.isfinite(objective) else None,
                             "alpha_trials": trials,
                             "coefficients": (dict(zip(factor_names, coefficient.tolist()))
                                              if coefficient is not None else None),
                             "status": status}
                    fits.append(audit)

                if "pool" in requested_routes:
                    selected: list[int] = []
                    selection_steps: list[dict[str, Any]] = []
                    current_objective = 0.0
                    remaining = list(range(len(factor_names)))
                    x_priority, y_priority = _stack_sample(dataset, train_times)
                    if len(y_priority):
                        numerator = np.einsum("ni,n->i", x_priority, y_priority)
                        denominator = np.sqrt(np.square(x_priority).sum(axis=0) * np.square(y_priority).sum())
                        marginal = np.divide(np.abs(numerator), denominator,
                                             out=np.zeros(len(factor_names)), where=denominator > 0)
                        candidate_order = sorted(remaining, key=lambda index: (-float(marginal[index]), index))
                    else:
                        candidate_order = remaining.copy()
                    remaining_budget = policy["pool_search_budget"]
                    candidate_evaluations = 0
                    selected_alpha: float | None = None
                    for candidate in candidate_order:
                        if len(selected) >= policy["pool_capacity"] or remaining_budget <= 0:
                            break
                        if candidate not in remaining:
                            continue
                        trial_columns = selected + [candidate]
                        alpha, objective, alpha_trials = _select_alpha(
                            "ridge", dataset, train_times, validation_times, trial_columns, policy,
                        )
                        candidate_evaluations += 1
                        remaining_budget -= 1
                        improvement = objective - current_objective if np.isfinite(objective) else np.nan
                        proposal = {
                            "proposal": candidate_evaluations,
                            "candidate": factor_names[candidate],
                            "candidate_index": candidate,
                            "selected_alpha": alpha,
                            "validation_r2": float(objective) if np.isfinite(objective) else None,
                            "marginal_r2_improvement": float(improvement) if np.isfinite(improvement) else None,
                            "alpha_trials": alpha_trials,
                        }
                        if alpha is None or not np.isfinite(objective):
                            proposal["status"] = "no_valid_candidate_objective"
                        elif improvement <= policy["min_objective_improvement"]:
                            proposal["status"] = "rejected_below_minimum_improvement"
                        else:
                            selected.append(candidate)
                            remaining.remove(candidate)
                            current_objective = float(objective)
                            selected_alpha = alpha
                            proposal["status"] = "selected"
                        selection_steps.append(proposal)
                    pool_coefficient = None
                    if selected and selected_alpha is not None:
                        x_pool, y_pool = _stack_sample(dataset, matured_times, selected)
                        target_scale = float(np.sqrt(np.mean(np.square(y_pool))))
                        _require(np.isfinite(target_scale) and target_scale > 0,
                                 "matured pool targets have zero or invalid scale")
                        y_pool = y_pool / target_scale
                        pool_coefficient = _ridge_coefficients(x_pool, y_pool, selected_alpha)
                    states["pool"] = (selected, pool_coefficient) if pool_coefficient is not None else None
                    fits.append({**_audit_base("pool", timestamp, matured_times, train_times,
                                               validation_times, horizon_hours),
                                 "capacity": policy["pool_capacity"],
                                 "candidate_search_budget_total": policy["pool_search_budget"],
                                 "candidate_evaluations": candidate_evaluations,
                                 "candidate_order": [factor_names[index] for index in candidate_order],
                                 "selected_factors": [factor_names[index] for index in selected],
                                 "selected_alpha": selected_alpha,
                                 "validation_r2": current_objective if selected else None,
                                 "selection_steps": selection_steps,
                                 "coefficients": (dict(zip([factor_names[index] for index in selected],
                                                           pool_coefficient.tolist()))
                                                  if pool_coefficient is not None else None),
                                 "status": "active" if states["pool"] is not None
                                 else "cash_no_positive_marginal_gain"})

        if due and stepwise_requested:
            if not common_ready:
                stepwise_active = False
                states[STEPWISE_ROUTE] = None
            else:
                stepwise_active = True
                selection = _stepwise_select(dataset, train_times, validation_times,
                                             factor_names, policy)
                selected = selection["selected"]
                selected_alpha = selection["selected_alpha"]
                coefficient = None
                if selected and selected_alpha is not None:
                    x_fit, y_fit = _stack_sample(dataset, matured_times, selected)
                    target_scale = float(np.sqrt(np.mean(np.square(y_fit))))
                    _require(np.isfinite(target_scale) and target_scale > 0,
                             "matured stepwise targets have zero or invalid scale")
                    coefficient = _ridge_coefficients(x_fit, y_fit / target_scale, selected_alpha)
                states[STEPWISE_ROUTE] = ((selected, coefficient)
                                          if coefficient is not None else None)
                audit_base = _audit_base(STEPWISE_ROUTE, timestamp, matured_times,
                                         train_times, validation_times, horizon_hours)
                fits.append({**audit_base,
                             "capacity": policy["pool_capacity"],
                             "proposal_search_budget_total": selection["proposal_search_budget_total"],
                             "proposal_evaluations": selection["proposal_evaluations"],
                             "remaining_proposal_budget": selection["remaining_proposal_budget"],
                             "proposal_budget_unit": selection["proposal_budget_unit"],
                             "alpha_trial_evaluations": selection["alpha_trial_evaluations"],
                             "unique_subset_evaluations": selection["unique_subset_evaluations"],
                             "factor_order": factor_names,
                             "selected_factors": [factor_names[index] for index in selected],
                             "selection_order": selection["selection_order"],
                             "selected_alpha": selected_alpha,
                             "objective_name": selection["objective_name"],
                             "validation_objective": float(selection["objective"]),
                             "validation_r2": float(selection["validation_r2"]),
                             "selection_rounds": selection["rounds"],
                             "stop_reason": selection["stop_reason"],
                             "fit_source_through": audit_base["train_source_end"],
                             "fit_label_matured_through": audit_base["latest_train_label_maturity"],
                             "coefficients": (dict(zip([factor_names[index] for index in selected],
                                                       coefficient.tolist()))
                                              if coefficient is not None else None),
                             "status": "active" if states[STEPWISE_ROUTE] is not None
                             else "cash_no_positive_stepwise_gain"})

        row_values = values.xs(timestamp, level="timestamp")
        row_matrix = row_values.to_numpy(dtype=float)
        complete = np.isfinite(row_matrix).all(axis=1)
        enough_symbols = int(complete.sum()) >= policy["min_cross_section_symbols"]
        shared_ready_values[position] = bool(shared_legacy_active and enough_symbols)
        if not legacy_active and not stepwise_active:
            continue

        # The legacy rolling-ICIR control updates its matured weights hourly,
        # independent of the refit cadence used by the learned routes.
        if legacy_active and "rolling_icir" in requested_routes:
            current_icir_weights = icir_weights.loc[timestamp].to_numpy(dtype=float)
            current_icir_directions = icir_directions.loc[timestamp].to_numpy(dtype=float)
            icir_signed_weights = current_icir_weights * current_icir_directions
            icir_active = np.isfinite(icir_signed_weights).all() and current_icir_weights.sum() > 0
            fits.append({**_audit_base("rolling_icir", timestamp, matured_times, train_times,
                                       validation_times, horizon_hours),
                         "factor_weights": dict(zip(factor_names, current_icir_weights.tolist())),
                         "factor_directions": dict(zip(factor_names, current_icir_directions.tolist())),
                         "status": "active" if icir_active else "cash_no_active_icir"})
        else:
            icir_active = False
            icir_signed_weights = None

        if not enough_symbols:
            continue
        output_positions = np.arange(position * len(symbols), (position + 1) * len(symbols))

        if legacy_active:
            if "equal" in requested_routes:
                scores["equal"].iloc[output_positions[complete]] = row_matrix[complete].mean(axis=1)

            if "cluster" in requested_routes and states["cluster"] is not None and complete.any():
                cluster_scores = np.array([_score_row(row, states["cluster"])
                                           for row in row_matrix[complete]])
                scores["cluster"].iloc[output_positions[complete]] = cluster_scores

            if icir_active and complete.any():
                scores["rolling_icir"].iloc[output_positions[complete]] = np.einsum(
                    "ni,i->n", row_matrix[complete], icir_signed_weights,
                )

            for method in ("ridge", "elastic_net"):
                if method not in requested_routes:
                    continue
                coefficient = states[method]
                if coefficient is not None and complete.any():
                    scores[method].iloc[output_positions[complete]] = np.einsum(
                        "ni,i->n", row_matrix[complete], coefficient,
                    )

            if "pool" in requested_routes:
                pool_state = states["pool"]
                if pool_state is not None and complete.any():
                    selected, coefficient = pool_state
                    scores["pool"].iloc[output_positions[complete]] = np.einsum(
                        "ni,i->n", row_matrix[complete][:, selected], coefficient,
                    )

        if stepwise_active and states[STEPWISE_ROUTE] is not None and complete.any():
            selected, coefficient = states[STEPWISE_ROUTE]
            scores[STEPWISE_ROUTE].iloc[output_positions[complete]] = np.einsum(
                "ni,i->n", row_matrix[complete][:, selected], coefficient,
            )

    shared_model_ready = pd.Series(shared_ready_values, index=timestamps,
                                   name="shared_model_ready", dtype=bool)
    return SelectionResult(scores=scores, fits=fits,
                           shared_model_ready=shared_model_ready)
