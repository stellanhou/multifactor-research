"""Single-subset execution and deterministic fixed-arm ordering for P4.

Every subset uses the baseline's frozen all-factor common mask and standardized
panel. This module does not load source data, call a model, choose the next
subset, or execute a batch of fixed-arm comparisons.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .multifactor_workflow import _code_version, _run_one, _write_json


COMPARISON_METRICS = (
    "net_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_turnover",
    "total_fees",
    "total_slippage_cost",
    "total_funding",
)


def fixed_subset_order(pool_ids: Iterable[str]) -> list[tuple[str, ...]]:
    """Return every fixed subset in sorted-ID, size-ascending order."""
    ids = list(pool_ids)
    if any(not isinstance(card_id, str) or not card_id for card_id in ids):
        raise ValueError("pool IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("pool IDs must be unique")
    ordered = sorted(ids)
    return [
        subset
        for size in range(2, len(ordered))
        for subset in combinations(ordered, size)
    ]


def _pool_cards(snapshot) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    cards = list(snapshot.cards)
    card_ids = [card["id"] for card in cards]
    if not card_ids or any(not isinstance(card_id, str) or not card_id for card_id in card_ids):
        raise ValueError("baseline cards must have nonempty IDs")
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("baseline card IDs must be unique")

    baseline_factors = snapshot.result["factors"]
    factor_records = {item["id"]: item for item in baseline_factors}
    if set(factor_records) != set(card_ids):
        raise ValueError("baseline result factors do not match the frozen card pool")
    factor_codes = {card_id: factor_records[card_id]["code"] for card_id in card_ids}
    return card_ids, {card["id"]: card for card in cards}, factor_codes


def _sample(snapshot) -> tuple[dict[str, Any], dict[str, str]]:
    contract = snapshot.contract
    start, end = contract.bounds
    account_signal_start = start - pd.Timedelta(hours=1)
    account_signal_end = end - pd.Timedelta(hours=1)
    factor_index = snapshot.factors.common_mask.index
    timestamps = factor_index.get_level_values("timestamp")
    account_period = (timestamps >= account_signal_start) & (timestamps < account_signal_end)
    account_eligible = snapshot.factors.eligible & account_period
    account_common = snapshot.factors.common_mask & account_period
    if not bool(account_common.any()):
        raise ValueError("baseline common mask has no valid rows in the frozen signal window")

    sample = {
        "factor_panel_eligible_rows_including_warmup": int(snapshot.factors.eligible.sum()),
        "factor_panel_common_rows_including_warmup": int(snapshot.factors.common_mask.sum()),
        "account_signal_window_eligible_rows": int(account_eligible.sum()),
        "account_signal_window_common_rows": int(account_common.sum()),
        "account_signal_window_start": account_signal_start.isoformat(),
        "account_signal_window_end_exclusive": account_signal_end.isoformat(),
    }
    signalwindow = {
        "signal_start_inclusive": account_signal_start.isoformat(),
        "signal_end_exclusive": account_signal_end.isoformat(),
        "account_start": start.isoformat(),
        "account_end_exclusive": end.isoformat(),
        "signal_grid_hours": int((account_signal_end - account_signal_start) / pd.Timedelta(hours=1)),
    }
    return sample, signalwindow


def _equal_weight_metrics(snapshot) -> dict[str, float]:
    experiments = snapshot.result["experiments"]
    matches = [
        experiment
        for experiment in experiments
        if experiment.get("kind") == "equal_weight" or experiment.get("name") == "equal_weight"
    ]
    if len(matches) != 1:
        raise ValueError("baseline result must contain exactly one equal-weight experiment")
    metrics = matches[0]["metrics"]
    missing = set(COMPARISON_METRICS) - set(metrics)
    if missing:
        raise ValueError(f"baseline equal-weight metrics missing: {sorted(missing)}")
    return metrics


def _definition(
    snapshot,
    *,
    experiment_id: str,
    selected_ids: list[str],
    pool_ids: list[str],
    cards_by_id: dict[str, dict[str, Any]],
    factor_codes: dict[str, str],
) -> dict[str, Any]:
    factor_snapshots = {item["id"]: item for item in snapshot.result["factors"]}
    result = snapshot.result
    contract = snapshot.contract
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "kind": "frozen_equal_weight_subset",
        "baseline_run_id": contract.run_id,
        "baseline_equal_weight_record_id": "strategy:equal_weight",
        "selected_card_ids": selected_ids,
        "selected_cards": [
            {
                "id": card_id,
                "code": factor_codes[card_id],
                "title": cards_by_id[card_id]["title"],
                "snapshot_path": factor_snapshots[card_id]["snapshot_path"],
            }
            for card_id in selected_ids
        ],
        "baseline_pool_card_ids": pool_ids,
        "weighting": "equal_weight_mean",
        "factor_values": "baseline snapshot factors.standardized; columns only",
        "sample_mask": "baseline snapshot factors.common_mask for the complete baseline pool",
        "portfolio_and_costs": "copied without change from the baseline contract",
        "fixed_snapshot_refs": {
            "baseline_input_snapshot": result["input_snapshot"],
            "baseline_card_snapshot": result["card_snapshot"],
            "baseline_factor_diagnostics": result["factor_diagnostics"],
            "baseline_common_mask": "factor_panels/common_mask.csv",
            "baseline_standardized_panel": result["factor_diagnostics"]["standardized"],
        },
        "input_provenance": snapshot.inputs.diagnostics,
        "inherited_prior_results_exposed": True,
    }


def execute_subset(snapshot, selected_ids: list[str], root: Path) -> dict[str, Any]:
    """Execute one frozen equal-weight subset on baseline inputs.

    The baseline contract is copied unchanged. Subset membership is recorded
    separately, preserving the original card, universe, date, cost, and
    portfolio definitions. The returned metrics compare only to the baseline
    equal-weight run; fixed-arm outputs are never read or returned here.
    """
    if not isinstance(selected_ids, list) or any(
        not isinstance(card_id, str) or not card_id for card_id in selected_ids
    ):
        raise ValueError("selected_ids must be a list of nonempty card IDs")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_ids cannot contain duplicates")

    pool_ids, cards_by_id, factor_codes = _pool_cards(snapshot)
    if not 2 <= len(selected_ids) < len(pool_ids):
        raise ValueError("subset size must be at least two and smaller than the full pool")
    if not set(selected_ids).issubset(pool_ids):
        raise ValueError("selected_ids must be a subset of the frozen baseline pool")
    # Canonicalize order to the frozen card order; selection is set-valued.
    selection = set(selected_ids)
    selected = [card_id for card_id in pool_ids if card_id in selection]
    baseline_metrics = _equal_weight_metrics(snapshot)

    experiment_root = Path(root)
    experiment_root.mkdir(parents=True, exist_ok=False)
    _write_json(experiment_root / "contract.json", snapshot.contract.as_dict())
    sample, signalwindow = _sample(snapshot)
    score = snapshot.factors.standardized[selected].mean(axis=1).rename("score")
    experiment_directory = experiment_root / "experiments" / "frozen_equal_weight_subset"
    experiment_directory.parent.mkdir()
    experiment = _run_one(
        "frozen_equal_weight_subset",
        "frozen_equal_weight_subset",
        selected,
        score,
        snapshot.inputs,
        snapshot.factors,
        snapshot.contract,
        experiment_directory,
        sample,
        factor_codes,
    )
    deltas = {
        metric: experiment["metrics"][metric] - baseline_metrics[metric]
        for metric in COMPARISON_METRICS
    }
    definition = _definition(
        snapshot,
        experiment_id=experiment_root.name,
        selected_ids=selected,
        pool_ids=pool_ids,
        cards_by_id=cards_by_id,
        factor_codes=factor_codes,
    )
    definition["code_version"] = _code_version()
    definition["agent_module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write_json(experiment_root / "definition.json", definition)

    artifacts = {
        "contract": "contract.json",
        "definition": "definition.json",
        "result": "result.json",
        **experiment["artifacts"],
    }
    result = {
        "experiment_id": experiment_root.name,
        "kind": "frozen_equal_weight_subset",
        "baseline_run_id": snapshot.contract.run_id,
        "baseline_equal_weight_record_id": "strategy:equal_weight",
        "selected_card_ids": selected,
        "included_factors": experiment["included_factors"],
        "metrics": experiment["metrics"],
        "delta_vs_equal_weight": deltas,
        "coverage": experiment["sample"],
        "sample": experiment["sample"],
        "signalwindow": signalwindow,
        "artifacts": artifacts,
        "inherited_prior_results_exposed": True,
    }
    _write_json(experiment_root / "result.json", result)
    return result
