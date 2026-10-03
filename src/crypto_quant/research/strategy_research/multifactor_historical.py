"""Frozen A+B subset research and one-shot C internal validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import validate_universe
from crypto_quant.research.factor_mining.contracts import identifier, require, text
from .multifactor_agent_contracts import AgentSessionContract
from .multifactor_evidence import load_baseline_snapshot
from .multifactor_factors import process_factors, read_cards
from .multifactor_workflow import (
    _run_one,
    _run_stress,
    _snapshot_factor_panels,
    _snapshot_inputs,
    _write_json,
)


APPROVED_CARD_IDS = (
    "fmv6-history-multi-20261002-0021--candidate-0001",
    "fmv6-history-multi-20261002-0071--candidate-0003",
    "fmv6-history-multi-20261002-0109--candidate-0002",
    "fmv6-history-multi-20261002-0109--candidate-0003",
)
CANDIDATE_SELECTION = "max_AB_net_return_then_lower_drawdown_then_sorted_card_ids"
CODE_VERSION_FILES = (
    "src/crypto_quant/research/strategy_research/multifactor_historical.py",
    "src/crypto_quant/research/strategy_research/multifactor_raw_dataset.py",
    "src/crypto_quant/research/strategy_research/multifactor_contracts.py",
    "src/crypto_quant/research/strategy_research/multifactor_data.py",
    "src/crypto_quant/research/strategy_research/multifactor_factors.py",
    "src/crypto_quant/research/strategy_research/multifactor_workflow.py",
    "src/crypto_quant/research/strategy_research/multifactor_evidence.py",
    "src/crypto_quant/research/strategy_research/multifactor_account.py",
    "src/crypto_quant/research/strategy_research/multifactor_agent_contracts.py",
    "src/crypto_quant/research/strategy_research/multifactor_agent_experiments.py",
    "src/crypto_quant/research/strategy_research/multifactor_agent_workflow.py",
    "src/crypto_quant/research/strategy_research/multifactor_forward.py",
    "src/crypto_quant/research/strategy_research/cli.py",
    "src/crypto_quant/cli.py",
    "src/crypto_quant/features/factor_expressions.py",
    "src/crypto_quant/features/factor_inputs.py",
    "src/crypto_quant/features/factors.py",
    "src/crypto_quant/data_access/market_data.py",
    "src/crypto_quant/data_access/futures_backfill.py",
    "src/crypto_quant/research/data_policy.py",
    "src/crypto_quant/research/factor_mining/contracts.py",
    "src/crypto_quant/research/factor_mining/structured_output.py",
)
STAGE_BOUNDS = {
    "A": ("2022-08-01T00:00:00Z", "2024-08-01T00:00:00Z"),
    "B": ("2024-08-01T00:00:00Z", "2025-08-01T00:00:00Z"),
    "C": ("2025-08-01T00:00:00Z", "2026-08-01T00:00:00Z"),
}


def _exact(value: Any, fields: set[str], label: str) -> None:
    require(isinstance(value, dict) and set(value) == fields,
            f"{label} fields do not match the frozen schema")


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")
    return float(value)


def _utc_hour(value: Any, label: str) -> pd.Timestamp:
    require(isinstance(value, str), f"{label} must be a UTC ISO timestamp")
    timestamp = pd.Timestamp(value)
    require(timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0
            and timestamp == timestamp.floor("h"), f"{label} must be an exact UTC hour")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class HistoricalContract:
    schema_version: int
    run_id: str
    dataset_manifest: str
    cards: list[str]
    universe: str
    horizon_hours: int
    warmup_hours: int
    development_start: str
    validation_start: str
    validation_end: str
    prior_data_use: str
    data_processing: str
    costs: dict[str, Any]
    portfolio: dict[str, Any]
    experiment_budget: int
    context_bytes: int
    qualification_gates: dict[str, Any]
    candidate_selection: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HistoricalContract":
        _exact(value, set(cls.__dataclass_fields__), "historical contract")
        result = cls(**value)
        require(type(result.schema_version) is int and result.schema_version == 1,
                "historical contract schema_version must be 1")
        identifier(result.run_id)
        identifier(f"{result.run_id}-ab")
        identifier(f"{result.run_id}-c")
        identifier(f"{result.run_id}-agent")
        for name in ("dataset_manifest", "universe", "prior_data_use", "data_processing"):
            text(getattr(result, name), name)
        require(isinstance(result.cards, list) and len(result.cards) == len(APPROVED_CARD_IDS),
                "historical research requires the four frozen 24h candidates")
        require(all(isinstance(path, str) and path.strip() for path in result.cards),
                "cards must be explicit paths")
        require(len(result.cards) == len(set(result.cards)), "card paths cannot repeat")
        require(type(result.horizon_hours) is int and result.horizon_hours == 24,
                "historical research horizon is frozen at 24 hours")
        require(type(result.warmup_hours) is int and result.warmup_hours == 168,
                "historical research warmup is frozen at 168 hours")
        boundaries = {
            "development_start": "2022-08-01T00:00:00Z",
            "validation_start": "2025-08-01T00:00:00Z",
            "validation_end": "2026-08-01T00:00:00Z",
        }
        for name, expected in boundaries.items():
            require(_utc_hour(getattr(result, name), name) == _utc_hour(expected, name),
                    f"{name} differs from the frozen historical split")

        _exact(result.costs, {"initial_capital", "fee_bps", "slippage_bps", "stress_multiplier"}, "costs")
        frozen_costs = {"initial_capital": 10000.0, "fee_bps": 10.0,
                        "slippage_bps": 5.0, "stress_multiplier": 2.0}
        for name, expected in frozen_costs.items():
            require(_finite_number(result.costs[name], f"costs.{name}") == expected,
                    f"costs.{name} differs from the frozen research contract")

        _exact(result.portfolio, {"long_count", "short_count", "gross_exposure", "max_asset_weight",
                                  "rebalance_hours", "margin_fraction"}, "portfolio")
        frozen_portfolio = {"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                            "max_asset_weight": 0.2, "rebalance_hours": 24, "margin_fraction": 0.1}
        for name, expected in frozen_portfolio.items():
            if type(expected) is int:
                require(type(result.portfolio[name]) is int and result.portfolio[name] == expected,
                        f"portfolio.{name} differs from the frozen research contract")
            else:
                require(_finite_number(result.portfolio[name], f"portfolio.{name}") == expected,
                        f"portfolio.{name} differs from the frozen research contract")

        require(type(result.experiment_budget) is int and result.experiment_budget == 2,
                "Agent and fixed arms are frozen at two experiments each")
        require(type(result.context_bytes) is int and result.context_bytes == 256000,
                "Agent context limit is frozen at 256000 bytes")
        _exact(result.qualification_gates,
               {"min_net_return", "max_drawdown", "min_traded_bars", "min_stress_return"},
               "qualification_gates")
        frozen_gates = {"min_net_return": 0.0, "max_drawdown": 0.15,
                        "min_traded_bars": 1, "min_stress_return": 0.0}
        for name, expected in frozen_gates.items():
            if name == "min_traded_bars":
                require(type(result.qualification_gates[name]) is int
                        and result.qualification_gates[name] == expected,
                        f"qualification_gates.{name} differs from the frozen gate")
            else:
                require(_finite_number(result.qualification_gates[name], f"qualification_gates.{name}") == expected,
                        f"qualification_gates.{name} differs from the frozen gate")
        require(result.candidate_selection == CANDIDATE_SELECTION,
                "candidate selection rule differs from the frozen contract")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def input_start(self) -> pd.Timestamp:
        return _utc_hour(self.development_start, "development_start") - pd.Timedelta(hours=self.warmup_hours + 1)


def _resolve_path(path: str, base_dir: Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (base_dir / result).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_code_version() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", *CODE_VERSION_FILES],
                            cwd=project_root, check=True, capture_output=True, text=True).stdout.splitlines()
    source_hashes = {}
    for relative in CODE_VERSION_FILES:
        path = project_root / relative
        require(path.is_file(), f"research code version source is missing: {relative}")
        source_hashes[relative] = _sha256(path)
    return {
        "git_commit": commit,
        "working_tree_clean_for_sources": not status,
        "working_tree_status_for_sources": status,
        "source_sha256": source_hashes,
        "captured_before_dataset_access": True,
        "source_builder_code_version_status": "not_captured_in_raw_dataset_manifest",
    }


def _contained_path(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative.strip(), f"{label} path is required")
    path = Path(relative)
    require(not path.is_absolute(), f"{label} path must be relative")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its run directory") from exc
    return resolved


def _source_manifest_gate(manifest_path: Path, contract_data: dict[str, Any],
                          contract_dir: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    from .multifactor_contracts import ResearchContract
    from .multifactor_data import ResearchDatasetManifest

    research_contract = ResearchContract.from_dict(contract_data)
    dataset = ResearchDatasetManifest.from_path(manifest_path, research_contract, contract_dir)
    document = dataset.document
    database_path = dataset.database_path
    require(database_path is not None, "strict dataset manifest did not resolve an isolated database")
    primary_database = Path(__file__).resolve().parents[4] / "market_data" / "crypto_quant.sqlite"
    require(database_path.resolve() != primary_database.resolve(),
            "historical research cannot read the primary database")
    provenance = {
        "dataset_id": document["dataset_id"],
        "manifest_path": str(dataset.path),
        "manifest_sha256": _sha256(dataset.path),
        "database_path": str(database_path),
        "database_sha256": document["database_signature"]["sha256"],
        "source_file_count": len(document["source_files"]),
        "symbols": sorted(document["symbols"]),
        "spot_feature_missing_rows": document["spot_feature_missing_rows"],
        "source_causality": document["source_causality"],
        "source_causality_notes": document["source_causality_notes"],
        "historical_causality_certified": document["historical_causality_certified"],
        "repair_state": document["repair_state"],
        "excluded_documented_repair_rows": document["excluded_documented_repair_rows"],
        "funding_mark": document["funding_mark"],
        "primary_source_fallback": False,
    }
    return document, database_path, provenance

def _validate_universe(path: Path, contract: HistoricalContract,
                       manifest_symbols: list[str]) -> pd.Series:
    frame = pd.read_csv(path)
    require(set(frame.columns) == {"timestamp", "symbol", "eligible"},
            "historical universe CSV requires timestamp,symbol,eligible")
    require(frame.eligible.isin([True, False, 0, 1]).all(), "universe membership must be explicit boolean/0/1")
    require(frame.timestamp.map(lambda value: pd.Timestamp(value).tzinfo is not None).all(),
            "universe timestamps must include an explicit timezone")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="ISO8601")
    require(frame.symbol.notna().all(), "universe symbols are required")
    universe = validate_universe(frame.set_index(["timestamp", "symbol"]).eligible.astype(bool))
    expected = pd.date_range(contract.input_start,
                             _utc_hour(contract.validation_end, "validation_end"),
                             freq="h", inclusive="left")
    timestamps = universe.index.get_level_values("timestamp").unique()
    require(timestamps.equals(expected), "historical universe must cover the full A+B+C and warmup hour grid")
    symbols = sorted(universe.index.get_level_values("symbol").unique())
    require(len(symbols) == 10 and symbols == sorted(manifest_symbols),
            "historical universe differs from the isolated dataset ten-symbol pool")
    return universe


def _stage_contract(contract: HistoricalContract, *, stage: str,
                    card_paths: list[Path], universe_path: Path,
                    manifest_path: Path, stage_contract_dir: Path) -> dict[str, Any]:
    if stage == "development":
        start, end = contract.development_start, contract.validation_start
        suffix = "ab"
    elif stage == "internal_validation":
        start, end = contract.validation_start, contract.validation_end
        suffix = "c"
    else:
        raise ValueError(f"unsupported historical stage: {stage}")
    run_id = identifier(f"{contract.run_id}-{suffix}")
    relative_manifest = os.path.relpath(manifest_path.resolve(), stage_contract_dir.resolve())
    return {
        "schema_version": 2,
        "run_id": run_id,
        "purpose": "research",
        "stage": stage,
        "start": start,
        "end": end,
        "warmup_hours": contract.warmup_hours,
        "horizon_hours": contract.horizon_hours,
        "cards": [str(path.resolve()) for path in card_paths],
        "universe": str(universe_path.resolve()),
        "dataset_manifest": relative_manifest,
        "prior_data_use": contract.prior_data_use,
        "data_processing": contract.data_processing,
        "costs": dict(contract.costs),
        "portfolio": dict(contract.portfolio),
    }


def _run_development_baseline(contract_path: Path, output: Path) -> dict[str, Any]:
    from .multifactor_workflow import run_research_baseline

    return run_research_baseline(contract_path, output)


def _write_stage_contract(root: Path, name: str, value: dict[str, Any]) -> Path:
    path = root / "contracts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)
    return path


def _card_pool(cards: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    ids = [card["id"] for card in cards]
    require(ids == list(APPROVED_CARD_IDS), "loaded card pool differs from the approved four-card order")
    codes = {card_id: f"F{index}" for index, card_id in enumerate(ids, start=1)}
    return ids, codes


def _require_agent_session(result: Any, contract: AgentSessionContract) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    require(isinstance(result, dict), "Agent runner must return a result object")
    require(result.get("status") == "engineering_complete", "Agent session did not complete")
    model_calls = result.get("model_calls")
    require(type(model_calls) is int and 0 <= model_calls <= contract.max_model_calls,
            "Agent session exceeded its derived model call ceiling")
    require(result.get("sealed_initial_drop_results") is True,
            "Agent session did not seal initial full-pool drop results")
    require(result.get("fixed_results_exposed_to_agent") is False,
            "fixed-arm outcomes must remain hidden from the Agent")
    require(result.get("independent_agent_value_established") is False,
            "historical Agent result cannot claim independent value")
    for field in ("paper_started", "forward_validation_started", "published"):
        require(result.get(field) is False, f"Agent session unexpectedly set {field}")
    agent_trials, fixed_trials = result.get("agent_experiments"), result.get("fixed_experiments")
    require(isinstance(agent_trials, list) and len(agent_trials) <= contract.experiment_budget,
            "Agent experiment count exceeds the frozen budget")
    require(isinstance(fixed_trials, list) and len(fixed_trials) == contract.experiment_budget,
            "fixed policy arm must execute the full frozen budget")
    session_root = Path(result.get("root", ""))
    require(session_root.is_absolute() and session_root.is_dir(), "Agent session root is missing")
    return agent_trials, fixed_trials, session_root


def _count_traded_bars(ledger: Path) -> int:
    require(ledger.is_file(), "account ledger is missing")
    frame = pd.read_csv(ledger)
    require("trade_notional" in frame, "account ledger is missing trade_notional")
    values = pd.to_numeric(frame["trade_notional"], errors="raise").to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), "account ledger trade_notional is invalid")
    return int((values > 0).sum())


def _ledger_traded_bars(session_root: Path, trial: dict[str, Any]) -> int:
    require(isinstance(trial.get("path"), str), "Agent trial path is required")
    trial_root = _contained_path(session_root, trial["path"], "Agent trial")
    artifacts = trial.get("artifacts")
    require(isinstance(artifacts, dict) and isinstance(artifacts.get("ledger"), str),
            "Agent trial ledger artifact path is required")
    ledger = _contained_path(trial_root, artifacts["ledger"], "Agent trial ledger")
    return _count_traded_bars(ledger)


def _candidate_failures(metrics: dict[str, Any], stress_metrics: dict[str, Any],
                        traded_bars: int, gates: dict[str, Any]) -> list[str]:
    net_return = _finite_number(metrics["net_return"], "net_return")
    max_drawdown = abs(_finite_number(metrics["max_drawdown"], "max_drawdown"))
    stress_return = _finite_number(stress_metrics["net_return"], "stress_net_return")
    failures = []
    if net_return < gates["min_net_return"]:
        failures.append("min_net_return")
    if max_drawdown > gates["max_drawdown"]:
        failures.append("max_drawdown")
    if traded_bars < gates["min_traded_bars"]:
        failures.append("min_traded_bars")
    if stress_return < gates["min_stress_return"]:
        failures.append("min_stress_return")
    return failures


def _run_ab_candidate_stress(trial: dict[str, Any], trial_id: str, snapshot,
                             root: Path, factor_codes: dict[str, str]) -> dict[str, Any]:
    selected = trial["selected_card_ids"]
    score = snapshot.factors.standardized[selected].mean(axis=1).rename("score")
    stress = _run_stress({"name": trial_id}, score, snapshot.inputs, snapshot.factors,
                         snapshot.contract, root / "development_stress" / trial_id)
    require(stress is not None, "historical candidate stress costs must be enabled")
    stress["path"] = str((root / "development_stress" / trial_id).relative_to(root))
    stress["included_factors"] = [factor_codes[card_id] for card_id in selected]
    return stress


def _collect_ab_candidates(agent_trials: list[dict[str, Any]], fixed_trials: list[dict[str, Any]],
                           session_root: Path, snapshot, contract: HistoricalContract,
                           root: Path, pool_ids: list[str], factor_codes: dict[str, str],
                           baseline_result: dict[str, Any], baseline_root: Path) -> list[dict[str, Any]]:
    candidates = []
    initial_configurations = {tuple([card_id]) for card_id in pool_ids}
    initial_configurations.add(tuple(sorted(pool_ids)))
    initial_configurations.update(tuple(sorted(set(pool_ids) - {card_id})) for card_id in pool_ids)
    for arm, trials in (("agent", agent_trials), ("fixed", fixed_trials)):
        seen_by_arm = set()
        for index, trial in enumerate(trials, start=1):
            selected = trial.get("selected_card_ids")
            require(isinstance(selected, list) and len(selected) == 2 and len(set(selected)) == 2,
                    f"{arm} candidate must be a novel two-card subset")
            require(set(selected) <= set(pool_ids), f"{arm} candidate contains a card outside the frozen pool")
            canonical = tuple(sorted(selected))
            require(canonical not in initial_configurations,
                    f"{arm} candidate repeats a previously evaluated baseline configuration")
            require(canonical not in seen_by_arm, f"{arm} arm repeated a subset within its budget")
            seen_by_arm.add(canonical)
            metrics = trial.get("metrics")
            require(isinstance(metrics, dict), f"{arm} candidate metrics are missing")
            for metric in ("net_return", "max_drawdown", "total_turnover", "total_fees",
                           "total_slippage_cost", "total_funding"):
                _finite_number(metrics[metric], f"{arm}.{metric}")
            trial_id = f"{arm}-{index:03d}"
            stress = _run_ab_candidate_stress(trial, trial_id, snapshot, root, factor_codes)
            traded_bars = _ledger_traded_bars(session_root, trial)
            failures = _candidate_failures(metrics, stress["metrics"], traded_bars,
                                           contract.qualification_gates)
            candidates.append({
                "trial_id": trial_id,
                "arm": arm,
                "selected_card_ids": list(canonical),
                "included_factors": [factor_codes[card_id] for card_id in canonical],
                "metrics": metrics,
                "traded_bars": traded_bars,
                "stress_costs": stress,
                "qualification_failures": failures,
                "qualified_ab": not failures,
                "source_trial_path": trial["path"],
                "source_artifacts": trial["artifacts"],
            })

    baseline_matches = [
        experiment for experiment in baseline_result["experiments"]
        if experiment.get("kind") == "equal_weight" or experiment.get("name") == "equal_weight"
    ]
    require(len(baseline_matches) == 1, "A+B baseline must have one full-pool equal-weight control")
    baseline_equal = baseline_matches[0]
    baseline_stress = baseline_result.get("stress_costs")
    require(isinstance(baseline_stress, dict), "A+B full-pool equal-weight cost stress is required")
    baseline_ledger_ref = baseline_equal.get("artifacts", {}).get("ledger")
    require(isinstance(baseline_ledger_ref, str), "full-pool equal-weight ledger reference is required")
    baseline_ledger = _contained_path(baseline_root, baseline_ledger_ref, "full-pool equal-weight ledger")
    baseline_traded_bars = _count_traded_bars(baseline_ledger)
    baseline_metrics = baseline_equal["metrics"]
    baseline_failures = _candidate_failures(baseline_metrics, baseline_stress["metrics"],
                                            baseline_traded_bars, contract.qualification_gates)
    candidates.append({
        "trial_id": "full-pool-equal-weight",
        "arm": "full_pool_equal_weight",
        "selected_card_ids": list(pool_ids),
        "included_factors": [factor_codes[card_id] for card_id in pool_ids],
        "metrics": baseline_metrics,
        "traded_bars": baseline_traded_bars,
        "stress_costs": baseline_stress,
        "qualification_failures": baseline_failures,
        "qualified_ab": not baseline_failures,
        "source_trial_path": baseline_equal["path"],
        "source_artifacts": baseline_equal["artifacts"],
        "shared_baseline_reference": True,
        "experiment_budget_charge": 0,
    })
    require(bool(candidates), "no executable candidate trial was returned by the two research arms")
    return candidates


def _selection_key(candidate: dict[str, Any]) -> tuple[float, float, tuple[str, ...]]:
    return (-float(candidate["metrics"]["net_return"]),
            abs(float(candidate["metrics"]["max_drawdown"])),
            tuple(sorted(candidate["selected_card_ids"])))


def _select_ab_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [candidate for candidate in candidates if candidate["qualified_ab"]]
    selected = min(candidates, key=_selection_key)
    return {
        "selection_rule": CANDIDATE_SELECTION,
        "selected_trial_id": selected["trial_id"],
        "selected_arm": selected["arm"],
        "selected_source_trial_path": selected["source_trial_path"],
        "selected_card_ids": selected["selected_card_ids"],
        "selected_factors": selected["included_factors"],
        "selected_ab_metrics": selected["metrics"],
        "selected_ab_traded_bars": selected["traded_bars"],
        "selected_ab_stress_metrics": selected["stress_costs"]["metrics"],
        "selected_ab_qualified": selected["qualified_ab"],
        "selected_ab_qualification_failures": selected["qualification_failures"],
        "selection_mode": "best_ab_candidate" if selected["qualified_ab"] else "best_ab_diagnostic_only",
        "candidate_count": len(candidates),
        "qualified_candidate_count": len(qualified),
        "ranked_trial_ids": [item["trial_id"] for item in sorted(candidates, key=_selection_key)],
        "selected_before_c_access": True,
        "candidate_definition_frozen_for_c": True,
        "c_data_accessed": False,
        "forward_validation_started": False,
    }


def _sample_for_c(factors, contract) -> dict[str, Any]:
    start, end = contract.bounds
    signal_start, signal_end = start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1)
    timestamps = factors.common_mask.index.get_level_values("timestamp")
    period = (timestamps >= signal_start) & (timestamps < signal_end)
    eligible = factors.eligible & period
    common = factors.common_mask & period
    require(bool(common.any()), "C has no common factor rows in its account signal window")
    return {
        "factor_panel_eligible_rows_including_warmup": int(factors.eligible.sum()),
        "factor_panel_common_rows_including_warmup": int(factors.common_mask.sum()),
        "account_signal_window_eligible_rows": int(eligible.sum()),
        "account_signal_window_common_rows": int(common.sum()),
        "account_signal_window_start": signal_start.isoformat(),
        "account_signal_window_end_exclusive": signal_end.isoformat(),
    }


def _run_validation_stage(contract_path: Path, contract_data: dict[str, Any],
                          manifest_path: Path, card_paths: list[Path],
                          universe_path: Path, selected_card_ids: list[str],
                          output_root: Path, qualification_gates: dict[str, Any],
                          candidate_freeze_sha256: str) -> dict[str, Any]:
    from .multifactor_contracts import ResearchContract
    from .multifactor_data import load_research_inputs

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    stage_contract = ResearchContract.from_dict(contract_data)
    inputs = load_research_inputs(manifest_path, stage_contract, contract_path.parent)
    cards = read_cards(card_paths, stage_contract.horizon_hours, stage_contract.warmup_hours)
    factors = process_factors(inputs.panel, cards)
    _write_json(output_root / "contract.json", contract_data)
    _snapshot_inputs(inputs, output_root)
    _snapshot_factor_panels(factors, output_root)
    sample = _sample_for_c(factors, stage_contract)
    pool_ids = [card["id"] for card in cards]
    require(2 <= len(selected_card_ids) <= len(pool_ids)
            and len(selected_card_ids) == len(set(selected_card_ids))
            and set(selected_card_ids) <= set(pool_ids),
            "C candidate IDs must be a valid frozen-pool subset")
    selected_set = set(selected_card_ids)
    selected_card_ids = [card_id for card_id in pool_ids if card_id in selected_set]
    factor_codes = {card_id: f"F{index}" for index, card_id in enumerate(pool_ids, start=1)}

    equal_score = factors.standardized[pool_ids].mean(axis=1).rename("score")
    equal_directory = output_root / "experiments" / "equal_weight_control"
    equal = _run_one("equal_weight_control", "equal_weight", pool_ids, equal_score,
                     inputs, factors, stage_contract, equal_directory, sample, factor_codes)

    reference_equals_candidate = selected_card_ids == pool_ids
    if reference_equals_candidate:
        selected_score = equal_score
        selected_directory = equal_directory
        selected = equal
    else:
        selected_score = factors.standardized[selected_card_ids].mean(axis=1).rename("score")
        selected_directory = output_root / "experiments" / "selected_candidate"
        selected = _run_one("selected_candidate", "selected_candidate", selected_card_ids,
                            selected_score, inputs, factors, stage_contract,
                            selected_directory, sample, factor_codes)
    stress_directory = output_root / "experiments" / "selected_candidate_stress"
    stress = _run_stress(selected, selected_score, inputs, factors, stage_contract, stress_directory)
    require(stress is not None, "C candidate stress costs must be enabled")
    stress["path"] = str(stress_directory.relative_to(output_root))
    traded_bars = _count_traded_bars(selected_directory / "ledger.csv")
    failures = _candidate_failures(selected["metrics"], stress["metrics"], traded_bars,
                                   qualification_gates)
    result = {
        "stage": "C_internal_validation",
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "selected_candidate_card_ids": selected_card_ids,
        "candidate": selected,
        "equal_weight_control": equal,
        "reference_equals_candidate": reference_equals_candidate,
        "candidate_artifact_ref": selected["path"],
        "equal_weight_artifact_ref": equal["path"],
        "candidate_stress_costs": stress,
        "traded_bars": traded_bars,
        "qualification_failures": failures,
        "qualification_gates": dict(qualification_gates),
        "qualified_c": not failures,
        "same_full_pool_common_mask": True,
        "agent_called_after_c_access": False,
        "forward_validation_started": False,
        "path": str(output_root),
    }
    _write_json(output_root / "result.json", result)
    return result


def _report(root: Path, contract: HistoricalContract, provenance: dict[str, Any],
            baseline: dict[str, Any], candidates: list[dict[str, Any]],
            freeze: dict[str, Any], validation: dict[str, Any] | None,
            agent_session_path: str, code_version: dict[str, Any], final_status: str) -> None:
    baseline_rows = []
    for experiment in baseline["experiments"]:
        metrics = experiment["metrics"]
        baseline_rows.append(
            f"| {experiment['name']} | {', '.join(experiment['included_factors'])} | "
            f"{metrics['net_return']:.2%} | {metrics['max_drawdown']:.2%} | "
            f"{metrics['total_fees']:.4g} | {metrics['total_slippage_cost']:.4g} | "
            f"{metrics['total_funding']:.4g} |"
        )
    candidate_rows = []
    for candidate in candidates:
        metrics, stress = candidate["metrics"], candidate["stress_costs"]["metrics"]
        candidate_rows.append(
            f"| {candidate['trial_id']} ({candidate['arm']}) | {', '.join(candidate['included_factors'])} | "
            f"{metrics['net_return']:.2%} | {metrics['max_drawdown']:.2%} | "
            f"{stress['net_return']:.2%} | {'pass' if candidate['qualified_ab'] else 'fail'} |"
        )
    lines = [
        "# 多因子历史研究与候选冻结",
        "",
        f"- 运行：`{contract.run_id}`；状态：`{final_status}`；阶段：A+B开发与C内部验证。",
        f"- 数据集：`{provenance['dataset_id']}`；源清单：[`dataset_manifest.json`](dataset_manifest.json)；因果来源标签：`{provenance['source_causality']}`。",
        f"- 执行代码 Git commit：`{code_version['git_commit']}`；逐文件版本见 [code_version.json](code_version.json)。原始数据构建时版本未记入源清单，不以当前哈希代替。",
        f"- 交易门槛在访问历史行情前由合同冻结：`{json.dumps(contract.qualification_gates, ensure_ascii=False, sort_keys=True)}`。",
        "- Agent 只接收全池、等权与单因子初始证据和自己的逐轮试验；全池消融收益保留但封存，固定策略臂结果不反馈给Agent。",
        "",
        "## A+B全池基线",
        "",
        "| 实验 | 因子代码 | 净收益 | 最大回撤 | 手续费 | 滑点 | 资金费 |",
        "|---|---|---:|---:|---:|---:|---:|",
        *baseline_rows,
        "",
        "## A+B候选试验",
        "",
        "| 试验 | 因子代码 | 净收益 | 最大回撤 | 2倍成本净收益 | 门槛 |",
        "|---|---|---:|---:|---:|---|",
        *candidate_rows,
        "",
        f"C阶段只检查在A+B冻结的子集 `{', '.join(freeze['selected_card_ids'])}` 与全池等权控制。",
        f"冻结方式：`{freeze['selection_mode']}`；依据规则：`{freeze['selection_rule']}`；选择发生在C数据读取前。",
    ]
    if validation is not None:
        candidate = validation["candidate"]["metrics"]
        control = validation["equal_weight_control"]["metrics"]
        stress = validation["candidate_stress_costs"]["metrics"]
        if validation["reference_equals_candidate"]:
            lines.append("C阶段选中方案与全池等权控制相同，使用同一账户回放及工件引用。")
        lines.extend([
            "",
            "## C内部验证",
            "",
            "| 组合 | 净收益 | 最大回撤 | 2倍成本净收益 | 门槛 |",
            "|---|---:|---:|---:|---|",
            f"| 冻结子集 | {candidate['net_return']:.2%} | {candidate['max_drawdown']:.2%} | "
            f"{stress['net_return']:.2%} | {'pass' if validation['qualified_c'] else 'fail'} |",
            f"| 全池等权控制 | {control['net_return']:.2%} | {control['max_drawdown']:.2%} | — | control |",
        ])
    lines.extend([
        "",
        "## 边界",
        "",
        "C是已暴露的内部历史验证，不是独立测试；C数据不回传本轮Agent，C结果也不用于重新挑选子集。资金费标记价按合同披露使用结算时点前已完成的1分钟标记价代理，最长年龄不超过60秒。",
        "通过门槛只表示候选达到预先写明的历史条件。未启动Paper、Demo、前向验证或发布。收益未通过门槛时仍保留所有实验结果并报告 `no_candidate_passed`。",
        "",
        "## 产物",
        "",
        "- [合同与源数据溯源](contract.json) · [卡片快照与清单](cards/) / [cards.json](cards.json) · [候选冻结记录](candidate-freeze.json)",
        f"- [Agent审查与逐轮记录]({agent_session_path}/report.md) · [全体Agent与固定臂试验](agent-session-result.json) · [候选资格与成本压力结果](candidate-trials.json)",
        "- [最终结构化结果](result.json) · A+B全池报告见 `development_full_pool/`，C见 `validation/`。",
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_record(root: Path, path: Path) -> dict[str, str]:
    resolved_root, resolved_path = root.resolve(), path.resolve()
    require(resolved_path.is_file(), f"historical result artifact is missing: {resolved_path}")
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"historical result artifact escapes its run root: {resolved_path}") from exc
    return {"path": str(relative), "sha256": _sha256(resolved_path)}


def run_historical(
    contract_path: Path,
    output: Path,
    model,
    *,
    model_mode: str,
    model_settings: dict[str, Any],
    agent_runner: Callable,
) -> dict[str, Any]:
    """Run A+B research, freeze one AB-ranked candidate, then access C once."""
    contract_path = Path(contract_path).resolve()
    contract = HistoricalContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
    code_version = _capture_code_version()
    contract_dir = contract_path.parent
    manifest_path = _resolve_path(contract.dataset_manifest, contract_dir)
    card_paths = [_resolve_path(path, contract_dir) for path in contract.cards]
    cards = read_cards(card_paths, contract.horizon_hours, contract.warmup_hours)
    pool_ids, factor_codes = _card_pool(cards)
    require(set(pool_ids) == set(APPROVED_CARD_IDS), "selected cards do not match the approved mechanism pool")
    universe_path = _resolve_path(contract.universe, contract_dir)
    require(universe_path.is_file(), f"historical universe is missing: {universe_path}")
    card_sources = [
        {"id": card["id"], "title": card["title"], "source_path": str(path),
         "source_sha256": _sha256(path), "snapshot_path": f"cards/{index:03d}_{card['id']}.json"}
        for index, (card, path) in enumerate(zip(cards, card_paths), start=1)
    ]
    output_root = Path(output) / contract.run_id
    contracts_dir = output_root / "contracts"
    ab_contract_data = _stage_contract(contract, stage="development", card_paths=card_paths,
                                       universe_path=universe_path, manifest_path=manifest_path,
                                       stage_contract_dir=contracts_dir)
    manifest, _, provenance = _source_manifest_gate(manifest_path, ab_contract_data, contracts_dir)
    _validate_universe(universe_path, contract, manifest["symbols"])

    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "contract.json", contract.as_dict())
    _write_json(output_root / "code_version.json", code_version)
    shutil.copyfile(manifest_path, output_root / "dataset_manifest.json")
    shutil.copyfile(universe_path, output_root / "universe.csv")
    card_dir = output_root / "cards"
    card_dir.mkdir()
    for card, path, source in zip(cards, card_paths, card_sources):
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == source["source_sha256"],
                f"factor card changed after validation: {card['id']}")
        destination = output_root / source["snapshot_path"]
        destination.write_bytes(payload)
    _write_json(output_root / "cards.json", card_sources)
    provenance["universe_path"] = str(universe_path)
    provenance["universe_sha256"] = _sha256(universe_path)
    provenance["card_sources"] = card_sources
    _write_json(output_root / "dataset_provenance.json", provenance)
    _write_json(output_root / "source_gate.json", {
        "passed": True,
        "dataset_id": provenance["dataset_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "database_sha256": provenance["database_sha256"],
        "primary_source_fallback": False,
        "native_hourly_source": "independent official Binance Vision downloads",
        "funding_event_mark_source": "previous completed 1m mark close with <=60s age",
        "C_read_before_AB_freeze": False,
    })
    contracts_dir.mkdir()
    ab_contract_path = _write_stage_contract(output_root, "development", ab_contract_data)

    baseline = _run_development_baseline(ab_contract_path, output_root / "development_full_pool")
    require(baseline.get("status") == "engineering_complete", "A+B full-pool baseline did not complete")
    baseline_root = Path(baseline.get("root", ""))
    require(baseline_root.is_absolute() and baseline_root.is_dir(), "A+B baseline artifact root is missing")

    session_contract = AgentSessionContract.from_dict({
        "schema_version": 1,
        "run_id": f"{contract.run_id}-agent",
        "experiment_budget": contract.experiment_budget,
        "context_bytes": contract.context_bytes,
    })
    agent_result = agent_runner(
        baseline_root,
        session_contract,
        output_root / "development_agent",
        model,
        model_mode=model_mode,
        model_settings=model_settings,
    )
    agent_trials, fixed_trials, session_root = _require_agent_session(agent_result, session_contract)
    agent_session_path = str(session_root.resolve().relative_to(output_root.resolve()))
    _write_json(output_root / "agent-session-result.json", agent_result)

    snapshot = load_baseline_snapshot(baseline_root)
    candidates = _collect_ab_candidates(agent_trials, fixed_trials, session_root, snapshot,
                                        contract, output_root, pool_ids, factor_codes,
                                        baseline, baseline_root)
    freeze = _select_ab_candidate(candidates)
    freeze["baseline_run_id"] = snapshot.contract.run_id
    freeze["qualification_gates"] = contract.qualification_gates
    freeze["prior_candidate_results_hidden_from_agent"] = agent_result["sealed_initial_drop_results"]
    _write_json(output_root / "candidate-trials.json", candidates)
    _write_json(output_root / "candidate-freeze.json", freeze)
    candidate_freeze_sha256 = _sha256(output_root / "candidate-freeze.json")

    c_contract_data = _stage_contract(contract, stage="internal_validation", card_paths=card_paths,
                                      universe_path=universe_path, manifest_path=manifest_path,
                                      stage_contract_dir=contracts_dir)
    c_contract_path = _write_stage_contract(output_root, "internal_validation", c_contract_data)
    validation = _run_validation_stage(c_contract_path, c_contract_data, manifest_path,
                                       card_paths, universe_path, freeze["selected_card_ids"],
                                       output_root / "validation", contract.qualification_gates,
                                       candidate_freeze_sha256)

    c_pass = validation["qualified_c"]
    qualified = freeze["selected_ab_qualified"] and c_pass
    final_status = "qualified_candidate_frozen" if qualified else "no_candidate_passed"
    artifacts = {
        "contract": _artifact_record(output_root, output_root / "contract.json"),
        "code_version": _artifact_record(output_root, output_root / "code_version.json"),
        "dataset_manifest": _artifact_record(output_root, output_root / "dataset_manifest.json"),
        "dataset_provenance": _artifact_record(output_root, output_root / "dataset_provenance.json"),
        "source_gate": _artifact_record(output_root, output_root / "source_gate.json"),
        "universe": _artifact_record(output_root, output_root / "universe.csv"),
        "cards_manifest": _artifact_record(output_root, output_root / "cards.json"),
        "development_contract": _artifact_record(output_root, output_root / "contracts/development.json"),
        "internal_validation_contract": _artifact_record(
            output_root, output_root / "contracts/internal_validation.json"),
        "development_baseline_result": _artifact_record(output_root, baseline_root / "result.json"),
        "agent_session_result": _artifact_record(output_root, session_root / "result.json"),
        "agent_session_report": _artifact_record(output_root, session_root / "report.md"),
        "candidate_trials": _artifact_record(output_root, output_root / "candidate-trials.json"),
        "candidate_freeze": _artifact_record(output_root, output_root / "candidate-freeze.json"),
        "internal_validation_result": _artifact_record(
            output_root, output_root / "validation/result.json"),
    }
    result = {
        "schema_version": 1,
        "engine": "multifactor_historical_v1",
        "run_id": contract.run_id,
        "status": final_status,
        "root": str(output_root.resolve()),
        "code_version": code_version,
        "stage_run_ids": {
            "historical": contract.run_id,
            "development_baseline": baseline["run_id"],
            "agent_session": session_contract.run_id,
            "internal_validation": c_contract_data["run_id"],
        },
        "purpose": "research",
        "dataset_provenance": provenance,
        "full_pool_baseline": baseline,
        "agent_session": {
            "root": agent_result["root"],
            "path": agent_session_path,
            "model_mode": agent_result.get("model_mode"),
            "model_calls": agent_result["model_calls"],
            "max_model_calls": session_contract.max_model_calls,
            "agent_experiment_count": len(agent_trials),
            "fixed_experiment_count": len(fixed_trials),
            "initial_drop_results_sealed": agent_result["sealed_initial_drop_results"],
            "fixed_results_exposed_to_agent": False,
            "independent_agent_value_established": False,
        },
        "trials": candidates,
        "candidates": candidates,
        "candidate_freeze": freeze,
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "internal_validation": validation,
        "selected_candidate": {
            "card_ids": freeze["selected_card_ids"],
            "trial_id": freeze["selected_trial_id"],
            "reference_equals_candidate": validation["reference_equals_candidate"],
            "candidate_artifact_ref": validation["candidate_artifact_ref"],
            "equal_weight_artifact_ref": validation["equal_weight_artifact_ref"],
            "qualified_ab": freeze["selected_ab_qualified"],
            "ab_metrics": freeze["selected_ab_metrics"],
            "ab_traded_bars": freeze["selected_ab_traded_bars"],
            "ab_stress_metrics": freeze["selected_ab_stress_metrics"],
            "ab_qualification_failures": freeze["selected_ab_qualification_failures"],
            "qualified_c": c_pass,
            "c_metrics": validation["candidate"]["metrics"],
            "c_traded_bars": validation["traded_bars"],
            "c_stress_metrics": validation["candidate_stress_costs"]["metrics"],
            "c_qualification_failures": validation["qualification_failures"],
            "qualification_gates": contract.qualification_gates,
            "qualified": qualified,
        },
        "selected_candidates": [freeze["selected_card_ids"]] if qualified else [],
        "forward_readiness": {
            "ready_for_separate_forward_review": qualified,
            "forward_validation_started": False,
            "paper_started": False,
            "published": False,
        },
        "c_data_accessed_after_candidate_freeze": True,
        "c_results_returned_to_agent": False,
        "reference_equals_candidate": validation["reference_equals_candidate"],
        "agent_used": True,
        "independent_agent_value_established": False,
        "historical_causality_certified": provenance["historical_causality_certified"],
        "artifacts": artifacts,
    }
    _write_json(output_root / "result.json", result)
    _report(output_root, contract, provenance, baseline, candidates,
            freeze, validation, agent_session_path, code_version, final_status)
    return result
