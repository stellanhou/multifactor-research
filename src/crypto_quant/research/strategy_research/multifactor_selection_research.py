"""Frozen multi-route and cross-horizon multifactor selection research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, require
from crypto_quant.research.progress import ProgressLog
from .multifactor_account import ContinuousTargetPolicy, run_perpetual_account
from .multifactor_contracts import ResearchContract
from .multifactor_historical import _candidate_failures
from .multifactor_horizon import combine_horizon_targets
from .multifactor_rebalance import _behavior_metrics, _decisions, _run_arm
from .multifactor_selection import generate_selection_scores
from .multifactor_workflow import _account_kwargs, _write_account_result, _write_json


HORIZONS = (1, 4, 24)
STAGES = ("development", "internal_validation")
METHODS = ("equal", "rolling_icir", "cluster", "ridge", "elastic_net", "pool")
STEPWISE_METHODS = ("stepwise",)
NET_SHARPE_METHODS = ("stepwise_net_sharpe",)
REBALANCE_ROUTES = (
    {"name": "R0", "rank_buffer": False, "position_buffer": False, "hourly": False},
    {"name": "R4", "rank_buffer": True, "position_buffer": True, "hourly": True},
)
SELECTION_POLICY = {
    "fit_window_hours": 2160,
    "min_train_periods": 720,
    "refit_every_hours": 168,
    "min_cross_section_symbols": 3,
    "icir_window_hours": 720,
    "icir_min_periods": 168,
    "correlation_window_hours": 720,
    "cluster_abs_correlation_threshold": 0.8,
    "inner_validation_hours": 504,
    "min_inner_validation_periods": 168,
    "alpha_grid": [0.00001, 0.001, 0.1, 10.0],
    "l1_ratio": 0.5,
    "pool_capacity": 8,
    "pool_search_budget": 32,
    "min_objective_improvement": 0.000001,
}
STEPWISE_POLICY = {**SELECTION_POLICY, "pool_search_budget": 2048}
OUTER_POLICIES = (
    {
        "method": "fixed",
        "gross_exposure": 0.8,
        "max_asset_weight": 0.2,
        "lookback_hours": 720,
        "min_history": 168,
        "mean_gate": False,
        "covariance_shrinkage": 0.1,
    },
    {
        "method": "risk_budget",
        "gross_exposure": 0.8,
        "max_asset_weight": 0.2,
        "lookback_hours": 720,
        "min_history": 168,
        "mean_gate": True,
        "covariance_shrinkage": 0.1,
    },
)
MODEL_POLICY_FIELDS = set(SELECTION_POLICY)


@dataclass(frozen=True)
class SelectionResearchContract:
    schema_version: int
    run_id: str
    source_run: str
    methods: list[str]
    rebalance_routes: list[dict[str, Any]]
    selection_policy: dict[str, Any]
    second_layer: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SelectionResearchContract:
        fields = set(cls.__dataclass_fields__)
        require(isinstance(value, dict) and set(value) == fields,
                "selection research contract fields differ from the schema")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "selection research schema_version must be 1")
        identifier(value["run_id"])
        require(isinstance(value["source_run"], str) and value["source_run"].strip(),
                "source_run is required")
        require(value["methods"] in (list(METHODS), list(STEPWISE_METHODS), list(NET_SHARPE_METHODS)),
                "selection methods must be the original six routes, stepwise, or net-Sharpe stepwise")
        require(value["rebalance_routes"] == list(REBALANCE_ROUTES),
                "rebalance routes are frozen to R0 and R4")
        policy = value["selection_policy"]
        require(isinstance(policy, dict) and set(policy) == MODEL_POLICY_FIELDS,
                "selection_policy fields differ from the frozen schema")
        expected_policy = SELECTION_POLICY if value["methods"] == list(METHODS) else STEPWISE_POLICY
        require(policy == expected_policy,
                "selection model parameters differ from the frozen research plan")
        second = value["second_layer"]
        require(isinstance(second, dict) and set(second) == {"continuous_target_weight_buffer", "policies"},
                "second_layer fields differ from the frozen schema")
        require(type(second["continuous_target_weight_buffer"]) in {int, float}
                and second["continuous_target_weight_buffer"] == 0.02,
                "continuous_target_weight_buffer is frozen at 0.02")
        require(second["policies"] == list(OUTER_POLICIES),
                "cross-horizon policies differ from the frozen research plan")
        return cls(**value)


@dataclass(frozen=True)
class StageInputs:
    contract: ResearchContract
    factors: pd.DataFrame
    opens: pd.Series
    frames: dict[str, pd.DataFrame]
    funding: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _utc_timestamps(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, format="mixed", utc=True))


def _read_indexed_panel(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    usecols = None if columns is None else ["timestamp", "symbol", *columns]
    frame = pd.read_csv(path, usecols=usecols, float_precision="round_trip")
    require({"timestamp", "symbol"}.issubset(frame.columns),
            f"panel source lacks timestamp/symbol fields: {path}")
    frame["timestamp"] = _utc_timestamps(frame["timestamp"])
    panel = frame.set_index(["timestamp", "symbol"]).sort_index()
    panel.index = panel.index.set_names(["timestamp", "symbol"])
    require(panel.index.is_unique and panel.index.is_monotonic_increasing,
            f"panel source index must be unique and sorted: {path}")
    return panel


def _read_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    frame.index = _utc_timestamps(frame.index)
    frame.index.name = "timestamp"
    require(frame.index.is_unique and frame.index.is_monotonic_increasing,
            f"market timestamps must be unique and sorted: {path}")
    require({"open", "close", "mark_close"}.issubset(frame.columns),
            f"market data lacks required account fields: {path}")
    return frame


def _read_funding(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"timestamp", "symbol", "funding_rate", "mark_price"}
    require(required.issubset(frame.columns), f"funding data lacks required fields: {path}")
    frame["timestamp"] = _utc_timestamps(frame["timestamp"])
    frame = frame.loc[:, ["timestamp", "symbol", "funding_rate", "mark_price"]]
    require(not frame.duplicated(["timestamp", "symbol"]).any(),
            f"funding event keys must be unique within a stage: {path}")
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _load_stage_inputs(source: Path, stage: str, horizon: int,
                       pool: dict[str, Any]) -> tuple[StageInputs, list[Path]]:
    stage_root = source / stage
    contract_path = stage_root / f"h{horizon}" / "contract.json"
    contract = ResearchContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
    require(contract.stage == stage and contract.horizon_hours == horizon,
            f"source {stage}/h{horizon} contract does not match its path")
    require(contract.start == (pool["development_start"] if stage == "development"
                               else pool["validation_start"]),
            f"source {stage}/h{horizon} start differs from the frozen pool split")
    require(contract.end == (pool["validation_start"] if stage == "development"
                             else pool["validation_end"]),
            f"source {stage}/h{horizon} end differs from the frozen pool split")
    require(contract.warmup_hours == pool["warmup_hours"],
            f"source {stage}/h{horizon} warmup differs from the frozen pool contract")
    require(contract.costs == pool["costs"],
            f"source {stage}/h{horizon} costs differ from the frozen pool contract")
    require(contract.portfolio == {**pool["portfolio"], "rebalance_hours": horizon},
            f"source {stage}/h{horizon} portfolio differs from the frozen pool contract")
    factors_path = stage_root / f"h{horizon}" / "factor_panels" / "standardized.csv"
    factors = _read_indexed_panel(factors_path)
    require(len(factors.columns) > 0 and not factors.columns.has_duplicates,
            f"source factor columns must be nonempty and unique: {factors_path}")
    require(not np.isinf(factors.to_numpy(dtype=float)).any(),
            f"standardized factor panel contains infinity: {factors_path}")

    panel_path = stage_root / "inputs" / "panel.values.csv"
    panel = _read_indexed_panel(panel_path, ["perp_open"])
    require(panel.index.equals(factors.index),
            f"factor and opening-price panels differ in {stage}/h{horizon}")
    opens = panel["perp_open"].astype(float).rename("perp_open")
    prices = opens.to_numpy(dtype=float)
    require(not np.isinf(prices).any() and np.all(prices[np.isfinite(prices)] > 0),
            f"opening prices must be finite when present and positive: {panel_path}")

    market_dir = stage_root / "inputs" / "market"
    market_paths = sorted(market_dir.glob("*.csv"))
    require(bool(market_paths), f"source market directory is empty: {market_dir}")
    frames = {path.stem: _read_market(path) for path in market_paths}
    funding_path = stage_root / "inputs" / "funding.csv"
    funding = _read_funding(funding_path)
    used = [contract_path, factors_path, panel_path, funding_path, *market_paths]
    return StageInputs(contract, factors, opens, frames, funding), used


def _compare_overlap(left: pd.Series | pd.DataFrame, right: pd.Series | pd.DataFrame,
                     *, label: str, exact: bool) -> dict[str, Any]:
    overlap = left.index.intersection(right.index)
    if not len(overlap):
        return {
            "overlap_rows": 0,
            "availability_mismatches": 0,
            "different_values": 0,
            "max_absolute_difference": 0.0,
        }
    a, b = left.loc[overlap], right.loc[overlap]
    if exact:
        require(a.equals(b), f"{label} differs across the development/validation overlap")
        return {"overlap_rows": int(len(overlap)), "availability_mismatches": 0,
                "different_values": 0,
                "max_absolute_difference": 0.0}
    a_values, b_values = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    finite_a, finite_b = np.isfinite(a_values), np.isfinite(b_values)
    paired = finite_a & finite_b
    difference = np.abs(a_values - b_values)
    different = paired & (difference > 1e-12)
    return {
        "overlap_rows": int(len(overlap)),
        "availability_mismatches": int((finite_a != finite_b).sum()),
        "different_values": int(different.sum()),
        "max_absolute_difference": float(difference[paired].max()) if paired.any() else 0.0,
    }


def _combine_model_inputs(dev: StageInputs, validation: StageInputs, *,
                          min_symbols: int, horizon: int) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], pd.DataFrame]:
    require(list(dev.factors.columns) == list(validation.factors.columns),
            f"h{horizon} factor universe differs across stages")
    require(dev.opens.index.get_level_values("symbol").unique().sort_values().equals(
        validation.opens.index.get_level_values("symbol").unique().sort_values()),
        f"h{horizon} symbol universe differs across stages")
    open_overlap = _compare_overlap(dev.opens, validation.opens, label=f"h{horizon} opens", exact=True)
    factor_overlap = _compare_overlap(dev.factors, validation.factors,
                                      label=f"h{horizon} factors", exact=False)

    new_factor_rows = validation.factors.loc[~validation.factors.index.isin(dev.factors.index)]
    new_open_rows = validation.opens.loc[~validation.opens.index.isin(dev.opens.index)]
    factors = pd.concat([dev.factors, new_factor_rows]).sort_index()
    opens = pd.concat([dev.opens, new_open_rows]).sort_index()
    require(factors.index.is_unique and factors.index.equals(opens.index),
            f"h{horizon} continuous model panel is not unique and aligned")
    times = factors.index.get_level_values("timestamp")
    require(times.unique().equals(pd.date_range(times.min(), times.max(), freq="h", tz="UTC")),
            f"h{horizon} model timestamps are not a complete hourly grid")

    complete = np.isfinite(factors.to_numpy(dtype=float)).all(axis=1)
    counts = pd.Series(complete, index=factors.index).groupby(level="timestamp", sort=True).sum()
    sufficient = counts >= min_symbols
    eligible_rows = complete & times.map(sufficient).to_numpy(dtype=bool)
    common = factors.copy()
    common.iloc[~eligible_rows, :] = np.nan
    availability = pd.DataFrame({
        "complete_factor_symbols": counts.astype(int),
        "meets_minimum_cross_section": sufficient,
    })
    availability.index.name = "timestamp"
    return common, opens, {"open_overlap": open_overlap, "factor_overlap": factor_overlap,
                           "overlap_preference": "development rows retained"}, availability


def _load_all_inputs(source: Path, pool: dict[str, Any]) -> tuple[
        dict[int, dict[str, StageInputs]], dict[int, tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]],
        list[Path]]:
    stage_inputs: dict[int, dict[str, StageInputs]] = {}
    all_paths: list[Path] = [source / "pool_contract.json", source / "dataset_manifest.json"]
    for horizon in HORIZONS:
        stage_inputs[horizon] = {}
        for stage in STAGES:
            loaded, used = _load_stage_inputs(source, stage, horizon, pool)
            stage_inputs[horizon][stage] = loaded
            all_paths.extend(used)
    model_inputs = {}
    for horizon in HORIZONS:
        model_inputs[horizon] = _combine_model_inputs(
            stage_inputs[horizon]["development"], stage_inputs[horizon]["internal_validation"],
            min_symbols=SELECTION_POLICY["min_cross_section_symbols"], horizon=horizon,
        )
    return stage_inputs, model_inputs, all_paths


def _source_file_paths(source: Path) -> list[Path]:
    paths = [source / "pool_contract.json", source / "plan.json", source / "dataset_manifest.json"]
    engineering_provenance = source / "engineering_provenance.json"
    if engineering_provenance.is_file():
        paths.append(engineering_provenance)
    for stage in STAGES:
        stage_root = source / stage
        paths.extend([
            stage_root / "inputs" / "panel.values.csv",
            stage_root / "inputs" / "funding.csv",
            *sorted((stage_root / "inputs" / "market").glob("*.csv")),
        ])
        for horizon in HORIZONS:
            paths.extend([
                stage_root / f"h{horizon}" / "contract.json",
                stage_root / f"h{horizon}" / "factor_panels" / "standardized.csv",
            ])
    missing = [path for path in paths if not path.is_file()]
    require(not missing, f"source run is missing required input files: {missing[:3]}")
    return paths


def _combined_frames_and_funding(stages: dict[str, StageInputs]) -> tuple[
        dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    development, validation = stages["development"], stages["internal_validation"]
    require(set(development.frames) == set(validation.frames),
            "market symbol sets differ across the two account stages")
    result, market_overlap = {}, {}
    for symbol in sorted(development.frames):
        left, right = development.frames[symbol], validation.frames[symbol]
        overlap = left.index.intersection(right.index)
        if len(overlap):
            require(left.loc[overlap].equals(right.loc[overlap]),
                    f"{symbol} market bars differ across the stage boundary")
        market_overlap[symbol] = {"overlap_rows": int(len(overlap)), "exact_match": True}
        tail = right.loc[~right.index.isin(left.index)]
        joined = pd.concat([left, tail]).sort_index()
        require(joined.index.is_unique, f"{symbol} combined market index is not unique")
        result[symbol] = joined
    left_funding, right_funding = development.funding, validation.funding
    key = ["timestamp", "symbol"]
    left_keys = pd.MultiIndex.from_frame(left_funding[key])
    right_keys = pd.MultiIndex.from_frame(right_funding[key])
    overlap = left_keys.intersection(right_keys)
    if len(overlap):
        left_shared = left_funding.set_index(key).loc[overlap].sort_index()
        right_shared = right_funding.set_index(key).loc[overlap].sort_index()
        require(left_shared.equals(right_shared), "funding events differ across the stage boundary")
    right_tail = right_funding.loc[~right_funding.set_index(key).index.isin(left_keys)]
    funding = pd.concat([left_funding, right_tail], ignore_index=True)
    funding = funding.sort_values(key, kind="stable").reset_index(drop=True)
    require(not funding.duplicated(key).any(), "combined funding event keys are not unique")
    return result, funding, {
        "market_overlap": market_overlap,
        "funding_overlap_rows": int(len(overlap)),
        "funding_overlap_exact": True,
    }


def _generate_model_scores(request: SelectionResearchContract, factors: pd.DataFrame,
                           opens: pd.Series, *, horizon: int,
                           stages: dict[str, StageInputs]):
    if request.methods == list(NET_SHARPE_METHODS):
        from .multifactor_net_sharpe import generate_net_sharpe_scores

        frames, funding, _ = _combined_frames_and_funding(stages)
        return generate_net_sharpe_scores(
            factors, opens, horizon_hours=horizon, policy=request.selection_policy,
            frames=frames, funding=funding, contract=stages["development"].contract,
        )
    return generate_selection_scores(
        factors, opens, horizon_hours=horizon, policy=request.selection_policy,
        methods=request.methods,
    )


def _validate_shared_stage_inputs(stages: dict[int, dict[str, StageInputs]]) -> dict[str, Any]:
    reference = stages[HORIZONS[0]]
    overlap_audit = _combined_frames_and_funding(reference)[2]
    for horizon in HORIZONS[1:]:
        for stage in STAGES:
            current, expected = stages[horizon][stage], reference[stage]
            require(set(current.frames) == set(expected.frames),
                    f"h{horizon} {stage} market symbols differ across horizons")
            for symbol in expected.frames:
                require(current.frames[symbol].equals(expected.frames[symbol]),
                        f"{symbol} market inputs differ across horizons")
            require(current.funding.equals(expected.funding),
                    f"h{horizon} {stage} funding differs across horizons")
            require(current.opens.equals(expected.opens),
                    f"h{horizon} {stage} opening-price inputs differ across horizons")
    return overlap_audit


def _source_contracts(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pool_path = source / "pool_contract.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    require(isinstance(pool, dict) and set(pool) >= {
        "schema_version", "run_id", "horizons", "warmup_hours", "development_start",
        "validation_start", "validation_end", "costs", "portfolio", "qualification_gates",
    }, "source pool contract is missing required fields")
    require(pool["horizons"] == list(HORIZONS), "source horizons must be 1, 4, and 24 hours")
    require(type(pool["schema_version"]) is int and pool["schema_version"] == 2,
            "selection research requires the source rolling-ICIR pool contract")
    require(set(pool["costs"]) == {
        "initial_capital", "fee_bps", "slippage_bps", "stress_multiplier",
    } and pool["costs"] == {
        "initial_capital": 10000.0, "fee_bps": 10.0,
        "slippage_bps": 5.0, "stress_multiplier": 2.0,
    }, "source costs differ from the frozen historical account contract")
    require(pool["portfolio"].get("long_count") == 2
            and pool["portfolio"].get("short_count") == 2
            and pool["portfolio"].get("gross_exposure") == 0.8
            and pool["portfolio"].get("max_asset_weight") == 0.2
            and pool["portfolio"].get("margin_fraction") == 0.1,
            "source portfolio differs from the frozen 2x2, 0.8 gross plan")
    gates = pool["qualification_gates"]
    require(set(gates) == {"min_net_return", "max_drawdown", "min_traded_bars", "min_stress_return"},
            "source qualification gates differ from the account schema")
    require(0 < gates["max_drawdown"] < 1 and type(gates["min_traded_bars"]) is int
            and gates["min_traded_bars"] > 0,
            "source qualification gates are invalid")
    for name in ("min_net_return", "max_drawdown", "min_stress_return"):
        value = gates[name]
        require(type(value) in {int, float} and np.isfinite(value),
                f"source qualification gate {name} must be finite numeric")
    plan_path = source / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan.get("horizons") == list(HORIZONS) and plan.get("stages") == list(STAGES),
            "source run plan differs from the frozen horizon/stage grid")
    if plan.get("engineering_smoke_only") is True:
        require((source / "engineering_provenance.json").is_file(),
                "engineering smoke source must include its provenance manifest")
    return pool, {"pool_contract": pool_path, "source_plan": plan_path}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_safe(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, _json_safe(value))


def _capture_sources(root: Path) -> dict[str, str]:
    project = Path(__file__).resolve().parents[4]
    relative_paths = [
        "src/crypto_quant/research/strategy_research/multifactor_selection_research.py",
        "src/crypto_quant/research/strategy_research/multifactor_selection.py",
        "src/crypto_quant/research/strategy_research/multifactor_net_sharpe.py",
        "src/crypto_quant/research/strategy_research/multifactor_horizon.py",
        "src/crypto_quant/research/strategy_research/multifactor_account.py",
        "src/crypto_quant/research/strategy_research/multifactor_rebalance.py",
        "src/crypto_quant/research/strategy_research/multifactor_ablation.py",
        "src/crypto_quant/research/strategy_research/multifactor_pool_research.py",
        "src/crypto_quant/research/strategy_research/multifactor_combination.py",
        "src/crypto_quant/research/strategy_research/multifactor_contracts.py",
        "src/crypto_quant/research/strategy_research/multifactor_historical.py",
        "src/crypto_quant/research/strategy_research/multifactor_workflow.py",
        "src/crypto_quant/research/strategy_research/multifactor_data.py",
        "src/crypto_quant/research/strategy_research/multifactor_factors.py",
        "src/crypto_quant/research/strategy_research/cli.py",
        "src/crypto_quant/research/factor_mining/contracts.py",
        "src/crypto_quant/research/progress.py",
        "src/crypto_quant/features/factor_inputs.py",
        "pyproject.toml",
        "uv.lock",
    ]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=project, check=True,
                            capture_output=True, text=True).stdout.splitlines()
    manifest = {}
    for relative in relative_paths:
        source = project / relative
        require(source.is_file(), f"source code snapshot input is missing: {relative}")
        destination = root / "execution_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        require(_sha256(destination) == digest, "source code changed while being snapshotted")
        manifest[relative] = digest
    _write_json(root / "execution_source_manifest.json", manifest)
    _write_json(root / "code_version.json", {
        "git_commit": commit,
        "working_tree_clean": not status,
        "working_tree_status": status,
        "source_sha256": manifest,
        "captured_before_source_panel_load": True,
    })
    _write_json(root / "runtime.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy")
        },
    })
    return manifest


def _build_plan(request: SelectionResearchContract, source: Path,
                pool: dict[str, Any], source_meta: dict[str, Path]) -> dict[str, Any]:
    source_plan = json.loads(source_meta["source_plan"].read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "run_id": request.run_id,
        "source_run": str(source),
        "source_pool_contract_sha256": _sha256(source_meta["pool_contract"]),
        "source_plan_sha256": _sha256(source_meta["source_plan"]),
        "methods": request.methods,
        "horizons": list(HORIZONS),
        "stages": list(STAGES),
        "stage_dates": {
            "development": [pool["development_start"], pool["validation_start"]],
            "internal_validation": [pool["validation_start"], pool["validation_end"]],
        },
        "source_provenance": {
            "source_run_id": pool["run_id"],
            "engineering_smoke_only": source_plan.get("engineering_smoke_only", False),
        },
        "first_layer": {
            "routes": request.rebalance_routes,
            "account_runs": len(request.methods) * len(HORIZONS) * len(request.rebalance_routes) * 2 * len(STAGES),
            "position_policy": {"holding_rank": 4, "weight_buffer": 0.02},
            "cost_paths": {"base": 1.0, "stress": pool["costs"]["stress_multiplier"]},
        },
        "second_layer": {
            "account_runs": len(request.methods) * len(request.rebalance_routes) * len(OUTER_POLICIES) * 2 * len(STAGES),
            "policies": list(OUTER_POLICIES),
            "target_policy": {
                "kind": "continuous_signed_weights",
                "weight_buffer": request.second_layer["continuous_target_weight_buffer"],
                "cost_paths_decided_independently": True,
            },
            "return_timing": "ledger bar_end timestamps are retained; allocation at signal t uses rows strictly before t",
            "risk_budget_history": "development and earlier internal-validation child returns; no backward shifting",
            "risk_budget_label": "mean-gated risk budget; fixed and risk-budget differ in both covariance allocation and positive-mean gating",
        },
        "account_runs": (
            len(request.methods) * len(HORIZONS) * len(request.rebalance_routes) * 2 * len(STAGES)
            + len(request.methods) * len(request.rebalance_routes) * len(OUTER_POLICIES) * 2 * len(STAGES)
        ),
        "qualification_gates": pool["qualification_gates"],
        "costs": pool["costs"],
        "portfolio": pool["portfolio"],
        "selection_policy": request.selection_policy,
        "model_objective": (
            "inner-validation net-Sharpe improvement; complete best-forward rounds followed by iterative best-backward removal; final selected subset must have positive inner net return"
            if request.methods == list(NET_SHARPE_METHODS) else
            "inner-validation R2 improvement; complete best-forward rounds followed by iterative best-backward removal; each accepted change exceeds 1e-6"
            if request.methods == list(STEPWISE_METHODS) else
            "inner-validation R2 improvement; finite-pool accepts each sequential marginal gain only above 1e-6"
        ),
        **({"inner_account_objective": {
            "metric": "net_sharpe", "alpha_selection_metric": "net_sharpe",
            "risk_free_rate": 0.0, "return_frequency": "hourly", "annualization_hours": 8760,
            "validation_source_hours": request.selection_policy["inner_validation_hours"],
            "rebalance_route": "R0", "rebalance_phase": "development contract start minus one hour",
            "account_initial_state": "original initial capital, zero positions in every validation window",
            "last_signal": "last matured validation source bar; hold resulting positions until fit time",
            "account_end": "fit time exclusive; last market bar closes at fit time; funding timestamp strictly before fit time",
            "costs": "base fees and slippage; actual signed funding cashflows",
            "final_admission": "positive net return and a trading path; cash is not an eligible candidate",
        }} if request.methods == list(NET_SHARPE_METHODS) else {}),
        "factor_availability": "standardized factor rows before the legacy B/D paired mask; only complete-factor rows with at least three symbols enter models",
        "overlap": "continuous development-to-validation model history; retain development factor/open rows at duplicate timestamps; require exact duplicate opens and market bars",
        "model_readiness": "shared causal readiness independent of route-specific cash abstentions",
        "funding_timing": "preserve original account engine: decision at signal close, execution at next open, funding at execution open charges old holdings",
        "stage_accounting": "each development and internal-validation account starts with the original initial capital; outer model history remains continuous across stages",
        "comparisons": "run every predeclared method, horizon, route, cost, and stage; no C-based re-selection",
        "frozen_before_account_outcomes": True,
        "model_calls": 0,
        "paper_started": False,
        "forward_validation_started": False,
    }


def _account_signal_grid(score: pd.Series, contract: ResearchContract,
                         symbols: list[str]) -> pd.DataFrame:
    start, end = contract.bounds
    grid = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2),
                         freq="h", tz="UTC", name="signal_timestamp")
    timestamp = score.index.get_level_values("timestamp")
    selected = score.loc[(timestamp >= grid[0]) & (timestamp <= grid[-1])]
    panel = selected.unstack("symbol").reindex(index=grid, columns=symbols)
    panel.columns.name = "symbol"
    require(panel.index.equals(grid) and list(panel.columns) == symbols,
            "method scores do not cover the frozen account signal grid")
    return panel


def _write_model_outputs(root: Path, horizon: int, scores: dict[str, pd.Series],
                         fits: list[dict[str, Any]], readiness: pd.DataFrame,
                         overlap: dict[str, Any], factors: pd.DataFrame) -> None:
    directory = root / "models" / f"h{horizon}"
    directory.mkdir(parents=True)
    score_panel = pd.DataFrame(scores, index=factors.index)
    score_panel.to_csv(directory / "scores.csv")
    score_panel.notna().to_csv(directory / "availability.csv")
    readiness.to_csv(directory / "factor_availability.csv")
    _write_json_safe(directory / "fits.json", fits)
    _write_json(directory / "overlap_audit.json", overlap)
    summary = {
        "factor_columns": list(factors.columns),
        "factor_column_count": len(factors.columns),
        "model_rows": len(factors),
        "model_timestamps": int(factors.index.get_level_values("timestamp").nunique()),
        "shared_ready_timestamps": int(readiness["shared_model_ready"].sum()),
        "route_available_rows": {
            method: int(score_panel[method].notna().sum()) for method in scores
        },
        "route_available_timestamps": {
            method: int(score_panel[method].notna().groupby(level="timestamp").any().sum())
            for method in scores
        },
    }
    _write_json(directory / "summary.json", summary)


def _execute_first_layer(root: Path, request: SelectionResearchContract,
                         stages: dict[int, dict[str, StageInputs]],
                         model_scores: dict[int, dict[str, pd.Series]],
                         pool: dict[str, Any], progress: ProgressLog, *,
                         horizons: tuple[int, ...] = HORIZONS) -> list[dict[str, Any]]:
    trials = []
    for horizon in horizons:
        for stage in STAGES:
            inputs = stages[horizon][stage]
            contract = inputs.contract
            symbols = sorted(inputs.frames)
            for method in request.methods:
                signals = _account_signal_grid(model_scores[horizon][method], contract, symbols)
                if method == request.methods[0]:
                    signal_grid = signals.index
                else:
                    require(signals.index.equals(signal_grid),
                            "all methods must use the same account time grid")
                for route in request.rebalance_routes:
                    path_results = []
                    account_root = root / stage / f"h{horizon}" / method / route["name"]
                    for cost_name, multiplier in (("base", 1.0), ("stress", pool["costs"]["stress_multiplier"])):
                        account_path = account_root / cost_name
                        with progress.span("selection.first_layer_account", heartbeat=True,
                                           stage=stage, horizon=horizon, method=method,
                                           route=route["name"], cost=cost_name):
                            account, result = _run_arm(
                                account_path, signals, inputs.frames, inputs.funding, contract,
                                route, {"holding_rank": 4, "weight_buffer": 0.02}, multiplier,
                            )
                        path_results.append({"cost": cost_name, **result})
                    failures = _candidate_failures(
                        path_results[0]["metrics"], path_results[1]["metrics"],
                        path_results[0]["metrics"]["traded_bars"], pool["qualification_gates"],
                    )
                    trial = {
                        "layer": "single_horizon",
                        "method": method,
                        "route": route["name"],
                        "horizon_hours": horizon,
                        "stage": stage,
                        "metrics": path_results[0]["metrics"],
                        "path": path_results[0]["path"],
                        "stress_costs": path_results[1],
                        "qualification_failures": failures,
                        "qualified": not failures,
                        "score_availability": {
                            "available_symbol_rows": int(signals.notna().sum().sum()),
                            "available_hours": int(signals.notna().any(axis=1).sum()),
                            "account_grid_hours": len(signals),
                        },
                    }
                    _write_json(root / stage / f"h{horizon}" / method / route["name"] / "result.json", trial)
                    trials.append(trial)
                    for path_result in path_results:
                        saved_target = Path(path_result["path"]) / "targets.csv"
                        require(saved_target.is_file(), "inner account target history is missing")
    return trials


def _read_inner_history(root: Path, horizon: int, method: str, route: str,
                        stage: str, cost: str, contract: ResearchContract,
                        symbols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    path = root / stage / f"h{horizon}" / method / route / cost
    targets = pd.read_csv(path / "targets.csv", index_col=0, parse_dates=[0],
                          float_precision="round_trip")
    targets.index = _utc_timestamps(targets.index)
    targets.index.name = "signal_timestamp"
    require(not targets.columns.has_duplicates and set(targets.columns) == set(symbols),
            "inner account target symbols differ from the account universe")
    require(targets.index.is_unique and targets.index.is_monotonic_increasing
            and np.isfinite(targets.to_numpy(dtype=float)).all(),
            "inner account targets must be unique, sorted, and finite")
    targets = targets.loc[:, symbols]
    start, end = contract.bounds
    grid = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2),
                         freq="h", tz="UTC", name="signal_timestamp")
    require(len(targets.index) > 0 and targets.index[0] == grid[0],
            "inner account targets are missing the initial signal row")
    require(targets.index.isin(grid).all(), "inner target timestamps lie outside the account grid")
    expanded = targets.reindex(grid).ffill()
    require(expanded.index.equals(grid) and np.isfinite(expanded.to_numpy(dtype=float)).all(),
            "inner targets do not form a finite account signal grid")
    ledger = pd.read_csv(path / "ledger.csv", index_col=0, parse_dates=[0],
                         float_precision="round_trip")
    ledger.index = _utc_timestamps(ledger.index)
    ledger.index.name = "timestamp"
    require("return" in ledger.columns, "inner account ledger has no realized return column")
    return expanded, ledger["return"].astype(float)


def _run_outer_account(directory: Path, targets: pd.DataFrame, frames: dict[str, pd.DataFrame],
                       funding: pd.DataFrame, contract: ResearchContract, multiplier: float,
                       weight_buffer: float) -> tuple[Any, dict[str, Any]]:
    directory.mkdir(parents=True)
    targets.to_csv(directory / "targets.csv")
    account = run_perpetual_account(
        frames, targets, funding,
        **_account_kwargs(contract, contract.costs["fee_bps"] * multiplier,
                          contract.costs["slippage_bps"] * multiplier),
        continuous_target_policy=ContinuousTargetPolicy(
            gross_exposure=contract.portfolio["gross_exposure"],
            max_asset_weight=contract.portfolio["max_asset_weight"],
            weight_buffer=weight_buffer,
        ),
    )
    _write_account_result(directory, account)
    audit = _decisions(account, None)
    audit.to_csv(directory / "decisions.csv", index=False)
    metrics, categories = _behavior_metrics(
        account, audit, contract.costs["initial_capital"],
    )
    categories.to_csv(directory / "turnover_categories.csv")
    _write_json(directory / "metrics.json", metrics)
    ledger = account.ledger
    monthly = pd.DataFrame({
        "net_return": (1 + ledger["return"]).resample("ME").prod() - 1,
        **{name: ledger[name].resample("ME").sum()
           for name in ("fees", "slippage_cost", "funding_cashflow", "turnover")},
        "average_gross_exposure": (ledger.gross_notional / ledger.equity).resample("ME").mean(),
    })
    monthly.to_csv(directory / "monthly.csv")
    return account, metrics


def _execute_second_layer(root: Path, request: SelectionResearchContract,
                          stages: dict[int, dict[str, StageInputs]], pool: dict[str, Any],
                          progress: ProgressLog, *,
                          route_names: tuple[str, ...] = ("R0", "R4")) -> list[dict[str, Any]]:
    development_contract = stages[HORIZONS[0]]["development"].contract
    validation_contract = stages[HORIZONS[0]]["internal_validation"].contract
    global_grid = pd.date_range(
        development_contract.bounds[0] - pd.Timedelta(hours=1),
        validation_contract.bounds[1] - pd.Timedelta(hours=2),
        freq="h", tz="UTC", name="signal_timestamp",
    )
    histories = {"development": {}, "internal_validation": {}}
    symbols = sorted(stages[HORIZONS[0]]["development"].frames)
    for stage in STAGES:
        for horizon in HORIZONS:
            require(stages[horizon][stage].contract.bounds == stages[HORIZONS[0]][stage].contract.bounds,
                    "cross-horizon outer sleeves require matching stage boundaries")
    trials = []
    for method in request.methods:
        for route in route_names:
            for cost_name in ("base", "stress"):
                target_panels, return_columns = {}, {}
                for horizon in HORIZONS:
                    target_parts, return_parts = [], []
                    for stage in STAGES:
                        inputs = stages[horizon][stage]
                        target, realized = _read_inner_history(
                            root, horizon, method, route, stage, cost_name,
                            inputs.contract, symbols,
                        )
                        target_parts.append(target)
                        return_parts.append(realized)
                    target = pd.concat(target_parts).sort_index()
                    require(target.index.is_unique and target.index.equals(global_grid),
                            f"{method}/{route}/h{horizon} child targets do not align across stages")
                    target_panels[horizon] = target
                    returns = pd.concat(return_parts).sort_index()
                    require(returns.index.is_unique, "child return timestamps overlap across stages")
                    return_columns[horizon] = returns.reindex(global_grid)
                realized = pd.DataFrame(return_columns, index=global_grid)
                for outer_policy in OUTER_POLICIES:
                    policy_name = ("gated_risk_budget" if outer_policy["method"] == "risk_budget"
                                   else "fixed")
                    combination_root = (root / "horizon_combination" / method / route
                                        / cost_name / policy_name)
                    with progress.span("selection.combine_horizons", heartbeat=True,
                                       method=method, route=route, cost=cost_name,
                                       policy=policy_name):
                        combination = combine_horizon_targets(
                            target_panels, realized, policy=outer_policy,
                        )
                    combination_root.mkdir(parents=True)
                    combination.targets.to_csv(combination_root / "targets.csv")
                    combination.budgets.to_csv(combination_root / "budgets.csv")
                    combination.audit.to_csv(combination_root / "audit.csv", index=False)
                    combination.fits.to_csv(combination_root / "fits.csv", index=False)
                    fit_timestamps = _utc_timestamps(combination.fits["timestamp"])
                    for stage in STAGES:
                        contract = stages[HORIZONS[0]][stage].contract
                        start, end = contract.bounds
                        account_grid = pd.date_range(
                            start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2),
                            freq="h", tz="UTC", name="signal_timestamp",
                        )
                        stage_fit = combination.fits.loc[
                            (fit_timestamps >= account_grid[0]) & (fit_timestamps <= account_grid[-1])
                        ]
                        stage_used_times = pd.to_datetime(
                            stage_fit["latest_return_timestamp"],
                            format="mixed", utc=True, errors="raise",
                        ).dropna()
                        max_used_return_time = (
                            stage_used_times.max().isoformat() if not stage_used_times.empty else None
                        )
                        targets = combination.targets.reindex(account_grid)
                        require(targets.index.equals(account_grid)
                                and np.isfinite(targets.to_numpy(dtype=float)).all(),
                                "combined targets do not cover a full account signal grid")
                        directory = (root / stage / "outer" / method / route
                                     / policy_name / cost_name)
                        inputs = stages[HORIZONS[0]][stage]
                        multiplier = 1.0 if cost_name == "base" else pool["costs"]["stress_multiplier"]
                        with progress.span("selection.outer_account", heartbeat=True,
                                           stage=stage, method=method, route=route,
                                           cost=cost_name, policy=policy_name):
                            account, metrics = _run_outer_account(
                                directory, targets, inputs.frames, inputs.funding, contract,
                                multiplier, request.second_layer["continuous_target_weight_buffer"],
                            )
                        histories[stage][(method, route, cost_name, policy_name)] = {
                            "metrics": metrics,
                            "path": str(directory),
                        }
                        _write_json(directory / "policy.json", {
                            "route": route,
                            "outer_policy": outer_policy,
                            "account_target_policy": {
                                "gross_exposure": contract.portfolio["gross_exposure"],
                                "max_asset_weight": contract.portfolio["max_asset_weight"],
                                "weight_buffer": request.second_layer["continuous_target_weight_buffer"],
                            },
                            "cost_multiplier": multiplier,
                            "history_max_timestamp_used_for_budget": max_used_return_time,
                        })
    # Summarize every full-grid combination without selecting a winner.
    for method in request.methods:
        for route in route_names:
            for outer_policy in OUTER_POLICIES:
                policy_name = ("gated_risk_budget" if outer_policy["method"] == "risk_budget"
                               else "fixed")
                for stage in STAGES:
                    base = histories[stage][(method, route, "base", policy_name)]
                    stress = histories[stage][(method, route, "stress", policy_name)]
                    failures = _candidate_failures(
                        base["metrics"], stress["metrics"],
                        base["metrics"]["traded_bars"], pool["qualification_gates"],
                    )
                    trial = {
                        "layer": "cross_horizon",
                        "method": method,
                        "route": route,
                        "outer_policy": policy_name,
                        "stage": stage,
                        "metrics": base["metrics"],
                        "path": base["path"],
                        "stress_costs": {
                            "metrics": stress["metrics"],
                            "path": stress["path"],
                        },
                        "qualification_failures": failures,
                        "qualified": not failures,
                    }
                    trials.append(trial)
                    _write_json(root / stage / "outer" / method / route / policy_name / "result.json", trial)
    return trials


def run_selection_research(contract_path: Path, output: Path) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    request = SelectionResearchContract.from_dict(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    source = _resolve(request.source_run, contract_path.parent)
    require(source.is_dir(), f"source run directory does not exist: {source}")
    source_paths = _source_file_paths(source)
    source_manifest = {str(path): _sha256(path) for path in source_paths}
    pool, source_meta = _source_contracts(source)
    for path, digest in source_manifest.items():
        require(_sha256(Path(path)) == digest, "source input changed while the frozen plan was prepared")
    root = Path(output).resolve() / request.run_id
    require(not root.exists(), "selection research run directory already exists; refusing overwrite")
    root.mkdir(parents=True)
    progress = ProgressLog.for_run(root)
    _write_json(root / "contract.json", request.__dict__)
    with progress.span("selection.freeze_plan"):
        _capture_sources(root)
        plan = _build_plan(request, source, pool, source_meta)
        _write_json(root / "plan.json", plan)
        plan_sha = _sha256(root / "plan.json")
        _write_json(root / "plan_manifest.json", {
            "plan_sha256": plan_sha,
            "frozen_before_account_outcomes": True,
        })
    _write_json(root / "source_manifest.json", source_manifest)

    with progress.span("selection.load_source", heartbeat=True):
        stages, model_inputs, used_paths = _load_all_inputs(source, pool)
        unused_manifest_paths = {source / "plan.json"}
        engineering_provenance = source / "engineering_provenance.json"
        if engineering_provenance.is_file():
            unused_manifest_paths.add(engineering_provenance)
        require(set(used_paths) == set(source_paths) - unused_manifest_paths,
                "loaded source inputs differ from the frozen source manifest")
        overlap_audit = _validate_shared_stage_inputs(stages)
        _write_json(root / "input_overlap_audit.json", {
            "factor_overlap_by_horizon": {str(h): model_inputs[h][2] for h in HORIZONS},
            "market_and_funding": overlap_audit,
        })
        dataset_manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
        require(isinstance(dataset_manifest, dict)
                and type(dataset_manifest.get("historical_causality_certified")) is bool,
                "source dataset manifest must declare its historical-causality certification")
        _write_json(root / "dataset_manifest.json", dataset_manifest)

    model_scores: dict[int, dict[str, pd.Series]] = {}
    for horizon in HORIZONS:
        factors, opens, overlap, availability = model_inputs[horizon]
        with progress.span("selection.fit_and_score", heartbeat=True, horizon=horizon):
            generated = _generate_model_scores(
                request, factors, opens, horizon=horizon, stages=stages[horizon],
            )
        require(tuple(generated.scores) == tuple(request.methods),
                f"h{horizon} signal generator did not return the frozen methods")
        for method, score in generated.scores.items():
            require(isinstance(score, pd.Series) and score.index.equals(factors.index)
                    and score.name == "score",
                    f"h{horizon}/{method} score does not preserve the full factor index")
            require(not np.isinf(score.to_numpy(dtype=float)).any(),
                    f"h{horizon}/{method} score contains infinity")
        readiness = availability.copy()
        ready_by_timestamp = generated.shared_model_ready
        readiness["shared_model_ready"] = ready_by_timestamp.reindex(
            readiness.index, fill_value=False,
        ).to_numpy(dtype=bool)
        _write_model_outputs(root, horizon, generated.scores, generated.fits,
                             readiness, overlap, factors)
        model_scores[horizon] = generated.scores

    first_layer = _execute_first_layer(root, request, stages, model_scores, pool, progress)
    second_layer = _execute_second_layer(root, request, stages, pool, progress)
    for path, digest in source_manifest.items():
        require(_sha256(Path(path)) == digest, "source input changed during selection research")
    comparisons = first_layer + second_layer
    pd.DataFrame([{
        "layer": row["layer"], "stage": row["stage"], "horizon_hours": row.get("horizon_hours"),
        "method": row["method"], "route": row["route"],
        "outer_policy": row.get("outer_policy"),
        **row["metrics"],
        "stress_net_return": row["stress_costs"]["metrics"]["net_return"],
        "qualified": row["qualified"],
    } for row in comparisons]).to_csv(root / "comparison.csv", index=False)
    result = {
        "schema_version": 1,
        "run_id": request.run_id,
        "root": str(root),
        "status": "selection_research_complete",
        "plan": plan,
        "plan_sha256": plan_sha,
        "first_layer_trials": len(first_layer),
        "second_layer_trials": len(second_layer),
        "first_layer_account_runs": len(first_layer) * 2,
        "second_layer_account_runs": len(second_layer) * 2,
        "account_runs": (len(first_layer) + len(second_layer)) * 2,
        "trials": comparisons,
        "selection": "none; every frozen route/method/horizon/stage/cost combination is reported",
        "historical_causality_certified": dataset_manifest.get("historical_causality_certified"),
        "model_called": False,
        "paper_started": False,
        "forward_validation_started": False,
    }
    _write_json(root / "result.json", result)
    (root / "report.md").write_text(_render_report(result, pool), encoding="utf-8")
    return result


def _render_report(result: dict[str, Any], pool: dict[str, Any]) -> str:
    gates = pool["qualification_gates"]
    trials = result["trials"]
    development = pool["development_start"]
    validation = pool["validation_start"]
    end = pool["validation_end"]

    def metric_cell(row: dict[str, Any]) -> str:
        metrics = row["metrics"]
        body = (
            f"net {metrics['net_return']:+.1%}, DD {abs(metrics['max_drawdown']):.1%}, "
            f"2x {row['stress_costs']['metrics']['net_return']:+.1%}, "
            f"gross {metrics['average_gross_exposure']:.2f}"
        )
        return body + (", Q=1/1" if row["qualified"] else ", Q=0/1")

    def one(method: str, layer: str, route: str, horizon: int | None = None,
            policy: str | None = None) -> dict[str, Any]:
        return next(row for row in trials
                    if row["method"] == method and row["layer"] == layer
                    and row["route"] == route and row.get("horizon_hours") == horizon
                    and row.get("outer_policy") == policy
                    and row["stage"] == "internal_validation")

    summary_rows = []
    for method in result["plan"]["methods"]:
        fixed = "fixed"
        risk = "gated_risk_budget"
        summary_rows.append(
            "| " + method + " | "
            + "；".join(metric_cell(one(method, "single_horizon", "R0", horizon))
                        for horizon in HORIZONS) + " | "
            + "；".join(metric_cell(one(method, "single_horizon", "R4", horizon))
                        for horizon in HORIZONS) + " | "
            + "；".join(metric_cell(one(method, "cross_horizon", route, policy=fixed))
                        for route in ("R0", "R4")) + " | "
            + "；".join(metric_cell(one(method, "cross_horizon", route, policy=risk))
                        for route in ("R0", "R4")) + " |"
        )

    provenance = (
        "输入标记为工程冒烟数据，结果只验证运行流程，不作为正式历史比较。"
        if result["plan"]["source_provenance"]["engineering_smoke_only"]
        else "输入来自冻结的历史研究快照。"
    )
    return "\n".join([
        f"# {result['run_id']}",
        "",
        provenance + " 本轮按冻结方案执行全部比较，没有根据内部验证段重新选择方法。",
        "",
        f"- 账户运行：{result['account_runs']}（单期限 {result['first_layer_account_runs']}，跨期限 {result['second_layer_account_runs']}）",
        f"- 阶段：development {development} 至 {validation}；internal_validation（C）{validation} 至 {end}",
        f"- 期限：{', '.join(str(value) + 'h' for value in HORIZONS)}",
        f"- 方法：{', '.join(result['plan']['methods'])}",
        "- 单期限路由：R0 固定期限调仓；R4 小时级持仓排名与目标权重双缓冲",
        "- 跨期限路由：固定等预算；均值门控风险预算（协方差风险预算与正收益均值门控一并启用）",
        "- 成本：基准成本和独立的 2 倍成本账户",
        f"- 账户门槛：净收益不低于 {gates['min_net_return']:.4g}，最大回撤不超过 {gates['max_drawdown']:.4g}，压力成本净收益不低于 {gates['min_stress_return']:.4g}",
        f"- 历史因果认证：{result['historical_causality_certified']}",
        "",
        "C 段结果（收益、回撤、2 倍成本收益；跨期限另列平均毛敞口；达门槛数值表示单项账户通过原门槛）：",
        "",
        "| 方法 | 单期限 R0（1h / 4h / 24h） | 单期限 R4（1h / 4h / 24h） | 跨期限 fixed（R0 / R4） | 跨期限均值门控风险预算（R0 / R4） |",
        "| --- | --- | --- | --- | --- |",
        *summary_rows,
        "",
        "各组合均按完整账户时间网格计算。模型单独弃权保留为该方法的现金路径。跨期限风险预算只读取信号时间之前、且实际已有的已实现子账户收益。",
        "",
        "研究文件：[完整比较表](comparison.csv)、[机器可读结果](result.json)、[模型与信号审计](models/)、[冻结计划](plan.json)、[输入哈希](source_manifest.json)。",
        "",
    ])
