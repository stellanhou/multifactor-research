"""Prepared, event-driven forward research for qualified multi-factor portfolios.

This module has no network, database, broker, or exchange-order integration.
It freezes a qualified candidate and fresh seed data, then exposes explicit
activation and event-ingestion methods. Offline event replay exercises the same
state transitions while leaving ``forward_started`` false.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from .multifactor_account import (
    HOUR,
    _apply_fill,
    _check_margin,
    _fill_terms,
    _funding_cashflow,
    _target_quantity,
    generate_rank_targets,
)
from .multifactor_factors import process_factors


FORWARD_SCHEMA_VERSION = 1
DEFAULT_MAX_DRAWDOWN = 0.15
DEFAULT_OBSERVATION_HOURS = 720
DEFAULT_MIN_REBALANCES = 20


class ForwardSessionError(RuntimeError):
    """A frozen forward-session contract or its event history is invalid."""


@dataclass(frozen=True)
class FreshSeedData:
    """Causally timestamped seed panel and account inputs from a collector."""

    panel: FactorInputPanel
    frames: dict[str, pd.DataFrame]
    funding: pd.DataFrame | None
    feature_available_at: pd.DataFrame
    received_at: pd.Series
    diagnostics: dict[str, Any]


def _stamp(value: Any, label: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None or result.utcoffset() != pd.Timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC")
    return result.tz_convert("UTC")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_indexed(frame: pd.DataFrame | pd.Series, path: Path) -> None:
    value = frame.to_frame() if isinstance(frame, pd.Series) else frame
    value.rename_axis(index=["timestamp", "symbol"]).reset_index().to_csv(path, index=False)


def _read_indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame.index = pd.MultiIndex.from_arrays(
        [frame.pop("timestamp"), frame.pop("symbol")], names=["timestamp", "symbol"]
    )
    return frame


def _write_seed(seed: FreshSeedData, root: Path) -> dict[str, Any]:
    seed_root = root / "seed"
    market_root = seed_root / "market"
    market_root.mkdir(parents=True)
    _write_indexed(seed.panel.values, seed_root / "factor_inputs.csv")
    _write_indexed(seed.panel.universe.rename("eligible"), seed_root / "universe.csv")
    _write_indexed(seed.feature_available_at, seed_root / "feature_available_at.csv")
    _write_indexed(seed.received_at.rename("received_at"), seed_root / "received_at.csv")
    for symbol, frame in sorted(seed.frames.items()):
        frame.rename_axis("timestamp").reset_index().to_csv(market_root / f"{symbol}.csv", index=False)
    funding = (
        seed.funding.copy()
        if seed.funding is not None
        else pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    )
    if "timestamp" not in funding.columns:
        if not funding.empty:
            raise ValueError("nonempty seed funding requires a timestamp column")
        funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    funding.to_csv(seed_root / "funding.csv", index=False)
    _atomic_json(seed_root / "diagnostics.json", seed.diagnostics)
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(seed_root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "files": hashes,
        "symbols": sorted(seed.frames),
        "first_timestamp": seed.panel.values.index.get_level_values("timestamp").min().isoformat(),
        "last_timestamp": seed.panel.values.index.get_level_values("timestamp").max().isoformat(),
        "source_diagnostics": seed.diagnostics,
    }
    _atomic_json(seed_root / "manifest.json", manifest)
    return manifest


def _verify_seed_manifest(root: Path, contract: dict[str, Any]) -> None:
    manifest_path = root / str(contract["seed"]["manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != contract["seed"]["manifest_sha256"]:
        raise ForwardSessionError("frozen seed manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ForwardSessionError("frozen seed manifest path escapes the session") from exc
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ForwardSessionError(f"frozen seed input changed: {relative}")


def _read_seed(root: Path) -> FreshSeedData:
    seed_root = root / "seed"
    values = _read_indexed(seed_root / "factor_inputs.csv")
    universe_frame = _read_indexed(seed_root / "universe.csv")
    universe = universe_frame["eligible"].astype(bool).rename("eligible")
    panel = FactorInputPanel(values=values, universe=universe, diagnostics=json.loads(
        (seed_root / "diagnostics.json").read_text(encoding="utf-8")
    ))
    available = _read_indexed(seed_root / "feature_available_at.csv")
    for column in available:
        available[column] = pd.to_datetime(available[column], utc=True, format="ISO8601")
    received_frame = _read_indexed(seed_root / "received_at.csv")
    received = pd.to_datetime(received_frame["received_at"], utc=True, format="ISO8601")
    received.name = "received_at"
    market = {}
    for path in sorted((seed_root / "market").glob("*.csv")):
        frame = pd.read_csv(path, float_precision="round_trip")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
        market[path.stem] = frame.set_index("timestamp")
    funding = pd.read_csv(seed_root / "funding.csv", float_precision="round_trip")
    if not funding.empty:
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True, format="ISO8601")
    return FreshSeedData(
        panel=panel,
        frames=market,
        funding=funding,
        feature_available_at=available,
        received_at=received,
        diagnostics=panel.diagnostics,
    )


def _validate_seed(seed: FreshSeedData, required_fields: set[str], start: pd.Timestamp,
                   mode: str) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    if not isinstance(seed.panel, FactorInputPanel):
        raise ValueError("fresh seed requires a FactorInputPanel")
    if not seed.frames:
        raise ValueError("fresh seed requires market frames")
    symbols = sorted(seed.frames)
    index = seed.panel.values.index
    if not isinstance(index, pd.MultiIndex) or index.names != ["timestamp", "symbol"]:
        raise ValueError("fresh seed panel must use a (timestamp, symbol) index")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("fresh seed panel index must be unique and sorted")
    if list(seed.panel.values.columns) != list(INPUT_COLUMNS):
        raise ValueError("fresh seed must contain the complete FactorInputPanel field set")
    if not seed.panel.values.index.equals(seed.panel.universe.index):
        raise ValueError("fresh seed panel and explicit universe indexes differ")
    if set(index.get_level_values("symbol")) != set(symbols):
        raise ValueError("fresh seed panel symbols differ from market frames")
    if not required_fields.issubset(seed.feature_available_at.columns):
        raise ValueError(f"fresh seed availability is missing fields: {sorted(required_fields - set(seed.feature_available_at.columns))}")
    if not seed.feature_available_at.index.equals(index) or not seed.received_at.index.equals(index):
        raise ValueError("fresh seed available_at/received_at indexes must match the factor panel")
    if not seed.panel.universe.astype(bool).all():
        raise ValueError("forward v1 freezes a fixed eligible universe; seed membership must be true")

    timestamps = index.get_level_values("timestamp").unique()
    if len(timestamps) < 2:
        raise ValueError("fresh seed needs at least two hourly bars")
    expected = pd.date_range(timestamps[0], timestamps[-1], freq=HOUR, tz="UTC")
    if not timestamps.equals(expected):
        raise ValueError("fresh seed timestamps must form a complete hourly grid")
    for (timestamp, symbol), feature_row in seed.panel.values.iterrows():
        row_received = _stamp(seed.received_at.loc[(timestamp, symbol)], "seed received_at")
        if row_received > start:
            raise ValueError("fresh seed rows must be received by the frozen start")
        available_times = []
        for field in required_fields:
            value = seed.feature_available_at.loc[(timestamp, symbol), field]
            feature_value = feature_row[field]
            if pd.isna(feature_value):
                continue
            if not np.isfinite(float(feature_value)):
                raise ValueError(f"fresh seed feature {field} must be finite or missing")
            if pd.isna(value):
                raise ValueError(f"fresh seed lacks available_at for {field} at {(timestamp, symbol)}")
            available = _stamp(value, f"seed {field} available_at")
            if available > start:
                raise ValueError("fresh seed fields must be available by the frozen start")
            if mode == "forward" and available < timestamp + HOUR:
                raise ValueError(f"forward seed feature {field} predates its completed bar")
            available_times.append(available)
        if available_times and max(available_times) > row_received:
            raise ValueError(f"seed received_at precedes a feature's available_at at {(timestamp, symbol)}")
    last_timestamp = timestamps[-1]
    expected_seed_end = start - 2 * HOUR
    if last_timestamp != expected_seed_end:
        raise ValueError("fresh seed must end at the frozen signal anchor for its session mode")

    for symbol in symbols:
        frame = seed.frames[symbol]
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{symbol} market timestamps must be UTC DatetimeIndex values")
        frame_index = frame.index.tz_convert("UTC")
        if not frame_index.equals(timestamps):
            raise ValueError(f"{symbol} market grid differs from the fresh seed panel")
        missing = {"open", "close", "mark_close"} - set(frame.columns)
        if missing:
            raise ValueError(f"{symbol} market frame missing fields: {sorted(missing)}")
        last = frame.loc[last_timestamp, ["open", "close", "mark_close"]].to_numpy(dtype=float)
        if not np.isfinite(last).all() or (last <= 0).any():
            raise ValueError(f"{symbol} latest seed prices must be finite and positive")
        panel_row = seed.panel.values.loc[(last_timestamp, symbol)]
        if not np.isclose(float(panel_row["perp_close"]), float(frame.loc[last_timestamp, "close"])):
            raise ValueError(f"{symbol} seed perp_close differs between feature and account inputs")
        if not np.isclose(float(panel_row["mark_close"]), float(frame.loc[last_timestamp, "mark_close"])):
            raise ValueError(f"{symbol} seed mark_close differs between feature and account inputs")

    available_last: list[pd.Timestamp] = []
    received_last: list[pd.Timestamp] = []
    for symbol in symbols:
        row = (last_timestamp, symbol)
        for field in sorted(required_fields):
            value = seed.feature_available_at.loc[row, field]
            feature_value = seed.panel.values.loc[row, field]
            if pd.notna(feature_value):
                if pd.isna(value):
                    raise ValueError(f"fresh seed lacks availability for {field} at {row}")
                available_last.append(_stamp(value, f"seed {field} available_at"))
        received_last.append(_stamp(seed.received_at.loc[row], "seed received_at"))
    latest_seed_cutoff = start
    if max(available_last + received_last) > latest_seed_cutoff:
        raise ValueError("fresh seed fields must be available and received by their closed-bar cutoff")
    return symbols, timestamps[0], last_timestamp


def _qualification_failures(metrics: Mapping[str, Any], stress_metrics: Mapping[str, Any],
                             traded_bars: Any, gates: Mapping[str, Any]) -> list[str]:
    failures = []
    if float(metrics["net_return"]) < float(gates["min_net_return"]):
        failures.append("min_net_return")
    if abs(float(metrics["max_drawdown"])) > float(gates["max_drawdown"]):
        failures.append("max_drawdown")
    if type(traded_bars) is not int or traded_bars < int(gates["min_traded_bars"]):
        failures.append("min_traded_bars")
    if float(stress_metrics["net_return"]) < float(gates["min_stress_return"]):
        failures.append("min_stress_return")
    return failures


def _qualified_historical_bundle(snapshot, historical_result_path: Path) -> dict[str, Any]:
    from .multifactor_historical import HistoricalContract
    from .multifactor_contracts import ResearchContract

    result_path = Path(historical_result_path).resolve()
    if not result_path.is_file() or result_path.name != "result.json":
        raise ValueError("historical_result_path must identify a completed result.json")
    history_root = result_path.parent.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("engine") != "multifactor_historical_v1":
        raise ValueError("historical result engine is not multifactor_historical_v1")
    if result.get("status") != "qualified_candidate_frozen":
        raise ValueError("historical result has no frozen qualified candidate")
    if Path(result.get("root", "")).resolve() != history_root:
        raise ValueError("historical result root does not match historical_result_path")
    readiness = result.get("forward_readiness", {})
    if (readiness.get("forward_validation_started") is not False
            or readiness.get("paper_started") is not False
            or readiness.get("published") is not False):
        raise ValueError("historical forward-readiness flags are not clean")
    if result.get("c_results_returned_to_agent") is not False:
        raise ValueError("historical C results were returned to the Agent")
    if result.get("c_data_accessed_after_candidate_freeze") is not True:
        raise ValueError("historical C validation did not follow the A+B candidate freeze")

    required_artifacts = {
        "contract", "source_gate", "dataset_manifest", "universe", "cards_manifest",
        "development_contract", "development_baseline_result", "agent_session_result", "candidate_trials",
        "candidate_freeze", "internal_validation_result",
    }
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        raise ValueError(f"historical result is missing artifact references: {sorted(required_artifacts - set(artifacts or {}))}")

    resolved_artifacts: dict[str, tuple[Path, str, str]] = {}
    for name in sorted(required_artifacts):
        record = artifacts[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"historical artifact reference is malformed: {name}")
        relative = Path(record["path"])
        if relative.is_absolute():
            raise ValueError(f"historical artifact path must be root-relative: {name}")
        artifact_path = (history_root / relative).resolve()
        try:
            artifact_path.relative_to(history_root)
        except ValueError as exc:
            raise ValueError(f"historical artifact escapes its run root: {name}") from exc
        if not artifact_path.is_file():
            raise ValueError(f"historical artifact is missing: {name}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"historical artifact hash mismatch: {name}")
        resolved_artifacts[name] = (artifact_path, str(relative), digest)

    historical_contract_data = json.loads(resolved_artifacts["contract"][0].read_text(encoding="utf-8"))
    historical_contract = HistoricalContract.from_dict(historical_contract_data)
    if historical_contract.run_id != result.get("run_id"):
        raise ValueError("historical run ID differs from its frozen contract")
    gates = dict(historical_contract.qualification_gates)

    source_gate = json.loads(resolved_artifacts["source_gate"][0].read_text(encoding="utf-8"))
    provenance = result.get("dataset_provenance", {})
    dataset_manifest = json.loads(resolved_artifacts["dataset_manifest"][0].read_text(encoding="utf-8"))
    if (source_gate.get("passed") is not True
            or source_gate.get("primary_source_fallback") is not False
            or source_gate.get("C_read_before_AB_freeze") is not False
            or source_gate.get("dataset_id") != provenance.get("dataset_id")
            or source_gate.get("manifest_sha256") != provenance.get("manifest_sha256")
            or source_gate.get("database_sha256") != provenance.get("database_sha256")):
        raise ValueError("historical source gate does not match its frozen provenance")
    database_signature = dataset_manifest.get("database_signature", {})
    manifest_symbols = dataset_manifest.get("symbols")
    if (database_signature.get("sha256") != provenance.get("database_sha256")
            or not isinstance(manifest_symbols, list)
            or len(manifest_symbols) != 10
            or len(manifest_symbols) != len(set(manifest_symbols))
            or not all(isinstance(symbol, str) and symbol.endswith("USDT") for symbol in manifest_symbols)
            or provenance.get("symbols") != sorted(manifest_symbols)):
        raise ValueError("historical dataset manifest symbols or database signature differ from its provenance")

    development_contract_data = json.loads(
        resolved_artifacts["development_contract"][0].read_text(encoding="utf-8")
    )
    development_contract = ResearchContract.from_dict(development_contract_data)
    baseline_result_path = resolved_artifacts["development_baseline_result"][0]
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    baseline_root = baseline_result_path.parent.resolve()
    if (Path(baseline_result.get("root", "")).resolve() != baseline_root
            or baseline_result.get("engine") != "deterministic_multifactor_v1"
            or baseline_result.get("status") != "engineering_complete"
            or baseline_result.get("model_called") is not False
            or baseline_result.get("agent_used") is not False
            or baseline_result.get("purpose") != "research"
            or baseline_result.get("stage") != "development"
            or baseline_result.get("forward_validation_started") is not False
            or baseline_result.get("paper_started") is not False
            or baseline_result.get("published") is not False
            or result.get("full_pool_baseline") != baseline_result
            or result.get("stage_run_ids", {}).get("development_baseline") != baseline_result.get("run_id")
            or getattr(snapshot, "result", None) != baseline_result):
        raise ValueError("supplied baseline snapshot result differs from the frozen historical baseline")
    if baseline_result.get("contract") != "contract.json":
        raise ValueError("historical baseline result has an unexpected contract reference")
    baseline_contract_path = (baseline_root / baseline_result["contract"]).resolve()
    try:
        baseline_contract_path.relative_to(baseline_root)
    except ValueError as exc:
        raise ValueError("historical baseline contract escapes its baseline root") from exc
    if not baseline_contract_path.is_file():
        raise ValueError("historical baseline contract is missing")
    baseline_contract_data = json.loads(baseline_contract_path.read_text(encoding="utf-8"))
    try:
        supplied_contract_data = snapshot.contract.as_dict()
    except AttributeError as exc:
        raise ValueError("supplied baseline snapshot has no frozen contract") from exc
    if (baseline_contract_data != development_contract_data
            or supplied_contract_data != development_contract_data):
        raise ValueError("development contract differs from the supplied baseline snapshot")
    if (development_contract.stage != "development"
            or development_contract.start != historical_contract.development_start
            or development_contract.end != historical_contract.validation_start
            or development_contract.warmup_hours != historical_contract.warmup_hours
            or development_contract.horizon_hours != historical_contract.horizon_hours
            or development_contract.costs != historical_contract.costs
            or development_contract.portfolio != historical_contract.portfolio
            or development_contract.prior_data_use != historical_contract.prior_data_use
            or development_contract.data_processing != historical_contract.data_processing):
        raise ValueError("development contract differs from the frozen historical configuration")
    manifest_source = Path(provenance.get("manifest_path", "")).resolve()
    stage_manifest_path = (history_root / "contracts" / development_contract.dataset_manifest).resolve()
    if manifest_source != stage_manifest_path:
        raise ValueError("development contract dataset path differs from historical source provenance")
    if getattr(getattr(snapshot, "inputs", None), "dataset_manifest", None) != dataset_manifest:
        raise ValueError("supplied baseline snapshot dataset differs from the frozen historical manifest")
    snapshot_frames = getattr(getattr(snapshot, "inputs", None), "frames", None)
    if not isinstance(snapshot_frames, dict) or sorted(snapshot_frames) != sorted(manifest_symbols):
        raise ValueError("supplied baseline snapshot market pool differs from the historical dataset")

    card_manifest = json.loads(resolved_artifacts["cards_manifest"][0].read_text(encoding="utf-8"))
    cards = list(snapshot.cards)
    pool_ids = [card["id"] for card in cards]
    if [item.get("id") for item in card_manifest] != pool_ids:
        raise ValueError("historical card manifest differs from the supplied frozen card pool")
    if development_contract.cards != [item.get("source_path") for item in card_manifest]:
        raise ValueError("development contract cards differ from the historical frozen card manifest")
    snapshot_by_id = {card["id"]: card for card in cards}
    copied_card_paths: list[tuple[Path, str, str]] = []
    for card_item in card_manifest:
        card_path = (history_root / card_item["snapshot_path"]).resolve()
        try:
            card_path.relative_to(history_root)
        except ValueError as exc:
            raise ValueError("historical card snapshot escapes its run root") from exc
        if not card_path.is_file():
            raise ValueError(f"historical card snapshot is missing: {card_item['id']}")
        digest = hashlib.sha256(card_path.read_bytes()).hexdigest()
        if digest != card_item.get("source_sha256"):
            raise ValueError(f"historical card snapshot hash mismatch: {card_item['id']}")
        raw_card = json.loads(card_path.read_text(encoding="utf-8"))
        if raw_card != snapshot_by_id[card_item["id"]].get("card_snapshot"):
            raise ValueError(f"historical card contents differ from the supplied snapshot: {card_item['id']}")
        copied_card_paths.append((card_path, str(card_path.relative_to(history_root)), digest))

    freeze_path = resolved_artifacts["candidate_freeze"][0]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validation = json.loads(resolved_artifacts["internal_validation_result"][0].read_text(encoding="utf-8"))
    freeze_digest = resolved_artifacts["candidate_freeze"][2]
    if (result.get("candidate_freeze_sha256") != freeze_digest
            or validation.get("candidate_freeze_sha256") != freeze_digest):
        raise ValueError("C validation is not hash-bound to the pre-C candidate freeze")
    if result.get("candidate_freeze") != freeze or result.get("internal_validation") != validation:
        raise ValueError("historical result does not match its candidate freeze or C validation artifact")
    selected = freeze.get("selected_card_ids")
    if freeze.get("baseline_run_id") != development_contract.run_id:
        raise ValueError("historical candidate freeze references a different development baseline")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("historical frozen candidate card IDs are invalid")
    if not set(selected).issubset(pool_ids):
        raise ValueError("historical frozen candidate references cards outside the supplied pool")
    candidate_trial_id = freeze.get("selected_trial_id")
    trials = json.loads(resolved_artifacts["candidate_trials"][0].read_text(encoding="utf-8"))
    if result.get("trials") != trials or result.get("candidates") != trials:
        raise ValueError("historical trial list differs from its qualification artifact")
    selected_trials = [item for item in trials if item.get("trial_id") == candidate_trial_id]
    if len(selected_trials) != 1:
        raise ValueError("historical freeze does not resolve to exactly one A+B trial")
    selected_trial = selected_trials[0]
    if selected_trial.get("selected_card_ids") != selected:
        raise ValueError("selected A+B trial card IDs differ from candidate freeze")
    if (freeze.get("selected_ab_qualified") is not True
            or selected_trial.get("qualified_ab") is not True
            or selected_trial.get("qualification_failures") != []
            or freeze.get("selected_before_c_access") is not True
            or freeze.get("c_data_accessed") is not False
            or freeze.get("selection_rule") != historical_contract.candidate_selection
            or freeze.get("qualification_gates") != gates):
        raise ValueError("A+B candidate freeze did not pass the predeclared qualification gates")
    ab_failures = _qualification_failures(
        selected_trial["metrics"], selected_trial["stress_costs"]["metrics"],
        selected_trial["traded_bars"], gates,
    )
    if ab_failures or freeze.get("selected_ab_metrics") != selected_trial["metrics"]:
        raise ValueError("selected A+B candidate fails or disagrees with its metric gates")
    if freeze.get("selected_ab_stress_metrics") != selected_trial["stress_costs"]["metrics"]:
        raise ValueError("selected A+B stress metrics differ from the frozen trial")

    selected_result = result.get("selected_candidate", {})
    if result.get("agent_session", {}).get("fixed_results_exposed_to_agent") is not False:
        raise ValueError("historical fixed-arm results were exposed to the Agent")
    c_candidate = validation.get("candidate", {})
    if (selected_result.get("card_ids") != selected
            or selected_result.get("qualified_ab") is not True
            or selected_result.get("qualified_c") is not True
            or selected_result.get("qualified") is not True
            or result.get("selected_candidates") != [selected]):
        raise ValueError("top-level selected candidate does not pass both frozen validation gates")
    if (validation.get("selected_candidate_card_ids") != selected
            or c_candidate.get("included_card_ids") != selected
            or validation.get("qualified_c") is not True
            or validation.get("qualification_failures") != []
            or validation.get("qualification_gates") != gates
            or validation.get("same_full_pool_common_mask") is not True
            or validation.get("agent_called_after_c_access") is not False):
        raise ValueError("C validation artifact does not qualify the exact frozen candidate")
    c_failures = _qualification_failures(
        c_candidate["metrics"], validation["candidate_stress_costs"]["metrics"],
        validation["traded_bars"], gates,
    )
    if c_failures:
        raise ValueError("C metrics fail the historical candidate gates")

    return {
        "root": history_root,
        "result_path": result_path,
        "result": result,
        "contract": historical_contract.as_dict(),
        "candidate_id": str(candidate_trial_id),
        "selected_card_ids": list(selected),
        "qualification_gates": gates,
        "universe_symbols": sorted(manifest_symbols),
        "ab_metrics": selected_trial["metrics"],
        "ab_stress_metrics": selected_trial["stress_costs"]["metrics"],
        "c_metrics": c_candidate["metrics"],
        "c_stress_metrics": validation["candidate_stress_costs"]["metrics"],
        "qualification_refs": ["candidate-freeze.json", "candidate-trials.json", "validation/result.json"],
        "verified_artifacts": list(resolved_artifacts.values()) + copied_card_paths,
    }


def freeze_forward_candidate(
    snapshot,
    historical_result_path: Path,
    seed: FreshSeedData,
    root: Path,
    *,
    start: Any,
    mode: str = "forward",
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
    observation_hours: int = DEFAULT_OBSERVATION_HOURS,
    minimum_rebalances: int = DEFAULT_MIN_REBALANCES,
) -> "ForwardSession":
    """Freeze one qualified candidate and a fresh, timestamped seed package.

    Freezing only creates a prepared local session. It does not activate it,
    fetch data, contact a service, or submit orders.
    """
    start_ts = _stamp(start, "forward start")
    if start_ts != start_ts.floor("h"):
        raise ValueError("forward start must be a UTC hour boundary")
    if not np.isfinite(max_drawdown) or not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be in (0, 1)")
    if type(observation_hours) is not int or observation_hours <= 0:
        raise ValueError("observation_hours must be a positive integer")
    if type(minimum_rebalances) is not int or minimum_rebalances <= 0:
        raise ValueError("minimum_rebalances must be a positive integer")
    if mode not in {"forward", "offline_event_replay"}:
        raise ValueError("mode must be forward or offline_event_replay")

    qualification = _qualified_historical_bundle(snapshot, historical_result_path)
    cards = list(snapshot.cards)
    selected_ids = qualification["selected_card_ids"]
    candidate_id = qualification["candidate_id"]
    qualification_refs = qualification["qualification_refs"]
    pool_ids = [card["id"] for card in cards]
    if not set(selected_ids).issubset(pool_ids):
        raise ValueError("qualified historical candidate is outside the supplied snapshot card pool")
    required_fields = {"perp_close", "mark_close"}
    for card in cards:
        required_fields.update(card["fields"])
    symbols, seed_first, seed_last = _validate_seed(seed, required_fields, start_ts, mode)
    if symbols != qualification["universe_symbols"]:
        raise ValueError("fresh seed symbol pool differs from the qualified historical universe")
    if minimum_rebalances > observation_hours // snapshot.contract.portfolio["rebalance_hours"]:
        raise ValueError("minimum_rebalances exceeds the predeclared observation opportunity")
    if {
        "long_count", "short_count", "gross_exposure", "max_asset_weight",
        "rebalance_hours", "margin_fraction",
    } - set(snapshot.contract.portfolio):
        raise ValueError("baseline portfolio contract is incomplete")

    factors = process_factors(seed.panel, cards)
    seed_mask = factors.common_mask.xs(seed_last, level="timestamp")
    if not bool(seed_mask.any()):
        raise ValueError("latest fresh seed bar has no full-pool common-mask assets")
    standardized = factors.standardized.xs(seed_last, level="timestamp")
    score = standardized[selected_ids].mean(axis=1).where(seed_mask)
    initial_equity = float(snapshot.contract.costs["initial_capital"])
    seed_received = max(
        _stamp(seed.received_at.loc[(seed_last, symbol)], "seed received_at")
        for symbol in symbols
    )
    last_available = []
    for symbol in symbols:
        for field in required_fields:
            value = seed.feature_available_at.loc[(seed_last, symbol), field]
            if pd.notna(value) and pd.notna(seed.panel.values.loc[(seed_last, symbol), field]):
                last_available.append(_stamp(value, f"seed {field} available_at"))
    seed_available = max(last_available)
    if seed_received > start_ts or seed_available > start_ts:
        raise ValueError("the latest seed signal must be available by forward start")

    session_root = Path(root)
    session_root.mkdir(parents=True, exist_ok=False)
    historical_output = session_root / "historical_qualification"
    historical_output.mkdir()
    copied_history_artifacts = {}
    for source_path, relative_path, digest in qualification["verified_artifacts"]:
        destination = historical_output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if copied_digest != digest:
            raise ForwardSessionError(f"copied historical qualification artifact changed: {relative_path}")
        copied_history_artifacts[relative_path] = {"sha256": copied_digest}
    historical_result_copy = historical_output / "result.json"
    shutil.copy2(qualification["result_path"], historical_result_copy)
    historical_result_digest = hashlib.sha256(historical_result_copy.read_bytes()).hexdigest()
    _atomic_json(historical_output / "manifest.json", {
        "historical_run_id": qualification["result"]["run_id"],
        "candidate_id": candidate_id,
        "selected_card_ids": selected_ids,
        "source_result_path": str(qualification["result_path"]),
        "source_result_sha256": historical_result_digest,
        "copied_artifacts": copied_history_artifacts,
    })
    seed_manifest = _write_seed(seed, session_root)
    factor_root = session_root / "seed" / "factor_panels"
    factor_root.mkdir()
    factors.standardized.to_csv(factor_root / "standardized.csv")
    factors.common_mask.rename("common_mask").to_csv(factor_root / "common_mask.csv")
    for file_path in (factor_root / "standardized.csv", factor_root / "common_mask.csv"):
        seed_manifest["files"][str(file_path.relative_to(session_root))] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    _atomic_json(session_root / "seed" / "manifest.json", seed_manifest)

    factor_results = {item["id"]: item for item in snapshot.result["factors"]}
    card_definitions = []
    for card in cards:
        card_definitions.append({
            "id": card["id"],
            "code": factor_results[card["id"]]["code"],
            "title": card["title"],
            "expression": card["expression"],
            "direction": card["direction"],
            "fields": list(card["fields"]),
            "lookback_hours": int(card["lookback_hours"]),
            "horizon_hours": int(card["horizon_hours"]),
            "snapshot_path": factor_results[card["id"]]["snapshot_path"],
            "card_snapshot": card["card_snapshot"],
        })
    order_intents: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}

    baseline_contract = snapshot.contract.as_dict()
    contract = {
        "schema_version": FORWARD_SCHEMA_VERSION,
        "session_id": session_root.name,
        "mode": mode,
        "status": "prepared",
        "candidate_id": candidate_id,
        "candidate_qualified": True,
        "qualification_refs": qualification_refs,
        "historical_qualification": {
            "result_path": "historical_qualification/result.json",
            "result_sha256": historical_result_digest,
            "manifest_path": "historical_qualification/manifest.json",
            "historical_run_id": qualification["result"]["run_id"],
            "qualification_gates": qualification["qualification_gates"],
            "ab_metrics": qualification["ab_metrics"],
            "ab_stress_metrics": qualification["ab_stress_metrics"],
            "c_metrics": qualification["c_metrics"],
            "c_stress_metrics": qualification["c_stress_metrics"],
        },
        "baseline_run_id": snapshot.contract.run_id,
        "baseline_contract": baseline_contract,
        "full_pool_card_ids": [card["id"] for card in cards],
        "selected_card_ids": selected_ids,
        "cards": card_definitions,
        "required_feature_fields": sorted(required_fields),
        "factor_standardization": {
            "method": "direction-adjusted cross-sectional z-score",
            "std_ddof": 0,
            "common_mask": "all frozen pool factors must be valid for the row",
            "universe_symbols": symbols,
        },
        "portfolio": dict(snapshot.contract.portfolio),
        "costs": dict(snapshot.contract.costs),
        "seed": {
            "first_timestamp": seed_first.isoformat(),
            "last_signal_timestamp": seed_last.isoformat(),
            "last_available_at": seed_available.isoformat(),
            "last_received_at": seed_received.isoformat(),
            "manifest": "seed/manifest.json",
            "manifest_sha256": hashlib.sha256((session_root / "seed/manifest.json").read_bytes()).hexdigest(),
        },
        "risk_stop": {
            "max_drawdown": float(max_drawdown),
            "action": "halt_new_orders_and_preserve_positions; no forced liquidation",
        },
        "observation": {
            "start": start_ts.isoformat(),
            "end_exclusive": (start_ts + pd.Timedelta(hours=observation_hours)).isoformat(),
            "duration_hours": observation_hours,
            "minimum_rebalances": minimum_rebalances,
            "minimum_rebalances_basis": "complete scheduled signal decisions; independent of fill count",
        },
        "execution_assumptions": {
            "product": "linear USD-M perpetual",
            "fill_model": "full fills at first observed quote after signal receipt with fixed-bps slippage",
            "reference": "first_observed_quote_after_signal_receipt",
            "historical_baseline_reference": "scheduled next-hour open",
            "quantity": "frozen at signal close using then-known equity and close",
            "contract_units": "continuous; no lot-size rounding",
            "funding": "event mark, signed quantity; late event is preserved unapplied and halts",
            "liquidation": "not simulated",
        },
        "input_provenance": dict(seed.diagnostics),
        "paper_started": False,
        "forward_started": False,
    }
    _atomic_json(session_root / "contract.json", contract)
    (session_root / "events.jsonl").write_text("", encoding="utf-8")
    account = {
        "cash": initial_equity,
        "quantities": {symbol: 0.0 for symbol in symbols},
        "entries": {symbol: None for symbol in symbols},
        "latest_marks": {
            symbol: float(seed.frames[symbol].loc[seed_last, "mark_close"]) for symbol in symbols
        },
        "equity": initial_equity,
        "peak_equity": initial_equity,
        "drawdown": 0.0,
    }
    contract_hash = hashlib.sha256((session_root / "contract.json").read_bytes()).hexdigest()
    state = {
        "schema_version": FORWARD_SCHEMA_VERSION,
        "session_id": session_root.name,
        "status": "prepared",
        "mode": mode,
        "forward_started": False,
        "paper_started": False,
        "live_bars": 0,
        "replay_bars": 0,
        "event_sequence": 0,
        "last_event_hash": "genesis",
        "frozen_contract_sha256": contract_hash,
        "last_received_at": seed_received.isoformat(),
        "account": account,
        "seed_signal_timestamp": seed_last.isoformat(),
        "last_signal_timestamp": seed_last.isoformat(),
        "last_ledger_timestamp": None,
        "last_market_mark_timestamp": seed_last.isoformat(),
        "score_history": [{
            "timestamp": seed_last.isoformat(),
            "scores": {symbol: (float(score[symbol]) if pd.notna(score[symbol]) else None) for symbol in symbols},
        }],
        "feature_rows": [],
        "close_buffer": {},
        "pending_orders": ({start_ts.isoformat(): pending} if pending else {}),
        "orders": order_intents,
        "fills": [],
        "funding_events": [],
        "missed_executions": [],
        "positions": [],
        "ledger": [],
        "bar_totals": {},
        "seen_events": {},
        "rebalances": 0,
    }
    _save_state(session_root, state)
    return ForwardSession(session_root, contract, seed, cards, required_fields, state)


def _state_hash(state: dict[str, Any]) -> str:
    return _hash({key: value for key, value in state.items() if key != "state_hash"})


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["state_hash"] = _state_hash(state)
    _atomic_json(root / "state.json", state)


def _journal_hash(row: dict[str, Any]) -> str:
    return _hash({key: value for key, value in row.items() if key != "event_hash"})


class ForwardSession:
    """Append-only local session with explicit live activation and offline replay."""

    def __init__(self, root: Path, contract: dict[str, Any], seed: FreshSeedData,
                 cards: list[dict[str, Any]], required_fields: set[str], state: dict[str, Any]):
        self.root = Path(root)
        self.contract = contract
        self.seed = seed
        self.cards = cards
        self.required_fields = set(required_fields)
        self.symbols = list(contract["factor_standardization"]["universe_symbols"])
        self._state = state
        self._panel = seed.panel
        self._seen: dict[str, str] = dict(state.get("seen_events", {}))
        self._last_received = _stamp(state["last_received_at"], "last_received_at")
        self._factor_panels = process_factors(self._panel, self.cards)
        self._restore_feature_rows()

    def _restore_feature_rows(self) -> None:
        rows = self._state.get("feature_rows", [])
        if not rows:
            return
        additions = []
        universe_rows = []
        for row in rows:
            timestamp = _stamp(row["timestamp"], "saved feature timestamp")
            for symbol in self.symbols:
                values = {field: np.nan for field in INPUT_COLUMNS}
                values.update(row["features"][symbol])
                additions.append(pd.Series(values, name=(timestamp, symbol)))
                universe_rows.append(((timestamp, symbol), True))
        frame = pd.DataFrame(additions)
        frame.index = pd.MultiIndex.from_tuples(frame.index, names=["timestamp", "symbol"])
        self._panel = FactorInputPanel(
            values=pd.concat([self._panel.values, frame]).sort_index(),
            universe=pd.concat([
                self._panel.universe,
                pd.Series([eligible for _, eligible in universe_rows],
                          index=pd.MultiIndex.from_tuples([item for item, _ in universe_rows], names=["timestamp", "symbol"]),
                          name="eligible", dtype=bool),
            ]).sort_index(),
            diagnostics=self._panel.diagnostics,
        )
        self._factor_panels = process_factors(self._panel, self.cards)

    @property
    def state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def status(self) -> dict[str, Any]:
        state = self._state
        return {
            "session_id": state["session_id"],
            "status": state["status"],
            "candidate_id": self.contract["candidate_id"],
            "forward_started": bool(state["forward_started"]),
            "paper_started": bool(state["paper_started"]),
            "live_bars": int(state["live_bars"]),
            "replay_bars": int(state["replay_bars"]),
            "event_sequence": int(state["event_sequence"]),
            "last_ledger_timestamp": state["last_ledger_timestamp"],
            "equity": float(state["account"]["equity"]),
            "drawdown": float(state["account"]["drawdown"]),
            "rebalances": int(state["rebalances"]),
        }

    def activate(self, activated_at: Any) -> None:
        timestamp = _stamp(activated_at, "activated_at")
        start = _stamp(self.contract["observation"]["start"], "observation start")
        if self.contract["mode"] != "forward":
            raise ForwardSessionError("offline event-replay sessions cannot be activated")
        if self._state["forward_started"]:
            raise ForwardSessionError("forward session is already active")
        if self._state["status"] != "prepared":
            raise ForwardSessionError("only a prepared forward session can be activated")
        if timestamp < self._last_received or timestamp > start:
            raise ForwardSessionError("activation must follow seed receipt and precede the frozen start")
        self._commit_event(
            {"kind": "activate", "timestamp": timestamp.isoformat()},
            timestamp,
            "control",
            f"activate|{timestamp.isoformat()}",
        )

    def ingest_event(self, event: Mapping[str, Any], *, received_at: Any) -> str:
        if self.contract["mode"] != "forward":
            raise ForwardSessionError("offline event-replay sessions cannot ingest live events")
        if not self._state["forward_started"]:
            raise ForwardSessionError("activate explicitly before ingesting live events")
        return self._ingest(event, received_at=received_at, origin="live")

    def replay_event(self, event: Mapping[str, Any], *, received_at: Any) -> str:
        """Apply a local fixture event without marking forward observation started."""
        if self.contract["mode"] != "offline_event_replay":
            raise ForwardSessionError("replay_event requires an offline_event_replay session")
        if self._state["forward_started"]:
            raise ForwardSessionError("offline replay cannot append to an active forward session")
        return self._ingest(event, received_at=received_at, origin="offline_replay")

    def _normalize_event(self, event: Mapping[str, Any], received_at: Any) -> tuple[dict[str, Any], pd.Timestamp, str, str]:
        if not isinstance(event, Mapping):
            raise ValueError("event must be a mapping")
        kind = event.get("kind")
        if kind == "bar_close":
            expected = {"kind", "timestamp", "symbol", "feature_values", "feature_available_at"}
            if set(event) != expected:
                raise ValueError(f"bar_close event fields must be exactly {sorted(expected)}")
            timestamp = _stamp(event["timestamp"], "bar_close timestamp")
            if timestamp != timestamp.floor("h"):
                raise ValueError("bar_close timestamp must be an exact UTC bar-open label")
            symbol = event["symbol"]
            if symbol not in self.symbols:
                raise ValueError(f"unsupported forward symbol: {symbol}")
            feature_values = event["feature_values"]
            available = event["feature_available_at"]
            if not isinstance(feature_values, Mapping) or set(feature_values) != self.required_fields:
                raise ValueError(f"bar_close feature_values must match frozen fields: {sorted(self.required_fields)}")
            if not isinstance(available, Mapping) or set(available) != self.required_fields:
                raise ValueError("feature_available_at fields must match feature_values exactly")
            values = {}
            times = {}
            for name in sorted(self.required_fields):
                raw = feature_values[name]
                if raw is None or pd.isna(raw):
                    values[name] = None
                    if available.get(name) is not None and not pd.isna(available.get(name)):
                        times[name] = _stamp(available[name], f"{name} available_at").isoformat()
                    else:
                        times[name] = None
                else:
                    numeric = float(raw)
                    if not np.isfinite(numeric):
                        raise ValueError(f"feature {name} must be finite or null")
                    values[name] = numeric
                    times[name] = _stamp(available[name], f"{name} available_at").isoformat()
            if values["perp_close"] is None or values["mark_close"] is None:
                raise ValueError("perp_close and mark_close are required for account valuation")
            if self.contract["mode"] == "forward":
                bar_complete = timestamp + HOUR
                if any(
                    _stamp(value, f"{name} available_at") < bar_complete
                    for name, value in times.items() if value is not None
                ):
                    raise ValueError("forward feature available_at predates its completed bar")
            payload = {"kind": kind, "timestamp": timestamp.isoformat(), "symbol": symbol,
                       "feature_values": values, "feature_available_at": times}
            logical_id = f"bar_close|{timestamp.isoformat()}|{symbol}"
        elif kind == "quote":
            expected = {"kind", "execution_timestamp", "symbol", "quote_at", "price", "available_at"}
            if set(event) != expected:
                raise ValueError(f"quote event fields must be exactly {sorted(expected)}")
            timestamp = _stamp(event["quote_at"], "quote_at")
            execution = _stamp(event["execution_timestamp"], "execution_timestamp")
            if execution != execution.floor("h"):
                raise ValueError("execution_timestamp must be an exact UTC hour boundary")
            symbol = event["symbol"]
            price = float(event["price"])
            available_at = _stamp(event["available_at"], "quote available_at")
            if symbol not in self.symbols or not np.isfinite(price) or price <= 0:
                raise ValueError("quote symbol and price must be valid")
            if available_at < timestamp:
                raise ValueError("quote cannot be available before quote_at")
            payload = {"kind": kind, "timestamp": timestamp.isoformat(),
                       "execution_timestamp": execution.isoformat(), "symbol": symbol,
                       "quote_at": timestamp.isoformat(), "price": price,
                       "available_at": available_at.isoformat()}
            logical_id = f"quote|{timestamp.isoformat()}|{symbol}"
        elif kind == "funding":
            expected = {"kind", "timestamp", "symbol", "funding_rate", "mark_price", "available_at"}
            if set(event) != expected:
                raise ValueError(f"funding event fields must be exactly {sorted(expected)}")
            timestamp = _stamp(event["timestamp"], "funding timestamp")
            symbol = event["symbol"]
            rate = float(event["funding_rate"])
            mark = float(event["mark_price"])
            available_at = _stamp(event["available_at"], "funding available_at")
            if symbol not in self.symbols or not np.isfinite(rate) or not np.isfinite(mark) or mark <= 0:
                raise ValueError("funding event symbol, rate, or mark is invalid")
            if available_at < timestamp:
                raise ValueError("funding data cannot be available before its settlement event")
            payload = {"kind": kind, "timestamp": timestamp.isoformat(), "symbol": symbol,
                       "funding_rate": rate, "mark_price": mark,
                       "available_at": available_at.isoformat()}
            logical_id = f"funding|{timestamp.isoformat()}|{symbol}"
        else:
            raise ValueError("supported event kinds are bar_close, quote, and funding")
        received = _stamp(received_at, "received_at")
        available_values = (
            [_stamp(value, "feature available_at") for value in payload["feature_available_at"].values() if value is not None]
            if kind == "bar_close" else [_stamp(payload["available_at"], "available_at")]
        )
        if available_values and max(available_values) > received:
            raise ValueError("received_at cannot precede any field's available_at")
        return payload, received, logical_id, _hash(payload)

    def _ingest(self, event: Mapping[str, Any], *, received_at: Any, origin: str) -> str:
        payload, received, logical_id, fingerprint = self._normalize_event(event, received_at)
        existing = self._seen.get(logical_id)
        if existing is not None:
            if existing == fingerprint:
                return "duplicate"
            raise ForwardSessionError(f"conflicting duplicate event: {logical_id}")
        if received < self._last_received:
            raise ForwardSessionError("new events must be appended in nondecreasing received_at order")
        return self._commit_event(payload, received, origin, logical_id, fingerprint)

    def _commit_event(self, payload: dict[str, Any], received: pd.Timestamp, origin: str,
                      logical_id: str, fingerprint: str | None = None) -> str:
        fingerprint = fingerprint or _hash(payload)
        if logical_id in self._seen:
            if self._seen[logical_id] == fingerprint:
                return "duplicate"
            raise ForwardSessionError(f"conflicting duplicate event: {logical_id}")
        if received < self._last_received:
            raise ForwardSessionError("new events must be appended in nondecreasing received_at order")
        next_state = copy.deepcopy(self._state)
        outcome = self._apply_payload(next_state, payload, received, origin)
        record = {
            "sequence": int(self._state["event_sequence"]) + 1,
            "previous_event_hash": self._state["last_event_hash"],
            "logical_id": logical_id,
            "fingerprint": fingerprint,
            "received_at": received.isoformat(),
            "origin": origin,
            "payload": payload,
            "outcome": outcome,
        }
        record["event_hash"] = _journal_hash(record)
        event_path = self.root / "events.jsonl"
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        next_state["event_sequence"] = record["sequence"]
        next_state["last_event_hash"] = record["event_hash"]
        next_state["last_received_at"] = received.isoformat()
        next_state["seen_events"][logical_id] = fingerprint
        _save_state(self.root, next_state)
        self._state = next_state
        self._seen = dict(next_state["seen_events"])
        self._last_received = received
        return outcome

    def _apply_payload(self, state: dict[str, Any], payload: dict[str, Any], received: pd.Timestamp, origin: str) -> str:
        kind = payload["kind"]
        if kind == "activate":
            start = _stamp(self.contract["observation"]["start"], "observation start")
            if received > start or state["forward_started"]:
                raise ForwardSessionError("activation must occur before the frozen observation start")
            state["forward_started"] = True
            state["status"] = "active"
            state["activated_at"] = received.isoformat()
            return "activated"
        if state["status"].startswith("halted") or state["status"] in {"observation_complete", "insufficient_rebalances"}:
            return "ignored_halted"

        timestamp = _stamp(payload["timestamp"], "event timestamp")
        start = _stamp(self.contract["observation"]["start"], "observation start")
        end = _stamp(self.contract["observation"]["end_exclusive"], "observation end")
        lower_bound = start - HOUR if kind == "bar_close" else start
        if timestamp < lower_bound or timestamp >= end:
            raise ForwardSessionError("new event timestamp must lie inside the frozen observation window")
        if kind == "quote":
            return self._accept_quote(state, payload, received, origin)
        if kind == "bar_close":
            return self._accept_close(state, payload, received, origin)
        return self._accept_funding(state, payload, received)

    def _accept_quote(self, state: dict[str, Any], payload: dict[str, Any], received: pd.Timestamp, origin: str) -> str:
        quote_at = _stamp(payload["quote_at"], "quote_at")
        execution = _stamp(payload["execution_timestamp"], "execution_timestamp")
        symbol = payload["symbol"]
        orders = state["pending_orders"].get(execution.isoformat())
        if orders is None or symbol not in orders:
            state.setdefault("quotes_without_pending_order", []).append({
                "quote_at": quote_at.isoformat(), "received_at": received.isoformat(),
                "execution_timestamp": execution.isoformat(), "symbol": symbol,
                "price": float(payload["price"]),
            })
            return "quote_recorded_without_pending_order"

        order = orders[symbol]
        signal_received = _stamp(order["signal_received_at"], "signal received_at")
        available_at = _stamp(payload["available_at"], "quote available_at")
        window_end = execution + HOUR
        if quote_at < signal_received or available_at < signal_received:
            state.setdefault("ineligible_quotes", []).append({
                "quote_at": quote_at.isoformat(), "received_at": received.isoformat(),
                "execution_timestamp": execution.isoformat(), "symbol": symbol,
                "reason": "quote_predates_signal_receipt",
            })
            return "ineligible_quote"
        if received >= window_end or quote_at >= window_end:
            state["missed_executions"].append({
                "timestamp": received.isoformat(), "execution_timestamp": execution.isoformat(),
                "signal_timestamp": order["signal_timestamp"], "symbol": symbol,
                "reason": "no_executable_quote_inside_signal_window",
            })
            orders.pop(symbol)
            if not orders:
                state["pending_orders"].pop(execution.isoformat(), None)
            return "missed_execution"

        delta = float(order["signed_order_quantity"])
        old_quantity = float(state["account"]["quantities"][symbol])
        if delta != 0.0:
            reference = float(payload["price"])
            fill_price, notional, fee, slippage = _fill_terms(
                delta,
                reference,
                self.contract["costs"]["fee_bps"] / 10_000.0,
                self.contract["costs"]["slippage_bps"] / 10_000.0,
            )
            new_cash, realized, new_quantity, new_entry = _apply_fill(
                float(state["account"]["cash"]), old_quantity,
                float(state["account"]["entries"][symbol])
                if state["account"]["entries"][symbol] is not None else np.nan,
                delta, fill_price, fee,
            )
            state["account"]["cash"] = new_cash
            state["account"]["quantities"][symbol] = new_quantity
            state["account"]["entries"][symbol] = new_entry if np.isfinite(new_entry) else None
            totals = state["bar_totals"].setdefault(execution.isoformat(), self._empty_totals())
            totals["realized_pnl"] += realized
            totals["fees"] += fee
            totals["slippage_cost"] += slippage
            totals["trade_notional"] += notional
            state["fills"].append({
                "timestamp": received.isoformat(), "quote_at": quote_at.isoformat(),
                "scheduled_open": execution.isoformat(),
                "quote_lag_seconds": (received - execution).total_seconds(),
                "symbol": symbol, "side": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta), "signed_quantity": delta,
                "reference_price": reference, "fill_price": fill_price,
                "notional": notional, "fee": fee, "slippage_cost": slippage,
                "realized_pnl": realized, "previous_quantity": old_quantity,
                "new_quantity": new_quantity,
                "average_entry_price": new_entry if np.isfinite(new_entry) else None,
                "signal_timestamp": order["signal_timestamp"],
            })
        orders.pop(symbol)
        if not orders:
            state["pending_orders"].pop(execution.isoformat(), None)
        if origin == "live":
            state["live_quotes"] = int(state.get("live_quotes", 0)) + 1
        else:
            state["replay_quotes"] = int(state.get("replay_quotes", 0)) + 1
        return "quote_filled"

    @staticmethod
    def _empty_totals() -> dict[str, float]:
        return {name: 0.0 for name in ("realized_pnl", "funding_cashflow", "fees", "slippage_cost", "trade_notional")}

    def _accept_funding(self, state: dict[str, Any], payload: dict[str, Any], received: pd.Timestamp) -> str:
        timestamp = _stamp(payload["timestamp"], "funding timestamp")
        last_mark = state["last_market_mark_timestamp"]
        late = (last_mark is not None and timestamp < _stamp(last_mark, "last market mark timestamp"))
        last_fill = max(
            (_stamp(fill["timestamp"], "fill timestamp") for fill in state["fills"]),
            default=None,
        )
        late = late or (last_fill is not None and timestamp <= last_fill)
        symbol = payload["symbol"]
        quantity = float(state["account"]["quantities"][symbol])
        cashflow = _funding_cashflow(quantity, payload["funding_rate"], payload["mark_price"])
        row = {
            "timestamp": timestamp.isoformat(), "received_at": received.isoformat(),
            "symbol": symbol, "quantity": quantity, "funding_rate": payload["funding_rate"],
            "mark_price": payload["mark_price"], "cashflow": cashflow,
            "applied": not late,
        }
        state["funding_events"].append(row)
        if late:
            state["status"] = "halted_late_funding"
            state["late_event"] = {"kind": "funding", "timestamp": timestamp.isoformat(),
                                    "reason": "event_arrived_after_its_accounting_watermark"}
            return "late_unapplied"
        state["account"]["cash"] += cashflow
        bar_open = timestamp.floor("h")
        totals = state["bar_totals"].setdefault(bar_open.isoformat(), self._empty_totals())
        totals["funding_cashflow"] += cashflow
        event_marks = dict(state["account"]["latest_marks"])
        event_marks[symbol] = float(payload["mark_price"])
        try:
            equity, _, _, _ = _check_margin(
                timestamp=timestamp, cash=float(state["account"]["cash"]),
                quantities={key: float(value) for key, value in state["account"]["quantities"].items()},
                entries={key: float(value) if value is not None else np.nan for key, value in state["account"]["entries"].items()},
                marks={key: float(value) for key, value in event_marks.items()},
                margin_fraction=float(self.contract["portfolio"]["margin_fraction"]),
            )
            state["account"]["latest_marks"] = event_marks
            state["account"]["equity"] = equity
        except ValueError:
            state["status"] = "halted_margin_guard"
            state["late_event"] = {"kind": "funding", "timestamp": timestamp.isoformat(),
                                    "reason": "sampled_margin_guard_breach"}
            return "halted_margin_guard"
        return "funding_applied"

    def _accept_close(self, state: dict[str, Any], payload: dict[str, Any], received: pd.Timestamp, origin: str) -> str:
        timestamp = _stamp(payload["timestamp"], "bar_close timestamp")
        last_signal = _stamp(state["last_signal_timestamp"], "last signal timestamp")
        if timestamp <= last_signal:
            raise ForwardSessionError("bar_close timestamp is not after the last signal")
        group = state["close_buffer"].setdefault(timestamp.isoformat(), {})
        feature_values = payload["feature_values"]
        group[payload["symbol"]] = {
            "features": feature_values,
            "feature_available_at": payload["feature_available_at"],
            "received_at": received.isoformat(),
        }
        if set(group) != set(self.symbols):
            return "close_buffered"
        expected = last_signal + HOUR
        if timestamp != expected:
            state["status"] = "halted_missing_bar"
            state["missed_executions"].append({"timestamp": timestamp.isoformat(), "reason": "nonconsecutive_close"})
            return "halted_missing_bar"
        close_time = timestamp + HOUR
        last_mark = state["last_market_mark_timestamp"]
        if last_mark is not None and close_time <= _stamp(last_mark, "last market mark timestamp"):
            state["status"] = "halted_late_bar"
            state["late_event"] = {"kind": "bar_close", "timestamp": timestamp.isoformat(),
                                    "reason": "closed_feature_arrived_after_ledger_watermark"}
            return "halted_late_bar"
        max_received = max(_stamp(item["received_at"], "close received_at") for item in group.values())
        max_available = max(
            _stamp(value, "feature available_at")
            for item in group.values() for value in item["feature_available_at"].values()
            if value is not None
        )
        rows = []
        for symbol in self.symbols:
            values = {field: None for field in INPUT_COLUMNS}
            values.update(group[symbol]["features"])
            rows.append({"timestamp": timestamp.isoformat(), "symbol": symbol, "features": values})
        state["feature_rows"].append({
            "timestamp": timestamp.isoformat(),
            "features": {item["symbol"]: item["features"] for item in rows},
        })
        self._append_feature_rows(rows)
        close_prices = {symbol: float(group[symbol]["features"]["perp_close"]) for symbol in self.symbols}
        mark_prices = {symbol: float(group[symbol]["features"]["mark_close"]) for symbol in self.symbols}
        try:
            equity, unrealized, gross, maintenance = _check_margin(
                timestamp=close_time,
                cash=float(state["account"]["cash"]),
                quantities={key: float(value) for key, value in state["account"]["quantities"].items()},
                entries={key: float(value) if value is not None else np.nan for key, value in state["account"]["entries"].items()},
                marks=mark_prices,
                margin_fraction=float(self.contract["portfolio"]["margin_fraction"]),
            )
        except ValueError:
            state["status"] = "halted_margin_guard"
            state["late_event"] = {"kind": "bar_close", "timestamp": timestamp.isoformat(),
                                    "reason": "sampled_margin_guard_breach"}
            return "halted_margin_guard"
        account = state["account"]
        account["latest_marks"] = mark_prices
        account["equity"] = equity
        account["peak_equity"] = max(float(account["peak_equity"]), equity)
        account["drawdown"] = 1.0 - equity / float(account["peak_equity"])
        totals = state["bar_totals"].setdefault(timestamp.isoformat(), self._empty_totals())
        state["ledger"].append({
            "timestamp": max_received.isoformat(), "received_at": max_received.isoformat(),
            "market_timestamp": close_time.isoformat(), "cash": float(account["cash"]),
            **totals, "gross_notional": gross,
            "net_notional": sum(float(account["quantities"][s]) * mark_prices[s] for s in self.symbols),
            "unrealized_pnl": unrealized, "equity": equity,
            "return": equity / float(state["ledger"][-1]["equity"] if state["ledger"] else self.contract["costs"]["initial_capital"]) - 1.0,
            "drawdown": account["drawdown"], "maintenance_margin": maintenance,
        })
        state["last_ledger_timestamp"] = max_received.isoformat()
        state["last_market_mark_timestamp"] = close_time.isoformat()
        state["last_signal_timestamp"] = timestamp.isoformat()
        for symbol in self.symbols:
            quantity = float(account["quantities"][symbol])
            entry = account["entries"][symbol]
            state["positions"].append({
                "timestamp": close_time.isoformat(), "symbol": symbol,
                "quantity": quantity, "average_entry_price": entry,
                "mark_price": mark_prices[symbol], "signed_notional": quantity * mark_prices[symbol],
                "unrealized_pnl": quantity * (mark_prices[symbol] - float(entry)) if entry is not None else 0.0,
            })
        state["bar_totals"].pop(timestamp.isoformat(), None)
        expired = state["pending_orders"].pop(timestamp.isoformat(), {})
        for symbol, order in expired.items():
            state["missed_executions"].append({
                "timestamp": close_time.isoformat(),
                "execution_timestamp": timestamp.isoformat(),
                "signal_timestamp": order["signal_timestamp"],
                "symbol": symbol,
                "reason": "no_quote_received_inside_execution_window",
            })
        if state["forward_started"]:
            state["live_bars"] += 1
        else:
            state["replay_bars"] += 1

        if account["drawdown"] >= float(self.contract["risk_stop"]["max_drawdown"]):
            state["status"] = "halted_max_drawdown"
            return "halted_max_drawdown"
        if close_time >= _stamp(self.contract["observation"]["end_exclusive"], "observation end"):
            state["status"] = (
                "observation_complete"
                if state["rebalances"] >= int(self.contract["observation"]["minimum_rebalances"])
                else "insufficient_rebalances"
            )
            return state["status"]

        scores = self._scores_at(timestamp)
        state["score_history"].append({
            "timestamp": timestamp.isoformat(),
            "scores": {symbol: (float(scores[symbol]) if pd.notna(scores[symbol]) else None) for symbol in self.symbols},
        })
        history = pd.DataFrame(
            [row["scores"] for row in state["score_history"]],
            index=pd.DatetimeIndex([_stamp(row["timestamp"], "score timestamp") for row in state["score_history"]]),
        ).reindex(columns=self.symbols)
        targets = generate_rank_targets(
            history,
            long_count=self.contract["portfolio"]["long_count"],
            short_count=self.contract["portfolio"]["short_count"],
            gross_exposure=self.contract["portfolio"]["gross_exposure"],
            max_asset_weight=self.contract["portfolio"]["max_asset_weight"],
            rebalance_hours=self.contract["portfolio"]["rebalance_hours"],
            start=_stamp(self.contract["observation"]["start"], "observation start"),
        )
        if timestamp not in targets.index:
            return "bar_closed"
        execution = close_time
        if max_received >= execution + HOUR or max_available >= execution + HOUR:
            state["missed_executions"].append({
                "timestamp": execution.isoformat(), "signal_timestamp": timestamp.isoformat(),
                "reason": "signal_arrived_after_execution_window",
            })
            return "missed_execution"

        order_intents = {}
        for symbol in self.symbols:
            target_weight = float(targets.loc[timestamp, symbol])
            signal_close = close_prices[symbol]
            target_quantity = _target_quantity(target_weight, equity, signal_close)
            current_quantity = float(account["quantities"][symbol])
            order = {
                "signal_timestamp": timestamp.isoformat(),
                "signal_received_at": max_received.isoformat(),
                "execution_timestamp": execution.isoformat(),
                "symbol": symbol, "target_weight": target_weight,
                "signal_close": signal_close, "signal_equity": equity,
                "current_quantity": current_quantity,
                "target_quantity": target_quantity,
                "signed_order_quantity": target_quantity - current_quantity,
                "order_status": "pending",
            }
            order_intents[symbol] = order
            state["orders"].append(order)
        state["pending_orders"][execution.isoformat()] = order_intents
        state["rebalances"] += 1
        return "signal_prepared"

    def _append_feature_rows(self, rows: list[dict[str, Any]]) -> None:
        timestamp = _stamp(rows[0]["timestamp"], "feature timestamp")
        values = []
        for row in rows:
            aligned = {field: np.nan for field in INPUT_COLUMNS}
            aligned.update({key: (np.nan if value is None else value) for key, value in row["features"].items()})
            values.append(pd.Series(aligned, name=(timestamp, row["symbol"])))
        frame = pd.DataFrame(values)
        frame.index = pd.MultiIndex.from_tuples(frame.index, names=["timestamp", "symbol"])
        member = pd.Series(True, index=frame.index, name="eligible", dtype=bool)
        self._panel = FactorInputPanel(
            values=pd.concat([self._panel.values, frame]).sort_index(),
            universe=pd.concat([self._panel.universe, member]).sort_index(),
            diagnostics=self._panel.diagnostics,
        )
        self._factor_panels = process_factors(self._panel, self.cards)

    def _scores_at(self, timestamp: pd.Timestamp) -> pd.Series:
        common = self._factor_panels.common_mask.xs(timestamp, level="timestamp")
        standardized = self._factor_panels.standardized.xs(timestamp, level="timestamp")
        score = standardized[self.contract["selected_card_ids"]].mean(axis=1).where(common)
        return score.reindex(self.symbols)

    @classmethod
    def open(cls, root: Path) -> "ForwardSession":
        root = Path(root).resolve()
        contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        _verify_seed_manifest(root, contract)
        seed = _read_seed(root)
        card_records = contract["cards"]
        state_saved = json.loads((root / "state.json").read_text(encoding="utf-8"))
        actual_contract_hash = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
        if state_saved.get("frozen_contract_sha256") != actual_contract_hash:
            raise ForwardSessionError("frozen forward contract changed")
        initial_state = _initial_state_from_contract(root, contract, seed, card_records)
        session = cls(root, contract, seed, card_records,
                      set(contract["required_feature_fields"]), initial_state)
        session._seen = {}
        events_path = root / "events.jsonl"
        previous = "genesis"
        rows = []
        saved_sequence = int(state_saved.get("event_sequence", -1))
        if saved_sequence < 0:
            raise ForwardSessionError("forward state lacks an event sequence")
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if saved_sequence > len(rows):
            raise ForwardSessionError("forward state is ahead of its event journal")
        state_prefix_hash = _state_hash(session._state) if saved_sequence == 0 else None
        for expected_sequence, record in enumerate(rows, start=1):
            if record.get("sequence") != expected_sequence or record.get("previous_event_hash") != previous:
                raise ForwardSessionError("forward event journal sequence/hash chain is broken")
            if _journal_hash(record) != record.get("event_hash"):
                raise ForwardSessionError("forward event journal hash mismatch")
            logical_id = record["logical_id"]
            if logical_id in session._seen:
                raise ForwardSessionError("forward event journal repeats a logical event ID")
            payload = record["payload"]
            if record.get("fingerprint") != _hash(payload):
                raise ForwardSessionError("forward event payload fingerprint mismatch")
            received = _stamp(record["received_at"], "journal received_at")
            if received < session._last_received:
                raise ForwardSessionError("forward event receipt order moved backwards")
            outcome = session._apply_payload(session._state, payload, received, record["origin"])
            if outcome != record["outcome"]:
                raise ForwardSessionError("forward event replay outcome differs from journal")
            session._state["event_sequence"] = expected_sequence
            session._state["last_event_hash"] = record["event_hash"]
            session._state["last_received_at"] = received.isoformat()
            session._state["seen_events"][logical_id] = record["fingerprint"]
            session._seen[logical_id] = record["fingerprint"]
            session._last_received = received
            previous = record["event_hash"]
            if expected_sequence == saved_sequence:
                state_prefix_hash = _state_hash(session._state)
        if state_prefix_hash != state_saved.get("state_hash"):
            raise ForwardSessionError("forward state snapshot does not match its journal prefix")
        if saved_sequence < len(rows):
            _save_state(root, session._state)
        else:
            session._state = state_saved
        session._seen = dict(session._state["seen_events"])
        session._last_received = _stamp(session._state["last_received_at"], "last_received_at")
        return session


def _initial_state_from_contract(root: Path, contract: dict[str, Any], seed: FreshSeedData,
                                 cards: list[dict[str, Any]]) -> dict[str, Any]:
    start = _stamp(contract["observation"]["start"], "observation start")
    seed_last = _stamp(contract["seed"]["last_signal_timestamp"], "seed last signal")
    symbols = list(contract["factor_standardization"]["universe_symbols"])
    factors = process_factors(seed.panel, cards)
    common = factors.common_mask.xs(seed_last, level="timestamp")
    standardized = factors.standardized.xs(seed_last, level="timestamp")
    score = standardized[contract["selected_card_ids"]].mean(axis=1).where(common).reindex(symbols)
    initial_equity = float(contract["costs"]["initial_capital"])
    seed_received = max(_stamp(seed.received_at.loc[(seed_last, symbol)], "seed received_at") for symbol in symbols)
    pending = {}
    orders = []
    mode = contract["mode"]
    return {
        "schema_version": FORWARD_SCHEMA_VERSION, "session_id": root.name,
        "mode": mode,
        "status": "prepared", "forward_started": False, "paper_started": False,
        "live_bars": 0, "replay_bars": 0, "event_sequence": 0,
        "last_event_hash": "genesis", "last_received_at": seed_received.isoformat(),
        "frozen_contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
        "account": {
            "cash": initial_equity, "quantities": {symbol: 0.0 for symbol in symbols},
            "entries": {symbol: None for symbol in symbols},
            "latest_marks": {symbol: float(seed.frames[symbol].loc[seed_last, "mark_close"]) for symbol in symbols},
            "equity": initial_equity, "peak_equity": initial_equity, "drawdown": 0.0,
        },
        "seed_signal_timestamp": seed_last.isoformat(),
        "last_signal_timestamp": seed_last.isoformat(), "last_ledger_timestamp": None,
        "last_market_mark_timestamp": seed_last.isoformat(),
        "score_history": [{"timestamp": seed_last.isoformat(),
                           "scores": {symbol: (float(score[symbol]) if pd.notna(score[symbol]) else None) for symbol in symbols}}],
        "feature_rows": [], "close_buffer": {},
        "pending_orders": ({start.isoformat(): pending} if pending else {}), "orders": orders, "fills": [],
        "funding_events": [], "missed_executions": [], "positions": [], "ledger": [],
        "bar_totals": {}, "seen_events": {},
        "rebalances": 0,
    }
