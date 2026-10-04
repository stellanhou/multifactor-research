"""Orchestrate the frozen 90-day multifactor combination experiment.

The search evaluator and account engine remain the authorities for candidate
and account performance. This module loads the already frozen h24 source,
builds causal windows, checkpoints complete candidate tables, and resumes route
runs without replacing completed evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

# The experiment's numerical routines are single-threaded unless the runtime
# explicitly launches a bounded worker process.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

from . import multifactor_account as account
from . import multifactor_combination_search as search
from . import multifactor_net_sharpe as net
from . import multifactor_selection as selection
from . import multifactor_selection_research as research
from .multifactor_contracts import ResearchContract


HORIZON = 24
HOUR = pd.Timedelta(hours=1)
WEEK_HOURS = 168
TRAINING_HOURS = 90 * 24
VALIDATION_HOURS = 90 * 24
REFIT_HOURS = 90 * 24
MIN_TRAINING_LABEL_PERIODS = 720
MIN_VALIDATION_LABEL_PERIODS = 168
CLUSTER_COUNT = 8
CLUSTER_CANDIDATES = 2
SUBSET_SIZE = 5
ALPHAS = [0.00001, 0.001, 0.1, 10.0]
L1_RATIO = 0.5
INITIAL_OBJECTIVE_KIND = "unscored_empty_subset"
ADMISSION_RULE = {
    "validation_net_return_strictly_positive": True,
    "validation_net_sharpe_strictly_positive": True,
    "trades_required": True,
    "positive_annualized_volatility_required": True,
    "failure_action": "cash_until_next_weekly_update",
}
EXECUTION_PROFILE = {"max_heavy_processes": 2, "blas_threads_per_process": 1,
                     "candidate_batch_size": 1024,
                     "execution_start_utc": "2026-10-03T15:46:35Z",
                     "extension_cutoff_utc": "2026-10-03T21:01:35Z",
                     "target_deadline_utc": "2026-10-03T21:46:35Z",
                     "extension_cutoff_elapsed_seconds": 18_900,
                     "target_total_elapsed_seconds": 21_600,
                     "execution_start_correction": {
                         "previously_recorded_start_utc": "2026-10-03T14:26:35Z",
                         "corrected_start_utc": "2026-10-03T15:46:35Z",
                         "basis": "goal.createdAt Unix timestamp 1791042395",
                     }}


def route_definitions() -> list[dict[str, Any]]:
    """Return the exact E1-E5 route grid frozen in the execution plan."""
    routes: list[dict[str, Any]] = []
    for days in (21, 90):
        routes.append({
            "route_id": f"E1_stepwise_ridge_v{days}", "experiment": "E1",
            "route_family": "stepwise", "method": "stepwise",
            "algorithm": "stepwise",
            "primary_selection_eligible": False,
            "synthesis": "ridge", "selection_window_days": days,
            "validation_hours": days * 24, "capacity": 8,
            "budget": 2048, "budget_unit": "feature_subset_proposal",
            "proposal_budget": True, "seed": 0,
        })
    for synthesis in ("equal_weight", "ridge", "elastic_net"):
        routes.append({
            "route_id": f"E2_full_{synthesis}", "experiment": "E2",
            "route_family": "full_pool", "method": "full_pool",
            "primary_selection_eligible": True,
            "synthesis": synthesis, "selection_window_days": 90,
            "validation_hours": VALIDATION_HOURS, "capacity": 53,
            "budget": 1, "budget_unit": "fixed_candidate", "seed": 0,
        })
    for algorithm, method in (("stepwise", "stepwise"), ("grid", "grid"), ("ga", "GA")):
        for synthesis in ("equal_weight", "ridge"):
            routes.append({
                "route_id": f"E3_{method.lower()}{'0' if algorithm == 'ga' else ''}_{synthesis}",
                "experiment": "E3", "route_family": method, "method": method,
                "primary_selection_eligible": True,
                "algorithm": algorithm, "synthesis": synthesis,
                "selection_window_days": 90, "validation_hours": VALIDATION_HOURS,
                "capacity": SUBSET_SIZE, "budget": "grid_unique_subsets",
                "budget_unit": "unique_nonempty_subset", "seed": 0,
            })
    for synthesis in ("equal_weight", "ridge"):
        routes.append({
            "route_id": f"E4_random_{synthesis}", "experiment": "E4",
            "route_family": "random", "method": "random", "algorithm": "random",
            "primary_selection_eligible": False,
            "synthesis": synthesis, "selection_window_days": 90,
            "validation_hours": VALIDATION_HOURS, "capacity": SUBSET_SIZE,
            "budget": "grid_unique_subsets", "budget_unit": "unique_nonempty_subset",
            "seed": 0,
        })
    for seed in (1, 2):
        for synthesis in ("equal_weight", "ridge"):
            routes.append({
                "route_id": f"E5_ga{seed}_{synthesis}", "experiment": "E5",
                "route_family": "GA", "method": "GA", "algorithm": "ga",
                "primary_selection_eligible": False,
                "synthesis": synthesis, "selection_window_days": 90,
                "validation_hours": VALIDATION_HOURS, "capacity": SUBSET_SIZE,
                "budget": "grid_unique_subsets", "budget_unit": "unique_nonempty_subset",
                "seed": seed,
            })
    return routes


def primary_selection_rule() -> dict[str, Any]:
    """Return the frozen development-only winner rule for the nine E2/E3 routes."""
    eligible_routes = [route for route in route_definitions()
                       if route["primary_selection_eligible"]]
    if len(eligible_routes) != 9 or {route["experiment"] for route in eligible_routes} != {"E2", "E3"}:
        raise RuntimeError("primary winner rule requires exactly the nine E2/E3 routes")
    route_ids = sorted(route["route_id"] for route in eligible_routes)
    return {
        "schema_version": 1,
        "decision_stage": "development",
        "eligible_experiments": ["E2", "E3"],
        "eligible_route_ids": route_ids,
        "excluded_experiments": ["E1", "E4", "E5"],
        "internal_validation_use": "descriptive_only; never selects or changes the winner",
        "admission": {
            "stage": "development",
            "base": {
                "net_return": {"operator": ">", "value": 0.0},
                "net_sharpe": {"operator": ">", "value": 0.0, "finite": True},
                "traded_bars": {"operator": ">", "value": 0},
                "max_drawdown": {"operator": "<=", "value": 0.15},
            },
            "stress": {"net_return": {"operator": ">=", "value": 0.0}},
        },
        "ranking": {
            "primary_metric": "development_base_net_sharpe",
            "primary_direction": "descending",
            "tie_break_field": "route_id",
            "tie_break_direction": "ascending_lexicographic",
            "fixed_route_identity_order": route_ids,
        },
    }


def route_priority_order() -> list[str]:
    """Plan order, including the prescribed E3 cutoff priority."""
    by_id = {route["route_id"]: route for route in route_definitions()}
    order = [route["route_id"] for route in route_definitions() if route["experiment"] in {"E1", "E2"}]
    for method in ("grid", "ga0", "stepwise"):
        order.extend(route_id for route_id in by_id
                     if route_id.startswith(f"E3_{method}_"))
    order.extend(route_id for route_id in by_id if route_id.startswith("E4_"))
    order.extend(route_id for route_id in by_id if route_id.startswith("E5_ga1_"))
    order.extend(route_id for route_id in by_id if route_id.startswith("E5_ga2_"))
    return order


@dataclass
class PreparedInputs:
    source_root: Path
    factors: pd.DataFrame
    opens: pd.Series
    frames: dict[str, pd.DataFrame]
    funding: pd.DataFrame
    dev_contract: ResearchContract
    validation_contract: ResearchContract
    pool: dict[str, Any]
    factor_names: list[str]
    card_order: list[str]
    directions: dict[str, int]
    symbols: list[str]
    timestamps: pd.DatetimeIndex
    factor_cube: np.ndarray
    dataset: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]
    market_full: net._MarketArrays
    market_funding: pd.DataFrame
    source_hashes: dict[str, str]
    hash_paths: dict[str, Path]
    source_freeze_reference_sha256: str
    overlap_audit: dict[str, Any]
    factor_availability: pd.DataFrame

    def verify_source_hashes(self) -> None:
        changed = [relative for relative, digest in self.source_hashes.items()
                   if _sha256(self.hash_paths[relative]) != digest]
        if changed:
            raise ValueError(f"frozen source inputs changed during the run: {changed[:5]}")


@dataclass
class WindowContext:
    prepared: PreparedInputs
    windows: search.Windows
    training_sample_times: list[pd.Timestamp]
    x_train: np.ndarray
    y_train: np.ndarray
    refit_source_times: list[pd.Timestamp]
    x_refit: np.ndarray
    y_refit: np.ndarray
    training_values: np.ndarray
    validation_values: np.ndarray
    validation_complete: np.ndarray
    refit_values: np.ndarray
    validation_times: pd.DatetimeIndex
    training_market: net._MarketArrays
    validation_market: net._MarketArrays
    training_funding: pd.DataFrame
    validation_funding: pd.DataFrame
    evaluator_factory: Callable[..., search.SubsetEvaluator]
    initial_objective_kind: str = INITIAL_OBJECTIVE_KIND
    training_label_periods: int = 0
    validation_label_periods: int = 0
    readiness_status: str = "ready"

    def prescreen_evaluator(self, batch_candidates: int = 1024) -> search.SubsetEvaluator:
        empty_x = np.empty((0, len(self.prepared.factor_names)), dtype=float)
        empty_y = np.empty(0, dtype=float)
        return search.SubsetEvaluator(
            method="equal", x_train=empty_x, y_train=empty_y,
            values=self.training_values,
            source_times=self.windows.training,
            fit_time=self.windows.validation[0],
            market=self.training_market, funding=self.training_funding,
            contract=self.prepared.dev_contract,
            batch_candidates=batch_candidates,
        )

    def refit_coefficients(self, synthesis: str, subset: tuple[int, ...],
                           alpha: float | None) -> tuple[np.ndarray, dict[str, Any]]:
        return _refit_coefficients(self, synthesis=synthesis, subset=subset, alpha=alpha)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(research._json_safe(value), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    payload = json.dumps(research._json_safe(value), ensure_ascii=False, indent=2,
                         sort_keys=True, allow_nan=False) + "\n"
    return payload.encode("utf-8")


def _atomic_write(path: Path, payload: bytes, *, refuse_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                          dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_existing and path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace completed evidence: {path}")
            return
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any, *, refuse_existing: bool = False) -> None:
    _atomic_write(path, _json_bytes(value), refuse_existing=refuse_existing)


def _write_gzip_json(path: Path, value: Any, *, refuse_existing: bool = True) -> None:
    body = json.dumps(research._json_safe(value), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = gzip.compress(body, compresslevel=1, mtime=0)
    _atomic_write(path, compressed, refuse_existing=refuse_existing)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _relative_to_source(path: Path, source: Path) -> str:
    return path.resolve().relative_to(source.resolve()).as_posix()


def _input_paths(source: Path, stages: dict[str, research.StageInputs]) -> list[Path]:
    paths = [source / name for name in (
        "pool_contract.json", "plan.json", "dataset_manifest.json",
        "cards_manifest.json", "preparation_complete.json", "lifecycle.json", "universe.csv",
    )]
    for stage in ("development", "internal_validation"):
        inputs = stages[stage]
        paths.extend([
            source / stage / "h24" / "contract.json",
            source / stage / "h24" / "factor_panels" / "standardized.csv",
            source / stage / "inputs" / "panel.values.csv",
            source / stage / "inputs" / "funding.csv",
            *[source / stage / "inputs" / "market" / f"{symbol}.csv"
              for symbol in sorted(inputs.frames)],
        ])
        paths.extend(Path(card_path) for card_path in inputs.contract.cards)
    return sorted({path.resolve() for path in paths})


def _verify_input_freeze(source: Path, paths: list[Path]) -> tuple[dict[str, str], dict[str, Path], str]:
    preparation_path = source / "preparation_complete.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("status") != "passed" or preparation.get("symbol_count") != 30:
        raise ValueError("selection30 source preparation is not a passed original 30-asset freeze")
    prepared_files = preparation.get("source_files")
    if not isinstance(prepared_files, dict):
        raise ValueError("selection30 preparation manifest lacks source file hashes")

    selection_run = source.parent / "multifactor-selection30-20261003"
    previous_manifest_path = selection_run / "source_manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    hash_paths: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = _relative_to_source(path, source)
        digest = _sha256(path)
        expected_preparation = prepared_files.get(relative)
        if expected_preparation is not None and expected_preparation != digest:
            raise ValueError(f"source differs from preparation freeze: {relative}")
        previous_digest = previous_manifest.get(str(path.resolve()))
        if previous_digest is not None and previous_digest != digest:
            raise ValueError(f"h24 source differs from completed selection30 run: {relative}")
        hashes[relative] = digest
        hash_paths[relative] = path

    previous_h24_paths = {
        name for name in previous_manifest
        if "/development/" in name or "/internal_validation/" in name
        if "/h24/" in name or "/inputs/" in name
    }
    expected_h24_paths = {
        str(path.resolve()) for path in paths
        if "/development/" in str(path) or "/internal_validation/" in str(path)
    }
    if previous_h24_paths != expected_h24_paths:
        missing = sorted(expected_h24_paths - previous_h24_paths)
        extra = sorted(previous_h24_paths - expected_h24_paths)
        if missing or extra:
            raise ValueError(f"selection30 h24 execution manifest path set differs: missing={missing[:3]}, extra={extra[:3]}")
    reference_digest = _sha256(previous_manifest_path)
    hashes["../selection30/multifactor-selection30-20261003/source_manifest.json"] = reference_digest
    hash_paths["../selection30/multifactor-selection30-20261003/source_manifest.json"] = previous_manifest_path
    return hashes, hash_paths, reference_digest


def prepare_inputs(source_root: Path) -> PreparedInputs:
    """Load the original h24 development/C inputs and verify every frozen hash."""
    source = Path(source_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    pool, _ = research._source_contracts(source)
    source_plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    if source_plan.get("engineering_smoke_only") is not False:
        raise ValueError("combination90 requires the original historical 30-asset source")

    dev, _ = research._load_stage_inputs(source, "development", HORIZON, pool)
    validation, _ = research._load_stage_inputs(source, "internal_validation", HORIZON, pool)
    if dev.contract.cards != validation.contract.cards:
        raise ValueError("development and C h24 card lists differ")
    factors, opens, factor_overlap, availability = research._combine_model_inputs(
        dev, validation, min_symbols=3, horizon=HORIZON,
    )
    frames, funding, account_overlap = research._combined_frames_and_funding({
        "development": dev, "internal_validation": validation,
    })
    card_order = [Path(card).stem for card in dev.contract.cards]
    if list(factors.columns) != card_order:
        raise ValueError("h24 standardized factor columns differ from the frozen contract card order")
    if len(card_order) != 53 or len(set(card_order)) != 53:
        raise ValueError("the h24 frozen factor pool must contain 53 unique card identities")
    sorted_names = sorted(card_order)
    factors = factors.loc[:, sorted_names]
    card_paths = {Path(card).name: Path(card) for card in dev.contract.cards}
    cards_manifest = json.loads((source / "cards_manifest.json").read_text(encoding="utf-8"))
    directions: dict[str, int] = {}
    for card_name in card_order:
        filename = f"{card_name}.json"
        if filename not in cards_manifest:
            raise ValueError(f"frozen copied-card manifest misses {filename}")
        card_path = card_paths[filename]
        card = json.loads(card_path.read_text(encoding="utf-8"))
        direction = card.get("original_claim", {}).get("direction")
        if type(direction) is not int or direction not in {-1, 1}:
            raise ValueError(f"factor direction is not frozen for {card_name}")
        if cards_manifest[filename].get("sha256") != _sha256(card_path):
            raise ValueError(f"card snapshot checksum differs from its copied-card manifest: {filename}")
        directions[card_name] = direction

    timestamps, symbols = selection._validate_inputs(factors, opens, HORIZON)
    if symbols != sorted(frames):
        raise ValueError("h24 factor and market universes differ")
    factor_cube = factors.to_numpy(dtype=float).reshape(len(timestamps), len(symbols), len(sorted_names))
    dataset = selection._make_labels(factors, opens, timestamps, symbols, HORIZON, 3)
    if not dataset:
        raise ValueError("h24 source contains no complete matured cross-sectional labels")

    price_wide = opens.unstack("symbol").reindex(index=timestamps, columns=symbols)
    market_open = pd.DataFrame({symbol: frames[symbol]["open"].reindex(timestamps)
                                for symbol in symbols}, index=timestamps)
    source_root_by_stage = {
        stage: {
            "factor_label_open": str(source / stage / "inputs" / "panel.values.csv") + "#perp_open",
            "account_market_open": str(source / stage / "inputs" / "market" / "{symbol}.csv") + "#open",
        }
        for stage in ("development", "internal_validation")
    }
    development_timestamps = set(dev.opens.index.get_level_values("timestamp").unique())
    open_source_audit = _audit_label_account_open_consistency(
        price_wide,
        market_open,
        source_files_by_stage=source_root_by_stage,
        development_timestamps=development_timestamps,
    )

    market_full, market_funding = net._market_arrays(frames, symbols, funding)
    source_paths = _input_paths(source, {"development": dev, "internal_validation": validation})
    source_hashes, hash_paths, reference_sha = _verify_input_freeze(source, source_paths)
    return PreparedInputs(
        source_root=source, factors=factors, opens=opens, frames=frames, funding=funding,
        dev_contract=dev.contract, validation_contract=validation.contract, pool=pool,
        factor_names=sorted_names, card_order=card_order, directions=directions,
        symbols=symbols, timestamps=timestamps, factor_cube=factor_cube, dataset=dataset,
        market_full=market_full, market_funding=market_funding,
        source_hashes=source_hashes, hash_paths=hash_paths,
        source_freeze_reference_sha256=reference_sha,
        overlap_audit={"factor_overlap": factor_overlap,
                       "factor_availability_rows": len(availability),
                       "factor_complete_symbols_min": int(availability["complete_factor_symbols"].min()),
                       "factor_complete_symbols_median": float(availability["complete_factor_symbols"].median()),
                       "factor_complete_symbols_max": int(availability["complete_factor_symbols"].max()),
                       "factor_meets_minimum_hours": int(availability["meets_minimum_cross_section"].sum()),
                       "market_and_funding_overlap": account_overlap,
                       "label_account_open_consistency": open_source_audit},
        factor_availability=availability,
    )


def _index_cube(prepared: PreparedInputs, times: pd.DatetimeIndex) -> np.ndarray:
    positions = prepared.timestamps.get_indexer(times)
    if (positions < 0).any():
        raise ValueError("requested source calendar is missing from the frozen factor panel")
    return prepared.factor_cube[positions]


def _slice_market(prepared: PreparedInputs, start: pd.Timestamp,
                  end_exclusive: pd.Timestamp) -> net._MarketArrays:
    source = prepared.market_full
    start = pd.Timestamp(start).tz_convert("UTC")
    end_exclusive = pd.Timestamp(end_exclusive).tz_convert("UTC")
    left = source.position.get(start)
    right_last = source.position.get(end_exclusive - HOUR)
    if left is None or right_last is None or right_last < left:
        raise ValueError(f"requested bounded market slice is unavailable: [{start}, {end_exclusive})")
    right = right_last + 1
    index = source.index[left:right]
    return net._MarketArrays(
        symbols=source.symbols, index=index,
        position={pd.Timestamp(value): offset for offset, value in enumerate(index)},
        opens=source.opens[left:right], closes=source.closes[left:right],
        marks=source.marks[left:right], inactive=source.inactive[left:right],
    )


def _scope_market_funding(prepared: PreparedInputs, start: pd.Timestamp,
                          end_exclusive: pd.Timestamp) -> pd.DataFrame:
    return prepared.market_funding.loc[
        (prepared.market_funding["timestamp"] >= pd.Timestamp(start) + HOUR)
        & (prepared.market_funding["timestamp"] < pd.Timestamp(end_exclusive))
    ].copy()


def _full_calendar(prepared: PreparedInputs, window: search.Windows) -> bool:
    for name in ("training", "validation", "refit"):
        grid = getattr(window, name)
        positions = prepared.timestamps.get_indexer(grid)
        if (positions < 0).any():
            return False
    market = prepared.market_full
    for start, end_exclusive in ((window.training[0], window.validation[0]),
                                 (window.validation[0], window.fit_time)):
        start = pd.Timestamp(start).tz_convert("UTC")
        end_exclusive = pd.Timestamp(end_exclusive).tz_convert("UTC")
        duration = end_exclusive - start
        if duration <= pd.Timedelta(0) or duration % HOUR != pd.Timedelta(0):
            return False
        left = market.position.get(start)
        right = market.position.get(end_exclusive - HOUR)
        if left is None or right is None or right < left \
                or right - left + 1 != int(duration / HOUR):
            return False
    if window.validation[0] + HOUR < prepared.frames[prepared.symbols[0]].index[0] + HOUR:
        return False
    if window.fit_time > prepared.frames[prepared.symbols[0]].index[-1] + HOUR:
        return False
    return True


def _audit_label_account_open_consistency(
    label_opens: pd.DataFrame,
    account_opens: pd.DataFrame,
    *,
    source_files_by_stage: Mapping[str, Mapping[str, str]],
    development_timestamps: set[pd.Timestamp],
) -> dict[str, Any]:
    """Allow only the observed one-ULP SHIB price conversion difference."""
    if not label_opens.index.equals(account_opens.index):
        raise ValueError("factor label/account open timestamp grids differ")
    if not label_opens.columns.equals(account_opens.columns):
        raise ValueError("factor label/account open symbol columns differ")
    label_values = label_opens.to_numpy(dtype=np.float64)
    account_values = account_opens.to_numpy(dtype=np.float64)
    common = np.isfinite(label_values) & np.isfinite(account_values)
    if not common.any():
        raise ValueError("factor label and account opens have no overlapping finite observations")

    source_prices = np.ascontiguousarray(label_values[common])
    account_prices = np.ascontiguousarray(account_values[common])
    if np.any(source_prices <= 0.0) or np.any(account_prices <= 0.0):
        raise ValueError("finite label/account opening prices must be positive")
    source_bits = source_prices.view(np.uint64)
    account_bits = account_prices.view(np.uint64)
    # Positive finite IEEE-754 binary64 encodings sort in numeric order, so
    # their unsigned integer distance is the exact representable-step distance.
    ulp_distance = np.maximum(source_bits, account_bits) - np.minimum(source_bits, account_bits)
    max_ulp_distance = int(ulp_distance.max(initial=0))
    if max_ulp_distance > 1:
        raise ValueError(
            "factor label/account opens differ by more than one binary64 ULP: "
            f"max_ulp_distance={max_ulp_distance}"
        )

    different = label_values != account_values
    mismatch_rows, mismatch_columns = np.where(common & different)
    mismatch_bits = ulp_distance[source_prices != account_prices]
    histogram_values, histogram_counts = np.unique(mismatch_bits, return_counts=True)
    by_symbol = {
        symbol: int(np.sum(common[:, position] & different[:, position]))
        for position, symbol in enumerate(label_opens.columns)
        if np.any(common[:, position] & different[:, position])
    }
    examples = []
    for row, column in zip(mismatch_rows[:5], mismatch_columns[:5]):
        timestamp = pd.Timestamp(label_opens.index[row])
        symbol = str(label_opens.columns[column])
        stage = "development" if timestamp in development_timestamps else "internal_validation"
        paths = source_files_by_stage[stage]
        label_value = float(label_values[row, column])
        account_value = float(account_values[row, column])
        examples.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "factor_label_open": label_value,
            "account_market_open": account_value,
            "absolute_difference": abs(label_value - account_value),
            "relative_difference": abs(label_value - account_value) / max(label_value, account_value),
            "ulp_distance": int(abs(
                int(np.asarray(label_value, dtype=np.float64).view(np.uint64))
                - int(np.asarray(account_value, dtype=np.float64).view(np.uint64))
            )),
            "stage": stage,
            "factor_label_source": paths["factor_label_open"],
            "account_market_source": paths["account_market_open"].format(symbol=symbol),
        })
    absolute = np.abs(source_prices - account_prices)
    relative = absolute / np.maximum(source_prices, account_prices)
    return {
        "finite_comparisons": int(common.sum()),
        "different_values": int(different[common].sum()),
        "mismatch_fraction": float(different[common].mean()),
        "max_abs_difference": float(absolute.max(initial=0.0)),
        "max_relative_difference": float(relative.max(initial=0.0)),
        "max_ulp_distance": max_ulp_distance,
        "accepted_max_ulp_distance": 1,
        "mismatch_ulp_histogram": {
            str(int(distance)): int(count)
            for distance, count in zip(histogram_values, histogram_counts)
        },
        "mismatch_by_symbol": by_symbol,
        "examples": examples,
        "difference_context": "SHIB native scale uses reciprocal multiplication in factor inputs and integer division in account frames",
    }


def common_fit_times(prepared: PreparedInputs, *, validation_hours: int = VALIDATION_HOURS,
                     start: pd.Timestamp | None = None,
                     end: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """Return complete weekly fit timestamps on the original source refit phase."""
    due = pd.date_range(prepared.timestamps[0], prepared.timestamps[-1] + HOUR,
                        freq=f"{WEEK_HOURS}h", tz="UTC")
    selected = []
    for fit_time in due:
        if start is not None and fit_time < start:
            continue
        if end is not None and fit_time >= end:
            continue
        window = search.calendar_windows(
            fit_time, horizon=HORIZON, training_hours=TRAINING_HOURS,
            validation_hours=validation_hours, refit_hours=REFIT_HOURS,
        )
        if not _full_calendar(prepared, window):
            continue
        selected.append(pd.Timestamp(fit_time))
    if not selected:
        raise ValueError("no weekly update has complete 90-day training, validation, and refit calendars")
    return selected


def calibration_fit_times(prepared: PreparedInputs) -> dict[int, pd.Timestamp]:
    complete = common_fit_times(prepared, validation_hours=VALIDATION_HOURS)
    selected: dict[int, pd.Timestamp] = {}
    for year in (2023, 2024, 2025):
        lower = pd.Timestamp(f"{year}-03-01T00:00:00Z")
        upper = pd.Timestamp(f"{year}-04-01T00:00:00Z")
        candidates = [fit for fit in complete if lower <= fit < upper]
        if not candidates:
            raise ValueError(f"no complete original-phase weekly update in March {year}")
        selected[year] = candidates[0]
    return selected


def build_window_context(prepared: PreparedInputs, fit_time: pd.Timestamp, *,
                         validation_hours: int = VALIDATION_HOURS,
                         batch_candidates: int = 1024) -> WindowContext:
    """Build the exact train/selection/refit data slices for one weekly fit."""
    fit_time = pd.Timestamp(fit_time).tz_convert("UTC")
    windows = search.calendar_windows(
        fit_time, horizon=HORIZON, training_hours=TRAINING_HOURS,
        validation_hours=validation_hours, refit_hours=REFIT_HOURS,
    )
    if not _full_calendar(prepared, windows):
        raise ValueError(f"fit timestamp lacks a full source/market calendar: {fit_time}")
    times = pd.DatetimeIndex(sorted(prepared.dataset))
    train_sample_times = [pd.Timestamp(t) for t in times.intersection(windows.training)]
    x_train, y_train = selection._stack_sample(prepared.dataset, train_sample_times)
    training_values = _index_cube(prepared, windows.training)
    validation_values = _index_cube(prepared, windows.validation)
    refit_values = _index_cube(prepared, windows.refit)
    refit_source_times = [pd.Timestamp(t) for t in times.intersection(windows.refit)]
    x_refit, y_refit = selection._stack_sample(prepared.dataset, refit_source_times)
    validation_complete = np.isfinite(validation_values).all(axis=2)

    training_market = _slice_market(prepared, windows.training[0], windows.validation[0])
    training_funding = _scope_market_funding(prepared, windows.training[0], windows.validation[0])
    validation_market = _slice_market(prepared, windows.validation[0], fit_time)
    validation_funding = _scope_market_funding(prepared, windows.validation[0], fit_time)
    validation_times = windows.validation.rename("timestamp")
    validation_label_periods = len(times.intersection(windows.validation))
    readiness_status = "ready"
    if len(train_sample_times) < MIN_TRAINING_LABEL_PERIODS:
        readiness_status = "insufficient_matured_training_periods"
    elif validation_label_periods < MIN_VALIDATION_LABEL_PERIODS:
        readiness_status = "insufficient_matured_validation_periods"

    def evaluator_factory(method: str, candidate_batch: int = batch_candidates):
        return search.SubsetEvaluator(
            method=method, x_train=x_train, y_train=y_train,
            values=validation_values, source_times=validation_times,
            fit_time=fit_time, market=validation_market,
            funding=validation_funding, contract=prepared.dev_contract,
            batch_candidates=candidate_batch,
        )

    return WindowContext(
        prepared=prepared, windows=windows,
        training_sample_times=train_sample_times, x_train=x_train, y_train=y_train,
        refit_source_times=refit_source_times, x_refit=x_refit, y_refit=y_refit,
        training_values=training_values, validation_values=validation_values,
        validation_complete=validation_complete, refit_values=refit_values,
        validation_times=validation_times, training_market=training_market,
        validation_market=validation_market, training_funding=training_funding,
        validation_funding=validation_funding, evaluator_factory=evaluator_factory,
        training_label_periods=len(train_sample_times),
        validation_label_periods=validation_label_periods,
        readiness_status=readiness_status,
    )


def _cluster_record(context: WindowContext, *, batch_candidates: int = 1024) -> dict[str, Any]:
    if context.readiness_status != "ready":
        return {
            "fit_timestamp": context.windows.fit_time,
            "training_source_start": context.windows.training[0],
            "training_source_end": context.windows.training[-1],
            "training_account_start": context.windows.training[0] + HOUR,
            "training_account_end_exclusive": context.windows.validation[0],
            "cluster_count": CLUSTER_COUNT,
            "factor_names": context.prepared.factor_names,
            "groups": [], "group_factor_ids": [], "mean_correlation": None,
            "prescreen_candidates": [], "prescreen_candidate_ids": [],
            "prescreen_rows": [], "grid_unique_subsets": 0,
            "grid_subsets": [], "grid_factor_ids": [],
            "prescreen_evaluator_audit": {"unique_subset_evaluations": 0,
                                           "cache_hits": 0,
                                           "alpha_trial_evaluations": 0,
                                           "actual_account_evaluations": 0,
                                           "evaluation_seconds": 0.0,
                                           "coefficient_fit_seconds": 0.0,
                                           "account_scoring_seconds": 0.0},
            "training_sample_times": context.training_sample_times,
            "status": context.readiness_status,
            "training_label_periods": context.training_label_periods,
            "validation_label_periods": context.validation_label_periods,
        }
    groups, correlation = search.hierarchical_clusters(
        context.training_values, cluster_count=CLUSTER_COUNT,
    )
    evaluator = context.prescreen_evaluator(batch_candidates)
    candidates, prescreen_rows = search.prescreen(groups, evaluator)
    grid = search.enumerate_grid(candidates, SUBSET_SIZE)
    return {
        "fit_timestamp": context.windows.fit_time,
        "training_source_start": context.windows.training[0],
        "training_source_end": context.windows.training[-1],
        "training_account_start": context.windows.training[0] + HOUR,
        "training_account_end_exclusive": context.windows.validation[0],
        "cluster_count": CLUSTER_COUNT,
        "factor_names": context.prepared.factor_names,
        "groups": groups,
        "group_factor_ids": [[context.prepared.factor_names[i] for i in group]
                              for group in groups],
        "mean_correlation": correlation,
        "prescreen_candidates": candidates,
        "prescreen_candidate_ids": [[context.prepared.factor_names[i] for i in group]
                                     for group in candidates],
        "prescreen_rows": prescreen_rows,
        "grid_unique_subsets": len(grid),
        "grid_subsets": grid,
        "grid_factor_ids": [[context.prepared.factor_names[i] for i in subset]
                            for subset in grid],
        "prescreen_evaluator_audit": evaluator.audit(),
        "training_sample_times": context.training_sample_times,
        "status": "ready",
        "training_label_periods": context.training_label_periods,
        "validation_label_periods": context.validation_label_periods,
    }


def _refit_coefficients(context: WindowContext, *, synthesis: str,
                        subset: tuple[int, ...], alpha: float | None) -> tuple[np.ndarray, dict[str, Any]]:
    factor_count = len(context.prepared.factor_names)
    coefficients = np.zeros(factor_count, dtype=float)
    if synthesis == "equal_weight":
        coefficients[list(subset)] = 1.0 / len(subset)
        return coefficients, {"fit_sample_count": None, "target_rms": None,
                              "method": "standardized_equal_weight_no_regression"}
    if synthesis not in {"ridge", "elastic_net"}:
        raise ValueError(f"unknown synthesis method: {synthesis}")
    x_refit = context.x_refit[:, list(subset)]
    y_refit = context.y_refit
    if not len(y_refit):
        raise ValueError(f"selected model has no mature refit samples at {context.windows.fit_time}")
    target_rms = float(np.sqrt(np.mean(np.square(y_refit))))
    if not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError(f"selected model has invalid refit target scale at {context.windows.fit_time}")
    if alpha is None:
        raise ValueError("regularized models require the validation-selected alpha")
    if synthesis == "ridge":
        selected_coefficients = selection._ridge_coefficients(x_refit, y_refit / target_rms, float(alpha))
    else:
        selected_coefficients = selection._elastic_net_coefficients(
            x_refit, y_refit / target_rms, float(alpha), L1_RATIO,
        )
    coefficients[list(subset)] = selected_coefficients
    return coefficients, {
        "fit_sample_count": len(y_refit), "target_rms": target_rms,
        "method": "recent_2160_hours_matured_cross_sectional_returns",
    }


def _route_search(context: WindowContext, route: dict[str, Any],
                  cluster: dict[str, Any] | None) -> tuple[dict[str, Any], search.SubsetEvaluator, tuple[int, ...] | None]:
    synthesis = route["synthesis"]
    synthesis_key = {"equal_weight": "equal", "ridge": "ridge", "elastic_net": "elastic_net"}[synthesis]
    evaluator = context.evaluator_factory(synthesis_key)
    algorithm = route.get("algorithm", "full")
    if algorithm == "full":
        result = search.run_search(
            evaluator, algorithm="full", groups=[], grid=[], budget=1, size=53,
            seed=int(route["seed"]), factor_names=context.prepared.factor_names,
        )
    else:
        if route["experiment"] == "E1":
            groups, grid = [], []
        elif cluster is None:
            raise ValueError(f"{route['route_id']} requires its shared training cluster/prescreen record")
        else:
            groups = cluster["groups"]
            grid = [tuple(subset) for subset in cluster["grid_subsets"]]
        if algorithm == "stepwise" and route["experiment"] == "E1":
            budget = int(route["budget"])
            proposal_budget = True
        else:
            budget = len(grid)
            proposal_budget = False
        result = search.run_search(
            evaluator, algorithm=algorithm, groups=groups, grid=grid,
            budget=budget, size=int(route["capacity"]), seed=int(route["seed"]),
            factor_names=context.prepared.factor_names,
            proposal_budget=proposal_budget,
        )
    selected = tuple(result["selected"]) if result.get("selected") else None
    return result, evaluator, selected


def _candidate_rows(route: dict[str, Any], fit_time: pd.Timestamp,
                    factor_names: list[str], evaluator: search.SubsetEvaluator) -> list[dict[str, Any]]:
    rows = []
    for subset in sorted(evaluator.cache):
        alpha, objective, trials = evaluator.cache[subset]
        for trial in trials:
            rows.append({
                "route_id": route["route_id"], "fit_timestamp": fit_time,
                "window_id": fit_time.isoformat(),
                "experiment": route["experiment"],
                "route_family": route["route_family"],
                "method": route["method"], "synthesis": route["synthesis"],
                "selection_window_days": route["selection_window_days"],
                "seed": route["seed"],
                "subset_indices": list(subset),
                "subset_id": [factor_names[index] for index in subset],
                "identity": [factor_names[index] for index in subset],
                "alpha": trial.get("alpha"), "selected_alpha": alpha,
                "validation_net_sharpe": trial.get("validation_net_sharpe"),
                "validation_net_return": trial.get("validation_net_return"),
                "trade_count": trial.get("trade_count"),
                "traded_bars": trial.get("traded_bars"),
                "annualized_volatility": trial.get("annualized_volatility"),
                "valid_objective": trial.get("valid_objective"),
                "status": trial.get("status"),
                "regularization": trial.get("alpha"),
                "unique_subset_budget": route.get("budget"),
            })
    return rows


def _checkpoint_identity(run_contract_sha256: str, route: dict[str, Any],
                         fit_time: pd.Timestamp, prepared: PreparedInputs) -> dict[str, Any]:
    return {
        "run_contract_sha256": run_contract_sha256,
        "route": route,
        "fit_timestamp": fit_time,
        "source_freeze_reference_sha256": prepared.source_freeze_reference_sha256,
    }


def evaluate_window(prepared: PreparedInputs, run_root: Path, run_contract_sha256: str,
                    route: dict[str, Any], fit_time: pd.Timestamp, *,
                    cluster_cache: dict[pd.Timestamp, dict[str, Any]] | None = None,
                    batch_candidates: int = 1024) -> dict[str, Any]:
    """Evaluate one route update and save all candidates and its refit checkpoint."""
    root = Path(run_root)
    fit_time = pd.Timestamp(fit_time).tz_convert("UTC")
    window_dir = root / "routes" / route["route_id"] / "windows" / fit_time.strftime("%Y%m%dT%H%M%SZ")
    trials_path = window_dir / "candidate_trials.json.gz"
    fit_path = window_dir / "fit.json.gz"
    expected_identity = _checkpoint_identity(run_contract_sha256, route, fit_time, prepared)
    if trials_path.exists() or fit_path.exists():
        if not (trials_path.is_file() and fit_path.is_file()):
            raise ValueError(f"incomplete route checkpoint requires inspection: {window_dir}")
        saved_fit = _read_gzip_json(fit_path)
        saved_trials = _read_gzip_json(trials_path)
        if saved_fit.get("identity_sha256") != _hash_json(expected_identity):
            raise ValueError(f"existing fit checkpoint belongs to another frozen contract: {fit_path}")
        if saved_trials.get("identity_sha256") != _hash_json(expected_identity):
            raise ValueError(f"existing candidate audit belongs to another frozen contract: {trials_path}")
        return saved_fit

    context = build_window_context(
        prepared, fit_time, validation_hours=int(route["validation_hours"]),
        batch_candidates=batch_candidates,
    )
    needs_clusters = route["experiment"] in {"E3", "E4", "E5"}
    cluster = None
    if needs_clusters:
        if cluster_cache is None or fit_time not in cluster_cache:
            raise ValueError(f"shared training cluster/prescreen record is missing for {fit_time}")
        cluster = cluster_cache[fit_time]
    synthesis_key = {"equal_weight": "equal", "ridge": "ridge", "elastic_net": "elastic_net"}[route["synthesis"]]
    if context.readiness_status == "ready":
        search_result, evaluator, selected = _route_search(context, route, cluster)
    else:
        evaluator = context.evaluator_factory(synthesis_key, batch_candidates)
        selected = None
        search_result = {"selected": None, "stop_reason": context.readiness_status,
                         "budget": route.get("budget"), **evaluator.audit()}
    proposed = selected
    admitted, chosen_trial = search.admitted(evaluator, proposed)
    if not admitted:
        selected = None
        alpha = None
        coefficients = None
        refit_audit = None
    else:
        alpha = chosen_trial.get("alpha") if chosen_trial else None
        coefficients, refit_audit = _refit_coefficients(
            context, synthesis=route["synthesis"], subset=selected, alpha=alpha,
        )
    fit_record = {
        "identity_sha256": _hash_json(expected_identity),
        "identity": expected_identity,
        "route": route,
        "fit_timestamp": fit_time,
        "windows": context.windows.audit(HORIZON),
        "factor_names": prepared.factor_names,
        "proposed_indices": list(proposed) if proposed is not None else None,
        "proposed_factors": [prepared.factor_names[index] for index in proposed]
        if proposed is not None else [],
        "selected_indices": list(selected) if selected is not None else None,
        "selected_factors": [prepared.factor_names[index] for index in selected]
        if selected is not None else [],
        "selected_alpha": alpha,
        "coefficients": dict(zip(prepared.factor_names, coefficients.tolist()))
        if coefficients is not None else None,
        "coefficient_vector": coefficients,
        "validation_net_sharpe": chosen_trial.get("validation_net_sharpe") if chosen_trial else None,
        "validation_net_return": chosen_trial.get("validation_net_return") if chosen_trial else None,
        "validation_metrics": chosen_trial,
        "admitted": admitted,
        "status": "active" if admitted else
        f"cash_{context.readiness_status}" if context.readiness_status != "ready" else
        "cash_rejected_by_common_admission_gate",
        "initial_objective_kind": INITIAL_OBJECTIVE_KIND,
        "search": search_result,
        "evaluator_audit": evaluator.audit(),
        "refit_audit": refit_audit,
        "candidate_trial_count": sum(len(value[2]) for value in evaluator.cache.values()),
        "candidate_unique_subset_count": len(evaluator.cache),
        "budget_used": search_result.get("budget", len(evaluator.cache)),
        "training_sample_count": len(context.y_train),
        "validation_sample_periods": context.validation_label_periods,
        "readiness_status": context.readiness_status,
        "source_hashes_sha256": _hash_json(prepared.source_hashes),
    }
    candidate_payload = {
        "identity_sha256": fit_record["identity_sha256"],
        "route": route,
        "fit_timestamp": fit_time,
        "factor_names": prepared.factor_names,
        "initial_objective_kind": INITIAL_OBJECTIVE_KIND,
        "candidate_rows": _candidate_rows(
            {**route, "budget": (cluster.get("grid_unique_subsets", 0) if cluster is not None
                                  and route["experiment"] in {"E3", "E4", "E5"}
                                  else route.get("budget"))},
            fit_time, prepared.factor_names, evaluator,
        ),
    }
    _write_gzip_json(trials_path, candidate_payload, refuse_existing=True)
    _write_gzip_json(fit_path, fit_record, refuse_existing=True)
    return fit_record


def build_cluster_cache(prepared: PreparedInputs, run_root: Path,
                        run_contract_sha256: str, fit_times: list[pd.Timestamp], *,
                        batch_candidates: int = 1024) -> dict[pd.Timestamp, dict[str, Any]]:
    """Calculate each training-only clustering/prescreen window once for E3-E5."""
    root = Path(run_root)
    return {pd.Timestamp(fit_time).tz_convert("UTC"): _cluster_window(
        prepared, root, run_contract_sha256, pd.Timestamp(fit_time).tz_convert("UTC"),
        batch_candidates=batch_candidates,
    ) for fit_time in fit_times}


def _cluster_window(prepared: PreparedInputs, root: Path, run_contract_sha256: str,
                    fit_time: pd.Timestamp, *, batch_candidates: int) -> dict[str, Any]:
    path = root / "shared" / "cluster_windows" / f"{fit_time.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    identity = {
        "run_contract_sha256": run_contract_sha256,
        "fit_timestamp": fit_time,
        "source_freeze_reference_sha256": prepared.source_freeze_reference_sha256,
        "cluster_count": CLUSTER_COUNT,
        "cluster_candidates": CLUSTER_CANDIDATES,
        "training_hours": TRAINING_HOURS,
    }
    if path.exists():
        record = _read_gzip_json(path)
        if record.get("identity_sha256") != _hash_json(identity):
            raise ValueError(f"existing cluster checkpoint belongs to another frozen contract: {path}")
        return record
    context = build_window_context(prepared, fit_time, batch_candidates=batch_candidates)
    record = _cluster_record(context, batch_candidates=batch_candidates)
    record["identity"] = identity
    record["identity_sha256"] = _hash_json(identity)
    _write_gzip_json(path, record, refuse_existing=True)
    return record


def _module_hashes(project_root: Path) -> dict[str, str]:
    paths = [
        project_root / "docs/plans/多因子组合算法90天选优实验Plan_20261003.md",
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_combination_research.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_combination_search.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_selection.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_net_sharpe.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_account.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_selection_research.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_contracts.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_rebalance.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_combination_report.py",
        project_root / "src/crypto_quant/research/strategy_research/multifactor_combination_verify.py",
        project_root / "experiments/strategy_research/combination90_20261003/run_combination90.py",
        project_root / "tests/test_multifactor_combination_research.py",
        project_root / "tests/test_multifactor_combination_search.py",
        project_root / "tests/test_multifactor_combination_report.py",
        project_root / "tests/test_multifactor_combination_verify.py",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"combination90 source freeze is incomplete: {missing}")
    return {path.relative_to(project_root).as_posix(): _sha256(path) for path in paths}


def _snapshot_execution_sources(run_root: Path, project_root: Path,
                                source_hashes: dict[str, str]) -> None:
    """Copy every declared execution source byte-for-byte and verify its hash."""
    snapshot_root = Path(run_root) / "execution_source"
    for relative, digest in source_hashes.items():
        source = project_root / relative
        destination = snapshot_root / relative
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != digest:
                raise FileExistsError(f"existing execution-source snapshot differs: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256(destination) != digest:
            raise IOError(f"execution-source snapshot checksum mismatch: {destination}")
    _write_json(Path(run_root) / "execution_source_manifest.json", source_hashes,
                refuse_existing=True)


def freeze_run(prepared: PreparedInputs, run_root: Path, *,
               run_id: str = "multifactor-combination90-20261003") -> dict[str, Any]:
    """Create or verify the run freeze; an identical run may be resumed."""
    root = Path(run_root).resolve()
    project_root = Path(__file__).resolve().parents[4]
    prepared.verify_source_hashes()
    if prepared.source_root.resolve() != (project_root / "experiments/strategy_research/selection30_20261003/source_inputs").resolve():
        raise ValueError("combination90 source must be selection30_20261003/source_inputs")
    selection_rule = primary_selection_rule()
    run_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run": str(prepared.source_root),
        "source_freeze_reference_sha256": prepared.source_freeze_reference_sha256,
        "source_hashes": prepared.source_hashes,
        "source_hash_policy": "h24 execution dependencies plus frozen h24 card snapshots; no h1/h4 model inputs",
        "factor_count": len(prepared.factor_names),
        "factor_order": prepared.factor_names,
        "source_card_order": prepared.card_order,
        "factor_directions": prepared.directions,
        "factor_direction_policy": "use source standardized panel with the frozen card direction already applied",
        "horizon_hours": HORIZON,
        "rebalance": "R0",
        "portfolio": dict(prepared.dev_contract.portfolio),
        "costs": dict(prepared.dev_contract.costs),
        "selection": {
            "training_hours": TRAINING_HOURS,
            "validation_hours": VALIDATION_HOURS,
            "refit_hours": REFIT_HOURS,
            "minimum_matured_training_periods": MIN_TRAINING_LABEL_PERIODS,
            "minimum_matured_validation_periods": MIN_VALIDATION_LABEL_PERIODS,
            "update_hours": WEEK_HOURS,
            "update_anchor": prepared.timestamps[0],
            "r0_anchor": prepared.dev_contract.bounds[0] - HOUR,
            "horizon_label_gap_hours": HORIZON + 1,
            "strict_train_purge": "latest train label matures at least one hour before first validation source",
            "cluster_count": CLUSTER_COUNT,
            "cluster_metric": "average linkage over 1 - abs(time-mean hourly cross-sectional Pearson correlation)",
            "cluster_prescreen": "top two valid singleton net-Sharpe candidates per cluster; training period only",
            "alpha_grid": ALPHAS,
            "elastic_net_l1_ratio": L1_RATIO,
            "subset_size": SUBSET_SIZE,
            "ga": {"population_limit": 100, "generation_limit": 50,
                   "elite_count": 3, "tournament_size": 3,
                   "mutation_probability": 0.2, "cluster_mutation_probability": 0.8},
            "initial_objective_kind": INITIAL_OBJECTIVE_KIND,
            "admission": ADMISSION_RULE,
        },
        "routes": route_definitions(),
        "selection_rule": selection_rule,
        "stages": {
            "development": {"start": prepared.dev_contract.start, "end": prepared.dev_contract.end},
            "internal_validation": {"start": prepared.validation_contract.start,
                                    "end": prepared.validation_contract.end},
        },
        "execution_profile": EXECUTION_PROFILE,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": metadata.version("numpy"),
            "pandas": metadata.version("pandas"),
            "scipy": metadata.version("scipy"),
            "scikit_learn": metadata.version("scikit-learn"),
        },
        "forward_validation_started": False,
        "paper_started": False,
        "source_overlap_audit": prepared.overlap_audit,
        "source_manifest_hashes": _module_hashes(project_root),
        "execution_source_snapshot": "execution_source/",
    }
    contract_sha = _hash_json(run_contract)
    root.mkdir(parents=True, exist_ok=True)
    _snapshot_execution_sources(root, project_root, run_contract["source_manifest_hashes"])
    contract_path = root / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if _hash_json(existing) != contract_sha:
            raise FileExistsError(f"run root already contains a different frozen contract: {contract_path}")
    else:
        _write_json(contract_path, run_contract, refuse_existing=True)
    _write_json(root / "selection_rule.json", selection_rule, refuse_existing=True)
    source_manifest_path = root / "source_manifest.json"
    if source_manifest_path.exists():
        existing = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if existing != prepared.source_hashes:
            raise FileExistsError(f"run root source manifest differs from current frozen inputs: {source_manifest_path}")
    else:
        _write_json(source_manifest_path, prepared.source_hashes, refuse_existing=True)
    factor_path = root / "factor_manifest.json"
    factor_manifest = {
        "factor_order": prepared.factor_names,
        "source_card_order": prepared.card_order,
        "factors": [{"factor_id": factor, "direction": prepared.directions[factor],
                     "card_sha256": prepared.source_hashes[
                         f"cards/{factor}.json"]}
                    for factor in prepared.factor_names],
    }
    if factor_path.exists():
        if _hash_json(json.loads(factor_path.read_text(encoding="utf-8"))) != _hash_json(factor_manifest):
            raise FileExistsError(f"factor manifest differs from source cards: {factor_path}")
    else:
        _write_json(factor_path, factor_manifest, refuse_existing=True)

    availability_path = root / "input_audits" / "factor_availability.csv"
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    if availability_path.exists():
        existing = pd.read_csv(availability_path, index_col=0, parse_dates=True)
        if not existing.equals(prepared.factor_availability):
            raise FileExistsError(f"factor availability audit differs from source: {availability_path}")
    else:
        prepared.factor_availability.to_csv(availability_path)

    state_path = root / "machine_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_contract_sha256") != contract_sha:
            raise ValueError("machine state belongs to another run contract")
        if state.get("execution_profile") != EXECUTION_PROFILE:
            raise ValueError("machine state execution profile differs from the original run budget")
    else:
        state = {
            "schema_version": 1, "run_id": run_id,
            "run_contract_sha256": contract_sha,
            "execution_profile": dict(EXECUTION_PROFILE),
            "created_at_utc": pd.Timestamp.now(tz="UTC"),
            "statuses": {f"E{i}": {"status": "pending", "completed_fit_times": [],
                                    "stage_accounts": 0, "elapsed_seconds": 0.0,
                                    "outputs": [], "failure_reason": None,
                                    "resume_entrypoint": "run_combination90.py"}
                         for i in range(7)},
            "routes": {}, "forward_validation_started": False, "paper_started": False,
            "expected_stage_accounts": {"core_E1_E3": 44, "expanded_E1_E5": 68},
            "execution_scope": "core",
            "account_manifest_path": "account_manifest.json",
        }
        _write_json(state_path, state, refuse_existing=True)
    return {"root": str(root), "run_contract": run_contract,
            "run_contract_sha256": contract_sha, "machine_state": state}


def mark_e0_passed(run_root: Path, verification_path: Path) -> dict[str, Any]:
    """Open the historical routes only after E0 audit evidence says passed."""
    root = Path(run_root)
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    status = verification.get("status", verification.get("audit_status"))
    if status not in {"passed", "e0_passed", "verification_passed"}:
        raise ValueError(f"E0 verification has not passed: {status}")
    state_path = root / "machine_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    e0 = state["statuses"]["E0"]
    fit_times = verification.get("fit_times", {})
    if isinstance(fit_times, dict):
        fit_values = [pd.Timestamp(value).tz_convert("UTC") for value in fit_times.values()]
    else:
        fit_values = [pd.Timestamp(value).tz_convert("UTC") for value in fit_times]
    elapsed = float(verification.get("source_preparation_seconds", 0.0))
    elapsed += float(verification.get("eligible_fit_time_discovery_seconds", 0.0))
    elapsed += sum(float(window.get("timing", {}).get("total_window_seconds", 0.0))
                   for window in verification.get("windows", []))
    e0.update({
        "status": "passed", "verification_path": str(Path(verification_path).resolve()),
        "verification_sha256": _sha256(Path(verification_path)),
        "completed_fit_times": [value.isoformat() for value in sorted(fit_values)],
        "elapsed_seconds": elapsed,
        "outputs": [str(Path(verification_path).resolve().relative_to(root.resolve()))]
        if Path(verification_path).resolve().is_relative_to(root.resolve()) else
        [str(Path(verification_path).resolve())],
        "completed_at_utc": pd.Timestamp.now(tz="UTC"),
    })
    _write_json(state_path, state)
    return state["statuses"]["E0"]


def _load_cluster_checkpoints(prepared: PreparedInputs, run_root: Path,
                              fit_times: list[pd.Timestamp]) -> dict[pd.Timestamp, dict[str, Any]]:
    root = Path(run_root)
    result = {}
    for fit_time in fit_times:
        path = root / "shared" / "cluster_windows" / f"{fit_time.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(f"shared training prescreen is not checkpointed: {path}")
        result[pd.Timestamp(fit_time)] = _read_gzip_json(path)
    return result


def run_route(prepared: PreparedInputs, run_root: Path, route_id: str, *,
              fit_times: list[pd.Timestamp] | None = None,
              max_new_windows: int | None = None,
              batch_candidates: int = 1024) -> dict[str, Any]:
    """Resume one declared route, checkpointing each full weekly fit."""
    root = Path(run_root).resolve()
    state_path = root / "machine_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["statuses"]["E0"]["status"] != "passed":
        raise RuntimeError("E0 numerical and account verification must pass before historical route execution")
    run_contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[4]
    if run_contract.get("source_manifest_hashes") != _module_hashes(project_root):
        raise ValueError("execution source changed after the frozen run contract")
    route_by_id = {route["route_id"]: route for route in run_contract["routes"]}
    if route_id not in route_by_id:
        raise KeyError(f"unknown frozen route: {route_id}")
    route = route_by_id[route_id]
    run_contract_sha = _hash_json(run_contract)
    if fit_times is None:
        fit_times = common_fit_times(
            prepared, validation_hours=VALIDATION_HOURS,
            end=prepared.validation_contract.bounds[1] - HOUR,
        )
    fit_times = [pd.Timestamp(value).tz_convert("UTC") for value in fit_times]
    if route["experiment"] != "E1":
        fit_times = [fit for fit in fit_times if fit < prepared.dev_contract.bounds[1] - HOUR]
        fit_times += [fit for fit in common_fit_times(
            prepared, validation_hours=VALIDATION_HOURS,
            start=prepared.dev_contract.bounds[1] - HOUR,
            end=prepared.validation_contract.bounds[1] - HOUR,
        ) if fit not in fit_times]
        fit_times = sorted(set(fit_times))

    needs_clusters = route["experiment"] in {"E3", "E4", "E5"}
    cluster_cache = {} if needs_clusters else None
    prepared.verify_source_hashes()
    started = time.monotonic()
    launcher = project_root / "experiments/strategy_research/combination90_20261003/run_combination90.py"
    resume_entrypoint = shlex.join([
        sys.executable, str(launcher), "--action", "route",
        "--run-root", str(root), "--run-id", run_contract["run_id"],
        "--route", route_id,
    ])
    route_state = state["routes"].setdefault(route_id, {
        "status": "running", "completed_fit_times": [], "stage_accounts": 0,
        "elapsed_seconds": 0.0, "outputs": [], "failure_reason": None,
        "resume_entrypoint": resume_entrypoint,
    })
    route_state["resume_entrypoint"] = resume_entrypoint
    elapsed_before = float(route_state["elapsed_seconds"])
    experiment_state = state["statuses"][route["experiment"]]
    experiment_state["status"] = "running"
    if route["experiment"] in {"E4", "E5"}:
        state["execution_scope"] = "expanded"
    route_state["status"] = "running"
    _write_json(state_path, state)
    completed = set(route_state["completed_fit_times"])
    new_count = 0
    try:
        for fit_time in fit_times:
            fit_key = fit_time.isoformat()
            if fit_key in completed:
                continue
            if cluster_cache is not None:
                cluster_cache[fit_time] = _cluster_window(
                    prepared, root, run_contract_sha, fit_time,
                    batch_candidates=batch_candidates,
                )
            record = evaluate_window(
                prepared, root, run_contract_sha, route, fit_time,
                cluster_cache=cluster_cache, batch_candidates=batch_candidates,
            )
            route_state["completed_fit_times"].append(fit_key)
            route_state["outputs"].append(
                str(Path("routes") / route_id / "windows" / fit_time.strftime("%Y%m%dT%H%M%SZ")
                    / "fit.json.gz")
            )
            completed.add(fit_key)
            new_count += 1
            route_state["last_fit_status"] = record["status"]
            route_state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            route_state["peak_rss_bytes"] = _peak_rss_bytes()
            _write_json(state_path, state)
            if max_new_windows is not None and new_count >= max_new_windows:
                route_state["status"] = "partial"
                experiment_state["status"] = "partial"
                _write_json(state_path, state)
                return route_state
        route_state["status"] = "complete"
        experiment_state["status"] = "partial"
        route_state["finished_at_utc"] = pd.Timestamp.now(tz="UTC")
        route_state["expected_fit_count"] = len(fit_times)
        route_state["completed_fit_count"] = len(completed)
    except Exception as exc:
        route_state["status"] = "failed"
        route_state["failure_reason"] = f"{type(exc).__name__}: {exc}"
        experiment_state["status"] = "failed"
        experiment_state["failure_reason"] = route_state["failure_reason"]
        raise
    finally:
        route_state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        route_state["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(state_path, state)
    prepared.verify_source_hashes()
    return route_state


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _fit_records(root: Path, route: dict[str, Any], prepared: PreparedInputs,
                 run_contract_sha256: str) -> list[dict[str, Any]]:
    directory = Path(root) / "routes" / route["route_id"] / "windows"
    records = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*/fit.json.gz")):
        record = _read_gzip_json(path)
        if record.get("route", {}).get("route_id") != route["route_id"]:
            raise ValueError(f"fit checkpoint route id mismatch: {path}")
        fit_time = pd.Timestamp(record["fit_timestamp"])
        if path.parent.name != fit_time.strftime("%Y%m%dT%H%M%SZ"):
            raise ValueError(f"fit checkpoint timestamp differs from its path: {path}")
        identity = _checkpoint_identity(run_contract_sha256, route, fit_time, prepared)
        if record.get("identity") != research._json_safe(identity):
            raise ValueError(f"fit checkpoint frozen identity differs: {path}")
        if record.get("identity_sha256") != _hash_json(identity):
            raise ValueError(f"fit checkpoint identity checksum differs: {path}")
        records.append(record)
    timestamps = [pd.Timestamp(record["fit_timestamp"]) for record in records]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"route {route['route_id']} contains duplicate weekly fit timestamps")
    return records


def _route_score_frame(prepared: PreparedInputs, fits: list[dict[str, Any]],
                       start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    if end <= start:
        raise ValueError("outer account date range is empty")
    signal_grid = pd.date_range(start - HOUR, end - 2 * HOUR, freq="h", tz="UTC")
    positions = prepared.timestamps.get_indexer(signal_grid)
    if (positions < 0).any():
        raise ValueError("outer account source panel does not cover its full signal grid")
    factor_values = prepared.factor_cube[positions]
    complete = np.isfinite(factor_values).all(axis=2)
    score_values = np.full((len(signal_grid), len(prepared.symbols)), np.nan, dtype=float)
    fit_times = pd.DatetimeIndex([pd.Timestamp(item["fit_timestamp"]) for item in fits])
    if len(fit_times) == 0:
        raise ValueError("outer account requires at least one weekly fit checkpoint")
    fit_positions = np.searchsorted(fit_times.asi8, signal_grid.asi8, side="right") - 1
    if (fit_positions < 0).any():
        raise ValueError("outer account begins before the first completed weekly fit")
    r0_anchor = prepared.dev_contract.bounds[0] - HOUR
    if ((start - HOUR).value - r0_anchor.value) % (HORIZON * HOUR.value):
        raise ValueError("outer account start does not preserve the original R0 anchor phase")
    schedule_entries = []
    for fit_index in sorted(set(fit_positions.tolist())):
        fit = fits[int(fit_index)]
        fit_timestamp = pd.Timestamp(fit["fit_timestamp"])
        next_fit = (pd.Timestamp(fits[int(fit_index) + 1]["fit_timestamp"])
                    if int(fit_index) + 1 < len(fits) else end - HOUR)
        application_start = max(start, fit_timestamp + HOUR)
        application_end = min(end, next_fit + HOUR)
        if application_start >= application_end:
            continue
        rows = np.flatnonzero(fit_positions == fit_index)
        schedule_entries.append({
            "fit_timestamp": fit_timestamp,
            "selected_factors": fit["selected_factors"],
            "proposed_factors": fit.get("proposed_factors", []),
            "coefficients": fit["coefficients"] or {},
            "alpha": fit["selected_alpha"],
            "regularization": fit["selected_alpha"],
            "validation_net_sharpe": fit["validation_net_sharpe"],
            "validation_net_return": fit["validation_net_return"],
            "status": fit["status"],
            "train_start": fit["windows"]["training_source_start"],
            "train_end": fit["windows"]["training_source_end"],
            "selection_start": fit["windows"]["validation_source_start"],
            "selection_end": fit["windows"]["validation_source_end"],
            "validation_account_start": fit["windows"]["validation_account_start"],
            "validation_account_end": fit["windows"]["validation_account_end_exclusive"],
            "application_start": application_start,
            "application_end": application_end,
            "account_score_start": application_start,
            "account_score_end": application_end,
        })
        coeff = fit.get("coefficient_vector")
        if coeff is None or fit.get("status") != "active":
            continue
        coeff = np.asarray(coeff, dtype=float)
        if coeff.shape != (len(prepared.factor_names),) or not np.isfinite(coeff).all():
            raise ValueError("active weekly fit checkpoint has invalid coefficient vector")
        current = np.einsum("tsf,f->ts", factor_values[rows], coeff, optimize=True)
        current[~complete[rows]] = np.nan
        score_values[rows] = current
    frame = pd.DataFrame(score_values, index=signal_grid, columns=prepared.symbols)
    return frame, schedule_entries


def run_stage_account(prepared: PreparedInputs, run_root: Path, route: dict[str, Any],
                      fits: list[dict[str, Any]], stage: str, *,
                      stress: bool = False) -> dict[str, Any]:
    """Replay a frozen model schedule through the common full account engine."""
    started = time.monotonic()
    if stage == "development":
        first_common = min(pd.Timestamp(item["fit_timestamp"]) for item in fits)
        start = first_common + HOUR
        end = prepared.dev_contract.bounds[1]
    elif stage in {"C", "internal_validation"}:
        stage = "C"
        start, end = prepared.validation_contract.bounds
    else:
        raise ValueError(f"unsupported outer account stage: {stage}")
    score_frame, model_schedule = _route_score_frame(prepared, fits, start, end)
    portfolio = prepared.dev_contract.portfolio
    targets = account.generate_rank_targets(
        score_frame, long_count=portfolio["long_count"],
        short_count=portfolio["short_count"],
        gross_exposure=portfolio["gross_exposure"],
        max_asset_weight=portfolio["max_asset_weight"],
        rebalance_hours=HORIZON, start=start,
    )
    frame_index = pd.date_range(start - HOUR, end - HOUR, freq="h", tz="UTC")
    market_frames = {symbol: frame.reindex(frame_index)
                     for symbol, frame in prepared.frames.items()}
    funding = prepared.funding.loc[
        (prepared.funding["timestamp"] >= start)
        & (prepared.funding["timestamp"] < end)
    ].copy()
    costs = prepared.dev_contract.costs
    multiplier = float(costs["stress_multiplier"] if stress else 1.0)
    cost_label = "stress" if stress else "base"
    account_dir = Path(run_root) / "accounts" / route["route_id"] / stage / cost_label
    account_meta = {
        "route_id": route["route_id"], "route_family": route["route_family"],
        "experiment": route["experiment"], "method": route["method"],
        "synthesis": route["synthesis"], "seed": route["seed"],
        "selection_window_days": route["selection_window_days"],
        "primary_selection_eligible": route["primary_selection_eligible"],
        "stage": stage, "cost_path": cost_label, "cost_multiplier": multiplier,
        "account_status": "complete",
        "initial_capital": costs["initial_capital"],
        "fee_bps": float(costs["fee_bps"]) * multiplier,
        "slippage_bps": float(costs["slippage_bps"]) * multiplier,
        "margin_fraction": portfolio["margin_fraction"], "start": start, "end": end,
        "portfolio": dict(portfolio),
        "lifecycle_events": json.loads((prepared.source_root / "lifecycle.json").read_text(encoding="utf-8"))["events"],
        "r0_anchor": prepared.dev_contract.bounds[0] - HOUR,
        "r0_rebalance_hours": HORIZON, "model_schedule": model_schedule,
        "target_rows": len(targets), "signal_rows": len(score_frame),
        "accounting_engine": "multifactor_account.run_perpetual_account",
        "source_freeze_reference_sha256": prepared.source_freeze_reference_sha256,
        "source_hashes": prepared.source_hashes,
    }
    if account_dir.exists():
        existing_meta = json.loads((account_dir / "account.json").read_text(encoding="utf-8"))
        expected_meta = {**account_meta, "account_status": existing_meta.get("account_status")}
        if existing_meta.get("account_status") not in {"complete", "failed_at_account_guard"} \
                or _hash_json(existing_meta) != _hash_json(expected_meta):
            raise FileExistsError(f"existing account evidence differs from current model schedule: {account_dir}")
        if existing_meta["account_status"] == "failed_at_account_guard":
            return {"route_id": route["route_id"], "stage": stage,
                    "cost_path": cost_label, "path": str(account_dir.relative_to(Path(run_root))),
                    "metrics": None, "status": existing_meta["account_status"],
                    "elapsed_seconds": json.loads((account_dir / "runtime.json").read_text(encoding="utf-8"))["elapsed_seconds"]}
        return {"route_id": route["route_id"], "stage": stage,
                "cost_path": cost_label, "path": str(account_dir.relative_to(Path(run_root))),
                "metrics": json.loads((account_dir / "metrics.json").read_text(encoding="utf-8")),
                "elapsed_seconds": json.loads((account_dir / "runtime.json").read_text(encoding="utf-8"))["elapsed_seconds"]}
    try:
        account_result = account.run_perpetual_account(
            market_frames, targets, funding,
            initial_capital=float(costs["initial_capital"]),
            fee_bps=float(costs["fee_bps"]) * multiplier,
            slippage_bps=float(costs["slippage_bps"]) * multiplier,
            start=start, end=end,
            margin_fraction=float(portfolio["margin_fraction"]),
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("non-positive account equity at "):
            failure_kind = "non_positive_account_equity"
        elif message.startswith("sampled margin guard breached at "):
            failure_kind = "sampled_margin_guard_breach"
        else:
            raise
        account_meta["account_status"] = "failed_at_account_guard"
        account_dir.mkdir(parents=True)
        _write_json(account_dir / "account.json", account_meta, refuse_existing=True)
        _write_json(account_dir / "account_failure.json", {
            "status": "failed_at_account_guard", "kind": failure_kind,
            "reason": message, "route_id": route["route_id"],
            "stage": stage, "cost_path": cost_label, "start": start, "end": end,
        }, refuse_existing=True)
        elapsed = time.monotonic() - started
        _write_json(account_dir / "runtime.json", {
            "elapsed_seconds": elapsed, "peak_rss_bytes": _peak_rss_bytes(),
        }, refuse_existing=True)
        return {"route_id": route["route_id"], "stage": stage,
                "cost_path": cost_label, "path": str(account_dir.relative_to(Path(run_root))),
                "metrics": None, "status": "failed_at_account_guard",
                "elapsed_seconds": elapsed}
    account_dir.mkdir(parents=True)
    for name in ("orders", "fills", "funding_events", "positions", "ledger"):
        getattr(account_result, name).to_csv(account_dir / f"{name}.csv")
    _write_json(account_dir / "metrics.json", account_result.metrics, refuse_existing=True)
    _write_json(account_dir / "account.json", account_meta, refuse_existing=True)
    elapsed = time.monotonic() - started
    _write_json(account_dir / "runtime.json", {
        "elapsed_seconds": elapsed, "peak_rss_bytes": _peak_rss_bytes(),
    }, refuse_existing=True)
    return {"route_id": route["route_id"], "stage": stage,
            "cost_path": cost_label, "path": str(account_dir.relative_to(Path(run_root))),
            "metrics": account_result.metrics, "elapsed_seconds": elapsed}


def run_final_accounts(prepared: PreparedInputs, run_root: Path, route_id: str) -> list[dict[str, Any]]:
    """Write development/C, base/stress continuous accounts for a complete route."""
    root = Path(run_root)
    contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    routes = {item["route_id"]: item for item in contract["routes"]}
    route = routes[route_id]
    state = json.loads((root / "machine_state.json").read_text(encoding="utf-8"))
    if state["routes"].get(route_id, {}).get("status") != "complete":
        raise RuntimeError(f"route {route_id} is not complete")
    run_contract_sha256 = _hash_json(contract)
    fits = _fit_records(root, route, prepared, run_contract_sha256)
    expected_fits = common_fit_times(
        prepared, validation_hours=VALIDATION_HOURS,
        end=prepared.validation_contract.bounds[1] - HOUR,
    )
    actual_fits = [pd.Timestamp(fit["fit_timestamp"]) for fit in fits]
    if actual_fits != expected_fits:
        missing = [stamp for stamp in expected_fits if stamp not in set(actual_fits)]
        extra = [stamp for stamp in actual_fits if stamp not in set(expected_fits)]
        raise ValueError(f"route {route_id} weekly fit schedule differs: missing={missing[:3]}, extra={extra[:3]}")
    manifest_path = root / "account_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest_by_key = {(row["route_id"], row["stage"], row["cost_path"]): row for row in manifest}
    records = []
    elapsed_new_accounts = 0.0
    for stage in ("development", "C"):
        for stress in (False, True):
            cost_path = "stress" if stress else "base"
            key = (route_id, stage, cost_path)
            if key in manifest_by_key:
                row = manifest_by_key[key]
                account_meta = json.loads((root / row["path"] / "account.json").read_text(encoding="utf-8"))
                if (account_meta["route_id"], account_meta["stage"], account_meta["cost_path"]) != key:
                    raise ValueError(f"account manifest points to mismatched account metadata: {key}")
                records.append(row)
                continue
            row = run_stage_account(prepared, root, route, fits, stage, stress=stress)
            manifest.append(row)
            manifest_by_key[key] = row
            records.append(row)
            elapsed_new_accounts += float(row.get("elapsed_seconds", 0.0))
    _write_json(manifest_path, manifest)
    for stage in ("development", "C"):
        base = json.loads((root / "accounts" / route_id / stage / "base" / "account.json").read_text(encoding="utf-8"))
        stress = json.loads((root / "accounts" / route_id / stage / "stress" / "account.json").read_text(encoding="utf-8"))
        if _hash_json(base["model_schedule"]) != _hash_json(stress["model_schedule"]):
            raise ValueError(f"base/stress model schedules differ for {route_id}/{stage}")
    experiment_rows = [row for row in manifest if routes[row["route_id"]]["experiment"] == route["experiment"]]
    state["statuses"][route["experiment"]]["stage_accounts"] = len(experiment_rows)
    state["statuses"][route["experiment"]]["outputs"] = [row["path"] for row in experiment_rows]
    state["routes"][route_id]["stage_accounts"] = sum(row["route_id"] == route_id for row in manifest)
    state["statuses"][route["experiment"]]["elapsed_seconds"] += elapsed_new_accounts
    state["account_count"] = len(manifest)
    _write_json(root / "machine_state.json", state)
    refresh_report_and_status(root)
    return records


def refresh_report_and_status(run_root: Path) -> dict[str, Any]:
    """Reconcile saved accounts and refresh the report/status after each batch."""
    from .multifactor_combination_report import build_report, refresh_morning_report

    root = Path(run_root).resolve()
    manifest_path = root / "account_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"account manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("account manifest cannot be empty")
    started = time.monotonic()
    report = build_report(root, manifest)
    state_path = root / "machine_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    progress = report["experiment_progress"]
    for experiment in ("E1", "E2", "E3", "E4", "E5"):
        item = progress[experiment]
        row = state["statuses"][experiment]
        row["stage_accounts"] = item["received_accounts"]
        row["expected_stage_accounts"] = item["expected_accounts"]
        if item["status"] == "complete":
            row["status"] = "passed"
            row["failure_reason"] = None
        elif item["status"] == "failed":
            row["status"] = "failed"
            row["failure_reason"] = "independent account reconciliation failed"
        elif item["received_accounts"]:
            row["status"] = "partial"
        row["outputs"] = [record["path"] for record in manifest
                           if record["route_id"] in {
                               route["route_id"] for route in route_definitions()
                               if route["experiment"] == experiment
                           }]
        route_state_rows = [value for route_id, value in state["routes"].items()
                            if any(route["route_id"] == route_id
                                   and route["experiment"] == experiment
                                   for route in route_definitions())]
        route_elapsed = sum(float(route.get("elapsed_seconds", 0.0))
                            for route in route_state_rows)
        account_elapsed = sum(
            float(json.loads((root / record["path"] / "runtime.json").read_text(encoding="utf-8"))
                  .get("elapsed_seconds", 0.0))
            for record in manifest
            if record["route_id"] in {
                route["route_id"] for route in route_definitions()
                if route["experiment"] == experiment
            }
        )
        row["elapsed_seconds"] = route_elapsed + account_elapsed
        row["completed_fit_times"] = sorted({fit_time
                                             for route in route_state_rows
                                             for fit_time in route.get("completed_fit_times", [])})
        row["updated_at_utc"] = pd.Timestamp.now(tz="UTC")
    core_complete = all(progress[name]["status"] == "complete" for name in ("E1", "E2", "E3"))
    scope_complete = core_complete and (
        state.get("execution_scope") != "expanded"
        or all(progress[name]["status"] == "complete" for name in ("E4", "E5"))
    )
    e6 = state["statuses"]["E6"]
    verification = report["verification"]
    if verification["failed_accounts"]:
        e6["status"] = "failed"
        e6["failure_reason"] = f"{verification['failed_accounts']} account reconciliations failed"
    elif scope_complete and report["status"] == "complete" \
            and verification["status"] == "passed":
        e6["status"] = "passed"
        e6["failure_reason"] = None
    elif manifest:
        e6["status"] = "partial"
    e6["stage_accounts"] = verification["passed_accounts"]
    e6["expected_stage_accounts"] = state["expected_stage_accounts"][
        "expanded_E1_E5" if state.get("execution_scope") == "expanded" else "core_E1_E3"
    ]
    e6["outputs"] = report["outputs"]
    e6["elapsed_seconds"] = float(e6.get("elapsed_seconds", 0.0)) + time.monotonic() - started
    e6["updated_at_utc"] = pd.Timestamp.now(tz="UTC")
    state["account_count"] = len(manifest)
    state["verified_account_count"] = verification["passed_accounts"]
    state["failed_account_count"] = verification["failed_accounts"]
    state["last_report_status"] = report["status"]
    _write_json(state_path, state)
    refresh_morning_report(root)
    return report


def run_route_queue(prepared: PreparedInputs, run_root: Path, route_ids: list[str], *,
                    max_new_windows_per_route: int | None = None,
                    batch_candidates: int = 1024) -> list[dict[str, Any]]:
    """Run routes in the frozen priority order, sharing training-only prescreens."""
    root = Path(run_root)
    state = json.loads((root / "machine_state.json").read_text(encoding="utf-8"))
    if state["statuses"]["E0"]["status"] != "passed":
        raise RuntimeError("E0 verification is a required gate")
    contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    route_by_id = {route["route_id"]: route for route in contract["routes"]}
    unknown = [route for route in route_ids if route not in route_by_id]
    if unknown:
        raise KeyError(f"unknown route ids: {unknown}")
    all_fit_times = common_fit_times(
        prepared, validation_hours=VALIDATION_HOURS,
        end=prepared.validation_contract.bounds[1] - HOUR,
    )
    results = []
    for route_id in route_ids:
        route_fits = all_fit_times
        results.append(run_route(
            prepared, root, route_id, fit_times=route_fits,
            max_new_windows=max_new_windows_per_route,
            batch_candidates=batch_candidates,
        ))
        if results[-1]["status"] == "complete":
            run_final_accounts(prepared, root, route_id)
        else:
            break
    return results


def e0_calibration(prepared: PreparedInputs, run_root: Path, *,
                   batch_candidates: int = 1024) -> dict[str, Any]:
    """Call the canonical independent E0 verifier and timing harness."""
    from .multifactor_combination_verify import run_e0_calibration

    return run_e0_calibration(
        prepared,
        eligible_fit_times=list(calibration_fit_times(prepared).values()),
        context_factory=lambda fit_time: build_window_context(
            prepared, fit_time, batch_candidates=batch_candidates,
        ),
        output_dir=Path(run_root) / "verification" / "E0",
        input_hashes=prepared.source_hashes,
        batch_candidates=batch_candidates,
    )


def mark_experiment_status(run_root: Path, experiment: str, *, status: str,
                           failure_reason: str | None = None,
                           outputs: list[str] | None = None) -> dict[str, Any]:
    if experiment not in {f"E{i}" for i in range(7)}:
        raise ValueError(f"unknown experiment status key: {experiment}")
    if status not in {"pending", "running", "partial", "passed", "failed", "not_run"}:
        raise ValueError(f"unknown experiment status: {status}")
    path = Path(run_root) / "machine_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    row = state["statuses"][experiment]
    row["status"] = status
    row["failure_reason"] = failure_reason
    if outputs is not None:
        row["outputs"] = outputs
    row["updated_at_utc"] = pd.Timestamp.now(tz="UTC")
    _write_json(path, state)
    return row
