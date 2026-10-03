"""Read frozen multi-factor snapshots and expose compact evidence records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.data_access.market_data import resolve_market_symbols
from crypto_quant.research.factor_mining.contracts import require
from crypto_quant.research.data_policy import POLICY_ID, PRIMARY_DATA_PROCESSING
from .multifactor_contracts import MultifactorContract, ResearchContract
from .multifactor_data import (
    MarketInputs,
    ResearchDatasetManifest,
    ResearchMarketInputs,
    _daily_supplement_date,
    _stage_funding_start,
    _manifest_timestamp,
    load_universe,
)
from .multifactor_factors import FactorPanels, process_factors, read_cards


@dataclass(frozen=True)
class BaselineSnapshot:
    contract: MultifactorContract
    inputs: MarketInputs
    cards: list[dict[str, Any]]
    factors: FactorPanels
    result: dict[str, Any]
    records: dict[str, dict[str, Any]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_snapshot_json(root: Path, relative: str, label: str) -> Any:
    return _read_json(_snapshot_path(root, relative, label))


def _snapshot_path(root: Path, relative: str, label: str) -> Path:
    resolved = _root_relative_path(root, relative, label)
    require(resolved.is_file(), f"{label} snapshot is missing: {relative}")
    return resolved


def _root_relative_path(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative.strip(), f"{label} path is required")
    path = Path(relative)
    require(not path.is_absolute(), f"{label} path must be relative to the snapshot")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the snapshot root") from exc
    return resolved


def _utc_index(values: pd.Series, label: str) -> pd.DatetimeIndex:
    stamps = []
    for value in values:
        stamp = pd.Timestamp(value)
        require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0),
                f"{label} timestamps must be explicit UTC ISO8601 values")
        stamps.append(stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(stamps, name="timestamp")


def _indexed_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    require({"timestamp", "symbol"}.issubset(frame.columns),
            f"{label} CSV requires timestamp and symbol columns")
    timestamps = _utc_index(frame["timestamp"], label)
    symbols = frame["symbol"]
    require(symbols.notna().all() and symbols.map(lambda value: isinstance(value, str)).all(),
            f"{label} symbols must be strings")
    frame = frame.drop(columns=["timestamp", "symbol"]).copy()
    frame.index = pd.MultiIndex.from_arrays([timestamps, symbols.to_numpy()],
                                            names=["timestamp", "symbol"])
    require(frame.index.is_unique, f"{label} rows must be unique")
    return frame


def _read_panel(root: Path, contract: MultifactorContract, universe: pd.Series,
                diagnostics: dict[str, Any]) -> FactorInputPanel:
    values = _indexed_frame(_snapshot_path(root, "inputs/panel.values.csv", "factor input panel"),
                            "factor input panel")
    require(list(values.columns) == list(INPUT_COLUMNS),
            "factor input snapshot columns differ from the current input contract")
    require(values.index.equals(universe.index), "factor input and universe grids differ")
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors="raise").astype(float)

    start, end = contract.input_start, contract.bounds[1]
    expected_times = pd.date_range(start, end, freq="h", inclusive="left")
    expected_index = pd.MultiIndex.from_product(
        [expected_times, universe.index.get_level_values("symbol").unique()],
        names=["timestamp", "symbol"],
    )
    require(values.index.equals(expected_index), "factor input snapshot differs from the contract window")
    require(diagnostics.get("eligible_rows") == int(universe.sum()),
            "input diagnostics eligible row count differs from the saved universe")
    require(pd.Timestamp(diagnostics.get("start")).tz_convert("UTC") == start,
            "input diagnostics start differs from the contract warmup")
    require(pd.Timestamp(diagnostics.get("end")).tz_convert("UTC") == expected_times[-1],
            "input diagnostics end differs from the saved input grid")
    require(set(diagnostics.get("symbols", {})) == set(universe.index.get_level_values("symbol").unique()),
            "input diagnostics symbols differ from the saved universe")
    return FactorInputPanel(values=values, universe=universe, diagnostics=diagnostics)


def _read_market_frames(root: Path, contract: MultifactorContract,
                        symbols: list[str]) -> dict[str, pd.DataFrame]:
    start, end = contract.bounds
    expected_index = pd.date_range(start - pd.Timedelta(hours=1), end, freq="h", inclusive="left",
                                   name="timestamp")
    frames = {}
    for symbol in symbols:
        path = _snapshot_path(root, f"inputs/market/{symbol}.csv", f"market input {symbol}")
        frame = pd.read_csv(path, float_precision="round_trip")
        require(list(frame.columns) == ["Unnamed: 0", "open", "close", "mark_close"],
                f"market input columns differ from the frozen CSV schema: {symbol}")
        frame.index = _utc_index(frame.pop("Unnamed: 0"), f"market input {symbol}")
        require(frame.index.is_unique and frame.index.equals(expected_index),
                f"market input grid differs from the account window: {symbol}")
        require(set(frame.columns) == {"open", "close", "mark_close"},
                f"market input fields differ from the account contract: {symbol}")
        require(np.isfinite(frame.to_numpy(dtype=float)).all(), f"market inputs contain missing values: {symbol}")
        frames[symbol] = frame[["open", "close", "mark_close"]]
    return frames


def _read_funding(root: Path, contract: MultifactorContract, symbols: list[str]) -> pd.DataFrame:
    path = _snapshot_path(root, "inputs/funding.csv", "funding input")
    funding = pd.read_csv(path, float_precision="round_trip")
    require(list(funding.columns) == ["timestamp", "funding_rate", "mark_price", "symbol"],
            "funding snapshot columns differ from the account contract")
    funding["timestamp"] = _utc_index(funding["timestamp"], "funding snapshot")
    require(funding["symbol"].isin(symbols).all(), "funding snapshot has symbols outside the universe")
    require(not funding.duplicated(["timestamp", "symbol"]).any(),
            "funding snapshot contains duplicate symbol timestamps")
    start, end = contract.bounds
    require(((funding["timestamp"] >= start) & (funding["timestamp"] < end)).all(),
            "funding snapshot falls outside the account window")
    values = funding[["funding_rate", "mark_price"]].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values[:, 1] > 0).all(),
            "funding snapshot contains invalid values")
    return funding.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _verify_card_manifest(root: Path, contract: MultifactorContract,
                          manifest: Any, result: dict[str, Any]) -> list[dict[str, Any]]:
    require(isinstance(manifest, list) and len(manifest) == len(contract.cards),
            "card manifest differs from contract card count")
    result_factors = result.get("factors")
    require(isinstance(result_factors, list) and len(result_factors) == len(manifest),
            "result card list differs from manifest")
    contract_ids = [Path(path).stem for path in contract.cards]
    ids = [entry.get("id") if isinstance(entry, dict) else None for entry in manifest]
    require(ids == contract_ids, "contract card IDs differ from the frozen manifest")
    require([item.get("id") for item in result_factors] == ids,
            "result card IDs differ from the frozen manifest")

    paths = []
    for position, (entry, result_factor) in enumerate(zip(manifest, result_factors), start=1):
        require(isinstance(entry, dict), "card manifest entry must be an object")
        for field in ("id", "title", "source_path", "snapshot_path", "sha256", "direction",
                      "horizon_hours", "lookback_hours", "expression"):
            require(field in entry, f"card manifest entry is missing {field}")
        require(isinstance(entry["source_path"], str), "card source_path must be text")
        snapshot = _snapshot_path(root, entry["snapshot_path"], "card")
        payload = snapshot.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == entry["sha256"],
                f"card snapshot sha256 differs from manifest: {entry['id']}")
        card = json.loads(payload.decode("utf-8"))
        require(card.get("id") == entry["id"], "card snapshot ID differs from manifest")
        require(result_factor.get("code") == f"F{position}"
                and result_factor.get("title") == entry["title"]
                and result_factor.get("snapshot_path") == entry["snapshot_path"],
                f"result factor metadata differs from manifest: {entry['id']}")
        paths.append(snapshot)

    cards = read_cards(paths, contract.horizon_hours, contract.warmup_hours)
    for card, entry in zip(cards, manifest):
        require(card["id"] == entry["id"] and card["title"] == entry["title"],
                "validated factor card differs from manifest identity")
        for field in ("direction", "horizon_hours", "lookback_hours", "expression"):
            require(card[field] == entry[field], f"card {field} differs from manifest: {entry['id']}")
    return cards


def _read_standardized(root: Path, expected: pd.DataFrame) -> None:
    path = _snapshot_path(root, "factor_panels/standardized.csv", "saved standardized factor panel")
    saved = _indexed_frame(path, "saved standardized factor panel")
    require(list(saved.columns) == list(expected.columns),
            "saved standardized factor IDs differ from the card manifest")
    actual = saved.astype(float)
    require(expected.index.equals(actual.index), "saved standardized factor index differs from the input grid")
    left, right = expected.to_numpy(dtype=float), actual.to_numpy(dtype=float)
    same_missing = np.isnan(left) == np.isnan(right)
    close = np.isclose(left, right, rtol=1e-14, atol=1e-14, equal_nan=True)
    require(bool(same_missing.all() and close.all()),
            "recomputed standardized factors differ from the frozen snapshot")


def _read_common_mask(root: Path, expected: pd.Series) -> None:
    path = _snapshot_path(root, "factor_panels/common_mask.csv", "saved common mask")
    frame = pd.read_csv(path)
    require(list(frame.columns) == ["timestamp", "symbol", "common_mask"],
            "saved common mask columns differ from the factor snapshot contract")
    index = pd.MultiIndex.from_arrays(
        [_utc_index(frame["timestamp"], "saved common mask"), frame["symbol"].to_numpy()],
        names=["timestamp", "symbol"],
    )
    require(frame["common_mask"].dtype == bool, "saved common mask must contain explicit booleans")
    saved = pd.Series(frame["common_mask"].to_numpy(), index=index, name=expected.name)
    try:
        pd.testing.assert_series_equal(expected, saved, check_exact=True, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("recomputed common mask differs from the frozen snapshot") from exc


def _read_coverage(root: Path, contract: MultifactorContract, factors: FactorPanels,
                   factor_codes: dict[str, str], sample: dict[str, Any]) -> dict[str, Any]:
    path = _snapshot_path(root, "coverage_impact.csv", "coverage impact")
    table = pd.read_csv(path, float_precision="round_trip")
    start, end = contract.bounds
    signal_start, signal_end = start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1)
    times = factors.eligible.index.get_level_values("timestamp")
    account_period = (times >= signal_start) & (times < signal_end)
    account_eligible = factors.eligible & account_period
    account_common = factors.common_mask & account_period
    eligible_count = int(account_eligible.sum())
    common_count = int(account_common.sum())
    require(eligible_count > 0, "account signal window has no eligible rows")
    require(sample.get("factor_panel_eligible_rows_including_warmup") == int(factors.eligible.sum())
            and sample.get("factor_panel_common_rows_including_warmup") == int(factors.common_mask.sum()),
            "result factor-panel coverage differs from recomputed factors")

    expected = factors.coverage.copy()
    expected["panel_shared_common_rows_including_warmup"] = int(factors.common_mask.sum())
    expected["panel_shared_coverage_ratio_including_warmup"] = (
        int(factors.common_mask.sum()) / int(factors.eligible.sum())
    )
    expected["account_eligible_rows"] = eligible_count
    expected["account_raw_valid_rows"] = factors.valid_masks.loc[account_eligible].sum(axis=0).reindex(expected.index)
    expected["account_raw_coverage_ratio"] = expected["account_raw_valid_rows"] / eligible_count
    expected["account_shared_common_rows"] = common_count
    expected["account_shared_coverage_ratio"] = common_count / eligible_count
    expected["account_rows_excluded_by_shared_mask"] = (
        expected["account_raw_valid_rows"] - common_count
    )
    expected = expected.reset_index()
    require(list(table.columns) == list(expected.columns),
            "coverage CSV columns differ from recomputed factor and account-window coverage")
    require(table["factor_id"].tolist() == expected["factor_id"].tolist()
            and set(table["factor_id"]) == set(factor_codes),
            "coverage factor IDs/order differ from the card manifest")
    integer_columns = [
        "eligible_rows", "valid_rows", "missing_rows", "panel_shared_common_rows_including_warmup",
        "account_eligible_rows", "account_raw_valid_rows", "account_shared_common_rows",
        "account_rows_excluded_by_shared_mask",
    ]
    float_columns = ["coverage_ratio", "panel_shared_coverage_ratio_including_warmup",
                     "account_raw_coverage_ratio", "account_shared_coverage_ratio"]
    for column in integer_columns:
        actual = table[column].to_numpy()
        recomputed = expected[column].to_numpy()
        require(np.issubdtype(actual.dtype, np.integer) and np.array_equal(actual, recomputed),
                f"coverage count differs from recomputed values: {column}")
    for column in float_columns:
        require(np.isclose(table[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float),
                           rtol=1e-14, atol=1e-14, equal_nan=True).all(),
                f"coverage ratio differs from recomputed values: {column}")

    per_factor = {}
    record_fields = ["account_eligible_rows", "account_raw_valid_rows", "account_raw_coverage_ratio",
                     "account_shared_common_rows", "account_shared_coverage_ratio",
                     "account_rows_excluded_by_shared_mask"]
    for row in expected.to_dict(orient="records"):
        per_factor[factor_codes[row["factor_id"]]] = {
            field: _evidence_scalar(row[field]) for field in record_fields
        }
    summary_keys = ["account_signal_window_start", "account_signal_window_end_exclusive",
                    "account_signal_window_eligible_rows", "account_signal_window_common_rows",
                    "account_shared_coverage_ratio"]
    require(all(key in sample for key in summary_keys), "result lacks account signal-window coverage")
    expected_summary = {
        "account_signal_window_start": signal_start.isoformat(),
        "account_signal_window_end_exclusive": signal_end.isoformat(),
        "account_signal_window_eligible_rows": eligible_count,
        "account_signal_window_common_rows": common_count,
        "account_shared_coverage_ratio": common_count / eligible_count,
    }
    for key in summary_keys:
        if key.endswith("ratio"):
            require(np.isclose(float(sample[key]), float(expected_summary[key]), rtol=1e-14, atol=1e-14),
                    f"result account coverage differs from recomputed values: {key}")
        elif key.endswith("start") or key.endswith("end_exclusive"):
            require(pd.Timestamp(sample[key]).tz_convert("UTC") == pd.Timestamp(expected_summary[key]),
                    f"result account coverage window differs from contract: {key}")
        else:
            require(sample[key] == expected_summary[key],
                    f"result account coverage differs from recomputed values: {key}")
    return {"scope": "account_signal_window", "summary": {key: sample[key] for key in summary_keys},
            "per_factor": per_factor}


def _read_correlations(root: Path, factors: FactorPanels,
                       factor_codes: dict[str, str]) -> dict[str, Any]:
    path = _snapshot_path(root, "factor_panels/correlation_summary.csv", "correlation summary")
    table = pd.read_csv(path, float_precision="round_trip")
    expected = ["factor_a", "factor_b", "valid_periods", "mean", "median", "mean_abs",
                "p10", "p90", "positive_share"]
    require(list(table.columns) == expected, "correlation summary columns differ from current schema")
    recomputed = factors.correlation_summary.reset_index()
    require(table["factor_a"].tolist() == recomputed["factor_a"].tolist()
            and table["factor_b"].tolist() == recomputed["factor_b"].tolist()
            and table["factor_a"].isin(factor_codes).all()
            and table["factor_b"].isin(factor_codes).all(),
            "correlation pair IDs/order differ from recomputed factor panels")
    require(np.array_equal(table["valid_periods"].to_numpy(dtype=np.int64),
                           recomputed["valid_periods"].to_numpy(dtype=np.int64)),
            "correlation valid periods differ from recomputed factor panels")
    numeric = ["mean", "median", "mean_abs", "p10", "p90", "positive_share"]
    require(np.isclose(table[numeric].to_numpy(dtype=float), recomputed[numeric].to_numpy(dtype=float),
                       rtol=1e-14, atol=1e-14, equal_nan=True).all(),
            "correlation summary differs from recomputed factor panels")
    return {
        "method": "cross-sectional Spearman",
        "scope": "full input panel, including prewarm rows",
        "pairs": [
            {
                "factor_a": factor_codes[row["factor_a"]],
                "factor_b": factor_codes[row["factor_b"]],
                **{key: _evidence_scalar(row[key]) for key in expected[2:]},
            }
            for row in recomputed.to_dict(orient="records")
        ],
    }


def _evidence_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _require_close(actual: Any, expected: Any, label: str) -> None:
    require(np.isclose(float(actual), float(expected), rtol=1e-14, atol=1e-14, equal_nan=False),
            f"{label} differs from recomputed values")


def _account_window(contract: MultifactorContract, factors: FactorPanels) -> tuple[dict[str, Any], pd.DatetimeIndex]:
    start, end = contract.bounds
    signal_start, signal_end = start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1)
    expected_signals = pd.date_range(signal_start, signal_end, freq="h", inclusive="left")
    timestamps = factors.eligible.index.get_level_values("timestamp")
    account_period = (timestamps >= signal_start) & (timestamps < signal_end)
    eligible = factors.eligible & account_period
    common = factors.common_mask & account_period
    require(int(eligible.sum()) > 0 and int(common.sum()) > 0,
            "account signal window has no valid common observations")
    return ({
        "factor_panel_eligible_rows_including_warmup": int(factors.eligible.sum()),
        "factor_panel_common_rows_including_warmup": int(factors.common_mask.sum()),
        "account_signal_window_eligible_rows": int(eligible.sum()),
        "account_signal_window_common_rows": int(common.sum()),
        "account_signal_window_start": signal_start.isoformat(),
        "account_signal_window_end_exclusive": signal_end.isoformat(),
    }, expected_signals)


def _verify_sample(sample: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        require(key in sample, f"{label} sample is missing {key}")
        if key.endswith("start") or key.endswith("end_exclusive"):
            require(pd.Timestamp(sample[key]).tz_convert("UTC") == pd.Timestamp(value),
                    f"{label} sample window differs from the frozen contract")
        elif key.endswith("ratio"):
            _require_close(sample[key], value, f"{label} sample {key}")
        else:
            require(sample[key] == value, f"{label} sample {key} differs from recomputed values")


def _read_signals(path: Path, symbols: list[str], expected_index: pd.DatetimeIndex,
                  label: str) -> tuple[int, int]:
    frame = pd.read_csv(path, float_precision="round_trip")
    require(list(frame.columns) == ["timestamp", *symbols], f"{label} signal columns differ from the universe")
    timestamps = _utc_index(frame["timestamp"], f"{label} signals")
    require(timestamps.equals(expected_index), f"{label} signal window differs from the contract")
    values = frame[symbols].apply(pd.to_numeric, errors="raise")
    require(not np.isinf(values.to_numpy(dtype=float)).any(), f"{label} signals contain infinite values")
    return int(values.notna().any(axis=1).sum()), len(values)


def _artifact_directory(root: Path, experiment: dict[str, Any], label: str) -> tuple[Path, dict[str, Path]]:
    relative_dir = experiment.get("path")
    require(isinstance(relative_dir, str) and relative_dir.strip(), f"{label} experiment path is required")
    path = Path(relative_dir)
    require(not path.is_absolute() and path.parts and path.parts[0] == "experiments"
            and ".." not in path.parts, f"{label} path is not a baseline experiment path")
    directory = _root_relative_path(root, relative_dir, f"{label} experiment")
    require(directory.is_dir(), f"{label} experiment directory is missing")
    names = ("signals", "targets", "orders", "fills", "funding_events", "positions", "ledger", "metrics")
    artifact_names = experiment.get("artifacts")
    require(isinstance(artifact_names, dict) and set(artifact_names) == set(names),
            f"{label} artifact list differs from the baseline schema")
    paths = {}
    for name in names:
        suffix = "json" if name == "metrics" else "csv"
        expected_relative = f"{relative_dir}/{name}.{suffix}"
        require(artifact_names[name] == expected_relative,
                f"{label} {name} path differs from its experiment directory")
        paths[name] = _snapshot_path(root, artifact_names[name], f"{label} {name}")
        require(paths[name].parent == directory, f"{label} {name} artifact escapes its experiment directory")
    return directory, paths


def _verify_ledger_metrics(contract: MultifactorContract, metrics_path: Path,
                           ledger_path: Path, fills_path: Path, funding_path: Path, label: str,
                           result_metrics: dict[str, Any]) -> None:
    saved_metrics = _read_json(metrics_path)
    require(saved_metrics == result_metrics, f"{label} result metrics differ from metrics.json")
    ledger = pd.read_csv(ledger_path, float_precision="round_trip")
    require("timestamp" in ledger.columns, f"{label} ledger lacks timestamps")
    start, end = contract.bounds
    expected_index = pd.date_range(start + pd.Timedelta(hours=1), end, freq="h")
    timestamps = _utc_index(ledger["timestamp"], f"{label} ledger")
    require(timestamps.equals(expected_index), f"{label} ledger timestamps differ from the account window")
    required = {"cash", "realized_pnl", "unrealized_pnl", "equity", "fees",
                "slippage_cost", "funding_cashflow", "turnover"}
    require(required.issubset(ledger.columns), f"{label} ledger lacks account totals")
    initial = contract.costs["initial_capital"]
    accounting_equity = ledger["cash"].to_numpy(dtype=float) + ledger["unrealized_pnl"].to_numpy(dtype=float)
    require(np.isclose(ledger["equity"].to_numpy(dtype=float), accounting_equity,
                       rtol=1e-14, atol=1e-10, equal_nan=False).all(),
            f"{label} ledger equity differs from cash and unrealized PnL")
    accounting_cash = initial + (
        ledger["realized_pnl"] - ledger["fees"] + ledger["funding_cashflow"]
    ).cumsum().to_numpy(dtype=float)
    require(np.isclose(ledger["cash"].to_numpy(dtype=float), accounting_cash,
                       rtol=1e-14, atol=1e-10, equal_nan=False).all(),
            f"{label} ledger cash differs from realized PnL, fees, and funding")
    equity = np.concatenate(([initial], ledger["equity"].to_numpy(dtype=float)))
    require(np.isfinite(equity).all() and (equity > 0).all(), f"{label} ledger equity is invalid")
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    elapsed_hours = (contract.bounds[1] - contract.bounds[0]) / pd.Timedelta(hours=1)
    returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)
    annualized_volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(365.0 * 24.0)) if len(returns) > 1 else 0.0
    )
    mean_annual_return = float(np.mean(returns) * 365.0 * 24.0) if len(returns) else 0.0
    fills = pd.read_csv(fills_path, float_precision="round_trip")
    funding_events = pd.read_csv(funding_path, float_precision="round_trip")
    require({"fee", "slippage_cost"}.issubset(fills.columns), f"{label} fills lack cost fields")
    require("cashflow" in funding_events.columns, f"{label} funding events lack cashflow")
    final_equity = float(equity[-1])
    totals = {
        "net_return": final_equity / initial - 1.0,
        "annualized_return": (final_equity / initial) ** (365.0 * 24.0 / elapsed_hours) - 1.0,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": mean_annual_return / annualized_volatility if annualized_volatility > 0 else 0.0,
        "max_drawdown": float(-np.min(drawdown)),
        "total_fees": float(ledger["fees"].sum()),
        "total_slippage_cost": float(ledger["slippage_cost"].sum()),
        "total_funding": float(ledger["funding_cashflow"].sum()),
        "total_turnover": float(ledger["turnover"].sum()),
        "final_equity": final_equity,
    }
    require(set(result_metrics) == set(totals), f"{label} metric fields differ from the account schema")
    _require_close(totals["total_fees"], fills["fee"].sum(), f"{label} ledger fees vs fills")
    _require_close(totals["total_slippage_cost"], fills["slippage_cost"].sum(),
                   f"{label} ledger slippage vs fills")
    _require_close(totals["total_funding"], funding_events["cashflow"].sum(),
                   f"{label} ledger funding vs funding events")
    for name, value in totals.items():
        _require_close(result_metrics[name], value, f"{label} {name} vs ledger")


def _verify_experiments(root: Path, contract: MultifactorContract, result: dict[str, Any],
                        cards: list[dict[str, Any]], factors: FactorPanels,
                        symbols: list[str]) -> None:
    card_ids = [card["id"] for card in cards]
    factor_codes = [f"F{position}" for position in range(1, len(cards) + 1)]
    expected_names = ([f"single:{code}" for code in factor_codes] + ["equal_weight"]
                      + [f"drop:{code}" for code in factor_codes])
    experiments = result.get("experiments")
    require(isinstance(experiments, list)
            and [item.get("name") for item in experiments] == expected_names,
            "result comparison set differs from the frozen factor count")

    expected_sample, expected_signal_index = _account_window(contract, factors)
    coverage = result.get("coverage_impact")
    require(isinstance(coverage, dict), "result coverage summary is missing")
    require(coverage.get("path") == "coverage_impact.csv",
            "result coverage artifact path differs from baseline root")
    _verify_sample(coverage, expected_sample, "result coverage")

    expected_by_name = {}
    for position, code in enumerate(factor_codes):
        expected_by_name[f"single:{code}"] = ("single_factor", [code], [card_ids[position]])
    expected_by_name["equal_weight"] = ("equal_weight", factor_codes, card_ids)
    for dropped_code in factor_codes:
        kept = [(code, card_id) for code, card_id in zip(factor_codes, card_ids) if code != dropped_code]
        expected_by_name[f"drop:{dropped_code}"] = (
            "leave_one_out", [code for code, _ in kept], [card_id for _, card_id in kept],
        )

    for experiment in experiments:
        name = experiment["name"]
        kind, included_codes, included_ids = expected_by_name[name]
        require(experiment.get("kind") == kind
                and experiment.get("included_factors") == included_codes
                and experiment.get("included_card_ids") == included_ids,
                f"{name} experiment factor membership differs from the frozen cards")
        _, artifacts = _artifact_directory(root, experiment, name)
        signal_rows, signal_hours = _read_signals(artifacts["signals"], symbols,
                                                  expected_signal_index, name)
        sample = dict(expected_sample)
        sample["account_signal_rows"] = signal_rows
        sample["account_signal_grid_hours"] = signal_hours
        _verify_sample(experiment.get("sample", {}), sample, name)
        metrics = experiment.get("metrics")
        require(isinstance(metrics, dict), f"{name} metrics are missing")
        _verify_ledger_metrics(contract, artifacts["metrics"], artifacts["ledger"],
                               artifacts["fills"], artifacts["funding_events"], name, metrics)

    by_name = {experiment["name"]: experiment for experiment in experiments}
    baseline = by_name["equal_weight"]["metrics"]
    comparison_metrics = ("net_return", "max_drawdown", "sharpe_ratio", "total_turnover",
                          "total_fees", "total_slippage_cost", "total_funding")
    for name, experiment in by_name.items():
        metrics = experiment["metrics"]
        require("net_return_delta_vs_equal_weight" in experiment
                and "delta_vs_equal_weight" in experiment,
                f"{name} equal-weight comparison metrics are missing")
        expected_delta = {metric: metrics[metric] - baseline[metric] for metric in comparison_metrics}
        _require_close(experiment["net_return_delta_vs_equal_weight"], expected_delta["net_return"],
                       f"{name} net return delta")
        require(set(experiment["delta_vs_equal_weight"]) == set(expected_delta),
                f"{name} equal-weight metric delta fields differ")
        for metric, value in expected_delta.items():
            _require_close(experiment["delta_vs_equal_weight"][metric], value,
                           f"{name} {metric} delta")


def _record(record_id: str, kind: str, run_id: str, source: Path,
            data: dict[str, Any], scope: str) -> dict[str, Any]:
    return {"id": record_id, "kind": kind, "run_id": run_id,
            "source": str(source), "scope": scope, "data": data}


def _build_records(root: Path, contract: MultifactorContract, inputs: MarketInputs,
                   cards: list[dict[str, Any]], factors: FactorPanels,
                   result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    run_id = contract.run_id
    records = {}
    result_path = root / "result.json"
    contract_path = root / "contract.json"
    provenance = inputs.diagnostics
    data_usage = {"purpose": contract.purpose, "stage": contract.stage,
                  "prior_data_use": contract.prior_data_use, "data_processing": contract.data_processing,
                  "source_causality": provenance["source_causality"],
                  "historical_causality_certified": provenance["historical_causality_certified"],
                  "independence": provenance["independence"],
                  "forward_validation_started": result["forward_validation_started"]}
    if isinstance(contract, ResearchContract):
        data_usage.update({
            "dataset_id": provenance["dataset_id"],
            "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "source_causality_notes": provenance["source_causality_notes"],
            "row_repair_mask": provenance["row_repair_mask"],
            "repair_state": provenance["repair_state"],
            "funding_mark": provenance["funding_mark"],
            "partial_spot_source_hours_masked": provenance["partial_spot_source_hours_masked"],
        })
        if isinstance(inputs, ResearchMarketInputs) and inputs.dataset_manifest["schema_version"] == 2:
            data_usage.update({
                "daily_mark_supplement_file_count": provenance["daily_mark_supplement_file_count"],
                "daily_hourly_mark_rows_restored": provenance["daily_hourly_mark_rows_restored"],
                "daily_minute_rows_for_funding_proxies": provenance[
                    "daily_minute_rows_for_funding_proxies"],
            })
    else:
        data_usage.update({
            "policy_id": provenance["policy_id"],
            "row_repair_mask": provenance["row_repair_mask"],
        })
    records["data_usage"] = _record(
        "data_usage", "data_usage", run_id, root / "inputs/data-provenance.json",
        data_usage,
        "snapshot provenance and historical data-use boundary",
    )
    records["portfolio_rules"] = _record(
        "portfolio_rules", "portfolio_rules", run_id, contract_path,
        {"start": contract.start, "end_exclusive": contract.end,
         "warmup_hours": contract.warmup_hours, "horizon_hours": contract.horizon_hours,
         "universe": contract.universe, "costs": contract.costs, "portfolio": contract.portfolio},
        "frozen contract",
    )

    manifest = _read_snapshot_json(root, "cards.json", "card manifest")
    factor_codes = {card["id"]: f"F{position}" for position, card in enumerate(cards, start=1)}
    card_entries = {entry["id"]: entry for entry in manifest}
    for card in cards:
        code = factor_codes[card["id"]]
        source = _snapshot_path(root, card_entries[card["id"]]["snapshot_path"], "card")
        snapshot = card["card_snapshot"]
        formula = snapshot["original_claim"]["formula"]
        records[f"card:{code}"] = _record(
            f"card:{code}", "factor_card", run_id, source,
            {"factor_code": code, "card_id": card["id"], "title": card["title"],
             "formula": {"expression": formula["expression"],
                         "expanded_expression": formula["expanded_expression"],
                         "fields": formula["fields"]},
             "direction": card["direction"], "horizon_hours": card["horizon_hours"],
             "lookback_hours": card["lookback_hours"],
             "meaning": snapshot["original_claim"].get("meaning"),
             "hypothesis": snapshot["original_claim"].get("hypothesis"),
             "economic_mechanism": snapshot.get("economic_mechanism"),
             "unverified_assumptions": snapshot.get("unverified_assumptions"),
             "falsification_conditions": snapshot.get("falsification_conditions"),
             "market_and_horizon": snapshot.get("market_and_horizon"),
             "strategy_validation_status": snapshot.get("strategy_validation_status")},
            "frozen FM-v6 card snapshot",
        )

    sample = result["coverage_impact"]
    coverage = _read_coverage(root, contract, factors, factor_codes, sample)
    records["coverage"] = _record("coverage", "coverage", run_id, root / "coverage_impact.csv",
                                  coverage, "account signal window only")
    correlations = _read_correlations(root, factors, factor_codes)
    records["correlations"] = _record("correlations", "correlations", run_id,
                                      root / "factor_panels/correlation_summary.csv",
                                      correlations, "full input panel including prewarm rows")

    experiments = result["experiments"]
    by_name = {item["name"]: item for item in experiments}
    equal_weight = by_name["equal_weight"]
    records["baseline_metrics"] = _record(
        "baseline_metrics", "baseline_metrics", run_id, result_path,
        {"strategy_id": "strategy:equal_weight", "metrics": equal_weight["metrics"],
         "included_factors": equal_weight["included_factors"],
         "sample": _account_sample(equal_weight["sample"])},
        "fixed-rule equal-weight baseline",
    )
    for name, experiment in by_name.items():
        record_id = "strategy:equal_weight" if name == "equal_weight" else name
        included_codes = experiment["included_factors"]
        record_data = {
            "strategy_id": record_id,
            "kind": experiment["kind"],
            "included_factors": included_codes,
            "included_card_ids": experiment["included_card_ids"],
            "sample": _account_sample(experiment["sample"]),
            "metrics": experiment["metrics"],
            "net_return_delta_vs_equal_weight": experiment["net_return_delta_vs_equal_weight"],
            "delta_vs_equal_weight": experiment["delta_vs_equal_weight"],
        }
        records[record_id] = _record(record_id, "strategy_comparison", run_id,
                                     result_path, record_data, "same account window and common sample")
    return records


def _account_sample(sample: dict[str, Any]) -> dict[str, Any]:
    keys = ["account_signal_window_start", "account_signal_window_end_exclusive",
            "account_signal_rows", "account_signal_grid_hours"]
    require(all(key in sample for key in keys), "strategy result lacks account-window sample metadata")
    return {key: sample[key] for key in keys}


def _research_snapshot_inputs(root: Path, contract: ResearchContract,
                              panel: FactorInputPanel, universe: pd.Series,
                              frames: dict[str, pd.DataFrame], funding: pd.DataFrame,
                              provenance: dict[str, Any]) -> ResearchMarketInputs:
    manifest = _read_snapshot_json(root, "inputs/dataset-manifest.json", "frozen research dataset manifest")
    dataset = ResearchDatasetManifest.from_dict(manifest)
    dataset.validate_contract(contract)
    document = dataset.document
    require(provenance.get("dataset_id") == document["dataset_id"]
            and _valid_sha256(provenance.get("dataset_manifest_sha256"))
            and provenance.get("source_causality") == document["source_causality"]
            and provenance.get("source_causality_notes") == document["source_causality_notes"]
            and provenance.get("historical_causality_certified") is False
            and provenance.get("repair_state") == document["repair_state"]
            and provenance.get("funding_mark") == document["funding_mark"],
            "research snapshot data provenance differs from the frozen dataset manifest")
    require(provenance.get("row_repair_mask") ==
            "market_row_provenance maps source rows; no price/rate interpolation or primary database fallback"
            and provenance.get("causality_scope") ==
            "raw archive close times and row provenance; funding event marks use the declared prior-minute close proxy; publication/receipt timestamps are unavailable",
            "research snapshot overstates row repair or causal evidence")
    if document["schema_version"] == 2:
        supplements = document["supplement_source_files"]
        require(provenance.get("daily_mark_supplement_file_count") == len(supplements)
                and provenance.get("daily_hourly_mark_rows_restored") == sum(
                    entry["selected_row_count"] for entry in supplements if entry["interval"] == "1h"
                )
                and provenance.get("daily_minute_rows_for_funding_proxies") == sum(
                    entry["selected_row_count"] for entry in supplements if entry["interval"] == "1m"
                ), "research snapshot daily-source counts differ from the frozen manifest")

    summary = _read_snapshot_json(root, "inputs/price-row-provenance-summary.json",
                                  "frozen price row provenance summary")
    require(isinstance(summary, dict) and set(summary) == {"prices", "funding_rate_event_rows"}
            and isinstance(summary["prices"], list)
            and type(summary["funding_rate_event_rows"]) is int
            and summary["funding_rate_event_rows"] > 0,
            "research price row provenance summary has an invalid schema")
    expected_price_rows = []
    start, end = contract.input_start, contract.bounds[1]
    restored_in_window = {symbol: 0 for symbol in document["symbols"]}
    if document["schema_version"] == 2:
        for source in document["supplement_source_files"]:
            if source["interval"] != "1h":
                continue
            day = pd.Timestamp(_daily_supplement_date(source["source_key"], source["symbol"], "1h"), tz="UTC")
            clipped_start, clipped_end = max(start, day), min(end, day + pd.Timedelta(days=1))
            if clipped_start < clipped_end:
                restored_in_window[source["symbol"]] += int(
                    (clipped_end - clipped_start) / pd.Timedelta(hours=1)
                )
    for symbol in document["symbols"]:
        for data_name, coverage_name in (("spot_trade", "spot_trade"),
                                         ("perpetual_trade", "perpetual_trade"),
                                         ("mark_price", "mark_price")):
            coverage = next(row for row in document["coverage"]
                            if row["symbol"] == symbol and row["dataset"] == coverage_name)
            complete_start = _manifest_timestamp(document["window"]["input_start"], "dataset input_start")
            stage_end = _manifest_timestamp(document["window"]["end_exclusive"], "dataset end")
            coverage_index = pd.date_range(complete_start, stage_end, freq="h", inclusive="left")
            stage_index = coverage_index[(coverage_index >= start) & (coverage_index < end)]
            hole_hours = set()
            for hole in coverage["holes"]:
                hole_start = _manifest_timestamp(hole["start"], "coverage hole start")
                hole_end = _manifest_timestamp(hole["end_exclusive"], "coverage hole end")
                hole_hours.update(pd.date_range(hole_start, hole_end, freq="h", inclusive="left").to_list())
            partial_hours = {
                _manifest_timestamp(row["open_time"], "partial spot open_time")
                for row in coverage["partial_bars"]
            }
            observed = len(set(stage_index) - hole_hours) + len(set(stage_index) & partial_hours)
            summary_item = next((item for item in summary["prices"]
                                 if item.get("symbol") == symbol and item.get("dataset") == data_name), None)
            expected_summary_fields = {"symbol", "dataset", "selected_rows", "derived_rows", "provenance_rows"}
            if document["schema_version"] == 2:
                expected_summary_fields.add("restored_rows")
            require(isinstance(summary_item, dict)
                    and set(summary_item) == expected_summary_fields
                    and all(type(summary_item[field]) is int and summary_item[field] >= 0
                            for field in expected_summary_fields - {"symbol", "dataset"})
                    and summary_item["provenance_rows"] == observed
                    and summary_item["selected_rows"] + summary_item["derived_rows"]
                    + (summary_item["restored_rows"] if document["schema_version"] == 2 else 0) == observed,
                    f"research {data_name} provenance counts differ from the frozen source coverage: {symbol}")
            if document["schema_version"] == 2:
                require(summary_item["restored_rows"] ==
                        (restored_in_window[symbol] if data_name == "mark_price" else 0),
                        f"research {data_name} restored rows differ from the declared daily supplements: {symbol}")
            if data_name != "mark_price":
                require(summary_item["derived_rows"] == 0,
                        f"research {data_name} rows may not be synthesized: {symbol}")
            else:
                require(summary_item["derived_rows"] <= coverage["derived_rows"],
                        f"research derived mark rows exceed frozen source coverage: {symbol}")
            expected_price_rows.append(summary_item)
    expected_keys = [(item["symbol"], item["dataset"]) for item in expected_price_rows]
    summary_keys = [(item.get("symbol"), item.get("dataset")) for item in summary["prices"]
                    if isinstance(item, dict)]
    require(len(summary["prices"]) == len(expected_price_rows) and summary_keys == expected_keys
            and summary["prices"] == expected_price_rows,
            "research snapshot price provenance counts differ from frozen coverage")

    funding_path = _snapshot_path(root, "inputs/funding-mark-provenance.csv",
                                  "frozen funding mark provenance")
    funding_marks = pd.read_csv(funding_path, float_precision="round_trip")
    expected_funding_columns = ["symbol", "timestamp", "funding_rate", "funding_interval_hours",
                                "native_event_mark_price", "proxy_price", "proxy_method", "source_key",
                                "source_row_number", "source_open_time", "source_close_time", "age_ms"]
    if document["schema_version"] == 2:
        expected_funding_columns.append("source_id")
    require(list(funding_marks.columns) == expected_funding_columns,
            "research funding mark provenance columns differ from the frozen schema")
    for column in ("timestamp", "source_open_time", "source_close_time"):
        funding_marks[column] = _utc_index(funding_marks[column], f"funding mark {column}")
    require(funding_marks["symbol"].isin(document["symbols"]).all()
            and not funding_marks.duplicated(["symbol", "timestamp"]).any(),
            "research funding mark provenance has unknown or duplicate event identities")
    require(funding_marks["timestamp"].ge(_stage_funding_start(contract, document)).all()
            and funding_marks["timestamp"].lt(end).all(),
            "research funding mark provenance falls outside the frozen input window")
    numeric = funding_marks[["funding_rate", "funding_interval_hours", "proxy_price", "age_ms",
                             "source_row_number"]].to_numpy(dtype=float)
    require(np.isfinite(numeric).all() and (funding_marks["funding_interval_hours"] > 0).all()
            and (funding_marks["proxy_price"] > 0).all()
            and (funding_marks["source_row_number"] > 0).all()
            and np.equal(funding_marks["source_row_number"].to_numpy(dtype=float),
                         np.floor(funding_marks["source_row_number"].to_numpy(dtype=float))).all()
            and (funding_marks["age_ms"] >= 0).all()
            and np.equal(funding_marks["age_ms"].to_numpy(dtype=float),
                         np.floor(funding_marks["age_ms"].to_numpy(dtype=float))).all()
            and (funding_marks["age_ms"] <= 60_000).all()
            and funding_marks["native_event_mark_price"].isna().all()
            and funding_marks["proxy_method"].eq(document["funding_mark"]["method"]).all()
            and (funding_marks["source_open_time"].astype("int64") // 1_000_000 % 60_000 == 0).all()
            and (funding_marks["source_close_time"] <= funding_marks["timestamp"]).all()
            and ((funding_marks["timestamp"] - funding_marks["source_close_time"]).dt.total_seconds()
                 .mul(1000).eq(funding_marks["age_ms"])).all(),
            "research funding mark provenance contains an invalid proxy")
    all_source_files = document["source_files"] + document.get("supplement_source_files", [])
    archive_keys = {(entry["symbol"], entry["source_key"])
                    for entry in all_source_files
                    if entry["category"] == "markPriceKlines" and entry["interval"] == "1m"}
    require(all((row.symbol, row.source_key) in archive_keys for row in funding_marks.itertuples()),
            "research funding mark references a source archive outside the frozen manifest")
    if document["schema_version"] == 2:
        require(funding_marks["source_id"].notna().all()
                and np.equal(funding_marks["source_id"].to_numpy(dtype=float),
                             np.floor(funding_marks["source_id"].to_numpy(dtype=float))).all(),
                "research funding provenance source IDs must be integers")
        source_by_id = {entry["archive_id"]: entry for entry in all_source_files}
    for row in funding_marks.itertuples():
        archive_month = pd.Timestamp(row.source_open_time).strftime("%Y-%m")
        if document["schema_version"] == 2:
            source = source_by_id.get(int(row.source_id))
            require(source is not None and source["symbol"] == row.symbol
                    and source["source_key"] == row.source_key and source["interval"] == "1m"
                    and int(row.source_row_number) <= source["raw_row_count"] + 1,
                    "research funding minute source ID/row differs from its archive metadata")
            if source["source_kind"] == "official_public_daily_archive":
                require(_daily_supplement_date(source["source_key"], row.symbol, "1m")
                        == pd.Timestamp(row.source_open_time).strftime("%Y-%m-%d"),
                        "research funding minute timestamp differs from its daily archive date")
            else:
                require(row.source_key.endswith(f"{row.symbol}-1m-{archive_month}.zip"),
                        "research funding minute proxy source month differs from its monthly archive")
        else:
            require(row.source_key.endswith(f"{row.symbol}-1m-{archive_month}.zip"),
                    "research funding minute proxy source month differs from its observation time")
    account_funding_marks = funding_marks.loc[
        (funding_marks["timestamp"] >= contract.bounds[0])
        & (funding_marks["timestamp"] < contract.bounds[1])
    ]
    require(len(account_funding_marks) == len(funding),
            "research account funding rows differ from frozen funding mark provenance")
    for symbol, events in funding.groupby("symbol", sort=True):
        marked = account_funding_marks.loc[account_funding_marks["symbol"] == symbol].sort_values("timestamp")
        events = events.sort_values("timestamp")
        multiplier = resolve_market_symbols(symbol).perpetual_multiplier
        require(events["timestamp"].reset_index(drop=True).equals(marked["timestamp"].reset_index(drop=True))
                and np.isclose(events["funding_rate"].to_numpy(dtype=float),
                               marked["funding_rate"].to_numpy(dtype=float), rtol=1e-14, atol=1e-14).all()
                and np.isclose(events["mark_price"].to_numpy(dtype=float),
                               marked["proxy_price"].to_numpy(dtype=float) / multiplier,
                               rtol=1e-14, atol=1e-12).all(),
                f"research account funding values differ from frozen proxy provenance: {symbol}")

    partial_path = _snapshot_path(root, "inputs/partial-spot-source-rows.csv",
                                  "frozen partial spot source rows")
    partial = pd.read_csv(partial_path, float_precision="round_trip")
    partial_columns = ["timestamp", "symbol", "observed_close_time", "expected_close_time"]
    require(list(partial.columns) == partial_columns,
            "research partial spot source rows differ from the frozen schema")
    for column in ("timestamp", "observed_close_time", "expected_close_time"):
        partial[column] = _utc_index(partial[column], f"partial spot {column}")
    require(partial["symbol"].isin(document["symbols"]).all()
            and not partial.duplicated(["symbol", "timestamp"]).any(),
            "research partial spot source rows contain unknown or duplicate bars")
    expected_partial = []
    for item in document["coverage"]:
        if item["dataset"] != "spot_trade":
            continue
        for row in item["partial_bars"]:
            stamp = _manifest_timestamp(row["open_time"], "partial spot open_time")
            if start <= stamp < end:
                expected_partial.append((stamp, item["symbol"],
                                         _manifest_timestamp(row["observed_close_time"], "partial close"),
                                         _manifest_timestamp(row["expected_close_time"], "expected close")))
    actual_partial = sorted(partial.itertuples(index=False, name=None), key=lambda row: (row[1], row[0]))
    expected_partial = sorted(expected_partial, key=lambda row: (row[1], row[0]))
    require(actual_partial == expected_partial
            and provenance.get("partial_spot_source_hours_masked") == len(expected_partial),
            "research partial spot rows differ from frozen coverage provenance")
    require(summary["funding_rate_event_rows"] == len(funding_marks),
            "research funding provenance count differs from its row summary")
    return ResearchMarketInputs(
        panel=panel, frames=frames, funding=funding, universe=universe, diagnostics=provenance,
        dataset_manifest=document, funding_mark_provenance=funding_marks,
        partial_spot_source_rows=partial, price_row_provenance_summary=summary,
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def load_baseline_snapshot(root: Path, *, for_research_agent: bool = False) -> BaselineSnapshot:
    """Load one completed traditional baseline using only its frozen files."""
    root = Path(root).resolve()
    require(root.is_dir(), f"baseline snapshot directory does not exist: {root}")
    contract = MultifactorContract.from_dict(_read_snapshot_json(root, "contract.json", "contract"))
    result = _read_snapshot_json(root, "result.json", "result")
    require(result.get("engine") == "deterministic_multifactor_v1"
            and result.get("status") == "engineering_complete"
            and result.get("model_called") is False and result.get("agent_used") is False,
            "result is not a completed no-Agent deterministic baseline")
    require(result.get("run_id") == contract.run_id
            and Path(result.get("root", "")).resolve() == root,
            "result run identity differs from snapshot root")
    require(result.get("purpose") == contract.purpose and result.get("stage") == contract.stage
            and result.get("contract") == "contract.json",
            "result purpose, stage, or contract reference differs from the frozen contract")
    require(result.get("forward_validation_started") is False
            and result.get("paper_started") is False and result.get("published") is False,
            "snapshot includes an out-of-scope forward, paper, or publication action")
    if for_research_agent and isinstance(contract, ResearchContract):
        require(contract.stage == "development",
                "research Agent may load only a development-stage snapshot")

    manifest = _read_snapshot_json(root, "cards.json", "card manifest")
    cards = _verify_card_manifest(root, contract, manifest, result)

    provenance = _read_snapshot_json(root, "inputs/data-provenance.json", "data provenance")
    require(provenance.get("purpose") == contract.purpose and provenance.get("stage") == contract.stage
            and provenance.get("prior_data_use") == contract.prior_data_use,
            "input provenance differs from contract data-use declarations")
    require(provenance.get("declared_data_processing") == contract.data_processing,
            "input provenance differs from contract data-use declarations")
    start, end = contract.bounds
    require(pd.Timestamp(provenance.get("start")).tz_convert("UTC") == start
            and pd.Timestamp(provenance.get("end_exclusive")).tz_convert("UTC") == end
            and pd.Timestamp(provenance.get("warmup_start")).tz_convert("UTC") == contract.input_start,
            "input provenance time window differs from the contract")
    if isinstance(contract, ResearchContract):
        require(provenance.get("source_causality") == "original_archives_with_declared_funding_proxy"
                and provenance.get("historical_causality_certified") is False,
                "input provenance overstates historical causality or row-level repair evidence")
    else:
        require(provenance.get("policy_id") == POLICY_ID
                and provenance.get("project_database_processing") == PRIMARY_DATA_PROCESSING,
                "input provenance policy or actual database processing differs from the current data contract")
        require(provenance.get("source_causality") == "retrospective_or_unknown"
                and provenance.get("historical_causality_certified") is False
                and provenance.get("row_repair_mask") ==
                "unavailable; no row is certified as an original causal observation",
                "input provenance overstates historical causality or row-level repair evidence")
    expected_independence = (
        "historical A+B development and C internal validation; none is an untouched final test"
        if isinstance(contract, ResearchContract)
        else "historical development/internal validation; no independent final test"
    )
    require(provenance.get("independence") == expected_independence
            and provenance.get("final_test_started") is False,
            "input provenance differs from the frozen historical evidence boundary")
    panel_diagnostics = _read_snapshot_json(root, "inputs/panel-diagnostics.json", "input diagnostics")
    factor_diagnostics = _read_snapshot_json(root, "factor_panels/input-diagnostics.json",
                                             "factor input diagnostics")
    require(panel_diagnostics == factor_diagnostics,
            "factor and input snapshot diagnostics differ")

    universe_path = _snapshot_path(root, "inputs/universe.csv", "universe")
    universe = load_universe(universe_path, contract)
    inputs_panel = _read_panel(root, contract, universe, panel_diagnostics)
    symbols = list(universe.index.get_level_values("symbol").unique())
    frames = _read_market_frames(root, contract, symbols)
    funding = _read_funding(root, contract, symbols)
    if isinstance(contract, ResearchContract):
        inputs = _research_snapshot_inputs(root, contract, inputs_panel, universe, frames, funding, provenance)
    else:
        inputs = MarketInputs(panel=inputs_panel, frames=frames, funding=funding,
                              universe=universe, diagnostics=provenance)

    factors = process_factors(inputs_panel, cards)
    _read_standardized(root, factors.standardized)
    _read_common_mask(root, factors.common_mask)

    _verify_experiments(root, contract, result, cards, factors, symbols)
    records = _build_records(root, contract, inputs, cards, factors, result)
    return BaselineSnapshot(contract=contract, inputs=inputs, cards=cards,
                            factors=factors, result=result, records=records)
