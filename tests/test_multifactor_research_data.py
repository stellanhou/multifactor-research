"""Research-contract and isolated-dataset loader tests."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.research.strategy_research.multifactor_data import _FIELD_SOURCES
from crypto_quant.research.strategy_research.multifactor_contracts import (
    MultifactorContract,
    ResearchContract,
)
from crypto_quant.research.strategy_research.multifactor_data import (
    ResearchDatasetManifest,
    _load_spot_source_rows,
    _mask_partial_spot_rows,
    load_research_inputs,
)
from crypto_quant.research.strategy_research.multifactor_raw_dataset import (
    FIELD_SOURCES_V2,
    SOURCE_V2,
)
from crypto_quant.research.strategy_research.multifactor_evidence import load_baseline_snapshot
from crypto_quant.research.strategy_research.multifactor_workflow import run_research_baseline


def _contract_v1():
    return {
        "schema_version": 1,
        "run_id": "research-contract-test",
        "purpose": "engineering",
        "stage": "development",
        "start": "2022-08-01T00:00:00Z",
        "end": "2024-08-01T00:00:00Z",
        "warmup_hours": 168,
        "horizon_hours": 24,
        "cards": ["one.json", "two.json"],
        "universe": "universe.csv",
        "prior_data_use": "B horizon was previously exposed; evidence is exploratory",
        "data_processing": "isolated raw archives with explicitly declared funding mark proxy",
        "costs": {"initial_capital": 10000, "fee_bps": 10, "slippage_bps": 5,
                  "stress_multiplier": 2},
        "portfolio": {"long_count": 1, "short_count": 1, "gross_exposure": .8,
                      "max_asset_weight": .4, "rebalance_hours": 24, "margin_fraction": .1},
    }


def test_v1_contract_roundtrip_stays_engineering_only():
    raw = _contract_v1()

    contract = MultifactorContract.from_dict(raw)

    assert type(contract) is MultifactorContract
    assert contract.as_dict() == raw
    assert "dataset_manifest" not in contract.as_dict()


def test_v2_research_contract_dispatches_with_shared_contract_validation():
    raw = _contract_v1()
    raw.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json")

    contract = MultifactorContract.from_dict(raw)

    assert isinstance(contract, ResearchContract)
    assert contract.purpose == "research"
    assert contract.stage == "development"
    assert contract.input_start.isoformat() == "2022-07-24T23:00:00+00:00"
    assert contract.as_dict() == raw


@pytest.mark.parametrize("changes", [
    {"purpose": "engineering"},
    {"stage": "final_test"},
    {"dataset_manifest": ""},
    {"dataset_manifest": "/external/manifest.json"},
    {"unexpected_switch": True},
])
def test_v2_research_contract_rejects_wrong_scope_or_extra_fields(changes):
    raw = _contract_v1()
    raw.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json")
    raw.update(copy.deepcopy(changes))

    with pytest.raises(ValueError):
        MultifactorContract.from_dict(raw)


def _manifest_v1():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"]
    archive_months = [str(value) for value in pd.period_range("2022-07", "2026-07", freq="M")]
    source_specs = [
        ("spot", "klines", "1h"),
        ("usd_m_perpetual", "klines", "1h"),
        ("usd_m_perpetual", "markPriceKlines", "1h"),
        ("usd_m_perpetual", "fundingRate", "native"),
        ("usd_m_perpetual", "markPriceKlines", "1m"),
    ]
    source_files = []
    archive_id = 1
    for symbol in symbols:
        for month in archive_months:
            for market, category, interval in source_specs:
                if market == "spot":
                    source_key = f"data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
                elif category == "fundingRate":
                    source_key = f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
                else:
                    source_key = (f"data/futures/um/monthly/{category}/{symbol}/{interval}/"
                                  f"{symbol}-{interval}-{month}.zip")
                source_files.append({
                    "archive_id": archive_id,
                    "source_key": source_key,
                    "source_url": f"https://data.binance.vision/{source_key}",
                    "raw_path": f"raw/{source_key}",
                    "market": market,
                    "category": category,
                    "symbol": symbol,
                    "interval": interval,
                    "month": month,
                    "source_kind": "official_public_download",
                    "sha256": hashlib.sha256(str(archive_id).encode()).hexdigest(),
                    "byte_size": 100,
                    "raw_row_count": 100,
                    "row_count": 100,
                    "selected_row_count": 100,
                    "issue_count": 0,
                    "status": "complete",
                    "csv_member": f"{symbol}.csv",
                })
                archive_id += 1

    input_start = pd.Timestamp("2022-07-24T23:00:00Z")
    end = pd.Timestamp("2026-08-01T00:00:00Z")
    expected_hours = int((end - input_start) / pd.Timedelta(hours=1))
    coverage = []
    spot_feature_missing = 0
    for symbol in symbols:
        for dataset in ("spot_trade", "perpetual_trade", "mark_price"):
            partial_bars = []
            hole_rows = 0
            derived_rows = 1 if dataset == "mark_price" and symbol == "BTCUSDT" else 0
            partial_close_rows = 0
            if dataset == "spot_trade" and symbol in {"BTCUSDT", "ETHUSDT"}:
                ts = "2023-03-24T12:00:00+00:00"
                partial_bars = [{
                    "open_time": ts,
                    "observed_close_time": "2023-03-24T12:59:59.000+00:00",
                    "expected_close_time": "2023-03-24T12:59:59.999+00:00",
                }]
                partial_close_rows = 1
                hole_rows = 2
                spot_feature_missing += hole_rows
            holes = ([{"start": "2023-03-24T12:00:00+00:00",
                       "end_exclusive": "2023-03-24T14:00:00+00:00", "count": 2}]
                     if hole_rows else [])
            selected = expected_hours - hole_rows
            source_observations = selected + partial_close_rows
            coverage.append({
                "symbol": symbol,
                "dataset": dataset,
                "expected_rows": expected_hours,
                "archive_rows": selected - derived_rows + partial_close_rows,
                "selected_rows": selected,
                "source_observation_rows": source_observations,
                "partial_close_rows": partial_close_rows,
                "partial_bars": partial_bars,
                "derived_rows": derived_rows,
                "holes": holes,
            })

    return {
        "schema_version": 1,
        "dataset_id": "multifactor_raw_20261002_v1",
        "database": "market_data.sqlite",
        "database_signature": {"size_bytes": 1, "sha256": "a" * 64},
        "database_schema": "MarketDataStore-compatible SQLite v1 with row provenance tables",
        "source": "Binance Vision official public monthly archives with explicitly named BTC/ETH local raw cache",
        "symbols": symbols,
        "window": {
            "timezone": "UTC",
            "input_start": input_start.isoformat(),
            "funding_start": "2022-07-18T00:00:00+00:00",
            "end_exclusive": end.isoformat(),
            "warmup_hours": 168,
            "stages": {
                "A": {"start": "2022-08-01T00:00:00+00:00", "end_exclusive": "2024-08-01T00:00:00+00:00"},
                "B": {"start": "2024-08-01T00:00:00+00:00", "end_exclusive": "2025-08-01T00:00:00+00:00"},
                "C": {"start": "2025-08-01T00:00:00+00:00", "end_exclusive": end.isoformat()},
            },
            "official_archive_months": ["2022-07", "2026-07"],
        },
        "source_files": source_files,
        "coverage": coverage,
        "field_sources": {
            "spot_*": "Binance Vision spot/monthly/klines 1h rows",
            "perpetual_*": "Binance Vision futures/um/monthly/klines 1h rows",
            "mark_*": "Binance Vision futures/um/monthly/markPriceKlines 1h rows; missing hours may use an exactly complete set of 60 source 1m rows",
            "funding_rate": "Binance Vision futures/um/monthly/fundingRate rows; original rate and interval columns",
            "funding_mark_price": "Previous completed Binance Vision 1m mark bar close proxy; no native event markPrice field is present in the source funding archives",
        },
        "repair_state": {
            "price_interpolation": False,
            "funding_rate_interpolation": False,
            "funding_mark_is_native_event_field": False,
            "funding_mark_proxy_declared": True,
            "minute_to_hour_mark_aggregation": "only complete 60 observed minute rows; no interpolation",
            "primary_source_fallback": False,
            "primary_database_read": False,
            "synthetic_archive_rows": False,
        },
        "funding_mark": {
            "proxy_count": 100,
            "missing_proxy_count": 0,
            "max_proxy_age_ms": 60000,
            "min_proxy_age_ms": 1000,
            "derived_hourly_mark_rows_by_symbol": {symbol: (1 if symbol == "BTCUSDT" else 0) for symbol in symbols},
            "native_event_mark_count": 0,
            "method": "previous_completed_1m_mark_close_at_or_before_funding_time",
            "max_allowed_age_ms": 60000,
            "status_counts": {"proxied": 100},
            "event_rows": 100,
            "causal_rule": "source minute bar close_time <= funding_time and funding_time - close_time <= 60000ms",
            "interpretation": "explicit cost proxy, not Binance's native settlement event mark price",
        },
        "source_causality": "original_archives_with_declared_funding_proxy",
        "source_causality_notes": "Official archived bars preserve exchange close times, but historical publication and receipt timestamps are unavailable.",
        "historical_causality_certified": False,
        "complete": True,
        "execution_grid_complete": True,
        "missing_archives": [],
        "parse_issue_count": 0,
        "spot_feature_missing_rows": spot_feature_missing,
        "feature_missing_rows_accepted": True,
        "unresolved_hourly_price_rows": 0,
        "download_policy": {"workers": 4, "retry_count": 0, "http_404": "recorded as missing archive"},
        "primary_source_fallback": False,
        "excluded_documented_repair_rows": [],
    }


def test_research_dataset_manifest_preserves_source_and_proxy_contract():
    manifest = ResearchDatasetManifest.from_dict(_manifest_v1())

    assert manifest.document["source_causality"] == "original_archives_with_declared_funding_proxy"
    assert manifest.document["historical_causality_certified"] is False
    assert manifest.document["funding_mark"]["native_event_mark_count"] == 0
    assert manifest.document["spot_feature_missing_rows"] == 4


def _manifest_v2_daily_supplement():
    document = _manifest_v1()
    document["schema_version"] = 2
    document["database_schema"] = "MarketDataStore-compatible SQLite v2 with row provenance tables"
    document["source"] = SOURCE_V2
    document["field_sources"] = copy.deepcopy(FIELD_SOURCES_V2)
    document["source_causality_notes"] = (
        "Official monthly and targeted daily archived mark bars preserve exchange close times. "
        "Funding marks use the previous fully completed mark minute close because the funding "
        "archives omit event markPrice; historical publication and receipt timestamps are unavailable."
    )
    document["supplement_source_files"] = []
    restored = {symbol: 0 for symbol in document["symbols"]}
    common_days = ("2022-10-02", "2023-02-24", "2026-06-29")
    extra_days = {"BTCUSDT", "BNBUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"}
    archive_id = 2451
    for symbol in document["symbols"]:
        days = list(common_days) + (["2022-07-31"] if symbol in extra_days else [])
        for day in days:
            for interval in ("1h", "1m"):
                source_key = (f"data/futures/um/daily/markPriceKlines/{symbol}/{interval}/"
                              f"{symbol}-{interval}-{day}.zip")
                rows = 24 if interval == "1h" else 1440
                selected = 24 if interval == "1h" else 3
                document["supplement_source_files"].append({
                    "archive_id": archive_id,
                    "source_key": source_key,
                    "source_url": f"https://data.binance.vision/{source_key}",
                    "raw_path": f"raw_supplement/{source_key}",
                    "market": "usd_m_perpetual",
                    "category": "markPriceKlines",
                    "symbol": symbol,
                    "interval": interval,
                    "month": day[:7],
                    "source_kind": "official_public_daily_archive",
                    "sha256": hashlib.sha256(str(archive_id).encode()).hexdigest(),
                    "byte_size": 100,
                    "raw_row_count": rows,
                    "row_count": rows,
                    "selected_row_count": selected,
                    "issue_count": 0,
                    "status": "complete",
                    "csv_member": f"{symbol}-{interval}-{day}.csv",
                })
                archive_id += 1
                if interval == "1h":
                    restored[symbol] += 24
    for item in document["coverage"]:
        item["restored_rows"] = restored[item["symbol"]] if item["dataset"] == "mark_price" else 0
        if item["dataset"] == "mark_price":
            item["archive_rows"] = (item["selected_rows"] - item["derived_rows"]
                                    - item["restored_rows"] + item["partial_close_rows"])
    document["funding_mark"].update({
        "proxy_count": 205,
        "missing_proxy_count": 0,
        "max_proxy_age_ms": 31,
        "min_proxy_age_ms": 1,
        "event_rows": 205,
        "status_counts": {"proxied": 205},
    })
    return document


def test_research_dataset_manifest_v2_accepts_only_declared_daily_mark_supplements():
    manifest = ResearchDatasetManifest.from_dict(_manifest_v2_daily_supplement())
    raw = _contract_v1()
    raw.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json",
               end="2025-08-01T00:00:00Z")
    contract = MultifactorContract.from_dict(raw)

    manifest.validate_contract(contract)

    assert len(manifest.document["source_files"]) == 2450
    assert len(manifest.document["supplement_source_files"]) == 70
    assert sum(row["restored_rows"] for row in manifest.document["coverage"]
               if row["dataset"] == "mark_price") == 840
    assert sum(row["selected_row_count"] for row in manifest.document["supplement_source_files"]
               if row["interval"] == "1m") == 105


@pytest.mark.parametrize("tamper", ["gap_key", "source_kind", "restored_rows", "missing_file"])
def test_research_dataset_manifest_v2_rejects_daily_supplement_mismatch(tamper):
    document = _manifest_v2_daily_supplement()
    if tamper == "gap_key":
        source = document["supplement_source_files"][0]
        source["source_key"] = source["source_key"].replace("2022-10-02", "2022-10-03")
    elif tamper == "source_kind":
        document["supplement_source_files"][0]["source_kind"] = "unverified_rest"
    elif tamper == "restored_rows":
        next(row for row in document["coverage"]
             if row["symbol"] == "BTCUSDT" and row["dataset"] == "mark_price")["restored_rows"] -= 24
    else:
        document["supplement_source_files"].pop()

    with pytest.raises(ValueError):
        ResearchDatasetManifest.from_dict(document)


def test_research_manifest_binds_development_and_C_contract_windows():
    manifest = ResearchDatasetManifest.from_dict(_manifest_v1())
    raw = _contract_v1()
    raw.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json",
               end="2025-08-01T00:00:00Z")
    development = MultifactorContract.from_dict(raw)
    manifest.validate_contract(development)

    raw["stage"] = "internal_validation"
    raw["start"] = "2025-08-01T00:00:00Z"
    raw["end"] = "2026-08-01T00:00:00Z"
    internal_validation = MultifactorContract.from_dict(raw)
    manifest.validate_contract(internal_validation)
    assert internal_validation.input_start == pd.Timestamp("2025-07-24T23:00:00Z")


def test_research_manifest_path_cannot_escape_contract_root_through_symlink(tmp_path):
    root = tmp_path / "run"
    (root / "data").mkdir(parents=True)
    external = tmp_path / "outside-manifest.json"
    external.write_text("{}", encoding="utf-8")
    (root / "data/manifest.json").symlink_to(external)
    raw = _contract_v1()
    raw.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json",
               end="2025-08-01T00:00:00Z")
    contract = MultifactorContract.from_dict(raw)

    with pytest.raises(ValueError, match="research dataset manifest path may not follow a symlink"):
        ResearchDatasetManifest.from_path(root / "data/manifest.json", contract, root)


def test_partial_spot_source_hours_are_masked_from_feature_panel():
    raw_contract = _contract_v1()
    raw_contract.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json")
    contract = MultifactorContract.from_dict(raw_contract)
    document = _manifest_v1()
    times = pd.date_range(contract.input_start, contract.end, freq="h", inclusive="left")
    missing = pd.Timestamp("2023-03-24T13:00:00Z")
    partial = pd.Timestamp("2023-03-24T12:00:00Z")
    observed_times = times[times != missing]
    close_times = observed_times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    close_times = close_times.where(observed_times != partial,
                                    partial + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1))
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1.0, "close_time": close_times}, index=observed_times)
    bars.index.name = "timestamp"

    class FixtureStore:
        def load_bars(self, *args, **kwargs):
            return bars.copy()

    manifest = ResearchDatasetManifest(
        document=document,
        verification={"price_row_provenance_summary": {
            "prices": [{"symbol": "BTCUSDT", "dataset": "spot_trade", "selected_rows": len(bars),
                        "derived_rows": 0, "provenance_rows": len(bars)}],
            "funding_rate_event_rows": 1,
        }},
    )
    spot_frames, partial_rows = _load_spot_source_rows(FixtureStore(), contract, ["BTCUSDT"], manifest)
    assert len(spot_frames["BTCUSDT"]) == len(bars)
    assert partial_rows["timestamp"].tolist() == [partial]

    index = pd.MultiIndex.from_product([times, ["BTCUSDT"]], names=["timestamp", "symbol"])
    values = pd.DataFrame(1.0, index=index, columns=INPUT_COLUMNS)
    for field in (column for column in INPUT_COLUMNS if column.startswith("spot_")):
        values.loc[(missing, "BTCUSDT"), field] = float("nan")
    universe = pd.Series(True, index=index, dtype=bool)
    diagnostics = {"symbols": {"BTCUSDT": {"coverage": {}}}}
    for field in INPUT_COLUMNS:
        missing_count = 1 if field.startswith("spot_") else 0
        valid_rows = len(times) - missing_count
        diagnostics["symbols"]["BTCUSDT"]["coverage"][field] = {
            "rows": len(times), "valid_rows": valid_rows,
            "coverage_ratio": valid_rows / len(times),
            "eligible_rows": len(times), "eligible_valid_rows": valid_rows,
            "eligible_coverage_ratio": valid_rows / len(times),
            "missing_reasons": ({"missing_or_invalid_source_value": 1} if missing_count else {}),
        }
    panel = FactorInputPanel(values=values, universe=universe, diagnostics=diagnostics)

    _mask_partial_spot_rows(panel, partial_rows)

    assert panel.values.loc[(partial, "BTCUSDT"), "spot_close"] != panel.values.loc[(partial, "BTCUSDT"), "spot_close"]
    assert panel.values.loc[(missing, "BTCUSDT"), "spot_close"] != panel.values.loc[(missing, "BTCUSDT"), "spot_close"]
    assert panel.values.loc[(partial, "BTCUSDT"), "perp_close"] == 1.0
    spot_close_coverage = diagnostics["symbols"]["BTCUSDT"]["coverage"]["spot_close"]
    assert spot_close_coverage["valid_rows"] == len(times) - 2
    assert spot_close_coverage["missing_reasons"] == {
        "missing_or_invalid_source_value": 1,
        "partial_source_hour_close_time": 1,
    }


def _write_minimal_raw_bundle(base: Path) -> tuple[Path, ResearchContract]:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"]
    stage_a_start = pd.Timestamp("2026-09-20T00:00:00Z")
    stage_b_end = pd.Timestamp("2026-09-22T00:00:00Z")
    dataset_end = pd.Timestamp("2026-09-23T00:00:00Z")
    input_start = stage_a_start - pd.Timedelta(hours=169)
    funding_start = pd.Timestamp("2026-09-05T00:00:00Z")
    hours = pd.date_range(input_start, dataset_end, freq="h", inclusive="left")
    base.mkdir(parents=True)
    data_dir = base / "data"
    (data_dir / "raw").mkdir(parents=True)

    document = _manifest_v1()
    document["dataset_id"] = "minimal-research-fixture-v1"
    document["window"] = {
        "timezone": "UTC", "input_start": input_start.isoformat(),
        "funding_start": funding_start.isoformat(), "end_exclusive": dataset_end.isoformat(),
        "warmup_hours": 168,
        "stages": {
            "A": {"start": stage_a_start.isoformat(), "end_exclusive": "2026-09-21T00:00:00+00:00"},
            "B": {"start": "2026-09-21T00:00:00+00:00", "end_exclusive": stage_b_end.isoformat()},
            "C": {"start": stage_b_end.isoformat(), "end_exclusive": dataset_end.isoformat()},
        },
        "official_archive_months": ["2026-09", "2026-09"],
    }
    document["symbols"] = symbols
    document["field_sources"] = copy.deepcopy(_FIELD_SOURCES)
    document["coverage"] = []
    for symbol in symbols:
        for dataset in ("spot_trade", "perpetual_trade", "mark_price"):
            document["coverage"].append({
                "symbol": symbol, "dataset": dataset, "expected_rows": len(hours),
                "archive_rows": len(hours), "selected_rows": len(hours),
                "source_observation_rows": len(hours), "partial_close_rows": 0,
                "partial_bars": [], "derived_rows": 0, "holes": [],
            })
    document["spot_feature_missing_rows"] = 0
    document["unresolved_hourly_price_rows"] = 0
    document["funding_mark"].update({
        "proxy_count": 540, "event_rows": 540, "missing_proxy_count": 0,
        "max_proxy_age_ms": 1, "min_proxy_age_ms": 1,
        "status_counts": {"proxied": 540},
        "derived_hourly_mark_rows_by_symbol": {symbol: 0 for symbol in symbols},
    })
    document["source_files"] = []
    source_specs = [
        ("spot", "klines", "1h"),
        ("usd_m_perpetual", "klines", "1h"),
        ("usd_m_perpetual", "markPriceKlines", "1h"),
        ("usd_m_perpetual", "fundingRate", "native"),
        ("usd_m_perpetual", "markPriceKlines", "1m"),
    ]
    archive_id = 0
    for symbol in symbols:
        for market, category, interval in source_specs:
            archive_id += 1
            if market == "spot":
                source_key = f"data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-2026-09.zip"
            elif category == "fundingRate":
                source_key = f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-2026-09.zip"
            else:
                source_key = (f"data/futures/um/monthly/{category}/{symbol}/{interval}/"
                              f"{symbol}-{interval}-2026-09.zip")
            entry = {
                "archive_id": archive_id, "source_key": source_key,
                "source_url": f"https://data.binance.vision/{source_key}",
                "raw_path": f"raw/{source_key}", "market": market, "category": category,
                "symbol": symbol, "interval": interval, "month": "2026-09",
                "source_kind": "official_public_download", "sha256": "0" * 64,
                "byte_size": 1, "raw_row_count": 100_000, "row_count": 100_000,
                "selected_row_count": 100_000, "issue_count": 0, "status": "complete",
                "csv_member": f"{symbol}.csv",
            }
            raw_path = data_dir / entry["raw_path"]
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            payload = (entry["source_key"] + "\nsynthetic fixture archive\n").encode()
            raw_path.write_bytes(payload)
            entry["byte_size"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            document["source_files"].append(entry)
    archive_ids = {
        (entry["symbol"], entry["market"], entry["category"], entry["interval"]): entry["archive_id"]
        for entry in document["source_files"]
    }

    database_path = data_dir / "market_data.sqlite"
    with sqlite3.connect(database_path) as db:
        db.execute("PRAGMA user_version=1")
        db.executescript("""
            CREATE TABLE source_archives (
                id INTEGER PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, source_url TEXT NOT NULL,
                raw_path TEXT, market TEXT NOT NULL, category TEXT NOT NULL, symbol TEXT NOT NULL,
                interval TEXT NOT NULL, month TEXT NOT NULL, source_kind TEXT NOT NULL,
                sha256 TEXT, byte_size INTEGER NOT NULL, raw_row_count INTEGER NOT NULL,
                row_count INTEGER NOT NULL, selected_row_count INTEGER NOT NULL,
                issue_count INTEGER NOT NULL, status TEXT NOT NULL, csv_member TEXT
            );
            CREATE TABLE futures_archive_files (
                id INTEGER PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, category TEXT NOT NULL,
                symbol TEXT NOT NULL, interval TEXT NOT NULL, byte_size INTEGER NOT NULL,
                etag TEXT, sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,
                issue_count INTEGER NOT NULL, processed_at TEXT NOT NULL
            );
            CREATE TABLE klines (
                symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                volume REAL NOT NULL, close_time INTEGER NOT NULL, quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL, taker_buy_base_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL, PRIMARY KEY(symbol,interval,open_time)
            ) WITHOUT ROWID;
            CREATE TABLE futures_price_bars (
                data_type TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
                open_time INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
                low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
                close_time INTEGER NOT NULL, quote_volume REAL NOT NULL, trades INTEGER NOT NULL,
                taker_buy_base_volume REAL NOT NULL, taker_buy_quote_volume REAL NOT NULL,
                source_file_id INTEGER NOT NULL, PRIMARY KEY(data_type,symbol,interval,open_time)
            ) WITHOUT ROWID;
            CREATE TABLE futures_funding_rates (
                symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
                funding_interval_hours INTEGER NOT NULL, funding_rate REAL NOT NULL,
                source_file_id INTEGER NOT NULL, mark_price REAL,
                PRIMARY KEY(symbol,funding_time)
            ) WITHOUT ROWID;
            CREATE TABLE market_row_provenance (
                target_table TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
                observed_time INTEGER NOT NULL, source_file_id INTEGER NOT NULL,
                source_interval TEXT NOT NULL, source_row_numbers_json TEXT NOT NULL,
                transformation TEXT NOT NULL,
                PRIMARY KEY(target_table,symbol,interval,observed_time)
            ) WITHOUT ROWID;
            CREATE TABLE funding_mark_provenance (
                symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
                native_event_mark_price REAL, proxy_price REAL, proxy_method TEXT NOT NULL,
                source_file_id INTEGER, source_row_number INTEGER, source_open_time INTEGER,
                source_close_time INTEGER, age_ms INTEGER, status TEXT NOT NULL,
                PRIMARY KEY(symbol,funding_time)
            ) WITHOUT ROWID;
        """)
        for entry in document["source_files"]:
            db.execute("INSERT INTO source_archives VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                entry["archive_id"], entry["source_key"], entry["source_url"], entry["raw_path"],
                entry["market"], entry["category"], entry["symbol"], entry["interval"], entry["month"],
                entry["source_kind"], entry["sha256"], entry["byte_size"], entry["raw_row_count"],
                entry["row_count"], entry["selected_row_count"], entry["issue_count"],
                entry["status"], entry["csv_member"],
            ))
            if entry["market"] == "usd_m_perpetual":
                db.execute("INSERT INTO futures_archive_files VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    entry["archive_id"], entry["source_key"], entry["category"], entry["symbol"],
                    entry["interval"], entry["byte_size"], None, entry["sha256"], entry["row_count"],
                    entry["issue_count"], "2026-10-02T00:00:00Z",
                ))

        def epoch_ms(stamp):
            return int(pd.Timestamp(stamp).value // 1_000_000)

        for symbol_index, symbol in enumerate(symbols):
            for bar_index, stamp in enumerate(hours):
                timestamp = epoch_ms(stamp)
                row_number = (stamp.day - 1) * 24 + stamp.hour + 1
                nominal_close = timestamp + 3_599_999
                for market, target, datatype in (
                    ("spot", "klines", None),
                    ("usd_m_perpetual", "futures_price_bars", "klines"),
                    ("usd_m_perpetual", "futures_price_bars", "markPriceKlines"),
                ):
                    category = "klines" if datatype in (None, "klines") else "markPriceKlines"
                    archive_id = archive_ids[(symbol, market, category, "1h")]
                    value = 80.0 + symbol_index * 8.0 + bar_index * (0.03 + symbol_index * 0.0007)
                    if datatype == "markPriceKlines":
                        value *= 1.001
                    record = (symbol, "1h", timestamp, value - 0.01, value + 0.4,
                              value - 0.4, value, 10.0 + symbol_index, nominal_close,
                              (10.0 + symbol_index) * value, 100 + bar_index,
                              4.0 + symbol_index, 4.0 * value)
                    if target == "klines":
                        db.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", record)
                        provenance_target = "klines"
                    else:
                        db.execute("INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (datatype, *record, archive_id))
                        provenance_target = f"futures_price_bars:{datatype}"
                    db.execute("INSERT INTO market_row_provenance VALUES (?,?,?,?,?,?,?,?)", (
                        provenance_target, symbol, "1h", timestamp, archive_id, "1h",
                        json.dumps([row_number]), "official_archive_row",
                    ))

            funding_source_id = archive_ids[(symbol, "usd_m_perpetual", "fundingRate", "native")]
            mark_source_id = archive_ids[(symbol, "usd_m_perpetual", "markPriceKlines", "1m")]
            event_times = pd.date_range(funding_start, dataset_end, freq="8h", inclusive="left")
            for event_number, event_time in enumerate(event_times, start=1):
                funding_time = epoch_ms(event_time)
                source_open = funding_time - 60_000
                source_close = source_open + 59_999
                age_ms = funding_time - source_close
                source_row = event_number * 60
                rate = (symbol_index - 4.5) * 1e-5 + (event_number % 3) * 1e-6
                bar_position = max(0, int((event_time - input_start) / pd.Timedelta(hours=1)))
                price = 80.0 + symbol_index * 8.0 + bar_position * (0.03 + symbol_index * 0.0007)
                price *= 1.001
                db.execute("INSERT INTO futures_funding_rates VALUES (?,?,?,?,?,?)", (
                    symbol, funding_time, 8, rate, funding_source_id, price,
                ))
                db.execute("INSERT INTO market_row_provenance VALUES (?,?,?,?,?,?,?,?)", (
                    "futures_funding_rates", symbol, "native", funding_time, funding_source_id,
                    "native", json.dumps([event_number]), "official_funding_event_row",
                ))
                db.execute("INSERT INTO funding_mark_provenance VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    symbol, funding_time, None, price,
                    "previous_completed_1m_mark_close_at_or_before_funding_time", mark_source_id,
                    source_row, source_open, source_close, age_ms, "proxied",
                ))

    document["database_signature"] = {
        "size_bytes": database_path.stat().st_size,
        "sha256": hashlib.sha256(database_path.read_bytes()).hexdigest(),
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    contract_raw = _contract_v1()
    contract_raw.update(
        schema_version=2, purpose="research", stage="development",
        start=stage_a_start.isoformat(), end=stage_b_end.isoformat(),
        dataset_manifest="data/manifest.json", cards=["cards/card-a.json", "cards/card-b.json"],
        data_processing="isolated official archive bundle with declared funding proxy",
        portfolio={"long_count": 1, "short_count": 1, "gross_exposure": 0.8,
                   "max_asset_weight": 0.4, "rebalance_hours": 24, "margin_fraction": 0.1},
    )
    contract = MultifactorContract.from_dict(contract_raw)
    universe_rows = [(stamp.isoformat(), symbol, True) for stamp in hours for symbol in symbols]
    pd.DataFrame(universe_rows, columns=["timestamp", "symbol", "eligible"]).to_csv(
        base / "universe.csv", index=False,
    )
    card_dir = base / "cards"
    card_dir.mkdir()
    for card_id, expression, direction in (
        ("card-a", "ts_return(perp_close, 2)", 1),
        ("card-b", "div(perp_close, ts_mean(perp_close, 3))", -1),
    ):
        compiled = compile_expression(expression)
        card = {
            "id": card_id, "title": card_id, "source_type": "factor_mining", "status": "research_idea",
            "source": {"run_id": "small-raw-bundle"},
            "original_claim": {"direction": direction, "meaning": "fixture factor",
                               "hypothesis": "fixture hypothesis", "formula": {
                                   "expression": expression,
                                   "expanded_expression": compiled.expanded_expression,
                                   "fields": list(compiled.fields),
                                   "lookback_hours": compiled.lookback_hours,
                               }},
            "market_and_horizon": {"venue": "Binance", "market": "USD-M perpetual", "inputs": "1h",
                                   "retained_horizons": [24], "passed_horizons": [24]},
            "b_validation_status": "passed",
            "admission_evidence": {"eligible_for_idea_pool": True},
        }
        (card_dir / f"{card_id}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    contract_path = base / "contract.json"
    contract_path.write_text(json.dumps(contract.as_dict(), ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return contract_path, contract


def test_minimal_raw_bundle_loads_and_runs_baseline_then_evidence(tmp_path, monkeypatch):
    contract_path, contract = _write_minimal_raw_bundle(tmp_path / "bundle")
    queries = []
    for method_name in ("load_bars", "load_funding", "load_metrics"):
        original = getattr(MarketDataStore, method_name)

        def bounded_query(self, *args, _original=original, _method=method_name, **kwargs):
            queries.append((_method, kwargs.get("start"), kwargs.get("end")))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(MarketDataStore, method_name, bounded_query)
    inputs = load_research_inputs(tmp_path / "bundle/data/manifest.json", contract, contract_path.parent)
    assert isinstance(inputs.price_row_provenance_summary, dict)
    assert inputs.price_row_provenance_summary["funding_rate_event_rows"] == len(inputs.funding_mark_provenance)
    assert inputs.diagnostics["source_causality"] == "original_archives_with_declared_funding_proxy"
    assert inputs.diagnostics["stage"] == "development"
    assert queries
    funding_floor = pd.Timestamp("2026-09-05T00:00:00Z")
    assert all(start is None or pd.Timestamp(start).tz_convert("UTC") >=
               (funding_floor if method == "load_funding" else contract.input_start)
               for method, start, _ in queries)
    assert all(end is None or pd.Timestamp(end).tz_convert("UTC") < contract.bounds[1]
               for _, _, end in queries)

    result = run_research_baseline(contract_path, tmp_path / "runs")
    snapshot = load_baseline_snapshot(Path(result["root"]), for_research_agent=True)

    assert snapshot.contract.stage == "development"
    assert snapshot.inputs.dataset_manifest["dataset_id"] == "minimal-research-fixture-v1"
    assert snapshot.records["data_usage"]["data"]["source_causality"] == (
        "original_archives_with_declared_funding_proxy"
    )


@pytest.mark.parametrize("change", ["archive_gap", "execution_hole", "proxy_future", "causality_claim"])
def test_research_dataset_manifest_rejects_incomplete_or_false_provenance(change):
    manifest = _manifest_v1()
    if change == "archive_gap":
        manifest["source_files"].pop()
    elif change == "execution_hole":
        coverage = next(row for row in manifest["coverage"] if row["dataset"] == "mark_price")
        coverage["selected_rows"] -= 1
        coverage["holes"] = [{"start": "2023-03-24T12:00:00+00:00",
                              "end_exclusive": "2023-03-24T13:00:00+00:00", "count": 1}]
        manifest["complete"] = False
        manifest["execution_grid_complete"] = False
        manifest["unresolved_hourly_price_rows"] = 1
    elif change == "proxy_future":
        manifest["funding_mark"]["max_proxy_age_ms"] = 60001
    else:
        manifest["historical_causality_certified"] = True

    with pytest.raises(ValueError):
        ResearchDatasetManifest.from_dict(manifest)
