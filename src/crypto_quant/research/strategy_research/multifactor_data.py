"""Explicit v1 inputs and isolated, provenance-checked historical research data."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd

from crypto_quant.data_access.market_data import MarketDataStore, SPOT, USD_M_PERPETUAL, resolve_market_symbols
from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS, load_factor_inputs, validate_universe
from crypto_quant.research.data_policy import POLICY_ID, PRIMARY_DATA_PROCESSING
from crypto_quant.research.factor_mining.contracts import require
from .multifactor_contracts import MultifactorContract, ResearchContract
from .multifactor_raw_dataset import (
    FIELD_SOURCES_V1 as _FIELD_SOURCES,
    FIELD_SOURCES_V2 as _FIELD_SOURCES_V2,
    SOURCE_V1 as _SOURCE_V1,
    SOURCE_V2 as _SOURCE_V2,
)


@dataclass
class MarketInputs:
    panel: FactorInputPanel
    frames: dict[str, pd.DataFrame]
    funding: pd.DataFrame
    universe: pd.Series
    diagnostics: dict


@dataclass
class ResearchMarketInputs(MarketInputs):
    dataset_manifest: dict
    funding_mark_provenance: pd.DataFrame
    partial_spot_source_rows: pd.DataFrame
    price_row_provenance_summary: dict


class _ResearchInputWindowStore:
    """Bound factor-input queries to the declared stage and funding warmup."""

    def __init__(self, store: MarketDataStore, contract: ResearchContract, funding_start: pd.Timestamp):
        self.store = store
        self.contract = contract
        self.funding_start = funding_start

    def _bounds(self, start, end, floor: pd.Timestamp):
        if start is not None:
            start = max(pd.Timestamp(start).tz_convert("UTC"), floor)
        last = self.contract.bounds[1] - pd.Timedelta(milliseconds=1)
        if end is not None:
            end = min(pd.Timestamp(end).tz_convert("UTC"), last)
        return start, end

    def load_bars(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["start"], kwargs["end"] = self._bounds(
            kwargs.get("start"), kwargs.get("end"), self.contract.input_start,
        )
        return self.store.load_bars(*args, **kwargs)

    def load_funding(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["start"], kwargs["end"] = self._bounds(
            kwargs.get("start"), kwargs.get("end"), self.funding_start,
        )
        return self.store.load_funding(*args, **kwargs)

    def load_metrics(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["start"], kwargs["end"] = self._bounds(
            kwargs.get("start"), kwargs.get("end"), self.contract.input_start,
        )
        return self.store.load_metrics(*args, **kwargs)


_RESEARCH_MANIFEST_FIELDS_V1 = {
    "schema_version", "dataset_id", "database", "database_signature", "database_schema", "source",
    "symbols", "window", "source_files", "coverage", "field_sources", "repair_state", "funding_mark",
    "source_causality", "source_causality_notes", "historical_causality_certified", "complete",
    "execution_grid_complete", "missing_archives", "parse_issue_count", "spot_feature_missing_rows",
    "feature_missing_rows_accepted", "unresolved_hourly_price_rows", "download_policy",
    "primary_source_fallback", "excluded_documented_repair_rows",
}
_RESEARCH_MANIFEST_FIELDS_V2 = _RESEARCH_MANIFEST_FIELDS_V1 | {"supplement_source_files"}
_SOURCE_FILE_FIELDS = {
    "archive_id", "source_key", "source_url", "raw_path", "market", "category", "symbol", "interval",
    "month", "source_kind", "sha256", "byte_size", "raw_row_count", "row_count", "selected_row_count",
    "issue_count", "status", "csv_member",
}
_COVERAGE_FIELDS = {"symbol", "dataset", "expected_rows", "archive_rows", "selected_rows",
                    "source_observation_rows", "partial_close_rows", "partial_bars", "derived_rows", "holes"}
_COVERAGE_FIELDS_V2 = _COVERAGE_FIELDS | {"restored_rows"}
_PARTIAL_BAR_FIELDS = {"open_time", "observed_close_time", "expected_close_time"}
_HOLE_FIELDS = {"start", "end_exclusive", "count"}
_DATASET_SPECS = (
    ("spot", "klines", "1h"),
    ("usd_m_perpetual", "klines", "1h"),
    ("usd_m_perpetual", "markPriceKlines", "1h"),
    ("usd_m_perpetual", "fundingRate", "native"),
    ("usd_m_perpetual", "markPriceKlines", "1m"),
)
_MARK_DAILY_GAPS_ALL = {"2022-10-02", "2023-02-24", "2026-06-29"}
_MARK_DAILY_GAPS_EXTRA = {"BTCUSDT", "BNBUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"}


def _known_daily_mark_supplements(symbols: list[str]) -> set[tuple[str, str, str]]:
    result = set()
    for symbol in symbols:
        days = set(_MARK_DAILY_GAPS_ALL)
        if symbol in _MARK_DAILY_GAPS_EXTRA:
            days.add("2022-07-31")
        for day in days:
            for interval in ("1h", "1m"):
                result.add((symbol, day, interval))
    return result


def _manifest_timestamp(value, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0),
            f"{label} must be timezone-aware UTC")
    return stamp.tz_convert("UTC")


def _stage_funding_start(contract: ResearchContract, document: dict) -> pd.Timestamp:
    manifest_start = _manifest_timestamp(document["window"]["funding_start"], "funding_start")
    factor_warmup_start = contract.input_start - pd.Timedelta(hours=168)
    return max(manifest_start, factor_warmup_start)


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative.strip(), f"{label} path is required")
    path = Path(relative)
    require(not path.is_absolute(), f"{label} path must be relative to the isolated dataset")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the isolated dataset") from exc
    require(resolved.is_file(), f"{label} file is missing: {relative}")
    return resolved


@dataclass(frozen=True)
class ResearchDatasetManifest:
    """Strict source and row-provenance contract for the isolated historical dataset."""

    document: dict
    path: Path | None = None
    database_path: Path | None = None
    verification: dict | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "ResearchDatasetManifest":
        require(isinstance(value, dict) and type(value.get("schema_version")) is int,
                "research dataset manifest must declare an integer schema_version")
        schema_version = value["schema_version"]
        if schema_version == 1:
            expected_manifest_fields = _RESEARCH_MANIFEST_FIELDS_V1
        elif schema_version == 2:
            expected_manifest_fields = _RESEARCH_MANIFEST_FIELDS_V2
        else:
            raise ValueError(f"unsupported research dataset manifest schema: {schema_version}")
        require(set(value) == expected_manifest_fields,
                f"research dataset manifest fields differ from schema v{schema_version}")
        require(isinstance(value["dataset_id"], str) and value["dataset_id"].strip(),
                "research dataset_id is required")
        database = Path(value["database"])
        require(isinstance(value["database"], str) and value["database"].strip()
                and not database.is_absolute() and ".." not in database.parts,
                "research database path must stay inside its dataset bundle")
        expected_database_schema = (
            "MarketDataStore-compatible SQLite v1 with row provenance tables"
            if schema_version == 1 else
            "MarketDataStore-compatible SQLite v2 with row provenance tables"
        )
        require(value["database_schema"] == expected_database_schema,
                "research dataset database schema is unsupported")
        expected_source = _SOURCE_V1 if schema_version == 1 else _SOURCE_V2
        require(value["source"] == expected_source,
                f"research dataset source differs from schema v{schema_version}")
        signature = value["database_signature"]
        require(isinstance(signature, dict) and set(signature) == {"size_bytes", "sha256"},
                "database_signature fields differ from schema")
        require(type(signature["size_bytes"]) is int and signature["size_bytes"] > 0,
                "database_signature.size_bytes must be positive")
        require(_valid_sha256(signature["sha256"]), "database_signature.sha256 is invalid")

        symbols = value["symbols"]
        require(isinstance(symbols, list) and len(symbols) == 10
                and all(isinstance(symbol, str) and symbol.endswith("USDT") for symbol in symbols)
                and len(set(symbols)) == len(symbols),
                "research dataset must declare the frozen ten-symbol USDT universe")
        window = value["window"]
        window_fields = {"timezone", "input_start", "funding_start", "end_exclusive",
                         "warmup_hours", "stages", "official_archive_months"}
        require(isinstance(window, dict) and set(window) == window_fields,
                "research dataset window fields differ from schema")
        require(window["timezone"] == "UTC", "research dataset timezone must be UTC")
        require(type(window["warmup_hours"]) is int and window["warmup_hours"] > 0,
                "dataset warmup_hours must be a positive integer")
        input_start = _manifest_timestamp(window["input_start"], "dataset input_start")
        funding_start = _manifest_timestamp(window["funding_start"], "dataset funding_start")
        end = _manifest_timestamp(window["end_exclusive"], "dataset end_exclusive")
        require(funding_start <= input_start - pd.Timedelta(hours=167),
                "funding archive does not cover the 7-day factor warmup")
        require(input_start < end, "dataset input window is empty")
        stages = window["stages"]
        require(isinstance(stages, dict) and set(stages) == {"A", "B", "C"},
                "research dataset must declare A, B, and C windows")
        parsed_stages = {}
        for name, bounds in stages.items():
            require(isinstance(bounds, dict) and set(bounds) == {"start", "end_exclusive"},
                    f"dataset stage {name} has invalid boundaries")
            stage_start = _manifest_timestamp(bounds["start"], f"stage {name} start")
            stage_end = _manifest_timestamp(bounds["end_exclusive"], f"stage {name} end")
            require(stage_start < stage_end, f"dataset stage {name} is empty")
            parsed_stages[name] = (stage_start, stage_end)
        require(parsed_stages["A"][1] == parsed_stages["B"][0]
                and parsed_stages["B"][1] == parsed_stages["C"][0]
                and parsed_stages["C"][1] == end,
                "dataset A/B/C stages must be contiguous and end at end_exclusive")
        require(input_start == parsed_stages["A"][0] - pd.Timedelta(hours=window["warmup_hours"] + 1),
                "dataset input_start differs from the frozen warmup contract")

        source_files = value["source_files"]
        require(isinstance(source_files, list) and source_files,
                "research dataset source_files must be a nonempty array")
        months = [str(period) for period in pd.period_range(
            funding_start.strftime("%Y-%m"), (end - pd.Timedelta(milliseconds=1)).strftime("%Y-%m"), freq="M")]
        archive_months = window["official_archive_months"]
        require(isinstance(archive_months, list) and len(archive_months) == 2
                and archive_months == [months[0], months[-1]],
                "official_archive_months must state the inclusive first/last source months")
        expected_archives = {
            (symbol, month, market, category, interval)
            for symbol in symbols for month in months
            for market, category, interval in _DATASET_SPECS
        }
        seen_archives = set()
        seen_archive_ids = set()
        for archive in source_files:
            require(isinstance(archive, dict) and set(archive) == _SOURCE_FILE_FIELDS,
                    "source_files entry fields differ from schema")
            require(type(archive["archive_id"]) is int and archive["archive_id"] > 0,
                    "source archive ID must be a positive integer")
            require(archive["archive_id"] not in seen_archive_ids, "duplicate source archive ID")
            seen_archive_ids.add(archive["archive_id"])
            key = (archive["symbol"], archive["month"], archive["market"],
                   archive["category"], archive["interval"])
            require(key in expected_archives and key not in seen_archives,
                    "source archive is outside or duplicated within the declared bundle window")
            seen_archives.add(key)
            require(re.fullmatch(r"\d{4}-\d{2}", archive["month"]) is not None,
                    "source archive month must use YYYY-MM")
            require(archive["source_kind"] in {"existing_local_raw_cache", "official_public_download"},
                    "source archive kind is unsupported")
            require(archive["status"] == "complete" and archive["issue_count"] == 0,
                    "research source archive is incomplete or has parser issues")
            require(type(archive["byte_size"]) is int and archive["byte_size"] > 0,
                    "source archive byte_size must be positive")
            for field in ("raw_row_count", "row_count", "selected_row_count", "issue_count"):
                require(type(archive[field]) is int and archive[field] >= 0,
                        f"source archive {field} must be a nonnegative integer")
            require(archive["row_count"] <= archive["raw_row_count"]
                    and archive["selected_row_count"] <= archive["row_count"],
                    "source archive row counts are inconsistent")
            require(_valid_sha256(archive["sha256"]), "source archive SHA-256 is invalid")
            require(isinstance(archive["source_key"], str) and archive["source_key"]
                    and archive["source_url"] == "https://data.binance.vision/" + archive["source_key"],
                    "source archive URL must be the official Vision key")
            require(archive["source_key"] == _expected_source_key(
                archive["market"], archive["category"], archive["symbol"], archive["interval"], archive["month"]
            ), "source archive key differs from its declared market/category/symbol/month")
            require(isinstance(archive["raw_path"], str) and archive["raw_path"].strip()
                    and not Path(archive["raw_path"]).is_absolute()
                    and ".." not in Path(archive["raw_path"]).parts,
                    "source archive path must remain inside the dataset bundle")
            require(archive["raw_path"] == f"raw/{archive['source_key']}",
                    "source archive path differs from the frozen bundle layout")
            require(isinstance(archive["csv_member"], str) and archive["csv_member"].strip(),
                    "source archive CSV member is required")
        require(seen_archives == expected_archives,
                "research dataset does not contain all monthly archives for its declared window")

        supplement_files = value.get("supplement_source_files", [])
        restored_by_symbol = {symbol: 0 for symbol in symbols}
        supplement_by_id = {}
        if schema_version == 1:
            require(supplement_files == [], "schema v1 cannot declare daily supplement archives")
        else:
            expected_supplements = _known_daily_mark_supplements(symbols)
            seen_supplements = set()
            maximum_monthly_id = max(seen_archive_ids)
            for source in supplement_files:
                require(isinstance(source, dict) and set(source) == _SOURCE_FILE_FIELDS,
                        "supplement_source_files entry fields differ from schema")
                require(type(source["archive_id"]) is int and source["archive_id"] > maximum_monthly_id
                        and source["archive_id"] not in seen_archive_ids,
                        "daily supplement source ID must be unique and follow monthly archive IDs")
                seen_archive_ids.add(source["archive_id"])
                supplement_by_id[source["archive_id"]] = source
                supplement_date = _daily_supplement_date(source["source_key"], source["symbol"], source["interval"])
                key = (source["symbol"], supplement_date, source["interval"])
                require(key in expected_supplements and key not in seen_supplements,
                        "daily supplement is outside or duplicated within the known source gaps")
                seen_supplements.add(key)
                require(source["market"] == "usd_m_perpetual"
                        and source["category"] == "markPriceKlines"
                        and source["source_kind"] == "official_public_daily_archive"
                        and source["month"] == supplement_date[:7]
                        and source["source_url"] == "https://data.binance.vision/" + source["source_key"]
                        and source["raw_path"] == f"raw_supplement/{source['source_key']}"
                        and source["status"] == "complete" and source["issue_count"] == 0,
                        "daily supplement metadata differs from the official Vision source contract")
                require(type(source["byte_size"]) is int and source["byte_size"] > 0
                        and all(type(source[field]) is int and source[field] >= 0
                                for field in ("raw_row_count", "row_count", "selected_row_count", "issue_count"))
                        and source["row_count"] == source["raw_row_count"]
                        and source["selected_row_count"] <= source["row_count"]
                        and _valid_sha256(source["sha256"])
                        and isinstance(source["csv_member"], str) and source["csv_member"].strip(),
                        "daily supplement archive counts, hash, or CSV member are invalid")
                if source["interval"] == "1h":
                    require(source["raw_row_count"] == source["row_count"] == source["selected_row_count"] == 24,
                            "daily hourly mark archive must contain exactly 24 restored rows")
                    restored_by_symbol[source["symbol"]] += source["selected_row_count"]
                else:
                    require(source["raw_row_count"] == source["row_count"] == 1440
                            and source["selected_row_count"] > 0,
                            "daily minute mark archive must contain a full day and selected proxy rows")
            require(seen_supplements == expected_supplements,
                    "schema v2 must include exactly the known daily mark supplements")
            require(len(supplement_files) == 70
                    and sum(item["selected_row_count"] for item in supplement_files
                            if item["interval"] == "1h") == 840
                    and sum(item["selected_row_count"] for item in supplement_files
                            if item["interval"] == "1m") == 105,
                    "daily mark supplement row totals differ from the frozen 840/105 restoration scope")

        coverage = value["coverage"]
        require(isinstance(coverage, list) and coverage,
                "research dataset coverage must be a nonempty array")
        expected_coverage = {(symbol, dataset) for symbol in symbols
                             for dataset in ("spot_trade", "perpetual_trade", "mark_price")}
        seen_coverage = set()
        expected_hours = int((end - input_start) / pd.Timedelta(hours=1))
        spot_missing = 0
        execution_missing = 0
        expected_coverage_fields = _COVERAGE_FIELDS if schema_version == 1 else _COVERAGE_FIELDS_V2
        for item in coverage:
            require(isinstance(item, dict) and set(item) == expected_coverage_fields,
                    "coverage row fields differ from schema")
            key = (item["symbol"], item["dataset"])
            require(key in expected_coverage and key not in seen_coverage,
                    "coverage has an unknown or duplicate symbol/dataset")
            seen_coverage.add(key)
            for field in ("expected_rows", "archive_rows", "selected_rows", "source_observation_rows",
                          "partial_close_rows", "derived_rows",
                          *( ("restored_rows",) if schema_version == 2 else ())):
                require(type(item[field]) is int and item[field] >= 0,
                        f"coverage {field} must be a nonnegative integer")
            require(item["expected_rows"] == expected_hours,
                    f"coverage hourly grid differs from the dataset window: {key}")
            partial_bars = item["partial_bars"]
            require(isinstance(partial_bars, list) and len(partial_bars) == item["partial_close_rows"],
                    f"coverage partial bar rows are inconsistent: {key}")
            partial_times = []
            for partial in partial_bars:
                require(isinstance(partial, dict) and set(partial) == _PARTIAL_BAR_FIELDS,
                        "partial bar provenance fields differ from schema")
                open_time = _manifest_timestamp(partial["open_time"], "partial bar open_time")
                observed_close = _manifest_timestamp(partial["observed_close_time"], "partial bar close_time")
                expected_close = _manifest_timestamp(partial["expected_close_time"], "partial bar expected close")
                require(open_time == open_time.floor("h")
                        and expected_close == open_time + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
                        and open_time <= observed_close and observed_close != expected_close,
                        "partial bar close_time does not match its nominal hour")
                partial_times.append(open_time)
            require(len(partial_times) == len(set(partial_times)), "duplicate partial bar timestamps")
            holes = item["holes"]
            require(isinstance(holes, list), "coverage holes must be an array")
            hole_rows = 0
            previous_end = None
            hole_intervals = []
            for hole in holes:
                require(isinstance(hole, dict) and set(hole) == _HOLE_FIELDS,
                        "coverage hole fields differ from schema")
                hole_start = _manifest_timestamp(hole["start"], "coverage hole start")
                hole_end = _manifest_timestamp(hole["end_exclusive"], "coverage hole end")
                require(type(hole["count"]) is int and hole["count"] > 0
                        and hole_start < hole_end and (hole_end - hole_start) / pd.Timedelta(hours=1) == hole["count"],
                        "coverage hole interval/count is inconsistent")
                require(previous_end is None or hole_start >= previous_end,
                        "coverage holes must be sorted and nonoverlapping")
                previous_end = hole_end
                hole_rows += hole["count"]
                hole_intervals.append((hole_start, hole_end))
            require(all(any(start <= stamp < end for start, end in hole_intervals) for stamp in partial_times),
                    f"partial bars are not included in source holes: {key}")
            require(item["selected_rows"] + hole_rows == expected_hours
                    and item["derived_rows"] <= item["selected_rows"],
                    f"coverage rows do not partition the expected grid: {key}")
            require(item["source_observation_rows"] >= item["selected_rows"]
                    and item["source_observation_rows"] <= expected_hours
                    and item["archive_rows"] <= item["source_observation_rows"]
                    and item["source_observation_rows"] == item["selected_rows"] + item["partial_close_rows"]
                    and item["archive_rows"] + (item["restored_rows"] if schema_version == 2 else 0)
                    == item["selected_rows"] - item["derived_rows"] + item["partial_close_rows"],
                    f"coverage source observation count is invalid: {key}")
            if item["dataset"] == "spot_trade":
                require(item["derived_rows"] == 0, "spot feature rows cannot be synthesized")
                if schema_version == 2:
                    require(item["restored_rows"] == 0,
                            "daily mark supplement cannot alter spot feature coverage")
                spot_missing += hole_rows
            else:
                require(item["partial_close_rows"] == 0,
                        f"partial execution source bars are unresolved: {key}")
                execution_missing += hole_rows
                if item["dataset"] == "perpetual_trade":
                    require(item["derived_rows"] == 0,
                            "perpetual execution trades cannot be synthesized")
                    if schema_version == 2:
                        require(item["restored_rows"] == 0,
                                "daily mark supplement cannot alter perpetual trade coverage")
                elif schema_version == 2:
                    require(item["restored_rows"] == restored_by_symbol[item["symbol"]],
                            f"mark restored_rows differs from daily source metadata: {key}")
        require(seen_coverage == expected_coverage, "coverage is missing declared symbols/datasets")

        field_sources = value["field_sources"]
        expected_field_sources = _FIELD_SOURCES if schema_version == 1 else _FIELD_SOURCES_V2
        require(isinstance(field_sources, dict) and field_sources == expected_field_sources,
                "research field source descriptions differ from the accepted source contract")
        repair = value["repair_state"]
        require(isinstance(repair, dict)
                and set(repair) == {"price_interpolation", "funding_rate_interpolation",
                                    "funding_mark_is_native_event_field", "funding_mark_proxy_declared",
                                    "minute_to_hour_mark_aggregation", "primary_source_fallback",
                                    "primary_database_read", "synthetic_archive_rows"}
                and repair["price_interpolation"] is False
                and repair["funding_rate_interpolation"] is False
                and repair["funding_mark_is_native_event_field"] is False
                and repair["funding_mark_proxy_declared"] is True
                and repair["minute_to_hour_mark_aggregation"] ==
                "only complete 60 observed minute rows; no interpolation"
                and repair["primary_source_fallback"] is False
                and repair["primary_database_read"] is False
                and repair["synthetic_archive_rows"] is False,
                "research dataset may not interpolate or fall back to the primary database")
        funding_mark = value["funding_mark"]
        funding_statuses = {"proxied", "no_prior_completed_mark_minute",
                            "prior_mark_minute_older_than_60s", "minute_mark_archive_missing"}
        require(isinstance(funding_mark, dict)
                and set(funding_mark) == {"proxy_count", "missing_proxy_count", "max_proxy_age_ms",
                                          "min_proxy_age_ms", "derived_hourly_mark_rows_by_symbol",
                                          "native_event_mark_count", "method", "max_allowed_age_ms",
                                          "status_counts", "event_rows", "causal_rule", "interpretation"},
                "funding mark provenance fields differ from schema")
        status_counts = funding_mark["status_counts"]
        require(funding_mark["native_event_mark_count"] == 0
                and funding_mark["method"] ==
                "previous_completed_1m_mark_close_at_or_before_funding_time"
                and isinstance(status_counts, dict) and set(status_counts) <= funding_statuses
                and "proxied" in status_counts
                and all(type(count) is int and count > 0 for count in status_counts.values())
                and type(funding_mark["proxy_count"]) is int and funding_mark["proxy_count"] > 0
                and type(funding_mark["missing_proxy_count"]) is int and funding_mark["missing_proxy_count"] == 0
                and type(funding_mark["max_proxy_age_ms"]) is int
                and 0 <= funding_mark["max_proxy_age_ms"] <= 60000
                and type(funding_mark["min_proxy_age_ms"]) is int
                and 0 <= funding_mark["min_proxy_age_ms"] <= funding_mark["max_proxy_age_ms"]
                and type(funding_mark["max_allowed_age_ms"]) is int
                and funding_mark["max_allowed_age_ms"] == 60000
                and type(funding_mark["event_rows"]) is int
                and funding_mark["event_rows"] == funding_mark["proxy_count"]
                and sum(status_counts.values()) == funding_mark["event_rows"]
                and status_counts["proxied"] == funding_mark["proxy_count"]
                and sum(status_counts.get(state, 0) for state in funding_statuses - {"proxied"}) ==
                funding_mark["missing_proxy_count"]
                and funding_mark["causal_rule"] ==
                "source minute bar close_time <= funding_time and funding_time - close_time <= 60000ms"
                and funding_mark["interpretation"] ==
                "explicit cost proxy, not Binance's native settlement event mark price"
                and isinstance(funding_mark["derived_hourly_mark_rows_by_symbol"], dict)
                and set(funding_mark["derived_hourly_mark_rows_by_symbol"]) == set(symbols)
                and all(type(count) is int and count >= 0
                        for count in funding_mark["derived_hourly_mark_rows_by_symbol"].values()),
                "funding mark proxy is missing, future-dated, or outside its declared age limit")

        require(value["source_causality"] == "original_archives_with_declared_funding_proxy",
                "research source causality label differs from the accepted archive/proxy method")
        expected_causality_notes = (
            "Official archived bars preserve exchange close times, but historical publication and receipt timestamps are unavailable."
            if schema_version == 1 else
            "Official monthly and targeted daily archived mark bars preserve exchange close times. "
            "Funding marks use the previous fully completed mark minute close because the funding "
            "archives omit event markPrice; historical publication and receipt timestamps are unavailable."
        )
        require(value["source_causality_notes"] == expected_causality_notes,
                "research source-causality limitations differ from the frozen raw archive scope")
        require(value["historical_causality_certified"] is False,
                "historical causality cannot be certified without publication/receipt timestamps")
        require(value["complete"] is True and value["missing_archives"] == []
                and type(value["parse_issue_count"]) is int and value["parse_issue_count"] == 0
                and value["execution_grid_complete"] is True
                and type(value["unresolved_hourly_price_rows"]) is int
                and value["unresolved_hourly_price_rows"] == execution_missing
                and execution_missing == 0
                and type(value["spot_feature_missing_rows"]) is int
                and value["spot_feature_missing_rows"] == spot_missing
                and value["feature_missing_rows_accepted"] is True
                and value["primary_source_fallback"] is False
                and value["excluded_documented_repair_rows"] == [],
                "research dataset is incomplete or silently repairs/falls back on hourly data")
        download_policy = value["download_policy"]
        require(isinstance(download_policy, dict)
                and set(download_policy) == {"workers", "retry_count", "http_404"}
                and type(download_policy["workers"]) is int and download_policy["workers"] > 0
                and type(download_policy["retry_count"]) is int and download_policy["retry_count"] >= 0
                and download_policy["http_404"] == "recorded as missing archive",
                "dataset download policy must record 404 as missing")
        return cls(document=value)

    @classmethod
    def from_path(cls, path: Path, contract: ResearchContract, base_dir: Path) -> "ResearchDatasetManifest":
        base_root = Path(base_dir).resolve()
        declared_path = base_root / contract.dataset_manifest
        expected = declared_path.resolve()
        lexical_path = Path(os.path.abspath(declared_path))
        actual = Path(path).resolve()
        require(actual == expected, "research dataset manifest path differs from the contract")
        require(lexical_path == expected,
                "research dataset manifest path may not follow a symlink")
        require(actual.is_file(), f"research dataset manifest is missing: {actual}")
        manifest = cls.from_dict(json.loads(actual.read_text(encoding="utf-8")))
        manifest = cls(document=manifest.document, path=actual)
        manifest.validate_contract(contract)
        database_path = _manifest_file(actual.parent, manifest.document["database"], "research dataset database")
        require(database_path.stat().st_size == manifest.document["database_signature"]["size_bytes"]
                and _manifest_sha256(database_path) == manifest.document["database_signature"]["sha256"],
                "research database signature differs from the frozen manifest")
        verification = manifest.verify_bundle(database_path, contract)
        return cls(document=manifest.document, path=actual, database_path=database_path,
                   verification=verification)

    def validate_contract(self, contract: ResearchContract) -> None:
        window = self.document["window"]
        stages = window["stages"]
        if contract.stage == "development":
            expected_start, expected_end = stages["A"]["start"], stages["B"]["end_exclusive"]
        else:
            expected_start, expected_end = stages["C"]["start"], stages["C"]["end_exclusive"]
        require(contract.bounds == (_manifest_timestamp(expected_start, "research stage start"),
                                    _manifest_timestamp(expected_end, "research stage end")),
                "research contract bounds differ from its predeclared A+B/C stage")
        manifest_input_start = _manifest_timestamp(window["input_start"], "research input_start")
        require(contract.warmup_hours == window["warmup_hours"]
                and (contract.input_start == manifest_input_start if contract.stage == "development"
                     else contract.input_start >= manifest_input_start),
                "research contract warmup starts outside the frozen dataset window")
        require(contract.bounds[1] <= _manifest_timestamp(window["end_exclusive"], "research dataset end"),
                "research contract extends past the raw dataset window")
        require(contract.bounds[0] >= _manifest_timestamp(window["funding_start"], "funding start"),
                "research account window precedes the raw funding dataset")

    def verify_bundle(self, database_path: Path, contract: ResearchContract) -> dict:
        root = self.path.parent
        monthly_entries = self.document["source_files"]
        supplement_entries = self.document.get("supplement_source_files", [])
        entries = monthly_entries + supplement_entries
        archive_by_id = {entry["archive_id"]: entry for entry in entries}
        for entry in entries:
            label = ("daily supplement archive" if entry["source_kind"] == "official_public_daily_archive"
                     else "raw source archive")
            raw_path = _manifest_file(root, entry["raw_path"], label)
            require(raw_path.stat().st_size == entry["byte_size"]
                    and _manifest_sha256(raw_path) == entry["sha256"],
                    f"{label} signature differs: {entry['source_key']}")

        with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            require(db.execute("PRAGMA user_version").fetchone()[0] == self.document["schema_version"],
                    "research database user_version differs from manifest")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            require({"source_archives", "market_row_provenance", "funding_mark_provenance",
                     "klines", "futures_price_bars", "futures_funding_rates"}.issubset(tables),
                    "research database is missing required source or market tables")
            source_columns = [row[1] for row in db.execute("PRAGMA table_info(source_archives)")]
            expected_columns = ["id", "source_key", "source_url", "raw_path", "market", "category", "symbol",
                                "interval", "month", "source_kind", "sha256", "byte_size", "raw_row_count",
                                "row_count", "selected_row_count", "issue_count", "status", "csv_member"]
            require(source_columns == expected_columns, "source_archives table schema differs from manifest")
            provenance_columns = [row[1] for row in db.execute("PRAGMA table_info(market_row_provenance)")]
            require(provenance_columns == ["target_table", "symbol", "interval", "observed_time", "source_file_id",
                                           "source_interval", "source_row_numbers_json", "transformation"],
                    "market_row_provenance table schema differs from manifest")
            funding_columns = [row[1] for row in db.execute("PRAGMA table_info(funding_mark_provenance)")]
            require(funding_columns == ["symbol", "funding_time", "native_event_mark_price", "proxy_price",
                                        "proxy_method", "source_file_id", "source_row_number", "source_open_time",
                                        "source_close_time", "age_ms", "status"],
                    "funding_mark_provenance table schema differs from manifest")
            rows = db.execute("SELECT id,source_key,source_url,raw_path,market,category,symbol,interval,month,"
                              "source_kind,sha256,byte_size,raw_row_count,row_count,selected_row_count,issue_count,"
                              "status,csv_member FROM source_archives ORDER BY id").fetchall()
            require(len(rows) == len(entries), "SQLite source archive rows differ from manifest count")
            for row in rows:
                archive_id = row[0]
                require(archive_id in archive_by_id, "SQLite source archive ID is absent from manifest")
                archive = archive_by_id[archive_id]
                for key, value in zip(expected_columns[1:], row[1:]):
                    require(archive[key] == value,
                            f"SQLite source archive metadata differs from manifest: {archive_id}/{key}")
            futures_archives = [entry for entry in entries if entry["market"] == "usd_m_perpetual"]
            futures_rows = db.execute(
                "SELECT id,source_key,category,symbol,interval,byte_size,sha256,row_count,issue_count "
                "FROM futures_archive_files ORDER BY id"
            ).fetchall()
            require(len(futures_rows) == len(futures_archives),
                    "futures_archive_files rows differ from monthly and daily source entries")
            futures_by_id = {entry["archive_id"]: entry for entry in futures_archives}
            for row in futures_rows:
                source = futures_by_id.get(row[0])
                require(source is not None and tuple(row[1:]) == (
                    source["source_key"], source["category"], source["symbol"], source["interval"],
                    source["byte_size"], source["sha256"], source["row_count"], source["issue_count"],
                ), f"futures_archive_files metadata differs from manifest: {row[0]}")
            symbols = self.document["symbols"]
            placeholders = ",".join("?" for _ in symbols)
            start_ms = contract.input_start.value // 1_000_000
            end_ms = contract.bounds[1].value // 1_000_000
            stage_funding_start = _stage_funding_start(contract, self.document)
            funding_start_ms = stage_funding_start.value // 1_000_000
            unmatched_spot = db.execute(
                "SELECT COUNT(*) FROM market_row_provenance p LEFT JOIN klines k "
                "ON p.target_table='klines' AND p.symbol=k.symbol AND p.interval=k.interval "
                "AND p.observed_time=k.open_time "
                f"WHERE p.target_table='klines' AND p.symbol IN ({placeholders}) "
                "AND p.observed_time>=? AND p.observed_time<? AND k.open_time IS NULL",
                (*symbols, start_ms, end_ms),
            ).fetchone()[0]
            missing_spot = db.execute(
                "SELECT COUNT(*) FROM klines k LEFT JOIN market_row_provenance p "
                "ON p.target_table='klines' AND p.symbol=k.symbol AND p.interval=k.interval "
                "AND p.observed_time=k.open_time "
                f"WHERE k.symbol IN ({placeholders}) AND k.interval='1h' AND k.open_time>=? AND k.open_time<? "
                "AND p.source_file_id IS NULL",
                (*symbols, start_ms, end_ms),
            ).fetchone()[0]
            unmatched_perpetual = db.execute(
                "SELECT COUNT(*) FROM market_row_provenance p LEFT JOIN futures_price_bars f "
                "ON p.target_table='futures_price_bars:'||f.data_type AND p.symbol=f.symbol "
                "AND p.interval=f.interval AND p.observed_time=f.open_time "
                "AND p.source_file_id=f.source_file_id "
                "WHERE p.target_table IN ('futures_price_bars:klines', "
                f"'futures_price_bars:markPriceKlines') AND p.symbol IN ({placeholders}) "
                "AND p.interval='1h' AND p.observed_time>=? AND p.observed_time<? AND f.open_time IS NULL",
                (*symbols, start_ms, end_ms),
            ).fetchone()[0]
            missing_perpetual = db.execute(
                "SELECT COUNT(*) FROM futures_price_bars f LEFT JOIN market_row_provenance p "
                "ON p.target_table='futures_price_bars:'||f.data_type AND p.symbol=f.symbol "
                "AND p.interval=f.interval AND p.observed_time=f.open_time "
                "AND p.source_file_id=f.source_file_id "
                f"WHERE f.symbol IN ({placeholders}) AND f.interval='1h' AND f.open_time>=? AND f.open_time<? "
                "AND p.source_file_id IS NULL",
                (*symbols, start_ms, end_ms),
            ).fetchone()[0]
            unmatched_funding = db.execute(
                "SELECT COUNT(*) FROM market_row_provenance p LEFT JOIN futures_funding_rates f "
                "ON p.target_table='futures_funding_rates' AND p.symbol=f.symbol "
                "AND p.observed_time=f.funding_time AND p.source_file_id=f.source_file_id "
                f"WHERE p.target_table='futures_funding_rates' AND p.symbol IN ({placeholders}) "
                "AND p.observed_time>=? AND p.observed_time<? AND f.funding_time IS NULL",
                (*symbols, funding_start_ms, end_ms),
            ).fetchone()[0]
            missing_funding = db.execute(
                "SELECT COUNT(*) FROM futures_funding_rates f LEFT JOIN market_row_provenance p "
                "ON p.target_table='futures_funding_rates' AND p.symbol=f.symbol "
                "AND p.observed_time=f.funding_time AND p.source_file_id=f.source_file_id "
                f"WHERE f.symbol IN ({placeholders}) AND f.funding_time>=? AND f.funding_time<? "
                "AND p.source_file_id IS NULL",
                (*symbols, funding_start_ms, end_ms),
            ).fetchone()[0]
            require(not any((unmatched_spot, missing_spot, unmatched_perpetual, missing_perpetual,
                             unmatched_funding, missing_funding)),
                    "loaded source rows and market_row_provenance do not map one-to-one")
            price_rows = _verify_market_row_provenance(
                db, archive_by_id, self.document["symbols"], contract,
                stage_funding_start,
                supplement_ids=set(source["archive_id"] for source in supplement_entries),
            )
            funding_rows = _verify_funding_mark_provenance(
                db, archive_by_id, self.document, contract, stage_funding_start,
            )
        return {"price_row_provenance_summary": price_rows,
                "funding_mark_provenance": funding_rows}


def _source_row_numbers(value: str, label: str) -> list[int]:
    try:
        row_numbers = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} source_row_numbers_json is invalid") from exc
    require(isinstance(row_numbers, list) and row_numbers
            and all(type(number) is int and number > 0 for number in row_numbers),
            f"{label} source row numbers must be positive integers")
    return row_numbers


def _verify_market_row_provenance(db, archive_by_id: dict, symbols: list[str], contract: ResearchContract,
                                  funding_start: pd.Timestamp,
                                  supplement_ids: set[int]) -> dict:
    start_ms = contract.input_start.value // 1_000_000
    end_ms = contract.bounds[1].value // 1_000_000
    symbols = list(symbols)
    placeholders = ",".join("?" for _ in symbols)
    counts: dict[tuple[str, str], dict[str, int]] = {}
    query = (
        "SELECT target_table,symbol,interval,observed_time,source_file_id,source_interval,"
        "source_row_numbers_json,transformation FROM market_row_provenance "
        "WHERE target_table IN ('klines','futures_price_bars:klines',"
        f"'futures_price_bars:markPriceKlines') AND symbol IN ({placeholders}) "
        "AND observed_time>=? AND observed_time<? ORDER BY symbol,observed_time,target_table"
    )
    for target, symbol, interval, observed_ms, source_id, source_interval, row_json, transform in db.execute(
            query, (*symbols, start_ms, end_ms)):
        require(type(observed_ms) is int and start_ms <= observed_ms < end_ms
                and observed_ms % 3_600_000 == 0 and interval == "1h",
                "price provenance row is outside the hourly research input window")
        archive = archive_by_id.get(source_id)
        require(archive is not None and archive["symbol"] == symbol
                and archive["month"] == pd.Timestamp(observed_ms, unit="ms", tz="UTC").strftime("%Y-%m"),
                "price provenance source archive differs from row symbol/time")
        row_numbers = _source_row_numbers(row_json, f"{symbol}/{target}")
        require(max(row_numbers) <= archive["raw_row_count"] + 1,
                "price row number exceeds its declared raw archive row count")
        if target == "klines":
            require(archive["market"] == "spot" and archive["category"] == "klines"
                    and archive["interval"] == source_interval == "1h"
                    and transform == "official_archive_row" and len(row_numbers) == 1,
                    "spot price row is not linked to one original hourly archive row")
            dataset, kind = "spot_trade", "selected_rows"
        else:
            require(archive["market"] == "usd_m_perpetual",
                    "perpetual price row references a non-perpetual archive")
            if target == "futures_price_bars:klines" and archive["category"] == "klines":
                require(archive["interval"] == source_interval == "1h"
                        and transform == "official_archive_row" and len(row_numbers) == 1,
                        "perpetual trade row is not linked to one original hourly archive row")
                dataset, kind = "perpetual_trade", "selected_rows"
            elif (target == "futures_price_bars:markPriceKlines"
                  and archive["category"] == "markPriceKlines" and archive["interval"] == "1h"):
                expected_transform = ("official_daily_archive_row" if source_id in supplement_ids
                                      else "official_archive_row")
                require(source_interval == "1h" and transform == expected_transform
                        and len(row_numbers) == 1,
                        "hourly mark row is not linked to its declared hourly archive source")
                if source_id in supplement_ids:
                    require(_daily_supplement_date(archive["source_key"], symbol, "1h")
                            == pd.Timestamp(observed_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                            "daily mark row timestamp differs from its source archive day")
                    dataset, kind = "mark_price", "restored_rows"
                else:
                    dataset, kind = "mark_price", "selected_rows"
            elif (target == "futures_price_bars:markPriceKlines"
                  and archive["category"] == "markPriceKlines" and archive["interval"] == "1m"):
                require(source_interval == "1m"
                        and transform == "aggregate_60_complete_observed_1m_mark_bars"
                        and len(row_numbers) == 60
                        and all(right == left + 1 for left, right in zip(row_numbers[:-1], row_numbers[1:])),
                        "derived hourly mark is not linked to 60 contiguous original minute rows")
                dataset, kind = "mark_price", "derived_rows"
            else:
                raise ValueError("perpetual price row points to an unsupported source archive")
        group = counts.setdefault((symbol, dataset),
                                  {"selected_rows": 0, "derived_rows": 0, "restored_rows": 0})
        group[kind] += 1

    summary = [
        {"symbol": symbol, "dataset": dataset,
         **counts.get((symbol, dataset),
                      {"selected_rows": 0, "derived_rows": 0, "restored_rows": 0}),
         "provenance_rows": sum(counts.get((symbol, dataset), {}).values())}
        for symbol in symbols for dataset in ("spot_trade", "perpetual_trade", "mark_price")
    ]
    if not supplement_ids:
        for row in summary:
            row.pop("restored_rows")
    funding_start_ms = funding_start.value // 1_000_000
    funding_sql = (
        "SELECT target_table,symbol,interval,observed_time,source_file_id,source_interval,"
        "source_row_numbers_json,transformation FROM market_row_provenance "
        f"WHERE target_table='futures_funding_rates' AND symbol IN ({placeholders}) "
        "AND observed_time>=? AND observed_time<? ORDER BY symbol,observed_time"
    )
    funding_rows = 0
    for target, symbol, interval, observed_ms, source_id, source_interval, row_json, transform in db.execute(
            funding_sql, (*symbols, funding_start_ms, end_ms)):
        require(target == "futures_funding_rates" and type(observed_ms) is int
                and funding_start_ms <= observed_ms < end_ms,
                "funding rate row provenance is outside the declared funding window")
        archive = archive_by_id.get(source_id)
        require(archive is not None and archive["symbol"] == symbol
                and archive["market"] == "usd_m_perpetual"
                and archive["category"] == "fundingRate" and archive["interval"] == "native"
                and archive["month"] == pd.Timestamp(observed_ms, unit="ms", tz="UTC").strftime("%Y-%m")
                and source_interval == "native" and transform == "official_funding_event_row",
                "funding event row is not linked to its original Vision archive")
        funding_row_numbers = _source_row_numbers(row_json, f"{symbol}/funding event")
        require(max(funding_row_numbers) <= archive["raw_row_count"] + 1,
                "funding row number exceeds its declared raw archive row count")
        funding_rows += 1
    require(funding_rows > 0, "research input window has no source-provenanced funding events")
    return {"prices": summary, "funding_rate_event_rows": funding_rows}


def _verify_funding_mark_provenance(db, archive_by_id: dict, manifest: dict,
                                    contract: ResearchContract,
                                    funding_start: pd.Timestamp) -> pd.DataFrame:
    symbols = sorted(manifest["symbols"])
    placeholders = ",".join("?" for _ in symbols)
    start_ms = funding_start.value // 1_000_000
    end_ms = contract.bounds[1].value // 1_000_000
    query = (
        "SELECT f.symbol,f.funding_time,f.funding_rate,f.funding_interval_hours,f.source_file_id,f.mark_price,"
        "p.native_event_mark_price,p.proxy_price,p.proxy_method,p.source_file_id,p.source_row_number,"
        "p.source_open_time,p.source_close_time,p.age_ms,p.status "
        "FROM futures_funding_rates f LEFT JOIN funding_mark_provenance p "
        "ON f.symbol=p.symbol AND f.funding_time=p.funding_time "
        f"WHERE f.symbol IN ({placeholders}) AND f.funding_time>=? AND f.funding_time<? "
        "ORDER BY f.symbol,f.funding_time"
    )
    all_source_entries = manifest["source_files"] + manifest.get("supplement_source_files", [])
    entries = {entry["archive_id"]: entry for entry in all_source_entries}
    rows = []
    source_use_rows: dict[int, set[int]] = {}
    for row in db.execute(query, (*symbols, start_ms, end_ms)):
        (symbol, funding_time, rate, interval, funding_file_id, mark_price, native_mark, proxy_price,
         method, mark_file_id, source_row, source_open, source_close, age_ms, status) = row
        require(symbol in symbols and type(funding_time) is int,
                "funding event provenance has an invalid event identity")
        funding_archive = entries.get(funding_file_id)
        mark_archive = entries.get(mark_file_id)
        require(funding_archive is not None and funding_archive["symbol"] == symbol
                and funding_archive["market"] == "usd_m_perpetual"
                and funding_archive["category"] == "fundingRate"
                and funding_archive["interval"] == "native"
                and funding_archive["month"] == pd.Timestamp(funding_time, unit="ms", tz="UTC").strftime("%Y-%m"),
                "funding event is not linked to its original Vision rate archive")
        require(native_mark is None and mark_archive is not None
                and mark_archive["symbol"] == symbol
                and mark_archive["market"] == "usd_m_perpetual"
                and mark_archive["category"] == "markPriceKlines"
                and mark_archive["interval"] == "1m",
                "funding event mark provenance is not linked to a declared minute proxy")
        require(type(source_row) is int and source_row > 0
                and source_row <= mark_archive["raw_row_count"] + 1
                and type(source_open) is int and type(source_close) is int
                and type(age_ms) is int and source_close == source_open + 59_999
                and source_open % 60_000 == 0
                and source_open <= source_close <= funding_time
                and age_ms == funding_time - source_close and 0 <= age_ms <= 60_000,
                "funding mark proxy time is future-dated or outside the 60-second limit")
        require(mark_archive["month"] == pd.Timestamp(source_open, unit="ms", tz="UTC").strftime("%Y-%m"),
                "funding mark source minute references a different archive month")
        if mark_archive["source_kind"] == "official_public_daily_archive":
            require(_daily_supplement_date(mark_archive["source_key"], symbol, "1m")
                    == pd.Timestamp(source_open, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                    "funding mark minute timestamp differs from its daily supplement source")
        require(method == "previous_completed_1m_mark_close_at_or_before_funding_time"
                and status == "proxied" and proxy_price is not None and mark_price is not None
                and np.isfinite(float(proxy_price)) and float(proxy_price) > 0
                and np.isclose(float(proxy_price), float(mark_price), rtol=1e-14, atol=1e-12)
                and np.isfinite(float(rate)) and type(interval) in (int, float) and float(interval) > 0,
                "funding event values differ from their explicit source proxy")
        source_use_rows.setdefault(mark_file_id, set()).add(source_row)
        provenance_row = {
            "symbol": symbol,
            "timestamp": pd.Timestamp(funding_time, unit="ms", tz="UTC"),
            "funding_rate": float(rate),
            "funding_interval_hours": float(interval),
            "native_event_mark_price": None,
            "proxy_price": float(proxy_price),
            "proxy_method": method,
            "source_key": mark_archive["source_key"],
            "source_row_number": source_row,
            "source_open_time": pd.Timestamp(source_open, unit="ms", tz="UTC"),
            "source_close_time": pd.Timestamp(source_close, unit="ms", tz="UTC"),
            "age_ms": age_ms,
        }
        if manifest["schema_version"] == 2:
            provenance_row["source_id"] = mark_file_id
        rows.append(provenance_row)
    require(bool(rows), "research contract window has no funding mark proxy events")
    for source_id, used_rows in source_use_rows.items():
        source = entries[source_id]
        if source["source_kind"] == "official_public_daily_archive":
            require(len(used_rows) <= source["selected_row_count"],
                    "funding proxy rows exceed the selected daily minute source rows")
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _valid_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _expected_source_key(market: str, category: str, symbol: str, interval: str, month: str) -> str:
    if market == "spot" and category == "klines" and interval == "1h":
        return f"data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
    if market == "usd_m_perpetual" and category in {"klines", "markPriceKlines"} and interval in {"1h", "1m"}:
        if category == "klines" and interval != "1h":
            raise ValueError("perpetual trade archive interval must be 1h")
        return (f"data/futures/um/monthly/{category}/{symbol}/{interval}/"
                f"{symbol}-{interval}-{month}.zip")
    if market == "usd_m_perpetual" and category == "fundingRate" and interval == "native":
        return f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    raise ValueError("source archive market/category/interval is outside the research dataset schema")


def _daily_supplement_date(source_key: str, symbol: str, interval: str) -> str:
    pattern = (rf"data/futures/um/daily/markPriceKlines/{re.escape(symbol)}/{re.escape(interval)}/"
               rf"{re.escape(symbol)}-{re.escape(interval)}-(\d{{4}}-\d{{2}}-\d{{2}})\.zip")
    match = re.fullmatch(pattern, source_key)
    require(match is not None, "daily mark source key differs from the frozen Vision layout")
    stamp = pd.Timestamp(match.group(1))
    require(stamp.strftime("%Y-%m-%d") == match.group(1),
            "daily mark source key date is invalid")
    return match.group(1)


def resolve_input_path(path: str, base_dir: Path) -> Path:
    result = Path(path)
    return result.resolve() if result.is_absolute() else (base_dir / result).resolve()


def load_universe(path: Path, contract: MultifactorContract) -> pd.Series:
    frame = pd.read_csv(path)
    require(set(frame.columns) == {"timestamp", "symbol", "eligible"},
            "universe CSV requires timestamp,symbol,eligible")
    require(frame.eligible.isin([True, False, 0, 1]).all(), "eligible must be explicit boolean/0/1")
    require(frame.timestamp.map(lambda t: pd.Timestamp(t).tzinfo is not None).all(),
            "universe timestamps must explicitly include a timezone")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="ISO8601")
    require(frame.symbol.notna().all(), "universe symbols required")
    universe = validate_universe(frame.set_index(["timestamp", "symbol"]).eligible.astype(bool))
    first, end = contract.input_start, contract.bounds[1]
    universe = universe.loc[(universe.index.get_level_values("timestamp") >= first)
                            & (universe.index.get_level_values("timestamp") < end)]
    universe = validate_universe(universe)
    times = universe.index.get_level_values("timestamp").unique()
    require(times.equals(pd.date_range(first, end, freq="h", inclusive="left")),
            "universe must include the complete frozen warmup and account window")
    symbols = universe.index.get_level_values("symbol").unique()
    require(len(symbols) >= contract.portfolio["long_count"] + contract.portfolio["short_count"],
            "universe has fewer assets than declared portfolio slots")
    return universe


def _bars(store, native, price_type, index):
    frame = store.load_bars(USD_M_PERPETUAL, native, "1h", start=index[0], end=index[-1],
                            price_type=price_type, derive=False)
    frame.index = frame.index.as_unit("ns")
    require(frame.index.equals(index), f"{native}/{price_type}: incomplete execution grid")
    require((frame.close_time == frame.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).all(),
            f"{native}/{price_type}: incomplete hourly bars")
    values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values > 0).all(), f"{native}/{price_type}: invalid prices")
    require((frame.high >= frame[["open", "close", "low"]].max(axis=1)).all()
            and (frame.low <= frame[["open", "close", "high"]].min(axis=1)).all(),
            f"{native}/{price_type}: invalid OHLC")
    return frame


def _funding(store, symbol, native, start, end, multiplier):
    # Read the last preceding event as coverage evidence, never manufacture a zero rate.
    frame = store.load_funding(native, start=start, end=end - pd.Timedelta(milliseconds=1), include_previous=True)
    frame.index = frame.index.as_unit("ns")
    require(frame.index.is_unique and frame.index.is_monotonic_increasing, f"{native}: duplicate funding times")
    values = frame[["funding_rate", "funding_interval_hours", "mark_price"]].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values[:, 1:] > 0).all(), f"{native}: invalid funding events")
    hours = frame.index.floor("h")
    require(hours[0] <= start, f"{native}: no funding coverage at account start")
    intervals = pd.to_timedelta(frame.funding_interval_hours.to_numpy(), unit="h")
    adjacent_intervals = [max(previous, following)
                          for previous, following in zip(intervals[:-1], intervals[1:])]
    require(all(b - a <= interval for a, b, interval in zip(hours[:-1], hours[1:], adjacent_intervals)),
            f"{native}: missing funding settlement")
    # Real settlement timestamps may be several milliseconds after the hour.
    require(hours[-1] + intervals[-1] >= end, f"{native}: funding coverage ends before account end")
    actual = frame.loc[(frame.index >= start) & (frame.index < end), ["funding_rate", "mark_price"]].copy()
    actual["mark_price"] /= multiplier
    actual["symbol"] = symbol
    actual = actual.reset_index()
    return actual, {"source_table": frame.attrs["source_table"], "events": len(actual),
                    "coverage_first": frame.index[0].isoformat(), "coverage_last": frame.index[-1].isoformat()}


def load_inputs(db: Path, contract: MultifactorContract, base_dir: Path) -> MarketInputs:
    require(contract.purpose == "engineering", "first-version loader supports engineering replay until row repair provenance is available")
    universe = load_universe(resolve_input_path(contract.universe, base_dir), contract)
    store = MarketDataStore(db)
    panel = load_factor_inputs(store, universe)
    start, end = contract.bounds
    index = pd.date_range(start - pd.Timedelta(hours=1), end, freq="h", inclusive="left")
    frames, events, sources = {}, [], {}
    for symbol in universe.index.get_level_values("symbol").unique():
        pair = resolve_market_symbols(symbol)
        trade = _bars(store, pair.perpetual, "trade", index)
        mark = _bars(store, pair.perpetual, "mark", index)
        frames[symbol] = pd.DataFrame({"open": trade.open / pair.perpetual_multiplier,
                                      "close": trade.close / pair.perpetual_multiplier,
                                      "mark_close": mark.close / pair.perpetual_multiplier}, index=index)
        event, funding_source = _funding(store, symbol, pair.perpetual, start, end, pair.perpetual_multiplier)
        events.append(event)
        sources[symbol] = {"native_perpetual": pair.perpetual, "base_multiplier": pair.perpetual_multiplier,
                           "trade_rows": len(trade), "mark_rows": len(mark), "funding": funding_source}
    funding = pd.concat(events, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    diagnostics = {"source_database": str(Path(db).resolve()), "policy_id": POLICY_ID,
                   "purpose": contract.purpose, "stage": contract.stage, "prior_data_use": contract.prior_data_use,
                   "declared_data_processing": contract.data_processing,
                   "project_database_processing": PRIMARY_DATA_PROCESSING,
                   "row_repair_mask": "unavailable; no row is certified as an original causal observation",
                   "source_causality": "retrospective_or_unknown",
                   "historical_causality_certified": False,
                   "independence": "historical development/internal validation; no independent final test",
                   "start": start.isoformat(), "end_exclusive": end.isoformat(),
                   "warmup_start": contract.input_start.isoformat(), "markets": sources,
                   "final_test_started": False}
    return MarketInputs(panel, frames, funding, universe, diagnostics)


def _coverage_by_symbol(manifest: ResearchDatasetManifest, dataset: str) -> dict[str, dict]:
    return {row["symbol"]: row for row in manifest.document["coverage"] if row["dataset"] == dataset}


def _price_rows_by_symbol(manifest: ResearchDatasetManifest) -> dict[tuple[str, str], dict]:
    summary = manifest.verification["price_row_provenance_summary"]
    require(isinstance(summary, dict) and set(summary) == {"prices", "funding_rate_event_rows"}
            and isinstance(summary["prices"], list),
            "verified price row provenance summary has an invalid schema")
    return {(row["symbol"], row["dataset"]): row for row in summary["prices"]}


def _expand_holes(coverage: dict, start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    holes = set()
    for item in coverage["holes"]:
        first = _manifest_timestamp(item["start"], "coverage hole start")
        last = _manifest_timestamp(item["end_exclusive"], "coverage hole end")
        clipped_start, clipped_end = max(first, start), min(last, end)
        if clipped_start < clipped_end:
            holes.update(pd.date_range(clipped_start, clipped_end, freq="h", inclusive="left").to_list())
    return holes


def _load_spot_source_rows(store: MarketDataStore, contract: ResearchContract,
                           symbols: list[str], manifest: ResearchDatasetManifest) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start, end = contract.input_start, contract.bounds[1]
    expected_index = pd.date_range(start, end, freq="h", inclusive="left", name="timestamp")
    coverage = _coverage_by_symbol(manifest, "spot_trade")
    source_summary = _price_rows_by_symbol(manifest)
    frames = {}
    partial_records = []
    for symbol in symbols:
        pair = resolve_market_symbols(symbol)
        frame = store.load_bars(SPOT, pair.spot, interval="1h", price_type="trade",
                                start=start, end=end - pd.Timedelta(milliseconds=1), derive=False)
        frame.index = frame.index.as_unit("ns")
        require(frame.index.is_unique and frame.index.is_monotonic_increasing
                and frame.index.isin(expected_index).all(),
                f"spot source rows fall outside the contract grid: {symbol}")
        require(source_summary[(symbol, "spot_trade")]["provenance_rows"] == len(frame),
                f"spot row provenance count differs from loaded archive rows: {symbol}")
        expected_close = frame.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        require((frame.close_time >= frame.index).all(), f"spot close_time precedes its source hour: {symbol}")
        partial = frame.close_time != expected_close
        observed = set(frame.index)
        complete = set(frame.index[~partial])
        actual_holes = set(expected_index) - complete
        manifest_holes = _expand_holes(coverage[symbol], start, end)
        require(actual_holes == manifest_holes,
                f"spot missing/partial hour mask differs from manifest coverage: {symbol}")

        partial_by_time = {
            _manifest_timestamp(row["open_time"], "partial spot bar"): row
            for row in coverage[symbol]["partial_bars"]
            if start <= _manifest_timestamp(row["open_time"], "partial spot bar") < end
        }
        actual_partial_times = set(frame.index[partial])
        require(actual_partial_times == set(partial_by_time),
                f"spot partial source rows differ from manifest coverage: {symbol}")
        for timestamp in frame.index[partial]:
            row = frame.loc[timestamp]
            expected_close_time = timestamp + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
            source_close_time = pd.Timestamp(row.close_time).tz_convert("UTC")
            manifest_partial = partial_by_time[timestamp]
            require(source_close_time == _manifest_timestamp(
                        manifest_partial["observed_close_time"], "manifest partial close_time")
                    and expected_close_time == _manifest_timestamp(
                        manifest_partial["expected_close_time"], "manifest expected close_time")
                    and source_close_time != expected_close_time,
                    f"spot partial close_time differs from the nominal boundary: {symbol}/{timestamp}")
            partial_records.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "observed_close_time": source_close_time,
                "expected_close_time": expected_close_time,
            })
        frames[symbol] = frame
    partial_rows = pd.DataFrame(partial_records,
                                columns=["timestamp", "symbol", "observed_close_time", "expected_close_time"])
    return frames, partial_rows


def _mask_partial_spot_rows(panel: FactorInputPanel, partial_rows: pd.DataFrame) -> None:
    if partial_rows.empty:
        return
    spot_columns = [column for column in panel.values.columns if column.startswith("spot_")]
    diagnostics = panel.diagnostics
    for symbol, group in partial_rows.groupby("symbol", sort=True):
        row_indices = pd.MultiIndex.from_arrays([group["timestamp"], group["symbol"]],
                                                names=panel.values.index.names)
        eligible = panel.universe.reindex(row_indices)
        for column in spot_columns:
            before = panel.values.loc[row_indices, column].notna()
            all_count = int(before.sum())
            eligible_count = int((before.to_numpy() & eligible.to_numpy(dtype=bool)).sum())
            panel.values.loc[row_indices, column] = np.nan
            coverage = diagnostics["symbols"][symbol]["coverage"][column]
            coverage["valid_rows"] -= all_count
            coverage["coverage_ratio"] = coverage["valid_rows"] / coverage["rows"]
            coverage["eligible_valid_rows"] -= eligible_count
            eligible_rows = coverage["eligible_rows"]
            coverage["eligible_coverage_ratio"] = (
                coverage["eligible_valid_rows"] / eligible_rows if eligible_rows else None
            )
            reasons = coverage["missing_reasons"]
            reasons["partial_source_hour_close_time"] = (
                reasons.get("partial_source_hour_close_time", 0) + all_count
            )
        diagnostics["symbols"][symbol]["partial_source_hours_masked"] = len(group)
    diagnostics["partial_source_hours_masked"] = int(len(partial_rows))


def load_research_inputs(dataset_manifest_path: Path, contract: ResearchContract,
                         base_dir: Path) -> ResearchMarketInputs:
    """Load raw-archive research inputs from one isolated manifest/database bundle."""
    require(type(contract) is ResearchContract,
            "load_research_inputs requires a schema-version-2 ResearchContract")
    manifest = ResearchDatasetManifest.from_path(dataset_manifest_path, contract, base_dir)
    require(manifest.database_path is not None and manifest.verification is not None,
            "research dataset manifest did not complete bundle verification")
    universe = load_universe(resolve_input_path(contract.universe, base_dir), contract)
    symbols = list(universe.index.get_level_values("symbol").unique())
    require(set(symbols) == set(manifest.document["symbols"]),
            "contract universe symbols differ from the frozen research dataset")
    store = MarketDataStore(manifest.database_path)
    funding_start = _stage_funding_start(contract, manifest.document)
    panel = load_factor_inputs(_ResearchInputWindowStore(store, contract, funding_start), universe)
    spot_frames, partial_spot_rows = _load_spot_source_rows(store, contract, symbols, manifest)
    _mask_partial_spot_rows(panel, partial_spot_rows)

    input_index = pd.date_range(contract.input_start, contract.bounds[1], freq="h", inclusive="left",
                                name="timestamp")
    account_index = pd.date_range(contract.bounds[0] - pd.Timedelta(hours=1), contract.bounds[1],
                                  freq="h", inclusive="left")
    source_summary = _price_rows_by_symbol(manifest)
    frames, sources = {}, {}
    funding_events = []
    for symbol in symbols:
        pair = resolve_market_symbols(symbol)
        trade_all = _bars(store, pair.perpetual, "trade", input_index)
        mark_all = _bars(store, pair.perpetual, "mark", input_index)
        require(source_summary[(symbol, "perpetual_trade")]["provenance_rows"] == len(trade_all)
                and source_summary[(symbol, "mark_price")]["provenance_rows"] == len(mark_all),
                f"perpetual execution row provenance differs from source grids: {symbol}")
        trade = trade_all.reindex(account_index)
        mark = mark_all.reindex(account_index)
        frames[symbol] = pd.DataFrame({
            "open": trade.open / pair.perpetual_multiplier,
            "close": trade.close / pair.perpetual_multiplier,
            "mark_close": mark.close / pair.perpetual_multiplier,
        }, index=account_index)
        events, funding_source = _funding(store, symbol, pair.perpetual,
                                         *contract.bounds, pair.perpetual_multiplier)
        events = events.sort_values("timestamp").reset_index(drop=True)
        marks = manifest.verification["funding_mark_provenance"]
        native_events = marks.loc[
            (marks["symbol"] == pair.perpetual)
            & (marks["timestamp"] >= contract.bounds[0])
            & (marks["timestamp"] < contract.bounds[1])
        ].sort_values("timestamp").reset_index(drop=True)
        require(len(native_events) == len(events),
                f"funding event mark provenance count differs from account events: {symbol}")
        require(events["timestamp"].equals(native_events["timestamp"])
                and np.isclose(events["funding_rate"].to_numpy(dtype=float),
                               native_events["funding_rate"].to_numpy(dtype=float), rtol=1e-14, atol=1e-14).all()
                and np.isclose(events["mark_price"].to_numpy(dtype=float),
                               native_events["proxy_price"].to_numpy(dtype=float)
                               / pair.perpetual_multiplier, rtol=1e-14, atol=1e-12).all(),
                f"funding account rates/events differ from the frozen minute-mark proxy: {symbol}")
        events_by_symbol = events
        sources[symbol] = {"native_perpetual": pair.perpetual,
                           "base_multiplier": pair.perpetual_multiplier,
                           "trade_rows_input_window": len(trade_all), "mark_rows_input_window": len(mark_all),
                           "spot_source_rows_input_window": len(spot_frames[symbol]),
                           "spot_partial_source_hours_masked": int(
                               (partial_spot_rows["symbol"] == symbol).sum()
                           ),
                           "funding": funding_source}
        funding_events.append(events_by_symbol)
    funding = pd.concat(funding_events, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    document = manifest.document
    verification = manifest.verification
    diagnostics = {
        "source_database": str(manifest.database_path),
        "dataset_id": document["dataset_id"],
        "dataset_manifest_sha256": _manifest_sha256(manifest.path),
        "verified_source_archive_count": len(document["source_files"]),
        "database_signature_verified": True,
        "source_causality": document["source_causality"],
        "source_causality_notes": document["source_causality_notes"],
        "historical_causality_certified": document["historical_causality_certified"],
        "causality_scope": "raw archive close times and row provenance; funding event marks use the declared prior-minute close proxy; publication/receipt timestamps are unavailable",
        "independence": "historical A+B development and C internal validation; none is an untouched final test",
        "purpose": contract.purpose,
        "stage": contract.stage,
        "prior_data_use": contract.prior_data_use,
        "declared_data_processing": contract.data_processing,
        "row_repair_mask": "market_row_provenance maps source rows; no price/rate interpolation or primary database fallback",
        "repair_state": document["repair_state"],
        "spot_feature_missing_rows_manifest": document["spot_feature_missing_rows"],
        "partial_spot_source_hours_masked": len(partial_spot_rows),
        "funding_mark": document["funding_mark"],
        "funding_mark_provenance_events_in_window": len(verification["funding_mark_provenance"]),
        "start": contract.start,
        "end_exclusive": contract.end,
        "warmup_start": contract.input_start.isoformat(),
        "markets": sources,
        "final_test_started": False,
    }
    if document["schema_version"] == 2:
        supplements = document["supplement_source_files"]
        diagnostics.update({
            "daily_mark_supplement_file_count": len(supplements),
            "daily_hourly_mark_rows_restored": sum(
                entry["selected_row_count"] for entry in supplements if entry["interval"] == "1h"
            ),
            "daily_minute_rows_for_funding_proxies": sum(
                entry["selected_row_count"] for entry in supplements if entry["interval"] == "1m"
            ),
        })
    return ResearchMarketInputs(
        panel=panel, frames=frames, funding=funding, universe=universe, diagnostics=diagnostics,
        dataset_manifest=document,
        funding_mark_provenance=verification["funding_mark_provenance"],
        partial_spot_source_rows=partial_spot_rows,
        price_row_provenance_summary=verification["price_row_provenance_summary"],
    )
