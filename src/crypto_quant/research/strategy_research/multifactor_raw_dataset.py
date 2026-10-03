"""Build an isolated multi-asset panel from official Binance Vision archives.

The source archives are retained verbatim. Hourly bars are parsed from the
official 1h files; missing mark-price hours may be reconstructed only from a
complete set of sixty observed 1m mark bars. Funding settlement marks use the
latest 1m mark close whose close time is no later than the settlement time.
That value is an explicit cost proxy, not Binance's native event mark price.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import shutil
import zipfile
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Iterable, NamedTuple

import pandas as pd
import requests

from crypto_quant.data_access.futures_backfill import (
    _csv_rows,
    _epoch_ms,
    _parse_funding,
    _parse_price,
)


DATASET_ID = "multifactor_raw_20261002_v1"
SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
)
FIRST_MONTH = "2022-07"
LAST_MONTH = "2026-07"
INPUT_START = pd.Timestamp("2022-07-24T23:00:00Z")
FUNDING_START = INPUT_START - pd.Timedelta(hours=168)
DATA_END = pd.Timestamp("2026-08-01T00:00:00Z")
WARMUP_HOURS = 168
HOUR_MS = 3_600_000
MINUTE_MS = 60_000
MAX_FUNDING_MARK_AGE_MS = MINUTE_MS
MAX_WORKERS = 4
VISION_ROOT = "https://data.binance.vision"
_HTTP_LOCAL = threading.local()
SOURCE_V1 = "Binance Vision official public monthly archives with explicitly named BTC/ETH local raw cache"
SOURCE_V2 = "Binance Vision official monthly archives and targeted official daily mark-price archives with explicitly named BTC/ETH local raw funding/minute cache"
SOURCE_CAUSALITY = "original_archives_with_declared_funding_proxy"
DAILY_SUPPLEMENT_METHOD = "official_binance_daily_mark_price_archives"
FIELD_SOURCES_V1 = {
    "spot_*": "Binance Vision spot/monthly/klines 1h rows",
    "perpetual_*": "Binance Vision futures/um/monthly/klines 1h rows",
    "mark_*": "Binance Vision futures/um/monthly/markPriceKlines 1h rows; missing hours may use an exactly complete set of 60 source 1m rows",
    "funding_rate": "Binance Vision futures/um/monthly/fundingRate rows; original rate and interval columns",
    "funding_mark_price": "Previous completed Binance Vision 1m mark bar close proxy; no native event markPrice field is present in the source funding archives",
}
FIELD_SOURCES_V2 = {
    **FIELD_SOURCES_V1,
    "mark_*": "Binance Vision futures/um monthly and daily markPriceKlines observed rows; complete 60-minute aggregation only, no interpolation",
}


class ArchiveSpec(NamedTuple):
    archive_id: int
    symbol: str
    month: str
    market: str
    category: str
    interval: str
    source_key: str
    source_url: str
    cache_relative_path: str | None


class MinuteBar(NamedTuple):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    row_number: int


class DailyMarkBars(NamedTuple):
    archive_id: int
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    record: dict[str, Any]
    hourly_rows: list[tuple[Any, ...]]
    row_numbers_by_time: dict[int, int]
    minute_bars: list[MinuteBar]


def _months() -> list[str]:
    return [stamp.strftime("%Y-%m") for stamp in pd.date_range(
        FIRST_MONTH + "-01", LAST_MONTH + "-01", freq="MS"
    )]


def _archive_specs() -> list[ArchiveSpec]:
    specs: list[ArchiveSpec] = []
    next_id = 1
    for month in _months():
        for symbol in SYMBOLS:
            year_month = month
            filename = f"{symbol}-1h-{year_month}.zip"
            entries = (
                ("spot", "klines", "1h", f"data/spot/monthly/klines/{symbol}/1h/{filename}"),
                ("usd_m_perpetual", "klines", "1h", f"data/futures/um/monthly/klines/{symbol}/1h/{filename}"),
                ("usd_m_perpetual", "markPriceKlines", "1h",
                 f"data/futures/um/monthly/markPriceKlines/{symbol}/1h/{filename}"),
                ("usd_m_perpetual", "fundingRate", "native",
                 f"data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year_month}.zip"),
                ("usd_m_perpetual", "markPriceKlines", "1m",
                 f"data/futures/um/monthly/markPriceKlines/{symbol}/1m/{symbol}-1m-{year_month}.zip"),
            )
            for market, category, interval, key in entries:
                cache_relative_path = None
                if symbol in {"BTCUSDT", "ETHUSDT"} and category == "fundingRate":
                    cache_relative_path = f"{symbol}/{symbol}-fundingRate-{year_month}.zip"
                elif symbol in {"BTCUSDT", "ETHUSDT"} and category == "markPriceKlines" and interval == "1m":
                    cache_relative_path = (
                        f"mark_price_klines/{symbol}/1m/{symbol}-1m-{year_month}.zip"
                    )
                specs.append(ArchiveSpec(
                    next_id, symbol, year_month, market, category, interval,
                    key, f"{VISION_ROOT}/{key}", cache_relative_path,
                ))
                next_id += 1
    return specs


def _read_local_cache(cache_root: Path, spec: ArchiveSpec) -> bytes:
    if spec.cache_relative_path is None:
        raise ValueError("archive is not assigned to the local raw cache")
    cache_root = cache_root.resolve()
    source = cache_root / spec.cache_relative_path
    resolved = source.resolve(strict=True)
    if not resolved.is_relative_to(cache_root):
        raise ValueError(f"local raw archive escapes its cache root: {source}")
    return resolved.read_bytes()


def _download_one(spec: ArchiveSpec) -> bytes | None:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        # requests.Session defaults to zero retries; retain the connection for
        # subsequent files handled by this worker thread.
        session = requests.Session()
        _HTTP_LOCAL.session = session
    response = session.get(spec.source_url, timeout=45)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return bytes(response.content)


def _archive_path(output_dir: Path, source_key: str) -> Path:
    return output_dir / "raw" / source_key


def _fetch_archive(
    spec: ArchiveSpec,
    output_dir: Path,
    existing_raw_root: Path,
) -> dict[str, Any]:
    if spec.cache_relative_path is not None:
        payload = _read_local_cache(existing_raw_root, spec)
        source_kind = "existing_local_raw_cache"
    else:
        payload = _download_one(spec)
        source_kind = "official_public_download"

    if payload is None:
        return {
            "archive_id": spec.archive_id,
            "source_key": spec.source_key,
            "source_url": spec.source_url,
            "raw_path": None,
            "market": spec.market,
            "category": spec.category,
            "symbol": spec.symbol,
            "interval": spec.interval,
            "month": spec.month,
            "source_kind": source_kind,
            "sha256": None,
            "byte_size": 0,
            "raw_row_count": 0,
            "row_count": 0,
            "selected_row_count": 0,
            "issue_count": 0,
            "status": "archive_missing_404",
            "csv_member": None,
        }

    # Validate the ZIP before retaining it as a source artifact.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"expected one CSV in {spec.source_key}, got {csv_members}")
        csv_member = csv_members[0]

    destination = _archive_path(output_dir, spec.source_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return {
        "archive_id": spec.archive_id,
        "source_key": spec.source_key,
        "source_url": spec.source_url,
        "raw_path": str(destination.relative_to(output_dir)),
        "market": spec.market,
        "category": spec.category,
        "symbol": spec.symbol,
        "interval": spec.interval,
        "month": spec.month,
        "source_kind": source_kind,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "raw_row_count": 0,
        "row_count": 0,
        "selected_row_count": 0,
        "issue_count": 0,
        "status": "source_retained",
        "csv_member": csv_member,
    }


def _fetch_archives(
    specs: list[ArchiveSpec], output_dir: Path, existing_raw_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    progress_path = output_dir / "progress.json"
    _write_progress(progress_path, {"stage": "downloading_archives",
                                    "archives_completed": 0,
                                    "archives_total": len(specs),
                                    "last_archive_key": None})
    # Submit at most four ZIPs at once; decompressed minute files are handled
    # later one symbol-month at a time.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for offset in range(0, len(specs), MAX_WORKERS):
            batch = specs[offset:offset + MAX_WORKERS]
            futures = [pool.submit(_fetch_archive, spec, output_dir, existing_raw_root)
                       for spec in batch]
            batch_records = [future.result() for future in futures]
            records.extend(batch_records)
            _write_progress(progress_path, {"stage": "downloading_archives",
                                            "archives_completed": len(records),
                                            "archives_total": len(specs),
                                            "last_archive_key": batch_records[-1]["source_key"]})
    return records


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _source_rows(raw: bytes) -> tuple[list[list[str]], bool, dict[int, int]]:
    rows = _csv_rows(raw)
    if not rows:
        return rows, False, {}
    has_header = not rows[0] or not rows[0][0].strip().lstrip("-").isdigit()
    open_time_to_line: dict[int, int] = {}
    for line_number, row in enumerate(rows, start=1):
        if has_header and line_number == 1:
            continue
        if not row:
            continue
        try:
            open_time = _epoch_ms(row[0])
        except (TypeError, ValueError, OverflowError):
            continue
        if open_time in open_time_to_line:
            raise ValueError(f"duplicate source timestamp {open_time}")
        open_time_to_line[open_time] = line_number
    return rows, has_header, open_time_to_line


def _source_row_count(rows: list[list[str]], has_header: bool) -> int:
    return max(0, len(rows) - int(has_header))


def _issue_rows(issues: list[dict[str, Any]], has_header: bool) -> list[dict[str, Any]]:
    if has_header:
        return issues
    return [{**item, "row_number": max(1, int(item["row_number"]) - 1)} for item in issues]


def _parse_spot_hourly(raw: bytes, symbol: str) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]], int]:
    """Parse one original Binance spot 1h ZIP and preserve physical CSV lines."""
    rows, has_header, _ = _source_rows(raw)
    start = 1 if has_header else 0
    output: list[tuple[Any, ...]] = []
    issues: list[dict[str, Any]] = []
    seen_times: set[int] = set()
    for position, row in enumerate(rows[start:], start=start + 1):
        stamp: int | None = None
        try:
            if len(row) < 12:
                raise ValueError("short spot kline row")
            stamp = _epoch_ms(row[0])
            close_time = _epoch_ms(row[6])
            values = [float(row[index]) for index in (1, 2, 3, 4, 5, 7, 9, 10)]
            open_price, high, low, close, volume, quote_volume, taker_base, taker_quote = values
            trades = int(float(row[8]))
            if stamp < 0 or stamp % HOUR_MS:
                raise ValueError("unaligned spot 1h open_time")
            if close_time <= stamp or close_time > stamp + HOUR_MS + 5_000:
                raise ValueError("invalid spot 1h close_time")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite spot kline value")
            if min(open_price, high, low, close) <= 0:
                raise ValueError("non-positive spot price")
            if high < max(open_price, close, low) or low > min(open_price, close, high):
                raise ValueError("invalid spot OHLC range")
            if min(volume, quote_volume, taker_base, taker_quote) < 0 or trades < 0:
                raise ValueError("negative spot volume or trade count")
            if taker_base > volume + abs(volume) * 1e-12:
                raise ValueError("spot taker buy base volume exceeds volume")
            if taker_quote > quote_volume + abs(quote_volume) * 1e-12:
                raise ValueError("spot taker buy quote volume exceeds quote volume")
            if stamp in seen_times:
                raise ValueError(f"duplicate spot 1h open_time: {stamp}")
            seen_times.add(stamp)
            output.append((symbol, "1h", stamp, open_price, high, low, close, volume,
                           close_time, quote_volume, trades, taker_base, taker_quote))
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            issues.append({"row_number": position, "open_time": stamp,
                           "reason": str(exc), "raw_preview": ",".join(row[:12])[:500]})
    return output, issues, _source_row_count(rows, has_header)


def _parse_minute_mark(raw: bytes, symbol: str) -> tuple[list[MinuteBar], list[dict[str, Any]], int]:
    rows, has_header, _ = _source_rows(raw)
    start = 1 if has_header else 0
    output: list[MinuteBar] = []
    issues: list[dict[str, Any]] = []
    seen_times: set[int] = set()
    for position, row in enumerate(rows[start:], start=start + 1):
        stamp: int | None = None
        try:
            if len(row) < 12:
                raise ValueError("short mark-price 1m row")
            stamp = _epoch_ms(row[0])
            close_time = _epoch_ms(row[6])
            values = [float(row[index]) for index in (1, 2, 3, 4, 5, 7, 9, 10)]
            open_price, high, low, close, volume, quote_volume, taker_base, taker_quote = values
            trades = int(float(row[8]))
            if stamp < 0 or stamp % MINUTE_MS:
                raise ValueError("unaligned mark-price 1m open_time")
            if close_time <= stamp or close_time > stamp + MINUTE_MS + 5_000:
                raise ValueError("invalid mark-price 1m close_time")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite mark-price 1m value")
            if min(open_price, high, low, close) <= 0:
                raise ValueError("non-positive mark price")
            if high < max(open_price, close, low) or low > min(open_price, close, high):
                raise ValueError("invalid mark-price OHLC range")
            if min(volume, quote_volume, taker_base, taker_quote) < 0 or trades < 0:
                raise ValueError("negative mark-price volume or trade count")
            if stamp in seen_times:
                raise ValueError(f"duplicate mark-price 1m open_time: {stamp}")
            seen_times.add(stamp)
            output.append(MinuteBar(stamp, open_price, high, low, close, volume,
                                    close_time, quote_volume, trades, taker_base,
                                    taker_quote, position))
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            issues.append({"row_number": position, "open_time": stamp,
                           "reason": str(exc), "raw_preview": ",".join(row[:12])[:500]})
    output.sort(key=lambda item: item.open_time)
    return output, issues, _source_row_count(rows, has_header)


def _db_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA user_version=1")
    conn.executescript(
        """
        CREATE TABLE source_archives (
            id INTEGER PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE,
            source_url TEXT NOT NULL,
            raw_path TEXT,
            market TEXT NOT NULL,
            category TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            month TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            sha256 TEXT,
            byte_size INTEGER NOT NULL,
            raw_row_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            selected_row_count INTEGER NOT NULL,
            issue_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            csv_member TEXT
        );
        CREATE TABLE klines (
            symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time INTEGER NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
            volume REAL NOT NULL, close_time INTEGER NOT NULL, quote_volume REAL NOT NULL,
            trades INTEGER NOT NULL, taker_buy_base_volume REAL NOT NULL,
            taker_buy_quote_volume REAL NOT NULL,
            PRIMARY KEY(symbol, interval, open_time)
        ) WITHOUT ROWID;
        CREATE TABLE futures_archive_files (
            id INTEGER PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            etag TEXT,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            issue_count INTEGER NOT NULL,
            processed_at TEXT NOT NULL,
            FOREIGN KEY(id) REFERENCES source_archives(id)
        );
        CREATE TABLE futures_price_bars (
            data_type TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            open_time INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
            close_time INTEGER NOT NULL, quote_volume REAL NOT NULL, trades INTEGER NOT NULL,
            taker_buy_base_volume REAL NOT NULL, taker_buy_quote_volume REAL NOT NULL,
            source_file_id INTEGER NOT NULL,
            PRIMARY KEY(data_type, symbol, interval, open_time),
            FOREIGN KEY(source_file_id) REFERENCES futures_archive_files(id)
        ) WITHOUT ROWID;
        CREATE TABLE futures_funding_rates (
            symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_interval_hours INTEGER NOT NULL, funding_rate REAL NOT NULL,
            source_file_id INTEGER NOT NULL, mark_price REAL,
            PRIMARY KEY(symbol, funding_time),
            FOREIGN KEY(source_file_id) REFERENCES futures_archive_files(id)
        ) WITHOUT ROWID;
        CREATE TABLE market_row_provenance (
            target_table TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
            observed_time INTEGER NOT NULL, source_file_id INTEGER NOT NULL,
            source_interval TEXT NOT NULL, source_row_numbers_json TEXT NOT NULL,
            transformation TEXT NOT NULL,
            PRIMARY KEY(target_table, symbol, interval, observed_time),
            FOREIGN KEY(source_file_id) REFERENCES source_archives(id)
        ) WITHOUT ROWID;
        CREATE TABLE source_archive_issues (
            source_file_id INTEGER NOT NULL, row_number INTEGER,
            observed_time INTEGER, reason TEXT NOT NULL, raw_preview TEXT,
            FOREIGN KEY(source_file_id) REFERENCES source_archives(id)
        );
        CREATE TABLE funding_mark_provenance (
            symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            native_event_mark_price REAL, proxy_price REAL,
            proxy_method TEXT NOT NULL, source_file_id INTEGER,
            source_row_number INTEGER, source_open_time INTEGER, source_close_time INTEGER,
            age_ms INTEGER, status TEXT NOT NULL,
            PRIMARY KEY(symbol, funding_time),
            FOREIGN KEY(source_file_id) REFERENCES source_archives(id)
        ) WITHOUT ROWID;
        CREATE INDEX idx_market_rows_time ON futures_price_bars(symbol, interval, open_time);
        CREATE INDEX idx_funding_time ON futures_funding_rates(symbol, funding_time);
        """
    )
    return conn


def _insert_archive_metadata(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
    processed_at = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO source_archives VALUES
           (:archive_id,:source_key,:source_url,:raw_path,:market,:category,:symbol,
            :interval,:month,:source_kind,:sha256,:byte_size,:raw_row_count,
            :row_count,:selected_row_count,:issue_count,:status,:csv_member)""",
        records,
    )
    conn.executemany(
        """INSERT INTO futures_archive_files
           (id,source_key,category,symbol,interval,byte_size,etag,sha256,row_count,
            issue_count,processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (r["archive_id"], r["source_key"], r["category"], r["symbol"],
             r["interval"], r["byte_size"], None, r["sha256"], 0, 0, processed_at)
            for r in records
            if r["market"] == "usd_m_perpetual" and r["sha256"] is not None
        ],
    )


def _read_archive_payload(output_dir: Path, record: dict[str, Any]) -> bytes | None:
    relative = record["raw_path"]
    if relative is None:
        return None
    path = output_dir / relative
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"retained archive hash changed: {relative}")
    return payload


def _record_issues(conn: sqlite3.Connection, archive_id: int,
                   issues: Iterable[dict[str, Any]]) -> None:
    conn.executemany(
        "INSERT INTO source_archive_issues VALUES (?,?,?,?,?)",
        [(archive_id, issue.get("row_number"), issue.get("open_time"),
          issue["reason"], issue.get("raw_preview")) for issue in issues],
    )


def _add_provenance(conn: sqlite3.Connection, target_table: str, symbol: str,
                    interval: str, observed_time: int, archive_id: int,
                    source_interval: str, row_numbers: Iterable[int],
                    transformation: str) -> None:
    conn.execute(
        """INSERT INTO market_row_provenance VALUES (?,?,?,?,?,?,?,?)""",
        (target_table, symbol, interval, observed_time, archive_id, source_interval,
         json.dumps(list(row_numbers), separators=(",", ":")), transformation),
    )


def _inside_window(open_time: int) -> bool:
    return int(INPUT_START.value // 1_000_000) <= open_time < int(DATA_END.value // 1_000_000)


def _parse_hourly_archives(output_dir: Path, records: list[dict[str, Any]],
                           conn: sqlite3.Connection, progress_path: Path) -> None:
    input_start_ms = int(INPUT_START.value // 1_000_000)
    funding_start_ms = int(FUNDING_START.value // 1_000_000)
    data_end_ms = int(DATA_END.value // 1_000_000)
    parsed_count = 0
    relevant_records = [r for r in records if not (
        r["category"] == "markPriceKlines" and r["interval"] == "1m"
    )]
    for record in records:
        if record["status"] == "archive_missing_404":
            if record in relevant_records:
                parsed_count += 1
                _write_progress(progress_path, {"stage": "parsing_hourly_and_funding",
                                                "archives_completed": parsed_count,
                                                "archives_total": len(relevant_records),
                                                "last_archive_key": record["source_key"]})
            continue
        if record["interval"] == "1m":
            continue
        if record["category"] == "markPriceKlines" and record["interval"] == "1h":
            parser_kind = "um_price"
        elif record["category"] == "klines" and record["market"] == "usd_m_perpetual":
            parser_kind = "um_price"
        elif record["category"] == "klines" and record["market"] == "spot":
            parser_kind = "spot_price"
        elif record["category"] == "fundingRate":
            parser_kind = "funding"
        else:
            raise ValueError(f"unknown archive category: {record}")

        payload = _read_archive_payload(output_dir, record)
        assert payload is not None
        csv_rows, has_header, row_map = _source_rows(payload)
        record["raw_row_count"] = _source_row_count(csv_rows, has_header)
        if parser_kind == "um_price":
            parsed, issues = _parse_price(record["category"], record["symbol"], payload)
            issues = _issue_rows(issues, has_header)
            selected = [row for row in parsed if _inside_window(int(row[3]))]
            if len({int(row[3]) for row in parsed}) != len(parsed):
                raise ValueError(f"duplicate parsed hourly timestamps in {record['source_key']}")
            conn.executemany(
                "INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(*row, record["archive_id"]) for row in selected],
            )
            for row in selected:
                _add_provenance(conn, f"futures_price_bars:{record['category']}",
                                record["symbol"], "1h",
                                int(row[3]), record["archive_id"], "1h",
                                [row_map[int(row[3])]], "official_archive_row")
        elif parser_kind == "spot_price":
            parsed, issues, source_rows = _parse_spot_hourly(payload, record["symbol"])
            record["raw_row_count"] = source_rows
            selected = [row for row in parsed if _inside_window(int(row[2]))]
            conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", selected)
            for row in selected:
                _add_provenance(conn, "klines", record["symbol"], "1h", int(row[2]),
                                record["archive_id"], "1h", [row_map[int(row[2])]],
                                "official_archive_row")
        else:
            parsed, issues = _parse_funding(record["symbol"], payload)
            issues = _issue_rows(issues, has_header)
            selected = [row for row in parsed
                        if funding_start_ms <= int(row[1]) < data_end_ms]
            conn.executemany(
                """INSERT INTO futures_funding_rates
                 (symbol,funding_time,funding_interval_hours,funding_rate,source_file_id,mark_price)
                 VALUES (?,?,?,?,?,NULL)""",
                [(row[0], int(row[1]), int(row[2]), float(row[3]), record["archive_id"])
                 for row in selected],
            )
            for row in selected:
                _add_provenance(conn, "futures_funding_rates", record["symbol"], "native",
                                int(row[1]), record["archive_id"], "native",
                                [row_map[int(row[1])]], "official_funding_event_row")

        record["row_count"] = len(parsed)
        record["selected_row_count"] = len(selected)
        record["issue_count"] = len(issues)
        if issues:
            _record_issues(conn, record["archive_id"], issues)
        record["status"] = "parsed_with_issues" if issues else "complete"
        if record["market"] == "usd_m_perpetual":
            conn.execute(
                "UPDATE futures_archive_files SET row_count=?, issue_count=? WHERE id=?",
                (len(parsed), len(issues), record["archive_id"]),
            )
        parsed_count += 1
        _write_progress(progress_path, {"stage": "parsing_hourly_and_funding",
                                        "archives_completed": parsed_count,
                                        "archives_total": len(relevant_records),
                                        "last_archive_key": record["source_key"]})


def _aggregate_hour(hour_open: int, bars: list[MinuteBar], archive_id: int,
                    conn: sqlite3.Connection, symbol: str) -> bool:
    bars = sorted(bars, key=lambda item: item.open_time)
    expected = [hour_open + i * MINUTE_MS for i in range(60)]
    if [bar.open_time for bar in bars] != expected:
        return False
    if any(bar.close_time != bar.open_time + MINUTE_MS - 1 for bar in bars):
        return False
    if bars[-1].close_time != hour_open + HOUR_MS - 1:
        return False
    if any(right.row_number != left.row_number + 1 for left, right in zip(bars[:-1], bars[1:])):
        return False
    values = (
        "markPriceKlines", symbol, "1h", hour_open,
        bars[0].open, max(bar.high for bar in bars), min(bar.low for bar in bars),
        bars[-1].close, sum(bar.volume for bar in bars), bars[-1].close_time,
        sum(bar.quote_volume for bar in bars), sum(bar.trades for bar in bars),
        sum(bar.taker_buy_base_volume for bar in bars),
        sum(bar.taker_buy_quote_volume for bar in bars), archive_id,
    )
    conn.execute("INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
    _add_provenance(conn, "futures_price_bars:markPriceKlines", symbol, "1h", hour_open,
                    archive_id, "1m", (bar.row_number for bar in bars),
                    "aggregate_60_complete_observed_1m_mark_bars")
    return True


def _find_funding_mark_proxy(
    funding_time: int,
    minute_sources: list[tuple[MinuteBar, int]],
) -> tuple[MinuteBar, int, int] | str:
    close_times = [bar.close_time for bar, _ in minute_sources]
    if any(right < left for left, right in zip(close_times, close_times[1:])):
        raise ValueError("minute mark close times are not monotone")
    index = bisect_right(close_times, funding_time) - 1
    if index < 0:
        return "no_prior_completed_mark_minute"
    bar, source_archive_id = minute_sources[index]
    age = funding_time - bar.close_time
    if age < 0 or age > MAX_FUNDING_MARK_AGE_MS:
        return "prior_mark_minute_older_than_60s"
    return bar, source_archive_id, age


def _process_minute_archives(output_dir: Path, records: list[dict[str, Any]],
                             conn: sqlite3.Connection,
                             progress_path: Path) -> dict[str, Any]:
    minute_records = {
        (record["symbol"], record["month"]): record
        for record in records
        if record["category"] == "markPriceKlines" and record["interval"] == "1m"
    }
    input_start_ms = int(INPUT_START.value // 1_000_000)
    data_end_ms = int(DATA_END.value // 1_000_000)
    funding_start_ms = int(FUNDING_START.value // 1_000_000)
    selected_rows: dict[int, set[int]] = {r["archive_id"]: set() for r in records}
    derived_counts: dict[str, int] = {symbol: 0 for symbol in SYMBOLS}
    proxy_rows = 0
    missing_proxy_rows = 0
    proxy_ages: list[int] = []
    minute_count = 0
    minute_total = len(minute_records)

    for symbol in SYMBOLS:
        previous_bar: tuple[MinuteBar, int] | None = None
        for month in _months():
            record = minute_records[(symbol, month)]
            if record["status"] == "archive_missing_404":
                # No missing-mark or funding-mark value is manufactured.
                next_month = (pd.Period(month, freq="M") + 1).start_time.tz_localize("UTC")
                month_start_ms = int(pd.Timestamp(month + "-01", tz="UTC").value // 1_000_000)
                month_end_ms = int(next_month.value // 1_000_000)
                _write_missing_proxies(conn, symbol, month_start_ms, month_end_ms)
                minute_count += 1
                _write_progress(progress_path, {"stage": "processing_mark_1m_archives",
                                                "archives_completed": minute_count,
                                                "archives_total": minute_total,
                                                "last_archive_key": record["source_key"]})
                continue

            payload = _read_archive_payload(output_dir, record)
            assert payload is not None
            bars, issues, raw_count = _parse_minute_mark(payload, symbol)
            record["raw_row_count"] = raw_count
            record["row_count"] = len(bars)
            record["issue_count"] = len(issues)
            record["status"] = "parsed_with_issues" if issues else "complete"
            if issues:
                _record_issues(conn, record["archive_id"], issues)

            groups: dict[int, list[MinuteBar]] = {}
            for bar in bars:
                groups.setdefault((bar.open_time // HOUR_MS) * HOUR_MS, []).append(bar)

            month_start = pd.Timestamp(month + "-01", tz="UTC")
            month_end = month_start + pd.offsets.MonthBegin(1)
            month_start_ms = int(month_start.value // 1_000_000)
            month_end_ms = int(month_end.value // 1_000_000)
            grid_start = max(input_start_ms, month_start_ms)
            grid_end = min(data_end_ms, month_end_ms)

            existing_hours = {
                int(row[0]) for row in conn.execute(
                    """SELECT open_time FROM futures_price_bars
                       WHERE data_type='markPriceKlines' AND symbol=? AND interval='1h'
                         AND open_time>=? AND open_time<?""",
                    (symbol, grid_start, grid_end),
                )
            }
            if grid_start < grid_end:
                for hour_open in range(grid_start, grid_end, HOUR_MS):
                    if hour_open in existing_hours:
                        continue
                    hour_bars = groups.get(hour_open, [])
                    if len(hour_bars) == 60 and _aggregate_hour(
                        hour_open, hour_bars, record["archive_id"], conn, symbol
                    ):
                        derived_counts[symbol] += 1
                        selected_rows[record["archive_id"]].update(
                            bar.row_number for bar in hour_bars
                        )

            # Determine explicit previous-completed-minute mark proxies for
            # this month's original funding event rows.
            events = [int(row[0]) for row in conn.execute(
                """SELECT funding_time FROM futures_funding_rates
                   WHERE symbol=? AND funding_time>=? AND funding_time<?
                   ORDER BY funding_time""",
                (symbol, max(funding_start_ms, month_start_ms), min(data_end_ms, month_end_ms)),
            )]
            prior = previous_bar
            minute_sources = ([(prior[0], prior[1])] if prior else []) + [
                (bar, record["archive_id"]) for bar in bars
            ]
            for funding_time in events:
                candidate = _find_funding_mark_proxy(funding_time, minute_sources)
                if isinstance(candidate, str):
                    _store_proxy_missing(conn, symbol, funding_time, candidate)
                    missing_proxy_rows += 1
                    continue
                bar, source_archive_id, age = candidate
                source_row = bar.row_number
                conn.execute(
                    "UPDATE futures_funding_rates SET mark_price=? WHERE symbol=? AND funding_time=?",
                    (bar.close, symbol, funding_time),
                )
                conn.execute(
                    """INSERT INTO funding_mark_provenance VALUES
                       (?,?,NULL,?,?,?,?,?,?,?,'proxied')""",
                    (symbol, funding_time, bar.close,
                     "previous_completed_1m_mark_close_at_or_before_funding_time",
                     source_archive_id, source_row, bar.open_time, bar.close_time, age),
                )
                proxy_rows += 1
                proxy_ages.append(age)
                selected_rows[source_archive_id].add(source_row)

            if bars:
                last_bar = bars[-1]
                previous_bar = (last_bar, record["archive_id"])
            elif previous_bar is None:
                previous_bar = None

            record["selected_row_count"] = len(selected_rows[record["archive_id"]])
            minute_count += 1
            _write_progress(progress_path, {"stage": "processing_mark_1m_archives",
                                            "archives_completed": minute_count,
                                            "archives_total": minute_total,
                                            "last_archive_key": record["source_key"]})

    for record in records:
        if record["category"] == "markPriceKlines" and record["interval"] == "1m":
            record["selected_row_count"] = len(selected_rows[record["archive_id"]])
    status_counts = {str(status): int(count) for status, count in conn.execute(
        "SELECT status,COUNT(*) FROM funding_mark_provenance GROUP BY status"
    )}
    proxy_rows = status_counts.get("proxied", 0)
    funding_events = int(conn.execute("SELECT COUNT(*) FROM futures_funding_rates").fetchone()[0])
    missing_proxy_rows = funding_events - proxy_rows
    proxy_ages = [int(row[0]) for row in conn.execute(
        "SELECT age_ms FROM funding_mark_provenance WHERE status='proxied' AND age_ms IS NOT NULL"
    )]
    return {
        "proxy_count": proxy_rows,
        "missing_proxy_count": missing_proxy_rows,
        "max_proxy_age_ms": max(proxy_ages) if proxy_ages else None,
        "min_proxy_age_ms": min(proxy_ages) if proxy_ages else None,
        "derived_hourly_mark_rows_by_symbol": derived_counts,
    }


def _store_proxy_missing(conn: sqlite3.Connection, symbol: str, funding_time: int,
                         reason: str) -> None:
    conn.execute(
        """INSERT INTO funding_mark_provenance
           (symbol,funding_time,native_event_mark_price,proxy_price,proxy_method,
            source_file_id,source_row_number,source_open_time,source_close_time,age_ms,status)
           VALUES (?, ?, NULL, NULL,
                   'previous_completed_1m_mark_close_at_or_before_funding_time',
                   NULL,NULL,NULL,NULL,NULL,?)""",
        (symbol, funding_time, reason),
    )


def _write_missing_proxies(conn: sqlite3.Connection, symbol: str,
                           start_ms: int, end_ms: int) -> None:
    rows = conn.execute(
        """SELECT funding_time FROM futures_funding_rates
           WHERE symbol=? AND funding_time>=? AND funding_time<?""",
        (symbol, start_ms, end_ms),
    ).fetchall()
    for (funding_time,) in rows:
        _store_proxy_missing(conn, symbol, int(funding_time), "minute_mark_archive_missing")


def _holes(expected: list[int], observed: set[int]) -> list[dict[str, Any]]:
    missing = [value for value in expected if value not in observed]
    if not missing:
        return []
    result = []
    run_start = previous = missing[0]
    for value in missing[1:]:
        if value != previous + HOUR_MS:
            result.append({"start": pd.to_datetime(run_start, unit="ms", utc=True).isoformat(),
                           "end_exclusive": pd.to_datetime(previous + HOUR_MS, unit="ms", utc=True).isoformat(),
                           "count": ((previous - run_start) // HOUR_MS) + 1})
            run_start = value
        previous = value
    result.append({"start": pd.to_datetime(run_start, unit="ms", utc=True).isoformat(),
                   "end_exclusive": pd.to_datetime(previous + HOUR_MS, unit="ms", utc=True).isoformat(),
                   "count": ((previous - run_start) // HOUR_MS) + 1})
    return result


def _coverage(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    start_ms = int(INPUT_START.value // 1_000_000)
    end_ms = int(DATA_END.value // 1_000_000)
    expected = list(range(start_ms, end_ms, HOUR_MS))
    expected_set = set(expected)
    output = []
    for symbol in SYMBOLS:
        spot_rows = [(int(row[0]), int(row[1])) for row in conn.execute(
            "SELECT open_time,close_time FROM klines WHERE symbol=? AND interval='1h' AND open_time>=? AND open_time<?",
            (symbol, start_ms, end_ms),
        )]
        perp_rows = [(int(row[0]), int(row[1])) for row in conn.execute(
            """SELECT open_time,close_time FROM futures_price_bars
               WHERE data_type='klines' AND symbol=? AND interval='1h'
                 AND open_time>=? AND open_time<?""",
            (symbol, start_ms, end_ms),
        )]
        mark_rows = [(int(row[0]), int(row[1])) for row in conn.execute(
            """SELECT open_time,close_time FROM futures_price_bars
               WHERE data_type='markPriceKlines' AND symbol=? AND interval='1h'
                 AND open_time>=? AND open_time<?""",
            (symbol, start_ms, end_ms),
        )]
        row_sets = (
            ("spot_trade", "klines", spot_rows),
            ("perpetual_trade", "futures_price_bars:klines", perp_rows),
            ("mark_price", "futures_price_bars:markPriceKlines", mark_rows),
        )
        for dataset, target_table, source_rows in row_sets:
            observed = {open_time for open_time, close_time in source_rows
                        if close_time == open_time + HOUR_MS - 1}
            partial_rows = [
                {"open_time": pd.to_datetime(open_time, unit="ms", utc=True).isoformat(),
                 "observed_close_time": pd.to_datetime(close_time, unit="ms", utc=True).isoformat(),
                 "expected_close_time": pd.to_datetime(open_time + HOUR_MS - 1, unit="ms", utc=True).isoformat()}
                for open_time, close_time in source_rows
                if open_time in expected_set and close_time != open_time + HOUR_MS - 1
            ]
            if dataset == "spot_trade":
                source_filter = "a.market='spot' AND a.category='klines' AND a.interval='1h'"
            elif dataset == "perpetual_trade":
                source_filter = "a.market='usd_m_perpetual' AND a.category='klines' AND a.interval='1h'"
            else:
                source_filter = "a.market='usd_m_perpetual' AND a.category='markPriceKlines' AND a.interval='1h'"
            archive_rows = int(conn.execute(
                f"""SELECT COUNT(*) FROM market_row_provenance AS p
                    JOIN source_archives AS a ON a.id=p.source_file_id
                    WHERE p.target_table=? AND p.symbol=? AND p.interval='1h'
                      AND p.observed_time>=? AND p.observed_time<?
                      AND p.transformation='official_archive_row' AND {source_filter}""",
                (target_table, symbol, start_ms, end_ms),
            ).fetchone()[0])
            output.append({"symbol": symbol, "dataset": dataset,
                           "expected_rows": len(expected),
                           "archive_rows": archive_rows,
                           "selected_rows": len(observed & expected_set),
                           "source_observation_rows": len(source_rows),
                           "partial_close_rows": len(partial_rows),
                           "partial_bars": partial_rows,
                           "derived_rows": (int(conn.execute(
            """SELECT COUNT(*) FROM market_row_provenance
                                  WHERE target_table='futures_price_bars:markPriceKlines' AND symbol=?
                                    AND interval='1h' AND observed_time>=? AND observed_time<?
                                    AND transformation='aggregate_60_complete_observed_1m_mark_bars'""",
                               (symbol, start_ms, end_ms),
                           ).fetchone()[0]) if dataset == "mark_price" else 0),
                           "holes": _holes(expected, observed)})
    return output


def _file_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _records_from_retained_archives(output_dir: Path,
                                    specs: list[ArchiveSpec]) -> list[dict[str, Any]]:
    output_root = output_dir.resolve(strict=True)
    if len(specs) != 2_450 or len({spec.source_key for spec in specs}) != 2_450:
        raise ValueError("offline finalization requires the fixed 2,450-archive source plan")
    expected_paths = {Path("raw") / spec.source_key for spec in specs}
    raw_root = output_root / "raw"
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("retained archive root is missing or is a symlink")
    actual_paths = {path.relative_to(output_root) for path in raw_root.rglob("*.zip") if path.is_file()}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(
            f"retained ZIP set differs from the fixed archive plan: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    records = []
    for spec in specs:
        destination = _archive_path(output_root, spec.source_key)
        if destination.is_symlink():
            raise ValueError(f"retained archive is a symlink: {destination}")
        resolved = destination.resolve(strict=True)
        if not resolved.is_relative_to(output_root):
            raise ValueError(f"retained archive escapes the dataset root: {destination}")
        payload = resolved.read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"retained archive CRC failed: {spec.source_key}/{bad_member}")
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(f"expected one CSV in {spec.source_key}, got {csv_members}")
        records.append({
            "archive_id": spec.archive_id,
            "source_key": spec.source_key,
            "source_url": spec.source_url,
            "raw_path": str(destination.relative_to(output_root)),
            "market": spec.market,
            "category": spec.category,
            "symbol": spec.symbol,
            "interval": spec.interval,
            "month": spec.month,
            "source_kind": "existing_local_raw_cache" if spec.cache_relative_path is not None
            else "official_public_download",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
            "raw_row_count": 0,
            "row_count": 0,
            "selected_row_count": 0,
            "issue_count": 0,
            "status": "source_retained",
            "csv_member": csv_members[0],
        })
    return records


def _daily_supplement_specs(manifest: dict[str, Any], first_archive_id: int
                           ) -> tuple[list[ArchiveSpec], list[dict[str, Any]]]:
    gaps = []
    seen = set()
    for item in manifest.get("coverage", []):
        if item.get("dataset") != "mark_price":
            continue
        symbol = item["symbol"]
        for hole in item["holes"]:
            start = pd.Timestamp(hole["start"])
            end = pd.Timestamp(hole["end_exclusive"])
            if (int(hole["count"]) != 24 or start.tzinfo is None or end.tzinfo is None
                    or start != start.floor("D") or end != start + pd.Timedelta(days=1)):
                raise ValueError("daily archive supplement only accepts complete midnight-to-midnight 24h mark gaps")
            day = start.strftime("%Y-%m-%d")
            key = (symbol, day)
            if key in seen:
                raise ValueError(f"duplicate daily mark gap in manifest: {key}")
            seen.add(key)
            gaps.append({"symbol": symbol, "day": day,
                         "start_ms": int(start.value // 1_000_000),
                         "end_ms": int(end.value // 1_000_000)})

    if len(gaps) != 35:
        raise ValueError(f"daily gap supplement expects the audited 35 symbol-days, found {len(gaps)}")
    specs = []
    next_id = first_archive_id
    for gap in gaps:
        symbol, day = gap["symbol"], gap["day"]
        month = day[:7]
        for interval in ("1h", "1m"):
            filename = f"{symbol}-{interval}-{day}.zip"
            source_key = (
                f"data/futures/um/daily/markPriceKlines/{symbol}/{interval}/{filename}"
            )
            specs.append(ArchiveSpec(
                next_id, symbol, month, "usd_m_perpetual", "markPriceKlines", interval,
                source_key, f"{VISION_ROOT}/{source_key}", None,
            ))
            next_id += 1
    if len(specs) != 70 or len({spec.source_key for spec in specs}) != 70:
        raise ValueError("daily mark supplement source plan is not the fixed 70-file set")
    gaps.sort(key=lambda item: (item["symbol"], item["day"]))
    return specs, gaps


def _daily_source_date(source_key: str) -> str:
    stem = Path(source_key).name.removesuffix(".zip")
    date = "-".join(stem.split("-")[-3:])
    return datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")


def _fetch_daily_supplement_one(
    spec: ArchiveSpec,
    staging_root: Path,
) -> tuple[dict[str, Any], bytes]:
    payload = _download_one(spec)
    if payload is None:
        raise FileNotFoundError(f"official daily mark archive returned HTTP 404: {spec.source_url}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"daily mark ZIP CRC failed: {spec.source_key}/{bad_member}")
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"expected one CSV in {spec.source_key}, got {csv_members}")
    destination = staging_root / spec.source_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    record = {
        "archive_id": spec.archive_id,
        "source_key": spec.source_key,
        "source_url": spec.source_url,
        "raw_path": str(Path("raw_supplement") / spec.source_key),
        "market": spec.market,
        "category": spec.category,
        "symbol": spec.symbol,
        "interval": spec.interval,
        "month": spec.month,
        "source_kind": "official_public_daily_archive",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "raw_row_count": 0,
        "row_count": 0,
        "selected_row_count": 0,
        "issue_count": 0,
        "status": "source_retained",
        "csv_member": csv_members[0],
    }
    return record, payload


def _fetch_daily_supplement(specs: list[ArchiveSpec], staging_root: Path,
                            progress_path: Path) -> list[tuple[dict[str, Any], bytes]]:
    output = []
    total = len(specs)
    _write_progress(progress_path, {"stage": "downloading_daily_mark_supplement",
                                    "archives_completed": 0, "archives_total": total,
                                    "last_archive_key": None})
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for offset in range(0, total, MAX_WORKERS):
            batch = specs[offset:offset + MAX_WORKERS]
            futures = [pool.submit(_fetch_daily_supplement_one, spec, staging_root)
                       for spec in batch]
            batch_results = [future.result() for future in futures]
            output.extend(batch_results)
            _write_progress(progress_path, {
                "stage": "downloading_daily_mark_supplement",
                "archives_completed": len(output),
                "archives_total": total,
                "last_archive_key": batch_results[-1][0]["source_key"],
            })
    return output


def _parse_daily_mark_supplement(
    record: dict[str, Any], payload: bytes, gap: dict[str, Any],
) -> DailyMarkBars:
    csv_rows, has_header, row_map = _source_rows(payload)
    raw_count = _source_row_count(csv_rows, has_header)
    start_ms, end_ms = gap["start_ms"], gap["end_ms"]
    if record["interval"] == "1h":
        parsed, issues = _parse_price("markPriceKlines", record["symbol"], payload)
        issues = _issue_rows(issues, has_header)
        expected = list(range(start_ms, end_ms, HOUR_MS))
        if (issues or raw_count != 24 or len(parsed) != 24
                or [int(row[3]) for row in parsed] != expected
                or any(int(row[9]) != int(row[3]) + HOUR_MS - 1 for row in parsed)):
            raise ValueError(f"daily 1h mark source is incomplete or has non-nominal closes: {record['source_key']}")
        record["raw_row_count"] = raw_count
        record["row_count"] = len(parsed)
        record["selected_row_count"] = len(parsed)
        record["issue_count"] = 0
        record["status"] = "complete"
        return DailyMarkBars(record["archive_id"], record["symbol"], "1h",
                             start_ms, end_ms, record, parsed, row_map, [])

    if record["interval"] != "1m":
        raise ValueError(f"unsupported daily mark interval: {record['interval']}")
    minutes, issues, minute_raw_count = _parse_minute_mark(payload, record["symbol"])
    expected = list(range(start_ms, end_ms, MINUTE_MS))
    if (issues or minute_raw_count != 1_440 or len(minutes) != 1_440
            or [bar.open_time for bar in minutes] != expected
            or any(bar.close_time != bar.open_time + MINUTE_MS - 1 for bar in minutes)):
        raise ValueError(f"daily 1m mark source is incomplete or has non-nominal closes: {record['source_key']}")
    record["raw_row_count"] = minute_raw_count
    record["row_count"] = len(minutes)
    record["selected_row_count"] = 0
    record["issue_count"] = 0
    record["status"] = "complete"
    return DailyMarkBars(record["archive_id"], record["symbol"], "1m",
                         start_ms, end_ms, record, [], {}, minutes)


def supplement_missing_native_mark_from_official_daily_archives(
    data_dir: Path,
) -> dict[str, Any]:
    """Restore only the audited hourly mark gaps from official daily ZIPs.

    Monthly ZIPs and funding rates are immutable.  The function performs one
    70-file public download batch, validates every full-day 1h/1m grid, copies
    the pre-supplement database and manifest, then transactionally adds the
    official hourly rows and approved prior-minute funding-mark proxies.
    """
    data_dir = Path(data_dir).resolve(strict=True)
    manifest_path = data_dir / "manifest.json"
    database_path = data_dir / "market_data.sqlite"
    if not manifest_path.is_file() or not database_path.is_file():
        raise FileNotFoundError("daily supplement requires the completed monthly-archive dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("complete") is not False or manifest.get("execution_grid_complete") is not False
            or manifest.get("unresolved_hourly_price_rows") != 840
            or manifest.get("funding_mark", {}).get("missing_proxy_count") != 105
            or len(manifest.get("source_files", [])) != 2_450
            or "supplement_source_files" in manifest):
        raise ValueError("daily supplement does not match the audited incomplete 2,450-ZIP dataset")
    if (manifest.get("missing_archives") != [] or manifest.get("parse_issue_count") != 0
            or any(record.get("status") != "complete" or record.get("issue_count") != 0
                   for record in manifest["source_files"])
            or manifest.get("repair_state", {}).get("price_interpolation") is not False
            or manifest.get("repair_state", {}).get("funding_rate_interpolation") is not False
            or manifest.get("repair_state", {}).get("primary_source_fallback") is not False):
        raise ValueError("daily supplement requires all monthly sources to be complete and unmodified")
    if (manifest.get("missing_archives") != [] or manifest.get("parse_issue_count") != 0
            or any(record.get("status") != "complete" or record.get("issue_count") != 0
                   for record in manifest["source_files"])
            or manifest.get("repair_state", {}).get("price_interpolation") is not False
            or manifest.get("repair_state", {}).get("funding_rate_interpolation") is not False
            or manifest.get("repair_state", {}).get("primary_source_fallback") is not False):
        raise ValueError("daily supplement requires all monthly sources to be complete and unmodified")
    if any((data_dir / name).exists() for name in (
        "raw_supplement", "raw_supplement.staging", "market_data.pre_supplement.sqlite",
        "manifest.pre_supplement.json", "supplement_recovery.json",
    )):
        raise FileExistsError("daily supplement output or preserved pre-supplement evidence already exists")
    if any((data_dir / f"market_data.sqlite{suffix}").exists() for suffix in ("-wal", "-shm", "-journal")):
        raise RuntimeError("cannot supplement while SQLite sidecars remain")

    original_database_signature = _file_signature(database_path)
    if original_database_signature != manifest["database_signature"]:
        raise ValueError("monthly dataset database differs from its frozen manifest signature")
    archive_ids = [record.get("archive_id") for record in manifest["source_files"]]
    if archive_ids != list(range(1, 2_451)):
        raise ValueError("monthly archive IDs differ from the fixed 1..2450 bundle")
    for record in manifest["source_files"]:
        source_path = data_dir / record["raw_path"]
        if source_path.is_symlink() or not source_path.is_file() \
                or not source_path.resolve(strict=True).is_relative_to(data_dir):
            raise ValueError(f"monthly archive path is missing or escapes the dataset: {record['raw_path']}")
        signature = _file_signature(source_path)
        if signature != {"size_bytes": record["byte_size"], "sha256": record["sha256"]}:
            raise ValueError(f"monthly source archive changed after finalization: {record['source_key']}")
    for record in manifest["source_files"]:
        source_path = data_dir / record["raw_path"]
        if source_path.is_symlink() or not source_path.is_file() \
                or not source_path.resolve(strict=True).is_relative_to(data_dir):
            raise ValueError(f"monthly archive path is missing or escapes the dataset: {record['raw_path']}")
        signature = _file_signature(source_path)
        if signature != {"size_bytes": record["byte_size"], "sha256": record["sha256"]}:
            raise ValueError(f"monthly source archive changed after finalization: {record['source_key']}")
    manifest_bytes = manifest_path.read_bytes()
    missing_funding_before = int(manifest["funding_mark"]["missing_proxy_count"])
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        source_archive_count, last_source_id = conn.execute(
            "SELECT COUNT(*),MAX(id) FROM source_archives"
        ).fetchone()
        if int(source_archive_count) != 2_450 or int(last_source_id) != 2_450:
            raise ValueError("database source archive IDs differ from the frozen monthly bundle")
        missing_events = [
            (str(symbol), int(funding_time))
            for symbol, funding_time in conn.execute(
                """SELECT symbol,funding_time FROM funding_mark_provenance
                   WHERE status!='proxied' ORDER BY symbol,funding_time"""
            )
        ]
        if len(missing_events) != missing_funding_before:
            raise ValueError("funding-mark status rows differ from the frozen manifest count")
        gaps_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        specs, gaps = _daily_supplement_specs(manifest, last_source_id + 1)
        for gap in gaps:
            key = (gap["symbol"], gap["day"])
            gaps_by_key[key] = gap
            existing = int(conn.execute(
                """SELECT COUNT(*) FROM futures_price_bars
                   WHERE data_type='markPriceKlines' AND symbol=? AND interval='1h'
                     AND open_time>=? AND open_time<?""",
                (gap["symbol"], gap["start_ms"], gap["end_ms"]),
            ).fetchone()[0])
            if existing:
                raise ValueError(f"manifest mark gap already has database bars: {key}")
    finally:
        conn.close()

    staging_root = data_dir / "raw_supplement.staging"
    staging_root.mkdir()
    supplement_progress = data_dir / "supplement_progress.json"
    supplement_started = datetime.now(timezone.utc).isoformat()
    fetched = _fetch_daily_supplement(specs, staging_root, supplement_progress)
    records = [record for record, _ in fetched]
    payloads = {record["archive_id"]: payload for record, payload in fetched}
    if len(records) != 70 or len({record["source_key"] for record in records}) != 70:
        raise ValueError("daily mark supplement did not retrieve the fixed 70 ZIP files")

    parsed_by_id: dict[int, DailyMarkBars] = {}
    for record in records:
        day = _daily_source_date(record["source_key"])
        gap = gaps_by_key.get((record["symbol"], day))
        if gap is None:
            raise ValueError(f"daily supplement source key does not map to a known gap: {record['source_key']}")
        parsed_by_id[record["archive_id"]] = _parse_daily_mark_supplement(
            record, payloads[record["archive_id"]], gap,
        )

    minute_sources_by_symbol: dict[str, list[tuple[MinuteBar, int]]] = {
        symbol: [] for symbol in SYMBOLS
    }
    minute_record_ids: set[int] = set()
    for parsed in parsed_by_id.values():
        if parsed.interval == "1m":
            minute_record_ids.add(parsed.archive_id)
            minute_sources_by_symbol[parsed.symbol].extend(
                (bar, parsed.archive_id) for bar in parsed.minute_bars
            )
    for symbol in SYMBOLS:
        minute_sources_by_symbol[symbol].sort(key=lambda item: item[0].close_time)

    restored_funding: list[dict[str, Any]] = []
    selected_minute_rows: dict[int, set[int]] = {source_id: set() for source_id in minute_record_ids}
    for symbol, funding_time in missing_events:
        candidate = _find_funding_mark_proxy(
            funding_time, minute_sources_by_symbol[symbol],
        )
        if isinstance(candidate, str):
            raise ValueError(f"daily ZIP has no causal <=60s mark donor for {symbol}/{funding_time}: {candidate}")
        minute_bar, source_id, age = candidate
        selected_minute_rows[source_id].add(minute_bar.row_number)
        restored_funding.append({
            "symbol": symbol,
            "funding_time": funding_time,
            "proxy_price": minute_bar.close,
            "source_archive_id": source_id,
            "source_row_number": minute_bar.row_number,
            "source_open_time": minute_bar.open_time,
            "source_close_time": minute_bar.close_time,
            "age_ms": age,
        })
    if len(restored_funding) != missing_funding_before:
        raise ValueError("daily minute ZIPs did not resolve every originally missing funding proxy")
    for record in records:
        if record["interval"] == "1m":
            record["selected_row_count"] = len(selected_minute_rows[record["archive_id"]])

    # Preserve the exact v1 source bundle and database before moving staged
    # daily ZIPs into their final, separately named directory.
    pre_database = data_dir / "market_data.pre_supplement.sqlite"
    pre_manifest = data_dir / "manifest.pre_supplement.json"
    shutil.copyfile(database_path, pre_database)
    shutil.copyfile(manifest_path, pre_manifest)
    if _file_signature(pre_database) != original_database_signature or pre_manifest.read_bytes() != manifest_bytes:
        raise ValueError("pre-supplement evidence copy differs from the frozen v1 dataset")
    final_raw = data_dir / "raw_supplement"
    staging_root.rename(final_raw)

    processed_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with conn:
            for record in records:
                conn.execute(
                    """INSERT INTO source_archives VALUES
                       (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (record["archive_id"], record["source_key"], record["source_url"],
                     record["raw_path"], record["market"], record["category"], record["symbol"],
                     record["interval"], record["month"], record["source_kind"], record["sha256"],
                     record["byte_size"], record["raw_row_count"], record["row_count"],
                     record["selected_row_count"], record["issue_count"], record["status"],
                     record["csv_member"]),
                )
                conn.execute(
                    """INSERT INTO futures_archive_files
                       (id,source_key,category,symbol,interval,byte_size,etag,sha256,row_count,
                        issue_count,processed_at) VALUES (?,?,?,?,?, ?,NULL,?,?,?,?)""",
                    (record["archive_id"], record["source_key"], record["category"],
                     record["symbol"], record["interval"], record["byte_size"], record["sha256"],
                     record["row_count"], record["issue_count"], processed_at),
                )

            for spec in specs:
                parsed = parsed_by_id[spec.archive_id]
                if spec.interval != "1h":
                    continue
                for row in parsed.hourly_rows:
                    open_time = int(row[3])
                    conn.execute(
                        "INSERT INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (*row, spec.archive_id),
                    )
                    _add_provenance(
                        conn, "futures_price_bars:markPriceKlines", spec.symbol, "1h",
                        open_time, spec.archive_id, "1h",
                        [parsed.row_numbers_by_time[open_time]], "official_daily_archive_row",
                    )

            for item in restored_funding:
                conn.execute(
                    """UPDATE futures_funding_rates SET mark_price=?
                       WHERE symbol=? AND funding_time=? AND mark_price IS NULL""",
                    (item["proxy_price"], item["symbol"], item["funding_time"]),
                )
                if conn.execute("SELECT changes()").fetchone()[0] != 1:
                    raise ValueError("refusing to overwrite a non-missing funding mark")
                conn.execute(
                    """UPDATE funding_mark_provenance SET proxy_price=?,source_file_id=?,
                       source_row_number=?,source_open_time=?,source_close_time=?,age_ms=?,status='proxied'
                       WHERE symbol=? AND funding_time=? AND status!='proxied'""",
                    (item["proxy_price"], item["source_archive_id"], item["source_row_number"],
                     item["source_open_time"], item["source_close_time"], item["age_ms"],
                     item["symbol"], item["funding_time"]),
                )
                if conn.execute("SELECT changes()").fetchone()[0] != 1:
                    raise ValueError("funding proxy provenance row changed before supplement commit")

            conn.execute("PRAGMA user_version=2")
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()

    # Recompute the coverage and event accounting from the committed database.
    conn = sqlite3.connect(database_path)
    try:
        coverage = _coverage(conn)
        restored_counts = {str(symbol): int(count) for symbol, count in conn.execute(
            """SELECT p.symbol,COUNT(*) FROM market_row_provenance p
               JOIN source_archives a ON a.id=p.source_file_id
               WHERE p.target_table='futures_price_bars:markPriceKlines'
                 AND p.transformation='official_daily_archive_row'
                 AND a.source_kind='official_public_daily_archive'
               GROUP BY p.symbol"""
        )}
        for item in coverage:
            item["restored_rows"] = restored_counts.get(item["symbol"], 0) \
                if item["dataset"] == "mark_price" else 0
        funding_rows = int(conn.execute("SELECT COUNT(*) FROM futures_funding_rates").fetchone()[0])
        statuses = {str(status): int(count) for status, count in conn.execute(
            "SELECT status,COUNT(*) FROM funding_mark_provenance GROUP BY status"
        )}
        ages = [int(row[0]) for row in conn.execute(
            "SELECT age_ms FROM funding_mark_provenance WHERE status='proxied' AND age_ms IS NOT NULL"
        )]
        derived = {str(symbol): int(count) for symbol, count in conn.execute(
            """SELECT symbol,COUNT(*) FROM market_row_provenance
               WHERE target_table='futures_price_bars:markPriceKlines'
                 AND transformation='aggregate_60_complete_observed_1m_mark_bars'
               GROUP BY symbol"""
        )}
    finally:
        conn.close()

    mark_holes = sum(
        int(hole["count"]) for item in coverage
        if item["dataset"] == "mark_price" for hole in item["holes"]
    )
    execution_holes = sum(
        int(hole["count"]) for item in coverage
        if item["dataset"] in {"perpetual_trade", "mark_price"} for hole in item["holes"]
    )
    missing_proxy_count = statuses.get("proxied", 0)
    missing_proxy_count = funding_rows - missing_proxy_count
    proxy_count = statuses.get("proxied", 0)
    complete = (
        execution_holes == 0 and missing_proxy_count == 0
        and manifest.get("missing_archives") == []
        and manifest.get("parse_issue_count") == 0
        and all(record.get("status") == "complete" and record.get("issue_count") == 0
                for record in manifest["source_files"])
        and all(record.get("status") == "complete" and record.get("issue_count") == 0
                for record in records)
    )
    source_records = manifest["source_files"]
    recovery_record = {
        "method": DAILY_SUPPLEMENT_METHOD,
        "source_endpoint": f"{VISION_ROOT}/data/futures/um/daily/markPriceKlines",
        "download_started_at": supplement_started,
        "download_finished_at": datetime.now(timezone.utc).isoformat(),
        "request_count": len(records),
        "supplement_source_files": [{"archive_id": record["archive_id"],
                                     "source_key": record["source_key"],
                                     "sha256": record["sha256"]} for record in records],
        "monthly_source_files_preserved": len(source_records) == 2_450,
        "monthly_manifest_sha256_before": hashlib.sha256(manifest_bytes).hexdigest(),
        "pre_supplement_database": "market_data.pre_supplement.sqlite",
        "pre_supplement_database_signature": original_database_signature,
        "pre_supplement_manifest": "manifest.pre_supplement.json",
        "native_hourly_mark_rows_restored": sum(item["restored_rows"] for item in coverage),
        "funding_proxy_rows_restored": len(restored_funding),
        "remaining_mark_holes": mark_holes,
        "remaining_funding_proxy_missing": missing_proxy_count,
        "complete": complete,
        "primary_database_read": False,
        "monthly_archives_modified": False,
        "price_or_funding_rate_interpolation": False,
    }
    database_signature = _file_signature(database_path)
    recovery_record["supplemented_database_signature"] = database_signature
    (data_dir / "supplement_recovery.json").write_text(
        json.dumps(recovery_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_v2 = dict(manifest)
    manifest_v2.update({
        "schema_version": 2,
        "database_signature": database_signature,
        "database_schema": "MarketDataStore-compatible SQLite v2 with row provenance tables",
        "source": SOURCE_V2,
        "field_sources": FIELD_SOURCES_V2,
        "coverage": coverage,
        "supplement_source_files": records,
        "funding_mark": {
            **manifest["funding_mark"],
            "proxy_count": proxy_count,
            "missing_proxy_count": missing_proxy_count,
            "max_proxy_age_ms": max(ages) if ages else None,
            "min_proxy_age_ms": min(ages) if ages else None,
            "status_counts": statuses,
            "event_rows": funding_rows,
            "derived_hourly_mark_rows_by_symbol": {
                **{symbol: 0 for symbol in SYMBOLS}, **derived,
            },
        },
        "execution_grid_complete": execution_holes == 0,
        "unresolved_hourly_price_rows": execution_holes,
        "complete": complete,
        "source_causality_notes": (
            "Official monthly and targeted daily archived mark bars preserve exchange close times. "
            "Funding marks use the previous fully completed mark minute close because the funding "
            "archives omit event markPrice; historical publication and receipt timestamps are unavailable."
        ),
    })
    for item in manifest_v2["coverage"]:
        if item["dataset"] in {"perpetual_trade", "mark_price"} and item["holes"]:
            manifest_v2["complete"] = False
    temporary_manifest = manifest_path.with_name(".manifest.v2.tmp")
    temporary_manifest.write_text(json.dumps(manifest_v2, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    _write_progress(data_dir / "supplement_progress.json", {
        "stage": "complete", "archives_completed": len(records),
        "archives_total": len(records), "last_archive_key": records[-1]["source_key"],
    })
    return manifest_v2


def _preserve_existing_file(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if destination.exists():
        raise FileExistsError(f"recovery evidence destination already exists: {destination}")
    source.rename(destination)
    return True


def finalize_from_retained_archives(output_dir: Path) -> dict[str, Any]:
    """Rebuild the failed isolated dataset from its retained ZIPs, without HTTP.

    This recovery path is for the known provenance-key collision in the first
    materialization. It requires every expected ZIP, verifies each ZIP CRC and
    SHA-256, preserves the partial database/manifest, and never downloads or
    silently substitutes another source.
    """
    output_dir = Path(output_dir).resolve(strict=True)
    specs = _archive_specs()
    records = _records_from_retained_archives(output_dir, specs)

    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = output_dir / f"market_data.sqlite{suffix}"
        if sidecar.exists():
            raise RuntimeError(f"cannot finalize while SQLite sidecar remains: {sidecar}")

    recovery_path = output_dir / "dataset_recovery.json"
    if recovery_path.exists():
        raise FileExistsError(f"recovery record already exists: {recovery_path}")
    for source_name, retained_name in (
        ("market_data.sqlite", "market_data.partial.sqlite"),
        ("manifest.json", "manifest.pre_finalization.json"),
        ("progress.json", "progress.pre_finalization.json"),
    ):
        source = output_dir / source_name
        retained = output_dir / retained_name
        if source.exists() and retained.exists():
            raise FileExistsError(f"recovery evidence destination already exists: {retained}")

    previous_progress_path = output_dir / "progress.json"
    previous_progress = None
    if previous_progress_path.is_file():
        previous_progress = json.loads(previous_progress_path.read_text(encoding="utf-8"))
    preserved_database = _preserve_existing_file(
        output_dir / "market_data.sqlite", output_dir / "market_data.partial.sqlite"
    )
    preserved_manifest = _preserve_existing_file(
        output_dir / "manifest.json", output_dir / "manifest.pre_finalization.json"
    )
    preserved_progress = _preserve_existing_file(
        previous_progress_path, output_dir / "progress.pre_finalization.json"
    )
    recovery_record = {
        "recovery_kind": "offline_finalize_retained_archives",
        "reason": "The first materialization used a provenance primary key that collided between USD-M trade and mark bars at the same symbol-hour.",
        "corrected_provenance_target_tables": {
            "perpetual_trade": "futures_price_bars:klines",
            "mark_price": "futures_price_bars:markPriceKlines",
            "spot_trade": "klines",
        },
        "network_requests": 0,
        "archives_verified": len(records),
        "archive_sha256_and_zip_crc_verified": True,
        "preserved_partial_database": "market_data.partial.sqlite" if preserved_database else None,
        "preserved_prior_manifest": "manifest.pre_finalization.json" if preserved_manifest else None,
        "preserved_prior_progress": "progress.pre_finalization.json" if preserved_progress else None,
        "prior_progress": previous_progress,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_progress(output_dir / "progress.json", {
        "stage": "offline_finalization",
        "archives_completed": 0,
        "archives_total": len(records),
        "last_archive_key": None,
    })
    recovery_path.write_text(json.dumps(recovery_record, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return _materialize_dataset(output_dir, specs, records, recovery_record=recovery_record)


def build_multifactor_raw_dataset(
    output_dir: Path,
    *,
    existing_raw_root: Path,
) -> dict[str, Any]:
    """Download/cache official monthly raw archives into a new isolated dataset.

    Existing BTC/ETH funding and 1m mark ZIPs are copied from the expressly
    named local cache. All other archives are fetched once from Binance Vision.
    A 404 is recorded as an archive hole; every other download or parse error
    stops the build and leaves retained source evidence for inspection.
    """
    output_dir = Path(output_dir).resolve()
    existing_raw_root = Path(existing_raw_root).resolve(strict=True)
    if output_dir == existing_raw_root or output_dir.is_relative_to(existing_raw_root) \
            or existing_raw_root.is_relative_to(output_dir):
        raise ValueError("dataset output and existing raw cache must not overlap")
    output_dir.mkdir(parents=True, exist_ok=False)

    specs = _archive_specs()
    records = _fetch_archives(specs, output_dir, existing_raw_root)
    return _materialize_dataset(output_dir, specs, records)


def _materialize_dataset(
    output_dir: Path,
    specs: list[ArchiveSpec],
    records: list[dict[str, Any]],
    *,
    recovery_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    database_path = output_dir / "market_data.sqlite"
    conn = _db_connect(database_path)
    try:
        _insert_archive_metadata(conn, records)
        _parse_hourly_archives(output_dir, records, conn, progress_path)
        proxy_summary = _process_minute_archives(output_dir, records, conn, progress_path)
        conn.commit()
        coverage = _coverage(conn)
        funding_rows = int(conn.execute("SELECT COUNT(*) FROM futures_funding_rates").fetchone()[0])
        funding_mark_status = conn.execute(
            "SELECT status,COUNT(*) FROM funding_mark_provenance GROUP BY status"
        ).fetchall()
        funding_mark_status = {str(status): int(count) for status, count in funding_mark_status}
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()

    # Persist final parser and proxy row counts to both the manifest and the
    # MarketDataStore-compatible source registry.
    conn = sqlite3.connect(database_path)
    for record in records:
        if record["market"] == "usd_m_perpetual" and record["sha256"] is not None:
            conn.execute("UPDATE futures_archive_files SET row_count=?,issue_count=? WHERE id=?",
                         (record["row_count"], record["issue_count"], record["archive_id"]))
        conn.execute(
            """UPDATE source_archives SET raw_row_count=?,row_count=?,selected_row_count=?,
               issue_count=?,status=? WHERE id=?""",
            (record["raw_row_count"], record["row_count"], record["selected_row_count"],
             record["issue_count"], record["status"], record["archive_id"]),
        )
    conn.commit()
    conn.execute("PRAGMA optimize")
    conn.close()

    missing_archives = [record["source_key"] for record in records
                        if record["status"] == "archive_missing_404"]
    parser_issue_count = sum(int(record["issue_count"]) for record in records)
    spot_feature_missing_rows = sum(
        sum(int(hole["count"]) for hole in item["holes"])
        for item in coverage if item["dataset"] == "spot_trade"
    )
    unresolved_price_hours = sum(
        sum(int(hole["count"]) for hole in item["holes"])
        for item in coverage if item["dataset"] in {"perpetual_trade", "mark_price"}
    )
    execution_grid_complete = unresolved_price_hours == 0
    funding_proxy_complete = proxy_summary["missing_proxy_count"] == 0 and \
        proxy_summary["proxy_count"] == funding_rows
    parsed_status_complete = all(record["status"] == "complete" for record in records)
    complete = parsed_status_complete and not missing_archives and parser_issue_count == 0 and \
        execution_grid_complete and funding_proxy_complete
    database_signature = _file_signature(database_path)
    if recovery_record is not None:
        recovery_record.update({
            "dataset_id": DATASET_ID,
            "final_database": "market_data.sqlite",
            "final_manifest": "manifest.json",
            "database_signature": database_signature,
            "complete": complete,
        })
        (output_dir / "dataset_recovery.json").write_text(
            json.dumps(recovery_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "database": "market_data.sqlite",
        "database_signature": database_signature,
        "database_schema": "MarketDataStore-compatible SQLite v1 with row provenance tables",
        "source": SOURCE_V1,
        "symbols": list(SYMBOLS),
        "window": {
            "timezone": "UTC",
            "input_start": INPUT_START.isoformat(),
            "funding_start": FUNDING_START.isoformat(),
            "end_exclusive": DATA_END.isoformat(),
            "warmup_hours": WARMUP_HOURS,
            "stages": {
                "A": {"start": "2022-08-01T00:00:00+00:00", "end_exclusive": "2024-08-01T00:00:00+00:00"},
                "B": {"start": "2024-08-01T00:00:00+00:00", "end_exclusive": "2025-08-01T00:00:00+00:00"},
                "C": {"start": "2025-08-01T00:00:00+00:00", "end_exclusive": "2026-08-01T00:00:00+00:00"},
            },
            "official_archive_months": [FIRST_MONTH, LAST_MONTH],
        },
        "source_files": records,
        "coverage": coverage,
        "field_sources": FIELD_SOURCES_V1,
        "funding_mark": {
            **proxy_summary,
            "native_event_mark_count": 0,
            "method": "previous_completed_1m_mark_close_at_or_before_funding_time",
            "max_allowed_age_ms": MAX_FUNDING_MARK_AGE_MS,
            "status_counts": funding_mark_status,
            "event_rows": funding_rows,
            "causal_rule": "source minute bar close_time <= funding_time and funding_time - close_time <= 60000ms",
            "interpretation": "explicit cost proxy, not Binance's native settlement event mark price",
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
        "primary_source_fallback": False,
        "excluded_documented_repair_rows": [],
        "source_causality": SOURCE_CAUSALITY,
        "source_causality_notes": "Official archived bars preserve exchange close times, but historical publication and receipt timestamps are unavailable.",
        "historical_causality_certified": False,
        "complete": complete,
        "execution_grid_complete": execution_grid_complete,
        "missing_archives": missing_archives,
        "parse_issue_count": parser_issue_count,
        "spot_feature_missing_rows": spot_feature_missing_rows,
        "feature_missing_rows_accepted": True,
        "unresolved_hourly_price_rows": unresolved_price_hours,
        "download_policy": {"workers": MAX_WORKERS, "retry_count": 0, "http_404": "recorded as missing archive"},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    _write_progress(progress_path, {"stage": "complete", "archives_completed": len(records),
                                    "archives_total": len(records),
                                    "last_archive_key": records[-1]["source_key"]})
    return manifest


__all__ = [
    "DATASET_ID",
    "DATA_END",
    "DAILY_SUPPLEMENT_METHOD",
    "FIELD_SOURCES_V1",
    "FIELD_SOURCES_V2",
    "FUNDING_START",
    "INPUT_START",
    "SYMBOLS",
    "WARMUP_HOURS",
    "build_multifactor_raw_dataset",
    "finalize_from_retained_archives",
    "supplement_missing_native_mark_from_official_daily_archives",
]
