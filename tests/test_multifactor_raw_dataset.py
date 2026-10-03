import csv
import hashlib
import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_raw_dataset import (
    ArchiveSpec,
    MAX_FUNDING_MARK_AGE_MS,
    MinuteBar,
    _aggregate_hour,
    _add_provenance,
    _archive_specs,
    _db_connect,
    _file_signature,
    _insert_archive_metadata,
    _find_funding_mark_proxy,
    _holes,
    _parse_minute_mark,
    _parse_spot_hourly,
    _source_rows,
    DATASET_ID,
    DATA_END,
    FIELD_SOURCES_V1,
    FUNDING_START,
    SOURCE_CAUSALITY,
    SOURCE_V1,
    _coverage,
    _daily_supplement_specs,
    _daily_source_date,
    _parse_daily_mark_supplement,
    INPUT_START,
    finalize_from_retained_archives,
    supplement_missing_native_mark_from_official_daily_archives,
)
from crypto_quant.data_access.market_data import MarketDataStore, SPOT, USD_M_PERPETUAL


def _zip_csv(name, rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerows(rows)
        archive.writestr(name, text.getvalue())
    return buffer.getvalue()


def _minute_row(open_time, close, row_number):
    return MinuteBar(
        open_time, close - 0.1, close + 0.5, close - 0.5, close,
        0.0, open_time + 59_999, 0.0, 0, 0.0, 0.0, row_number,
    )


def test_archive_plan_is_fixed_to_the_authorized_source_scope():
    specs = _archive_specs()
    assert len(specs) == 2_450
    assert len({spec.source_key for spec in specs}) == 2_450
    assert len([spec for spec in specs if spec.cache_relative_path is not None]) == 196
    assert {spec.category for spec in specs} == {"klines", "markPriceKlines", "fundingRate"}
    assert {spec.month for spec in specs} == set(
        [f"{year}-{month:02d}" for year in range(2022, 2027)
         for month in range(1, 13)
         if (year, month) >= (2022, 7) and (year, month) <= (2026, 7)]
    )


def test_spot_archives_normalize_microsecond_timestamps_and_preserve_taker_fields():
    open_ms = 1_735_689_600_000  # 2025-01-01 00:00 UTC
    open_us = open_ms * 1_000
    raw = _zip_csv("BTCUSDT-1h-2025-01.csv", [
        ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"],
        [open_us, 100, 102, 99, 101, 10, open_us + 3_599_999_000,
         1000, 7, 4, 400, 0],
    ])
    parsed, issues, row_count = _parse_spot_hourly(raw, "BTCUSDT")

    assert issues == []
    assert row_count == 1
    assert parsed == [("BTCUSDT", "1h", open_ms, 100.0, 102.0, 99.0, 101.0,
                       10.0, open_ms + 3_599_999, 1000.0, 7, 4.0, 400.0)]


def test_mark_minute_parser_and_hour_aggregation_require_all_sixty_observations():
    hour_open = 1_704_067_200_000
    rows = [["open_time", "open", "high", "low", "close", "volume", "close_time",
             "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]]
    for minute in range(60):
        stamp = hour_open + minute * 60_000
        price = 100 + minute / 100
        rows.append([stamp, price, price + 1, price - 1, price + .5,
                     0, stamp + 59_999, 0, 0, 0, 0, 0])
    raw = _zip_csv("BTCUSDT-1m-2024-01.csv", rows)
    parsed, issues, raw_count = _parse_minute_mark(raw, "BTCUSDT")
    assert issues == []
    assert raw_count == 60
    assert len(parsed) == 60

    with tempfile.TemporaryDirectory() as temp:
        conn = _db_connect(Path(temp) / "market_data.sqlite")
        conn.execute(
            "INSERT INTO source_archives VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (1, "mark-minute", "https://data.binance.vision/mark-minute", "raw/mark-minute",
             "usd_m_perpetual", "markPriceKlines", "BTCUSDT", "1m", "2024-01",
             "official_public_download", "a" * 64, 100, 60, 60, 60, 0,
             "complete", "mark.csv"),
        )
        conn.execute(
            """INSERT INTO futures_archive_files
               (id,source_key,category,symbol,interval,byte_size,etag,sha256,row_count,issue_count,processed_at)
               VALUES (1,'mark-minute','markPriceKlines','BTCUSDT','1m',100,NULL,?,60,0,'2026-10-02')""",
            ("a" * 64,),
        )
        assert _aggregate_hour(hour_open, parsed[:-1], 1, conn, "BTCUSDT") is False
        short_close = parsed[:-1] + [parsed[-1]._replace(close_time=parsed[-1].close_time - 1)]
        assert _aggregate_hour(hour_open, short_close, 1, conn, "BTCUSDT") is False
        assert _aggregate_hour(hour_open, parsed, 1, conn, "BTCUSDT") is True
        row = conn.execute(
            """SELECT open,high,low,close,close_time FROM futures_price_bars
               WHERE data_type='markPriceKlines' AND symbol='BTCUSDT' AND interval='1h'"""
        ).fetchone()
        provenance = conn.execute(
            """SELECT source_row_numbers_json,transformation FROM market_row_provenance"""
        ).fetchone()
        assert row == (100.0, 101.59, 99.0, 101.09, hour_open + 3_599_999)
        assert len(json.loads(provenance[0])) == 60
        assert provenance[1] == "aggregate_60_complete_observed_1m_mark_bars"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        conn.close()


def test_partial_spot_hour_is_retained_but_excluded_from_complete_feature_coverage():
    open_ms = int(INPUT_START.value // 1_000_000)
    with tempfile.TemporaryDirectory() as temp:
        conn = _db_connect(Path(temp) / "market_data.sqlite")
        conn.execute(
            "INSERT INTO source_archives VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (1, "spot-hour", "https://data.binance.vision/spot-hour", "raw/spot-hour",
             "spot", "klines", "BTCUSDT", "1h", "2022-07", "official_public_download",
             "b" * 64, 100, 1, 1, 1, 0, "complete", "spot.csv"),
        )
        conn.execute(
            """INSERT INTO klines VALUES
               ('BTCUSDT','1h',?,100,101,99,100,1,?,100,1,0.5,50)""",
            (open_ms, open_ms + 1_799_999),
        )
        conn.execute(
            "INSERT INTO market_row_provenance VALUES (?,?,?,?,?,?,?,?)",
            ("klines", "BTCUSDT", "1h", open_ms, 1, "1h", "[2]", "official_archive_row"),
        )
        coverage = _coverage(conn)
        btc_spot = next(item for item in coverage
                        if item["symbol"] == "BTCUSDT" and item["dataset"] == "spot_trade")
        assert btc_spot["source_observation_rows"] == 1
        assert btc_spot["selected_rows"] == 0
        assert btc_spot["partial_close_rows"] == 1
        assert btc_spot["partial_bars"][0]["open_time"] == INPUT_START.isoformat()
        conn.close()


def test_spot_perpetual_trade_and_mark_same_hour_keep_distinct_lineage_and_load():
    hour_open = 1_704_067_200_000
    close_time = hour_open + 3_599_999
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "market_data.sqlite"
        conn = _db_connect(database)
        archive_rows = [
            (1, "spot", "spot", "klines", "BTCUSDT", "1h"),
            (2, "perp-trade", "usd_m_perpetual", "klines", "BTCUSDT", "1h"),
            (3, "perp-mark", "usd_m_perpetual", "markPriceKlines", "BTCUSDT", "1h"),
        ]
        for archive_id, source_key, market, category, symbol, interval in archive_rows:
            conn.execute(
                """INSERT INTO source_archives VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (archive_id, source_key, f"https://data.binance.vision/{source_key}",
                 f"raw/{source_key}", market, category, symbol, interval, "2024-01",
                 "official_public_download", str(archive_id) * 64, 100, 1, 1, 1, 0,
                 "complete", f"{source_key}.csv"),
            )
            if market == USD_M_PERPETUAL:
                conn.execute(
                    """INSERT INTO futures_archive_files
                       (id,source_key,category,symbol,interval,byte_size,etag,sha256,
                        row_count,issue_count,processed_at)
                       VALUES (?,?,?,?,?,100,NULL,?,1,0,'2026-10-02')""",
                    (archive_id, source_key, category, symbol, interval, str(archive_id) * 64),
                )

        conn.execute(
            """INSERT INTO klines VALUES
               ('BTCUSDT','1h',?,100,101,99,100.5,1,?,100,3,.5,50)""",
            (hour_open, close_time),
        )
        for data_type, archive_id, price in (
            ("klines", 2, 200.0), ("markPriceKlines", 3, 201.0),
        ):
            conn.execute(
                """INSERT INTO futures_price_bars VALUES
                   (?,?, '1h', ?, ?, ?, ?, ?, 0, ?, 0, 0, 0, 0, ?)""",
                (data_type, "BTCUSDT", hour_open, price, price + 1, price - 1,
                 price + .5, close_time, archive_id),
            )

        _add_provenance(conn, "klines", "BTCUSDT", "1h", hour_open, 1,
                        "1h", [2], "official_archive_row")
        _add_provenance(conn, "futures_price_bars:klines", "BTCUSDT", "1h", hour_open,
                        2, "1h", [2], "official_archive_row")
        _add_provenance(conn, "futures_price_bars:markPriceKlines", "BTCUSDT", "1h",
                        hour_open, 3, "1h", [2], "official_archive_row")
        conn.commit()
        conn.close()

        store = MarketDataStore(database)
        timestamp = pd.Timestamp(hour_open, unit="ms", tz="UTC")
        spot = store.load_bars(SPOT, "BTCUSDT", "1h", start=timestamp, end=timestamp,
                               include_incomplete=True, derive=False)
        perp = store.load_bars(USD_M_PERPETUAL, "BTCUSDT", "1h", price_type="trade",
                               start=timestamp, end=timestamp, include_incomplete=True, derive=False)
        mark = store.load_bars(USD_M_PERPETUAL, "BTCUSDT", "1h", price_type="mark",
                               start=timestamp, end=timestamp, include_incomplete=True, derive=False)
        assert spot.close.iloc[0] == 100.5
        assert perp.close.iloc[0] == 200.5
        assert mark.close.iloc[0] == 201.5
        with sqlite3.connect(database) as conn:
            targets = {row[0] for row in conn.execute(
                "SELECT target_table FROM market_row_provenance")}
        assert targets == {
            "klines", "futures_price_bars:klines", "futures_price_bars:markPriceKlines"
        }


def test_offline_finalizer_rebuilds_from_exact_retained_zip_set_without_http():
    specs = [
        ArchiveSpec(index + 1, "BTCUSDT", "2022-07", "spot", "klines", "1h",
                    f"data/spot/monthly/klines/BTCUSDT/1h/{index}.zip",
                    f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/{index}.zip",
                    None)
        for index in range(2_450)
    ]
    payload = _zip_csv("rows.csv", [["timestamp"], ["1"]])
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "data"
        output.mkdir()
        raw_root = output / "raw"
        for spec in specs:
            destination = raw_root / spec.source_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (output / "market_data.sqlite").write_bytes(b"partial sqlite evidence")
        (output / "manifest.json").write_text("partial manifest\n", encoding="utf-8")
        (output / "progress.json").write_text('{"stage":"parsing"}\n', encoding="utf-8")

        with patch(
            "crypto_quant.research.strategy_research.multifactor_raw_dataset._archive_specs",
            return_value=specs,
        ), patch(
            "crypto_quant.research.strategy_research.multifactor_raw_dataset._download_one",
            side_effect=AssertionError("offline finalization must not request HTTP"),
        ), patch(
            "crypto_quant.research.strategy_research.multifactor_raw_dataset._materialize_dataset",
            return_value={"complete": False},
        ) as materialize:
            result = finalize_from_retained_archives(output)

        assert result == {"complete": False}
        assert (output / "market_data.partial.sqlite").read_bytes() == b"partial sqlite evidence"
        assert (output / "manifest.pre_finalization.json").read_text(encoding="utf-8") == "partial manifest\n"
        assert json.loads((output / "progress.pre_finalization.json").read_text()) == {"stage": "parsing"}
        recovery = json.loads((output / "dataset_recovery.json").read_text(encoding="utf-8"))
        assert recovery["network_requests"] == 0
        assert recovery["archives_verified"] == 2_450
        assert recovery["archive_sha256_and_zip_crc_verified"] is True
        passed_records = materialize.call_args.args[2]
        assert len(passed_records) == 2_450
        assert passed_records[0]["status"] == "source_retained"
        assert passed_records[0]["source_kind"] == "official_public_download"


def test_daily_supplement_plan_contains_only_the_35_audited_symbol_days():
    four_day_symbols = {"BTCUSDT", "BNBUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"}
    all_days = ("2022-07-31", "2022-10-02", "2023-02-24", "2026-06-29")
    ordinary_days = all_days[1:]
    coverage = []
    for symbol in (
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
    ):
        days = all_days if symbol in four_day_symbols else ordinary_days
        holes = []
        for day in days:
            start = pd.Timestamp(day, tz="UTC")
            holes.append({"start": start.isoformat(),
                          "end_exclusive": (start + pd.Timedelta(days=1)).isoformat(),
                          "count": 24})
        coverage.append({"dataset": "mark_price", "symbol": symbol, "holes": holes})
    specs, gaps = _daily_supplement_specs({"coverage": coverage}, 2_451)

    assert len(gaps) == 35
    assert len(specs) == 70
    assert {spec.interval for spec in specs} == {"1h", "1m"}
    assert all("/daily/markPriceKlines/" in spec.source_key for spec in specs)
    assert specs[0].archive_id == 2_451
    assert specs[-1].archive_id == 2_520
    assert _daily_source_date(
        "data/futures/um/daily/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2022-10-02.zip"
    ) == "2022-10-02"


def test_daily_mark_parser_requires_native_complete_hour_and_minute_grids():
    start_ms = int(pd.Timestamp("2022-10-02T00:00:00Z").value // 1_000_000)
    end_ms = start_ms + 24 * 3_600_000
    gap = {"symbol": "BTCUSDT", "day": "2022-10-02",
           "start_ms": start_ms, "end_ms": end_ms}
    hourly_header = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                     "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    hourly_rows = [hourly_header]
    for stamp in range(start_ms, end_ms, 3_600_000):
        hourly_rows.append([stamp, 100, 101, 99, 100.5, 0, stamp + 3_599_999,
                            0, 0, 0, 0, 0])
    hour_spec = ArchiveSpec(
        2_451, "BTCUSDT", "2022-10", "usd_m_perpetual", "markPriceKlines", "1h",
        "daily-hour.zip", "https://data.binance.vision/daily-hour.zip", None,
    )
    hourly = _parse_daily_mark_supplement(
        {"archive_id": 2_451, "symbol": "BTCUSDT", "interval": "1h", "source_key": hour_spec.source_key},
        _zip_csv("daily-hour.csv", hourly_rows), gap,
    )
    assert len(hourly.hourly_rows) == 24
    assert hourly.row_numbers_by_time[start_ms] == 2
    assert hourly.record["selected_row_count"] == 24

    minute_rows = [hourly_header]
    for stamp in range(start_ms, end_ms, 60_000):
        minute_rows.append([stamp, 100, 101, 99, 100.5, 0, stamp + 59_999,
                            0, 0, 0, 0, 0])
    minute_spec = ArchiveSpec(
        2_452, "BTCUSDT", "2022-10", "usd_m_perpetual", "markPriceKlines", "1m",
        "daily-minute.zip", "https://data.binance.vision/daily-minute.zip", None,
    )
    minute = _parse_daily_mark_supplement(
        {"archive_id": 2_452, "symbol": "BTCUSDT", "interval": "1m", "source_key": minute_spec.source_key},
        _zip_csv("daily-minute.csv", minute_rows), gap,
    )
    assert len(minute.minute_bars) == 1_440
    assert minute.minute_bars[0].row_number == 2
    assert minute.minute_bars[-1].close_time == end_ms - 1
    assert minute.record["selected_row_count"] == 0

    incomplete = _zip_csv("daily-hour.csv", hourly_rows[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        _parse_daily_mark_supplement(
            {"archive_id": 2_453, "symbol": "BTCUSDT", "interval": "1h", "source_key": "missing-hour.zip"},
            incomplete, gap,
        )


def test_daily_supplement_updates_only_declared_gaps_without_http(tmp_path):
    specs = _archive_specs()
    database = tmp_path / "market_data.sqlite"
    conn = _db_connect(database)
    month_specs = {(spec.symbol, spec.month): spec for spec in specs
                   if spec.market == "usd_m_perpetual" and spec.category == "fundingRate"}
    monthly_records = []
    monthly_payload = _zip_csv("source.csv", [["timestamp"], ["1"]])
    for spec in specs:
        monthly_path = tmp_path / "raw" / spec.source_key
        monthly_path.parent.mkdir(parents=True, exist_ok=True)
        monthly_path.write_bytes(monthly_payload)
        monthly_records.append({
            "archive_id": spec.archive_id, "source_key": spec.source_key,
            "source_url": spec.source_url, "raw_path": f"raw/{spec.source_key}",
            "market": spec.market, "category": spec.category, "symbol": spec.symbol,
            "interval": spec.interval, "month": spec.month,
            "source_kind": ("existing_local_raw_cache" if spec.cache_relative_path
                            else "official_public_download"),
            "sha256": hashlib.sha256(monthly_payload).hexdigest(),
            "byte_size": len(monthly_payload),
            "raw_row_count": 1, "row_count": 1, "selected_row_count": 1,
            "issue_count": 0, "status": "complete", "csv_member": "source.csv",
        })
    _insert_archive_metadata(conn, monthly_records)

    all_days = ("2022-07-31", "2022-10-02", "2023-02-24", "2026-06-29")
    fewer_days = all_days[1:]
    four_day_symbols = {"BTCUSDT", "BNBUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"}
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                   "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"):
        for day in (all_days if symbol in four_day_symbols else fewer_days):
            gap_start = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
            gap_end = gap_start + 24 * 3_600_000
            rate_archive = month_specs[(symbol, day[:7])]
            for hour in (8, 16):
                event_time = gap_start + hour * 3_600_000 + 11
                conn.execute(
                    """INSERT INTO futures_funding_rates
                       (symbol,funding_time,funding_interval_hours,funding_rate,source_file_id,mark_price)
                       VALUES (?,?,8,0.0001,?,NULL)""",
                    (symbol, event_time, rate_archive.archive_id),
                )
                conn.execute(
                    """INSERT INTO market_row_provenance VALUES
                       ('futures_funding_rates',?,'native',?,?, 'native','[2]',
                        'official_funding_event_row')""",
                    (symbol, event_time, rate_archive.archive_id),
                )
                conn.execute(
                    """INSERT INTO funding_mark_provenance
                       (symbol,funding_time,native_event_mark_price,proxy_price,proxy_method,
                        source_file_id,source_row_number,source_open_time,source_close_time,age_ms,status)
                       VALUES (?, ?, NULL, NULL,
                         'previous_completed_1m_mark_close_at_or_before_funding_time',
                         NULL,NULL,NULL,NULL,NULL,'prior_mark_minute_older_than_60s')""",
                    (symbol, event_time),
                )
            event_time = gap_end + 11
            rate_archive = month_specs[(symbol, (pd.Timestamp(gap_end, unit="ms", tz="UTC")).strftime("%Y-%m"))]
            conn.execute(
                """INSERT INTO futures_funding_rates
                   (symbol,funding_time,funding_interval_hours,funding_rate,source_file_id,mark_price)
                   VALUES (?,?,8,0.0001,?,NULL)""",
                (symbol, event_time, rate_archive.archive_id),
            )
            conn.execute(
                """INSERT INTO market_row_provenance VALUES
                   ('futures_funding_rates',?,'native',?,?, 'native','[2]',
                    'official_funding_event_row')""",
                (symbol, event_time, rate_archive.archive_id),
            )
            conn.execute(
                """INSERT INTO funding_mark_provenance
                   (symbol,funding_time,native_event_mark_price,proxy_price,proxy_method,
                    source_file_id,source_row_number,source_open_time,source_close_time,age_ms,status)
                   VALUES (?, ?, NULL, NULL,
                     'previous_completed_1m_mark_close_at_or_before_funding_time',
                     NULL,NULL,NULL,NULL,NULL,'prior_mark_minute_older_than_60s')""",
                (symbol, event_time),
            )
    conn.commit()
    conn.close()

    coverage = []
    expected_hours = int((DATA_END - INPUT_START) / pd.Timedelta(hours=1))
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                   "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"):
        mark_days = all_days if symbol in four_day_symbols else fewer_days
        holes = []
        for day in mark_days:
            start = pd.Timestamp(day, tz="UTC")
            holes.append({"start": start.isoformat(),
                          "end_exclusive": (start + pd.Timedelta(days=1)).isoformat(),
                          "count": 24})
        for dataset in ("spot_trade", "perpetual_trade", "mark_price"):
            if dataset == "mark_price":
                observed = expected_hours - 24 * len(mark_days)
                source_count = observed
                current_holes = holes
            else:
                observed = expected_hours
                source_count = expected_hours
                current_holes = []
            coverage.append({"symbol": symbol, "dataset": dataset,
                             "expected_rows": expected_hours, "archive_rows": source_count,
                             "selected_rows": observed, "source_observation_rows": observed,
                             "partial_close_rows": 0, "partial_bars": [], "derived_rows": 0,
                             "holes": current_holes})

    base_manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID, "database": "market_data.sqlite",
        "database_schema": "MarketDataStore-compatible SQLite v1 with row provenance tables",
        "database_signature": _file_signature(database),
        "source": SOURCE_V1, "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                                           "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"],
        "window": {"timezone": "UTC", "input_start": INPUT_START.isoformat(),
                   "funding_start": FUNDING_START.isoformat(),
                   "end_exclusive": DATA_END.isoformat(), "warmup_hours": 168,
                   "stages": {"A": {"start": "2022-08-01T00:00:00+00:00", "end_exclusive": "2024-08-01T00:00:00+00:00"},
                              "B": {"start": "2024-08-01T00:00:00+00:00", "end_exclusive": "2025-08-01T00:00:00+00:00"},
                              "C": {"start": "2025-08-01T00:00:00+00:00", "end_exclusive": "2026-08-01T00:00:00+00:00"}},
                   "official_archive_months": ["2022-07", "2026-07"]},
        "source_files": monthly_records, "coverage": coverage,
        "field_sources": FIELD_SOURCES_V1,
        "funding_mark": {"proxy_count": 0, "missing_proxy_count": 105,
                         "max_proxy_age_ms": None, "min_proxy_age_ms": None,
                         "derived_hourly_mark_rows_by_symbol": {symbol: 0 for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"]},
                         "native_event_mark_count": 0,
                         "method": "previous_completed_1m_mark_close_at_or_before_funding_time",
                         "max_allowed_age_ms": 60_000,
                         "status_counts": {"prior_mark_minute_older_than_60s": 105},
                         "event_rows": 105,
                         "causal_rule": "source minute bar close_time <= funding_time and funding_time - close_time <= 60000ms",
                         "interpretation": "explicit cost proxy, not Binance's native settlement event mark price"},
        "repair_state": {"price_interpolation": False, "funding_rate_interpolation": False,
                         "funding_mark_is_native_event_field": False, "funding_mark_proxy_declared": True,
                         "minute_to_hour_mark_aggregation": "only complete 60 observed minute rows; no interpolation",
                         "primary_source_fallback": False, "primary_database_read": False,
                         "synthetic_archive_rows": False},
        "primary_source_fallback": False, "excluded_documented_repair_rows": [],
        "source_causality": SOURCE_CAUSALITY,
        "source_causality_notes": "Official archived bars preserve exchange close times, but historical publication and receipt timestamps are unavailable.",
        "historical_causality_certified": False, "complete": False,
        "execution_grid_complete": False, "missing_archives": [], "parse_issue_count": 0,
        "spot_feature_missing_rows": 0, "feature_missing_rows_accepted": True,
        "unresolved_hourly_price_rows": 840,
        "download_policy": {"workers": 4, "retry_count": 0, "http_404": "recorded as missing archive"},
    }
    base_manifest["database_signature"] = _file_signature(database)
    original_sample_monthly_zip = tmp_path / "raw" / monthly_records[0]["source_key"]
    original_sample_bytes = original_sample_monthly_zip.read_bytes()
    (tmp_path / "manifest.json").write_text(
        json.dumps(base_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def fake_fetch(daily_specs, staging_root, progress_path):
        results = []
        for spec in daily_specs:
            day = _daily_source_date(spec.source_key)
            start_ms = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
            end_ms = start_ms + 24 * 3_600_000
            rows = [["open_time", "open", "high", "low", "close", "volume", "close_time",
                     "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]]
            step = 3_600_000 if spec.interval == "1h" else 60_000
            for stamp in range(start_ms, end_ms, step):
                rows.append([stamp, 100, 101, 99, 100.5, 0, stamp + step - 1,
                             0, 0, 0, 0, 0])
            payload = _zip_csv(f"{spec.symbol}-{spec.interval}-{day}.csv", rows)
            destination = staging_root / spec.source_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_member = [name for name in archive.namelist() if name.endswith(".csv")][0]
            record = {
                "archive_id": spec.archive_id, "source_key": spec.source_key,
                "source_url": spec.source_url,
                "raw_path": f"raw_supplement/{spec.source_key}",
                "market": spec.market, "category": spec.category, "symbol": spec.symbol,
                "interval": spec.interval, "month": spec.month,
                "source_kind": "official_public_daily_archive",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload), "raw_row_count": 0, "row_count": 0,
                "selected_row_count": 0, "issue_count": 0, "status": "source_retained",
                "csv_member": csv_member,
            }
            results.append((record, payload))
        return results

    def fake_coverage(_conn):
        result = [dict(item) for item in coverage]
        for item in result:
            if item["dataset"] == "mark_price":
                item["holes"] = []
                item["selected_rows"] = item["expected_rows"]
                item["source_observation_rows"] = item["expected_rows"]
        return result

    with patch(
        "crypto_quant.research.strategy_research.multifactor_raw_dataset._fetch_daily_supplement",
        side_effect=fake_fetch,
    ), patch(
        "crypto_quant.research.strategy_research.multifactor_raw_dataset._coverage",
        side_effect=fake_coverage,
    ), patch(
        "crypto_quant.research.strategy_research.multifactor_raw_dataset._download_one",
        side_effect=AssertionError("fixture supplement must not issue an HTTP request"),
    ):
        result = supplement_missing_native_mark_from_official_daily_archives(tmp_path)

    assert result["schema_version"] == 2
    assert result["complete"] is True
    assert len(result["source_files"]) == 2_450
    assert len(result["supplement_source_files"]) == 70
    assert sum(item["restored_rows"] for item in result["coverage"]
               if item["dataset"] == "mark_price") == 840
    assert result["funding_mark"]["missing_proxy_count"] == 0
    assert result["funding_mark"]["proxy_count"] == 105
    assert (tmp_path / "market_data.pre_supplement.sqlite").is_file()
    assert (tmp_path / "manifest.pre_supplement.json").is_file()
    assert (tmp_path / "supplement_recovery.json").is_file()
    assert original_sample_monthly_zip.read_bytes() == original_sample_bytes
    with sqlite3.connect(database) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute(
            """SELECT COUNT(*) FROM futures_price_bars
               WHERE data_type='markPriceKlines' AND source_file_id>=2451"""
        ).fetchone()[0] == 840
        assert conn.execute(
            """SELECT COUNT(*) FROM futures_funding_rates
               WHERE mark_price IS NOT NULL"""
        ).fetchone()[0] == 105
        assert conn.execute(
            "SELECT COUNT(*) FROM source_archives"
        ).fetchone()[0] == 2_520


def test_funding_proxy_uses_only_a_previously_closed_minute_mark():
    hour_open = 1_704_067_200_000
    prior = _minute_row(hour_open - 60_000, 100.0, 2)
    current = _minute_row(hour_open, 110.0, 3)
    candidate = _find_funding_mark_proxy(hour_open + 10, [(prior, 1), (current, 2)])
    assert candidate == (prior, 1, 11)

    stale = _minute_row(hour_open - 120_000, 99.0, 1)
    candidate = _find_funding_mark_proxy(hour_open, [(stale, 1)])
    assert candidate == "prior_mark_minute_older_than_60s"
    assert MAX_FUNDING_MARK_AGE_MS == 60_000


def test_hourly_holes_are_reported_as_left_closed_ranges():
    hour = 3_600_000
    expected = [0, hour, hour * 2, hour * 3, hour * 4]
    assert _holes(expected, {0, hour * 3, hour * 4}) == [
        {"start": "1970-01-01T01:00:00+00:00",
         "end_exclusive": "1970-01-01T03:00:00+00:00", "count": 2}
    ]
