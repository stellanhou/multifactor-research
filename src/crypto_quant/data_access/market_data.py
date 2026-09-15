"""Unified read-only access to local spot and USD-M research data."""

from __future__ import annotations

import calendar
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from crypto_quant.data_access.data import KLINE_COLUMNS, interval_to_ms


SPOT = "spot"
USD_M_PERPETUAL = "usd_m_perpetual"

PRICE_TYPE_TO_STORAGE = {
    "trade": "klines",
    "mark": "markPriceKlines",
    "index": "indexPriceKlines",
    "premium_index": "premiumIndexKlines",
}
STORAGE_TO_PRICE_TYPE = {value: key for key, value in PRICE_TYPE_TO_STORAGE.items()}

FUNDING_COLUMNS = ("funding_rate", "funding_interval_hours")
METRICS_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

# Explicit pairs present in the local Binance spot/USD-M archive. Numeric
# prefixes are not stripped: 1000SATS, 1000CAT and 1MBABYDOGE already have the
# same denomination in both markets.
SPOT_TO_PERPETUAL = {
    "PEPEUSDT": "1000PEPEUSDT",
    "SHIBUSDT": "1000SHIBUSDT",
    "BONKUSDT": "1000BONKUSDT",
    "FLOKIUSDT": "1000FLOKIUSDT",
    "LUNCUSDT": "1000LUNCUSDT",
    "XECUSDT": "1000XECUSDT",
    "BTTCUSDT": "1000BTTCUSDT",
}


@dataclass(frozen=True)
class MarketSymbols:
    spot: str
    perpetual: str
    perpetual_multiplier: int


def resolve_market_symbols(symbol: str) -> MarketSymbols:
    """Resolve either native symbol; multiplier is in spot base-asset units."""
    symbol = str(symbol).upper()
    for spot, perpetual in SPOT_TO_PERPETUAL.items():
        if symbol in (spot, perpetual):
            return MarketSymbols(spot, perpetual, 1000)
    return MarketSymbols(symbol, symbol, 1)


LIQUIDATION_COUNT_COLUMNS = (
    "liquidation_event_count", "long_liquidation_count", "short_liquidation_count",
)
LIQUIDATION_NOTIONAL_COLUMNS = (
    "long_liquidation_notional_usdt", "short_liquidation_notional_usdt",
)
LIQUIDATION_VALUE_COLUMNS = (*LIQUIDATION_COUNT_COLUMNS, *LIQUIDATION_NOTIONAL_COLUMNS)


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _to_utc_ms(value: object) -> int:
    return int(_to_utc_timestamp(value).value // 1_000_000)


def _iso_ms(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True).isoformat()


@dataclass(frozen=True)
class DatasetCoverage:
    dataset: str
    market: str
    symbol: Optional[str]
    interval: str
    price_type: Optional[str]
    symbols: int
    rows: int
    start: Optional[str]
    end: Optional[str]
    coverage_precision: str = "exact"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketDataStore:
    """One causal, read-only interface over the local research database.

    The adapter never downloads, mutates, fills, or interpolates market data.
    It uses the common stored 1-hour base for complete multi-hour or daily bars
    when available; exact stored intervals remain accessible with derive=False.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._metrics_snapshot_cache: Optional[
            tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        ] = None

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError(f"market database does not exist: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA query_only=ON")
        return conn

    @staticmethod
    def _has_table(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is not None

    @staticmethod
    def _has_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
    ) -> bool:
        return any(
            str(row[1]) == column
            for row in conn.execute(f"PRAGMA table_info({table})")
        )

    @staticmethod
    def _time_filters(
        column: str,
        start: Optional[object],
        end: Optional[object],
        params: List[Any],
    ) -> str:
        sql = ""
        if start is not None:
            sql += f" AND {column} >= ?"
            params.append(_to_utc_ms(start))
        if end is not None:
            sql += f" AND {column} <= ?"
            params.append(_to_utc_ms(end))
        return sql

    @staticmethod
    def _finish_bars(
        frame: pd.DataFrame,
        market: str,
        symbol: str,
        interval: str,
        price_type: str,
        include_incomplete: bool,
        derived_from: Optional[str] = None,
    ) -> pd.DataFrame:
        if frame.empty:
            raise ValueError(
                f"no local {market} {price_type} bars for {symbol} {interval}"
            )
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        frame = frame.set_index("open_time").sort_index()
        frame.index.name = "timestamp"
        for column in frame.columns:
            if column != "close_time":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not include_incomplete:
            complete_at = frame.index + pd.Timedelta(milliseconds=interval_to_ms(interval))
            frame = frame.loc[complete_at <= pd.Timestamp.now(tz="UTC")]
        if frame.empty:
            raise ValueError(f"no complete local bars for {symbol} {interval}")
        frame.attrs.update(
            {
                "market": market,
                "symbol": symbol,
                "interval": interval,
                "price_type": price_type,
                "derived_from": derived_from,
            }
        )
        return frame

    def _has_exact_bars(
        self,
        conn: sqlite3.Connection,
        market: str,
        symbol: str,
        interval: str,
        storage_type: str,
    ) -> bool:
        if market == SPOT:
            if not self._has_table(conn, "klines"):
                return False
            row = conn.execute(
                "SELECT 1 FROM klines WHERE symbol=? AND interval=? LIMIT 1",
                (symbol, interval),
            ).fetchone()
        else:
            if not self._has_table(conn, "futures_price_bars"):
                return False
            row = conn.execute(
                """SELECT 1 FROM futures_price_bars
                   WHERE data_type=? AND symbol=? AND interval=? LIMIT 1""",
                (storage_type, symbol, interval),
            ).fetchone()
        return row is not None

    def _read_exact_bars(
        self,
        conn: sqlite3.Connection,
        market: str,
        symbol: str,
        interval: str,
        storage_type: str,
        start: Optional[object],
        end: Optional[object],
    ) -> pd.DataFrame:
        params: List[Any]
        if market == SPOT:
            params = [symbol, interval]
            sql = f"SELECT {', '.join(KLINE_COLUMNS)} FROM klines WHERE symbol=? AND interval=?"
        else:
            params = [storage_type, symbol, interval]
            sql = (
                f"SELECT {', '.join(KLINE_COLUMNS)} FROM futures_price_bars "
                "WHERE data_type=? AND symbol=? AND interval=?"
            )
        sql += self._time_filters("open_time", start, end, params)
        sql += " ORDER BY open_time"
        return pd.read_sql_query(sql, conn, params=params)

    @staticmethod
    def _derived_query_bounds(
        interval: str,
        start: Optional[object],
        end: Optional[object],
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        target_ms = interval_to_ms(interval)
        source_ms = interval_to_ms("1h")
        query_start: Optional[pd.Timestamp] = None
        query_end: Optional[pd.Timestamp] = None
        if start is not None:
            value = _to_utc_ms(start)
            query_start = pd.to_datetime((value // target_ms) * target_ms, unit="ms", utc=True)
        if end is not None:
            value = _to_utc_ms(end)
            bucket = (value // target_ms) * target_ms
            query_end = pd.to_datetime(bucket + target_ms - source_ms, unit="ms", utc=True)
        return query_start, query_end

    @staticmethod
    def _derive_bars(
        source: pd.DataFrame,
        interval: str,
        start: Optional[object],
        end: Optional[object],
    ) -> pd.DataFrame:
        target_ms = interval_to_ms(interval)
        source_ms = interval_to_ms("1h")
        if target_ms <= source_ms or target_ms % source_ms != 0:
            raise ValueError(f"cannot derive {interval} bars from local 1h bars")
        ratio = target_ms // source_ms
        working = source.reset_index()
        open_ms = (working["timestamp"].astype("int64") // 1_000_000).to_numpy()
        working["bucket_ms"] = (open_ms // target_ms) * target_ms
        grouped = working.groupby("bucket_ms", sort=True)
        derived = grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            close_time=("close_time", "last"),
            quote_volume=("quote_volume", "sum"),
            trades=("trades", "sum"),
            taker_buy_base_volume=("taker_buy_base_volume", "sum"),
            taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
            source_count=("timestamp", "size"),
            source_first=("timestamp", "min"),
            source_last=("timestamp", "max"),
        )
        expected_first = pd.to_datetime(derived.index.to_numpy(), unit="ms", utc=True)
        expected_last = expected_first + pd.Timedelta(milliseconds=target_ms - source_ms)
        complete = (
            (derived["source_count"] == ratio)
            & (pd.DatetimeIndex(derived["source_first"]) == expected_first)
            & (pd.DatetimeIndex(derived["source_last"]) == expected_last)
        )
        derived = derived.loc[complete].drop(
            columns=["source_count", "source_first", "source_last"]
        )
        derived.insert(0, "open_time", derived.index.astype("int64"))
        derived = derived.reset_index(drop=True)
        if start is not None:
            derived = derived.loc[derived["open_time"] >= _to_utc_ms(start)]
        if end is not None:
            derived = derived.loc[derived["open_time"] <= _to_utc_ms(end)]
        return derived

    def load_bars(
        self,
        market: str,
        symbol: str,
        interval: str = "1h",
        price_type: str = "trade",
        start: Optional[object] = None,
        end: Optional[object] = None,
        include_incomplete: bool = False,
        derive: bool = True,
    ) -> pd.DataFrame:
        """Load consistently shaped OHLCV bars for spot or USD-M perpetuals."""
        market = str(market).lower()
        symbol = str(symbol).upper()
        price_type = str(price_type).lower()
        if market not in {SPOT, USD_M_PERPETUAL}:
            raise ValueError(f"unsupported market: {market}")
        if price_type not in PRICE_TYPE_TO_STORAGE:
            raise ValueError(f"unsupported price_type: {price_type}")
        if market == SPOT and price_type != "trade":
            raise ValueError("spot supports only price_type='trade'")
        interval_to_ms(interval)
        storage_type = PRICE_TYPE_TO_STORAGE[price_type]

        conn = self._connect()
        try:
            target_ms = interval_to_ms(interval)
            source_ms = interval_to_ms("1h")
            can_derive = (
                derive
                and target_ms > source_ms
                and target_ms % source_ms == 0
                and self._has_exact_bars(conn, market, symbol, "1h", storage_type)
            )
            if can_derive:
                query_start, query_end = self._derived_query_bounds(interval, start, end)
                source_raw = self._read_exact_bars(
                    conn,
                    market,
                    symbol,
                    "1h",
                    storage_type,
                    query_start,
                    query_end,
                )
                source = self._finish_bars(
                    source_raw,
                    market,
                    symbol,
                    "1h",
                    price_type,
                    include_incomplete=True,
                )
                derived_frame = self._derive_bars(source, interval, start, end)
                return self._finish_bars(
                    derived_frame,
                    market,
                    symbol,
                    interval,
                    price_type,
                    include_incomplete,
                    derived_from="1h",
                )

            if self._has_exact_bars(conn, market, symbol, interval, storage_type):
                frame = self._read_exact_bars(
                    conn, market, symbol, interval, storage_type, start, end
                )
                return self._finish_bars(
                    frame,
                    market,
                    symbol,
                    interval,
                    price_type,
                    include_incomplete,
                )
            raise ValueError(
                f"no local {market} {price_type} bars for {symbol} {interval}"
            )
        finally:
            conn.close()

    def _load_event_frame(
        self,
        table: str,
        time_column: str,
        value_columns: Sequence[str],
        symbol: str,
        start: Optional[object],
        end: Optional[object],
        include_previous: bool,
    ) -> pd.DataFrame:
        symbol = str(symbol).upper()
        conn = self._connect()
        try:
            if not self._has_table(conn, table):
                raise ValueError(f"local dataset is missing table: {table}")
            params: List[Any] = [symbol]
            sql = (
                f"SELECT {time_column}, {', '.join(value_columns)} "
                f"FROM {table} WHERE symbol=?"
            )
            sql += self._time_filters(time_column, start, end, params)
            sql += f" ORDER BY {time_column}"
            frame = pd.read_sql_query(sql, conn, params=params)
            if include_previous and start is not None:
                previous = pd.read_sql_query(
                    f"""SELECT {time_column}, {', '.join(value_columns)}
                        FROM {table} WHERE symbol=? AND {time_column} < ?
                        ORDER BY {time_column} DESC LIMIT 1""",
                    conn,
                    params=[symbol, _to_utc_ms(start)],
                )
                if not previous.empty:
                    frame = pd.concat([previous, frame], ignore_index=True)
        finally:
            conn.close()
        if frame.empty:
            raise ValueError(f"no local {table} data for {symbol}")
        frame[time_column] = pd.to_datetime(frame[time_column], unit="ms", utc=True)
        frame = frame.set_index(time_column).sort_index()
        frame.index.name = "timestamp"
        for column in value_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.attrs.update({"symbol": symbol, "source_table": table})
        return frame

    @staticmethod
    def _has_symbol(
        conn: sqlite3.Connection,
        table: str,
        symbol: str,
    ) -> bool:
        return conn.execute(
            f"SELECT 1 FROM {table} WHERE symbol=? LIMIT 1",
            (symbol,),
        ).fetchone() is not None

    def _read_events(
        self,
        conn: sqlite3.Connection,
        select_values: str,
        from_sql: str,
        where_sql: str,
        params: Sequence[Any],
        time_expression: str,
        start: Optional[object],
        end: Optional[object],
        include_previous: bool,
    ) -> pd.DataFrame:
        current_params = list(params)
        filters = self._time_filters(
            time_expression,
            start,
            end,
            current_params,
        )
        select_sql = f"{time_expression} AS event_time, {select_values}"
        frame = pd.read_sql_query(
            f"SELECT {select_sql} FROM {from_sql} "
            f"WHERE {where_sql}{filters} ORDER BY {time_expression}",
            conn,
            params=current_params,
        )
        if include_previous and start is not None:
            previous = pd.read_sql_query(
                f"SELECT {select_sql} FROM {from_sql} "
                f"WHERE {where_sql} AND {time_expression} < ? "
                f"ORDER BY {time_expression} DESC LIMIT 1",
                conn,
                params=[*params, _to_utc_ms(start)],
            )
            if not previous.empty:
                frame = pd.concat([previous, frame], ignore_index=True)
        if not frame.empty:
            frame = frame.drop_duplicates(subset=["event_time"], keep="last")
            frame = frame.sort_values("event_time")
        return frame

    @staticmethod
    def _finish_events(
        frame: pd.DataFrame,
        symbol: str,
        source_table: str,
        numeric_columns: Sequence[str],
    ) -> pd.DataFrame:
        if frame.empty:
            raise ValueError(f"no local {source_table} data for {symbol}")
        frame["event_time"] = pd.to_datetime(
            frame["event_time"], unit="ms", utc=True
        )
        frame = frame.set_index("event_time").sort_index()
        frame.index.name = "timestamp"
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.attrs.update({"symbol": symbol, "source_table": source_table})
        return frame

    def load_funding(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
        include_previous: bool = False,
    ) -> pd.DataFrame:
        """Load native-cadence USD-M funding events.

        The consolidated futures table is authoritative for a symbol when it
        contains any rows. The legacy table is used only as a compatibility
        fallback for symbols that have not been migrated yet.
        """
        symbol = str(symbol).upper()
        conn = self._connect()
        try:
            has_new = self._has_table(conn, "futures_funding_rates") and self._has_symbol(
                conn, "futures_funding_rates", symbol
            )
            if has_new:
                legacy_join = ""
                has_current_mark = self._has_column(
                    conn, "futures_funding_rates", "mark_price"
                )
                mark_price = (
                    "current.mark_price AS mark_price"
                    if has_current_mark
                    else "NULL AS mark_price"
                )
                if self._has_table(conn, "funding_rates"):
                    legacy_join = (
                        "LEFT JOIN funding_rates AS legacy "
                        "ON legacy.symbol=current.symbol "
                        "AND legacy.funding_time=current.funding_time"
                    )
                    mark_price = (
                        "COALESCE(current.mark_price, legacy.mark_price) AS mark_price"
                        if has_current_mark
                        else "legacy.mark_price AS mark_price"
                    )
                archive_join = ""
                source_path = "NULL AS source_path"
                if self._has_table(conn, "futures_archive_files"):
                    archive_join = (
                        "LEFT JOIN futures_archive_files AS archive "
                        "ON archive.id=current.source_file_id"
                    )
                    source_path = "archive.source_key AS source_path"
                frame = self._read_events(
                    conn,
                    (
                        "current.funding_rate, current.funding_interval_hours, "
                        f"{mark_price}, {source_path}"
                    ),
                    "futures_funding_rates AS current "
                    f"{legacy_join} {archive_join}",
                    "current.symbol=?",
                    [symbol],
                    "current.funding_time",
                    start,
                    end,
                    include_previous,
                )
                source_table = "futures_funding_rates"
            elif self._has_table(conn, "funding_rates"):
                frame = self._read_events(
                    conn,
                    (
                        "current.funding_rate, current.funding_interval_hours, "
                        "current.mark_price, current.source_path"
                    ),
                    "funding_rates AS current",
                    "current.symbol=?",
                    [symbol],
                    "current.funding_time",
                    start,
                    end,
                    include_previous,
                )
                source_table = "funding_rates"
            else:
                raise ValueError("local funding dataset is missing")
        finally:
            conn.close()
        return self._finish_events(
            frame,
            symbol,
            source_table,
            (*FUNDING_COLUMNS, "mark_price"),
        )

    def load_metrics(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
        include_previous: bool = False,
    ) -> pd.DataFrame:
        """Load native 5-minute USD-M open-interest and positioning metrics."""
        symbol = str(symbol).upper()
        conn = self._connect()
        try:
            has_new = self._has_table(conn, "futures_metrics") and self._has_symbol(
                conn, "futures_metrics", symbol
            )
            if has_new:
                archive_join = ""
                source_path = "NULL AS source_path"
                if self._has_table(conn, "futures_archive_files"):
                    archive_join = (
                        "LEFT JOIN futures_archive_files AS archive "
                        "ON archive.id=current.source_file_id"
                    )
                    source_path = "archive.source_key AS source_path"
                frame = self._read_events(
                    conn,
                    f"{', '.join(f'current.{column}' for column in METRICS_COLUMNS)}, "
                    f"{source_path}",
                    f"futures_metrics AS current {archive_join}",
                    "current.symbol=?",
                    [symbol],
                    "current.open_time",
                    start,
                    end,
                    include_previous,
                )
                source_table = "futures_metrics"
            else:
                legacy_frames: List[pd.DataFrame] = []
                if self._has_table(conn, "open_interest_history") and self._has_symbol(
                    conn, "open_interest_history", symbol
                ):
                    legacy_frames.append(
                        self._read_events(
                            conn,
                            (
                                "current.open_interest AS sum_open_interest, "
                                "current.open_interest_value AS sum_open_interest_value, "
                                "current.source_path AS open_interest_source_path"
                            ),
                            "open_interest_history AS current",
                            "current.symbol=? AND current.period='5m'",
                            [symbol],
                            "current.open_time",
                            start,
                            end,
                            include_previous,
                        )
                    )
                if self._has_table(conn, "positioning_history") and self._has_symbol(
                    conn, "positioning_history", symbol
                ):
                    legacy_frames.append(
                        self._read_events(
                            conn,
                            (
                                "current.count_toptrader_long_short_ratio, "
                                "current.sum_toptrader_long_short_ratio, "
                                "current.count_long_short_ratio, "
                                "current.sum_taker_long_short_vol_ratio, "
                                "current.source_path AS positioning_source_path"
                            ),
                            "positioning_history AS current",
                            "current.symbol=? AND current.period='5m'",
                            [symbol],
                            "current.open_time",
                            start,
                            end,
                            include_previous,
                        )
                    )
                if not legacy_frames:
                    raise ValueError("local metrics dataset is missing")
                frame = legacy_frames[0]
                for extra in legacy_frames[1:]:
                    frame = frame.merge(extra, on="event_time", how="outer")
                source_columns = [
                    column
                    for column in (
                        "open_interest_source_path",
                        "positioning_source_path",
                    )
                    if column in frame
                ]
                frame["source_path"] = (
                    frame[source_columns].bfill(axis=1).iloc[:, 0]
                    if source_columns
                    else None
                )
                frame = frame.drop(columns=source_columns)
                for column in METRICS_COLUMNS:
                    if column not in frame:
                        frame[column] = pd.NA
                frame = frame[["event_time", *METRICS_COLUMNS, "source_path"]]
                source_table = "legacy_metrics"
        finally:
            conn.close()
        output = self._finish_events(
            frame,
            symbol,
            source_table,
            METRICS_COLUMNS,
        )
        quality = {}
        for column in METRICS_COLUMNS:
            values = output[column]
            invalid = values.notna() & (~np.isfinite(values) | values.lt(0))
            quality[column] = {
                "rows": len(output),
                "missing_rows": int(values.isna().sum()),
                "invalid_rows": int(invalid.sum()),
                "valid_rows": int((values.notna() & ~invalid).sum()),
            }
            output.loc[invalid, column] = np.nan
        output.attrs["field_quality"] = quality
        return output

    def load_liquidations(
        self,
        symbol: str,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> pd.DataFrame:
        """Read hourly liquidations usable at that hour's close, preserving flags.

        Original aligned/raw files remain unchanged. Late hour totals retain
        their event hour and receive time but are unavailable to that hour's
        signal. Counts can remain usable when only notional is invalid.
        """
        symbol = resolve_market_symbols(symbol).perpetual
        path = (
            self.db_path.parent / "derived" / "liquidation_alignment"
            / f"symbol={symbol}" / "aligned_1h.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(f"no aligned liquidation history for {symbol}: {path}")
        frame = pd.read_parquet(path)
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.tz is None
            or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError("liquidation timestamps must be unique, ordered and timezone-aware")
        required = {
            "symbol", "available_at", "liquidation_available_at",
            "liquidation_received_at_max", "liquidation_coverage_status",
            "invalid_notional_count", "liquidation_partition_mismatch",
            *LIQUIDATION_VALUE_COLUMNS,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"liquidation data missing columns: {sorted(missing)}")
        if not frame["symbol"].eq(symbol).all():
            raise ValueError("liquidation symbol does not match its file")
        if not frame["liquidation_coverage_status"].isin(["valid_event", "valid_zero", "source_unknown"]).all():
            raise ValueError("unsupported liquidation coverage status")
        frame.index = frame.index.tz_convert("UTC")
        if start is not None:
            frame = frame.loc[frame.index >= _to_utc_timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= _to_utc_timestamp(end)]
        if frame.empty:
            raise ValueError(f"no liquidation rows for {symbol} in requested range")
        frame = frame.copy()
        expected_close = frame.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        if not frame["available_at"].eq(expected_close).all():
            raise ValueError("liquidation bars must use hourly close availability")
        if (frame["liquidation_received_at_max"] > frame["liquidation_available_at"]).any():
            raise ValueError("liquidation availability precedes receipt")
        known = frame["liquidation_coverage_status"].isin(["valid_event", "valid_zero"])
        on_time = frame["liquidation_available_at"].le(frame["available_at"])
        counts = frame[list(LIQUIDATION_COUNT_COLUMNS)]
        counts_valid = (
            np.isfinite(counts).all(axis=1) & counts.ge(0).all(axis=1)
            & counts["liquidation_event_count"].eq(
                counts["long_liquidation_count"] + counts["short_liquidation_count"]
            )
        )
        amounts = frame[list(LIQUIDATION_NOTIONAL_COLUMNS)]
        amounts_valid = (
            frame["invalid_notional_count"].eq(0)
            & np.isfinite(amounts).all(axis=1) & amounts.ge(0).all(axis=1)
        )
        valid_zero = frame["liquidation_coverage_status"].eq("valid_zero")
        if (valid_zero & frame[list(LIQUIDATION_VALUE_COLUMNS)].ne(0).any(axis=1)).any():
            raise ValueError("valid-zero liquidation hours must contain explicit zero values")
        count_usable = known & on_time & counts_valid
        notional_usable = count_usable & amounts_valid
        frame["liquidation_count_usable"] = count_usable
        frame["liquidation_notional_usable"] = notional_usable
        frame["liquidation_status"] = "valid"
        frame.loc[~amounts_valid, "liquidation_status"] = "invalid_notional"
        frame.loc[~counts_valid, "liquidation_status"] = "invalid_count"
        frame.loc[~on_time, "liquidation_status"] = "late"
        frame.loc[~known, "liquidation_status"] = "source_unknown"
        frame.loc[~count_usable, list(LIQUIDATION_COUNT_COLUMNS)] = np.nan
        frame.loc[~notional_usable, list(LIQUIDATION_NOTIONAL_COLUMNS)] = np.nan
        frame.attrs.update({
            "symbol": symbol,
            "market_symbols": asdict(resolve_market_symbols(symbol)),
            "source_path": str(path),
            "quality_counts": {str(k): int(v) for k, v in frame["liquidation_status"].value_counts().items()},
            "missing_policy": "mask_unavailable_fields_at_hour_close_keep_original_files",
        })
        return frame

    def load_open_interest(
        self,
        symbol: str,
        period: str = "5m",
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> pd.DataFrame:
        """Load open interest through the unified metrics source."""
        if period != "5m":
            raise ValueError(f"unified futures metrics support only 5m, not {period}")
        symbol = str(symbol).upper()
        metrics = self.load_metrics(symbol, start=start, end=end)
        output = metrics.rename(
            columns={
                "sum_open_interest": "open_interest",
                "sum_open_interest_value": "open_interest_value",
            }
        )
        output = output.dropna(subset=["open_interest", "open_interest_value"])
        if output.empty:
            raise ValueError(f"no open-interest history for {symbol} {period}")
        output.insert(0, "period", period)
        output.insert(0, "symbol", symbol)
        output = output[
            ["symbol", "period", "open_interest", "open_interest_value", "source_path"]
        ]
        output.index.name = "open_time"
        output.attrs.update(metrics.attrs)
        return output

    def load_positioning(
        self,
        symbol: str,
        period: str = "5m",
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> pd.DataFrame:
        """Load positioning ratios through the unified metrics source."""
        if period != "5m":
            raise ValueError(f"unified futures metrics support only 5m, not {period}")
        symbol = str(symbol).upper()
        metrics = self.load_metrics(symbol, start=start, end=end)
        columns = list(METRICS_COLUMNS[2:])
        output = metrics.dropna(subset=columns)
        if output.empty:
            raise ValueError(f"no positioning history for {symbol} {period}")
        output.insert(0, "period", period)
        output.insert(0, "symbol", symbol)
        output = output[["symbol", "period", *columns, "source_path"]]
        output.index.name = "open_time"
        output.attrs.update(metrics.attrs)
        return output

    @staticmethod
    def _asof_join(
        bars: pd.DataFrame,
        events: pd.DataFrame,
        observed_at: str,
    ) -> pd.DataFrame:
        left = bars.reset_index().sort_values("timestamp")
        right = events.reset_index().rename(columns={"timestamp": observed_at})
        right = right.sort_values(observed_at)
        merged = pd.merge_asof(
            left,
            right,
            left_on="timestamp",
            right_on=observed_at,
            direction="backward",
            allow_exact_matches=True,
        )
        return merged.set_index("timestamp")

    def load_feature_frame(
        self,
        market: str,
        symbol: str,
        interval: str = "1h",
        price_type: str = "trade",
        start: Optional[object] = None,
        end: Optional[object] = None,
        include_funding: bool = False,
        include_metrics: bool = False,
    ) -> pd.DataFrame:
        """Causally align optional derivatives events to base bars.

        Each auxiliary observation is joined only to a bar at or after the
        observation timestamp. The original event time is retained so a
        strategy can enforce its own staleness limit.
        """
        frame = self.load_bars(
            market,
            symbol,
            interval=interval,
            price_type=price_type,
            start=start,
            end=end,
        )
        attrs = dict(frame.attrs)
        if include_funding:
            funding = self.load_funding(
                symbol, start=start, end=end, include_previous=True
            )[list(FUNDING_COLUMNS)]
            frame = self._asof_join(frame, funding, "funding_observed_at")
        if include_metrics:
            metrics = self.load_metrics(
                symbol, start=start, end=end, include_previous=True
            )[list(METRICS_COLUMNS)]
            frame = self._asof_join(frame, metrics, "metrics_observed_at")
        frame.attrs.update(attrs)
        frame.attrs["causal_asof_join"] = True
        return frame

    @staticmethod
    def _snapshot_item(
        symbol: str,
        row_count: object,
        start: object,
        end: object,
        period: Optional[str] = None,
    ) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "symbol": str(symbol),
            "row_count": int(row_count),
            "start": _iso_ms(int(start)) if start is not None else None,
            "end": _iso_ms(int(end)) if end is not None else None,
        }
        if period is not None:
            item["period"] = period
        return item

    def spot_snapshot(self) -> List[Dict[str, Any]]:
        """Return exact spot coverage grouped by symbol and interval."""
        conn = self._connect()
        try:
            if not self._has_table(conn, "klines"):
                return []
            rows = conn.execute(
                """SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time)
                   FROM klines GROUP BY symbol, interval ORDER BY symbol, interval"""
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "symbol": str(symbol),
                "interval": str(interval),
                "rows": int(row_count),
                "start": _iso_ms(start),
                "end": _iso_ms(end),
            }
            for symbol, interval, row_count, start, end in rows
        ]

    def futures_price_snapshot(
        self, price_type: str = "trade"
    ) -> List[Dict[str, Any]]:
        """Return USD-M price-bar coverage without scanning per symbol.

        Registered archive manifests are preferred because they provide a
        compact, hash-backed inventory.  Exact table rows are used only for a
        database that predates that manifest layer.
        """
        price_type = str(price_type).lower()
        if price_type not in PRICE_TYPE_TO_STORAGE:
            raise ValueError(f"unsupported price_type: {price_type}")
        category = PRICE_TYPE_TO_STORAGE[price_type]
        conn = self._connect()
        try:
            has_manifests = (
                self._has_table(conn, "futures_archive_files")
                and conn.execute(
                    "SELECT 1 FROM futures_archive_files "
                    "WHERE category=? AND row_count>0 LIMIT 1",
                    (category,),
                ).fetchone()
                is not None
            )
            if has_manifests:
                monthly_period = "substr(source_key,-11,7)"
                rows = conn.execute(
                    f"""SELECT symbol, interval, COALESCE(SUM(row_count),0),
                               MIN({monthly_period}), MAX({monthly_period})
                        FROM futures_archive_files
                        WHERE category=? AND row_count>0
                          AND {monthly_period} GLOB
                              '[12][0-9][0-9][0-9]-[01][0-9]'
                        GROUP BY symbol, interval ORDER BY symbol, interval""",
                    (category,),
                ).fetchall()
                return [
                    {
                        "symbol": str(symbol),
                        "interval": str(interval),
                        "rows": int(row_count),
                        "start": self._period_bounds(str(start), False, False),
                        "end": self._period_bounds(str(end), False, True),
                        "coverage_precision": "archive_manifest",
                    }
                    for symbol, interval, row_count, start, end in rows
                ]
            if not self._has_table(conn, "futures_price_bars"):
                return []
            rows = conn.execute(
                """SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time)
                   FROM futures_price_bars WHERE data_type=?
                   GROUP BY symbol, interval ORDER BY symbol, interval""",
                (category,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "symbol": str(symbol),
                "interval": str(interval),
                "rows": int(row_count),
                "start": _iso_ms(start),
                "end": _iso_ms(end),
                "coverage_precision": "exact",
            }
            for symbol, interval, row_count, start, end in rows
        ]

    def funding_snapshot(self) -> List[Dict[str, Any]]:
        """Return funding coverage, preferring consolidated rows per symbol."""
        conn = self._connect()
        try:
            selected: Dict[str, Dict[str, Any]] = {}
            if self._has_table(conn, "futures_funding_rates"):
                for symbol, row_count, start, end in conn.execute(
                    """SELECT symbol, COUNT(*), MIN(funding_time), MAX(funding_time)
                       FROM futures_funding_rates GROUP BY symbol ORDER BY symbol"""
                ):
                    selected[str(symbol)] = self._snapshot_item(
                        str(symbol), row_count, start, end
                    )
            if self._has_table(conn, "funding_rates"):
                for symbol, row_count, start, end in conn.execute(
                    """SELECT symbol, COUNT(*), MIN(funding_time), MAX(funding_time)
                       FROM funding_rates GROUP BY symbol ORDER BY symbol"""
                ):
                    selected.setdefault(
                        str(symbol),
                        self._snapshot_item(str(symbol), row_count, start, end),
                    )
        finally:
            conn.close()
        return [selected[symbol] for symbol in sorted(selected)]

    def _metrics_snapshots(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self._metrics_snapshot_cache is not None:
            return self._metrics_snapshot_cache
        conn = self._connect()
        try:
            open_interest: Dict[tuple[str, str], Dict[str, Any]] = {}
            positioning: Dict[tuple[str, str], Dict[str, Any]] = {}
            has_metric_manifests = (
                self._has_table(conn, "futures_archive_files")
                and conn.execute(
                    "SELECT 1 FROM futures_archive_files "
                    "WHERE category='metrics' LIMIT 1"
                ).fetchone()
                is not None
            )
            if has_metric_manifests:
                daily_period = "substr(source_key,-14,10)"
                rows = conn.execute(
                    f"""SELECT symbol, COALESCE(SUM(row_count),0),
                               MIN({daily_period}), MAX({daily_period})
                        FROM futures_archive_files
                        WHERE category='metrics'
                          AND {daily_period} GLOB
                              '[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]'
                        GROUP BY symbol ORDER BY symbol"""
                ).fetchall()
                for symbol, row_count, start, end in rows:
                    if int(row_count or 0) <= 0:
                        continue
                    key = (str(symbol), "5m")
                    item = {
                        "symbol": str(symbol),
                        "period": "5m",
                        "row_count": int(row_count),
                        "start": self._period_bounds(str(start), True, False),
                        "end": self._period_bounds(str(end), True, True),
                    }
                    open_interest[key] = dict(item)
                    positioning[key] = dict(item)
            elif self._has_table(conn, "futures_metrics"):
                rows = conn.execute(
                    """SELECT symbol,
                              SUM(CASE WHEN sum_open_interest IS NOT NULL
                                            AND sum_open_interest_value IS NOT NULL
                                       THEN 1 ELSE 0 END),
                              MIN(CASE WHEN sum_open_interest IS NOT NULL
                                            AND sum_open_interest_value IS NOT NULL
                                       THEN open_time END),
                              MAX(CASE WHEN sum_open_interest IS NOT NULL
                                            AND sum_open_interest_value IS NOT NULL
                                       THEN open_time END),
                              SUM(CASE WHEN count_toptrader_long_short_ratio IS NOT NULL
                                            AND sum_toptrader_long_short_ratio IS NOT NULL
                                            AND count_long_short_ratio IS NOT NULL
                                            AND sum_taker_long_short_vol_ratio IS NOT NULL
                                       THEN 1 ELSE 0 END),
                              MIN(CASE WHEN count_toptrader_long_short_ratio IS NOT NULL
                                            AND sum_toptrader_long_short_ratio IS NOT NULL
                                            AND count_long_short_ratio IS NOT NULL
                                            AND sum_taker_long_short_vol_ratio IS NOT NULL
                                       THEN open_time END),
                              MAX(CASE WHEN count_toptrader_long_short_ratio IS NOT NULL
                                            AND sum_toptrader_long_short_ratio IS NOT NULL
                                            AND count_long_short_ratio IS NOT NULL
                                            AND sum_taker_long_short_vol_ratio IS NOT NULL
                                       THEN open_time END)
                       FROM futures_metrics GROUP BY symbol ORDER BY symbol"""
                ).fetchall()
                for symbol, oi_count, oi_start, oi_end, pos_count, pos_start, pos_end in rows:
                    key = (str(symbol), "5m")
                    if int(oi_count or 0) > 0:
                        open_interest[key] = self._snapshot_item(
                            str(symbol), oi_count, oi_start, oi_end, "5m"
                        )
                    if int(pos_count or 0) > 0:
                        positioning[key] = self._snapshot_item(
                            str(symbol), pos_count, pos_start, pos_end, "5m"
                        )
            if self._has_table(conn, "open_interest_history"):
                for symbol, period, row_count, start, end in conn.execute(
                    """SELECT symbol, period, COUNT(*), MIN(open_time), MAX(open_time)
                       FROM open_interest_history
                       GROUP BY symbol, period ORDER BY symbol, period"""
                ):
                    key = (str(symbol), str(period))
                    open_interest.setdefault(
                        key,
                        self._snapshot_item(
                            str(symbol), row_count, start, end, str(period)
                        ),
                    )
            if self._has_table(conn, "positioning_history"):
                for symbol, period, row_count, start, end in conn.execute(
                    """SELECT symbol, period, COUNT(*), MIN(open_time), MAX(open_time)
                       FROM positioning_history
                       GROUP BY symbol, period ORDER BY symbol, period"""
                ):
                    key = (str(symbol), str(period))
                    positioning.setdefault(
                        key,
                        self._snapshot_item(
                            str(symbol), row_count, start, end, str(period)
                        ),
                    )
        finally:
            conn.close()
        self._metrics_snapshot_cache = (
            [open_interest[key] for key in sorted(open_interest)],
            [positioning[key] for key in sorted(positioning)],
        )
        return self._metrics_snapshot_cache

    def open_interest_snapshot(self) -> List[Dict[str, Any]]:
        """Return exact coverage for usable open-interest observations."""
        return self._metrics_snapshots()[0]

    def positioning_snapshot(self) -> List[Dict[str, Any]]:
        """Return exact coverage for usable positioning observations."""
        return self._metrics_snapshots()[1]

    @staticmethod
    def _exact_coverage(
        conn: sqlite3.Connection,
        dataset: str,
        market: str,
        table: str,
        time_column: str,
        interval: str,
        price_type: Optional[str],
        symbol: str,
        extra_sql: str = "",
        extra_params: Sequence[Any] = (),
    ) -> Optional[DatasetCoverage]:
        row = conn.execute(
            f"""SELECT COUNT(*), MIN({time_column}), MAX({time_column})
                FROM {table} WHERE symbol=? {extra_sql}""",
            (symbol, *extra_params),
        ).fetchone()
        if row is None or int(row[0]) == 0:
            return None
        return DatasetCoverage(
            dataset=dataset,
            market=market,
            symbol=symbol,
            interval=interval,
            price_type=price_type,
            symbols=1,
            rows=int(row[0]),
            start=_iso_ms(row[1]),
            end=_iso_ms(row[2]),
        )

    @staticmethod
    def _period_bounds(period: str, daily: bool, end: bool) -> str:
        if daily:
            stamp = pd.Timestamp(period, tz="UTC")
            if end:
                stamp += pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            return stamp.isoformat()
        year, month = (int(part) for part in period.split("-"))
        day = calendar.monthrange(year, month)[1] if end else 1
        stamp = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
        if end:
            stamp += pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        return stamp.isoformat()

    def _summary_catalog(self, conn: sqlite3.Connection) -> List[DatasetCoverage]:
        output: List[DatasetCoverage] = []
        if self._has_table(conn, "klines"):
            for interval, symbols, rows, start, end in conn.execute(
                """SELECT interval, COUNT(DISTINCT symbol), COUNT(*),
                          MIN(open_time), MAX(open_time)
                   FROM klines GROUP BY interval ORDER BY interval"""
            ):
                output.append(
                    DatasetCoverage(
                        "spot.trade_bars",
                        SPOT,
                        None,
                        str(interval),
                        "trade",
                        int(symbols),
                        int(rows),
                        _iso_ms(start),
                        _iso_ms(end),
                    )
                )

        if self._has_table(conn, "futures_archive_files"):
            daily_period = "substr(source_key,-14,10)"
            monthly_period = "substr(source_key,-11,7)"
            period = (
                "CASE "
                f"WHEN category='metrics' AND {daily_period} GLOB "
                "'[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]' "
                f"THEN {daily_period} "
                f"WHEN category<>'metrics' AND {monthly_period} GLOB "
                "'[12][0-9][0-9][0-9]-[01][0-9]' "
                f"THEN {monthly_period} ELSE NULL END"
            )
            rows = conn.execute(
                f"""SELECT category, interval, COUNT(DISTINCT symbol),
                           COALESCE(SUM(row_count),0), MIN({period}), MAX({period})
                    FROM futures_archive_files
                    GROUP BY category, interval ORDER BY category, interval"""
            ).fetchall()
            for category, interval, symbols, row_count, start, end in rows:
                if category in STORAGE_TO_PRICE_TYPE:
                    dataset = f"usd_m.{STORAGE_TO_PRICE_TYPE[category]}_bars"
                    price_type = STORAGE_TO_PRICE_TYPE[category]
                elif category == "fundingRate":
                    dataset, price_type = "usd_m.funding", None
                elif category == "metrics":
                    dataset, price_type = "usd_m.metrics", None
                else:
                    continue
                daily = category == "metrics"
                output.append(
                    DatasetCoverage(
                        dataset,
                        USD_M_PERPETUAL,
                        None,
                        str(interval),
                        price_type,
                        int(symbols),
                        int(row_count),
                        self._period_bounds(str(start), daily, False) if start else None,
                        self._period_bounds(str(end), daily, True) if end else None,
                        coverage_precision="archive_manifest",
                    )
                )
        existing = {item.dataset for item in output}
        if "usd_m.funding" not in existing:
            funding = self.funding_snapshot()
            if funding:
                output.append(
                    DatasetCoverage(
                        "usd_m.funding",
                        USD_M_PERPETUAL,
                        None,
                        "native",
                        None,
                        len(funding),
                        sum(int(item["row_count"]) for item in funding),
                        min(str(item["start"]) for item in funding if item["start"]),
                        max(str(item["end"]) for item in funding if item["end"]),
                        coverage_precision="exact_merged_symbols",
                    )
                )
        if "usd_m.metrics" not in existing:
            metrics = self.open_interest_snapshot()
            if metrics:
                output.append(
                    DatasetCoverage(
                        "usd_m.metrics",
                        USD_M_PERPETUAL,
                        None,
                        "5m",
                        None,
                        len(metrics),
                        sum(int(item["row_count"]) for item in metrics),
                        min(str(item["start"]) for item in metrics if item["start"]),
                        max(str(item["end"]) for item in metrics if item["end"]),
                        coverage_precision="merged_sources",
                    )
                )
        return output

    def _symbol_catalog(
        self, conn: sqlite3.Connection, symbols: Sequence[str]
    ) -> List[DatasetCoverage]:
        output: List[DatasetCoverage] = []
        for symbol in symbols:
            if self._has_table(conn, "klines"):
                intervals = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT interval FROM klines WHERE symbol=? ORDER BY interval",
                        (symbol,),
                    )
                ]
                for interval in intervals:
                    item = self._exact_coverage(
                        conn,
                        "spot.trade_bars",
                        SPOT,
                        "klines",
                        "open_time",
                        interval,
                        "trade",
                        symbol,
                        "AND interval=?",
                        (interval,),
                    )
                    if item:
                        output.append(item)
            if self._has_table(conn, "futures_price_bars"):
                types = conn.execute(
                    """SELECT DISTINCT data_type, interval FROM futures_price_bars
                       WHERE symbol=? ORDER BY data_type, interval""",
                    (symbol,),
                ).fetchall()
                for data_type, interval in types:
                    price_type = STORAGE_TO_PRICE_TYPE.get(str(data_type))
                    if price_type is None:
                        continue
                    item = self._exact_coverage(
                        conn,
                        f"usd_m.{price_type}_bars",
                        USD_M_PERPETUAL,
                        "futures_price_bars",
                        "open_time",
                        str(interval),
                        price_type,
                        symbol,
                        "AND data_type=? AND interval=?",
                        (data_type, interval),
                    )
                    if item:
                        output.append(item)
            if self._has_table(conn, "futures_funding_rates"):
                item = self._exact_coverage(
                    conn,
                    "usd_m.funding",
                    USD_M_PERPETUAL,
                    "futures_funding_rates",
                    "funding_time",
                    "native",
                    None,
                    symbol,
                )
                if item:
                    output.append(item)
            if (
                not any(
                    item.dataset == "usd_m.funding" and item.symbol == symbol
                    for item in output
                )
                and self._has_table(conn, "funding_rates")
            ):
                item = self._exact_coverage(
                    conn,
                    "usd_m.funding",
                    USD_M_PERPETUAL,
                    "funding_rates",
                    "funding_time",
                    "native",
                    None,
                    symbol,
                )
                if item:
                    output.append(item)
            if self._has_table(conn, "futures_metrics"):
                item = self._exact_coverage(
                    conn,
                    "usd_m.metrics",
                    USD_M_PERPETUAL,
                    "futures_metrics",
                    "open_time",
                    "5m",
                    None,
                    symbol,
                )
                if item:
                    output.append(item)
            if (
                not any(
                    item.dataset == "usd_m.metrics" and item.symbol == symbol
                    for item in output
                )
                and self._has_table(conn, "open_interest_history")
            ):
                item = self._exact_coverage(
                    conn,
                    "usd_m.metrics",
                    USD_M_PERPETUAL,
                    "open_interest_history",
                    "open_time",
                    "5m",
                    None,
                    symbol,
                    "AND period='5m'",
                )
                if item:
                    output.append(item)
        return output

    def catalog(self, symbols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Describe available datasets without downloading or changing data."""
        requested = sorted({str(symbol).upper() for symbol in symbols or []})
        conn = self._connect()
        try:
            entries = (
                self._symbol_catalog(conn, requested)
                if requested
                else self._summary_catalog(conn)
            )
        finally:
            conn.close()
        return {
            "database": str(self.db_path),
            "mode": "exact_symbols" if requested else "summary",
            "symbols_requested": requested,
            "datasets": [entry.to_dict() for entry in entries],
        }
