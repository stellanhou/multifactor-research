from __future__ import annotations

import sqlite3
import time
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from crypto_quant.backtesting.config import BINANCE_PUBLIC_BASE_URL


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
EXTENDED_CLOSE_TOLERANCE_MS = 5_000


def _to_utc_ms(value: object) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


def interval_to_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError as exc:
        raise ValueError(f"unsupported interval: {interval}") from exc


def _epoch_milliseconds(values: Any) -> np.ndarray:
    """Normalize millisecond or datetime-like timestamps to integer milliseconds."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not len(numeric):
        return numeric
    magnitude = float(np.nanmax(np.abs(numeric)))
    # pandas may expose DatetimeIndex values in ns, us, or ms depending on
    # the index resolution.  Normalize all three, while retaining seconds
    # for callers that supplied Unix-second values.
    if magnitude > 1e17:
        scale = 1_000_000.0
    elif magnitude > 1e14:
        scale = 1_000.0
    elif magnitude > 1e11:
        scale = 1.0
    else:
        scale = 0.001
    return (numeric / scale).astype(float)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            quote_volume REAL NOT NULL,
            trades INTEGER NOT NULL,
            taker_buy_base_volume REAL NOT NULL,
            taker_buy_quote_volume REAL NOT NULL,
            PRIMARY KEY (symbol, interval, open_time)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_log (
            fetched_at INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL,
            rows_received INTEGER NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.commit()


class BinancePublicClient:
    def __init__(
        self,
        base_url: str = BINANCE_PUBLIC_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1000,
    ) -> List[List[Any]]:
        response = self.session.get(
            f"{self.base_url}/api/v3/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": min(limit, 1000),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"unexpected kline response: {payload}")
        return payload

    def download_archive(self, url: str) -> Optional[bytes]:
        """Fetch one official public archive, returning None for a missing period."""
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return bytes(response.content)


def _kline_row(symbol: str, interval: str, row: Iterable[Any]) -> Tuple[Any, ...]:
    values = list(row)
    if len(values) < 12:
        raise ValueError(f"short kline row: {values}")
    def epoch_ms(value: Any) -> int:
        numeric = int(value)
        return numeric // 1_000 if abs(numeric) > 10**15 else numeric

    return (
        symbol.upper(),
        interval,
        epoch_ms(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
        epoch_ms(values[6]),
        float(values[7]),
        int(float(values[8])),
        float(values[9]),
        float(values[10]),
    )


def store_klines(conn: sqlite3.Connection, rows: Iterable[Tuple[Any, ...]]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    interval = str(payload[0][1])
    validate_kline_records(payload, interval)
    conn.executemany(
        """
        INSERT INTO klines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            close_time=excluded.close_time,
            quote_volume=excluded.quote_volume,
            trades=excluded.trades,
            taker_buy_base_volume=excluded.taker_buy_base_volume,
            taker_buy_quote_volume=excluded.taker_buy_quote_volume
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def validate_kline_records(
    rows: List[Tuple[Any, ...]],
    interval: str,
    expected_symbol: Optional[str] = None,
) -> None:
    """Reject malformed bars before they can contaminate local research data."""
    step = interval_to_ms(interval)
    price_fields = ("open", "high", "low", "close")
    volume_fields = ("volume", "quote_volume")
    taker_fields = (
        ("taker_buy_base_volume", "volume"),
        ("taker_buy_quote_volume", "quote_volume"),
    )
    next_opens: Dict[Tuple[str, str], List[int]] = {}
    for candidate in rows:
        if isinstance(candidate, tuple) and len(candidate) >= 3:
            next_opens.setdefault((str(candidate[0]).upper(), str(candidate[1])), []).append(int(candidate[2]))
    for values in next_opens.values():
        values.sort()

    for position, row in enumerate(rows, start=1):
        label = f"kline {position}"
        if not isinstance(row, tuple) or len(row) != 13:
            raise ValueError(f"{label} must contain exactly 13 fields")

        (
            symbol,
            row_interval,
            open_time,
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            close_time,
            quote_volume,
            trades,
            taker_base,
            taker_quote,
        ) = row

        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{label} has an invalid symbol")
        if expected_symbol and symbol.upper() != expected_symbol.upper():
            raise ValueError(
                f"{label} symbol {symbol} does not match {expected_symbol}"
            )
        if row_interval != interval:
            raise ValueError(
                f"{label} interval {row_interval} does not match {interval}"
            )

        try:
            open_ms = int(open_time)
            close_ms = int(close_time)
            trade_count = int(trades)
            numeric = {
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
                "quote_volume": float(quote_volume),
                "taker_buy_base_volume": float(taker_base),
                "taker_buy_quote_volume": float(taker_quote),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-numeric field") from exc

        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"{label} contains a non-finite value")
        if open_ms < 0 or open_ms % step != 0:
            raise ValueError(f"{label} open_time is not aligned to {interval}")
        close_duration = close_ms - open_ms
        if close_duration <= 0:
            raise ValueError(f"{label} close_time is not after open_time")
        if close_duration > step:
            following = [stamp for stamp in next_opens.get((symbol.upper(), row_interval), []) if stamp > open_ms]
            if close_duration > step + EXTENDED_CLOSE_TOLERANCE_MS or (following and following[0] <= close_ms):
                raise ValueError(f"{label} close_time exceeds its interval")
        if any(numeric[field] <= 0 for field in price_fields):
            raise ValueError(f"{label} contains a non-positive price")
        # A low-liquidity market may have a valid bar with no trades.  Keep
        # that bar in the research universe, but do not accept a half-empty
        # zero-volume record whose trade counters contradict one another.
        if numeric["volume"] == 0.0 or numeric["quote_volume"] == 0.0:
            if not (
                numeric["volume"] == 0.0
                and numeric["quote_volume"] == 0.0
                and trade_count == 0
                and numeric["taker_buy_base_volume"] == 0.0
                and numeric["taker_buy_quote_volume"] == 0.0
            ):
                raise ValueError(f"{label} zero-volume fields are inconsistent")
        if trade_count < 0:
            raise ValueError(f"{label} has a negative trade count")

        if high_price < max(open_price, close_price, low_price):
            raise ValueError(f"{label} high is below an OHLC value")
        if low_price > min(open_price, close_price, high_price):
            raise ValueError(f"{label} low is above an OHLC value")

        for taker_field, parent_field in taker_fields:
            tolerance = abs(numeric[parent_field]) * 1e-12
            if numeric[taker_field] < -tolerance:
                raise ValueError(f"{label} has a negative {taker_field}")
            if numeric[taker_field] > numeric[parent_field] + tolerance:
                raise ValueError(
                    f"{label} {taker_field} exceeds {parent_field}"
                )


def update_klines(
    db_path: Path,
    symbols: Iterable[str],
    interval: str,
    start: str,
    end: Optional[str] = None,
    client: Optional[BinancePublicClient] = None,
    request_sleep_seconds: float = 0.12,
) -> Dict[str, int]:
    client = client or BinancePublicClient()
    conn = connect(db_path)
    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end if end is not None else pd.Timestamp.now(tz="UTC"))
    counts: Dict[str, int] = {}

    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        cursor = start_ms
        received = 0
        while cursor <= end_ms:
            raw_rows = client.get_klines(symbol, interval, cursor, end_ms)
            if not raw_rows:
                break
            parsed = [_kline_row(symbol, interval, row) for row in raw_rows]
            validate_kline_records(parsed, interval, expected_symbol=symbol)
            received += store_klines(conn, parsed)
            last_open = int(parsed[-1][2])
            next_cursor = last_open + interval_to_ms(interval)
            if next_cursor <= cursor:
                raise RuntimeError("Binance kline pagination did not advance")
            cursor = next_cursor
            if last_open >= end_ms or len(raw_rows) < 1000:
                break
            time.sleep(request_sleep_seconds)

        conn.execute(
            "INSERT INTO fetch_log VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(pd.Timestamp.now(tz="UTC").timestamp() * 1000),
                symbol,
                interval,
                start_ms,
                end_ms,
                received,
                client.base_url,
            ),
        )
        conn.commit()
        counts[symbol] = received

    conn.close()
    return counts


def load_klines(
    db_path: Path,
    symbol: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Compatibility adapter for the unified local market-data store."""
    from crypto_quant.data_access.market_data import MarketDataStore, SPOT

    try:
        return MarketDataStore(db_path).load_bars(
            SPOT,
            symbol,
            interval=interval,
            start=start,
            end=end,
            include_incomplete=include_incomplete,
        )
    except ValueError as exc:
        raise ValueError(
            f"no data for {symbol} {interval}; run update-data first"
        ) from exc


def validate_klines(frame: pd.DataFrame, interval: str) -> Dict[str, Any]:
    required = ["open", "high", "low", "close", "volume", "quote_volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("timestamps are not monotonic")

    numeric = frame[required]
    price_numeric = frame[["open", "high", "low", "close"]]
    if (price_numeric <= 0).any().any():
        bad_columns = price_numeric.columns[(price_numeric <= 0).any()].tolist()
        raise ValueError(f"non-positive values in {bad_columns}")
    quality_columns = sorted(
        set(required)
        | {
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        }
    )
    missing_quality = sorted(set(quality_columns) - set(frame.columns))
    if missing_quality:
        raise ValueError(f"missing quality columns: {missing_quality}")
    quality_values = frame[quality_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if len(quality_values) and not np.isfinite(quality_values).all():
        raise ValueError("K-line quality fields contain non-finite values")
    zero_volume = (frame["volume"] == 0.0) | (frame["quote_volume"] == 0.0)
    inconsistent_zero = zero_volume & ~(
        (frame["volume"] == 0.0)
        & (frame["quote_volume"] == 0.0)
        & (frame["trades"] == 0)
        & (frame["taker_buy_base_volume"] == 0.0)
        & (frame["taker_buy_quote_volume"] == 0.0)
    )
    if inconsistent_zero.any():
        raise ValueError("zero-volume fields are inconsistent")
    if (frame["trades"] < 0).any():
        raise ValueError("K-line trades contain a negative value")
    if (frame["taker_buy_base_volume"] > frame["volume"] * (1 + 1e-12)).any():
        raise ValueError("taker_buy_base_volume exceeds volume")
    if (
        frame["taker_buy_quote_volume"]
        > frame["quote_volume"] * (1 + 1e-12)
    ).any():
        raise ValueError("taker_buy_quote_volume exceeds quote_volume")

    taker_base = frame["taker_buy_base_volume"].astype(float)
    taker_quote = frame["taker_buy_quote_volume"].astype(float)
    positive_taker_base = taker_base > 0.0
    implied_taker_price = taker_quote[positive_taker_base] / taker_base[
        positive_taker_base
    ]
    taker_price_outside_bar = (
        (implied_taker_price < frame.loc[positive_taker_base, "low"] * (1 - 1e-8))
        | (
            implied_taker_price
            > frame.loc[positive_taker_base, "high"] * (1 + 1e-8)
        )
    )
    zero_base_nonzero_quote = (~positive_taker_base) & (taker_quote > 0.0)
    taker_flow_inconsistencies = int(taker_price_outside_bar.sum()) + int(
        zero_base_nonzero_quote.sum()
    )

    invalid_range = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    )
    if invalid_range.any():
        raise ValueError(f"invalid high/low ranges at {invalid_range.idxmax()}")

    step = interval_to_ms(interval)
    epoch_index = _epoch_milliseconds(frame.index)
    misaligned = (epoch_index % step) != 0
    if misaligned.any():
        raise ValueError(
            f"open_time is not aligned to {interval} at {frame.index[misaligned][0]}"
        )
    if "close_time" in frame.columns:
        actual_close = _epoch_milliseconds(frame["close_time"])
        close_durations = actual_close - epoch_index
        extended = close_durations > step
        overlap = np.zeros(len(frame), dtype=bool)
        if len(frame) > 1:
            overlap[:-1] = extended[:-1] & (epoch_index[1:] <= actual_close[:-1])
        invalid_close = (close_durations <= 0) | (
            extended
            & ((close_durations > step + EXTENDED_CLOSE_TOLERANCE_MS) | overlap)
        )
        if invalid_close.any():
            raise ValueError(
                f"close_time is inconsistent with open_time at "
                f"{frame.index[invalid_close][0]}"
            )
    expected_count = int((epoch_index[-1] - epoch_index[0]) // step) + 1
    gap_positions = pd.Series(epoch_index).diff().fillna(step)
    short_close_count = 0
    extended_close_count = 0
    if "close_time" in frame.columns:
        close_durations = _epoch_milliseconds(frame["close_time"]) - epoch_index
        short_close_count = int(
            ((close_durations > 0) & (close_durations < step)).sum()
        )
        extended_close_count = int((close_durations > step).sum())
    report = {
        "rows": int(len(frame)),
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
        "missing_bars_by_span": expected_count - len(epoch_index),
        "gap_locations": int((gap_positions > step).sum()),
        "largest_gap_ms": int(gap_positions.max()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "short_close_bars": short_close_count,
        "extended_close_bars": extended_close_count,
        "taker_flow_inconsistencies": taker_flow_inconsistencies,
    }
    # Maintenance and listing transitions can create genuine gaps. Report them;
    # silently filling prices can fabricate returns.
    if report["missing_bars_by_span"] > 0:
        report["warning"] = "span contains missing intervals; prices were not filled"
    if short_close_count > 0:
        report["short_close_warning"] = (
            "span contains partial bars before maintenance/listing gaps; "
            "durations were retained rather than expanded"
        )
    if extended_close_count > 0:
        report["extended_close_warning"] = (
            "span contains terminal or maintenance bars with an extended close_time; "
            "prices were retained without alteration"
        )
    if taker_flow_inconsistencies > 0:
        report["taker_flow_warning"] = (
            "taker buy base/quote volumes imply prices outside the bar range; "
            "do not use taker-flow research until the affected history is refreshed"
        )
    return report


def latest_snapshot(db_path: Path) -> List[Dict[str, Any]]:
    """Compatibility adapter for :meth:`MarketDataStore.spot_snapshot`."""
    from crypto_quant.data_access.market_data import MarketDataStore

    return MarketDataStore(db_path).spot_snapshot()
