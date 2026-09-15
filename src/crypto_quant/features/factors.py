"""Causal derived factors built on the unified local market-data interface."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from crypto_quant.data_access.data import interval_to_ms
from crypto_quant.data_access.market_data import (
    FUNDING_COLUMNS,
    METRICS_COLUMNS,
    LIQUIDATION_VALUE_COLUMNS,
    MarketDataStore,
    SPOT,
    USD_M_PERPETUAL,
    _to_utc_timestamp,
    resolve_market_symbols,
)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    inputs: tuple[str, ...]
    formula: str
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FACTOR_DEFINITIONS: tuple[FactorDefinition, ...] = (
    FactorDefinition("basis_trade_spot", "basis", ("perp_close", "spot_close"), "perp_close / spot_close - 1", "Tradable perpetual premium to spot."),
    FactorDefinition("basis_mark_index", "basis", ("mark_close", "index_close"), "mark_close / index_close - 1", "Mark-price premium to the venue index."),
    FactorDefinition("trade_mark_spread", "basis", ("perp_close", "mark_close"), "perp_close / mark_close - 1", "Last-trade dislocation from mark price."),
    FactorDefinition("premium_index", "basis", ("premium_index_close",), "premium_index_close", "Venue premium-index close without reinterpretation."),
    FactorDefinition("basis_change_1bar", "basis", ("basis_trade_spot",), "basis_trade_spot.diff(1)", "One-bar change in trade-to-spot basis."),
    FactorDefinition("funding_rate", "carry", ("funding_events",), "latest funding_rate known by available_at", "Latest native funding payment rate."),
    FactorDefinition("funding_annualized", "carry", ("funding_rate", "funding_interval_hours"), "funding_rate * 24 / interval_hours * 365.25", "Simple annualized funding rate; not compounded."),
    FactorDefinition("funding_24h_sum", "carry", ("funding_events",), "rolling 24h event-rate sum", "Funding paid during the trailing 24 hours."),
    FactorDefinition("funding_7d_sum", "carry", ("funding_events",), "rolling 7d event-rate sum", "Funding paid during the trailing seven days."),
    FactorDefinition("open_interest", "leverage", ("futures_metrics",), "latest sum_open_interest known by available_at", "Aggregate contract open interest."),
    FactorDefinition("open_interest_value", "leverage", ("futures_metrics",), "latest sum_open_interest_value known by available_at", "Aggregate open-interest notional."),
    FactorDefinition("open_interest_change_1bar", "leverage", ("open_interest",), "open_interest.pct_change(1)", "One-bar open-interest change."),
    FactorDefinition("open_interest_change_24h", "leverage", ("open_interest",), "open_interest.pct_change(bars_per_24h)", "Trailing 24-hour open-interest change."),
    FactorDefinition("toptrader_account_net", "positioning", ("toptrader_account_long_short_ratio",), "(ratio - 1) / (ratio + 1)", "Bounded top-trader account imbalance."),
    FactorDefinition("toptrader_position_net", "positioning", ("toptrader_position_long_short_ratio",), "(ratio - 1) / (ratio + 1)", "Bounded top-trader position imbalance."),
    FactorDefinition("global_account_net", "positioning", ("global_account_long_short_ratio",), "(ratio - 1) / (ratio + 1)", "Bounded global account imbalance."),
    FactorDefinition("taker_flow_net", "positioning", ("taker_long_short_volume_ratio",), "(ratio - 1) / (ratio + 1)", "Bounded taker buy/sell volume imbalance."),
    FactorDefinition("spot_taker_buy_share", "flow", ("spot_taker_buy_quote_volume", "spot_quote_volume"), "taker_buy_quote_volume / quote_volume", "Spot taker-buy share of quoted turnover."),
    FactorDefinition("perp_taker_buy_share", "flow", ("perp_taker_buy_quote_volume", "perp_quote_volume"), "taker_buy_quote_volume / quote_volume", "Perpetual taker-buy share of quoted turnover."),
    FactorDefinition("spot_quote_volume_24h", "liquidity", ("spot_quote_volume",), "rolling 24h sum", "Trailing spot quote turnover."),
    FactorDefinition("perp_quote_volume_24h", "liquidity", ("perp_quote_volume",), "rolling 24h sum", "Trailing perpetual quote turnover."),
    FactorDefinition("perp_to_spot_quote_volume", "liquidity", ("perp_quote_volume_24h", "spot_quote_volume_24h"), "perp_quote_volume_24h / spot_quote_volume_24h", "Relative perpetual-to-spot trading activity."),
    FactorDefinition("spot_log_return_1bar", "returns", ("spot_close",), "log(spot_close).diff(1)", "One-bar spot log return."),
    FactorDefinition("perp_log_return_1bar", "returns", ("perp_close",), "log(perp_close).diff(1)", "One-bar perpetual log return."),
    FactorDefinition("spot_perp_return_spread_1bar", "returns", ("perp_log_return_1bar", "spot_log_return_1bar"), "perp return - spot return", "One-bar relative return."),
    FactorDefinition("perp_realized_vol_24h", "volatility", ("perp_log_return_1bar",), "rolling 24h std * sqrt(bars_per_year)", "Annualized trailing 24-hour perpetual volatility."),
    FactorDefinition("funding_age_hours", "freshness", ("funding_observed_at", "available_at"), "available_at - funding_observed_at", "Age of the funding observation at signal availability."),
    FactorDefinition("metrics_age_minutes", "freshness", ("metrics_observed_at", "available_at"), "available_at - metrics_observed_at", "Age of the metrics observation at signal availability."),
)

FACTOR_COLUMNS = tuple(definition.name for definition in FACTOR_DEFINITIONS)

BAR_SOURCES = (
    ("spot", SPOT, "trade"),
    ("perp", USD_M_PERPETUAL, "trade"),
    ("mark", USD_M_PERPETUAL, "mark"),
    ("index", USD_M_PERPETUAL, "index"),
    ("premium_index", USD_M_PERPETUAL, "premium_index"),
)


def factor_catalog() -> Dict[str, Any]:
    return {
        "factor_count": len(FACTOR_DEFINITIONS),
        "time_contract": (
            "Each row is known at available_at (the base bar close); "
            "orders may execute no earlier than the next bar."
        ),
        "missing_input_policy": "return_nan_and_report_input_availability",
        "definitions": [definition.to_dict() for definition in FACTOR_DEFINITIONS],
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = (
        numerator.notna() & np.isfinite(numerator)
        & denominator.notna() & np.isfinite(denominator) & (denominator != 0.0)
    )
    output = pd.Series(np.nan, index=numerator.index, dtype=float)
    output.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return output


def _log_return(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    clean = numeric.where(numeric > 0.0)
    return np.log(clean).diff()


def _bounded_ratio(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric) & numeric.ge(0))
    return _safe_ratio(numeric - 1.0, numeric + 1.0)


class FactorEngine:
    """Build a stable, causal factor table for one symbol and bar interval."""

    def __init__(self, data: MarketDataStore) -> None:
        self.data = data

    @staticmethod
    def _query_start(start: Optional[object], warmup_days: int) -> Optional[pd.Timestamp]:
        if start is None:
            return None
        return _to_utc_timestamp(start) - pd.Timedelta(days=warmup_days)

    def _load_optional_bars(
        self,
        market: str,
        symbol: str,
        interval: str,
        price_type: str,
        start: Optional[object],
        end: Optional[object],
    ) -> Optional[pd.DataFrame]:
        try:
            return self.data.load_bars(
                market,
                symbol,
                interval=interval,
                price_type=price_type,
                start=start,
                end=end,
            )
        except ValueError:
            return None

    @staticmethod
    def _attach_bar_source(
        frame: pd.DataFrame,
        source: Optional[pd.DataFrame],
        prefix: str,
        price_divisor: int = 1,
    ) -> bool:
        observed_column = f"{prefix}_observed_at"
        frame[observed_column] = pd.Series(
            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
        )
        columns = {"close": 1.0 / price_divisor}
        if prefix in {"spot", "perp"}:
            columns.update({
                "open": 1.0 / price_divisor, "high": 1.0 / price_divisor,
                "low": 1.0 / price_divisor, "volume": price_divisor,
                "taker_buy_base_volume": price_divisor,
                "quote_volume": 1, "taker_buy_quote_volume": 1, "trades": 1,
            })
        for column in columns:
            frame[f"{prefix}_{column}"] = np.nan
        if source is None:
            return False
        aligned = source.reindex(frame.index)
        known = aligned["close_time"].notna() & (
            aligned["close_time"] <= frame["available_at"]
        )
        frame.loc[known, observed_column] = aligned.loc[known, "close_time"]
        for column, multiplier in columns.items():
            values = aligned.loc[known, column] * multiplier
            frame.loc[known, f"{prefix}_{column}"] = values.where(np.isfinite(values))
        return bool(known.any())

    @staticmethod
    def _funding_window(
        funding: pd.DataFrame, available_at: pd.Series, hours: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum distinct events in (signal time - window, signal time].

        Coverage uses only events already observed at each signal time. A missing
        scheduled settlement invalidates the window rather than becoming zero.
        The schedule check uses the intervals stored on those observed events.
        """
        event_times = funding.index.as_unit("ns").asi8
        # Binance settlement records can be a few milliseconds after the nominal
        # hour. Use the settlement hour only to check the hourly schedule; actual
        # event timestamps still govern availability and window inclusion.
        settlement_hours = funding.index.floor("h").as_unit("ns").asi8
        query_times = pd.DatetimeIndex(available_at).as_unit("ns").asi8
        lower = query_times - pd.Timedelta(hours=hours).value
        rates = funding["funding_rate"].to_numpy(dtype=float)
        intervals = funding["funding_interval_hours"].to_numpy(dtype=float)
        valid = np.isfinite(rates) & np.isfinite(intervals) & (intervals > 0)
        right = np.searchsorted(event_times, query_times, side="right")
        left = np.searchsorted(event_times, lower, side="right")
        prefix = np.r_[0.0, np.cumsum(np.where(valid, rates, 0.0))]
        invalid = np.r_[0, np.cumsum(~valid)]
        values = prefix[right] - prefix[left]
        status = np.full(len(query_times), "valid", dtype=object)
        status[lower < event_times[0]] = "insufficient_history"
        status[invalid[right] != invalid[left]] = "invalid_event"
        latest = np.maximum(right - 1, 0)
        overdue = query_times - settlement_hours[latest] >= intervals[latest] * 3_600_000_000_000
        status[(right > 0) & overdue] = "missing_settlement"
        status[(right > 0) & ~valid[latest]] = "invalid_event"
        # A gap is knowable only once its right-hand event has been observed.
        gaps = np.flatnonzero(
            np.diff(settlement_hours) > intervals[1:] * 3_600_000_000_000
        ) + 1
        for position in gaps:
            gap_end = event_times[position]
            last_missing_hour = settlement_hours[position] - intervals[position] * 3_600_000_000_000
            status[(query_times >= gap_end) & (lower < last_missing_hour)] = "missing_settlement"
        status[right == 0] = "missing_event"
        values[status != "valid"] = np.nan
        return values, status

    @staticmethod
    def _asof_available(
        frame: pd.DataFrame,
        events: pd.DataFrame,
        observed_at: str,
    ) -> pd.DataFrame:
        left = frame.reset_index().sort_values("available_at")
        right = events.reset_index().rename(columns={"timestamp": observed_at})
        right = right.sort_values(observed_at)
        merged = pd.merge_asof(
            left,
            right,
            left_on="available_at",
            right_on=observed_at,
            direction="backward",
            allow_exact_matches=True,
        )
        return merged.set_index("timestamp").sort_index()

    def _attach_funding(
        self,
        frame: pd.DataFrame,
        symbol: str,
        start: Optional[object],
        end: Optional[object],
    ) -> tuple[pd.DataFrame, bool]:
        frame["funding_observed_at"] = pd.Series(
            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
        )
        frame["funding_rate"] = np.nan
        frame["funding_interval_hours"] = np.nan
        frame["funding_24h_sum"] = np.nan
        frame["funding_7d_sum"] = np.nan
        frame["funding_24h_status"] = "missing_event"
        frame["funding_7d_status"] = "missing_event"
        try:
            funding = self.data.load_funding(
                symbol, start=start, end=end, include_previous=True
            )[list(FUNDING_COLUMNS)]
        except ValueError:
            return frame, False
        funding = funding.copy()
        if not funding.index.is_unique:
            raise ValueError("funding event timestamps must be unique")
        windows = {
            name: self._funding_window(funding, frame["available_at"], hours)
            for name, hours in (("24h", 24), ("7d", 168))
        }
        for column in ("funding_rate", "funding_interval_hours"):
            funding[column] = funding[column].where(np.isfinite(funding[column]))
        funding["funding_interval_hours"] = funding["funding_interval_hours"].where(
            funding["funding_interval_hours"] > 0
        )
        empty_columns = [
            "funding_observed_at",
            "funding_rate",
            "funding_interval_hours",
            "funding_24h_sum",
            "funding_7d_sum",
        ]
        frame = frame.drop(columns=empty_columns)
        frame = self._asof_available(frame, funding, "funding_observed_at")
        for name, (values, status) in windows.items():
            frame[f"funding_{name}_sum"] = values
            frame[f"funding_{name}_status"] = status
        return frame, True

    def _attach_metrics(
        self,
        frame: pd.DataFrame,
        symbol: str,
        start: Optional[object],
        end: Optional[object],
    ) -> tuple[pd.DataFrame, bool]:
        output_columns = {
            "sum_open_interest": "open_interest",
            "sum_open_interest_value": "open_interest_value",
            "count_toptrader_long_short_ratio": "toptrader_account_long_short_ratio",
            "sum_toptrader_long_short_ratio": "toptrader_position_long_short_ratio",
            "count_long_short_ratio": "global_account_long_short_ratio",
            "sum_taker_long_short_vol_ratio": "taker_long_short_volume_ratio",
        }
        frame["metrics_observed_at"] = pd.Series(
            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
        )
        frame["metrics_status"] = "missing"
        frame.attrs["metrics_source_quality"] = {}
        for column in output_columns.values():
            frame[column] = np.nan
        try:
            metrics = self.data.load_metrics(
                symbol, start=start, end=end, include_previous=True
            )
        except ValueError:
            return frame, False
        source_quality = metrics.attrs["field_quality"]
        metrics = metrics[list(METRICS_COLUMNS)].rename(columns=output_columns)
        empty_columns = ["metrics_observed_at", *output_columns.values()]
        frame = frame.drop(columns=empty_columns)
        frame = self._asof_available(frame, metrics, "metrics_observed_at")
        age = frame["available_at"] - frame["metrics_observed_at"]
        # A native 5m observation expires when the next observation is due.
        # Never search farther back for a non-null value of one ratio.
        stale = age >= pd.Timedelta(minutes=5)
        columns = list(output_columns.values())
        frame.loc[stale, columns] = np.nan
        frame["metrics_status"] = "valid"
        frame.loc[frame[columns].isna().any(axis=1), "metrics_status"] = "partial"
        frame.loc[frame[columns].isna().all(axis=1), "metrics_status"] = "missing"
        frame.loc[stale, "metrics_status"] = "stale"
        frame.attrs["metrics_source_quality"] = source_quality
        return frame, True

    @staticmethod
    def _derive(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        interval_ms = interval_to_ms(interval)
        day_ms = interval_to_ms("1d")
        bars_24h = max(1, int(round(day_ms / interval_ms)))
        periods_per_year = (365.25 * day_ms) / interval_ms

        frame["basis_trade_spot"] = _safe_ratio(
            frame["perp_close"], frame["spot_close"]
        ) - 1.0
        frame["basis_mark_index"] = _safe_ratio(
            frame["mark_close"], frame["index_close"]
        ) - 1.0
        frame["trade_mark_spread"] = _safe_ratio(
            frame["perp_close"], frame["mark_close"]
        ) - 1.0
        frame["premium_index"] = frame["premium_index_close"]
        frame["basis_change_1bar"] = frame["basis_trade_spot"].diff()

        interval_hours = frame["funding_interval_hours"].where(
            frame["funding_interval_hours"] > 0.0
        )
        frame["funding_annualized"] = (
            frame["funding_rate"] * 24.0 / interval_hours * 365.25
        )
        frame["open_interest_change_1bar"] = _safe_ratio(
            frame["open_interest"], frame["open_interest"].shift(1)
        ) - 1.0
        frame["open_interest_change_24h"] = _safe_ratio(
            frame["open_interest"], frame["open_interest"].shift(bars_24h)
        ) - 1.0
        frame["toptrader_account_net"] = _bounded_ratio(
            frame["toptrader_account_long_short_ratio"]
        )
        frame["toptrader_position_net"] = _bounded_ratio(
            frame["toptrader_position_long_short_ratio"]
        )
        frame["global_account_net"] = _bounded_ratio(
            frame["global_account_long_short_ratio"]
        )
        frame["taker_flow_net"] = _bounded_ratio(
            frame["taker_long_short_volume_ratio"]
        )
        frame["spot_taker_buy_share"] = _safe_ratio(
            frame["spot_taker_buy_quote_volume"], frame["spot_quote_volume"]
        )
        frame["perp_taker_buy_share"] = _safe_ratio(
            frame["perp_taker_buy_quote_volume"], frame["perp_quote_volume"]
        )
        frame["spot_quote_volume_24h"] = frame["spot_quote_volume"].rolling(
            bars_24h, min_periods=bars_24h
        ).sum()
        frame["perp_quote_volume_24h"] = frame["perp_quote_volume"].rolling(
            bars_24h, min_periods=bars_24h
        ).sum()
        frame["perp_to_spot_quote_volume"] = _safe_ratio(
            frame["perp_quote_volume_24h"], frame["spot_quote_volume_24h"]
        )
        frame["spot_log_return_1bar"] = _log_return(frame["spot_close"])
        frame["perp_log_return_1bar"] = _log_return(frame["perp_close"])
        frame["spot_perp_return_spread_1bar"] = (
            frame["perp_log_return_1bar"] - frame["spot_log_return_1bar"]
        )
        frame["perp_realized_vol_24h"] = frame[
            "perp_log_return_1bar"
        ].rolling(bars_24h, min_periods=bars_24h).std(ddof=0) * np.sqrt(
            periods_per_year
        )
        frame["funding_age_hours"] = (
            frame["available_at"] - frame["funding_observed_at"]
        ).dt.total_seconds() / 3_600.0
        frame["metrics_age_minutes"] = (
            frame["available_at"] - frame["metrics_observed_at"]
        ).dt.total_seconds() / 60.0
        return frame

    def load(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[object] = None,
        end: Optional[object] = None,
        base_market: str = USD_M_PERPETUAL,
        warmup_days: int = 30,
        include_liquidations: bool = False,
    ) -> pd.DataFrame:
        """Load base bars plus every available causal factor for one symbol."""
        symbol = str(symbol).upper()
        symbols = resolve_market_symbols(symbol)
        base_market = str(base_market).lower()
        if base_market not in {SPOT, USD_M_PERPETUAL}:
            raise ValueError(f"unsupported base_market: {base_market}")
        if warmup_days < 7:
            raise ValueError("warmup_days must be at least 7")
        interval_to_ms(interval)
        if include_liquidations and interval != "1h":
            raise ValueError("liquidation research inputs currently require 1h bars")
        query_start = self._query_start(start, warmup_days)

        base = self.data.load_bars(
            base_market,
            symbols.spot if base_market == SPOT else symbols.perpetual,
            interval=interval,
            price_type="trade",
            start=query_start,
            end=end,
        )
        frame = base[
            [
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
        ].copy()
        # Keep missing bars in their actual time positions before any rolling or
        # shift operation; raw OHLCV at those positions remains missing.
        grid = pd.date_range(
            frame.index.min(), frame.index.max(),
            freq=pd.Timedelta(milliseconds=interval_to_ms(interval)), name="timestamp",
        )
        frame = frame.reindex(grid)
        frame["base_bar_present"] = frame["close_time"].notna()
        frame["available_at"] = frame.index + pd.Timedelta(
            milliseconds=interval_to_ms(interval) - 1
        )

        loaded: Dict[str, Optional[pd.DataFrame]] = {}
        for prefix, market, price_type in BAR_SOURCES:
            if market == base_market and price_type == "trade":
                loaded[prefix] = base
            else:
                loaded[prefix] = self._load_optional_bars(
                    market,
                    symbols.spot if market == SPOT else symbols.perpetual,
                    interval,
                    price_type,
                    query_start,
                    end,
                )
        availability = {
            f"{prefix}_bars": self._attach_bar_source(
                frame, source, prefix,
                symbols.perpetual_multiplier if prefix in {"perp", "mark", "index"} else 1,
            )
            for prefix, source in loaded.items()
        }
        event_end = frame["available_at"].max()
        frame, availability["funding_events"] = self._attach_funding(
            frame, symbols.perpetual, query_start, event_end
        )
        frame, availability["futures_metrics"] = self._attach_metrics(
            frame, symbols.perpetual, query_start, event_end
        )
        frame["open_interest_base"] = frame["open_interest"] * symbols.perpetual_multiplier
        frame = self._derive(frame, interval)
        if include_liquidations:
            liquidations = self.data.load_liquidations(symbol, start=query_start, end=end)
            columns = [
                *LIQUIDATION_VALUE_COLUMNS, "liquidation_coverage_status",
                "liquidation_available_at", "liquidation_received_at_max",
                "liquidation_partition_mismatch", "invalid_notional_count",
                "liquidation_count_usable", "liquidation_notional_usable",
                "liquidation_status",
            ]
            attrs = dict(frame.attrs)
            frame = frame.join(liquidations[columns], how="left")
            frame.attrs.update(attrs)
            frame["liquidation_status"] = frame["liquidation_status"].fillna("missing")
            for column in ("liquidation_count_usable", "liquidation_notional_usable"):
                frame[column] = frame[column].eq(True)
            availability["liquidations"] = True

        future_funding = frame["funding_observed_at"].notna() & (
            frame["funding_observed_at"] > frame["available_at"]
        )
        future_metrics = frame["metrics_observed_at"].notna() & (
            frame["metrics_observed_at"] > frame["available_at"]
        )
        if future_funding.any() or future_metrics.any():
            raise RuntimeError("causal factor join selected a future observation")

        if start is not None:
            frame = frame.loc[frame.index >= _to_utc_timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= _to_utc_timestamp(end)]
        if frame.empty:
            raise ValueError(f"no factor rows for {symbol} {interval} in requested range")

        available_factors = [
            column for column in FACTOR_COLUMNS if frame[column].notna().any()
        ]
        coverage_columns = list(dict.fromkeys([
            "open_interest", "open_interest_base", "open_interest_value",
            "toptrader_account_long_short_ratio", "toptrader_position_long_short_ratio",
            "global_account_long_short_ratio", "taker_long_short_volume_ratio",
            *FACTOR_COLUMNS,
            *(LIQUIDATION_VALUE_COLUMNS if include_liquidations else ()),
        ]))
        field_coverage = {}
        for column in coverage_columns:
            valid = np.isfinite(frame[column])
            field_coverage[column] = {
                "rows": len(frame), "valid_rows": int(valid.sum()),
                "unavailable_rows": int((~valid).sum()),
                "coverage_ratio": float(valid.mean()),
            }
        frame.attrs.update(
            {
                "symbol": symbol,
                "interval": interval,
                "base_market": base_market,
                "time_contract": factor_catalog()["time_contract"],
                "input_availability": availability,
                "available_factors": available_factors,
                "unavailable_factors": sorted(set(FACTOR_COLUMNS) - set(available_factors)),
                "warmup_days": warmup_days,
                "market_symbols": asdict(symbols),
                "price_units": {
                    "base_ohlcv": "native_market_units",
                    "spot_perp_mark_index_close": "per_spot_base_asset_unit",
                    "spot_perp_ohlc": "per_spot_base_asset_unit",
                    "spot_perp_base_volume_and_open_interest_base": "spot_base_asset_units",
                    "premium_index": "dimensionless",
                },
                "field_coverage": field_coverage,
                "metrics_status_counts": {
                    str(k): int(v) for k, v in frame["metrics_status"].value_counts().items()
                },
                "funding_window_status_counts": {
                    name: {str(k): int(v) for k, v in frame[f"funding_{name}_status"].value_counts().items()}
                    for name in ("24h", "7d")
                },
            }
        )
        return frame

    @staticmethod
    def snapshot(frame: pd.DataFrame, tail: int = 3) -> Dict[str, Any]:
        """Return a compact JSON-safe summary of a factor frame."""
        sample = frame.tail(max(0, int(tail))).reset_index()
        records: List[Dict[str, Any]] = json.loads(
            sample.to_json(orient="records", date_format="iso", date_unit="ms")
        )
        return {
            "symbol": frame.attrs.get("symbol"),
            "interval": frame.attrs.get("interval"),
            "base_market": frame.attrs.get("base_market"),
            "rows": int(len(frame)),
            "start": frame.index[0].isoformat(),
            "end": frame.index[-1].isoformat(),
            "time_contract": frame.attrs.get("time_contract"),
            "input_availability": frame.attrs.get("input_availability", {}),
            "available_factors": frame.attrs.get("available_factors", []),
            "unavailable_factors": frame.attrs.get("unavailable_factors", []),
            "market_symbols": frame.attrs["market_symbols"],
            "price_units": frame.attrs["price_units"],
            "field_coverage": frame.attrs["field_coverage"],
            "metrics_status_counts": frame.attrs["metrics_status_counts"],
            "metrics_source_quality": frame.attrs["metrics_source_quality"],
            "funding_window_status_counts": frame.attrs["funding_window_status_counts"],
            "liquidation_status_counts": (
                {str(k): int(v) for k, v in frame["liquidation_status"].value_counts().items()}
                if "liquidation_status" in frame else None
            ),
            "tail": records,
        }


def load_factor_frame(
    db_path: Path,
    symbol: str,
    interval: str = "1h",
    start: Optional[object] = None,
    end: Optional[object] = None,
    base_market: str = USD_M_PERPETUAL,
    include_liquidations: bool = False,
) -> pd.DataFrame:
    """Convenience entry point for strategy scripts."""
    return FactorEngine(MarketDataStore(db_path)).load(
        symbol,
        interval=interval,
        start=start,
        end=end,
        base_market=base_market,
        include_liquidations=include_liquidations,
    )
