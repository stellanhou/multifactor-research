from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std(ddof=0)


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def realized_volatility(
    close: pd.Series,
    window: int,
    bars_per_year: int,
) -> pd.Series:
    returns = simple_returns(close)
    return returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(
        bars_per_year
    )


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def previous_donchian(
    high: pd.Series,
    low: pd.Series,
    entry_window: int,
    exit_window: int,
) -> tuple[pd.Series, pd.Series]:
    """Channels known before the current bar begins."""
    upper = high.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = low.rolling(exit_window, min_periods=exit_window).min().shift(1)
    return upper, lower
