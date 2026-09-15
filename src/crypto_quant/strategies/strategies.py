from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

from crypto_quant.features.indicators import previous_donchian, realized_volatility, sma, zscore


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    hypothesis: str
    rationale: str
    failure_modes: str
    parameters: dict[str, float | int]
    generate: Callable[[pd.DataFrame, dict], pd.Series]


def _buy_hold(data: pd.DataFrame, params: dict) -> pd.Series:
    return pd.Series(1.0, index=data.index, name="buy_and_hold")


def _moving_average_trend(data: pd.DataFrame, params: dict) -> pd.Series:
    window = int(params["window"])
    trend = sma(data["close"], window)
    signal = (data["close"] > trend).astype(float)
    signal[trend.isna()] = 0.0
    return signal.rename(f"sma_trend_{window}")


def _time_series_momentum(data: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params["lookback"])
    momentum = data["close"].pct_change(lookback)
    signal = (momentum > float(params.get("threshold", 0.0))).astype(float)
    signal[momentum.isna()] = 0.0
    return signal.rename(f"tsmom_{lookback}")


def _donchian_breakout(data: pd.DataFrame, params: dict) -> pd.Series:
    entry_window = int(params["entry_window"])
    exit_window = int(params["exit_window"])
    upper, lower = previous_donchian(
        data["high"], data["low"], entry_window, exit_window
    )
    output = pd.Series(np.nan, index=data.index)
    position = 0.0
    for timestamp, close in data["close"].items():
        upper_value = upper.loc[timestamp]
        lower_value = lower.loc[timestamp]
        if np.isfinite(upper_value) and np.isfinite(lower_value):
            if close >= upper_value:
                position = 1.0
            elif close <= lower_value:
                position = 0.0
        output.loc[timestamp] = position
    return output.fillna(0.0).rename(f"donchian_{entry_window}_{exit_window}")


def _bollinger_mean_reversion(data: pd.DataFrame, params: dict) -> pd.Series:
    window = int(params["window"])
    entry_z = float(params["entry_z"])
    if entry_z <= 0:
        raise ValueError("entry_z must be a positive deviation magnitude")
    scores = zscore(data["close"], window)
    output = pd.Series(np.nan, index=data.index)
    position = 0.0
    for timestamp, value in scores.items():
        if np.isfinite(value):
            if value <= -entry_z:
                position = 1.0
            elif value >= 0.0:
                position = 0.0
        output.loc[timestamp] = position
    return output.fillna(0.0).rename(f"bollinger_mr_{window}_{entry_z:g}z")


def _volatility_targeted_trend(data: pd.DataFrame, params: dict) -> pd.Series:
    trend_window = int(params["trend_window"])
    vol_window = int(params["vol_window"])
    target_vol = float(params["target_annual_vol"])
    max_leverage = float(params["max_leverage"])
    bars_per_year = int(params["bars_per_year"])
    trend = sma(data["close"], trend_window)
    direction = np.sign(data["close"] - trend)
    # First version uses Binance spot. Negative-direction targets stay flat until a
    # futures execution layer explicitly supports borrowing or shorting.
    direction = direction.clip(lower=0.0)
    vol = realized_volatility(data["close"], vol_window, bars_per_year)
    sizing = (target_vol / vol).clip(upper=max_leverage).fillna(0.0)
    output = (direction * sizing).fillna(0.0)
    return output.rename(f"voltgt_trend_{trend_window}_{vol_window}")


def strategy_registry() -> dict[str, StrategySpec]:
    return {
        "buy_and_hold": StrategySpec(
            name="buy_and_hold",
            family="baseline",
            hypothesis="Crypto has delivered a positive equity risk premium over the sample.",
            rationale="The baseline separates market beta from timing skill.",
            failure_modes="Long drawdowns and regime-dependent terminal values.",
            parameters={},
            generate=_buy_hold,
        ),
        "sma_trend_200": StrategySpec(
            name="sma_trend_200",
            family="trend_following",
            hypothesis="Persistent order flow and slow information diffusion create trends.",
            rationale="A 200-bar filter owns strong regimes and steps aside after weakness.",
            failure_modes="Whipsaws near the mean and delayed entry after reversals.",
            parameters={"window": 200},
            generate=_moving_average_trend,
        ),
        "tsmom_168": StrategySpec(
            name="tsmom_168",
            family="time_series_momentum",
            hypothesis="Recent seven-day returns continue briefly before crowding reverses them.",
            rationale="Time-series momentum is widely studied across liquid futures and crypto.",
            failure_modes="Sharp reversal days and crowded liquidation cascades.",
            parameters={"lookback": 168, "threshold": 0.0},
            generate=_time_series_momentum,
        ),
        "donchian_96_48": StrategySpec(
            name="donchian_96_48",
            family="breakout",
            hypothesis="Breakouts from a 16-day range attract follow-through and survive eight days.",
            rationale="The rule buys confirmed highs and exits on a shorter adverse range.",
            failure_modes="False breakouts in compressed, low-volume ranges.",
            parameters={"entry_window": 96, "exit_window": 48},
            generate=_donchian_breakout,
        ),
        "bollinger_mr_96_2z": StrategySpec(
            name="bollinger_mr_96_2z",
            family="mean_reversion",
            hypothesis="Liquidation-driven dislocations below two deviations partially revert.",
            rationale="Only long/flat spot exposure is taken, with a zero-z exit.",
            failure_modes="Buying persistent impairment during structural drawdowns.",
            parameters={"window": 96, "entry_z": 2.0},
            generate=_bollinger_mean_reversion,
        ),
        "vol_target_trend_200_168": StrategySpec(
            name="vol_target_trend_200_168",
            family="volatility_targeting",
            hypothesis="Trend exposure improves risk-adjusted returns when sized inversely to volatility.",
            rationale="Volatility targeting stabilizes risk without changing the directional signal.",
            failure_modes="Volatility estimates lag regime shifts; spot cannot express negative trends.",
            parameters={
                "trend_window": 200,
                "vol_window": 168,
                "target_annual_vol": 0.40,
                "max_leverage": 1.0,
                "bars_per_year": 2190,
            },
            generate=_volatility_targeted_trend,
        ),
    }
