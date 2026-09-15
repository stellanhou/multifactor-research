from dataclasses import dataclass


BINANCE_PUBLIC_BASE_URL = "https://data-api.binance.vision"


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    min_trade_fraction: float = 0.001
    allow_short: bool = False
