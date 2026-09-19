"""One supported strategy adapter; no model-generated Python or downloads."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.strategies.relative_strength import generate_relative_strength_targets, run_relative_strength_backtest
from .contracts import RULES, StrategyResearchContract


class DataGap(ValueError):
    """Local data cannot support this explicitly requested experiment."""


def load_segment(db: Path, contract: StrategyResearchContract, stage: str) -> dict:
    if stage not in {"development", "validation"}:
        raise ValueError("unknown research stage")
    start = pd.Timestamp(contract.development_start if stage == "development" else contract.validation_start)
    end = pd.Timestamp(contract.validation_start if stage == "development" else contract.validation_end)
    # Same grid for every parameter choice, including the longest warmup.
    first = start - pd.Timedelta(hours=max(contract.parameter_space["lookback"]) + 1)
    expected = pd.date_range(first, end, freq="h", inclusive="left").as_unit("ns")
    store = MarketDataStore(db)
    frames = {}
    for symbol in RULES["symbols"]:
        try:
            frame = store.load_bars("spot", symbol, "1h", start=first, end=end - pd.Timedelta(hours=1))
        except (ValueError, FileNotFoundError) as exc:
            raise DataGap(f"{symbol}: {exc}") from exc
        frame.index = frame.index.as_unit("ns")
        if not frame.index.equals(expected):
            missing = expected.difference(frame.index)
            raise DataGap(f"{symbol}: hourly grid mismatch; missing={len(missing)}, first={list(missing[:3])}")
        values = frame[["open", "high", "low", "close", "quote_volume"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise DataGap(f"{symbol}: nonfinite or invalid prices/quote volume")
        if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
                (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
            raise DataGap(f"{symbol}: inconsistent OHLC prices")
        expected_close = frame.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        if frame.close_time.isna().any() or not (frame.close_time == expected_close).all():
            raise DataGap(f"{symbol}: close_time differs from hourly availability contract")
        frames[symbol] = frame
    return frames


def save_snapshot(frames: dict, root: Path) -> dict:
    root.mkdir()
    evidence = {}
    for symbol, frame in frames.items():
        path = root / f"{symbol}.csv"
        frame.to_csv(path)
        evidence[symbol] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rows": len(frame), "first": frame.index[0].isoformat(), "last": frame.index[-1].isoformat()}
    return evidence


def metrics(result) -> dict:
    # Include the initial cash anchor so first-fill costs affect return/drawdown.
    path = np.r_[result.config.initial_capital, result.equity.to_numpy()]
    returns = path[1:] / path[:-1] - 1
    std = float(np.std(returns, ddof=1))
    return {
        "net_return": float(path[-1] / path[0] - 1),
        "max_drawdown": float(-(path / np.maximum.accumulate(path) - 1).min()),
        "annualized_sharpe": float(np.mean(returns) / std * np.sqrt(365 * 24)) if std > 0 else None,
        "traded_bars": int((result.weights.trade_notional > 0).sum()),
        "closed_trade_count": len(result.trades),
        "turnover": float(result.weights.turnover.sum()),
        "fees": float(result.weights.fee.sum()), "slippage_cost": float(result.weights.slippage_cost.sum()),
        "terminal_equity": float(path[-1]), "valuation": "mark to final close; no forced liquidation",
    }


def evaluate(frames: dict, parameters: dict, contract: StrategyResearchContract, stage: str, root: Path) -> dict:
    contract.parameters(parameters)
    start = pd.Timestamp(contract.development_start if stage == "development" else contract.validation_start)
    targets, _ = generate_relative_strength_targets(frames, **parameters)
    # One pre-start decision bar, fresh cash account at each segment boundary.
    index = frames["BTCUSDT"].index
    index = index[index >= start - pd.Timedelta(hours=1)]
    market = {s: f.loc[index] for s, f in frames.items()}
    target = {s: t.loc[index] for s, t in targets.items()}
    costs = {k: v for k, v in contract.costs.items() if k != "stress_multiplier"}
    root.mkdir()
    evidence = {}
    results = {}
    for name in ("strategy", "benchmark", "stress"):
        config = dict(costs)
        weights = target
        if name == "benchmark":
            weights = {s: pd.Series(0.5, index=index) for s in frames}
        if name == "stress":
            for cost in ("fee_bps", "slippage_bps"):
                config[cost] *= contract.costs["stress_multiplier"]
        result = run_relative_strength_backtest(market, weights, config=BacktestConfig(**config))
        results[name] = metrics(result)
        paths = {"equity": result.equity.to_frame(), "executions": result.weights, "closed_trades": result.trades,
                 "targets": pd.DataFrame(weights)}
        evidence[name] = {}
        for kind, table in paths.items():
            path = root / f"{name}-{kind}.csv"
            table.to_csv(path)
            evidence[name][kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    results["excess_return"] = results["strategy"]["net_return"] - results["benchmark"]["net_return"]
    return {"stage": stage, "start": start.isoformat(), "end_exclusive":
            contract.validation_start if stage == "development" else contract.validation_end,
            "parameters": parameters, "results": results, "artifacts": evidence,
            "scope": "adaptive development" if stage == "development" else "held out within this run; see prior_data_use"}


def validation_verdict(report: dict, contract: StrategyResearchContract) -> dict:
    r, g = report["results"], contract.gates
    checks = {"net_return": r["strategy"]["net_return"] >= g["min_net_return"],
              "excess_return": r["excess_return"] >= g["min_excess_return"],
              "drawdown": r["strategy"]["max_drawdown"] <= g["max_drawdown"],
              "activity": r["strategy"]["traded_bars"] >= g["min_traded_bars"],
              "stress": r["stress"]["net_return"] >= g["min_stress_return"]}
    return {"checks": checks, "passed_declared_checks": all(checks.values()),
            "scope": "single frozen strategy, descriptive holdout checks; no statistical significance claim",
            "paper_started": False, "demo_approved": False}
