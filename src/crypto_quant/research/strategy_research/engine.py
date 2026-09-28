"""Declared strategy definitions mapped to deterministic spot execution."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.strategies.relative_strength import generate_relative_strength_targets, run_relative_strength_backtest
from .contracts import RULES, StrategyResearchContract
from .rule_strategy import FAMILY, INPUTS, compile_definition, generate_rule_targets
from . import basis_strategy
from crypto_quant.research.data_policy import strategy_bounds, strategy_usage


def supported_rules(contract: StrategyResearchContract) -> dict:
    if contract.task["strategy_family"] == basis_strategy.FAMILY:
        return basis_strategy.rules(contract)
    rules = dict(RULES)
    if contract.imputed_hours:
        rules["inputs"] = ("BTCUSDT and ETHUSDT spot 1h OHLC and quote_volume; only contract.imputed_hours are repaired: "
                  "OHLC uses the mean of the previous two complete real bars, excluding shortened or synthetic bars; "
                  "synthetic volume and trade counts are zero; a shortened bar immediately before an authorized gap is retained; "
                  "signals still wait for the nominal hour close; other missing/shortened bars are rejected")
        rules["execution"] += ("; no buys or sells on synthetic hours; holdings carry through; "
                               "synthetic close is an estimated valuation, not a tradable price")
    if contract.task["strategy_family"] == FAMILY:
        rules.update(family=FAMILY,
                     signal="Agent defines named signal expressions, score, entry and exit using causal operators; "
                            "score/entry/exit are numeric expressions; entry > 0 admits a new holding; exit > 0 takes precedence; "
                            "rules are checked only at rebalance times; undefined rule values stop execution",
                     entry="entry > 0 and exit <= 0 at rebalance",
                     exit="exit > 0 or exclusion from the selected ranked portfolio at rebalance",
                     position="spot long/cash; eligible holdings ranked by descending score, ties by ascending symbol; "
                              "keep at most max_positions; equal weight min(gross_exposure/count, max_asset_weight); "
                              "unallocated capital stays in cash; no trading between scheduled rebalances; "
                              "actual buys constrained by available cash including fees")
        rules["inputs"] += "; supported formula inputs=" + ",".join(INPUTS)
    return rules


def execution_plan(parameters: dict, contract: StrategyResearchContract) -> dict:
    """Describe the fixed adapter and the exact parameters passed to its engine."""
    contract.parameters(parameters)
    if contract.task["strategy_family"] == basis_strategy.FAMILY:
        return basis_strategy.execution_plan(parameters, contract)
    rules = supported_rules(contract)
    if contract.task["strategy_family"] == FAMILY:
        compiled = compile_definition(parameters, contract.parameter_space["warmup_hours"])
        from crypto_quant.research.factor_mining.contracts import dumps
        return {"inputs": rules["inputs"],
                "signal": rules["signal"] + "; expanded rules=" + dumps(
                    {name: {**item.description(), "steps": item.calculation_steps()} for name, item in compiled.items()}),
                "execution": rules["execution"] + f"; rebalance every {parameters['rebalance_hours']} hours "
                             "anchored at the one pre-start signal bar; start with no selected holdings; "
                             "a scheduled fill on a synthetic hour waits for the next real hour",
                "position": rules["position"] + "; allocation=" + dumps(parameters["allocation"]),
                "costs": f"costs={dumps(contract.costs)}; stress multiplies both fees and slippage; "
                         "benchmark uses 50/50 hourly targets with the same base costs and no-trade band",
                "parameters": "complete executable strategy definition=" + dumps(parameters)}
    return {
        "inputs": rules["inputs"],
        "signal": f"close[t] / close[t-{parameters['lookback']}h] - 1 for each asset; BTC return minus ETH return; "
                  f"abs(spread) >= {parameters['spread_threshold']} and winner_return > {parameters['min_winner_return']}; "
                  "ties stay in cash",
        "execution": rules["execution"],
        "position": RULES["position"],
        "costs": f"initial_capital={contract.costs['initial_capital']}; fee_bps={contract.costs['fee_bps']}; "
                 f"slippage_bps={contract.costs['slippage_bps']}; min_trade_fraction={contract.costs['min_trade_fraction']}; "
                 f"stress multiplies both fees and slippage by {contract.costs['stress_multiplier']}; "
                 "benchmark uses 50/50 hourly targets with the same base costs and no-trade band",
        "parameters": f"lookback={parameters['lookback']} hours; spread_threshold={parameters['spread_threshold']} "
                      f"return fraction; min_winner_return={parameters['min_winner_return']} return fraction; "
                      "only these three signal parameters are variable",
    }


class DataGap(ValueError):
    """Local data cannot support this explicitly requested experiment."""


def load_segment(db: Path, contract: StrategyResearchContract, stage: str) -> dict:
    start, end = strategy_bounds(contract, stage)
    if contract.task["strategy_family"] == basis_strategy.FAMILY:
        try:
            return basis_strategy.load_segment(db, contract, stage)
        except (ValueError, FileNotFoundError) as exc:
            raise DataGap(str(exc)) from exc
    # Same grid for every parameter choice, including the longest warmup.
    warmup = (contract.parameter_space["warmup_hours"] if contract.task["strategy_family"] == FAMILY
              else max(contract.parameter_space["lookback"]))
    first = start - pd.Timedelta(hours=warmup + 1)
    expected = pd.date_range(first, end, freq="h", inclusive="left").as_unit("ns")
    store = MarketDataStore(db)
    frames = {}
    for symbol in RULES["symbols"]:
        try:
            frame = store.load_bars("spot", symbol, "1h", start=first, end=end - pd.Timedelta(hours=1))
        except (ValueError, FileNotFoundError) as exc:
            raise DataGap(f"{symbol}: {exc}") from exc
        frame.index = frame.index.as_unit("ns")
        authorized = pd.DatetimeIndex([pd.Timestamp(t).tz_convert("UTC") for t in contract.imputed_hours], tz="UTC")
        authorized = authorized[(authorized >= first) & (authorized < end)]
        missing = expected.difference(frame.index)
        if not missing.equals(authorized.sort_values()):
            raise DataGap(f"{symbol}: hourly grid mismatch; missing={len(missing)}, first={list(missing[:3])}; "
                          "missing hours must exactly match explicit imputed_hours in this segment")
        expected_close = frame.index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        complete = frame.close_time == expected_close
        shortened = (~complete) & frame.index.isin(authorized - pd.Timedelta(hours=1)) & \
                    (frame.close_time >= frame.index) & (frame.close_time < expected_close)
        if not (complete | shortened).all():
            raise DataGap(f"{symbol}: close_time differs from hourly availability contract")
        frame["synthetic"] = False
        frame["shortened"] = shortened
        for timestamp in authorized.sort_values():
            prior = frame.loc[(frame.index < timestamp) & ~frame.synthetic & ~frame.shortened].tail(2)
            if len(prior) != 2:
                raise DataGap(f"{symbol}: two complete real bars required before {timestamp}")
            row = prior.iloc[-1].copy()
            for column in ("open", "high", "low", "close"):
                row[column] = prior[column].mean()
            for column in ("volume", "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume"):
                row[column] = 0
            row["close_time"] = timestamp + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
            row["synthetic"], row["shortened"] = True, False
            frame.loc[timestamp] = row
        frame = frame.sort_index()
        if not frame.index.equals(expected):
            missing = expected.difference(frame.index)
            raise DataGap(f"{symbol}: hourly grid mismatch; missing={len(missing)}, first={list(missing[:3])}")
        columns = ["open", "high", "low", "close", "quote_volume"]
        if contract.task["strategy_family"] == FAMILY:
            columns.append("volume")
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
            raise DataGap(f"{symbol}: nonfinite or invalid prices/quote volume")
        if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
                (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
            raise DataGap(f"{symbol}: inconsistent OHLC prices")
        frames[symbol] = frame
    return frames


def save_snapshot(frames: dict, root: Path) -> dict:
    root.mkdir()
    evidence = {}
    for symbol, frame in frames.items():
        path = root / f"{symbol}.csv"
        frame.to_csv(path)
        evidence[symbol] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rows": len(frame), "first": frame.index[0].isoformat(), "last": frame.index[-1].isoformat(),
                            "synthetic_hours": [] if symbol == "funding" else [t.isoformat() for t in frame.index[frame.synthetic]],
                            "shortened_hours": [] if symbol == "funding" else [t.isoformat() for t in frame.index[frame.shortened]]}
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
    if contract.task["strategy_family"] == basis_strategy.FAMILY:
        return basis_strategy.evaluate(frames, parameters, contract, stage, root)
    plan = execution_plan(parameters, contract)
    start = pd.Timestamp(contract.development_start if stage == "development" else contract.validation_start)
    rule_trace = None
    if contract.task["strategy_family"] == FAMILY:
        targets, rule_trace = generate_rule_targets(frames, parameters, contract.parameter_space["warmup_hours"], start)
    else:
        targets, _ = generate_relative_strength_targets(frames, **parameters)
    # One pre-start decision bar, fresh cash account at each segment boundary.
    index = frames["BTCUSDT"].index
    index = index[index >= start - pd.Timedelta(hours=1)]
    market = {s: f.loc[index] for s, f in frames.items()}
    target = {s: t.loc[index] for s, t in targets.items()}
    costs = {k: v for k, v in contract.costs.items() if k != "stress_multiplier"}
    root.mkdir()
    evidence = {}
    if rule_trace is not None:
        path = root / "rule-values.csv"
        rule_trace.loc[index].to_csv(path)
        evidence["rule_values"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    results = {}
    for name in ("strategy", "benchmark", "stress"):
        config = dict(costs)
        weights = target
        if name == "benchmark":
            weights = {s: pd.Series(0.5, index=index) for s in frames}
        if name == "stress":
            for cost in ("fee_bps", "slippage_bps"):
                config[cost] *= contract.costs["stress_multiplier"]
        tradable = ~pd.DataFrame({s: f.synthetic for s, f in market.items()}).any(axis=1)
        if contract.task["strategy_family"] == FAMILY and name != "benchmark":
            scheduled = pd.Series(False, index=index)
            pending = False
            for timestamp in index[1:]:
                if int((timestamp - start) / pd.Timedelta(hours=1)) % parameters["rebalance_hours"] == 0:
                    pending = True
                if pending and tradable.loc[timestamp]:
                    scheduled.loc[timestamp] = True
                    pending = False
            tradable = scheduled
        result = run_relative_strength_backtest(market, weights, config=BacktestConfig(**config), tradable=tradable)
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
            "parameters": parameters, "execution_plan": plan, "results": results, "artifacts": evidence,
            "scope": "adaptive development" if stage == "development" else "strategy internal validation on previously exposed history; not final testing"}


def validation_verdict(report: dict, contract: StrategyResearchContract) -> dict:
    r, g = report["results"], contract.gates
    checks = {"net_return": r["strategy"]["net_return"] >= g["min_net_return"],
              "excess_return": r["excess_return"] >= g["min_excess_return"],
              "drawdown": r["strategy"]["max_drawdown"] <= g["max_drawdown"],
              "activity": r["strategy"]["traded_bars"] >= g["min_traded_bars"],
              "stress": r["stress"]["net_return"] >= g["min_stress_return"]}
    return {"checks": checks, "passed_declared_checks": all(checks.values()),
            "scope": "single frozen strategy, internal validation checks; not final forward testing",
            "data_usage": strategy_usage(contract),
            "paper_started": False, "demo_approved": False}
