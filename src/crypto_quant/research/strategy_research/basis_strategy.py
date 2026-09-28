"""Long spot / equal-quantity short USD-M perpetual with explicit cash accounting."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_quant.backtesting.backtest import BacktestResult
from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.research.factor_mining.contracts import dumps, number, require
from .contracts import fields
from crypto_quant.research.data_policy import strategy_bounds

FAMILY = "spot_perp_basis"
PARAMETERS = ("entry_basis", "exit_basis", "capital_fraction")
DEFINITION_FORMAT = {"entry_basis": "positive basis fraction from contract choices",
                     "exit_basis": "nonnegative basis fraction below entry_basis from contract choices",
                     "capital_fraction": "spot notional / account equity at entry, from contract choices; < 0.5"}


def check_space(space: dict):
    fields(space, ("symbol", "margin_guard_ratio", *PARAMETERS))
    require(space["symbol"] in {"BTCUSDT", "ETHUSDT"}, "basis symbol must be BTCUSDT or ETHUSDT")
    require(0 < number(space["margin_guard_ratio"], "margin_guard_ratio") < 1,
            "margin_guard_ratio must be in (0,1); explicit research guard, not exchange maintenance margin")
    for name in PARAMETERS:
        values = space[name]
        require(isinstance(values, list) and bool(values), "basis parameter choices must be nonempty lists")
        for value in values:
            number(value, name)
            require(value >= 0, f"{name} must be nonnegative")
            if name == "entry_basis":
                require(value > 0, "entry_basis must be positive")
            if name == "capital_fraction":
                require(0 < value < 0.5, "capital_fraction must leave cash for fully collateralized short")
        require(len(set(values)) == len(values), "duplicate basis parameter choices")


def check_parameters(value: dict, space: dict):
    fields(value, PARAMETERS)
    for name in PARAMETERS:
        number(value[name], name)
        require(value[name] in space[name], f"{name} outside frozen parameter space")
    require(value["exit_basis"] < value["entry_basis"], "exit_basis must be below entry_basis")
    return value


def rules(contract):
    return {"family": FAMILY, "symbols": [contract.parameter_space["symbol"]], "interval": "1h",
            "inputs": "same-symbol spot and USD-M perpetual trade OHLC plus mark OHLC on identical complete hourly grids; "
                      "bar t available at t+1h-1ms; no imputation; native funding timestamps, rates, intervals and settlement mark prices required",
            "signal": "basis=perp trade close/spot close-1; enter when basis >= entry_basis; exit when basis <= exit_basis; otherwise keep state",
            "position": "long spot and equal base quantity short linear USDT perpetual; quantities fixed until close; "
                        "spot entry notional=capital_fraction*equity; reserve 100% of perp entry notional as cash collateral; "
                        "short notional is never credited as sale proceeds; no leverage, borrowing or collateral top-ups",
            "execution": "one pre-start signal bar; previous closed-hour signal executes at next hour open on both trade-price series; "
                         "funding at the exact open timestamp settles on the old position before trading; later events settle on the new position; "
                         "end marked at spot close and perp mark close, no forced closing; fees/slippage on both legs; "
                         "margin checked before trading at mark open and conservatively at hourly mark high without crediting positive intrahour funding; "
                         "guard breach stops with partial ledger, not an invented liquidation fill",
            "benchmark": "uninvested cash, zero yield; appropriate neutral baseline, not 50/50 directional spot portfolio"}


def execution_plan(parameters, contract):
    check_parameters(parameters, contract.parameter_space)
    r = rules(contract)
    return {"inputs": r["inputs"], "signal": r["signal"] + "; " + dumps(parameters),
            "execution": r["execution"], "position": r["position"] +
            f"; margin guard ratio={contract.parameter_space['margin_guard_ratio']} of current short mark notional; this is not historical exchange margin tiers",
            "costs": "same fee_bps and slippage_bps charged separately on each leg; stress multiplies both by stress_multiplier; "
                     "funding cashflow for short=quantity*event mark price*rate; funding is never multiplied by fee stress; " + dumps(contract.costs),
            "parameters": "symbol and margin guard fixed by contract; only " + dumps(parameters) + " are model choices"}


def load_segment(db: Path, contract, stage: str):
    start, end = strategy_bounds(contract, stage)
    first = start - pd.Timedelta(hours=1)
    expected = pd.date_range(first, end, freq="h", inclusive="left").as_unit("ns")
    store = MarketDataStore(db)
    symbol = contract.parameter_space["symbol"]
    frames = {}
    for name, market, kind in (("spot", "spot", "trade"), ("perp", "usd_m_perpetual", "trade"),
                                ("mark", "usd_m_perpetual", "mark")):
        frame = store.load_bars(market, symbol, "1h", price_type=kind, start=first, end=end-pd.Timedelta(hours=1))
        frame.index = frame.index.as_unit("ns")
        require(frame.index.equals(expected), f"{name}: incomplete basis hourly grid")
        require((frame.close_time == frame.index + pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)).all(),
                f"{name}: invalid bar availability")
        prices = frame[["open", "high", "low", "close"]]
        require(np.isfinite(prices.to_numpy()).all() and (prices > 0).all().all(), f"{name}: invalid prices")
        require((frame.high >= prices.max(axis=1)).all() and (frame.low <= prices.min(axis=1)).all(),
                f"{name}: inconsistent OHLC")
        frame["synthetic"], frame["shortened"] = False, False
        frames[name] = frame
    funding = store.load_funding(symbol, start=first, end=end-pd.Timedelta(nanoseconds=1), include_previous=True)
    funding.index = funding.index.as_unit("ns")
    require(not funding.empty and funding.index.is_unique and funding.index.is_monotonic_increasing,
            "funding events must be nonempty, unique and sorted")
    values = funding[["funding_rate", "funding_interval_hours", "mark_price"]]
    require(np.isfinite(values.to_numpy()).all() and (funding.mark_price > 0).all() and
            (funding.funding_interval_hours > 0).all(), "funding rates, intervals and settlement mark prices must be present")
    stamps = funding.index.floor("h")
    require(stamps.is_unique and funding.index[0] <= first, "funding coverage must include a preceding event")
    require(np.array_equal(np.diff(stamps.asi8) / 3.6e12, funding.funding_interval_hours.iloc[1:].to_numpy()),
            "missing funding settlement or interval mismatch")
    require(stamps[-1] + pd.Timedelta(hours=float(funding.funding_interval_hours.iloc[-1])) >= end,
            "funding coverage ends before the research segment")
    frames["funding"] = funding
    return frames


def run_account(frames, parameters, contract, start, costs, root):
    spot, perp, mark = (frames[k].loc[start-pd.Timedelta(hours=1):] for k in ("spot", "perp", "mark"))
    basis = perp.close / spot.close - 1
    events = frames["funding"]
    cash, collateral, units, entry_perp = costs["initial_capital"], 0.0, 0.0, 0.0
    fee_rate, slip = costs["fee_bps"] / 10000, costs["slippage_bps"] / 10000
    rows, payments, trades = [], [], []
    entry_equity, trade_fees = None, 0.0
    guard = contract.parameter_space["margin_guard_ratio"]

    def pay(group):
        nonlocal collateral
        total = 0.0
        for timestamp, event in group.iterrows():
            amount = units * event.mark_price * event.funding_rate
            collateral += amount
            total += amount
            payments.append({"timestamp": timestamp, "quantity": units, "rate": event.funding_rate,
                             "mark_price": event.mark_price, "cashflow": amount})
        return total

    def margin_check(timestamp, price, wallet):
        if units and wallet + units * (entry_perp-price) <= guard * units * price:
            pd.DataFrame(rows).to_csv(root / "partial-ledger.csv", index=False)
            pd.DataFrame(payments).to_csv(root / "partial-funding.csv", index=False)
            (root / "margin-breach.json").write_text(dumps({"timestamp": timestamp.isoformat(), "quantity": units,
                "mark_price": float(price), "margin_equity": float(wallet+units*(entry_perp-price)),
                "required_guard": float(guard*units*price)}))
            raise ValueError(f"basis margin guard breached at {timestamp}; partial ledger preserved")

    for i in range(1, len(spot)):
        t = spot.index[i]
        bar_events = events.loc[(events.index >= t) & (events.index < t+pd.Timedelta(hours=1))]
        funding = pay(bar_events.loc[bar_events.index == t])
        margin_check(t, mark.open.iloc[i], collateral)
        opening_equity = cash + units*spot.open.iloc[i] + collateral + units*(entry_perp-mark.open.iloc[i])
        fee, slippage, notional = 0.0, 0.0, 0.0
        signal = basis.iloc[i-1]
        if units == 0 and signal >= parameters["entry_basis"]:
            sfill, pfill = spot.open.iloc[i]*(1+slip), perp.open.iloc[i]*(1-slip)
            units = parameters["capital_fraction"]*opening_equity/sfill
            notional = units*(sfill+pfill)
            fee = notional*fee_rate
            collateral = units*pfill
            require(cash >= units*sfill+collateral+fee, "insufficient cash for spot and 100% perp collateral")
            cash -= units*sfill+collateral+fee
            entry_perp = pfill
            slippage = units*(abs(sfill-spot.open.iloc[i])+abs(pfill-perp.open.iloc[i]))
            entry_equity, trade_fees = opening_equity, fee
        elif units > 0 and signal <= parameters["exit_basis"]:
            sfill, pfill = spot.open.iloc[i]*(1-slip), perp.open.iloc[i]*(1+slip)
            notional = units*(sfill+pfill)
            fee = notional*fee_rate
            slippage = units*(abs(sfill-spot.open.iloc[i])+abs(pfill-perp.open.iloc[i]))
            cash += units*sfill + collateral + units*(entry_perp-pfill) - fee
            trades.append({"exit_time": t, "quantity": units, "net_pnl": cash-entry_equity,
                           "return": cash/entry_equity-1, "fees": trade_fees+fee})
            units, collateral, entry_perp = 0.0, 0.0, 0.0
        wallet_before_events = collateral
        later_funding = pay(bar_events.loc[bar_events.index > t])
        funding += later_funding
        # With hourly OHLC the order of the high and funding events is unknown.
        negative_funding = sum(min(p["cashflow"], 0.0) for p in payments if t < p["timestamp"] < t+pd.Timedelta(hours=1))
        margin_check(t, mark.high.iloc[i], wallet_before_events+negative_funding)
        unrealized = units*(entry_perp-mark.close.iloc[i])
        equity = cash + units*spot.close.iloc[i] + collateral + unrealized
        require(np.isfinite(equity) and equity > 0 and cash >= 0, "basis account cash/equity invariant failed")
        rows.append({"timestamp": t, "cash": cash, "collateral_cash": collateral, "spot_units": units,
                     "perp_units": -units, "spot_value": units*spot.close.iloc[i], "perp_unrealized_pnl": unrealized,
                     "equity": equity, "funding_cashflow": funding, "basis_signal": signal,
                     "trade_notional": notional, "fee": fee, "slippage_cost": slippage,
                     "turnover": notional/opening_equity})
    ledger = pd.DataFrame(rows).set_index("timestamp")
    ledger.to_csv(root / "ledger.csv")
    pd.DataFrame(payments, columns=["timestamp", "quantity", "rate", "mark_price", "cashflow"]).to_csv(root / "funding-payments.csv", index=False)
    closed = pd.DataFrame(trades, columns=["exit_time", "quantity", "net_pnl", "return", "fees"])
    closed.to_csv(root / "closed-trades.csv", index=False)
    return BacktestResult(ledger.equity, pd.Series(costs["initial_capital"], index=ledger.index), ledger,
                          closed, BacktestConfig(**costs)), ledger


def evaluate(frames, parameters, contract, stage, root):
    from .engine import metrics
    plan = execution_plan(parameters, contract)
    start = pd.Timestamp(contract.development_start if stage == "development" else contract.validation_start)
    root.mkdir()
    costs = {k: v for k, v in contract.costs.items() if k != "stress_multiplier"}
    results = {}
    for name in ("strategy", "stress"):
        path = root / name
        path.mkdir()
        scenario = dict(costs)
        if name == "stress":
            for field in ("fee_bps", "slippage_bps"):
                scenario[field] *= contract.costs["stress_multiplier"]
        backtest, ledger = run_account(frames, parameters, contract, start, scenario, path)
        results[name] = {**metrics(backtest), "funding_cashflow": float(ledger.funding_cashflow.sum())}
    benchmark = ledger.copy()
    benchmark[["fee", "slippage_cost", "turnover", "trade_notional"]] = 0.0
    benchmark["equity"] = costs["initial_capital"]
    benchmark[["equity"]].to_csv(root / "cash-benchmark.csv")
    results["benchmark"] = metrics(BacktestResult(benchmark.equity, benchmark.equity, benchmark,
                                                 pd.DataFrame(), BacktestConfig(**costs)))
    results["excess_return"] = results["strategy"]["net_return"]
    artifacts = {str(p.relative_to(root)): {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in root.rglob("*.csv")}
    return {"stage": stage, "start": start.isoformat(), "end_exclusive":
            contract.validation_start if stage == "development" else contract.validation_end,
            "parameters": parameters, "execution_plan": plan, "results": results, "artifacts": artifacts,
            "scope": "hourly paired-account research; explicit conservative margin guard, not exchange liquidation replay"}
