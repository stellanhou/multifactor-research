"""Separate proposal scope from the adapters currently able to execute it."""
import sqlite3
from pathlib import Path


from crypto_quant.features.factor_inputs import INPUT_FIELDS
from crypto_quant.research.factor_mining.contracts import require, text
from .contracts import fields, strings, RULES
from .rule_strategy import FAMILY, INPUTS, compile_definition
from . import basis_strategy
from crypto_quant.research.data_policy import strategy_bounds


PROPOSAL_FORMAT = {
    "family": "desired strategy family, including one not implemented",
    "symbols": ["native Binance symbol, not restricted to BTC/ETH"],
    "required_fields": ["research field name, including unavailable fields"],
    "required_capabilities": ["execution capability ID from catalog, or a new explicit ID"],
    "definition": "executable parameters object if expressible; otherwise null; preserve full rules in calculation_meaning",
}


def sources():
    result = {}
    bar_columns = {"open", "high", "low", "close", "volume", "quote_volume", "trades",
                   "taker_buy_base_volume", "taker_buy_quote_volume"}
    for prefix in ("spot", "perp"):
        for column in bar_columns:
            result[f"{prefix}_{column}"] = ("klines" if prefix == "spot" else "futures_price_bars",
                column, "open_time", "interval='1h'" + (" AND data_type='klines'" if prefix == "perp" else ""))
    for name, category in (("mark_close", "markPriceKlines"), ("index_close", "indexPriceKlines"),
                           ("premium_index", "premiumIndexKlines")):
        result[name] = ("futures_price_bars", "close", "open_time", f"interval='1h' AND data_type='{category}'")
    for name, column in (("open_interest_base", "sum_open_interest"), ("open_interest_value", "sum_open_interest_value"),
                         ("toptrader_account_long_short_ratio", "count_toptrader_long_short_ratio"),
                         ("toptrader_position_long_short_ratio", "sum_toptrader_long_short_ratio"),
                         ("global_account_long_short_ratio", "count_long_short_ratio")):
        result[name] = ("futures_metrics", column, "open_time", "1=1")
    for name in ("funding_rate", "funding_interval_hours", "funding_24h_sum", "funding_7d_sum"):
        result[name] = ("futures_funding_rates", "funding_interval_hours" if name == "funding_interval_hours" else
                        "funding_rate", "funding_time", "1=1")
    return result


def bounds(contract):
    return tuple(int(t.value // 1_000_000) for t in strategy_bounds(contract, "development"))


def data_catalog(db: Path, contract):
    """Only query the development interval; listings are not completeness claims."""
    start, end = bounds(contract)
    catalog = {}
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, _, stamp, condition in set(sources().values()):
            key = f"{table}:{condition}"
            if key in catalog:
                continue
            symbols = [] if table not in tables else [r[0] for r in conn.execute(
                f"SELECT DISTINCT symbol FROM {table} WHERE {condition} AND {stamp}>=? AND {stamp}<? ORDER BY symbol",
                (start, end))]
            catalog[key] = symbols
    return {"scope": "development only; symbol presence does not prove complete or non-null fields",
            "start": contract.development_start, "end_exclusive": contract.validation_start,
            "sources": catalog, "fields": [{"name": f.name, "meaning": f.meaning, "unit": f.unit}
                for f in INPUT_FIELDS if f.source != "liquidations"]}


def execution_catalog(contract):
    return {"scope": "execution capabilities, NOT proposal restrictions; dates/costs remain contractual",
            "configured_family": contract.task["strategy_family"],
            "families": {
                RULES["family"]: {"symbols": RULES["symbols"], "fields": list(INPUTS),
                    "capabilities": ["spot_long_cash", "next_hour_open", "relative_strength_rotation"]},
                FAMILY: {"symbols": RULES["symbols"], "fields": list(INPUTS),
                    "capabilities": ["spot_long_cash", "next_hour_open", "equal_weight_capped", "periodic_rebalance"]},
                basis_strategy.FAMILY: {"symbols": ["BTCUSDT", "ETHUSDT"],
                    "fields": ["spot_close", "perp_close", "mark_close", "funding_rate", "funding_interval_hours"],
                    "capabilities": ["spot_long_perp_short", "next_hour_open", "fixed_pair_quantity", "cash_collateral"]}}}


def assess_proposal(proposal, contract, db):
    fields(proposal, tuple(PROPOSAL_FORMAT))
    family = text(proposal["family"], "proposal family")
    symbols = strings(proposal["symbols"], "proposal symbols")
    required = strings(proposal["required_fields"], "required_fields")
    capabilities = strings(proposal["required_capabilities"], "required_capabilities")
    missing_data, missing_execution, coverage = [], [], []
    start, end = bounds(contract)
    mapping = sources()
    from crypto_quant.data_access.market_data import resolve_market_symbols
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in required:
            if name not in mapping:
                missing_data.append({"field": name, "reason": "no registered local source"})
                continue
            table, column, stamp, condition = mapping[name]
            for symbol in symbols:
                resolved = resolve_market_symbols(symbol)
                native = resolved.spot if table == "klines" else resolved.perpetual
                row = (0, None, None) if table not in tables else conn.execute(
                    f"SELECT COUNT(*),MIN({stamp}),MAX({stamp}) FROM {table} WHERE {condition} "
                    f"AND symbol=? AND {stamp}>=? AND {stamp}<? AND {column} IS NOT NULL AND ABS({column})<1.7976931348623157e308",
                    (native, start, end)).fetchone()
                item = {"field": name, "symbol": native, "valid_source_rows": row[0],
                        "first_ms": row[1], "last_ms": row[2], "interval": [start, end]}
                coverage.append(item)
                if not row[0]:
                    missing_data.append({**item, "reason": "no finite source values in development interval"})
    route = execution_catalog(contract)["families"].get(family)
    if route is None:
        missing_execution.append(f"execution family not implemented: {family}")
        missing_execution.append(f"no adapter for requested execution capabilities: {capabilities}")
    else:
        if family != contract.task["strategy_family"]:
            missing_execution.append(f"route requires its own execution contract: {family}")
        if not set(symbols) <= set(route["symbols"]):
            missing_execution.append(f"symbols not supported by this adapter: {sorted(set(symbols)-set(route['symbols']))}")
        configured_symbols = ([contract.parameter_space["symbol"]] if family == basis_strategy.FAMILY and
                              family == contract.task["strategy_family"] else route["symbols"])
        if family == contract.task["strategy_family"] and set(symbols) != set(configured_symbols):
            missing_execution.append(f"adapter executes fixed symbols {configured_symbols}; cannot silently change proposal universe")
        if not set(required) <= set(route["fields"]):
            missing_execution.append(f"fields not connected to this adapter: {sorted(set(required)-set(route['fields']))}")
        if not set(capabilities) <= set(route["capabilities"]):
            missing_execution.append(f"execution capabilities not implemented: {sorted(set(capabilities)-set(route['capabilities']))}")
    if proposal["definition"] is None:
        missing_execution.append("proposal has no executable definition; original rules preserved")
    elif route and family == contract.task["strategy_family"] and not missing_execution:
        if family == FAMILY:
            compiled = compile_definition(proposal["definition"], contract.parameter_space["warmup_hours"])
            used = set().union(*(set(item.fields) for item in compiled.values()))
            require(used <= set(required), "proposal omits fields used by executable rules")
        contract.parameters(proposal["definition"])
    return {"status": "missing_data" if missing_data else "missing_execution_capability" if missing_execution else "ready_for_data_check",
            "missing_data": missing_data, "missing_execution": missing_execution, "coverage": coverage,
            "coverage_scope": "source presence only; exact warmup, gaps and execution inputs checked by load_segment before backtest"}
