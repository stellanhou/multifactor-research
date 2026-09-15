"""Field contract and point-in-time input panels for factor ideation/calculation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.data_access.market_data import MarketDataStore, resolve_market_symbols
from crypto_quant.features.factors import FactorEngine


@dataclass(frozen=True)
class InputField:
    name: str
    meaning: str
    unit: str
    # Powers of USDT, this asset's base unit, hours, and counts.
    dimensions: tuple[int, int, int, int]
    source: str
    native_frequency: str
    alignment: str
    missing_rule: str


def _fields() -> tuple[InputField, ...]:
    result = []
    for prefix, market in (("spot", "现货"), ("perp", "永续")):
        for column, meaning in (("open", "开盘价"), ("high", "最高价"), ("low", "最低价"), ("close", "收盘价")):
            result.append(InputField(
                f"{prefix}_{column}", f"{market}当小时{meaning}", "USDT / base_asset",
                (1, -1, 0, 0), prefix, "1h", "该小时K线已结束且close_time不晚于信号时点",
                "缺失或非有限值为空，不补前值",
            ))
        for column, meaning, unit, dimensions in (
            ("volume", "成交基础币数量", "base_asset", (0, 1, 0, 0)),
            ("quote_volume", "成交额", "USDT", (1, 0, 0, 0)),
            ("taker_buy_base_volume", "主动买入基础币数量", "base_asset", (0, 1, 0, 0)),
            ("taker_buy_quote_volume", "主动买入成交额", "USDT", (1, 0, 0, 0)),
            ("trades", "成交笔数", "count", (0, 0, 0, 1)),
        ):
            result.append(InputField(
                f"{prefix}_{column}", f"{market}当小时{meaning}", unit, dimensions,
                prefix, "1h", "同一已结束小时的区间合计", "缺失或非法值为空，真实零保留",
            ))
    for name, meaning in (("mark_close", "永续标记价格"), ("index_close", "永续指数价格")):
        result.append(InputField(name, f"当小时结束时的{meaning}", "USDT / base_asset",
                                 (1, -1, 0, 0), name.split("_")[0], "1h",
                                 "该小时K线已结束且close_time不晚于信号时点", "缺失或非有限值为空"))
    result.append(InputField("premium_index", "交易所溢价指数收盘值", "ratio", (0, 0, 0, 0),
                             "premium_index", "1h", "已结束小时的收盘值", "缺失或非有限值为空"))
    for name, meaning, unit, dimensions in (
        ("funding_rate", "最近已知结算事件的资金费率；正值多头付给空头", "rate_per_settlement", (0, 0, 0, 0)),
        ("funding_interval_hours", "这期费率对应的结算间隔", "hour", (0, 0, 1, 0)),
    ):
        result.append(InputField(name, meaning, unit, dimensions, "funding", "原生结算周期",
                                 "截至信号时点最近已知的结算事件，非下一期预测费率", "无已知事件或非法值为空"))
    for period, hours in (("24h", 24), ("7d", 168)):
        result.append(InputField(f"funding_{period}_sum", f"过去{hours}小时实际结算费率之和", "rate_sum",
                                 (0, 0, 0, 0), "funding", "原生结算周期",
                                 f"在(信号时点-{hours}小时, 信号时点]内每个结算事件只计一次",
                                 "历史不足、缺结算或窗口内非法事件为空，附覆盖状态"))
    for name, meaning, unit, dimensions in (
        ("open_interest_base", "未平仓持仓量，已换算成对应基础币数量", "base_asset", (0, 1, 0, 0)),
        ("open_interest_value", "未平仓持仓名义价值", "USDT", (1, 0, 0, 0)),
        ("toptrader_account_long_short_ratio", "保证金余额前20%用户的净多账户数/净空账户数", "ratio", (0, 0, 0, 0)),
        ("toptrader_position_long_short_ratio", "保证金余额前20%用户的净多持仓量/净空持仓量", "ratio", (0, 0, 0, 0)),
        ("global_account_long_short_ratio", "Binance该合约整体净多账户数/净空账户数", "ratio", (0, 0, 0, 0)),
    ):
        result.append(InputField(name, meaning, unit, dimensions, "metrics", "5m",
                                 "小时结束前最近一条有效快照；非小时均值",
                                 "最新记录逐字段保留缺失；年龄达到5分钟后过期，不查找更早非空值"))
    for name, meaning, unit, dimensions in (
        ("long_liquidation_notional_usdt", "观察到的多头清算名义金额", "USDT", (1, 0, 0, 0)),
        ("short_liquidation_notional_usdt", "观察到的空头清算名义金额", "USDT", (1, 0, 0, 0)),
        ("long_liquidation_count", "观察到的多头清算记录数", "count", (0, 0, 0, 1)),
        ("short_liquidation_count", "观察到的空头清算记录数", "count", (0, 0, 0, 1)),
        ("liquidation_event_count", "观察到的全部清算记录数，非人数", "count", (0, 0, 0, 1)),
    ):
        result.append(InputField(name, meaning, unit, dimensions, "liquidations", "逐事件",
                                 "按事件小时汇总，实际接收时间不晚于小时信号时点",
                                 "按需接入；未知或晚到为空；异常金额不抹去有效次数；确认无事件才为零"))
    return tuple(result)


INPUT_FIELDS = _fields()
FIELD_BY_NAME = {field.name: field for field in INPUT_FIELDS}
INPUT_COLUMNS = tuple(FIELD_BY_NAME)
DEFERRED_FIELDS = {
    "taker_long_short_volume_ratio": "已核对：2023样本更匹配前五分钟区间，2025/2026样本更匹配同起点区间，且数量口径未完全复现；全历史缺少统一可用时间定义，继续暂缓",
    "taker_flow_net": "继承上述五分钟主动买卖比的时间与数量口径限制；不能视为整小时资金流",
}


def input_catalog() -> dict[str, Any]:
    return {
        "contract_version": "crypto_factor_inputs_v1",
        "exchange": "Binance", "quote_asset": "USDT", "interval": "1h", "timezone": "UTC",
        "fields": [asdict(field) for field in INPUT_FIELDS],
        "deferred_fields": dict(DEFERRED_FIELDS),
        "unit_rules": "显式倍率下价格除以倍率，基础币数量乘以倍率；USDT金额、费率、比例和次数不缩放。不同资产的基础币单位不能直接比较规模。",
        "availability_assumption": "K线用收盘时点，资金费率/持仓按源时间；后两者无独立历史发布延迟记录。清算另检查实际接收时间。",
        "labels": "未来收益标签仅供评估，禁止进入因子公式",
    }


def validate_universe(universe: pd.Series) -> pd.Series:
    """Require explicit hourly membership; never infer missing membership."""
    if not isinstance(universe, pd.Series) or not isinstance(universe.index, pd.MultiIndex):
        raise ValueError("universe must be a boolean Series indexed by (timestamp, symbol)")
    if universe.index.names != ["timestamp", "symbol"] or universe.empty:
        raise ValueError("universe requires nonempty (timestamp, symbol) index")
    if not pd.api.types.is_bool_dtype(universe.dtype) or universe.isna().any():
        raise ValueError("universe membership must be explicit boolean values")
    times = universe.index.get_level_values("timestamp")
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None or str(times.tz) != "UTC":
        raise ValueError("universe timestamps must be UTC")
    if not times.equals(times.floor("h")):
        raise ValueError("universe timestamps must be exact hourly bar-open labels")
    raw_symbols = universe.index.get_level_values("symbol")
    unique_symbols = raw_symbols.unique()
    if any(not isinstance(s, str) or not s.upper().endswith("USDT") for s in unique_symbols):
        raise ValueError("universe symbols must be USDT pairs")
    names = {s: resolve_market_symbols(s).spot for s in unique_symbols}
    if len(set(names.values())) != len(names):
        raise ValueError("duplicate asset aliases in universe")
    result = universe.copy()
    result.index = pd.MultiIndex.from_arrays([times, raw_symbols.map(names)], names=universe.index.names)
    if not result.index.is_unique:
        raise ValueError("duplicate universe rows")
    result = result.sort_index()
    expected = pd.MultiIndex.from_product([
        pd.date_range(times.min(), times.max(), freq="h"), sorted(names.values()),
    ], names=["timestamp", "symbol"])
    if not result.index.equals(expected):
        raise ValueError("universe must explicitly cover every hourly timestamp and symbol, including false membership")
    if not result.any():
        raise ValueError("universe contains no eligible observations")
    return result.astype(bool)


@dataclass
class FactorInputPanel:
    values: pd.DataFrame
    universe: pd.Series
    diagnostics: dict[str, Any]

    def ideation_context(self) -> dict[str, Any]:
        # Imported here to keep input loading independent of expression execution.
        from crypto_quant.features.factor_expressions import operator_catalog, template_catalog
        return {
            **input_catalog(), "operators": operator_catalog(), "templates": template_catalog(),
            "data": self.diagnostics,
            "universe_rule": "显式传入各历史小时的成员资格；仅据当时成员做截面计算，不在此接口重建选币规则",
        }

    def ideation_message(self) -> dict[str, str]:
        """Ready-to-send context shared by ideation and formula repair callers."""
        return {"role": "user", "content": json.dumps(self.ideation_context(), ensure_ascii=False, allow_nan=False)}


def load_factor_inputs(
    data: MarketDataStore, universe: pd.Series, *, include_liquidations: bool = False,
) -> FactorInputPanel:
    """Load the declared research interval, including caller-declared warm-up rows.

    Membership affects cross-sectional operations and final output. Raw historical
    values remain available to single-asset time-series windows outside membership.
    No labels or non-whitelisted columns enter the returned value panel.
    """
    universe = validate_universe(universe)
    times = universe.index.get_level_values("timestamp").unique()
    symbols = universe.index.get_level_values("symbol").unique()
    engine = FactorEngine(data)
    pieces = []
    diagnostics: dict[str, Any] = {
        "start": times.min().isoformat(), "end": times.max().isoformat(),
        "eligible_rows": int(universe.sum()), "symbols": {},
        "window_history": "时序算子使用本面板覆盖的小时；调用方须在所需研究起点前显式包含窗口预热期",
    }
    for symbol in symbols:
        member = universe.xs(symbol, level="symbol").reindex(times)
        if not member.any():
            # A historical universe can include an asset that first becomes
            # eligible in B. Explicit all-false membership needs no A prices.
            values = pd.DataFrame(np.nan, index=times, columns=INPUT_COLUMNS)
            values["symbol"] = symbol
            pieces.append(values.reset_index().set_index(["timestamp", "symbol"]))
            diagnostics["symbols"][symbol] = {
                "market_symbols": asdict(resolve_market_symbols(symbol)),
                "status": "not_member_in_requested_window", "eligible_rows": 0,
                "data_read": False, "missing_reason": "no eligible output in this entire panel; source data was not requested",
                "coverage": {field: {"rows": len(times), "valid_rows": 0, "coverage_ratio": 0.0,
                             "eligible_rows": 0, "eligible_valid_rows": 0, "eligible_coverage_ratio": None,
                             "missing_reasons": {"not_member_in_requested_window": len(times)}} for field in INPUT_COLUMNS},
            }
            continue
        frame = engine.load(symbol, start=times.min(), end=times.max(), include_liquidations=include_liquidations)
        frame = frame.reindex(times)
        values = pd.DataFrame(index=times)
        coverage = {}
        for field in INPUT_FIELDS:
            if field.source == "liquidations" and not include_liquidations:
                column = pd.Series(np.nan, index=times)
                reasons = pd.Series("not_requested", index=times)
            else:
                column = frame[field.name].astype(float)
                valid = np.isfinite(column)
                if field.unit in {"base_asset", "USDT", "count"}:
                    valid &= column.ge(0)
                if field.unit == "USDT / base_asset" or field.name == "funding_interval_hours":
                    valid &= column.gt(0)
                if field.unit == "ratio" and field.source == "metrics":
                    valid &= column.ge(0)
                column = column.where(valid)
                reasons = pd.Series("missing_or_invalid_source_value", index=times)
                if field.source == "metrics":
                    reasons.loc[frame["metrics_status"].eq("stale")] = "stale"
                elif field.name in {"funding_24h_sum", "funding_7d_sum"}:
                    reasons = frame[field.name.replace("_sum", "_status")].fillna("missing_source")
                elif field.source == "liquidations":
                    reasons = frame["liquidation_status"].fillna("missing_source")
            values[field.name] = column
            missing = column.isna()
            coverage[field.name] = {
                "rows": len(times), "valid_rows": int((~missing).sum()),
                "coverage_ratio": float((~missing).mean()),
                "eligible_rows": int(member.sum()), "eligible_valid_rows": int((~missing & member).sum()),
                "eligible_coverage_ratio": float((~missing & member).sum() / member.sum()) if member.any() else None,
                "missing_reasons": {str(k): int(v) for k, v in reasons[missing].value_counts().items()},
            }
        values["symbol"] = symbol
        pieces.append(values.reset_index().set_index(["timestamp", "symbol"]))
        diagnostics["symbols"][symbol] = {
            "market_symbols": frame.attrs["market_symbols"], "coverage": coverage,
            "metrics_source_quality": frame.attrs["metrics_source_quality"],
            "metrics_source_quality_scope": "包含读取层预热期的原生记录，小时覆盖以上方逐字段统计为准",
            "funding_window_status_counts": frame.attrs["funding_window_status_counts"],
            "observed_time_ranges": {
                column: {
                    "first": frame[column].min().isoformat() if frame[column].notna().any() else None,
                    "last": frame[column].max().isoformat() if frame[column].notna().any() else None,
                }
                for column in ("spot_observed_at", "perp_observed_at", "funding_observed_at", "metrics_observed_at")
            },
            "observation_age": {
                column: {
                    "min": float(frame[column].min()) if frame[column].notna().any() else None,
                    "max": float(frame[column].max()) if frame[column].notna().any() else None,
                }
                for column in ("funding_age_hours", "metrics_age_minutes")
            },
        }
    panel = pd.concat(pieces).sort_index()
    return FactorInputPanel(panel, universe, diagnostics)
