"""Build Chinese HTML reports for the paired 1h/4h turnover study."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "experiments/strategy_research/horizon_turnover_20261004"
LEGACY_24H_ROOT = ROOT / "experiments/strategy_research/continuous_positions_20261004/fixed_models_v1"
STAGES = ("development", "C")
ARMS = ("R0", "TAPER5", "EMA50", "SCORE", "HOLD4")
COSTS = ("base", "stress")
KEYS = ("route", "stage", "arm", "cost")
REQUIRED_METRICS = (
    "net_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "total_fees",
    "total_slippage_cost", "total_funding", "total_turnover", "final_equity",
    "average_gross_exposure", "average_net_exposure", "mean_absolute_net_exposure",
    "average_target_gross", "average_target_holdings", "position_time_fraction", "fills",
    "verification",
)
PAIR_METRICS = (
    "total_turnover", "daily_turnover", "daily_turnover_per_gross_exposure",
    "estimated_fill_notional", "estimated_fill_notional_per_day", "total_fees", "fee_to_initial_capital",
    "total_slippage_cost", "slippage_to_initial_capital", "total_cost_to_initial_capital",
    "net_return", "annualized_volatility", "sharpe_ratio", "max_drawdown",
    "average_gross_exposure", "average_net_exposure", "mean_absolute_net_exposure",
    "average_target_holdings", "fills",
)
DISPLAY_NAMES = {
    "route": "路线", "stage": "区间", "arm": "调仓方式", "cost": "成本情景",
    "account_count": "全部路线数", "window_account_count": "账户数", "mean_daily_turnover": "日均换手",
    "median_daily_turnover": "日均换手中位数", "mean_daily_turnover_delta_pct": "日均换手差异（对R0）",
    "median_daily_turnover_delta_pct": "日均换手差异中位数", "mean_turnover_per_gross": "日均换手/平均敞口",
    "r0_zero_turnover_accounts": "R0零换手账户数",
    "zero_turnover_accounts": "本方式零换手账户数",
    "zero_exposure_accounts": "零敞口账户数", "all_cash_accounts": "全现金账户数",
    "valid_traded_pair_count": "双方有交易配对数", "return_improved_routes": "收益改善路线数（有效交易配对）",
    "sharpe_improved_routes": "夏普改善路线数（有效交易配对）",
    "mean_fill_notional_per_day": "估算日成交名义金额", "mean_fee_to_capital": "手续费/初始资金",
    "mean_slippage_to_capital": "滑点成本/初始资金", "mean_total_cost_to_capital": "手续费+滑点/初始资金",
    "mean_net_return": "净收益率均值", "mean_sharpe_ratio": "净夏普均值",
    "mean_max_drawdown": "最大回撤均值", "mean_gross_exposure": "平均总敞口",
    "mean_net_exposure": "平均净敞口", "mean_target_holdings": "目标持仓数均值",
    "mean_fills": "成交笔数均值", "qualified_routes": "通过开发门槛路线数",
    "mean_delta_sharpe_ratio": "净夏普差异（对R0）", "mean_delta_net_return": "净收益差异（对R0）",
    "base_net_return": "R0开发段base净收益", "base_sharpe": "R0开发段base净夏普",
    "base_max_drawdown": "R0开发段base最大回撤", "stress_net_return": "R0开发段stress净收益",
    "base_volatility": "开发段base年化波动率", "route_label": "路线", "qualified": "资格",
    "qualified_routes": "通过组合数", "evaluated_routes": "已评估组合数", "route_total": "预期组合数",
    "start_date": "起始日期（UTC）", "end_date": "结束日期（UTC）",
    "duration_days": "区间日数", "account_count": "账户数",
    "annualized_volatility": "年化波动率", "net_return": "净收益率", "sharpe_ratio": "净夏普",
    "max_drawdown": "最大回撤", "average_gross_exposure": "平均总敞口",
    "average_net_exposure": "平均净敞口", "average_target_holdings": "目标持仓数",
    "mean_absolute_net_exposure": "平均绝对净敞口", "fills": "成交笔数",
    "daily_turnover": "日均换手", "daily_turnover_per_gross_exposure": "日均换手/平均总敞口",
    "estimated_fill_notional_per_day": "估算日成交名义金额", "fee_to_initial_capital": "手续费/初始资金",
    "estimated_fill_notional": "估算成交名义金额", "total_fees": "手续费总额",
    "total_slippage_cost": "滑点总额", "total_turnover": "累计换手", "verification": "账户核验",
    "valid_traded_pair": "双方有交易", "strategy_daily_turnover": "调仓方式日均换手",
    "r0_daily_turnover": "R0日均换手", "delta_daily_turnover": "日均换手差",
    "delta_daily_turnover_pct": "日均换手相对差（有效交易配对）",
    "strategy_daily_turnover_per_gross_exposure": "调仓方式敞口归一换手",
    "r0_daily_turnover_per_gross_exposure": "R0敞口归一换手",
    "strategy_estimated_fill_notional_per_day": "调仓方式估算日成交金额",
    "r0_estimated_fill_notional_per_day": "R0估算日成交金额",
    "strategy_total_cost_to_initial_capital": "调仓方式成本/初始资金",
    "r0_total_cost_to_initial_capital": "R0成本/初始资金",
    "delta_net_return": "净收益率差", "delta_sharpe_ratio": "净夏普差",
    "strategy_max_drawdown": "调仓方式最大回撤",
    "strategy_average_gross_exposure": "调仓方式平均总敞口",
    "strategy_average_net_exposure": "调仓方式平均净敞口",
    "strategy_average_target_holdings": "调仓方式目标持仓数",
    "strategy_fills": "调仓方式成交笔数",
    "factor_count": "因子数量",
    "schedule_windows": "model_schedule记录数",
    "active_schedule_windows": "active窗口数",
    "active_alpha_values": "active窗口alpha",
    "all_zero_active_windows": "active且系数全零窗口数",
    "all_zero_active_pct": "active窗口全零比例",
    "all_zero_route_label": "路线",
    "gross_exposure": "平均总敞口",
    "net_exposure": "平均净敞口",
    "traded_pair": "R0与本方式均有成交",
    "zero_model_windows": "active全零模型窗口数",
    "zero_model_window_share": "active窗口全零比例",
    "cash_state": "现金/无成交状态",
    "arm_label": "调仓方式",
    "valid_pairs": "有效交易配对数/9",
    "mean_daily_turnover_delta": "平均日均换手差",
    "mean_relative_turnover_delta": "平均相对换手差",
    "return_improved": "净收益改善路线数/9",
    "sharpe_improved": "净夏普改善路线数/9",
}
ARM_LABELS = {
    "R0": "原版 R0", "TAPER5": "五档梯形", "EMA50": "仓位 EMA",
    "SCORE": "分数加权", "HOLD4": "前四保留",
}
ROUTE_LABELS = {
    "E2_full_equal_weight": "E2 全池等权", "E2_full_ridge": "E2 全池 Ridge",
    "E2_full_elastic_net": "E2 全池 Elastic Net",
    "E3_stepwise_equal_weight": "E3 逐步等权", "E3_stepwise_ridge": "E3 逐步 Ridge",
    "E3_grid_equal_weight": "E3 网格等权", "E3_grid_ridge": "E3 网格 Ridge",
    "E3_ga0_equal_weight": "E3 GA 等权", "E3_ga0_ridge": "E3 GA Ridge",
}
STAGE_LABELS = {"development": "开发段", "C": "C段"}
COST_LABELS = {"base": "基准成本", "stress": "2倍成本"}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _ratio(numerator: Any, denominator: Any) -> float:
    numerator, denominator = _number(numerator), _number(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _account_meta(root: Path, row: pd.Series) -> tuple[dict[str, Any], str | None]:
    path = root / "paired" / "accounts" / str(row.route) / str(row.stage) / str(row.arm) / str(row.cost) / "account.json"
    metadata = read_json(path)
    if metadata is None:
        return {}, f"缺少 {path.relative_to(root)}"
    start = pd.to_datetime(metadata.get("start"), utc=True, errors="coerce")
    end = pd.to_datetime(metadata.get("end"), utc=True, errors="coerce")
    duration_days = (end - start).total_seconds() / 86400 if pd.notna(start) and pd.notna(end) and end > start else float("nan")
    missing_fields = [field for field in ("start", "end", "initial_capital", "fee_bps") if metadata.get(field) in (None, "")]
    if pd.isna(start) or pd.isna(end) or end <= start:
        missing_fields.append("valid start/end")
    missing_fields.extend(field for field in ("initial_capital", "fee_bps") if not math.isfinite(_number(metadata.get(field))))
    missing_fields = sorted(set(missing_fields))
    return {
        "duration_days": duration_days,
        "start_utc": start.isoformat() if pd.notna(start) else "",
        "end_utc": end.isoformat() if pd.notna(end) else "",
        "initial_capital": _number(metadata.get("initial_capital")),
        "fee_bps": _number(metadata.get("fee_bps")),
        "account_metadata_present": True,
    }, f"{path.relative_to(root)} 缺少或无法解析字段 {', '.join(missing_fields)}" if missing_fields else None


def load_comparison(root: Path, *, legacy: bool = False) -> tuple[pd.DataFrame, list[str]]:
    paired_root = root if legacy else root / "paired"
    comparison_path = paired_root / "comparison.csv"
    if not comparison_path.is_file():
        return pd.DataFrame(), [f"缺少 {comparison_path.relative_to(root)}"]
    frame = pd.read_csv(comparison_path)
    if frame.empty:
        return frame, []
    missing_columns = [column for column in (*KEYS, *REQUIRED_METRICS) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"comparison is missing required columns {missing_columns}: {comparison_path}")
    frame["route"] = frame["route"].astype(str)
    for column in ("stage", "arm", "cost"):
        frame[column] = frame[column].astype(str)
    frame["verification"] = frame["verification"].astype(str).str.strip().str.lower()
    unverified = ~frame["verification"].eq("passed")
    if unverified.any():
        rows = frame.loc[unverified, [*KEYS, "verification"]].to_dict("records")
        raise ValueError(f"comparison includes accounts without passed verification: {rows[:3]}")
    duplicated = frame.duplicated(list(KEYS), keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, list(KEYS)].to_dict("records")
        raise ValueError(f"duplicate account rows in {comparison_path}: {keys[:3]}")
    return frame, []


def enrich_comparison(root: Path, frame: pd.DataFrame, *, legacy: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        return frame.copy(), []
    result = frame.copy()
    diagnostics: list[str] = []
    durations, starts, ends, capitals, fee_bps, metadata_present = [], [], [], [], [], []
    for _, row in result.iterrows():
        if legacy:
            account_path = root / "accounts" / str(row.route) / str(row.stage) / str(row.arm) / str(row.cost) / "account.json"
            metadata = read_json(account_path)
            warning = None if metadata is not None else f"缺少 {account_path.relative_to(root)}"
            if metadata is None:
                values = {}
            else:
                start = pd.to_datetime(metadata.get("start"), utc=True, errors="coerce")
                end = pd.to_datetime(metadata.get("end"), utc=True, errors="coerce")
                values = {
                    "duration_days": (end - start).total_seconds() / 86400 if pd.notna(start) and pd.notna(end) and end > start else float("nan"),
                    "start_utc": start.isoformat() if pd.notna(start) else "",
                    "end_utc": end.isoformat() if pd.notna(end) else "",
                    "initial_capital": _number(metadata.get("initial_capital")),
                    "fee_bps": _number(metadata.get("fee_bps")),
                    "account_metadata_present": True,
                }
        else:
            values, warning = _account_meta(root, row)
        if warning:
            diagnostics.append(warning)
        durations.append(values.get("duration_days", float("nan")))
        starts.append(values.get("start_utc", ""))
        ends.append(values.get("end_utc", ""))
        capitals.append(values.get("initial_capital", float("nan")))
        fee_bps.append(values.get("fee_bps", float("nan")))
        metadata_present.append(bool(values.get("account_metadata_present", False)))
    result["duration_days"] = durations
    result["start_utc"] = starts
    result["end_utc"] = ends
    result["initial_capital"] = capitals
    result["fee_bps"] = fee_bps
    result["account_metadata_present"] = metadata_present
    result["daily_turnover"] = result.apply(lambda row: _ratio(row.total_turnover, row.duration_days), axis=1)
    result["daily_turnover_per_gross_exposure"] = result.apply(
        lambda row: _ratio(row.daily_turnover, row.average_gross_exposure), axis=1
    )
    result["estimated_fill_notional"] = result.apply(
        lambda row: _ratio(row.total_fees, row.fee_bps / 10000.0), axis=1
    )
    result["estimated_fill_notional_per_day"] = result.apply(
        lambda row: _ratio(row.estimated_fill_notional, row.duration_days), axis=1
    )
    result["fee_to_initial_capital"] = result.apply(lambda row: _ratio(row.total_fees, row.initial_capital), axis=1)
    result["slippage_to_initial_capital"] = result.apply(lambda row: _ratio(row.total_slippage_cost, row.initial_capital), axis=1)
    result["total_cost_to_initial_capital"] = result.apply(
        lambda row: _ratio(row.total_fees + row.total_slippage_cost, row.initial_capital), axis=1
    )
    return result, sorted(set(diagnostics))


def paired_effects(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    columns = ["horizon", *KEYS, "paired_with", "duration_days", "initial_capital", "fee_bps", "account_metadata_present",
               "verification", "r0_verification",
               "valid_traded_pair", "return_improved_on_valid_traded_pair", "sharpe_improved_on_valid_traded_pair"]
    for metric in PAIR_METRICS:
        columns.extend((f"strategy_{metric}", f"r0_{metric}", f"delta_{metric}", f"delta_{metric}_pct"))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    baseline = frame.loc[frame.arm.eq("R0"), ["route", "stage", "cost", *PAIR_METRICS, "verification"]].copy()
    baseline = baseline.rename(columns={**{metric: f"r0_{metric}" for metric in PAIR_METRICS}, "verification": "r0_verification"})
    paired = frame.merge(baseline, on=["route", "stage", "cost"], how="left", validate="many_to_one")
    paired["horizon"] = horizon
    paired["paired_with"] = "R0"
    valid_traded_pair = (
        (pd.to_numeric(paired["total_turnover"], errors="coerce") > 0)
        & (pd.to_numeric(paired["r0_total_turnover"], errors="coerce") > 0)
        & (pd.to_numeric(paired["fills"], errors="coerce") > 0)
        & (pd.to_numeric(paired["r0_fills"], errors="coerce") > 0)
        & paired["verification"].astype(str).str.lower().eq("passed")
        & paired["r0_verification"].astype(str).str.lower().eq("passed")
    )
    for metric in PAIR_METRICS:
        paired[f"strategy_{metric}"] = paired[metric]
        paired[f"delta_{metric}"] = paired[metric] - paired[f"r0_{metric}"]
        paired[f"delta_{metric}_pct"] = paired.apply(
            lambda row: _ratio(row[f"delta_{metric}"], abs(row[f"r0_{metric}"])) * 100, axis=1
        )
    paired.loc[~valid_traded_pair, ["delta_total_turnover_pct", "delta_daily_turnover_pct"]] = np.nan
    paired["valid_traded_pair"] = (
        valid_traded_pair
    )
    paired["return_improved_on_valid_traded_pair"] = paired.valid_traded_pair & (paired.delta_net_return > 0)
    paired["sharpe_improved_on_valid_traded_pair"] = paired.valid_traded_pair & (paired.delta_sharpe_ratio > 0)
    return paired.reindex(columns=columns).sort_values(["stage", "route", "cost", "arm"], kind="stable")


def _expected_routes(model_contract: dict[str, Any]) -> list[str]:
    routes = model_contract.get("routes")
    if not isinstance(routes, list):
        raise ValueError("models/run_contract.json must contain a routes list")
    route_ids = [item.get("route_id") for item in routes if isinstance(item, dict)]
    if len(route_ids) != len(routes) or any(not isinstance(route_id, str) or not route_id for route_id in route_ids):
        raise ValueError("every models/run_contract.json route must have a nonempty route_id")
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("models/run_contract.json route_id values must be unique")
    return route_ids


def expected_keys(routes: list[str]) -> set[tuple[str, str, str, str]]:
    return {(route, stage, arm, cost) for route in routes for stage in STAGES for arm in ARMS for cost in COSTS}


def qualification_table(frame: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    records = []
    development = frame.loc[frame.stage.eq("development")] if not frame.empty else frame
    for route in routes:
        for arm in ARMS:
            base = development.loc[development.route.eq(route) & development.arm.eq(arm) & development.cost.eq("base")]
            stress = development.loc[development.route.eq(route) & development.arm.eq(arm) & development.cost.eq("stress")]
            if base.empty or stress.empty:
                records.append({"route": route, "arm": arm, "base_net_return": np.nan, "base_sharpe": np.nan,
                                "base_volatility": np.nan, "base_max_drawdown": np.nan, "stress_net_return": np.nan,
                                "qualified": "待完成"})
                continue
            b, s = base.iloc[0], stress.iloc[0]
            passed = (
                str(b.verification).lower() == "passed"
                and str(s.verification).lower() == "passed"
                and _number(b.net_return) > 0
                and _number(b.sharpe_ratio) > 0
                and math.isfinite(_number(b.sharpe_ratio))
                and _number(b.annualized_volatility) > 0
                and math.isfinite(_number(b.annualized_volatility))
                and _number(b.fills) > 0
                and _number(s.fills) > 0
                and _number(b.max_drawdown) <= .15
                and _number(s.net_return) >= 0
            )
            records.append({"route": route, "arm": arm, "base_net_return": b.net_return, "base_sharpe": b.sharpe_ratio,
                            "base_volatility": b.annualized_volatility, "base_max_drawdown": b.max_drawdown,
                            "stress_net_return": s.net_return, "qualified": "通过" if passed else "未通过"})
    return pd.DataFrame(records)


def qualification_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        group = details.loc[details.arm.eq(arm)] if not details.empty else details
        rows.append({"arm": arm, "qualified_routes": int(group.qualified.eq("通过").sum()),
                     "evaluated_routes": int(group.qualified.ne("待完成").sum()), "route_total": len(group)})
    return pd.DataFrame(rows)


def interval_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stage", "start_date", "end_date", "duration_days", "window_account_count"]
    if frame.empty or "start_utc" not in frame:
        return pd.DataFrame(columns=columns)
    records = []
    for stage, rows in frame.groupby("stage", sort=False):
        valid = rows.loc[rows.start_utc.ne("") & rows.end_utc.ne("")]
        if valid.empty:
            records.append({"stage": stage, "start_date": "待完成", "end_date": "待完成", "duration_days": np.nan, "window_account_count": 0})
            continue
        starts = pd.to_datetime(valid.start_utc, utc=True)
        ends = pd.to_datetime(valid.end_utc, utc=True)
        start_min, start_max = starts.min(), starts.max()
        end_min, end_max = ends.min(), ends.max()

        def display_range(low: pd.Timestamp, high: pd.Timestamp) -> str:
            low_text, high_text = low.strftime("%Y-%m-%d %H:%M UTC"), high.strftime("%Y-%m-%d %H:%M UTC")
            return low_text if low == high else f"{low_text} — {high_text}"

        durations = (ends - starts).dt.total_seconds() / 86400
        duration_low, duration_high = durations.min(), durations.max()
        duration_text = f"{duration_low:.3f}" if math.isclose(duration_low, duration_high, abs_tol=1e-9) else f"{duration_low:.3f} — {duration_high:.3f}"
        records.append({"stage": stage, "start_date": display_range(start_min, start_max),
                        "end_date": display_range(end_min, end_max),
                        "duration_days": duration_text,
                        "window_account_count": len(valid)})
    return pd.DataFrame(records).reindex(columns=columns)


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stage", "arm", "cost", "account_count", "mean_daily_turnover", "median_daily_turnover",
               "mean_daily_turnover_delta_pct", "median_daily_turnover_delta_pct", "mean_turnover_per_gross",
               "valid_traded_pair_count", "r0_zero_turnover_accounts",
               "zero_turnover_accounts", "zero_exposure_accounts", "all_cash_accounts",
               "mean_fill_notional_per_day", "mean_fee_to_capital", "mean_slippage_to_capital", "mean_total_cost_to_capital",
               "return_improved_routes", "sharpe_improved_routes", "mean_net_return", "mean_sharpe_ratio",
               "mean_max_drawdown", "mean_gross_exposure", "mean_net_exposure", "mean_target_holdings", "mean_fills",
               "mean_delta_sharpe_ratio", "mean_delta_net_return"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    paired = paired_effects(frame, "")
    group_rows = []
    for (stage, arm, cost), group in frame.groupby(["stage", "arm", "cost"], sort=False):
        pair_group = paired.loc[paired.stage.eq(stage) & paired.arm.eq(arm) & paired.cost.eq(cost)]
        valid_group = pair_group.loc[pair_group.valid_traded_pair]
        r0_group = frame.loc[frame.stage.eq(stage) & frame.cost.eq(cost) & frame.arm.eq("R0")]
        group_rows.append({
            "stage": stage, "arm": arm, "cost": cost, "account_count": len(group),
            "mean_daily_turnover": _finite_aggregate(group.daily_turnover, "mean"),
            "median_daily_turnover": _finite_aggregate(group.daily_turnover, "median"),
            "mean_daily_turnover_delta_pct": _finite_aggregate(pair_group.delta_daily_turnover_pct, "mean"),
            "median_daily_turnover_delta_pct": _finite_aggregate(pair_group.delta_daily_turnover_pct, "median"),
            "mean_turnover_per_gross": _finite_aggregate(group.daily_turnover_per_gross_exposure, "mean"),
            "valid_traded_pair_count": int(pair_group.valid_traded_pair.sum()),
            "r0_zero_turnover_accounts": int((_as_numeric_series(r0_group.total_turnover) <= 0).sum()),
            "zero_turnover_accounts": int((_as_numeric_series(group.total_turnover) <= 0).sum()),
            "zero_exposure_accounts": int((_as_numeric_series(group.average_gross_exposure) <= 0).sum()),
            "all_cash_accounts": int(((_as_numeric_series(group.average_gross_exposure) <= 0) & (_as_numeric_series(group.fills) <= 0)).sum()),
            "mean_fill_notional_per_day": _finite_aggregate(group.estimated_fill_notional_per_day, "mean"),
            "mean_fee_to_capital": _finite_aggregate(group.fee_to_initial_capital, "mean"),
            "mean_slippage_to_capital": _finite_aggregate(group.slippage_to_initial_capital, "mean"),
            "mean_total_cost_to_capital": _finite_aggregate(group.total_cost_to_initial_capital, "mean"),
            "return_improved_routes": int(pair_group.return_improved_on_valid_traded_pair.sum()),
            "sharpe_improved_routes": int(pair_group.sharpe_improved_on_valid_traded_pair.sum()),
            "mean_net_return": _finite_aggregate(group.net_return, "mean"),
            "mean_sharpe_ratio": _finite_aggregate(group.sharpe_ratio, "mean"),
            "mean_max_drawdown": _finite_aggregate(group.max_drawdown, "mean"),
            "mean_gross_exposure": _finite_aggregate(group.average_gross_exposure, "mean"),
            "mean_net_exposure": _finite_aggregate(group.average_net_exposure, "mean"),
            "mean_target_holdings": _finite_aggregate(group.average_target_holdings, "mean"),
            "mean_fills": _finite_aggregate(group.fills, "mean"),
            "mean_delta_sharpe_ratio": _finite_aggregate(valid_group.delta_sharpe_ratio, "mean"),
            "mean_delta_net_return": _finite_aggregate(valid_group.delta_net_return, "mean"),
        })
    stage_order = {stage: i for i, stage in enumerate(STAGES)}
    arm_order = {arm: i for i, arm in enumerate(ARMS)}
    cost_order = {cost: i for i, cost in enumerate(COSTS)}
    result = pd.DataFrame(group_rows)
    result["_stage"] = result.stage.map(stage_order)
    result["_arm"] = result.arm.map(arm_order)
    result["_cost"] = result.cost.map(cost_order)
    return result.sort_values(["_stage", "_arm", "_cost"]).drop(columns=["_stage", "_arm", "_cost"]).reindex(columns=columns)


def _as_numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(np.nan)


def _finite_aggregate(values: pd.Series, operation: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return float("nan")
    return float(np.mean(numeric) if operation == "mean" else np.median(numeric))


def model_schedule_diagnostic(root: Path, routes: list[str], frame: pd.DataFrame, effects: pd.DataFrame) -> str:
    schedule_rows = []
    for route in routes:
        for stage in STAGES:
            path = root / "models" / "accounts" / route / stage / "base" / "account.json"
            metadata = read_json(path)
            if metadata is None:
                raise ValueError(f"missing frozen R0 model metadata: {path}")
            schedule = metadata.get("model_schedule")
            if not isinstance(schedule, list):
                raise ValueError(f"model_schedule must be a list: {path}")
            active = [entry for entry in schedule if entry.get("status") == "active"]
            all_zero = 0
            alphas = sorted({float(entry["alpha"]) for entry in active if entry.get("alpha") is not None})
            for entry in active:
                coefficients = entry.get("coefficients")
                if not isinstance(coefficients, dict) or not coefficients:
                    raise ValueError(f"active schedule entry has no coefficient map: {path}")
                if all(math.isfinite(float(value)) and float(value) == 0.0 for value in coefficients.values()):
                    all_zero += 1
            active_count = len(active)
            schedule_rows.append({
                "route": route,
                "route_label": ROUTE_LABELS.get(route, route),
                "stage": stage,
                "schedule_windows": len(schedule),
                "active_schedule_windows": active_count,
                "active_alpha_values": ", ".join(f"{value:g}" for value in alphas) if alphas else "—",
                "all_zero_active_windows": all_zero,
                "all_zero_active_pct": all_zero / active_count if active_count else np.nan,
            })
    schedule_frame = pd.DataFrame(schedule_rows)
    affected = schedule_frame.loc[schedule_frame.all_zero_active_windows.gt(0)]
    window_counts_html = _render_table(schedule_frame.assign(stage=schedule_frame.stage.map(STAGE_LABELS)),
        ["route_label", "stage", "schedule_windows", "active_schedule_windows", "active_alpha_values", "all_zero_active_windows", "all_zero_active_pct"], {
        "schedule_windows": "int", "active_schedule_windows": "int", "all_zero_active_windows": "int", "all_zero_active_pct": "pct"})
    if affected.empty:
        return f'''<section><h2>活跃模型系数诊断</h2><p>已检查 {len(schedule_frame)} 个 route × stage 的 R0 模型 schedule；未发现 active 窗口中全系数为零的情况。</p><details><summary>各 route × stage schedule 窗口计数</summary>{window_counts_html}</details></section>'''

    effect_by_key = {(row.route, row.stage, row.arm, row.cost): row for row in effects.itertuples(index=False)}
    outcome_rows = []
    for diagnostic in affected.itertuples(index=False):
        for arm in ARMS:
            result = frame.loc[frame.route.eq(diagnostic.route) & frame.stage.eq(diagnostic.stage)
                               & frame.arm.eq(arm) & frame.cost.eq("base")]
            paired = effect_by_key.get((diagnostic.route, diagnostic.stage, arm, "base"))
            row = result.iloc[0] if not result.empty else None
            no_trade = bool(row is not None and _number(row.total_turnover) <= 0 and _number(row.fills) <= 0)
            outcome_rows.append({
                "all_zero_route_label": ROUTE_LABELS.get(diagnostic.route, diagnostic.route),
                "stage": STAGE_LABELS[diagnostic.stage],
                "arm": ARM_LABELS[arm],
                "zero_model_windows": f"{diagnostic.all_zero_active_windows}/{diagnostic.active_schedule_windows}",
                "gross_exposure": row.average_gross_exposure if row is not None else np.nan,
                "net_exposure": row.average_net_exposure if row is not None else np.nan,
                "total_turnover": row.total_turnover if row is not None else np.nan,
                "daily_turnover": row.daily_turnover if row is not None else np.nan,
                "delta_daily_turnover_pct": paired.delta_daily_turnover_pct if paired is not None else np.nan,
                "net_return": row.net_return if row is not None else np.nan,
                "sharpe_ratio": row.sharpe_ratio if row is not None else np.nan,
                "fills": row.fills if row is not None else np.nan,
                "cash_state": "全现金/无成交" if no_trade else ("未完成" if row is None else "有成交"),
                "traded_pair": bool(paired.valid_traded_pair) if paired is not None else False,
            })
    outcome_frame = pd.DataFrame(outcome_rows)
    affected_summary = affected.copy()
    affected_summary["route_label"] = affected_summary.apply(
        lambda row: f"{ROUTE_LABELS.get(row.route, row.route)}（{STAGE_LABELS[row.stage]}）", axis=1
    )
    affected_summary["stage"] = affected_summary.stage.map(STAGE_LABELS)
    affected_html = _render_table(affected_summary, ["route_label", "schedule_windows", "active_schedule_windows",
        "active_alpha_values", "all_zero_active_windows", "all_zero_active_pct"], {
        "schedule_windows": "int", "active_schedule_windows": "int", "all_zero_active_windows": "int", "all_zero_active_pct": "pct"})
    outcomes_html = _render_table(outcome_frame, ["all_zero_route_label", "stage", "arm", "zero_model_windows",
        "gross_exposure", "net_exposure", "total_turnover", "daily_turnover", "delta_daily_turnover_pct",
        "net_return", "sharpe_ratio", "fills", "cash_state", "traded_pair"], {
        "gross_exposure": "pct2", "net_exposure": "pct2", "total_turnover": "num", "daily_turnover": "turnover",
        "delta_daily_turnover_pct": "pct100", "net_return": "pct", "sharpe_ratio": "num", "fills": "int"})
    notes = ("统计单位是 model_schedule 中的一条拟合窗口；active全零表示该窗口 status=active 且该模型全部系数精确为0，未按窗口长度加权。"
             " 同一行按 route × stage 对齐各调仓方式的开发/C段 base 成本账户，方便将模型窗口占比与敞口、换手、收益并看。"
             " SCORE 对横截面分数去均值，分数全平时按规则持现金；HOLD4 在并列排名下按 symbol 顺序留仓再补席位，R0 使用原版long-first与多空去重排名。"
             " 因此 SCORE 全现金或 HOLD4 总/净敞口改变应按仓位/方向变化解释，不能单独认作纯稳定性改善。")
    return f'''<section><h2>活跃模型系数诊断</h2><p class="warning">发现 {len(affected)} 个 route × stage 含 active 且全系数为零的拟合窗口。{html.escape(notes)}</p>
<h3>出现该情况的路线/区间</h3>{affected_html}<h3>对应调仓方式结果（base成本）</h3><div class="data-wrap">{outcomes_html}</div>
<details><summary>全部 route × stage 的 schedule 统计</summary>{window_counts_html}</details></section>'''


def _fmt(value: Any, kind: str = "num") -> str:
    number = _number(value)
    if not math.isfinite(number):
        return "—"
    if kind == "pct":
        return f"{number:.1%}"
    if kind == "pct2":
        return f"{number:.2%}"
    if kind == "pct100":
        return f"{number:.1f}%"
    if kind == "money":
        return f"{number:,.0f} USDT"
    if kind == "int":
        return f"{number:,.0f}"
    if kind == "turnover":
        return f"{number:.3f}×"
    return f"{number:.3f}"


def _render_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None,
                  display_maps: dict[str, dict[Any, str]] | None = None) -> str:
    if frame.empty:
        return '<p class="empty">当前没有可展示的账户结果。</p>'
    visible = frame.reindex(columns=[column for column in columns if column in frame.columns]).copy()
    for column, kind in (formats or {}).items():
        if column in visible.columns:
            visible[column] = visible[column].map(lambda value: _fmt(value, kind))
    for column, mapping in (display_maps or {}).items():
        if column in visible.columns:
            visible[column] = visible[column].map(lambda value: mapping.get(value, value))
    rename = {column: DISPLAY_NAMES.get(column, column) for column in visible.columns}
    return visible.rename(columns=rename).to_html(index=False, escape=True, border=0, classes="data")


def _figure_data(summary: pd.DataFrame, value: str, title: str, ylabel: str, *, percent: bool = False) -> str | None:
    if summary.empty or value not in summary.columns:
        return None
    arms = [arm for arm in ARMS if arm != "R0"]
    groups = [(stage, cost) for stage in STAGES for cost in COSTS]
    if not any(summary["arm"].eq(arm).any() for arm in arms):
        return None
    x = np.arange(len(arms))
    width = .18
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=150)
    colors = ("#3569a8", "#62a0d8", "#d47c2c", "#e8b65c")
    for index, (stage, cost) in enumerate(groups):
        subset = summary.loc[summary.stage.eq(stage) & summary.cost.eq(cost)].set_index("arm")
        values = [(_number(subset.loc[arm, value]) if arm in subset.index else float("nan")) for arm in arms]
        if percent:
            values = [v * 100 for v in values]
        offset = (index - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=f"{stage}/{cost}", color=colors[index])
    ax.axhline(0, color="#64748b", linewidth=.8)
    ax.set_xticks(x, arms)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _legacy_section() -> str:
    if not LEGACY_24H_ROOT.is_dir():
        return '<section><h2>旧版 24 小时结果对照</h2><p class="empty">未找到 fixed_models_v1 结果。</p></section>'
    contract = read_json(LEGACY_24H_ROOT / "contract.json") or {}
    frame, errors = load_comparison(LEGACY_24H_ROOT, legacy=True)
    frame, more_errors = enrich_comparison(LEGACY_24H_ROOT, frame, legacy=True)
    summary = summary_table(frame)
    table = _render_table(summary, ["stage", "arm", "cost", "account_count", "mean_daily_turnover",
        "mean_daily_turnover_delta_pct", "mean_net_return", "mean_sharpe_ratio", "mean_max_drawdown"], {
        "mean_daily_turnover": "turnover", "mean_daily_turnover_delta_pct": "pct100", "mean_net_return": "pct",
        "mean_max_drawdown": "pct", "mean_sharpe_ratio": "num", "account_count": "int"},
        {"stage": STAGE_LABELS, "arm": ARM_LABELS, "cost": COST_LABELS})
    notes = [f"旧结果路线数 {len(contract.get('routes', [])) or frame.route.nunique()}，仅作为独立参照。",
             "该版本使用旧模型、原生 24 小时决策节奏和 EMA α=0.5，日期与本轮 1h/4h 结果不同，不直接合并排名。"]
    if errors or more_errors:
        notes.append(f"旧结果数据不完整：缺少 {len(set(errors + more_errors))} 项文件。")
    return f'<section><h2>旧版 24 小时结果对照</h2><p>{html.escape(" ".join(notes))}</p>{table}</section>'


REPORT_STYLE = '''body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;background:#f4f7fb;margin:0;line-height:1.55}
main{max-width:1450px;margin:0 auto;padding:32px 26px 60px}h1{margin:0 0 8px;font-size:30px}h2{margin:28px 0 10px;font-size:21px}p{margin:8px 0 14px}.subtle,.empty{color:#64748b}.cards{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}.card{background:white;border:1px solid #dbe3ef;border-radius:10px;padding:12px 16px;min-width:140px}.card b{display:block;font-size:19px;color:#1c4b7a}.card span{font-size:12px;color:#64748b}.panel{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:18px;margin:16px 0}.data-wrap{overflow-x:auto}table.data{border-collapse:collapse;width:100%;font-size:12px;background:white}table.data th{background:#eaf0f8;color:#30435e;text-align:left;position:sticky;top:0}table.data td,table.data th{padding:8px 9px;border-bottom:1px solid #e4eaf2;white-space:nowrap}table.data tr:hover td{background:#f8fbff}.warning{background:#fff7e6;border-left:4px solid #df9c2b;padding:10px 12px}.good{background:#eef8f1;border-left:4px solid #40a56a;padding:10px 12px}figure{display:inline-block;vertical-align:top;width:min(100%,600px);margin:8px 14px 8px 0}figure img{width:100%;height:auto;border:1px solid #e4eaf2;border-radius:8px}figcaption{font-size:12px;color:#64748b}details{margin:12px 0;color:#59677a}footer{margin-top:30px;color:#64748b;font-size:12px}'''


def horizon_label(horizon: str) -> str:
    return {"h1": "1小时", "h4": "4小时"}.get(horizon, horizon)


def status_label(value: Any) -> str:
    return {
        "preparing": "准备中", "calibrating": "校准中", "refitting": "模型重新拟合中",
        "paired_accounts": "账户配对计算中", "running": "进行中", "complete": "已完成",
        "passed": "通过", "failed": "未通过",
        "failed_at_account_guard": "账户检查未通过",
    }.get(str(value), "待生成" if value is None else "状态待确认")


def primary_summary(frame: pd.DataFrame, effects: pd.DataFrame, routes: list[str]) -> tuple[str, str]:
    selected_stage = None
    for stage in ("C", "development"):
        expected_keys_for_stage = {(route, stage, arm, "base") for route in routes for arm in ARMS}
        rows = frame.loc[frame.stage.eq(stage) & frame.cost.eq("base")]
        observed = set(zip(rows.route, rows.stage, rows.arm, rows.cost))
        if len(rows) == len(expected_keys_for_stage) and observed == expected_keys_for_stage:
            selected_stage = stage
            break
    if selected_stage is None:
        return '<p class="warning">开发段或 C 段的 base 账户尚未形成完整45行；完整阶段齐备后会显示四种调仓方式的九路线汇总。</p>', ""

    records = []
    for arm in ARMS:
        if arm == "R0":
            continue
        group = effects.loc[effects.stage.eq(selected_stage) & effects.cost.eq("base") & effects.arm.eq(arm)]
        valid = group.loc[group.valid_traded_pair & pd.to_numeric(group.delta_daily_turnover, errors="coerce").notna()]
        count = len(valid)
        records.append({
            "arm_label": ARM_LABELS[arm],
            "valid_pairs": f"{count}/{len(routes)}",
            "mean_daily_turnover_delta": _fmt(_finite_aggregate(valid.delta_daily_turnover, "mean"), "turnover"),
            "mean_relative_turnover_delta": _fmt(_finite_aggregate(valid.delta_daily_turnover_pct, "mean"), "pct100"),
            "return_improved": f"{int(valid.return_improved_on_valid_traded_pair.sum())}/{len(routes)}",
            "sharpe_improved": f"{int(valid.sharpe_improved_on_valid_traded_pair.sum())}/{len(routes)}",
        })
    summary = pd.DataFrame(records)
    table = _render_table(summary, ["arm_label", "valid_pairs", "mean_daily_turnover_delta",
        "mean_relative_turnover_delta", "return_improved", "sharpe_improved"])
    stage_name = STAGE_LABELS[selected_stage]
    reason = f"汇总区间：{stage_name} base，覆盖全部 {len(routes)} 条路线 × 5 种调仓方式（45/45账户）。"
    if selected_stage == "C":
        reason += "C 仅作描述统计，不参与路线资格或选择。"
    else:
        reason += "C 段 base 尚未完整，因此暂用完整开发段；C 不参与路线资格或选择。"
    return table, reason


def _page(horizon: str, body: str) -> str:
    title = horizon_label(horizon)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} 多因子调仓换手率报告</title><style>{REPORT_STYLE}</style></head><body><main>{body}</main></body></html>'''


def _waiting_report(root: Path, horizon: str, root_state: dict[str, Any], model_contract: dict[str, Any] | None) -> tuple[Path, Path]:
    root_status = html.escape(status_label(root_state.get("status")))
    route_count = len(_expected_routes(model_contract)) if model_contract is not None else "未生成"
    factor_count = model_contract.get("factor_count", "未记录") if model_contract is not None else "未生成"
    expected_pairs = root_state.get("expected_paired_accounts", "未记录")
    body = f'''<h1>{html.escape(horizon_label(horizon))} 多因子调仓方式与换手率</h1>
<div class="cards"><div class="card"><b>{root_status}</b><span>模型进度</span></div><div class="card"><b>{route_count}</b><span>模型路线</span></div><div class="card"><b>{html.escape(str(factor_count))}</b><span>候选因子数</span></div><div class="card"><b>{html.escape(str(expected_pairs))}</b><span>预计账户数</span></div></div>
<section class="panel"><h2>配对结果尚未开始</h2><p>模型拟合仍在进行，账户配对与调仓方式比较尚未启动。完成后重新运行报告，即可查看九条路线的配对结果。</p></section>'''
    output_html = root / "report.html"
    output_html.write_text(_page(horizon, body), encoding="utf-8")
    return output_html, root / "paired" / "turnover_effects.csv"


def _cost_description(model_contract: dict[str, Any]) -> str:
    costs = model_contract["costs"]
    fee_bps = float(costs["fee_bps"])
    slippage_bps = float(costs["slippage_bps"])
    stress = float(costs["stress_multiplier"])
    return (f"base 手续费 {fee_bps:g} bp + 滑点 {slippage_bps:g} bp；stress 为 "
            f"{fee_bps * stress:g} bp + {slippage_bps * stress:g} bp。资金费按账户期间实际 funding 现金流计入收益。")


def build_report(horizon_root: Path) -> tuple[Path, Path]:
    root = horizon_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    horizon = root.name
    root_state = read_json(root / "state.json") or {}
    model_contract = read_json(root / "models" / "run_contract.json")
    paired_contract = read_json(root / "paired" / "contract.json")
    if model_contract is None or paired_contract is None:
        return _waiting_report(root, horizon, root_state, model_contract)

    paired_state = read_json(root / "paired" / "state.json")
    verification = read_json(root / "paired" / "verification.json")
    raw, diagnostics = load_comparison(root)
    frame, account_diagnostics = enrich_comparison(root, raw)
    diagnostics = sorted(set(diagnostics + account_diagnostics))
    routes = _expected_routes(model_contract)
    wanted = expected_keys(routes)
    contract_expected = paired_contract.get("expected_accounts")
    if contract_expected is not None and int(contract_expected) != len(wanted):
        raise ValueError(f"paired contract expects {contract_expected} accounts but the frozen route/stage/arm/cost matrix has {len(wanted)}")
    actual = set(tuple(str(row[key]) for key in KEYS) for _, row in raw.iterrows()) if not raw.empty else set()
    unexpected = sorted(actual - wanted)
    if unexpected:
        raise ValueError(f"comparison.csv contains rows outside the frozen account matrix: {unexpected[:3]}")
    missing = sorted(wanted - actual)
    effects = paired_effects(frame, horizon)
    summary = summary_table(frame)
    primary_table_html, primary_scope = primary_summary(frame, effects, routes)
    model_diagnostic_html = model_schedule_diagnostic(root, routes, frame, effects)
    csv_path = root / "paired" / "turnover_effects.csv"
    effects.to_csv(csv_path, index=False, float_format="%.10g")

    qualification = qualification_table(frame, routes)
    qual_summary = qualification_summary(qualification)
    qualification_counts = _render_table(qual_summary, ["arm", "qualified_routes", "evaluated_routes", "route_total"], {
        "qualified_routes": "int", "evaluated_routes": "int", "route_total": "int"}, {"arm": ARM_LABELS})
    qualification["route_label"] = qualification.route.map(ROUTE_LABELS).fillna(qualification.route)
    qualification_details = _render_table(qualification, ["route_label", "arm", "base_net_return", "base_sharpe",
        "base_volatility", "base_max_drawdown", "stress_net_return", "qualified"], {
        "base_net_return": "pct", "base_sharpe": "num", "base_volatility": "pct",
        "base_max_drawdown": "pct", "stress_net_return": "pct"}, {"arm": ARM_LABELS})
    windows_html = _render_table(interval_table(frame), ["stage", "start_date", "end_date", "duration_days", "window_account_count"], {
        "window_account_count": "int"}, {"stage": STAGE_LABELS})

    common_formats = {"account_count": "int", "mean_daily_turnover": "turnover", "median_daily_turnover": "turnover",
        "mean_daily_turnover_delta_pct": "pct100", "mean_turnover_per_gross": "turnover",
        "valid_traded_pair_count": "int", "r0_zero_turnover_accounts": "int",
        "zero_turnover_accounts": "int", "zero_exposure_accounts": "int", "all_cash_accounts": "int",
        "mean_fill_notional_per_day": "money", "mean_fee_to_capital": "pct", "mean_slippage_to_capital": "pct",
        "mean_total_cost_to_capital": "pct", "return_improved_routes": "int", "sharpe_improved_routes": "int",
        "mean_net_return": "pct", "mean_sharpe_ratio": "num", "mean_max_drawdown": "pct",
        "mean_gross_exposure": "pct", "mean_net_exposure": "pct", "mean_target_holdings": "num", "mean_fills": "int",
        "mean_delta_sharpe_ratio": "num", "mean_delta_net_return": "pct"}
    stage_arm_maps = {"stage": STAGE_LABELS, "arm": ARM_LABELS, "cost": COST_LABELS}
    turnover_table = _render_table(summary, ["stage", "arm", "cost", "account_count", "mean_daily_turnover",
        "median_daily_turnover", "mean_daily_turnover_delta_pct",
        "valid_traded_pair_count", "r0_zero_turnover_accounts", "zero_turnover_accounts", "zero_exposure_accounts",
        "all_cash_accounts", "mean_turnover_per_gross"], common_formats, stage_arm_maps)
    cost_table = _render_table(summary, ["stage", "arm", "cost", "mean_fill_notional_per_day", "mean_fee_to_capital",
        "mean_slippage_to_capital", "mean_total_cost_to_capital"], common_formats, stage_arm_maps)
    performance_table = _render_table(summary, ["stage", "arm", "cost", "account_count", "valid_traded_pair_count",
        "return_improved_routes", "sharpe_improved_routes", "mean_net_return", "mean_sharpe_ratio",
        "mean_max_drawdown", "mean_gross_exposure", "mean_net_exposure", "mean_target_holdings", "mean_fills",
        "mean_delta_net_return", "mean_delta_sharpe_ratio"], common_formats, stage_arm_maps)

    turnover_chart = _figure_data(summary, "mean_daily_turnover_delta_pct", "Paired daily turnover change vs R0", "Mean paired change (%)")
    sharpe_chart = _figure_data(summary, "mean_delta_sharpe_ratio", "Paired net Sharpe change vs R0", "Mean paired Sharpe change")
    chart_html = "".join(
        f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for data, title, caption in (
            (turnover_chart, "Paired daily turnover change vs R0", "只汇总双方都有成交的路线配对；按开发段/C、base/stress 分开。"),
            (sharpe_chart, "Paired net Sharpe change vs R0", "只统计双方都有成交的路线配对；C 段仅作描述。"),
        ) if data
    )

    detail = effects.copy()
    detail["route_label"] = detail.route.map(ROUTE_LABELS).fillna(detail.route)
    detail["stage"] = detail.stage.map(STAGE_LABELS).fillna(detail.stage)
    detail["arm"] = detail.arm.map(ARM_LABELS).fillna(detail.arm)
    detail["cost"] = detail.cost.map(COST_LABELS).fillna(detail.cost)
    detail_cols = ["route_label", "stage", "arm", "cost", "valid_traded_pair",
        "strategy_daily_turnover", "r0_daily_turnover", "delta_daily_turnover", "delta_daily_turnover_pct",
        "strategy_daily_turnover_per_gross_exposure", "r0_daily_turnover_per_gross_exposure",
        "strategy_estimated_fill_notional_per_day", "r0_estimated_fill_notional_per_day",
        "strategy_total_cost_to_initial_capital", "r0_total_cost_to_initial_capital",
        "delta_net_return", "delta_sharpe_ratio", "strategy_max_drawdown", "strategy_average_gross_exposure",
        "strategy_average_net_exposure", "strategy_average_target_holdings", "strategy_fills"]
    detail_formats = {column: "turnover" for column in ("strategy_daily_turnover", "r0_daily_turnover",
        "delta_daily_turnover", "strategy_daily_turnover_per_gross_exposure", "r0_daily_turnover_per_gross_exposure")}
    detail_formats.update({column: "money" for column in ("strategy_estimated_fill_notional_per_day", "r0_estimated_fill_notional_per_day")})
    detail_formats.update({column: "pct" for column in ("strategy_total_cost_to_initial_capital", "r0_total_cost_to_initial_capital",
        "delta_net_return", "strategy_max_drawdown", "strategy_average_gross_exposure", "strategy_average_net_exposure")})
    detail_formats.update({"delta_daily_turnover_pct": "pct100", "delta_sharpe_ratio": "num",
        "strategy_average_target_holdings": "num", "strategy_fills": "int"})
    details_html = _render_table(detail, detail_cols, detail_formats)

    paired_status = paired_state.get("status", "状态缺失") if paired_state else "state.json缺失"
    if paired_state:
        progress = f"{paired_state.get('completed_accounts', '—')} / {paired_state.get('expected_accounts', '—')}"
    else:
        progress = "未记录"
    root_status = html.escape(status_label(root_state.get("status")))
    verification_status = html.escape(status_label(verification.get("status"))) if verification else "待最终核验"
    paired_status_html = html.escape(status_label(paired_status))
    factor_count = html.escape(str(model_contract.get("factor_count", "未记录")))
    missing_preview = "；".join(" / ".join(item) for item in missing[:12])
    if len(missing) > 12:
        missing_preview += f"；另有 {len(missing) - 12} 个缺失账户"
    missing_block = (f'<p class="warning">尚有 {len(missing)} / {len(wanted)} 个账户结果未完成：{html.escape(missing_preview)}</p>'
                     if missing else '<p class="good">全部预期账户结果均已完成。</p>')
    diagnostics_block = ""
    if diagnostics:
        diagnostic_lines = "".join(f"<li>{html.escape(item)}</li>" for item in diagnostics[:30])
        diagnostics_block = f'<details><summary>缺少账户元数据（{len(diagnostics)} 项）</summary><ul>{diagnostic_lines}</ul></details>'
    qualified_count = int(qualification.qualified.eq("通过").sum()) if not qualification.empty else 0
    evaluated_count = int(qualification.qualified.ne("待完成").sum()) if not qualification.empty else 0
    expected_qualification_count = len(routes) * len(ARMS)
    current_scope = "使用既有历史开发段与内部验证段（C）；C 只作描述展示，不参与资格判定或路线选择。"
    costs_note = html.escape(_cost_description(model_contract))
    methods_note = "R0＝原版仓位；EMA50＝仓位 EMA；TAPER5＝五档梯形；SCORE＝分数加权；HOLD4＝前四保留。全池等权、全池 Ridge、全池 Elastic Net、逐步等权/Ridge、网格等权/Ridge 和 GA 等权/Ridge 使用中文短名展示，配对差异表保留原路线编号。模型内部路线选择按冻结选择规则的 R0 成本账户执行；最终开发段资格再对每条路线×调仓方式单独检验。1小时与4小时的因子池、预测目标和原生调仓间隔不同，跨周期结果只作描述比较；执行方式影响只在同周期、同一模型内配对。EMA α=0.5 每次调仓生效，因此两个周期的实际平滑时间不同。"
    definition_note = "日均换手 = total_turnover / account.json 的起止日数；total_turnover 沿用 engine ledger.turnover，买卖双边成交名义金额都计入，不除以 2。归一换手 = 日均换手 / 平均总敞口。成交名义金额 = total_fees ÷ (fee_bps / 10000)。成本同时列手续费、滑点与二者合计；资金费按账户期间实际资金费现金流计入收益。相对换手变化与收益/夏普改善计数只使用双方都有成交的有效路线配对，零交易路线不用于判断调仓贡献；R0 或调仓方式换手为零时相对换手变化留空，零敞口时归一换手留空。"
    output_html = root / "report.html"
    report_title = horizon_label(horizon)
    body = f'''<h1>{html.escape(report_title)} 多因子调仓方式与换手率</h1>
<p class="subtle">对九条冻结路线做同周期配对，展示仓位 EMA、五档梯形、分数加权和前四保留相对原版 R0 的影响。</p>
<div class="cards"><div class="card"><b>{root_status}</b><span>模型进度</span></div><div class="card"><b>{paired_status_html}</b><span>账户比较进度</span></div><div class="card"><b>{html.escape(progress)}</b><span>已完成账户 / 预期账户</span></div><div class="card"><b>{len(actual)} / {len(wanted)}</b><span>已有账户结果 / 总账户数</span></div><div class="card"><b>{factor_count}</b><span>候选因子数</span></div><div class="card"><b>{verification_status}</b><span>账户核验</span></div><div class="card"><b>{qualified_count} / {expected_qualification_count}</b><span>开发段门槛通过（已评估 {evaluated_count}）</span></div></div>
<section class="panel"><h2>主要结果</h2><p>{html.escape(primary_scope)}</p><p>以下为全部九条路线汇总；均值和改善计数只使用 R0 与该方式双方通过核验且有成交的有效配对。改善路线数以九条路线为总数，零交易路线不算作改善。</p><div class="data-wrap">{primary_table_html}</div></section>
<details class="panel"><summary><strong>方法、样本日期与统计口径</strong></summary><p>{html.escape(current_scope)}</p><p>{html.escape(methods_note)}</p><p>{costs_note}</p><p>{html.escape(definition_note)}</p><div class="data-wrap">{windows_html}</div>{missing_block}</details>
<section><h2>开发段资格</h2><p>按既有门槛逐条路线×调仓方式检验。{qualified_count} / {expected_qualification_count} 个组合通过，已评估 {evaluated_count} 个；C 段不参与资格判定或路线选择。</p>{qualification_counts}<details><summary>逐路线资格明细</summary>{qualification_details}</details></section>
{model_diagnostic_html}
<section><h2>换手影响</h2><div class="data-wrap">{turnover_table}</div></section>
<section><h2>成交额与成本</h2><div class="data-wrap">{cost_table}</div></section>
<section><h2>收益与风险</h2><p>收益/夏普差异列是有效交易配对的平均差；收益/风险均值列覆盖本表全部账户，因而差异不必等于两列全账户均值之差。改善路线数按有效配对计；零交易账户的 Sharpe=0 不代表策略改善或具备资格。</p><div class="data-wrap">{performance_table}</div></section>
<section><h2>静态配对图</h2>{chart_html or '<p class="empty">当前账户结果不足以生成图表。</p>'}</section>
<section><h2>逐路线配对明细</h2><details><summary>展开路线、区间、调仓方式和成本情景的配对值与差异</summary><div class="data-wrap">{details_html}</div></details><p><a href="paired/turnover_effects.csv" download>下载逐路线配对差异 CSV</a>（保留原路线编号，便于复核）。</p></section>
{diagnostics_block}{_legacy_section()}
<footer>HTML 与配对差异 CSV 可重复生成；账户和模型文件保持原样。</footer>'''
    output_html.write_text(_page(horizon, body), encoding="utf-8")
    return output_html, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 1h/4h 多因子调仓换手率配对报告")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="包含 h1 与 h4 目录的实验根目录")
    parser.add_argument("--horizon", choices=("all", "h1", "h4"), default="all", help="生成所有报告或指定 horizon")
    args = parser.parse_args()
    horizons = ("h1", "h4") if args.horizon == "all" else (args.horizon,)
    for horizon in horizons:
        output_html, output_csv = build_report(args.root / horizon)
        print(f"{horizon}: {output_html} ; {output_csv}")


if __name__ == "__main__":
    main()
