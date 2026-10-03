"""Frozen five-arm rebalance experiment over an existing ablation's B signals."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, require
from crypto_quant.research.progress import ProgressLog
from .multifactor_ablation import _capture_sources
from .multifactor_account import RebalancePolicy, generate_rank_targets, run_perpetual_account
from .multifactor_contracts import ResearchContract
from .multifactor_historical import _candidate_failures, _sha256
from .multifactor_pool_research import PoolResearchContract, _resolve
from .multifactor_workflow import _account_kwargs, _write_account_result, _write_json


ARMS = (
    {"name": "R0", "rank_buffer": False, "position_buffer": False, "hourly": False},
    {"name": "R1", "rank_buffer": False, "position_buffer": False, "hourly": True},
    {"name": "R2", "rank_buffer": True, "position_buffer": False, "hourly": True},
    {"name": "R3", "rank_buffer": False, "position_buffer": True, "hourly": True},
    {"name": "R4", "rank_buffer": True, "position_buffer": True, "hourly": True},
)
STAGES = ("development", "internal_validation")


def _decisions(account, signals):
    orders = account.orders.copy()
    if "action" not in orders:
        old, new = orders.current_quantity, orders.target_quantity
        changed = old != new
        orders["actual_weight"] = old * orders.signal_close / orders.signal_equity
        orders["execution_weight"] = new * orders.signal_close / orders.signal_equity
        orders["lower_weight"] = orders.target_weight.abs()
        orders["upper_weight"] = orders.target_weight.abs()
        orders["action"] = np.where(~changed, "hold", np.where(new.abs() < old.abs(), "reduce", "add"))
        orders["trade_category"] = np.select(
            [~changed, old * new < 0, (old == 0) | (new == 0)],
            ["hold", "direction_reversal", "symbol_change"], default="resize")
        orders["reason"] = np.select(
            [~changed, old * new < 0, new == 0, old == 0],
            ["at_target", "direction_reversal", "rank_exit", "entry"], default="exact_resize")
        keys = pd.MultiIndex.from_arrays([orders.signal_timestamp, orders.symbol])
        for name, panel in (("score", signals),
                            ("long_rank", signals.rank(axis=1, ascending=False, method="first")),
                            ("short_rank", signals.rank(axis=1, ascending=True, method="first"))):
            orders[name] = panel.stack(future_stack=True).reindex(keys).to_numpy()
        orders.loc[(orders.target_weight == 0) & orders.score.isna() & changed, "reason"] = "missing_signal"
    fills = account.fills.rename(columns={"timestamp": "execution_timestamp"})
    costs = ["fee", "slippage_cost", "notional", "fill_price", "realized_pnl", "new_quantity"]
    audit = orders.merge(fills[["execution_timestamp", "symbol", *costs]],
                         on=["execution_timestamp", "symbol"], how="left", validate="one_to_one")
    # Missing fills are the recorded hold decisions, which incur zero costs.
    audit.loc[audit.action == "hold", ["fee", "slippage_cost", "notional", "realized_pnl"]] = 0.0
    audit.loc[audit.action == "hold", "new_quantity"] = audit.loc[audit.action == "hold", "current_quantity"]
    require(audit.loc[audit.action != "hold", "fee"].notna().all(), "a trading decision has no fill")
    return audit


def _behavior_metrics(account, audit, initial_capital):
    ledger = account.ledger
    signs = np.sign(account.positions.pivot(index="timestamp", columns="symbol", values="quantity"))
    durations = []
    for symbol in signs:
        values = signs[symbol].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        durations.extend(int(right - left) for left, right in zip(boundaries[:-1], boundaries[1:]) if values[left])
    old, new = audit.current_quantity, audit.target_quantity
    switches = pd.DataFrame({
        "long_entry": (old == 0) & (new > 0), "long_exit": (old > 0) & (new == 0),
        "short_entry": (old == 0) & (new < 0), "short_exit": (old < 0) & (new == 0),
        "timestamp": audit.execution_timestamp,
    }).groupby("timestamp").sum()
    metrics = {
        **account.metrics,
        "gross_pnl_including_funding": account.metrics["final_equity"] - initial_capital
            + account.metrics["total_fees"] + account.metrics["total_slippage_cost"],
        "average_gross_exposure": float((ledger.gross_notional / ledger.equity).mean()),
        "average_net_exposure": float((ledger.net_notional / ledger.equity).mean()),
        "position_time_fraction": float((ledger.gross_notional > 0).mean()),
        "average_holding_hours_including_open_episodes": float(np.mean(durations)) if durations else 0.0,
        "entry_count": int(((old == 0) & (new != 0)).sum()),
        "exit_count": int(((old != 0) & (new == 0)).sum()),
        "reversal_count": int((old * new < 0).sum()),
        "coin_switch_count": int(switches[["long_entry", "long_exit"]].min(axis=1).sum()
                                 + switches[["short_entry", "short_exit"]].min(axis=1).sum()),
        "resize_count": int((audit.trade_category == "resize").sum()),
        "no_trade_check_fraction": float(audit.groupby("execution_timestamp").signed_order_quantity
            .apply(lambda row: row.eq(0).all()).mean()),
        "no_trade_hour_fraction": float(ledger.trade_notional.eq(0).mean()),
        "traded_bars": int(ledger.trade_notional.gt(0).sum()),
    }
    categories = audit.groupby("trade_category")[["notional", "fee", "slippage_cost"]].sum()
    return metrics, categories


def _run_arm(directory, signals, frames, funding, contract, arm, request, multiplier):
    directory.mkdir(parents=True)
    portfolio = contract.portfolio
    policy = None
    if arm["hourly"]:
        policy = RebalancePolicy(
            **{name: portfolio[name] for name in ("long_count", "short_count", "gross_exposure", "max_asset_weight")},
            holding_rank=request["holding_rank"] if arm["rank_buffer"] else None,
            weight_buffer=request["weight_buffer"] if arm["position_buffer"] else 0.0)
        targets = None
    else:
        targets = generate_rank_targets(signals, start=contract.start,
            **{name: portfolio[name] for name in ("long_count", "short_count", "gross_exposure",
                                                  "max_asset_weight", "rebalance_hours")})
    account = run_perpetual_account(frames, targets, funding,
        **_account_kwargs(contract, contract.costs["fee_bps"] * multiplier,
                          contract.costs["slippage_bps"] * multiplier),
        scores=signals if policy is not None else None, rebalance_policy=policy)
    _write_account_result(directory, account)
    audit = _decisions(account, signals)
    audit.to_csv(directory / "decisions.csv", index=False)
    audit.pivot(index="signal_timestamp", columns="symbol", values="target_weight").to_csv(directory / "targets.csv")
    metrics, categories = _behavior_metrics(account, audit, contract.costs["initial_capital"])
    categories.to_csv(directory / "turnover_categories.csv")
    monthly = pd.DataFrame({
        "net_return": (1 + account.ledger["return"]).resample("ME").prod() - 1,
        **{name: account.ledger[name].resample("ME").sum() for name in ("fees", "slippage_cost", "funding_cashflow", "turnover")},
        "average_gross_exposure": (account.ledger.gross_notional / account.ledger.equity).resample("ME").mean(),
    })
    monthly.to_csv(directory / "monthly.csv")
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "policy.json", {**arm, "policy": asdict(policy) if policy is not None else None,
                                           "cost_multiplier": multiplier})
    return account, {"metrics": metrics, "path": str(directory),
                     "turnover_categories": categories.reset_index().to_dict("records")}


def _verify_reproduction(account, reference):
    """Compare the old scheduled account's full order, fill and ledger path."""
    deviations = {}
    for name, index_col in (("orders", 0), ("fills", 0), ("ledger", 0)):
        old = pd.read_csv(reference / f"{name}.csv", index_col=index_col, float_precision="round_trip")
        new = getattr(account, name)
        require(len(old) == len(new), f"R0 {name} row count differs from original")
        numeric = new.select_dtypes(include="number").columns
        np.testing.assert_allclose(new[numeric].to_numpy(), old[numeric].to_numpy(),
                                   rtol=2e-11, atol=1e-8, equal_nan=True,
                                   err_msg=f"R0 {name} does not reproduce original B")
        finite = np.isfinite(new[numeric].to_numpy()) & np.isfinite(old[numeric].to_numpy())
        deviations[name] = float(np.max(np.abs(new[numeric].to_numpy()[finite] - old[numeric].to_numpy()[finite]))) if finite.any() else 0.0
    return deviations


def run_rebalance(contract_path: Path, output: Path) -> dict:
    contract_path = Path(contract_path).resolve()
    request = json.loads(contract_path.read_text())
    require(isinstance(request, dict) and set(request) ==
            {"schema_version", "run_id", "source_run", "holding_rank", "weight_buffer"}, "rebalance contract fields differ")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1, "rebalance schema_version must be 1")
    identifier(request["run_id"])
    require(type(request["holding_rank"]) is int and request["holding_rank"] == 4
            and type(request["weight_buffer"]) in (int, float) and request["weight_buffer"] == 0.02,
            "first rebalance experiment freezes holding_rank=4 and weight_buffer=0.02")
    require(isinstance(request["source_run"], str) and request["source_run"].strip(), "source_run is required")
    source = _resolve(request["source_run"], contract_path.parent)
    source_plan = json.loads((source / "plan.json").read_text())
    require(source_plan["arms"][1] == {"name": "B", "orthogonalize": False, "weighting": "rolling_icir"},
            "source must provide the original non-orthogonal rolling ICIR B arm")
    pool = PoolResearchContract.from_dict(json.loads((source / "pool_contract.json").read_text()))
    require(set(pool.horizons) == {1, 4, 24} and source_plan["horizons"] == pool.horizons,
            "rebalance requires frozen 1/4/24h source signals")
    require(pool.portfolio["long_count"] == pool.portfolio["short_count"] == 2
            and pool.portfolio["gross_exposure"] == 0.8 and pool.portfolio["max_asset_weight"] == 0.2
            and pool.costs["stress_multiplier"] == 2, "source portfolio/cost limits differ from the agreed Plan")
    root = Path(output).resolve() / request["run_id"]
    require(not root.exists(), "rebalance run directory already exists; refusing overwrite")
    root.mkdir(parents=True)
    progress = ProgressLog.for_run(root)
    _capture_sources(root)
    _write_json(root / "contract.json", request)
    # Snapshot inputs and all rules before opening any account outcome.
    paths = [source / name for name in ("plan.json", "pool_contract.json", "dataset_manifest.json", "cards_manifest.json")]
    for stage in STAGES:
        paths += sorted((source / stage / "inputs" / "market").glob("*.csv"))
        paths.append(source / stage / "inputs" / "funding.csv")
        for horizon in (24, 4, 1):
            paths += [source / stage / f"h{horizon}" / "contract.json",
                      source / stage / f"h{horizon}" / "experiments" / "B" / "signals.csv"]
    manifest = {}
    for path in paths:
        relative = path.relative_to(source)
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = _sha256(path)
        shutil.copyfile(path, destination)
        require(_sha256(destination) == digest == _sha256(path), "rebalance input changed during snapshot")
        manifest[str(relative)] = digest
    _write_json(root / "source_manifest.json", manifest)
    plan = {
        "arms": ARMS, "horizons": [24, 4, 1], "stages": STAGES, "account_runs": 60,
        "source_run": str(source), "source_arm": "B", "source_manifest_sha256": _sha256(root / "source_manifest.json"),
        "holding_rank": 4, "weight_buffer": 0.02, "portfolio": pool.portfolio, "costs": pool.costs,
        "qualification_gates": pool.qualification_gates, "icir": source_plan["icir"],
        "frozen_before_outcome_access": True, "selection": "all five arms at all horizons in both stages; no C-based reselection",
        "missing_signal": "exit the affected position; empty entry seats remain cash",
        "ties_and_conflicts": "score then symbol; retain old eligible seats first; fill longs then shorts; entry ranks use the full available universe",
        "sizing": "new entries at target; old positions outside band move to nearest boundary; exits bypass band",
        "limits": "signal-close pre-trade equity and prices; asset cap clips bands; gross cap precedes no-trade band",
        "funding_timing": "freeze decisions before execution-open funding; old position pays that event",
        "cost_paths": "base and 2x costs independently decide from their own holdings and equity",
        "prior_data_use": pool.prior_data_use, "model_calls": 0,
    }
    _write_json(root / "plan.json", plan)
    loaded = {}
    trials = []
    for horizon in plan["horizons"]:
        for stage in STAGES:
            stage_source = root / "source" / stage
            if stage not in loaded:
                frames = {p.stem: pd.read_csv(p, index_col=0, parse_dates=[0], float_precision="round_trip")
                          for p in sorted((stage_source / "inputs" / "market").glob("*.csv"))}
                funding = pd.read_csv(stage_source / "inputs" / "funding.csv", float_precision="round_trip")
                loaded[stage] = frames, funding
            frames, funding = loaded[stage]
            directory = stage_source / f"h{horizon}"
            contract = ResearchContract.from_dict(json.loads((directory / "contract.json").read_text()))
            require(contract.stage == stage and contract.horizon_hours == horizon
                    and contract.costs == pool.costs and contract.portfolio == {**pool.portfolio, "rebalance_hours": horizon},
                    "source stage contract differs from frozen pool")
            signals = pd.read_csv(directory / "experiments" / "B" / "signals.csv", index_col=0,
                                  parse_dates=[0], float_precision="round_trip")
            for arm in ARMS:
                arm_root = root / stage / f"h{horizon}" / arm["name"]
                paths = []
                for cost, multiplier in (("base", 1), ("stress", 2)):
                    with progress.span("rebalance.account", heartbeat=True, horizon=horizon,
                                       stage=stage, arm=arm["name"], cost=cost):
                        account, result = _run_arm(arm_root / cost, signals, frames, funding, contract, arm, request, multiplier)
                    if arm["name"] == "R0":
                        ref = source / stage / f"h{horizon}" / "experiments" / ("B" if cost == "base" else "B_stress")
                        result["reproduction_max_absolute_error"] = _verify_reproduction(account, ref)
                    paths.append(result)
                failures = _candidate_failures(paths[0]["metrics"], paths[1]["metrics"],
                                                paths[0]["metrics"]["traded_bars"], pool.qualification_gates)
                trial = {**arm, "stage": stage, "horizon_hours": horizon, **paths[0],
                         "stress_costs": paths[1], "qualification_failures": failures, "qualified": not failures}
                trials.append(trial)
                _write_json(arm_root / "result.json", trial)
                _write_json(root / "progress_results.json", trials)
    result = {"schema_version": 1, "engine": "position_aware_rebalance_v1", "run_id": request["run_id"],
              "root": str(root), "status": "rebalance_complete", "plan": plan, "plan_sha256": _sha256(root / "plan.json"),
              "trials": trials, "model_called": False, "paper_started": False, "forward_validation_started": False,
              "historical_causality_certified": json.loads((root / "source" / "dataset_manifest.json").read_text())["historical_causality_certified"]}
    rows = [{"stage": item["stage"], "horizon_hours": item["horizon_hours"], "arm": item["name"],
             **item["metrics"], "stress_net_return": item["stress_costs"]["metrics"]["net_return"],
             "qualified": item["qualified"]} for item in trials]
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    _write_json(root / "result.json", result)
    _report(root, result)
    return result


def _report(root, result):
    lines = ["# 信号触发调仓五组对照", "", "固定原四组消融实验的B组信号：不正交＋30天滚动ICIR。",
             "R0原周期；R1每小时精确；R2排名缓冲；R3仓位缓冲；R4双缓冲。入场前/后2名，持有前/后4名；仓位缓冲2个百分点。",
             "新仓开到20%；已有仓在18%～20%内保持合约数量，越界调整至最近边界；缺失信号退出。基础与两倍成本各自决策。", ""]
    for stage in STAGES:
        lines += [f"## {'A+B开发段' if stage == 'development' else 'C内部历史比较'}", "",
                  "| 期限 | 组别 | 净收益 | 夏普 | 波动 | 回撤 | 换手 | 两倍成本净收益 | 平均敞口 | 不交易小时 | 达标 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for item in result["trials"]:
            if item["stage"] != stage:
                continue
            m, stress = item["metrics"], item["stress_costs"]["metrics"]
            lines.append(f"| {item['horizon_hours']}h | {item['name']} | {m['net_return']:.2%} | {m['sharpe_ratio']:.3f} | "
                         f"{m['annualized_volatility']:.2%} | {m['max_drawdown']:.2%} | {m['total_turnover']:.1f} | "
                         f"{stress['net_return']:.2%} | {m['average_gross_exposure']:.2%} | {m['no_trade_hour_fraction']:.2%} | {'是' if item['qualified'] else '否'} |")
        lines += ["", "| 期限 | 组别 | 节省成本 | 毛盈亏变化（含资金费） | 净盈亏变化 | 平均敞口变化 | 评价依据 |",
                  "|---|---|---:|---:|---:|---:|---|"]
        stage_trials = [item for item in result["trials"] if item["stage"] == stage]
        controls = {item["horizon_hours"]: item["metrics"] for item in stage_trials if item["name"] == "R0"}
        for item in stage_trials:
            m, control = item["metrics"], controls[item["horizon_hours"]]
            saved = control["total_fees"] + control["total_slippage_cost"] - m["total_fees"] - m["total_slippage_cost"]
            gross = m["gross_pnl_including_funding"] - control["gross_pnl_including_funding"]
            net = m["final_equity"] - control["final_equity"]
            exposure = m["average_gross_exposure"] - control["average_gross_exposure"]
            assessment = "原版对照" if item["name"] == "R0" else (
                "净表现改善" if net > 0 else "净表现未改善")
            if item["name"] != "R0" and m["sharpe_ratio"] > control["sharpe_ratio"] and m["max_drawdown"] < control["max_drawdown"]:
                assessment += "；夏普/回撤改善"
            if exposure < -0.02:
                assessment += "；敞口下降超过2个百分点"
            if item["qualification_failures"]:
                assessment += "；未过门槛：" + ", ".join(item["qualification_failures"])
            lines.append(f"| {item['horizon_hours']}h | {item['name']} | {saved:+.2f} | {gross:+.2f} | {net:+.2f} | {exposure:+.2%} | {assessment} |")
        lines.append("")
    lines += ["R0逐项复核原B组订单、成交和账本；每组decisions.csv记录排名、目标/实际仓位、动作、原因、数量及成交成本。",
              "turnover_categories.csv拆分换币、方向反转、比例微调；monthly.csv给出分月份结果。平均持有时间包括期末未平仓区间。",
              "所有规则在读取结果前冻结；C已暴露，继续作为内部历史比较。逐段门槛为净收益≥0、回撤≤15%、存在成交、两倍成本净收益≥0。",
              "毛盈亏加回该账户实际手续费与滑点并包含资金费，用于会计拆分；它不等同于重新运行零成本账户。",
              "[冻结规则](plan.json) · [完整指标](comparison.csv) · [结构化结果](result.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
