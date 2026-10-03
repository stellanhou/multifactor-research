"""Run the deterministic, no-Agent multi-factor research baseline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .multifactor_account import generate_rank_targets, run_perpetual_account
from .multifactor_contracts import MultifactorContract
from .multifactor_data import load_inputs
from .multifactor_factors import FactorPanels, process_factors, read_cards


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n",
                    encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _code_version() -> dict[str, Any]:
    root = _project_root()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.splitlines()
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("multifactor_contracts.py"),
        Path(__file__).with_name("multifactor_data.py"),
        Path(__file__).with_name("multifactor_factors.py"),
        Path(__file__).with_name("multifactor_account.py"),
    ]
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    return {
        "git_commit": commit,
        "working_tree_clean": not status,
        "working_tree_status": status,
        "source_sha256": hashes,
    }


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")


def _snapshot_cards(cards: list[dict[str, Any]], card_paths: list[Path], root: Path) -> list[dict[str, Any]]:
    directory = root / "cards"
    directory.mkdir()
    manifest = []
    for index, (card, source) in enumerate(zip(cards, card_paths), start=1):
        payload = source.read_bytes()
        card_id = card["id"]
        filename = f"{index:03d}_{_safe_label(card_id)}.json"
        destination = directory / filename
        destination.write_bytes(payload)
        manifest.append({
            "id": card_id,
            "title": card["title"],
            "source_path": str(source),
            "snapshot_path": f"cards/{filename}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "direction": card["direction"],
            "horizon_hours": card["horizon_hours"],
            "lookback_hours": card["lookback_hours"],
            "expression": card["expression"],
        })
    _write_json(root / "cards.json", manifest)
    return manifest


def _snapshot_inputs(inputs, root: Path) -> None:
    directory = root / "inputs"
    market_directory = directory / "market"
    market_directory.mkdir(parents=True)
    inputs.panel.values.to_csv(directory / "panel.values.csv")
    inputs.universe.rename("eligible").to_csv(directory / "universe.csv")
    for symbol, frame in sorted(inputs.frames.items()):
        frame.to_csv(market_directory / f"{_safe_label(symbol)}.csv")
    inputs.funding.to_csv(directory / "funding.csv", index=False)
    _write_json(directory / "data-provenance.json", inputs.diagnostics)
    _write_json(directory / "panel-diagnostics.json", inputs.panel.diagnostics)


def _snapshot_factor_panels(factors: FactorPanels, root: Path) -> None:
    directory = root / "factor_panels"
    directory.mkdir()
    factors.raw.to_csv(directory / "raw.csv")
    factors.standardized.to_csv(directory / "standardized.csv")
    factors.eligible.rename("eligible").to_csv(directory / "eligible.csv")
    factors.valid_masks.to_csv(directory / "valid_masks.csv")
    factors.common_mask.rename("common_mask").to_csv(directory / "common_mask.csv")
    factors.coverage.to_csv(directory / "coverage.csv")
    factors.correlations.to_csv(directory / "correlations.csv", index=False)
    factors.correlation_summary.to_csv(directory / "correlation_summary.csv")
    _write_json(directory / "input-diagnostics.json", factors.input_diagnostics)


def _score_matrix(values: pd.Series, common_mask: pd.Series, start: pd.Timestamp,
                  end: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    usable = values.where(common_mask)
    timestamps = usable.index.get_level_values("timestamp")
    window = (timestamps >= start - pd.Timedelta(hours=1)) & (timestamps < end - pd.Timedelta(hours=1))
    selected = usable.loc[window]
    scores = selected.unstack("symbol").sort_index().sort_index(axis=1)
    complete_grid = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2), freq="h")
    if not scores.index.equals(complete_grid):
        raise ValueError("common-mask scores do not contain the complete account signal grid")
    if list(scores.columns) != sorted(symbols):
        raise ValueError("common-mask score symbols do not match the frozen account universe")
    return scores


def _account_kwargs(contract: MultifactorContract, fee_bps: float, slippage_bps: float) -> dict[str, Any]:
    start, end = contract.bounds
    return {
        "initial_capital": contract.costs["initial_capital"],
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "start": start,
        "end": end,
        "margin_fraction": contract.portfolio["margin_fraction"],
    }


def _write_account_result(directory: Path, result) -> dict[str, Any]:
    for name in ("orders", "fills", "funding_events", "positions", "ledger"):
        getattr(result, name).to_csv(directory / f"{name}.csv")
    _write_json(directory / "metrics.json", result.metrics)
    return result.metrics


def _run_one(name: str, kind: str, included: list[str], score: pd.Series,
             inputs, factors: FactorPanels, contract: MultifactorContract,
             directory: Path, sample: dict[str, Any], factor_codes: dict[str, str]) -> dict[str, Any]:
    start, end = contract.bounds
    signals = _score_matrix(score, factors.common_mask, start, end, list(inputs.frames))
    portfolio = contract.portfolio
    targets = generate_rank_targets(
        signals,
        long_count=portfolio["long_count"],
        short_count=portfolio["short_count"],
        gross_exposure=portfolio["gross_exposure"],
        max_asset_weight=portfolio["max_asset_weight"],
        rebalance_hours=portfolio["rebalance_hours"],
        start=start,
    )
    directory.mkdir(parents=True)
    signals.to_csv(directory / "signals.csv")
    targets.to_csv(directory / "targets.csv")
    account = run_perpetual_account(inputs.frames, targets, inputs.funding,
                                    **_account_kwargs(contract, contract.costs["fee_bps"],
                                                      contract.costs["slippage_bps"]))
    metrics = _write_account_result(directory, account)
    return {
        "name": name,
        "kind": kind,
        "included_factors": [factor_codes[card_id] for card_id in included],
        "included_card_ids": included,
        "sample": {
            **sample,
            "account_signal_rows": int(signals.notna().any(axis=1).sum()),
            "account_signal_grid_hours": len(signals),
        },
        "metrics": metrics,
        "path": f"experiments/{directory.name}",
        "artifacts": {
            name: f"experiments/{directory.name}/{name}.csv"
            for name in ("signals", "targets", "orders", "fills", "funding_events", "positions", "ledger")
        } | {"metrics": f"experiments/{directory.name}/metrics.json"},
    }


def _run_stress(equal_weight: dict[str, Any], score: pd.Series, inputs,
                factors: FactorPanels, contract: MultifactorContract,
                directory: Path) -> dict[str, Any] | None:
    multiplier = contract.costs["stress_multiplier"]
    if multiplier == 1:
        return None
    start, end = contract.bounds
    signals = _score_matrix(score, factors.common_mask, start, end, list(inputs.frames))
    portfolio = contract.portfolio
    targets = generate_rank_targets(
        signals,
        long_count=portfolio["long_count"],
        short_count=portfolio["short_count"],
        gross_exposure=portfolio["gross_exposure"],
        max_asset_weight=portfolio["max_asset_weight"],
        rebalance_hours=portfolio["rebalance_hours"],
        start=start,
    )
    account = run_perpetual_account(
        inputs.frames, targets, inputs.funding,
        **_account_kwargs(contract, contract.costs["fee_bps"] * multiplier,
                          contract.costs["slippage_bps"] * multiplier),
    )
    directory.mkdir(parents=True)
    signals.to_csv(directory / "signals.csv")
    targets.to_csv(directory / "targets.csv")
    metrics = _write_account_result(directory, account)
    return {"multiplier": multiplier, "metrics": metrics, "path": f"experiments/{directory.name}",
            "applies_to": equal_weight["name"]}


def run_experiments(inputs, contract: MultifactorContract, output_root: Path,
                    cards: list[dict[str, Any]], code_version: dict[str, Any]) -> dict[str, Any]:
    """Execute all single, equal-weight, and leave-one-out baseline portfolios."""
    output_root = Path(output_root)
    factors = process_factors(inputs.panel, cards)
    _snapshot_factor_panels(factors, output_root)
    start, end = contract.bounds
    account_start = start - pd.Timedelta(hours=1)
    signal_times = factors.common_mask.index.get_level_values("timestamp")
    account_signal_end = end - pd.Timedelta(hours=1)
    account_period = (signal_times >= account_start) & (signal_times < account_signal_end)
    account_eligible = factors.eligible & account_period
    account_common_mask = factors.common_mask & account_period
    if not account_common_mask.any():
        raise ValueError("no shared valid factor observations in the account interval")
    sample = {
        "factor_panel_eligible_rows_including_warmup": int(factors.eligible.sum()),
        "factor_panel_common_rows_including_warmup": int(factors.common_mask.sum()),
        "account_signal_window_eligible_rows": int(account_eligible.sum()),
        "account_signal_window_common_rows": int(account_common_mask.sum()),
        "account_signal_window_start": account_start.isoformat(),
        "account_signal_window_end_exclusive": account_signal_end.isoformat(),
    }

    card_ids = [card["id"] for card in cards]
    factor_codes = {card_id: f"F{position}" for position, card_id in enumerate(card_ids, start=1)}
    standardized = factors.standardized
    experiments = []
    experiment_root = output_root / "experiments"
    experiment_root.mkdir()
    for position, card_id in enumerate(card_ids, start=1):
        score = standardized[card_id].rename("score")
        label = f"single:{factor_codes[card_id]}"
        directory = experiment_root / f"single_{position:03d}_{_safe_label(card_id)}"
        experiments.append(_run_one(label, "single_factor", [card_id], score, inputs,
                                    factors, contract, directory, sample, factor_codes))

    equal_score = standardized[card_ids].mean(axis=1).rename("score")
    equal_weight = _run_one("equal_weight", "equal_weight", card_ids, equal_score, inputs,
                            factors, contract, experiment_root / "equal_weight", sample, factor_codes)
    experiments.append(equal_weight)
    stress = _run_stress(equal_weight, equal_score, inputs, factors, contract,
                         experiment_root / "equal_weight_stress")

    for position, dropped in enumerate(card_ids, start=1):
        included = [card_id for card_id in card_ids if card_id != dropped]
        score = standardized[included].mean(axis=1).rename("score")
        label = f"drop:{factor_codes[dropped]}"
        directory = experiment_root / f"drop_{position:03d}_{_safe_label(dropped)}"
        experiments.append(_run_one(label, "leave_one_out", included, score, inputs,
                                    factors, contract, directory, sample, factor_codes))

    equal_metrics = equal_weight["metrics"]
    comparison_metrics = (
        "net_return", "max_drawdown", "sharpe_ratio", "total_turnover",
        "total_fees", "total_slippage_cost", "total_funding",
    )
    for experiment in experiments:
        deltas = {
            metric: experiment["metrics"][metric] - equal_metrics[metric]
            for metric in comparison_metrics
        }
        experiment["net_return_delta_vs_equal_weight"] = deltas["net_return"]
        experiment["delta_vs_equal_weight"] = deltas

    coverage = factors.coverage.copy()
    coverage["panel_shared_common_rows_including_warmup"] = int(factors.common_mask.sum())
    coverage["panel_shared_coverage_ratio_including_warmup"] = (
        int(factors.common_mask.sum()) / int(factors.eligible.sum())
    )
    coverage["account_eligible_rows"] = int(account_eligible.sum())
    coverage["account_raw_valid_rows"] = factors.valid_masks.loc[account_eligible].sum(axis=0).reindex(coverage.index)
    coverage["account_raw_coverage_ratio"] = (
        coverage["account_raw_valid_rows"] / int(account_eligible.sum())
    )
    coverage["account_shared_common_rows"] = int(account_common_mask.sum())
    coverage["account_shared_coverage_ratio"] = (
        int(account_common_mask.sum()) / int(account_eligible.sum())
    )
    coverage["account_rows_excluded_by_shared_mask"] = (
        coverage["account_raw_valid_rows"] - coverage["account_shared_common_rows"]
    )
    coverage.to_csv(output_root / "coverage_impact.csv")

    result = {
        "engine": "deterministic_multifactor_v1",
        "model_called": False,
        "agent_used": False,
        "status": "engineering_complete",
        "run_id": contract.run_id,
        "root": str(output_root.resolve()),
        "purpose": contract.purpose,
        "stage": contract.stage,
        "forward_validation_started": False,
        "paper_started": False,
        "published": False,
        "upstream_cards_modified": False,
        "factors": [
            {
                "code": factor_codes[card["id"]],
                "id": card["id"],
                "title": card["title"],
                "source_path": card["path"],
                "snapshot_path": f"cards/{position:03d}_{_safe_label(card['id'])}.json",
            }
            for position, card in enumerate(cards, start=1)
        ],
        "contract": "contract.json",
        "code_version": code_version,
        "experiments": experiments,
        "stress_costs": stress,
        "coverage_impact": {
            **sample,
            "account_shared_coverage_ratio": int(account_common_mask.sum()) / int(account_eligible.sum()),
            "path": "coverage_impact.csv",
        },
        "factor_diagnostics": {
            "raw": "factor_panels/raw.csv",
            "standardized": "factor_panels/standardized.csv",
            "correlations": "factor_panels/correlations.csv",
            "correlation_summary": "factor_panels/correlation_summary.csv",
        },
        "input_snapshot": "inputs/",
        "card_snapshot": "cards/",
    }
    _write_json(output_root / "result.json", result)
    _write_report(output_root / "report.md", contract, result, cards)
    return result


def _write_report(path: Path, contract: MultifactorContract, result: dict[str, Any],
                  cards: list[dict[str, Any]]) -> None:
    start, end = contract.bounds
    lines = [
        "# 多因子传统基线结果",
        "",
        f"- 运行：`{contract.run_id}`；状态：`{result['status']}`；用途：`{contract.purpose}` / `{contract.stage}`",
        f"- 时段：`[{start.isoformat()}, {end.isoformat()})`；目标期限：{contract.horizon_hours}h；预热：{contract.warmup_hours}h",
        f"- 因子数：{len(cards)}；`model_called=false`；未改变上游卡片状态；未启动前向验证或发布。",
        f"- 数据处理合同：{contract.data_processing}",
        "- 数据来源的实际处理与因果性标记见 [data-provenance.json](inputs/data-provenance.json)。",
        "- Spearman逐时截面相关和汇总见 [correlations.csv](factor_panels/correlations.csv) 与 [correlation_summary.csv](factor_panels/correlation_summary.csv)；报告包含绝对相关与分位数，供检查阶段变化和正负抵消。",
        "- 因子比较、单因子和逐一移除结果均使用所有因子的共同有效样本；覆盖率主口径只计入账户信号窗，预热与全输入面板另列，见 [coverage_impact.csv](coverage_impact.csv)。",
        "- 每次逐一移除后，剩余因子在相同共同样本上重新等权；权重没有拟合，仓位、调仓、成交成本与账户规则保持合同不变。等权及各消融的基础指标差值保存在 result.json。",
        "",
        "## 因子代码",
        "",
        "| 代码 | 创意卡 ID | 标题 | 源文件 |",
        "|---|---|---|---|",
    ]
    for factor in result["factors"]:
        title = factor["title"].replace("|", "\\|")
        source = factor["source_path"].replace("|", "\\|")
        lines.append(f"| {factor['code']} | `{factor['id']}` | {title} | `{source}` |")
    lines.extend([
        "",
        "## 组合结果",
        "",
        "| 实验 | 纳入因子 | 净收益 | 对等权净收益差 | 最大回撤 | 对等权回撤差 | 换手 | 手续费 | 滑点 | 资金费 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for experiment in result["experiments"]:
        metrics = experiment["metrics"]
        included = ", ".join(experiment["included_factors"])
        lines.append(
            f"| {experiment['name']} | {included} | {metrics['net_return']:.2%} | "
            f"{experiment['net_return_delta_vs_equal_weight']:+.2%} | "
            f"{metrics['max_drawdown']:.2%} | "
            f"{experiment['delta_vs_equal_weight']['max_drawdown']:+.2%} | "
            f"{metrics['total_turnover']:.4g} | "
            f"{metrics['total_fees']:.4g} | {metrics['total_slippage_cost']:.4g} | "
            f"{metrics['total_funding']:.4g} |"
        )
    lines.append("最大回撤按正幅度记录；相对差值为负表示回撤小于等权组合。资金费为账户现金流，正值表示收入。")
    if result["stress_costs"] is not None:
        stress = result["stress_costs"]
        metrics = stress["metrics"]
        lines.extend([
            "",
            f"等权组合成本压力（费率与滑点各乘 {stress['multiplier']:g}）：净收益 "
            f"{metrics['net_return']:.2%}，最大回撤 {metrics['max_drawdown']:.2%}。",
        ])
    lines.extend([
        "",
        "## 解释边界",
        "",
        "报告描述的是指定历史区间内的工程回放。收益为负或组合没有增量不影响流程是否完成；该结果本身不证明独立样本外表现，也不构成前向验证。",
        "所有实验每段从独立初始资金开始；信号范围为 `[start-1h, end-1h)`，最后一小时只用于账户收盘估值，不再生成会在合同结束后成交的信号。订单、成交、逐事件资金费、持仓和逐时账本保存在各实验目录。信号时间按 UTC bar-open 标记，成交规则由确定性账户模块执行。",
        "",
        "## 产物",
        "",
        "- [冻结合同](contract.json)",
        "- [卡片快照清单](cards.json)",
        "- [输入快照](inputs/)",
        "- [因子面板与相关性](factor_panels/)",
        "- [运行结果与代码版本](result.json)",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baseline(contract_path: Path, db: Path, output: Path) -> dict[str, Any]:
    """Freeze the contract and source cards, then run the no-Agent baseline."""
    contract_path = Path(contract_path).resolve()
    contract = MultifactorContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
    if contract.schema_version != 1:
        raise ValueError("research datasets use the explicit research-baseline entry point, without --db")
    output_root = Path(output) / contract.run_id
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "contract.json", contract.as_dict())

    card_paths = []
    for path in contract.cards:
        resolved = Path(path).expanduser()
        card_paths.append(resolved.resolve() if resolved.is_absolute()
                          else (contract_path.parent / resolved).resolve())
    cards = read_cards(card_paths, contract.horizon_hours, contract.warmup_hours)
    _snapshot_cards(cards, card_paths, output_root)

    inputs = load_inputs(db, contract, contract_path.parent)
    _snapshot_inputs(inputs, output_root)
    code_version = _code_version()
    usage = {
        "purpose": contract.purpose,
        "stage": contract.stage,
        "prior_data_use": contract.prior_data_use,
        "data_processing": contract.data_processing,
        "contract_start": contract.start,
        "contract_end_exclusive": contract.end,
        "horizon_hours": contract.horizon_hours,
        "agent_used": False,
        "upstream_card_status_changed": False,
        "forward_validation_started": False,
        "published": False,
        "source_database": str(Path(db).resolve()),
    }
    _write_json(output_root / "data-usage.json", usage)
    return run_experiments(inputs, contract, output_root, cards, code_version)


def run_research_baseline(contract_path: Path, output: Path) -> dict[str, Any]:
    """Run a versioned raw-source research segment without a primary-DB fallback."""
    from .multifactor_data import load_research_inputs
    contract_path = Path(contract_path).resolve()
    contract = MultifactorContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
    if contract.schema_version != 2 or contract.purpose != "research":
        raise ValueError("research-baseline requires the versioned research contract")
    manifest = (contract_path.parent / contract.dataset_manifest).resolve()
    # Complete source validation precedes output creation and any model call.
    inputs = load_research_inputs(manifest, contract, contract_path.parent)
    card_paths = [(Path(path).resolve() if Path(path).is_absolute()
                   else (contract_path.parent / path).resolve()) for path in contract.cards]
    cards = read_cards(card_paths, contract.horizon_hours, contract.warmup_hours)
    output_root = Path(output) / contract.run_id
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "contract.json", contract.as_dict())
    _snapshot_cards(cards, card_paths, output_root)
    _snapshot_inputs(inputs, output_root)
    _write_json(output_root / "inputs/dataset-manifest.json", inputs.dataset_manifest)
    inputs.funding_mark_provenance.to_csv(output_root / "inputs/funding-mark-provenance.csv", index=False)
    inputs.partial_spot_source_rows.to_csv(output_root / "inputs/partial-spot-source-rows.csv", index=False)
    _write_json(output_root / "inputs/price-row-provenance-summary.json", inputs.price_row_provenance_summary)
    _write_json(output_root / "data-usage.json", {
        "purpose": "research", "stage": contract.stage, "prior_data_use": contract.prior_data_use,
        "data_processing": contract.data_processing, "source_causality": inputs.diagnostics["source_causality"],
        "historical_causality_certified": inputs.diagnostics["historical_causality_certified"],
        "dataset_manifest": str(manifest), "agent_used": False,
        "forward_validation_started": False, "published": False,
    })
    return run_experiments(inputs, contract, output_root, cards, _code_version())
