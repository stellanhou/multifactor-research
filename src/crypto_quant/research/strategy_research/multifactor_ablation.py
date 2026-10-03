"""Predeclared 2x2 experiment: symmetric orthogonalization x rolling ICIR."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, require
from crypto_quant.research.progress import ProgressLog
from .multifactor_combination import rolling_icir_combine, symmetric_orthogonalize
from .multifactor_contracts import ResearchContract
from .multifactor_data import load_research_inputs
from .multifactor_factors import process_factors, read_cards
from .multifactor_historical import _capture_code_version, _sha256
from .multifactor_pool import scan_idea_pool
from .multifactor_pool_research import AVAILABLE_FIELDS, PoolResearchContract, _resolve, _sample
from .multifactor_workflow import (
    _run_one, _run_stress, _snapshot_cards, _snapshot_factor_panels,
    _snapshot_inputs, _write_json,
)


ARMS = (
    {"name": "A", "orthogonalize": False, "weighting": "equal"},
    {"name": "B", "orthogonalize": False, "weighting": "rolling_icir"},
    {"name": "C", "orthogonalize": True, "weighting": "equal"},
    {"name": "D", "orthogonalize": True, "weighting": "rolling_icir"},
)
METRICS = ("net_return", "max_drawdown", "sharpe_ratio", "total_turnover",
           "total_fees", "total_slippage_cost", "total_funding")


def factorial_effects(results: list[dict]) -> dict:
    require({item["name"] for item in results} == {"A", "B", "C", "D"}
            and len(results) == 4, "ablation requires exactly four arms")
    arms = {item["name"]: item for item in results}
    effects = {}
    for cost in ("base", "stress"):
        metrics = {name: (item["metrics"] if cost == "base" else item["stress_costs"]["metrics"])
                   for name, item in arms.items()}
        deltas = {label: {metric: metrics[left][metric] - metrics[right][metric] for metric in METRICS}
                  for label, left, right in (("B-A", "B", "A"), ("C-A", "C", "A"),
                                             ("D-C", "D", "C"), ("D-B", "D", "B"), ("D-A", "D", "A"))}
        deltas["interaction_D-B-C+A"] = {
            metric: metrics["D"][metric] - metrics["B"][metric] - metrics["C"][metric] + metrics["A"][metric]
            for metric in METRICS
        }
        deltas["weighting_main_effect"] = {
            metric: (deltas["B-A"][metric] + deltas["D-C"][metric]) / 2 for metric in METRICS
        }
        deltas["orthogonalization_main_effect"] = {
            metric: (deltas["C-A"][metric] + deltas["D-B"][metric]) / 2 for metric in METRICS
        }
        effects[cost] = deltas
    return effects


def four_arm_scores(factors, opens: pd.Series, horizon: int, policy: dict,
                    histories: dict | None = None):
    """Use one factor pool and intersect availability before any account run."""
    values = factors.standardized.where(factors.common_mask, axis=0)
    transformed, diagnostics = symmetric_orthogonalize(values, factors.common_mask)
    histories = {} if histories is None else histories
    require(set(histories) in (set(), {"B", "D"}), "rolling history must contain both B and D")
    combined = {
        "B": rolling_icir_combine(values, pd.DataFrame(index=diagnostics.index), opens,
                                  horizon_hours=horizon, policy=policy, history=histories.get("B")),
        "D": rolling_icir_combine(transformed, diagnostics, opens,
                                  horizon_hours=horizon, policy=policy, history=histories.get("D")),
    }
    scores = {"A": values.mean(axis=1, skipna=False), "B": combined["B"].score,
              "C": transformed.mean(axis=1, skipna=False), "D": combined["D"].score}
    availability = pd.DataFrame({name: score.notna() for name, score in scores.items()})
    shared = factors.common_mask & availability.all(axis=1)
    scores = {name: score.where(shared).rename("score") for name, score in scores.items()}
    return scores, shared.rename("paired_mask"), availability, combined, transformed, diagnostics


def _read_history(root: Path) -> dict:
    history = {}
    for arm in ("B", "D"):
        directory = root / "combination" / arm
        values = pd.read_csv(directory / "history_factors.csv", index_col=[0, 1], parse_dates=[0],
                             float_precision="round_trip")
        opens = pd.read_csv(directory / "history_opens.csv", index_col=[0, 1], parse_dates=[0],
                            float_precision="round_trip")["perp_open"]
        history[arm] = values, opens
    return history


def _execute_horizon(root: Path, inputs, contract: ResearchContract, cards: list[dict],
                     policy: dict, progress: ProgressLog, history_root: Path | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "contract.json", contract.as_dict())
    _snapshot_cards(cards, [Path(card["path"]) for card in cards], root)
    with progress.span("ablation.factors", heartbeat=True, stage=contract.stage, horizon=contract.horizon_hours):
        factors = process_factors(inputs.panel, cards, include_correlations=False)
    histories = _read_history(history_root) if history_root is not None else None
    with progress.span("ablation.scores", heartbeat=True, stage=contract.stage, horizon=contract.horizon_hours):
        scores, shared, availability, combined, transformed, diagnostics = four_arm_scores(
            factors, inputs.panel.values["perp_open"], contract.horizon_hours, policy, histories,
        )
    factors = replace(factors, common_mask=shared)
    _snapshot_factor_panels(factors, root)
    transformed.to_csv(root / "factor_panels" / "orthogonalized.csv")
    diagnostics.to_csv(root / "factor_panels" / "orthogonalization_diagnostics.csv")
    availability.to_csv(root / "factor_panels" / "arm_availability.csv")
    shared.to_csv(root / "factor_panels" / "paired_mask.csv")
    cutoff = contract.bounds[1] - pd.Timedelta(hours=policy["window_hours"] + contract.horizon_hours + 1)
    tail = inputs.panel.values.index.get_level_values("timestamp") >= cutoff
    for name, combination in combined.items():
        directory = root / "combination" / name
        directory.mkdir(parents=True)
        for field in ("rank_ic", "mature_ic_count", "icir", "weights", "directions"):
            getattr(combination, field).to_csv(directory / f"{field}.csv")
        combination.orthogonalized.loc[tail].to_csv(directory / "history_factors.csv")
        inputs.panel.values.loc[tail, "perp_open"].to_csv(directory / "history_opens.csv")
    _write_json(root / "score_policy.json", {
        "arms": ARMS, "icir": policy, "ic_maturity_delay_hours": contract.horizon_hours + 1,
        "paired_mask": "all factors complete, at least 3 common symbols, B and D ICIR available; intersection of all four arms",
        "inactive_icir": "insufficient history or zero IC variance: zero factor weight",
        "negative_icir": "absolute ICIR weight and simultaneous factor direction flip in B and D",
        "history_root": str(history_root) if history_root is not None else None,
    })
    sample = _sample(factors, contract)
    selected = [card["id"] for card in cards]
    codes = {card_id: f"F{index}" for index, card_id in enumerate(selected, 1)}
    results = []
    for arm in ARMS:
        name = arm["name"]
        with progress.span("ablation.account", heartbeat=True, stage=contract.stage,
                           horizon=contract.horizon_hours, arm=name):
            result = _run_one(name, "factorial_ablation", selected, scores[name], inputs, factors,
                              contract, root / "experiments" / name, sample, codes)
            stress = _run_stress(result, scores[name], inputs, factors, contract,
                                 root / "experiments" / f"{name}_stress")
        require(stress is not None, "ablation requires the declared cost stress run")
        results.append({**result, **arm, "stress_costs": stress, "horizon_hours": contract.horizon_hours,
                        "stage": contract.stage})
    require(len({item["sample"]["account_signal_rows"] for item in results}) == 1,
            "four arms do not share signal availability")
    result = {"horizon_hours": contract.horizon_hours, "stage": contract.stage,
              "experiments": results, "effects": factorial_effects(results), "sample": sample}
    _write_json(root / "result.json", result)
    return result


def _capture_sources(root: Path) -> None:
    project = Path(__file__).resolve().parents[4]
    version = _capture_code_version()
    for filename in ("multifactor_ablation.py", "multifactor_combination.py", "multifactor_pool.py", "multifactor_pool_research.py"):
        path = Path(__file__).with_name(filename)
        version["source_sha256"][str(path.relative_to(project))] = _sha256(path)
    _write_json(root / "code_version.json", version)
    paths = sorted((project / "src" / "crypto_quant").rglob("*.py")) + [project / "pyproject.toml", project / "uv.lock"]
    manifest = {}
    for source in paths:
        relative = source.relative_to(project)
        destination = root / "execution_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        digest = _sha256(destination)
        require(digest == _sha256(source), "ablation source changed during capture")
        manifest[str(relative)] = digest
    require(all(manifest[path] == digest for path, digest in version["source_sha256"].items()),
            "ablation source snapshot differs from captured version")
    _write_json(root / "execution_source_manifest.json", manifest)
    _write_json(root / "runtime.json", {"python": sys.version, "platform": platform.platform(),
                                        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}})


def run_ablation(contract_path: Path, output: Path) -> dict:
    contract_path = Path(contract_path).resolve()
    request = json.loads(contract_path.read_text())
    require(isinstance(request, dict) and set(request) == {"schema_version", "run_id", "pool_contract"},
            "ablation contract fields differ from schema")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1,
            "ablation schema_version must be 1")
    identifier(request["run_id"])
    require(isinstance(request["pool_contract"], str) and request["pool_contract"].strip(), "pool_contract is required")
    pool_path = _resolve(request["pool_contract"], contract_path.parent)
    contract = PoolResearchContract.from_dict(json.loads(pool_path.read_text()))
    require(contract.schema_version == 2, "ablation requires the frozen rolling ICIR pool contract")
    contract = replace(contract, run_id=request["run_id"])
    pool, manifest_path, universe_path = (_resolve(getattr(contract, name), pool_path.parent)
                                         for name in ("idea_pool", "dataset_manifest", "universe"))
    root = Path(output).resolve() / contract.run_id
    require(not root.exists(), "ablation run directory already exists; refusing overwrite")
    root.mkdir(parents=True)
    progress = ProgressLog.for_run(root)
    _write_json(root / "contract.json", request)
    _write_json(root / "pool_contract.json", contract.__dict__)
    _capture_sources(root)
    raw_manifest = json.loads(manifest_path.read_text())
    _write_json(root / "dataset_manifest.json", raw_manifest)
    (root / "universe.csv").write_bytes(universe_path.read_bytes())
    groups, snapshots = {}, {}
    for horizon in contract.horizons:
        catalog = scan_idea_pool(pool, horizon, contract.warmup_hours, AVAILABLE_FIELDS)
        directory = root / "catalogs"
        directory.mkdir(exist_ok=True)
        _write_json(directory / f"h{horizon}.json", catalog)
        require(not any(entry["status"] == "error" for entry in catalog["entries"]), "malformed factor pool records")
        families = sorted((family for group in catalog["groups"] for family in group["structure_families"]),
                          key=lambda family: family["family_id"])
        require(len(families) >= 2, "ablation needs at least two family representatives per horizon")
        card_paths = []
        for family in families:
            card_id, source = family["representative_id"], Path(family["representative_path"])
            entry = next(entry for entry in catalog["entries"] if entry.get("id") == card_id and entry["status"] == "admitted")
            payload = source.read_bytes()
            destination = root / "cards" / f"{identifier(card_id)}.json"
            destination.parent.mkdir(exist_ok=True)
            require(hashlib.sha256(payload).hexdigest() == entry["card_sha256"], "factor card changed during scan")
            if card_id in snapshots:
                require(snapshots[card_id]["sha256"] == entry["card_sha256"], "cross-horizon factor card changed")
            else:
                destination.write_bytes(payload)
                snapshots[card_id] = {"source": str(source), "snapshot": str(destination.relative_to(root)),
                                      "sha256": entry["card_sha256"]}
            card_paths.append(str(destination))
        groups[str(horizon)] = {"families": families, "card_paths": card_paths,
                                "card_ids": [family["representative_id"] for family in families]}
    _write_json(root / "cards_manifest.json", snapshots)
    plan = {"arms": ARMS, "groups": groups, "horizons": contract.horizons,
            "icir": contract.combination, "minimum_common_symbols": 3,
            "minimum_available_factor_share": 1, "cold_start": "all four arms share B/D ICIR readiness",
            "stages": ["development", "internal_validation"], "account_runs": len(contract.horizons) * 4 * 2 * 2,
            "selection": "none; all four frozen arms run in every horizon and both stages",
            "frozen_before_market_panel_access": True,
            "attribution": "B-A, C-A, D-C, D-B and interaction D-B-C+A; descriptive paired performance differences",
            "weighting_treatment": "rolling ICIR includes dynamic direction flips; cannot separate those two effects",
            "c_results_feedback": False, "model_calls": 0}
    _write_json(root / "plan.json", plan)
    stages = []
    for stage in ("development", "internal_validation"):
        stage_root = root / stage
        stage_root.mkdir()
        first_horizon = contract.horizons[0]
        stage_values = contract.stage_dict(first_horizon, groups[str(first_horizon)]["card_paths"], str(universe_path),
                                           os.path.relpath(manifest_path, root), stage)
        stage_contract = ResearchContract.from_dict(stage_values)
        with progress.span("ablation.load_inputs", heartbeat=True, stage=stage):
            inputs = load_research_inputs(manifest_path, stage_contract, root)
        _snapshot_inputs(inputs, stage_root)
        for horizon in contract.horizons:
            values = contract.stage_dict(horizon, groups[str(horizon)]["card_paths"], str(universe_path),
                                         os.path.relpath(manifest_path, stage_root / f"h{horizon}"), stage)
            scoped_contract = ResearchContract.from_dict(values)
            cards = read_cards([Path(path) for path in scoped_contract.cards], horizon, contract.warmup_hours)
            history_root = root / "development" / f"h{horizon}" if stage == "internal_validation" else None
            stages.append(_execute_horizon(stage_root / f"h{horizon}", inputs, scoped_contract, cards,
                                            contract.combination, progress, history_root))
        _write_json(stage_root / "result.json", [item for item in stages if item["stage"] == stage])
    result = {"schema_version": 1, "engine": "deterministic_four_arm_ablation_v1", "run_id": contract.run_id,
              "root": str(root), "status": "ablation_complete", "plan_sha256": _sha256(root / "plan.json"),
              "plan": plan, "stages": stages, "model_called": False,
              "forward_validation_started": False, "paper_started": False,
              "historical_causality_certified": raw_manifest["historical_causality_certified"]}
    _write_json(root / "result.json", result)
    rows = [{"stage": item["stage"], "horizon_hours": item["horizon_hours"], "arm": arm["name"],
             **arm["metrics"], "stress_net_return": arm["stress_costs"]["metrics"]["net_return"]}
            for item in stages for arm in item["experiments"]]
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    _report(root, result)
    return result


def _report(root: Path, result: dict) -> None:
    policy = result["plan"]["icir"]
    lines = ["# 四组加权与正交化消融实验", "", "A：不正交+等权；B：不正交+滚动ICIR；C：对称正交+等权；D：对称正交+滚动ICIR。", "",
             "同期限四组固定相同卡池、完整因子截面、ICIR就绪时点、持仓与成本；A是原版算法在配对样本上的重新运行。",
             f"滚动ICIR窗口{policy['window_hours']}小时，至少{policy['min_periods']}个有效IC；B/D均按绝对ICIR分配权重，负ICIR同时翻转因子方向。",
             "10币池有效秩最多9，正交化用对称伪逆保留有效子空间；全部原因子列不能同时两两正交。", ""]
    for stage in ("development", "internal_validation"):
        label = "A+B历史开发" if stage == "development" else "C内部历史验证"
        lines += [f"## {label}", "", "| 期限 | 组别 | 净收益 | 最大回撤 | Sharpe | 换手 | 两倍成本净收益 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for item in result["stages"]:
            if item["stage"] != stage:
                continue
            for arm in item["experiments"]:
                metrics = arm["metrics"]
                lines.append(f"| {item['horizon_hours']}h | {arm['name']} | {metrics['net_return']:.2%} | "
                             f"{metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.3f} | {metrics['total_turnover']:.1f} | "
                             f"{arm['stress_costs']['metrics']['net_return']:.2%} |")
        lines += ["", "净收益差以百分点表示；回撤差为负表示回撤减小。", "",
                  "| 期限 | B−A 权重效果 | C−A 正交效果 | D−C 权重效果 | D−B 正交效果 | 交互 D−B−C+A |",
                  "|---|---:|---:|---:|---:|---:|"]
        for item in result["stages"]:
            if item["stage"] == stage:
                effects = item["effects"]["base"]
                deltas = " | ".join(f"{effects[name]['net_return'] * 100:+.2f}" for name in
                                    ("B-A", "C-A", "D-C", "D-B", "interaction_D-B-C+A"))
                lines.append(f"| {item['horizon_hours']}h | {deltas} |")
    lines += ["", "全部方案在行情访问前冻结，A+B和C均运行四组；不按C结果重选。C接续对应B/D的A+B滚动历史。",
              "加权效果包含动态方向翻转。差值描述这组历史账户的组合表现，不是独立因子的因果证明；A/B/C均有既往使用记录。",
              "原档缺少历史发布/接收延迟，资金费使用已披露的前一分钟标记价代理。", "",
              "[预定实验](plan.json) · [数据表](comparison.csv) · [结构化结果](result.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
