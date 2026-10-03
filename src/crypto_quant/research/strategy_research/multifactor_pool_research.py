"""Finite, predeclared whole-pool research on observed archive data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd

from crypto_quant.features.factor_inputs import INPUT_COLUMNS
from crypto_quant.research.factor_mining.contracts import identifier, require
from crypto_quant.research.progress import ProgressLog
from .multifactor_contracts import ResearchContract
from .multifactor_combination import (
    rolling_icir_combine, symmetric_orthogonalize, validate_combination,
)
from .multifactor_data import load_research_inputs
from .multifactor_factors import process_factors, read_cards
from .multifactor_historical import (
    _candidate_failures, _capture_code_version, _count_traded_bars, _sha256,
)
from .multifactor_pool import scan_idea_pool
from .multifactor_workflow import (
    _run_one, _run_stress, _snapshot_cards, _snapshot_factor_panels,
    _snapshot_inputs, _write_json,
)


SELECTION_RULE = "max_AB_net_return_then_lower_drawdown_then_sorted_card_ids_then_horizon"
AVAILABLE_FIELDS = tuple(
    field for field in INPUT_COLUMNS
    if field.startswith(("spot_", "perp_"))
    or field in {"mark_close", "funding_rate", "funding_interval_hours", "funding_24h_sum", "funding_7d_sum"}
)


@dataclass(frozen=True)
class PoolResearchContract:
    schema_version: int
    run_id: str
    idea_pool: str
    dataset_manifest: str
    universe: str
    horizons: list[int]
    warmup_hours: int
    development_start: str
    validation_start: str
    validation_end: str
    experiments_per_horizon: int
    minimum_available_factor_share: float
    prior_data_use: str
    data_processing: str
    costs: dict
    portfolio: dict
    qualification_gates: dict
    combination: dict | None = None

    @classmethod
    def from_dict(cls, value: dict) -> PoolResearchContract:
        require(isinstance(value, dict), "pool research contract must be an object")
        schema = value.get("schema_version")
        require(type(schema) is int and schema in {1, 2}, "pool research schema_version must be 1 or 2")
        fields = set(cls.__dataclass_fields__)
        if schema == 1:
            fields.remove("combination")
        require(set(value) == fields,
                "pool research contract fields differ from the schema")
        contract = cls(**value)
        if schema == 2:
            validate_combination(contract.combination)
            require(contract.minimum_available_factor_share == 1,
                    "symmetric orthogonalization requires complete factor cross-sections (share=1)")
        identifier(contract.run_id)
        require(isinstance(contract.horizons, list) and contract.horizons
                and contract.horizons == sorted(set(contract.horizons))
                and all(type(h) is int and h in {1, 4, 24} for h in contract.horizons),
                "horizons must be sorted distinct members of 1/4/24")
        require(type(contract.experiments_per_horizon) is int and contract.experiments_per_horizon > 0,
                "experiments_per_horizon must be a positive integer")
        share = contract.minimum_available_factor_share
        require(type(share) in {int, float} and math.isfinite(share) and 0 < share <= 1,
                "minimum_available_factor_share must be in (0,1]")
        for name in ("idea_pool", "dataset_manifest", "universe", "prior_data_use", "data_processing"):
            require(isinstance(getattr(contract, name), str) and getattr(contract, name).strip(), f"{name} is required")
        dates = [pd.Timestamp(getattr(contract, name)) for name in
                 ("development_start", "validation_start", "validation_end")]
        require(all(t.tzinfo is not None and t.utcoffset().total_seconds() == 0 and t == t.floor("h") for t in dates)
                and dates[0] < dates[1] < dates[2], "pool research requires ordered UTC hour splits")
        require(set(contract.qualification_gates) ==
                {"min_net_return", "max_drawdown", "min_traded_bars", "min_stress_return"},
                "qualification gates differ from the schema")
        for name, number in contract.qualification_gates.items():
            require(type(number) in {int, float} and math.isfinite(number), f"gate {name} must be finite")
        require(type(contract.qualification_gates["min_traded_bars"]) is int
                and contract.qualification_gates["min_traded_bars"] > 0
                and 0 < contract.qualification_gates["max_drawdown"] < 1, "invalid trading/drawdown gate")
        require(contract.costs["stress_multiplier"] > 1, "pool research requires a cost stress test")
        # Reuse the existing account contract's complete portfolio/cost validation.
        for horizon in contract.horizons:
            ResearchContract.from_dict(contract.stage_dict(horizon, ["shape-check-a", "shape-check-b"],
                                                            "shape-check-universe", "manifest.json", "development"))
        return contract

    def stage_dict(self, horizon: int, cards: list[str], universe: str,
                   dataset_manifest: str, stage: str) -> dict:
        development = stage == "development"
        require(stage in {"development", "internal_validation"}, "unknown research stage")
        return {
            "schema_version": 2, "run_id": f"{self.run_id}-h{horizon}-{'ab' if development else 'c'}",
            "purpose": "research", "stage": stage,
            "start": self.development_start if development else self.validation_start,
            "end": self.validation_start if development else self.validation_end,
            "warmup_hours": self.warmup_hours, "horizon_hours": horizon,
            "cards": cards, "universe": universe, "dataset_manifest": dataset_manifest,
            "prior_data_use": self.prior_data_use, "data_processing": self.data_processing,
            "costs": dict(self.costs), "portfolio": {**self.portfolio, "rebalance_hours": horizon},
        }


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def build_group_plan(catalog: dict, budget: int, combination: dict | None = None) -> dict:
    """Cover every structural family by deterministic block removal."""
    require(not any(entry["status"] == "error" for entry in catalog["entries"]),
            "idea pool contains malformed or conflicting records; inspect the catalog")
    families = sorted((family for group in catalog["groups"] for family in group["structure_families"]),
                      key=lambda family: family["family_id"])
    ids = [family["representative_id"] for family in families]
    require(len(ids) >= 2 and len(ids) == len(set(ids)), "a research horizon needs at least two distinct family representatives")
    count = min(budget, len(ids))
    control_name = "family_equal_weight" if combination is None else "family_rolling_icir"
    variants = [{"name": control_name, "kind": control_name,
                 "selected_card_ids": ids, "removed_card_ids": [], "removed_family_ids": [], "budget_charge": 0}]
    for index in range(count):
        removed = families[index::count]
        removed_ids = [family["representative_id"] for family in removed]
        remaining = [card_id for card_id in ids if card_id not in set(removed_ids)]
        require(bool(remaining), "a block removal cannot empty the pool")
        variants.append({"name": f"remove_block_{index + 1:02d}", "kind": "remove_family_block",
                         "selected_card_ids": remaining, "removed_card_ids": removed_ids,
                         "removed_family_ids": [family["family_id"] for family in removed], "budget_charge": 1})
    return {"families": families, "representative_card_ids": ids,
            "representative_paths": [family["representative_path"] for family in families],
            "variants": variants, "experiment_budget": budget, "experiments_planned": count,
            "all_families_covered_by_removal_blocks": True,
            "attribution_unit": "whole block; no member-level marginal claim",
            "weight_policy": ("equal weight among available representatives; recomputed for each fixed subset"
                              if combination is None else combination)}


def score_variants(factors, variants: list[dict], share: float):
    """Predeclared availability threshold, paired across all planned variants."""
    counts, masks, scores = {}, {}, {}
    for variant in variants:
        name, selected = variant["name"], variant["selected_card_ids"]
        values = factors.standardized[selected]
        counts[name] = values.notna().sum(axis=1)
        masks[name] = factors.eligible & (counts[name] >= math.ceil(len(selected) * share))
        scores[name] = values.mean(axis=1, skipna=True).rename("score")
    shared_mask = pd.concat(masks, axis=1).all(axis=1) & factors.eligible
    return scores, pd.DataFrame(counts), pd.DataFrame(masks), shared_mask


def _sample(factors, contract) -> dict:
    start, end = contract.bounds
    timestamps = factors.eligible.index.get_level_values("timestamp")
    period = (timestamps >= start - pd.Timedelta(hours=1)) & (timestamps < end - pd.Timedelta(hours=1))
    eligible, common = factors.eligible & period, factors.common_mask & period
    require(bool(common.any()), "no account signal rows meet the frozen factor-availability rule")
    return {"factor_panel_eligible_rows_including_warmup": int(factors.eligible.sum()),
            "factor_panel_common_rows_including_warmup": int(factors.common_mask.sum()),
            "account_signal_window_eligible_rows": int(eligible.sum()),
            "account_signal_window_common_rows": int(common.sum()),
            "account_signal_window_start": (start - pd.Timedelta(hours=1)).isoformat(),
            "account_signal_window_end_exclusive": (end - pd.Timedelta(hours=1)).isoformat(),
            "shared_signal_coverage": int(common.sum()) / int(eligible.sum())}


def _execute_stage(root: Path, stage_path: Path, manifest_path: Path, variants: list[dict],
                   share: float, gates: dict, progress: ProgressLog, *, mask_variants: list[dict],
                   combination: dict | None = None, history_root: Path | None = None) -> list[dict]:
    contract = ResearchContract.from_dict(json.loads(stage_path.read_text()))
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "contract.json", contract.as_dict())
    with progress.span("pool.load_inputs", heartbeat=True, horizon=contract.horizon_hours, stage=contract.stage):
        inputs = load_research_inputs(manifest_path, contract, stage_path.parent)
    cards = read_cards([Path(path) for path in contract.cards], contract.horizon_hours, contract.warmup_hours)
    with progress.span("pool.calculate_factors", heartbeat=True, cards=len(cards), horizon=contract.horizon_hours):
        original = process_factors(inputs.panel, cards, include_correlations=False)
        scores, counts, masks, shared = score_variants(original, mask_variants, share)
        factors = replace(original, common_mask=shared)
    if combination is not None:
        validate_combination(combination)
        require(share == 1, "symmetric orthogonalization requires complete factor cross-sections")
        combination_root = root / "combination"
        combination_root.mkdir()
        with progress.span("pool.orthogonalize_and_weight", heartbeat=True,
                           horizon=contract.horizon_hours, stage=contract.stage):
            opens = inputs.panel.values["perp_open"]
            for variant in variants:
                name, selected = variant["name"], variant["selected_card_ids"]
                transformed, diagnostics = symmetric_orthogonalize(original.standardized[selected], shared)
                history = None
                if history_root is not None:
                    previous_root = history_root / "combination" / name
                    previous = pd.read_csv(previous_root / "history_factors.csv", index_col=[0, 1],
                                           parse_dates=[0], float_precision="round_trip")
                    previous_opens = pd.read_csv(previous_root / "history_opens.csv", index_col=[0, 1],
                                                 parse_dates=[0], float_precision="round_trip")["perp_open"]
                    history = previous, previous_opens
                combined = rolling_icir_combine(transformed, diagnostics, opens,
                                               horizon_hours=contract.horizon_hours,
                                               policy=combination, history=history)
                scores[name] = combined.score
                directory = combination_root / name
                directory.mkdir()
                for field in ("orthogonalized", "cross_sections", "rank_ic", "mature_ic_count", "icir", "weights", "directions"):
                    getattr(combined, field).to_csv(directory / f"{field}.csv")
                # Save enough preceding factors to complete labels and the
                # entire rolling window across the A+B/C boundary.
                cutoff = contract.bounds[1] - pd.Timedelta(hours=combination["window_hours"] + contract.horizon_hours + 1)
                tail = combined.orthogonalized.index.get_level_values("timestamp") >= cutoff
                combined.orthogonalized.loc[tail].to_csv(directory / "history_factors.csv")
                opens.loc[tail].rename("perp_open").to_csv(directory / "history_opens.csv")
    _snapshot_cards(cards, [Path(path) for path in contract.cards], root)
    _snapshot_inputs(inputs, root)
    _snapshot_factor_panels(factors, root)
    original.common_mask.rename("all_factor_intersection").to_csv(root / "factor_panels" / "all_factor_intersection.csv")
    counts.to_csv(root / "factor_panels" / "available_factor_counts.csv")
    masks.to_csv(root / "factor_panels" / "variant_availability_masks.csv")
    score_policy = {
        "minimum_available_factor_share": share,
        "missing_rule": ("declared available-factor equal mean, never fill factor values" if combination is None
                         else "complete common cross-section; never fill missing factor values; fewer than 3 symbols gives no signal"),
        "paired_mask": "intersection of predeclared coverage-threshold masks across every planned variant",
        "mask_variant_names": [variant["name"] for variant in mask_variants],
        "correlations_calculated": False, "all_factor_intersection_used": combination is not None,
    }
    if combination is not None:
        score_policy.update({
            "combination": combination,
            "ic_label": f"perp_next_open_{contract.horizon_hours}h",
            "ic_maturity_delay_hours": contract.horizon_hours + 1,
            "ic_std_ddof": 1,
            "inactive_factor_rule": "insufficient IC history or undefined/zero-variance ICIR has zero weight",
            "empty_weight_rule": "no score; rebalance to cash",
            "negative_icir_rule": "absolute ICIR weight; negative ICIR multiplies orthogonalized factor score by -1",
            "orthogonalization": "symmetric covariance pseudoinverse square root; SVD machine-precision rank cutoff",
            "history_root": str(history_root) if history_root is not None else None,
        })
    _write_json(root / "score_policy.json", score_policy)
    sample = _sample(factors, contract)
    start, end = contract.bounds
    timestamps = masks.index.get_level_values("timestamp")
    period = (timestamps >= start - pd.Timedelta(hours=1)) & (timestamps < end - pd.Timedelta(hours=1))
    codes = {card["id"]: f"F{index}" for index, card in enumerate(cards, 1)}
    results = []
    for variant in variants:
        name, selected = variant["name"], variant["selected_card_ids"]
        directory = root / "experiments" / name
        with progress.span("pool.account", heartbeat=True, horizon=contract.horizon_hours, experiment=name, stage=contract.stage):
            experiment = _run_one(name, variant["kind"], selected, scores[name], inputs, factors, contract, directory, sample, codes)
            stress = _run_stress(experiment, scores[name], inputs, factors, contract,
                                 root / "experiments" / f"{name}_stress")
        require(stress is not None, "cost stress was not executed")
        traded = _count_traded_bars(directory / "ledger.csv")
        failures = _candidate_failures(experiment["metrics"], stress["metrics"], traded, gates)
        results.append({**variant, "horizon_hours": contract.horizon_hours, "metrics": experiment["metrics"],
                        "stress_costs": stress, "traded_bars": traded, "qualification_failures": failures,
                        "qualified": not failures, "sample": experiment["sample"], "path": str(directory),
                        "result_artifacts": experiment["artifacts"],
                        "individual_signal_eligible_rows_including_warmup": int(masks[name].sum()),
                        "individual_account_signal_rows": int((masks[name] & period).sum()),
                        "experiment_id": f"h{contract.horizon_hours}-{name}"})
    control_name = mask_variants[0]["name"]
    control = next(item for item in results if item["name"] == control_name)
    delta_name = "delta_vs_family_equal_weight" if combination is None else "delta_vs_family_rolling_icir"
    for item in results:
        item[delta_name] = {
            metric: item["metrics"][metric] - control["metrics"][metric]
            for metric in ("net_return", "max_drawdown", "total_turnover", "total_fees", "total_slippage_cost", "total_funding")
        }
        item["block_addition_delta"] = {metric: -number for metric, number in item[delta_name].items()}
        item["addition_comparison"] = ("full pool versus this block removed; same paired mask and recomputed equal weights"
                                       if combination is None else
                                       "full pool versus this block removed; same paired mask; subset orthogonalization and rolling ICIR recomputed")
        if combination is not None:
            item["combination"] = combination
            item["combination_artifacts"] = str(root / "combination" / item["name"])
            item["account_score_available_rows"] = int((scores[item["name"]].notna() & period).sum())
    _write_json(root / "result.json", {"contract": contract.as_dict(), "combination": combination, "experiments": results,
                                       "correlations_calculated": False, "model_called": False,
                                       "forward_validation_started": False})
    return results


def _selection_key(item: dict):
    return (-item["metrics"]["net_return"], abs(item["metrics"]["max_drawdown"]),
            tuple(sorted(item["selected_card_ids"])), item["horizon_hours"], item["experiment_id"])


def run_pool_research(contract_path: Path, output: Path) -> dict:
    contract_path = Path(contract_path).resolve()
    contract = PoolResearchContract.from_dict(json.loads(contract_path.read_text()))
    base = contract_path.parent
    pool, manifest_path, universe_path = (_resolve(getattr(contract, name), base)
                                         for name in ("idea_pool", "dataset_manifest", "universe"))
    require(pool.is_dir(), "idea pool directory does not exist")
    root = Path(output).resolve() / contract.run_id
    require(not root.exists(), "pool research run directory already exists; refusing overwrite")
    root.mkdir(parents=True)
    progress = ProgressLog.for_run(root)
    contract_snapshot = asdict(contract)
    if contract.schema_version == 1:
        del contract_snapshot["combination"]
    _write_json(root / "contract.json", contract_snapshot)
    version = _capture_code_version()
    for filename in ("multifactor_pool.py", "multifactor_pool_research.py", "multifactor_combination.py"):
        path = Path(__file__).with_name(filename)
        version["source_sha256"][str(path.relative_to(Path(__file__).resolve().parents[4]))] = _sha256(path)
    _write_json(root / "code_version.json", version)
    project = Path(__file__).resolve().parents[4]
    source_manifest = {}
    snapshot_root = root / "execution_source"
    for source in sorted((project / "src" / "crypto_quant").rglob("*.py")):
        relative = source.relative_to(project)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        digest = _sha256(destination)
        require(digest == _sha256(source), "execution source changed during capture")
        source_manifest[str(relative)] = digest
    for name in ("pyproject.toml", "uv.lock"):
        source = project / name
        require(source.is_file(), f"runtime dependency definition is missing: {name}")
        destination = snapshot_root / name
        destination.write_bytes(source.read_bytes())
        source_manifest[name] = _sha256(destination)
    require(all(source_manifest[relative] == digest for relative, digest in version["source_sha256"].items()),
            "source snapshot differs from captured code version")
    _write_json(root / "execution_source_manifest.json", source_manifest)
    _write_json(root / "runtime.json", {
        "python": sys.version, "platform": platform.platform(),
        "packages": {distribution.metadata["Name"]: distribution.version for distribution in metadata.distributions()},
        "captured_before_market_panel_access": True,
    })
    raw_manifest = json.loads(manifest_path.read_text())
    require(raw_manifest["complete"] is True and raw_manifest["execution_grid_complete"] is True,
            "raw dataset is not complete")
    require(set(raw_manifest["field_sources"]) ==
            {"spot_*", "perpetual_*", "mark_*", "funding_rate", "funding_mark_price"},
            "dataset fields differ from the archive-input source contract")
    _write_json(root / "dataset_manifest.json", raw_manifest)
    (root / "universe.csv").write_bytes(universe_path.read_bytes())
    (root / "cards").mkdir()
    (root / "catalogs").mkdir()
    (root / "contracts").mkdir()
    groups, catalogs, source_cards = {}, {}, {}
    for horizon in contract.horizons:
        catalog = scan_idea_pool(pool, horizon, contract.warmup_hours, AVAILABLE_FIELDS)
        _write_json(root / "catalogs" / f"h{horizon}.json", catalog)
        group = build_group_plan(catalog, contract.experiments_per_horizon, contract.combination)
        groups[str(horizon)], catalogs[str(horizon)] = group, catalog
        # Preserve every admitted member, not only the chosen representative.
        for entry in catalog["entries"]:
            if entry["status"] != "admitted":
                continue
            identifier(entry["id"])
            source = Path(entry["path"])
            payload = source.read_bytes()
            require(hashlib.sha256(payload).hexdigest() == entry["card_sha256"], "card changed during pool scan")
            destination = root / "cards" / f"{entry['id']}.json"
            if entry["id"] in source_cards:
                require(source_cards[entry["id"]]["sha256"] == entry["card_sha256"], "cross-horizon card contents conflict")
            else:
                destination.write_bytes(payload)
                source_cards[entry["id"]] = {"source_path": str(source), "snapshot_path": str(destination.relative_to(root)),
                                            "sha256": entry["card_sha256"]}
        group["snapshot_card_paths"] = [str(root / "cards" / f"{card_id}.json") for card_id in group["representative_card_ids"]]
    _write_json(root / "cards_manifest.json", source_cards)
    plan = {"schema_version": contract.schema_version, "groups": groups, "horizons": contract.horizons,
            "combination": contract.combination,
            "selection_rule": SELECTION_RULE, "frozen_before_market_panel_access": True,
            "experiments_per_horizon": contract.experiments_per_horizon,
            "new_experiments_planned": sum(group["experiments_planned"] for group in groups.values()),
            "maximum_account_runs_including_cost_stress_and_c":
                2 * sum(1 + group["experiments_planned"] for group in groups.values()) + 4,
            "minimum_available_factor_share": contract.minimum_available_factor_share,
            "qualification_gates": contract.qualification_gates, "model_calls": 0,
            "c_validation_count": 1, "c_result_feedback": False}
    _write_json(root / "plan.json", plan)
    plan_sha = _sha256(root / "plan.json")
    trials = []
    for horizon in contract.horizons:
        group = groups[str(horizon)]
        stage_path = root / "contracts" / f"h{horizon}-ab.json"
        stage = contract.stage_dict(horizon, group["snapshot_card_paths"], str(universe_path),
                                    os.path.relpath(manifest_path, stage_path.parent), "development")
        _write_json(stage_path, stage)
        trials.extend(_execute_stage(root / "development" / f"h{horizon}", stage_path, manifest_path,
                                     group["variants"], contract.minimum_available_factor_share,
                                     contract.qualification_gates, progress, mask_variants=group["variants"],
                                     combination=contract.combination))
    _write_json(root / "candidate_trials.json", trials)
    selected = min(trials, key=_selection_key)
    freeze = {"plan_sha256": plan_sha, "selection_rule": SELECTION_RULE, "selected": selected,
              "selected_before_c_market_panel_access": True, "c_outcomes_accessed": False,
              "selection_mode": "candidate" if selected["qualified"] else "diagnostic_only",
              "qualified_ab_candidates": sum(item["qualified"] for item in trials),
              "ranked_experiment_ids": [item["experiment_id"] for item in sorted(trials, key=_selection_key)]}
    _write_json(root / "candidate_freeze.json", freeze)
    freeze_sha = _sha256(root / "candidate_freeze.json")
    horizon = selected["horizon_hours"]
    group = groups[str(horizon)]
    control = group["variants"][0]
    selected_variant = next(variant for variant in group["variants"] if variant["name"] == selected["name"])
    variants = [control] if selected_variant == control else [control, selected_variant]
    stage_path = root / "contracts" / f"h{horizon}-c.json"
    _write_json(stage_path, contract.stage_dict(horizon, group["snapshot_card_paths"], str(universe_path),
                                               os.path.relpath(manifest_path, stage_path.parent), "internal_validation"))
    validation = _execute_stage(root / "validation", stage_path, manifest_path, variants,
                                contract.minimum_available_factor_share, contract.qualification_gates, progress,
                                mask_variants=group["variants"], combination=contract.combination,
                                history_root=(root / "development" / f"h{horizon}"
                                              if contract.combination is not None else None))
    c_selected = next(item for item in validation if item["name"] == selected["name"])
    qualified = selected["qualified"] and c_selected["qualified"]
    _write_json(root / "validation" / "candidate_binding.json", {
        "candidate_freeze_sha256": freeze_sha, "selected_experiment_id": selected["experiment_id"],
        "selected_card_ids": selected["selected_card_ids"], "results_returned_to_search": False,
    })
    result = {"schema_version": contract.schema_version,
              "engine": f"deterministic_pool_research_v{contract.schema_version}", "run_id": contract.run_id,
              "combination": contract.combination,
              "root": str(root), "status": "qualified_historical_candidate" if qualified else "no_candidate_passed",
              "catalogs": {str(h): f"catalogs/h{h}.json" for h in contract.horizons},
              "unique_admitted_card_count": len(source_cards), "plan_sha256": plan_sha,
              "plan": plan, "trials": trials, "candidate_freeze": freeze,
              "candidate_freeze_sha256": freeze_sha, "selected_ab": selected, "selected_c": c_selected,
              "validation": validation, "qualified": qualified, "model_called": False, "agent_used": False,
              "c_outcomes_accessed_after_freeze": True, "c_results_used_to_reselect": False,
              "forward_validation_started": False, "paper_started": False, "published": False,
              "historical_causality_certified": raw_manifest["historical_causality_certified"],
              "funding_mark": raw_manifest["funding_mark"]}
    _write_json(root / "result.json", result)
    _report(root, contract, catalogs, result)
    return result


def _report(root: Path, contract, catalogs: dict, result: dict):
    lines = ["# 全创意卡池组合研究", "", f"运行 `{contract.run_id}`：`{result['status']}`。",
             f"全池扫描 {len(next(iter(catalogs.values()))['entries'])} 份记录，"
             f"本数据可执行的不同卡片 {result['unique_admitted_card_count']} 张。", "",
             "所有卡都有逐条准入/拒绝原因；不改原卡，不用历史收益选代表。家族仅为已展开公式的窗口结构分组，不宣称经济机制等价。", "",
             "| 期限 | 可执行卡 | 结构家族代表 | 新增块实验 |", "|---|---:|---:|---:|"]
    for horizon in contract.horizons:
        group = result["plan"]["groups"][str(horizon)]
        admitted = sum(entry["status"] == "admitted" for entry in catalogs[str(horizon)]["entries"])
        lines.append(f"| {horizon}h | {admitted} | {len(group['families'])} | {group['experiments_planned']} |")
    lines += ["", "## A+B：固定块实验", "",
              "| 实验 | 因子数 | 净收益 | 最大回撤 | 两倍成本净收益 | 共同因子覆盖 |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in result["trials"]:
        lines.append(f"| {item['experiment_id']} | {len(item['selected_card_ids'])} | {item['metrics']['net_return']:.2%} | "
                     f"{item['metrics']['max_drawdown']:.2%} | {item['stress_costs']['metrics']['net_return']:.2%} | "
                     f"{item['sample']['shared_signal_coverage']:.2%} |")
    combination_description = ("剩余因子按预定规则重新等权。" if contract.combination is None else
                               "每个子集在相同共同截面上重新对称正交化，再计算自己的滚动ICIR与权重。")
    score_description = (
        f"每行至少 {contract.minimum_available_factor_share:.0%} 的子集因子可用才生成信号；可用因子等权均值，不填缺值。"
        if contract.combination is None else
        f"滚动窗口 {contract.combination['window_hours']} 小时，最少 {contract.combination['min_periods']} 个有效IC；"
        "ICIR为已成熟Rank IC的均值除以样本标准差，绝对ICIR归一化为正权重，负ICIR同时将正交因子分数乘−1。"
        "全部因子共同有效，截面少于3币不计算；SVD伪逆仅保留有效秩，秩不足时协方差是投影矩阵。"
        "无可用权重时空仓；IC方差为零的因子不分配权重。C接续A+B滚动历史与待成熟标签。"
        "每个信号只使用截至其bar-open已成熟的标签，成熟延迟为预测期限加1小时。"
        "共同因子覆盖与实际有分数的小时数分别记录。"
    )
    lines += ["", "每个结构家族进入一个移除块；块增量等于全池与该块移除方案的配对差值，不拆成成员的单独贡献。"
              + combination_description + "正的相对改善也不等于策略通过门槛。", "",
              score_description + "每个期限的所有预定方案共享覆盖掩码与完整账户时间网格，另存可用因子数及各方案覆盖。"
              "1h/4h/24h分别以1/4/24小时调仓。全池两两Spearman诊断未计算。", "",
              "## 冻结候选及C内部验证", "",
              f"选择 `{result['selected_ab']['experiment_id']}`，冻结在C行情面板和账户结果计算之前。",
              "| 阶段 | 净收益 | 最大回撤 | 两倍成本净收益 |", "|---|---:|---:|---:|"]
    for stage, item in (("A+B", result["selected_ab"]), ("C内部验证", result["selected_c"])):
        lines.append(f"| {stage} | {item['metrics']['net_return']:.2%} | {item['metrics']['max_drawdown']:.2%} | {item['stress_costs']['metrics']['net_return']:.2%} |")
    lines += ["", "C是既往已暴露的内部验证，不用于重选；完整来源签名和固定币池元数据预检覆盖C。"
              "资金费用事件前已完成的原始分钟标记价估算，未认证历史发布/接收延迟。", "",
              "本轮是全池确定性入口，不调用模型；未启动前向账户、Paper或发布。", "",
              "[预定预算与家族映射](plan.json) · [逐条卡片清单](catalogs/) · [结构化结果](result.json) · [候选冻结](candidate_freeze.json)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
