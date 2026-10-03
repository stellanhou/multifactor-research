import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.progress import ProgressLog
from crypto_quant.research.strategy_research import multifactor_pool_research as pool_research
from crypto_quant.research.strategy_research.multifactor_pool import scan_idea_pool


def _write_card(path: Path, card_id: str, expression: str, *, horizons=(1, 4),
                extra: dict | None = None) -> dict:
    compiled = compile_expression(expression)
    card = {
        "id": card_id,
        "title": card_id,
        "source_type": "factor_mining",
        "status": "research_idea",
        "source": {"run_id": "pool-runner-test", "candidate_id": card_id},
        "original_claim": {
            "direction": 1,
            "formula": {
                "expression": expression,
                "expanded_expression": compiled.expanded_expression,
                "fields": list(compiled.fields),
                "lookback_hours": compiled.lookback_hours,
            },
        },
        "market_and_horizon": {
            "venue": "Binance",
            "market": "USD-M perpetual",
            "inputs": "1h",
            "passed_horizons": list(horizons),
        },
        "b_validation_status": "passed",
        "admission_evidence": {"eligible_for_idea_pool": True},
    }
    if extra:
        card.update(extra)
    path.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    return card


def _pool_cards(directory: Path, *, horizons=(1, 4)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    cards = [
        ("a-close", "perp_close", {"prior_research_result": {"net_return": -0.8}}),
        ("a-mean-low", "ts_mean(perp_close, 2)", {"prior_research_result": {"net_return": -1.0}}),
        ("z-mean-high", "ts_mean(perp_close, 2)", {"prior_research_result": {"net_return": 99.0}}),
        ("b-return", "ts_return(perp_close, 1)", {}),
        ("c-volatility", "ts_std(perp_close, 2)", {}),
    ]
    paths = []
    for card_id, expression, extra in cards:
        path = directory / f"{card_id}.json"
        _write_card(path, card_id, expression, horizons=horizons, extra=extra)
        paths.append(path)
    return paths


def _pool_contract(*, horizons=(1, 4), run_id="pool_runner_test") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "idea_pool": "pool",
        "dataset_manifest": "manifest.json",
        "universe": "universe.csv",
        "horizons": list(horizons),
        "warmup_hours": 2,
        "development_start": "2024-01-01T00:00:00Z",
        "validation_start": "2024-01-02T00:00:00Z",
        "validation_end": "2024-01-03T00:00:00Z",
        "experiments_per_horizon": 2,
        "minimum_available_factor_share": 0.8,
        "prior_data_use": "synthetic test fixture",
        "data_processing": "unmodified synthetic observations",
        "costs": {
            "initial_capital": 10000.0,
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "stress_multiplier": 2.0,
        },
        "portfolio": {
            "long_count": 1,
            "short_count": 1,
            "gross_exposure": 0.4,
            "max_asset_weight": 0.2,
            "rebalance_hours": 1,
            "margin_fraction": 0.1,
        },
        "qualification_gates": {
            "min_net_return": -0.5,
            "max_drawdown": 0.9,
            "min_traded_bars": 1,
            "min_stress_return": -0.5,
        },
    }


def test_build_group_plan_covers_family_blocks_within_budget_and_ignores_returns(tmp_path):
    pool = tmp_path / "cards"
    _pool_cards(pool)

    assert len(pool_research.AVAILABLE_FIELDS) == 23
    catalog = scan_idea_pool(pool, 1, 168, pool_research.AVAILABLE_FIELDS)
    assert catalog["counts"]["admitted"] == 5
    plan = pool_research.build_group_plan(catalog, budget=2)

    assert plan["experiment_budget"] == 2
    assert plan["experiments_planned"] == 2
    assert len(plan["variants"]) == 3  # baseline plus two budgeted removals
    families = {family["family_id"] for family in plan["families"]}
    removed_once = [family_id for variant in plan["variants"][1:]
                    for family_id in variant["removed_family_ids"]]
    assert set(removed_once) == families
    assert len(removed_once) == len(families)
    representatives = set(plan["representative_card_ids"])
    assert set(plan["variants"][0]["selected_card_ids"]) == representatives
    for variant in plan["variants"][1:]:
        assert set(variant["selected_card_ids"]) == representatives - set(variant["removed_card_ids"])
        assert variant["budget_charge"] == 1

    mean_family = next(family for family in plan["families"] if "a-mean-low" in family["member_ids"])
    assert mean_family["representative_id"] == "a-mean-low"
    assert "z-mean-high" in mean_family["member_ids"]
    larger_budget = pool_research.build_group_plan(catalog, budget=99)
    assert larger_budget["experiments_planned"] == len(families)


def test_global_candidate_ranking_includes_each_horizon_baseline():
    metrics = lambda net, drawdown: {
        "net_return": net,
        "max_drawdown": drawdown,
        "total_turnover": 1.0,
        "total_fees": 2.0,
        "total_slippage_cost": 3.0,
        "total_funding": 4.0,
    }
    trials = [
        {"experiment_id": "h1-family_equal_weight", "horizon_hours": 1,
         "selected_card_ids": ["a", "b"], "metrics": metrics(0.1, 0.2)},
        {"experiment_id": "h4-remove_block_01", "horizon_hours": 4,
         "selected_card_ids": ["a"], "metrics": metrics(0.3, 0.4)},
        {"experiment_id": "h1-remove_block_01", "horizon_hours": 1,
         "selected_card_ids": ["b"], "metrics": metrics(0.2, 0.1)},
        {"experiment_id": "h4-family_equal_weight", "horizon_hours": 4,
         "selected_card_ids": ["a", "b"], "metrics": metrics(0.25, 0.3)},
    ]

    ranked = sorted(trials, key=pool_research._selection_key)

    assert [trial["experiment_id"] for trial in ranked] == [
        "h4-remove_block_01", "h4-family_equal_weight",
        "h1-remove_block_01", "h1-family_equal_weight",
    ]
    assert {trial["experiment_id"] for trial in ranked
            if trial["experiment_id"].endswith("family_equal_weight")} == {
        "h1-family_equal_weight", "h4-family_equal_weight",
    }


def test_score_variants_uses_available_factor_mean_and_intersects_all_planned_masks():
    times = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="h")
    index = pd.MultiIndex.from_arrays(
        [times, ["BTCUSDT"] * len(times)], names=["timestamp", "symbol"],
    )
    standardized = pd.DataFrame(
        [
            [1.0, 3.0, 5.0, np.nan, 7.0],
            [2.0, 4.0, 6.0, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, np.nan],
        ], index=index, columns=["f1", "f2", "f3", "f4", "f5"],
    )
    eligible = pd.Series([True, True, False, True], index=index, name="eligible")
    factors = SimpleNamespace(standardized=standardized, eligible=eligible)
    variants = [
        {"name": "full", "selected_card_ids": ["f1", "f2", "f3", "f4", "f5"]},
        {"name": "small", "selected_card_ids": ["f1", "f2", "f3"]},
        {"name": "medium", "selected_card_ids": ["f1", "f2", "f3", "f4"]},
    ]

    scores, counts, masks, shared = pool_research.score_variants(factors, variants, 0.8)

    assert counts.loc[index[0], "full"] == 4
    assert scores["full"].loc[index[0]] == pytest.approx(4.0)
    assert masks.loc[index[0], "full"]
    assert not masks.loc[index[0], "medium"]
    assert masks.loc[index[1], "small"]
    assert not masks.loc[index[1], "full"]
    assert not shared.loc[index[0]]  # every predeclared variant must meet its threshold
    assert not shared.loc[index[1]]
    assert not shared.loc[index[2]]  # ineligible rows stay excluded
    assert shared.loc[index[3]]


def _small_inputs(start: pd.Timestamp, end: pd.Timestamp):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    panel_times = pd.date_range(start - pd.Timedelta(hours=2), end - pd.Timedelta(hours=1), freq="h")
    index = pd.MultiIndex.from_product([panel_times, symbols], names=["timestamp", "symbol"]).sort_values()
    values = pd.DataFrame(index=index, columns=["perp_close", "perp_open"], dtype=float)
    frames = {}
    market_times = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1), freq="h")
    base = [100.0, 120.0, 90.0, 140.0]
    drift = [0.20, -0.15, 0.08, -0.05]
    curve = [0.010, -0.006, 0.014, -0.012]
    for symbol_index, symbol in enumerate(symbols):
        prices = []
        for step, timestamp in enumerate(panel_times):
            price = base[symbol_index] + drift[symbol_index] * step + curve[symbol_index] * step**2
            values.loc[(timestamp, symbol), "perp_close"] = price
            values.loc[(timestamp, symbol), "perp_open"] = price * (1.0 + 0.0002)
            if timestamp in market_times:
                prices.append(price)
        close = np.asarray(prices, dtype=float)
        frames[symbol] = pd.DataFrame(
            {"open": close * (1.0 + 0.0002), "close": close, "mark_close": close},
            index=market_times,
        )
    universe = pd.Series(True, index=index, name="eligible", dtype=bool)
    panel = FactorInputPanel(values, universe, {"panel_source": "synthetic test fixture"})
    inputs = SimpleNamespace(
        panel=panel,
        frames=frames,
        funding=pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"]),
        universe=universe,
        diagnostics={"historical_causality_certified": False, "fixture": True},
    )
    return inputs


def test_execute_stage_smoke_uses_factor_panels_and_costed_account(tmp_path, monkeypatch):
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    end = pd.Timestamp("2024-01-01T09:00:00Z")
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    card_paths = [
        cards_dir / "close.json",
        cards_dir / "return.json",
        cards_dir / "volatility.json",
    ]
    _write_card(card_paths[0], "close", "perp_close", horizons=(1,))
    _write_card(card_paths[1], "return", "ts_return(perp_close, 1)", horizons=(1,))
    _write_card(card_paths[2], "volatility", "ts_std(perp_close, 2)", horizons=(1,))
    contract_values = _pool_contract(horizons=(1,))
    contract_values["development_start"] = start.isoformat()
    contract_values["validation_start"] = end.isoformat()
    contract_values["validation_end"] = (end + pd.Timedelta(hours=8)).isoformat()
    contract = pool_research.PoolResearchContract.from_dict(contract_values)
    stage_path = tmp_path / "contracts" / "h1-ab.json"
    stage_path.parent.mkdir()
    stage_path.write_text(json.dumps(contract.stage_dict(
        1, [str(path) for path in card_paths], "universe.csv", "manifest.json", "development",
    )), encoding="utf-8")
    inputs = _small_inputs(start, end)
    monkeypatch.setattr(pool_research, "load_research_inputs", lambda *args: inputs)
    pool = scan_idea_pool(tmp_path / "cards", 1, contract.warmup_hours, pool_research.AVAILABLE_FIELDS)
    group = pool_research.build_group_plan(pool, budget=2)

    results = pool_research._execute_stage(
        tmp_path / "run" / "development" / "h1", stage_path, tmp_path / "manifest.json",
        group["variants"], contract.minimum_available_factor_share,
        contract.qualification_gates, ProgressLog.for_run(tmp_path / "run"),
        mask_variants=group["variants"],
    )

    assert len(results) == 3
    assert all(item["traded_bars"] > 0 for item in results)
    assert all(item["metrics"]["total_fees"] > 0 for item in results)
    assert all(item["metrics"]["total_slippage_cost"] > 0 for item in results)
    assert all(item["stress_costs"]["metrics"]["net_return"] != item["metrics"]["net_return"]
               for item in results)
    run_root = tmp_path / "run" / "development" / "h1"
    score_policy = json.loads((run_root / "score_policy.json").read_text(encoding="utf-8"))
    assert score_policy["mask_variant_names"] == [variant["name"] for variant in group["variants"]]
    assert (run_root / "experiments" / "family_equal_weight" / "ledger.csv").is_file()


@pytest.mark.parametrize("use_icir", [False, True])
def test_pool_runner_freezes_global_ab_ranking_then_runs_one_c_without_reselection(tmp_path, monkeypatch, use_icir):
    pool = tmp_path / "pool"
    _pool_cards(pool)
    contract_path = tmp_path / "pool-contract.json"
    contract_value = _pool_contract()
    control_name = "family_rolling_icir" if use_icir else "family_equal_weight"
    if use_icir:
        contract_value.update(schema_version=2, minimum_available_factor_share=1.0,
                              combination={"method": "symmetric_orthogonalized_rolling_icir", "window_hours": 720,
                                           "min_periods": 168, "negative_icir": "flip_factor"})
    contract_path.write_text(json.dumps(contract_value), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "complete": True,
        "execution_grid_complete": True,
        "field_sources": ["spot_*", "perpetual_*", "mark_*", "funding_rate", "funding_mark_price"],
        "historical_causality_certified": False,
        "funding_mark": {"source": "test fixture"},
    }), encoding="utf-8")
    (tmp_path / "universe.csv").write_text("symbol\nBTCUSDT\n", encoding="utf-8")
    calls = []

    def fake_execute(root, stage_path, manifest_path, variants, share, gates, progress, *, mask_variants,
                     combination=None, history_root=None):
        root.mkdir(parents=True, exist_ok=False)
        stage_contract = json.loads(stage_path.read_text(encoding="utf-8"))
        stage = stage_contract["stage"]
        horizon = stage_contract["horizon_hours"]
        run_root = stage_path.parent.parent
        if use_icir:
            assert combination == contract_value["combination"]
            assert share == 1
            assert history_root == (run_root / "development" / f"h{horizon}"
                                    if stage == "internal_validation" else None)
        if stage == "internal_validation":
            freeze = json.loads((run_root / "candidate_freeze.json").read_text(encoding="utf-8"))
            assert freeze["selected_before_c_market_panel_access"] is True
            assert freeze["c_outcomes_accessed"] is False
        calls.append({
            "stage": stage,
            "horizon": horizon,
            "variants": json.loads(json.dumps(variants)),
            "mask_variants": json.loads(json.dumps(mask_variants)),
        })
        results = []
        if stage == "development":
            returns = {
                1: {control_name: 0.10, "remove_block_01": 0.20, "remove_block_02": 0.15},
                4: {control_name: 0.30, "remove_block_01": 0.40, "remove_block_02": 0.35},
            }[horizon]
        else:
            # The frozen A+B winner fails C. This result must not send the search back to A+B.
            returns = {control_name: 0.05, "remove_block_01": -0.10}
        for variant in variants:
            net = returns[variant["name"]]
            metrics = {
                "net_return": net,
                "max_drawdown": 0.2,
                "total_turnover": 1.0,
                "total_fees": 2.0,
                "total_slippage_cost": 3.0,
                "total_funding": 4.0,
            }
            qualified = stage == "development" or variant["name"] == control_name
            results.append({
                **variant,
                "horizon_hours": horizon,
                "metrics": metrics,
                "stress_costs": {"metrics": {**metrics, "net_return": net - 0.02}},
                "traded_bars": 10,
                "qualification_failures": [] if qualified else ["min_net_return"],
                "qualified": qualified,
                "sample": {"shared_signal_coverage": 0.9},
                "path": str(root / "experiments" / variant["name"]),
                "result_artifacts": {},
                "individual_signal_eligible_rows": 100,
                "experiment_id": f"h{horizon}-{variant['name']}",
            })
        return results

    monkeypatch.setattr(pool_research, "_execute_stage", fake_execute)
    monkeypatch.setattr(pool_research, "_capture_code_version", lambda: {"source_sha256": {}})

    result = pool_research.run_pool_research(contract_path, tmp_path / "runs")

    assert [(call["stage"], call["horizon"]) for call in calls] == [
        ("development", 1), ("development", 4), ("internal_validation", 4),
    ]
    assert len(result["trials"]) == 6
    ranked = result["candidate_freeze"]["ranked_experiment_ids"]
    assert ranked == [
        "h4-remove_block_01", "h4-remove_block_02", f"h4-{control_name}",
        "h1-remove_block_01", "h1-remove_block_02", f"h1-{control_name}",
    ]
    assert result["selected_ab"]["experiment_id"] == "h4-remove_block_01"
    c_call = calls[-1]
    assert [variant["name"] for variant in c_call["variants"]] == [
        control_name, "remove_block_01",
    ]
    assert c_call["mask_variants"] == calls[1]["mask_variants"]
    assert result["selected_c"]["experiment_id"] == "h4-remove_block_01"
    assert result["selected_c"]["qualified"] is False
    assert result["qualified"] is False
    assert result["c_results_used_to_reselect"] is False
    assert result["forward_validation_started"] is False
    assert result["paper_started"] is False
    assert result["engine"] == ("deterministic_pool_research_v2" if use_icir else "deterministic_pool_research_v1")
    if use_icir:
        assert result["plan"]["combination"] == contract_value["combination"]

    run_root = Path(result["root"])
    result_bytes = (run_root / "result.json").read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        pool_research.run_pool_research(contract_path, tmp_path / "runs")
    assert len(calls) == 3
    assert (run_root / "result.json").read_bytes() == result_bytes


def _icir_contract():
    value = _pool_contract(horizons=(1,))
    value.update(schema_version=2, minimum_available_factor_share=1.0)
    value["combination"] = {
        "method": "symmetric_orthogonalized_rolling_icir", "window_hours": 8,
        "min_periods": 3, "negative_icir": "flip_factor",
    }
    return value


def test_pool_v2_contract_requires_explicit_combination_and_complete_cross_sections():
    value = _icir_contract()
    contract = pool_research.PoolResearchContract.from_dict(value)
    assert contract.combination == value["combination"]
    for change in ({"minimum_available_factor_share": 0.8}, {"combination": None},
                   {"schema_version": 1}, {"schema_version": 3}):
        with pytest.raises(ValueError):
            pool_research.PoolResearchContract.from_dict({**value, **change})
    value.pop("combination")
    with pytest.raises(ValueError, match="fields differ"):
        pool_research.PoolResearchContract.from_dict(value)


def test_pool_v2_stage_runs_icir_account_and_reuses_persisted_ab_history(tmp_path, monkeypatch):
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    split = start + pd.Timedelta(hours=24)
    end = split + pd.Timedelta(hours=24)
    value = _icir_contract()
    value.update(development_start=start.isoformat(), validation_start=split.isoformat(),
                 validation_end=end.isoformat())
    contract = pool_research.PoolResearchContract.from_dict(value)
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    paths = [cards_dir / "close.json", cards_dir / "return.json", cards_dir / "volatility.json"]
    for path, expression in zip(paths, ["perp_close", "ts_return(perp_close, 1)", "ts_std(perp_close, 2)"]):
        _write_card(path, path.stem, expression, horizons=(1,))
    catalog = scan_idea_pool(cards_dir, 1, 2, pool_research.AVAILABLE_FIELDS)
    group = pool_research.build_group_plan(catalog, 2, contract.combination)
    assert group["variants"][0]["name"] == "family_rolling_icir"
    inputs = _small_inputs(start, end)
    rng = np.random.default_rng(274)
    panel_times = inputs.panel.values.index.get_level_values("timestamp").unique()
    symbols = sorted(inputs.frames)
    prices = 100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=(len(panel_times), len(symbols))), axis=0))
    inputs.panel.values["perp_close"] = prices.reshape(-1)
    inputs.panel.values["perp_open"] = (prices * 1.0002).reshape(-1)
    for column, symbol in enumerate(symbols):
        frame = inputs.frames[symbol]
        frame["close"] = pd.Series(prices[:, column], index=panel_times).reindex(frame.index)
        frame["open"] = frame.close * 1.0002
        frame["mark_close"] = frame.close

    def load_inputs(manifest, stage_contract, base):
        first, last = stage_contract.input_start, stage_contract.bounds[1]
        times = inputs.panel.values.index.get_level_values("timestamp")
        scoped = (times >= first) & (times < last)
        universe = inputs.universe.loc[scoped]
        panel = FactorInputPanel(inputs.panel.values.loc[scoped], universe, inputs.panel.diagnostics)
        frames = {symbol: frame.loc[stage_contract.bounds[0] - pd.Timedelta(hours=1):last - pd.Timedelta(hours=1)]
                  for symbol, frame in inputs.frames.items()}
        return SimpleNamespace(panel=panel, frames=frames, funding=inputs.funding, universe=universe,
                               diagnostics=inputs.diagnostics)

    monkeypatch.setattr(pool_research, "load_research_inputs", load_inputs)
    stage_directory = tmp_path / "contracts"
    stage_directory.mkdir()
    development_root = tmp_path / "ab"
    progress = ProgressLog.for_run(tmp_path / "run")
    for stage, root, history in (("development", development_root, None),
                                  ("internal_validation", tmp_path / "c", development_root)):
        stage_path = stage_directory / f"{stage}.json"
        stage_path.write_text(json.dumps(contract.stage_dict(1, [str(path) for path in paths],
                                                            "universe.csv", "manifest.json", stage)))
        results = pool_research._execute_stage(root, stage_path, tmp_path / "manifest.json",
                                              group["variants"], 1, contract.qualification_gates, progress,
                                              mask_variants=group["variants"], combination=contract.combination,
                                              history_root=history)
        assert all(item["traded_bars"] > 0 for item in results)
        assert all(item["metrics"]["total_fees"] > 0 for item in results)
        assert all("delta_vs_family_rolling_icir" in item for item in results)
        assert all(item["account_score_available_rows"] > 0 for item in results)
        policy = json.loads((root / "score_policy.json").read_text())
        assert policy["ic_maturity_delay_hours"] == 2
        assert policy["all_factor_intersection_used"] is True
        assert policy["history_root"] == (str(history) if history is not None else None)
        for variant in group["variants"]:
            directory = root / "combination" / variant["name"]
            assert (directory / "directions.csv").is_file()
            assert (directory / "history_opens.csv").is_file()
            weights = pd.read_csv(directory / "weights.csv", index_col=0, parse_dates=[0])
            assert (weights >= 0).all().all()
            if history is not None:
                assert weights.loc[split - pd.Timedelta(hours=1)].sum() == pytest.approx(1)
