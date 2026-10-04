import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research import multifactor_selection_research as selection_research
from crypto_quant.research.strategy_research.multifactor_contracts import ResearchContract
from crypto_quant.research.strategy_research.multifactor_pool_research import PoolResearchContract
from test_multifactor_pool_research import _icir_contract


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HOUR = pd.Timedelta(hours=1)


def test_inner_history_rejects_missing_symbols_instead_of_creating_cash(tmp_path):
    directory = tmp_path / "development/h1/equal/R0/base"
    directory.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    index = pd.date_range(start - HOUR, periods=2, freq="h")
    pd.DataFrame({"BTCUSDT": [0.2, 0.2]}, index=index).to_csv(directory / "targets.csv")
    contract = SimpleNamespace(bounds=(start, start + 2 * HOUR))
    with pytest.raises(ValueError, match="target symbols differ"):
        selection_research._read_inner_history(
            tmp_path, 1, "equal", "R0", "development", "base", contract, SYMBOLS,
        )


def _write_source(root: Path) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    warmup = 1440
    development_hours = 180
    validation_hours = 12
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    development_start = base + (warmup + 1) * HOUR
    validation_start = development_start + development_hours * HOUR
    validation_end = validation_start + validation_hours * HOUR

    pool_value = _icir_contract()
    pool_value.update(
        horizons=[1, 4, 24],
        warmup_hours=warmup,
        development_start=development_start.isoformat(),
        validation_start=validation_start.isoformat(),
        validation_end=validation_end.isoformat(),
    )
    pool_value["portfolio"].update(
        long_count=2,
        short_count=2,
        gross_exposure=0.8,
        max_asset_weight=0.2,
        margin_fraction=0.1,
    )
    (root / "pool_contract.json").write_text(json.dumps(pool_value), encoding="utf-8")
    (root / "plan.json").write_text(json.dumps({
        "horizons": [1, 4, 24],
        "stages": list(selection_research.STAGES),
        "engineering_smoke_only": True,
    }), encoding="utf-8")
    (root / "engineering_provenance.json").write_text(json.dumps({
        "fixture": "synthetic integration test",
        "historical_data": False,
    }), encoding="utf-8")
    (root / "dataset_manifest.json").write_text(json.dumps({
        "historical_causality_certified": False,
        "fixture": "synthetic selection integration inputs",
    }), encoding="utf-8")

    model_start = development_start - (warmup + 1) * HOUR
    all_times = pd.date_range(model_start, validation_end - HOUR, freq="h", tz="UTC")
    rng = np.random.default_rng(824)
    values = rng.normal(size=(len(all_times), len(SYMBOLS), 3))
    values -= values.mean(axis=1, keepdims=True)
    values /= values.std(axis=1, keepdims=True)
    index = pd.MultiIndex.from_product(
        [all_times, SYMBOLS], names=["timestamp", "symbol"],
    )
    factors = pd.DataFrame(
        values.reshape(-1, 3), index=index,
        columns=["signal_a", "signal_b", "noise"],
    )
    opens = pd.Series(100.0, index=index, name="perp_open")
    pool = PoolResearchContract.from_dict(pool_value)

    for stage, start, end in (
        ("development", development_start, validation_start),
        ("internal_validation", validation_start, validation_end),
    ):
        stage_root = root / stage
        shared_end = end - HOUR
        shared_times = all_times[all_times <= shared_end]
        factor_index = pd.MultiIndex.from_product(
            [shared_times, SYMBOLS], names=["timestamp", "symbol"],
        )
        factor_panel = factors.loc[factor_index]
        open_panel = opens.loc[factor_index].to_frame()
        # The same continuous model history is copied into each horizon folder.
        for horizon in (1, 4, 24):
            horizon_root = stage_root / f"h{horizon}"
            (horizon_root / "factor_panels").mkdir(parents=True, exist_ok=True)
            contract = ResearchContract.from_dict(pool.stage_dict(
                horizon, ["card-a.json", "card-b.json"], "universe.csv",
                "manifest.json", stage,
            ))
            (horizon_root / "contract.json").write_text(
                json.dumps(contract.as_dict()), encoding="utf-8",
            )
            factor_panel.to_csv(horizon_root / "factor_panels" / "standardized.csv")

        inputs = stage_root / "inputs"
        inputs.mkdir(parents=True)
        open_panel.to_csv(inputs / "panel.values.csv")
        market_dir = inputs / "market"
        market_dir.mkdir()
        market_times = pd.date_range(start - HOUR, end - HOUR, freq="h", tz="UTC")
        for symbol in SYMBOLS:
            pd.DataFrame(
                {"open": 100.0, "close": 100.0, "mark_close": 100.0},
                index=market_times,
            ).to_csv(market_dir / f"{symbol}.csv")
        pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"]).to_csv(
            inputs / "funding.csv", index=False,
        )
    return development_start, validation_start, validation_end


def _request(source: Path, run_id="selection_integration_test") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_run": str(source),
        "methods": list(selection_research.METHODS),
        "rebalance_routes": list(selection_research.REBALANCE_ROUTES),
        "selection_policy": dict(selection_research.SELECTION_POLICY),
        "second_layer": {
            "continuous_target_weight_buffer": 0.02,
            "policies": list(selection_research.OUTER_POLICIES),
        },
    }


def test_overlap_audit_separates_value_differences_from_availability_mismatches():
    index = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="h")
    left = pd.DataFrame({"factor": [1.0, 2.0, np.nan]}, index=index)
    right = pd.DataFrame({"factor": [1.1, np.nan, 3.0]}, index=index)

    audit = selection_research._compare_overlap(
        left, right, label="test factors", exact=False,
    )

    assert audit["overlap_rows"] == 3
    assert audit["availability_mismatches"] == 2
    assert audit["different_values"] == 1
    assert audit["max_absolute_difference"] == pytest.approx(0.1)


def test_full_selection_research_freezes_runs_nets_returns_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, validation_start, _ = _write_source(source)
    contract_path = tmp_path / "selection_contract.json"
    contract_path.write_text(json.dumps(_request(source)), encoding="utf-8")
    output = tmp_path / "runs"
    run_root = output / "selection_integration_test"

    calls = {"models": 0, "inner_accounts": 0, "outer_accounts": 0}
    original_generate = selection_research.generate_selection_scores
    original_inner = selection_research._run_arm
    original_outer = selection_research.run_perpetual_account

    def frozen_plan_exists():
        plan_path = run_root / "plan.json"
        assert plan_path.is_file()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan["frozen_before_account_outcomes"] is True
        return plan

    def observe_model(*args, **kwargs):
        frozen_plan_exists()
        calls["models"] += 1
        return original_generate(*args, **kwargs)

    def observe_inner(*args, **kwargs):
        frozen_plan_exists()
        calls["inner_accounts"] += 1
        return original_inner(*args, **kwargs)

    def observe_outer(*args, **kwargs):
        frozen_plan_exists()
        calls["outer_accounts"] += 1
        return original_outer(*args, **kwargs)

    monkeypatch.setattr(selection_research, "generate_selection_scores", observe_model)
    monkeypatch.setattr(selection_research, "_run_arm", observe_inner)
    monkeypatch.setattr(selection_research, "run_perpetual_account", observe_outer)

    result = selection_research.run_selection_research(contract_path, output)
    assert calls == {"models": 3, "inner_accounts": 144, "outer_accounts": 96}
    assert result["account_runs"] == 240
    assert result["first_layer_account_runs"] == 144
    assert result["second_layer_account_runs"] == 96

    plan = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    assert plan["account_runs"] == 240
    assert plan["first_layer"]["account_runs"] == 144
    assert plan["second_layer"]["account_runs"] == 96
    assert json.loads((run_root / "plan_manifest.json").read_text(encoding="utf-8"))["frozen_before_account_outcomes"]

    score_panel = pd.read_csv(run_root / "models/h1/scores.csv", index_col=[0, 1])
    assert (score_panel["equal"].notna() & score_panel["pool"].isna()).any()
    fits = json.loads((run_root / "models/h1/fits.json").read_text(encoding="utf-8"))
    assert any(fit.get("route") == "pool" and fit.get("status") == "cash_no_positive_marginal_gain"
               for fit in fits)

    overlap = json.loads((run_root / "models/h1/overlap_audit.json").read_text(encoding="utf-8"))
    assert overlap["factor_overlap"]["overlap_rows"] > 0
    assert overlap["factor_overlap"]["different_values"] == 0
    assert overlap["open_overlap"]["different_values"] == 0
    input_overlap = json.loads((run_root / "input_overlap_audit.json").read_text(encoding="utf-8"))
    assert all(item["overlap_rows"] == 1 and item["exact_match"]
               for item in input_overlap["market_and_funding"]["market_overlap"].values())

    first_c_signal = validation_start - HOUR
    fixed_fits = pd.read_csv(
        run_root / "horizon_combination/equal/R0/base/fixed/fits.csv",
        parse_dates=["timestamp", "latest_return_timestamp"],
    )
    first_c_fit = fixed_fits.loc[fixed_fits.timestamp == first_c_signal].iloc[0]
    source_ledger = pd.read_csv(
        run_root / "development/h1/equal/R0/base/ledger.csv",
        index_col=0, parse_dates=[0],
    )
    source_ledger.index = pd.to_datetime(source_ledger.index, utc=True)
    available_times = source_ledger.index[source_ledger.index < first_c_signal]
    assert len(available_times) >= 168
    assert first_c_fit.latest_return_timestamp == available_times.max()
    assert first_c_fit.latest_return_timestamp < first_c_signal
    assert first_c_fit.prior_complete_observations == len(available_times)

    risk_budgets = pd.read_csv(
        run_root / "horizon_combination/equal/R0/base/gated_risk_budget/budgets.csv",
        index_col=0, parse_dates=[0],
    )
    assert risk_budgets.loc[first_c_signal].sum() == 0.0

    manifest = json.loads((run_root / "source_manifest.json").read_text(encoding="utf-8"))
    assert str((source / "engineering_provenance.json").resolve()) in manifest
    for filename, digest in manifest.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    result_bytes = (run_root / "result.json").read_bytes()
    with pytest.raises(ValueError, match="refusing overwrite"):
        selection_research.run_selection_research(contract_path, output)
    assert (run_root / "result.json").read_bytes() == result_bytes


def test_stepwise_contract_keeps_the_original_preset_and_freezes_its_larger_budget(tmp_path):
    value = _request(tmp_path)
    assert selection_research.SelectionResearchContract.from_dict(value).methods == list(selection_research.METHODS)
    value["methods"] = ["stepwise"]
    with pytest.raises(ValueError, match="model parameters differ"):
        selection_research.SelectionResearchContract.from_dict(value)
    value["selection_policy"] = dict(selection_research.STEPWISE_POLICY)
    assert selection_research.SelectionResearchContract.from_dict(value).selection_policy["pool_search_budget"] == 2048
    value["methods"] = ["stepwise_net_sharpe"]
    assert selection_research.SelectionResearchContract.from_dict(value).methods == ["stepwise_net_sharpe"]
    value["selection_policy"]["pool_capacity"] = 9
    with pytest.raises(ValueError, match="model parameters differ"):
        selection_research.SelectionResearchContract.from_dict(value)


def test_stepwise_supplement_runs_only_its_forty_accounts_and_keeps_cash_readiness(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_source(source)
    value = _request(source, "stepwise_integration_test")
    value["methods"] = ["stepwise"]
    value["selection_policy"] = dict(selection_research.STEPWISE_POLICY)
    contract_path = tmp_path / "stepwise_contract.json"
    contract_path.write_text(json.dumps(value), encoding="utf-8")
    result = selection_research.run_selection_research(contract_path, tmp_path / "runs")
    root = Path(result["root"])

    assert result["account_runs"] == result["plan"]["account_runs"] == 40
    assert result["first_layer_account_runs"] == 24
    assert result["second_layer_account_runs"] == 16
    assert len(result["trials"]) == 20
    assert {row["method"] for row in result["trials"]} == {"stepwise"}
    for horizon in (1, 4, 24):
        scores = pd.read_csv(root / f"models/h{horizon}/scores.csv", index_col=[0, 1])
        assert list(scores.columns) == ["stepwise"]
        assert scores["stepwise"].isna().all()  # Fixture prices give zero prediction targets.
        readiness = pd.read_csv(root / f"models/h{horizon}/factor_availability.csv")
        assert readiness["shared_model_ready"].any()
    assert "stepwise" in (root / "report.md").read_text(encoding="utf-8")
    assert not list(root.glob("**/pool"))
