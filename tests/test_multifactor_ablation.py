import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.progress import ProgressLog
from crypto_quant.research.strategy_research import multifactor_ablation as ablation
from crypto_quant.research.strategy_research.multifactor_combination import rolling_icir_combine, symmetric_orthogonalize
from crypto_quant.research.strategy_research.multifactor_contracts import ResearchContract
from test_multifactor_pool_research import _icir_contract, _pool_cards, _write_card


POLICY = {"method": "symmetric_orthogonalized_rolling_icir", "window_hours": 8,
          "min_periods": 3, "negative_icir": "flip_factor"}


def _factors(periods=60):
    times = pd.date_range("2024-01-01", periods=periods, tz="UTC", freq="h", name="timestamp")
    symbols = sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"])
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    rng = np.random.default_rng(72)
    raw = rng.normal(size=(periods, len(symbols), 3))
    raw[:, :, 1] += raw[:, :, 0]
    raw -= raw.mean(axis=1, keepdims=True)
    raw /= raw.std(axis=1, keepdims=True)
    values = pd.DataFrame(raw.reshape(-1, 3), index=index, columns=["f1", "f2", "f3"])
    opens = pd.Series((100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=(periods, len(symbols))), axis=0))).reshape(-1),
                      index=index, name="perp_open")
    common = pd.Series(True, index=index)
    return SimpleNamespace(standardized=values, common_mask=common), opens


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_four_arms_change_only_the_two_declared_treatments_and_share_availability(horizon):
    factors, opens = _factors()
    scores, shared, availability, combinations, transformed, diagnostics = ablation.four_arm_scores(factors, opens, horizon, POLICY)
    expected_orthogonal, _ = symmetric_orthogonalize(factors.standardized, factors.common_mask)
    pd.testing.assert_frame_equal(transformed, expected_orthogonal)
    expected_b = rolling_icir_combine(factors.standardized, pd.DataFrame(index=diagnostics.index), opens,
                                     horizon_hours=horizon, policy=POLICY)
    expected_d = rolling_icir_combine(transformed, diagnostics, opens, horizon_hours=horizon, policy=POLICY)
    pd.testing.assert_series_equal(scores["A"], factors.standardized.mean(axis=1).where(shared).rename("score"))
    pd.testing.assert_series_equal(scores["B"], expected_b.score.where(shared))
    pd.testing.assert_series_equal(scores["C"], transformed.mean(axis=1).where(shared).rename("score"))
    pd.testing.assert_series_equal(scores["D"], expected_d.score.where(shared))
    pd.testing.assert_series_equal(shared, availability.all(axis=1).rename("paired_mask"))
    for score in scores.values():
        pd.testing.assert_series_equal(score.notna(), shared.rename("score"))
    assert availability["A"].iloc[0]
    assert not shared.iloc[0]
    assert shared.any()
    assert not scores["A"].equals(scores["B"])
    assert not scores["C"].equals(scores["D"])


def test_factorial_effects_include_conditional_effects_and_interaction():
    results = []
    for name, net in zip("ABCD", [0.1, 0.3, 0.2, 0.7]):
        metrics = dict.fromkeys(ablation.METRICS, net)
        results.append({"name": name, "metrics": metrics,
                        "stress_costs": {"metrics": {**metrics, "net_return": net / 2}}})
    effects = ablation.factorial_effects(results)
    assert effects["base"]["B-A"]["net_return"] == pytest.approx(0.2)
    assert effects["base"]["C-A"]["net_return"] == pytest.approx(0.1)
    assert effects["base"]["D-C"]["net_return"] == pytest.approx(0.5)
    assert effects["base"]["D-B"]["net_return"] == pytest.approx(0.4)
    assert effects["base"]["interaction_D-B-C+A"]["net_return"] == pytest.approx(0.3)
    assert effects["base"]["weighting_main_effect"]["net_return"] == pytest.approx(0.35)
    assert effects["stress"]["B-A"]["net_return"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exactly four"):
        ablation.factorial_effects(results[:3])


def test_real_account_four_arms_have_identical_signal_masks_and_cost_stress(tmp_path):
    factors, opens = _factors()
    index, times = factors.standardized.index, factors.standardized.index.get_level_values("timestamp").unique()
    start, end = times[3], times[-1] + pd.Timedelta(hours=1)
    panel_values = pd.DataFrame({"perp_open": opens, "perp_close": opens * 1.0001}, index=index)
    universe = factors.common_mask
    frames = {symbol: pd.DataFrame({"open": opens.xs(symbol, level="symbol"),
                                   "close": opens.xs(symbol, level="symbol") * 1.0001,
                                   "mark_close": opens.xs(symbol, level="symbol") * 1.0001})
              for symbol in index.get_level_values("symbol").unique()}
    inputs = SimpleNamespace(panel=FactorInputPanel(panel_values, universe, {}), frames=frames,
                             funding=pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"]))
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    paths = [cards_dir / "close.json", cards_dir / "return.json", cards_dir / "std.json"]
    for path, expression in zip(paths, ["perp_close", "ts_return(perp_close, 1)", "ts_std(perp_close, 2)"]):
        _write_card(path, path.stem, expression, horizons=(1,))
    value = _icir_contract()
    value.update(development_start=start.isoformat(), validation_start=end.isoformat(),
                 validation_end=(end + pd.Timedelta(days=1)).isoformat())
    parent = ablation.PoolResearchContract.from_dict(value)
    stage = ResearchContract.from_dict(parent.stage_dict(1, [str(path) for path in paths], "universe.csv", "manifest.json", "development"))
    cards = ablation.read_cards(paths, 1, 2)
    result = ablation._execute_horizon(tmp_path / "result", inputs, stage, cards, POLICY,
                                       ProgressLog.for_run(tmp_path / "progress"))
    masks = []
    for arm in result["experiments"]:
        directory = tmp_path / "result" / arm["path"]
        signals = pd.read_csv(directory / "signals.csv", index_col=0)
        masks.append(signals.notna())
        assert arm["metrics"]["total_fees"] > 0
        assert arm["stress_costs"]["metrics"]["net_return"] != arm["metrics"]["net_return"]
        assert (tmp_path / "result" / arm["stress_costs"]["path"] / "ledger.csv").is_file()
    for mask in masks[1:]:
        pd.testing.assert_frame_equal(mask, masks[0])
    assert (tmp_path / "result" / "combination" / "B" / "history_factors.csv").is_file()
    assert (tmp_path / "result" / "combination" / "D" / "history_factors.csv").is_file()


def test_runner_freezes_all_four_arms_then_runs_every_horizon_in_both_stages(tmp_path, monkeypatch):
    _pool_cards(tmp_path / "pool")
    value = _icir_contract()
    value["horizons"] = [1, 4]
    pool_path = tmp_path / "pool-contract.json"
    pool_path.write_text(json.dumps(value))
    contract_path = tmp_path / "ablation-contract.json"
    contract_path.write_text(json.dumps({"schema_version": 1, "run_id": "four-arm-test", "pool_contract": "pool-contract.json"}))
    (tmp_path / "manifest.json").write_text(json.dumps({"historical_causality_certified": False}))
    (tmp_path / "universe.csv").write_text("test universe")
    calls = []

    def fake_load(manifest, contract, root):
        plan = json.loads((root / "plan.json").read_text())
        assert plan["frozen_before_market_panel_access"] is True
        assert [arm["name"] for arm in plan["arms"]] == list("ABCD")
        assert not (root / "candidate_freeze.json").exists()
        calls.append(("load", contract.stage))
        return None

    def fake_execute(root, inputs, contract, cards, policy, progress, history_root=None):
        root.mkdir()
        calls.append((contract.stage, contract.horizon_hours))
        assert history_root == (root.parent.parent / "development" / f"h{contract.horizon_hours}"
                                if contract.stage == "internal_validation" else None)
        experiments = []
        for i, arm in enumerate(ablation.ARMS):
            metrics = {**dict.fromkeys(ablation.METRICS, 0.0), "net_return": -0.1 * i}
            experiments.append({**arm, "metrics": metrics, "stress_costs": {"metrics": metrics}})
        return {"stage": contract.stage, "horizon_hours": contract.horizon_hours,
                "experiments": experiments, "effects": ablation.factorial_effects(experiments)}

    monkeypatch.setattr(ablation, "_capture_sources", lambda root: None)
    monkeypatch.setattr(ablation, "load_research_inputs", fake_load)
    monkeypatch.setattr(ablation, "_snapshot_inputs", lambda *args: None)
    monkeypatch.setattr(ablation, "_execute_horizon", fake_execute)
    result = ablation.run_ablation(contract_path, tmp_path / "runs")
    assert calls == [("load", "development"), ("development", 1), ("development", 4),
                     ("load", "internal_validation"), ("internal_validation", 1), ("internal_validation", 4)]
    assert result["status"] == "ablation_complete"
    assert result["plan"]["account_runs"] == 32
    assert result["model_called"] is False
    assert result["paper_started"] is False
    assert len(pd.read_csv(Path(result["root"]) / "comparison.csv")) == 16
    with pytest.raises(ValueError, match="refusing overwrite"):
        ablation.run_ablation(contract_path, tmp_path / "runs")
