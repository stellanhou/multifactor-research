import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.strategy_research.multifactor_factors import (
    process_factors,
    read_cards,
)


def _card(
    path,
    *,
    expression="perp_close",
    direction=-1,
    retained_horizons=(1, 4, 24),
    passed_horizons=(24,),
    b_status="passed",
    eligible=True,
):
    compiled = compile_expression(expression)
    value = {
        "id": path.stem,
        "title": path.stem,
        "source_type": "factor_mining",
        "status": "research_idea",
        "source": {"run_id": "test-run"},
        "original_claim": {
            "direction": direction,
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
            "retained_horizons": list(retained_horizons),
            "passed_horizons": list(passed_horizons),
        },
        "b_validation_status": b_status,
        "admission_evidence": {"eligible_for_idea_pool": eligible},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_read_cards_validates_direction_horizon_and_dsl_metadata(tmp_path):
    path = tmp_path / "valid.json"
    source = _card(path, expression="ts_return(perp_close, 2)", direction=1)

    cards = read_cards([path], horizon_hours=24, max_lookback_hours=2)

    assert cards[0]["id"] == source["id"]
    assert cards[0]["direction"] == 1
    assert cards[0]["lookback_hours"] == 2
    assert cards[0]["fields"] == ["perp_close"]
    assert cards[0]["card_snapshot"] == source

    with pytest.raises(ValueError, match="direction must be explicitly"):
        invalid_direction = tmp_path / "invalid-direction.json"
        _card(invalid_direction, direction=None)
        read_cards([invalid_direction], 24, 10)

    with pytest.raises(ValueError, match="did not pass"):
        read_cards([path], horizon_hours=1, max_lookback_hours=2)

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["original_claim"]["formula"]["fields"] = ["spot_close"]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="input fields differ"):
        read_cards([path], horizon_hours=24, max_lookback_hours=2)


def test_read_cards_rejects_lookback_beyond_configured_limit(tmp_path):
    path = tmp_path / "long.json"
    _card(path, expression="ts_return(perp_close, 3)")

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        read_cards([path], horizon_hours=24, max_lookback_hours=2)


def test_read_cards_requires_passed_b_admission_and_usdm_hourly_market(tmp_path):
    for name, changes, error in [
        ("b-status", {"b_status": "not_passed"}, "B validation status is not passed"),
        ("admission", {"eligible": False}, "not eligible for the idea pool"),
    ]:
        path = tmp_path / f"{name}.json"
        _card(path, **changes)
        with pytest.raises(ValueError, match=error):
            read_cards([path], horizon_hours=24, max_lookback_hours=2)

    path = tmp_path / "wrong-market.json"
    card = _card(path)
    card["market_and_horizon"]["market"] = "spot"
    path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(ValueError, match="Binance USD-M perpetual"):
        read_cards([path], horizon_hours=24, max_lookback_hours=2)


def test_read_cards_accepts_current_workflow_card_without_program_version(tmp_path):
    # Captured from the factor-mining producer; its matching export test verifies the contract.
    expression = "div(perp_close, ts_mean(perp_close, 3))"
    fixture = Path(__file__).parent / "fixtures/fm_v6_workflow_card.json"
    card = json.loads(fixture.read_text(encoding="utf-8"))
    path = tmp_path / "current-workflow-card.json"
    path.write_text(json.dumps(card), encoding="utf-8")

    loaded = read_cards([path], horizon_hours=24, max_lookback_hours=3)

    assert "program_version" not in card
    assert loaded[0]["expression"] == expression
    assert loaded[0]["direction"] == -1


def test_process_factors_orients_and_standardizes_without_filling_missing_values(monkeypatch):
    times = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    symbols = ["AUSDT", "BUSDT", "CUSDT", "DUSDT"]
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    perp = [10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0]
    spot = [40.0, 30.0, 20.0, 10.0, 41.0, 31.0, 21.0, 11.0]
    spot[2] = np.nan
    values = pd.DataFrame({"perp_close": perp, "spot_close": spot}, index=index)
    panel = FactorInputPanel(values, universe, {"quality": {"repaired_rows": 3}})
    cards = [
        {"id": "perp", "expression": "perp_close", "direction": 1,
         "fields": ["perp_close"], "lookback_hours": 0},
        {"id": "spot", "expression": "spot_close", "direction": -1,
         "fields": ["spot_close"], "lookback_hours": 0},
    ]

    factors = process_factors(panel, cards)

    assert factors.raw.loc[(times[0], "CUSDT"), "spot"] != factors.raw.loc[(times[0], "CUSDT"), "spot"]
    assert not factors.valid_masks.loc[(times[0], "CUSDT"), "spot"]
    assert not factors.common_mask.loc[(times[0], "CUSDT")]
    assert factors.common_mask.loc[(times[0], "AUSDT")]
    for timestamp in times:
        section = factors.standardized.xs(timestamp, level="timestamp")
        assert section["perp"].mean() == pytest.approx(0.0)
        assert section["perp"].std(ddof=0) == pytest.approx(1.0)
    # The opposite raw ordering is corrected by direction before standardization.
    first = factors.standardized.xs(times[0], level="timestamp")
    assert first.loc["AUSDT", "perp"] < first.loc["DUSDT", "perp"]
    assert first.loc["AUSDT", "spot"] < first.loc["DUSDT", "spot"]
    assert factors.coverage.loc["spot", "missing_rows"] == 1
    assert factors.input_diagnostics == panel.diagnostics
    assert len(factors.correlations) == len(times)
    assert factors.correlations["spearman"].dropna().iloc[0] == pytest.approx(1.0)
    assert factors.correlation_summary.iloc[0]["positive_share"] == pytest.approx(1.0)

    from crypto_quant.research.strategy_research import multifactor_factors

    def fail_if_correlations_are_calculated(*args, **kwargs):
        pytest.fail("correlation diagnostics should be skipped")

    monkeypatch.setattr(
        multifactor_factors, "_cross_section_correlations", fail_if_correlations_are_calculated,
    )
    without_correlations = process_factors(panel, cards, include_correlations=False)

    pd.testing.assert_frame_equal(without_correlations.raw, factors.raw)
    pd.testing.assert_frame_equal(without_correlations.standardized, factors.standardized)
    pd.testing.assert_series_equal(without_correlations.eligible, factors.eligible)
    pd.testing.assert_frame_equal(without_correlations.valid_masks, factors.valid_masks)
    pd.testing.assert_series_equal(without_correlations.common_mask, factors.common_mask)
    pd.testing.assert_frame_equal(without_correlations.coverage, factors.coverage)
    assert without_correlations.correlations.empty
    assert list(without_correlations.correlations.columns) == [
        "timestamp", "factor_a", "factor_b", "spearman", "pairwise_n",
    ]
    assert without_correlations.correlation_summary.empty
    assert without_correlations.correlation_summary.index.names == ["factor_a", "factor_b"]
    assert list(without_correlations.correlation_summary.columns) == [
        "valid_periods", "mean", "median", "mean_abs", "p10", "p90", "positive_share",
    ]


def test_process_factors_leaves_constant_cross_sections_missing():
    times = pd.date_range("2025-02-01", periods=1, freq="h", tz="UTC")
    symbols = ["AUSDT", "BUSDT"]
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    values = pd.DataFrame({"perp_close": [5.0, 5.0]}, index=index)
    panel = FactorInputPanel(values, universe, {})

    factors = process_factors(panel, [{"id": "constant", "expression": "perp_close", "direction": 1}])

    assert factors.valid_masks["constant"].all()
    assert factors.standardized["constant"].isna().all()
    assert not factors.common_mask.any()


def test_process_factors_are_causal_and_keep_source_gaps_in_masks():
    times = pd.date_range("2025-03-01", periods=8, freq="h", tz="UTC")
    symbols = ["AUSDT", "BUSDT", "CUSDT", "DUSDT"]
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    universe = pd.Series(True, index=index, dtype=bool)
    base = np.tile(np.arange(1.0, 9.0), (len(symbols), 1)).T
    scales = np.array([1.0, 2.0, 3.0, 4.0])
    perp = (base * scales).reshape(-1)
    spot = (base * (scales + 1.0)).reshape(-1)
    values = pd.DataFrame({"perp_close": perp, "spot_close": spot}, index=index)
    gap_index = (times[3], "BUSDT")
    values.loc[gap_index, "perp_close"] = np.nan
    panel = FactorInputPanel(values, universe, {"source": "local", "gaps": [str(gap_index)]})
    cards = [
        {"id": "return", "expression": "ts_return(perp_close, 1)", "direction": 1},
        {"id": "ratio", "expression": "div(spot_close, perp_close)", "direction": -1},
    ]
    original = process_factors(panel, cards)

    mutated_values = values.copy()
    future = index.get_level_values("timestamp") > times[4]
    mutated_values.loc[future, "perp_close"] *= 100.0
    mutated_values.loc[future, "spot_close"] /= 100.0
    mutated = process_factors(FactorInputPanel(mutated_values, universe, panel.diagnostics), cards)
    prior = original.raw.index.get_level_values("timestamp") <= times[4]
    pd.testing.assert_frame_equal(original.raw.loc[prior], mutated.raw.loc[prior])
    pd.testing.assert_frame_equal(original.standardized.loc[prior], mutated.standardized.loc[prior])

    assert not original.valid_masks.loc[gap_index, "return"]
    assert not original.valid_masks.loc[gap_index, "ratio"]
    assert not original.valid_masks.loc[(times[4], "BUSDT"), "return"]
    assert original.valid_masks.loc[(times[5], "BUSDT"), "return"]
    assert original.input_diagnostics == panel.diagnostics


def test_correlation_summary_exposes_sign_reversals_hidden_by_the_mean():
    from crypto_quant.research.strategy_research.multifactor_factors import _summarize_correlations

    correlations = pd.DataFrame({
        "factor_a": ["a", "a"],
        "factor_b": ["b", "b"],
        "spearman": [0.9, -0.9],
    })

    summary = _summarize_correlations(correlations).iloc[0]

    assert summary["mean"] == pytest.approx(0.0)
    assert summary["mean_abs"] == pytest.approx(0.9)
    assert summary["p10"] < 0 < summary["p90"]
