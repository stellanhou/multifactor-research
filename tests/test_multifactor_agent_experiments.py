from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research import multifactor_agent_experiments as experiments


class _Contract:
    run_id = "baseline_run"
    bounds = (
        pd.Timestamp("2025-01-01T02:00:00Z"),
        pd.Timestamp("2025-01-01T06:00:00Z"),
    )

    def as_dict(self):
        return {"run_id": self.run_id, "cards": ["full-pool-card-paths"]}


def _snapshot():
    timestamps = pd.date_range("2025-01-01T01:00:00Z", periods=5, freq="h")
    symbols = ["BTCUSDT", "ETHUSDT"]
    index = pd.MultiIndex.from_product(
        [timestamps, symbols], names=["timestamp", "symbol"]
    )
    standardized = pd.DataFrame(
        {
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "f1": [2.0, 4.0, 4.0, 8.0, 6.0, 12.0, 8.0, 16.0, 10.0, 20.0],
            "f3": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
        },
        index=index,
    )
    common_mask = pd.Series(True, index=index, name="common_mask")
    common_mask.loc[(timestamps[2], "ETHUSDT")] = False
    eligible = pd.Series(True, index=index, name="eligible")
    valid_masks = standardized.notna()
    cards = [
        {"id": "f2", "title": "second in code order", "path": "f2.json"},
        {"id": "f1", "title": "first in code order", "path": "f1.json"},
        {"id": "f3", "title": "third in code order", "path": "f3.json"},
    ]
    result = {
        "root": "/missing/stale/baseline/path",
        "factors": [
            {"id": "f2", "code": "F1", "snapshot_path": "cards/001_f2.json"},
            {"id": "f1", "code": "F2", "snapshot_path": "cards/002_f1.json"},
            {"id": "f3", "code": "F3", "snapshot_path": "cards/003_f3.json"},
        ],
        "input_snapshot": "inputs/",
        "card_snapshot": "cards/",
        "factor_diagnostics": {
            "standardized": "factor_panels/standardized.csv",
            "common_mask": "factor_panels/common_mask.csv",
        },
        "experiments": [
            {"name": "single:F1", "kind": "single_factor", "metrics": {}},
            {
                "name": "equal_weight",
                "kind": "equal_weight",
                "metrics": {
                    "net_return": 0.08,
                    "max_drawdown": 0.12,
                    "sharpe_ratio": 1.1,
                    "total_turnover": 4.0,
                    "total_fees": 2.0,
                    "total_slippage_cost": 1.0,
                    "total_funding": -0.5,
                },
            },
        ],
    }
    inputs = SimpleNamespace(frames={symbol: pd.DataFrame() for symbol in symbols},
                             diagnostics={"source_causality": "retrospective_or_unknown"})
    factors = SimpleNamespace(
        standardized=standardized,
        common_mask=common_mask,
        eligible=eligible,
        valid_masks=valid_masks,
    )
    return SimpleNamespace(
        contract=_Contract(),
        inputs=inputs,
        cards=cards,
        factors=factors,
        result=result,
        records={},
    )


def test_fixed_subset_order_is_sorted_and_size_ascending():
    assert experiments.fixed_subset_order(["c", "a", "b"]) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert experiments.fixed_subset_order(["d", "b", "a", "c"]) == [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "c", "d"),
        ("b", "c", "d"),
    ]
    with pytest.raises(ValueError, match="unique"):
        experiments.fixed_subset_order(["a", "a"])


def test_execute_subset_reuses_full_pool_standardization_and_mask(tmp_path, monkeypatch):
    snapshot = _snapshot()
    captured = {}

    def fake_run_one(name, kind, included, score, inputs, factors, contract,
                     directory, sample, factor_codes):
        captured.update(
            name=name,
            kind=kind,
            included=included,
            score=score.copy(),
            inputs=inputs,
            factors=factors,
            contract=contract,
            directory=directory,
            sample=sample.copy(),
            factor_codes=factor_codes.copy(),
        )
        directory.mkdir()
        metrics = {
            "net_return": 0.11,
            "max_drawdown": 0.10,
            "sharpe_ratio": 1.4,
            "total_turnover": 5.0,
            "total_fees": 2.5,
            "total_slippage_cost": 1.2,
            "total_funding": -0.25,
        }
        return {
            "included_factors": [factor_codes[card_id] for card_id in included],
            "metrics": metrics,
            "sample": {**sample, "account_signal_rows": 4, "account_signal_grid_hours": 4},
            "artifacts": {"ledger": "experiments/selected_subset/ledger.csv"},
        }

    monkeypatch.setattr(experiments, "_run_one", fake_run_one)
    monkeypatch.setattr(experiments, "_code_version", lambda: {"git_commit": "fixture"})
    root = tmp_path / "session" / "agent_arm" / "experiment-001"
    result = experiments.execute_subset(snapshot, ["f3", "f2"], root)

    assert captured["included"] == ["f2", "f3"]
    assert captured["factor_codes"] == {"f2": "F1", "f1": "F2", "f3": "F3"}
    expected_score = snapshot.factors.standardized[["f2", "f3"]].mean(axis=1)
    pd.testing.assert_series_equal(captured["score"], expected_score.rename("score"))
    assert captured["factors"].common_mask is snapshot.factors.common_mask
    # The selected values remain finite even where a third pool factor invalidated
    # the baseline mask; _run_one receives that unchanged full-pool mask to gate them.
    assert np.isfinite(captured["score"].loc[(pd.Timestamp("2025-01-01T03:00Z"), "ETHUSDT")])
    assert captured["sample"]["account_signal_window_start"] == "2025-01-01T01:00:00+00:00"
    assert captured["sample"]["account_signal_window_end_exclusive"] == "2025-01-01T05:00:00+00:00"
    assert result["selected_card_ids"] == ["f2", "f3"]
    assert result["baseline_run_id"] == "baseline_run"
    assert result["baseline_equal_weight_record_id"] == "strategy:equal_weight"
    assert result["metrics"]["net_return"] == 0.11
    assert result["delta_vs_equal_weight"]["net_return"] == pytest.approx(0.03)
    assert result["delta_vs_equal_weight"]["total_funding"] == pytest.approx(0.25)
    assert result["coverage"]["account_signal_rows"] == 4
    assert result["signalwindow"]["signal_grid_hours"] == 4
    assert result["inherited_prior_results_exposed"] is True
    assert result["artifacts"]["ledger"] == "experiments/selected_subset/ledger.csv"
    assert (root / "contract.json").is_file()
    definition = (root / "definition.json").read_text(encoding="utf-8")
    assert '"selected_card_ids": [\n    "f2",\n    "f3"' in definition
    assert '"baseline_run_id": "baseline_run"' in definition
    assert '"baseline_equal_weight_record_id": "strategy:equal_weight"' in definition
    assert '"baseline_common_mask": "factor_panels/common_mask.csv"' in definition
    assert '"inherited_prior_results_exposed": true' in definition
    assert (root / "result.json").is_file()


def test_execute_subset_rejects_full_pool_and_duplicate_selection(tmp_path):
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="smaller than the full pool"):
        experiments.execute_subset(snapshot, ["f1", "f2", "f3"], tmp_path / "full")
    with pytest.raises(ValueError, match="duplicates"):
        experiments.execute_subset(snapshot, ["f1", "f1"], tmp_path / "duplicate")
    assert not (tmp_path / "full").exists()
    assert not (tmp_path / "duplicate").exists()
