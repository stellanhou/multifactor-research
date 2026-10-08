import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/multifactor_position_validation.py"
SPEC = importlib.util.spec_from_file_location("multifactor_position_validation", SCRIPT)
position_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(position_validation)
raw_scores = position_validation.raw_scores
rank_reference = position_validation.rank_reference
retention_targets = position_validation.retention_targets


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_raw_scores_use_native_cadence_and_switch_models_at_fit_timestamp(horizon):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-05T00:00:00Z")
    hourly = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2), freq="h")
    symbols = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([hourly, symbols], names=["timestamp", "symbol"])
    per_symbol = {
        "A": (1.0, 10.0, 100.0),
        "B": (2.0, 20.0, 200.0),
        "C": (3.0, 30.0, 300.0),
        "D": (4.0, 40.0, 400.0),
    }
    panel = pd.DataFrame(
        [per_symbol[symbol] for _, symbol in index],
        index=index,
        columns=["f1", "f2", "f3"],
    )
    decision_times = pd.date_range(
        start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2), freq=f"{horizon}h"
    )
    boundary = decision_times[2]
    metadata = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "r0_rebalance_hours": horizon,
        "model_schedule": [
            {
                "fit_timestamp": decision_times[0].isoformat(),
                "status": "active",
                "coefficients": {"f1": 1.0, "f2": 0.0, "f3": 0.0},
            },
            {
                "fit_timestamp": boundary.isoformat(),
                "status": "active",
                "coefficients": {"f1": 0.0, "f2": 1.0, "f3": 0.0},
            },
        ],
    }

    scores = raw_scores(panel, ["f1", "f2", "f3"], symbols, metadata)

    pd.testing.assert_index_equal(scores.index, decision_times)
    assert scores.index.to_series().diff().dropna().eq(pd.Timedelta(hours=horizon)).all()
    assert scores.loc[decision_times[1], "A"] == pytest.approx(1.0)
    assert scores.loc[boundary, "A"] == pytest.approx(10.0)


def test_hold4_retains_inside_rank_four_exits_outside_rank_four_and_exits_missing_assets():
    index = pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="4h")
    symbols = list("ABCDEFGH")
    scores = pd.DataFrame(
        [
            [8, 7, 6, 5, 4, 3, 2, 1],  # Enter A/B long and G/H short.
            [5, 3, 8, 7, 6, 4, 2, 1],  # Retain A at long rank 4; B exits beyond rank 4.
            [3, 1, 8, 7, 6, 4, 2, 5],  # A and H leave their holding bands; D/B enter.
            [4, 1, np.nan, 6, 5, 3, np.nan, 2],  # Missing held C/G must exit immediately.
            [np.nan] * 8,  # A model-cash row liquidates all remaining seats.
        ],
        index=index,
        columns=symbols,
        dtype=float,
    )
    targets = retention_targets(scores, rank_reference(scores))

    assert targets.loc[index[0], ["A", "B"]].eq(0.2).all()
    assert targets.loc[index[0], ["G", "H"]].eq(-0.2).all()
    assert targets.loc[index[1], ["A", "C"]].eq(0.2).all()
    assert targets.loc[index[1], "B"] == 0.0
    assert targets.loc[index[1], ["G", "H"]].eq(-0.2).all()
    assert targets.loc[index[2], ["C", "D"]].eq(0.2).all()
    assert targets.loc[index[2], ["B", "G"]].eq(-0.2).all()
    assert targets.loc[index[2], ["A", "H"]].eq(0.0).all()
    assert targets.loc[index[3], ["C", "G"]].eq(0.0).all()
    assert targets.loc[index[4]].eq(0.0).all()
