"""Input-boundary tests for the new deterministic baseline."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_contracts import MultifactorContract
from crypto_quant.research.strategy_research.multifactor_data import load_universe, _funding, _bars


def raw_contract():
    return {"schema_version": 1, "run_id": "baseline-test", "purpose": "engineering", "stage": "development",
            "start": "2024-03-01T00:00:00Z", "end": "2024-03-02T00:00:00Z", "warmup_hours": 24,
            "horizon_hours": 24, "cards": ["one.json", "two.json"], "universe": "universe.csv",
            "prior_data_use": "historical engineering sample", "data_processing": "retrospective estimates",
            "costs": {"initial_capital": 10000, "fee_bps": 5, "slippage_bps": 5, "stress_multiplier": 2},
            "portfolio": {"long_count": 1, "short_count": 1, "gross_exposure": .8,
                          "max_asset_weight": .4, "rebalance_hours": 24, "margin_fraction": .1}}


@pytest.mark.parametrize("key,value", [("horizon_hours", 2), ("warmup_hours", 0), ("stage", "final_test"),
                                       ("end", "2024-02-01T00:00:00Z"), ("cards", ["one.json", "one.json"])])
def test_contract_rejects_invalid_frozen_inputs(key, value):
    raw = raw_contract()
    raw[key] = value
    with pytest.raises(ValueError):
        MultifactorContract.from_dict(raw)


def test_contract_rejects_nonfinite_or_overlevered_costs_and_allocations():
    for section, key, value in [("costs", "fee_bps", float("nan")), ("portfolio", "gross_exposure", 1.1),
                                 ("portfolio", "max_asset_weight", .3), ("portfolio", "long_count", True)]:
        raw = copy.deepcopy(raw_contract())
        raw[section][key] = value
        with pytest.raises(ValueError):
            MultifactorContract.from_dict(raw)


def test_universe_requires_explicit_full_warmup_membership():
    contract = MultifactorContract.from_dict(raw_contract())
    index = pd.MultiIndex.from_product([pd.date_range(contract.input_start, contract.bounds[1], freq="h", inclusive="left"),
                                        ["BTCUSDT", "ETHUSDT"]], names=["timestamp", "symbol"])
    frame = pd.Series(True, index=index, name="eligible").reset_index()
    frame.loc[0, "eligible"] = False
    with TemporaryDirectory() as directory:
        path = Path(directory) / "universe.csv"
        frame.to_csv(path, index=False)
        loaded = load_universe(path, contract)
        assert not loaded.iloc[0]
        frame.iloc[2:].to_csv(path, index=False)
        with pytest.raises(ValueError, match="warmup"):
            load_universe(path, contract)
        invalid = frame.copy()
        invalid["eligible"] = invalid.eligible.astype(object)
        invalid.loc[0, "eligible"] = None
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError, match="explicit boolean"):
            load_universe(path, contract)


def funding_store(times):
    frame = pd.DataFrame({"funding_rate": .0001, "funding_interval_hours": 8, "mark_price": 1000.},
                         index=pd.DatetimeIndex(times, tz="UTC", name="timestamp"))
    frame.attrs["source_table"] = "fixture"
    store = Mock()
    store.load_funding.return_value = frame
    return store


def test_funding_preserves_events_and_scales_native_contract_units():
    store = funding_store(["2024-03-01 00:00", "2024-03-01 08:00", "2024-03-01 16:00"])
    events, provenance = _funding(store, "SHIBUSDT", "1000SHIBUSDT", pd.Timestamp("2024-03-01", tz="UTC"),
                                  pd.Timestamp("2024-03-02", tz="UTC"), 1000)
    assert events.mark_price.eq(1).all()
    assert len(events) == provenance["events"] == 3
    assert store.load_funding.call_args.kwargs["include_previous"] is True


def test_funding_interval_transition_accepts_either_adjacent_declared_cadence():
    times = pd.to_datetime([
        "2024-03-01 00:00", "2024-03-01 04:00", "2024-03-01 06:00",
        "2024-03-01 08:00", "2024-03-01 16:00",
    ], utc=True)
    frame = pd.DataFrame({"funding_rate": .0001,
                          "funding_interval_hours": [4, 2, 2, 2, 8],
                          "mark_price": 1000.}, index=times)
    frame.index.name = "timestamp"
    frame.attrs["source_table"] = "fixture"
    store = Mock()
    store.load_funding.return_value = frame

    events, _ = _funding(store, "SOLUSDT", "SOLUSDT", times[0],
                         pd.Timestamp("2024-03-02 00:00", tz="UTC"), 1)

    assert len(events) == 5


def test_missing_funding_is_not_silently_zeroed():
    store = funding_store(["2024-03-01 00:00", "2024-03-01 16:00"])
    with pytest.raises(ValueError, match="missing funding"):
        _funding(store, "BTCUSDT", "BTCUSDT", pd.Timestamp("2024-03-01", tz="UTC"),
                 pd.Timestamp("2024-03-02", tz="UTC"), 1)
    store = funding_store(["2024-03-01 00:00", "2024-03-01 08:00"])
    with pytest.raises(ValueError, match="coverage ends"):
        _funding(store, "BTCUSDT", "BTCUSDT", pd.Timestamp("2024-03-01", tz="UTC"),
                 pd.Timestamp("2024-03-02", tz="UTC"), 1)


def test_execution_data_rejects_missing_or_shortened_hour():
    index = pd.date_range("2024-03-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                          "close_time": index + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)}, index=index)
    store = Mock()
    store.load_bars.return_value = frame.drop(index[2])
    with pytest.raises(ValueError, match="grid"):
        _bars(store, "BTCUSDT", "trade", index)
    frame.loc[index[2], "close_time"] -= pd.Timedelta(minutes=1)
    store.load_bars.return_value = frame
    with pytest.raises(ValueError, match="incomplete hourly"):
        _bars(store, "BTCUSDT", "trade", index)
