"""Bounded roles execute the same frozen account; invalid proposals never execute."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_inputs import FactorInputPanel
from crypto_quant.research.strategy_research.cli import ReplayModel
from crypto_quant.research.strategy_research.multifactor_contracts import MultifactorContract
from crypto_quant.research.strategy_research.multifactor_factors import process_factors
from crypto_quant.research.strategy_research.multifactor_workflow import run_experiments
from crypto_quant.research.strategy_research.multifactor_agent_workflow import run_agent_session

MODULE = "crypto_quant.research.strategy_research.multifactor_agent_workflow"


@pytest.fixture
def snapshot(tmp_path):
    contract = MultifactorContract.from_dict({
        "schema_version": 1, "run_id": "base", "purpose": "engineering", "stage": "development",
        "start": "2024-03-01T00:00:00Z", "end": "2024-03-01T04:00:00Z", "warmup_hours": 2,
        "horizon_hours": 24, "cards": ["f1.json", "f2.json", "f3.json"], "universe": "universe.csv",
        "prior_data_use": "known engineering fixture", "data_processing": "retrospective_or_unknown",
        "costs": {"initial_capital": 10000, "fee_bps": 10, "slippage_bps": 5, "stress_multiplier": 2},
        "portfolio": {"long_count": 1, "short_count": 1, "gross_exposure": .4,
                      "max_asset_weight": .2, "rebalance_hours": 1, "margin_fraction": .1}})
    times = pd.date_range(contract.input_start, contract.end, freq="h", inclusive="left")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    values = pd.DataFrame({"perp_close": np.tile([100., 110., 120.], len(times)),
                           "spot_close": np.tile([120., 100., 110.], len(times)),
                           "perp_quote_volume": np.tile([40., 60., 30.], len(times))}, index=index)
    universe = pd.Series(True, index=index, name="eligible")
    panel = FactorInputPanel(values, universe, {})
    start, end = contract.bounds
    market_times = pd.date_range(start - pd.Timedelta(hours=1), end, freq="h", inclusive="left")
    frames = {s: pd.DataFrame({"open": 100. + i, "close": 100. + i,
                               "mark_close": 100. + i}, index=market_times) for i, s in enumerate(symbols)}
    inputs = SimpleNamespace(panel=panel, universe=universe, frames=frames,
        funding=pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"]),
        diagnostics={"source_causality": "retrospective_or_unknown"})
    cards = [{"id": f"f{i}", "title": f"Factor {i}", "path": str(tmp_path / f"f{i}.json"),
              "expression": expression, "direction": 1}
             for i, expression in enumerate(("perp_close", "spot_close", "perp_quote_volume"), 1)]
    baseline = tmp_path / "base"
    baseline.mkdir()
    result = run_experiments(inputs, contract, baseline, cards, {"git_commit": "fixture"})
    factors = process_factors(panel, cards)
    records = {key: {"id": key, "kind": "fixture", "data": {}, "source": "fixture"}
               for key in ("data_usage", "baseline_metrics", "drop:F1")}
    return SimpleNamespace(contract=contract, inputs=inputs, cards=cards, factors=factors,
                           result=result, records=records)


def review(ref="data_usage"):
    return {"role": "review", "response": {"summary": "Historical engineering evidence.",
            "findings": [{"claim": "This is an engineering replay.", "evidence_refs": [ref]}],
            "limitations": ["Not independent validation."]}}


def design(ids=("f2", "f3")):
    return {"role": "design", "response": {"action": "experiment", "hypothesis": "Reproduce a fixed subset.",
            "strategy_card_ids": list(ids), "evidence_refs": ["drop:F1"]}}


def run(tmp_path, snapshot, replies, *, budget=1, context_bytes=160000):
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"schema_version": 1, "run_id": "agent-test",
                       "experiment_budget": budget, "context_bytes": context_bytes}))
    with patch(f"{MODULE}.load_baseline_snapshot", return_value=snapshot), \
            patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
        return run_agent_session(tmp_path / "base", session, tmp_path / "runs", ReplayModel(replies),
                                 model_mode="replay", model_settings={"provider": "fixed-replay", "model": "fixed-replay"})


def test_complete_loop_preserves_sample_budget_and_hides_fixed_arm(tmp_path, snapshot):
    result = run(tmp_path, snapshot, [review(), design(), review("agent-experiment:001")])
    root = Path(result["root"])
    assert result["model_calls"] == 3
    assert result["model_mode"] == "replay" and not result["live_agent_loop_executed"]
    assert not result["independent_agent_value_established"]
    assert result["comparison"]["maximum_experiment_budget_per_arm"] == 1
    assert result["agent_experiments"][0]["selected_card_ids"] == ["f2", "f3"]
    assert result["fixed_experiments"][0]["selected_card_ids"] == ["f1", "f2"]
    for item in result["agent_experiments"] + result["fixed_experiments"]:
        assert item["coverage"]["account_signal_window_common_rows"] == 12
        assert (root / item["path"] / item["artifacts"]["orders"]).is_file()
        assert json.loads((root / item["path"] / "contract.json").read_text()) == snapshot.contract.as_dict()
    calls = sorted((root / "model_calls").glob("*.request.json"))
    assert [json.loads(json.loads(p.read_text())[1]["content"])["role"] for p in calls] == ["review", "design", "review"]
    for p in calls:
        request = json.loads(json.loads(p.read_text())[1]["content"])
        assert all(not x["id"].startswith("fixed-experiment") for x in request["records"])
        assert "fixed_arm" not in p.read_text()
    assert (root / "fixed-plan.json").is_file() and (root / "report.md").is_file()
    with pytest.raises(FileExistsError):
        run(tmp_path, snapshot, [review(), design(), review()])


@pytest.mark.parametrize("change", ["unknown_reference", "immutable_costs"])
def test_rejected_proposal_is_saved_without_execution_or_retry(tmp_path, snapshot, change):
    proposal = design()
    if change == "unknown_reference":
        proposal["response"]["evidence_refs"] = ["missing-record"]
    else:
        proposal["response"]["costs"] = {"fee_bps": 0}
    with pytest.raises(ValueError):
        run(tmp_path, snapshot, [review(), proposal])
    root = tmp_path / "runs/agent-test"
    assert (root / "model_calls/call-0002.rejected.json").is_file()
    assert not (root / "agent_arm").exists() and not (root / "fixed_arm").exists()
    state = json.loads((root / "state.json").read_text())
    assert state["status"] == "failed" and state["model_calls"] == 2


def test_repeated_subset_consumes_call_then_stops(tmp_path, snapshot):
    with pytest.raises(ValueError, match="already been evaluated"):
        run(tmp_path, snapshot, [review(), design(), review("agent-experiment:001"), design(("f3", "f2"))], budget=2)
    root = tmp_path / "runs/agent-test"
    assert (root / "agent_arm/experiment-001/result.json").exists()
    assert not (root / "agent_arm/experiment-002").exists()
    assert json.loads((root / "state.json").read_text())["model_calls"] == 4


def test_agent_stop_preserves_predeclared_fixed_budget(tmp_path, snapshot):
    stop = {"role": "design", "response": {"action": "stop", "hypothesis": "No justified new experiment.",
            "strategy_card_ids": [], "evidence_refs": ["data_usage"]}}
    result = run(tmp_path, snapshot, [review(), stop], budget=2)
    assert result["stop_reason"] == "agent_stopped" and result["model_calls"] == 2
    assert not result["agent_experiments"] and len(result["fixed_experiments"]) == 2
    assert "agent_best_net_return" not in result["comparison"]


def test_context_limit_fails_before_model_call(tmp_path, snapshot):
    with pytest.raises(ValueError, match="context limit"):
        run(tmp_path, snapshot, [review()], context_bytes=1)
    root = tmp_path / "runs/agent-test"
    assert not (root / "model_calls").exists()
    assert json.loads((root / "state.json").read_text())["model_calls"] == 0
