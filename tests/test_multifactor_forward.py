import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.factor_inputs import FactorInputPanel, INPUT_COLUMNS
from crypto_quant.research.strategy_research.multifactor_forward import (
    ForwardSession,
    FreshSeedData,
    ForwardSessionError,
    freeze_forward_candidate,
)


SYMBOLS = sorted(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                  "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"])
REQUIRED_FIELDS = {"perp_close", "mark_close", "spot_quote_volume", "spot_trades"}


class _Contract:
    def __init__(self, data):
        self.data = json.loads(json.dumps(data))
        self.run_id = self.data["run_id"]
        self.costs = self.data["costs"]
        self.portfolio = self.data["portfolio"]

    def as_dict(self):
        return json.loads(json.dumps(self.data))


def _snapshot_for_history(template, historical_result_path):
    history_root = historical_result_path.parent
    result = json.loads(historical_result_path.read_text(encoding="utf-8"))
    artifacts = result["artifacts"]
    baseline_result = json.loads(
        (history_root / artifacts["development_baseline_result"]["path"]).read_text(encoding="utf-8")
    )
    contract_data = json.loads(
        (history_root / artifacts["development_contract"]["path"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (history_root / artifacts["dataset_manifest"]["path"]).read_text(encoding="utf-8")
    )
    return SimpleNamespace(
        contract=_Contract(contract_data),
        inputs=SimpleNamespace(
            frames={symbol: pd.DataFrame() for symbol in manifest["symbols"]},
            dataset_manifest=manifest,
        ),
        cards=template.cards,
        factors=None,
        result=baseline_result,
        records={},
    )


def _copy_snapshot(snapshot, *, contract_data=None, frame_symbols=None, result_data=None):
    inputs = SimpleNamespace(
        frames={symbol: pd.DataFrame() for symbol in (frame_symbols or snapshot.inputs.frames)},
        dataset_manifest=snapshot.inputs.dataset_manifest,
    )
    return SimpleNamespace(
        contract=_Contract(contract_data or snapshot.contract.as_dict()),
        inputs=inputs,
        cards=snapshot.cards,
        factors=snapshot.factors,
        result=result_data or snapshot.result,
        records=snapshot.records,
    )
def _write_historical_qualification(root, snapshot, selected_ids, *, qualified=True,
                                   c_stress_return=0.02, validation_qualified=None):
    root.mkdir(parents=True)
    gates = {
        "min_net_return": 0.0,
        "max_drawdown": 0.15,
        "min_traded_bars": 1,
        "min_stress_return": 0.0,
    }
    candidate_selection = "max_AB_net_return_then_lower_drawdown_then_sorted_card_ids"
    hist_contract = {
        "schema_version": 1,
        "run_id": "forward-history-fixture",
        "dataset_manifest": "dataset_manifest.json",
        "cards": [f"cards/{card['id']}.json" for card in snapshot.cards],
        "universe": "universe.csv",
        "horizon_hours": 24,
        "warmup_hours": 168,
        "development_start": "2022-08-01T00:00:00Z",
        "validation_start": "2025-08-01T00:00:00Z",
        "validation_end": "2026-08-01T00:00:00Z",
        "prior_data_use": "C is prior internal validation; no final test has been declared.",
        "data_processing": "Strict observed dataset; no fallback.",
        "costs": {"initial_capital": 10000.0, "fee_bps": 10.0,
                  "slippage_bps": 5.0, "stress_multiplier": 2.0},
        "portfolio": {"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                      "max_asset_weight": 0.2, "rebalance_hours": 24,
                      "margin_fraction": 0.1},
        "experiment_budget": 2,
        "context_bytes": 256000,
        "qualification_gates": gates,
        "candidate_selection": candidate_selection,
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(hist_contract, indent=2) + "\n", encoding="utf-8")
    database_sha256 = "c" * 64
    dataset_manifest = {
        "schema_version": 2,
        "dataset_id": "fresh-fixture",
        "database": "market_data.sqlite",
        "database_signature": {"size_bytes": 4096, "sha256": database_sha256},
        "database_schema": "MarketDataStore-compatible SQLite v2 with row provenance tables",
        "source": "Binance Vision official monthly archives",
        "symbols": SYMBOLS,
        "window": {
            "timezone": "UTC",
            "input_start": "2022-07-24T23:00:00+00:00",
            "funding_start": "2022-07-17T23:00:00+00:00",
            "end_exclusive": "2026-08-01T00:00:00+00:00",
            "warmup_hours": 168,
            "stages": {
                "A": {"start": "2022-08-01T00:00:00+00:00", "end_exclusive": "2024-08-01T00:00:00+00:00"},
                "B": {"start": "2024-08-01T00:00:00+00:00", "end_exclusive": "2025-08-01T00:00:00+00:00"},
                "C": {"start": "2025-08-01T00:00:00+00:00", "end_exclusive": "2026-08-01T00:00:00+00:00"},
            },
            "official_archive_months": ["2022-07", "2026-08"],
        },
        "source_files": [],
        "coverage": [],
        "field_sources": {"spot_*": "official archive", "perpetual_*": "official archive"},
        "repair_state": {
            "price_interpolation": False,
            "funding_rate_interpolation": False,
            "funding_mark_is_native_event_field": False,
            "funding_mark_proxy_declared": True,
            "minute_to_hour_mark_aggregation": "only complete 60 observed minute rows; no interpolation",
            "primary_source_fallback": False,
            "primary_database_read": False,
            "synthetic_archive_rows": False,
        },
        "funding_mark": {
            "proxy_count": 1,
            "missing_proxy_count": 0,
            "max_proxy_age_ms": 60000,
            "min_proxy_age_ms": 0,
            "derived_hourly_mark_rows_by_symbol": {},
        },
        "source_causality": "original_archives_with_declared_funding_proxy",
        "source_causality_notes": "Fixture mirrors the declared production manifest schema.",
        "historical_causality_certified": False,
        "complete": True,
        "execution_grid_complete": True,
        "missing_archives": [],
        "parse_issue_count": 0,
        "spot_feature_missing_rows": 0,
        "feature_missing_rows_accepted": True,
        "unresolved_hourly_price_rows": 0,
        "download_policy": {"workers": 4, "retry_count": 0, "http_404": "recorded as missing archive"},
        "primary_source_fallback": False,
        "excluded_documented_repair_rows": [],
        "supplement_source_files": [],
    }
    source_manifest_path = root / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    dataset_path = root / "dataset_manifest.json"
    dataset_path.write_bytes(source_manifest_path.read_bytes())
    source_universe_path = root / "source_universe.csv"
    universe_payload = "timestamp,symbol,eligible\n" + "".join(
        f"2026-01-01T00:00:00Z,{symbol},true\n" for symbol in sorted(SYMBOLS)
    )
    source_universe_path.write_text(universe_payload, encoding="utf-8")
    universe_path = root / "universe.csv"
    universe_path.write_bytes(source_universe_path.read_bytes())
    card_manifest = []
    for card in snapshot.cards:
        source_card_path = root / "source_cards" / f"{card['id']}.json"
        source_card_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(card["card_snapshot"], ensure_ascii=False, sort_keys=True) + "\n"
        source_card_path.write_text(payload, encoding="utf-8")
        card_path = root / "cards" / f"{card['id']}.json"
        card_path.parent.mkdir(exist_ok=True)
        card_path.write_text(payload, encoding="utf-8")
        card_manifest.append({
            "id": card["id"], "title": card["title"],
            "source_path": str(source_card_path.resolve()),
            "source_sha256": hashlib.sha256(source_card_path.read_bytes()).hexdigest(),
            "snapshot_path": f"cards/{card['id']}.json",
        })
    cards_manifest_path = root / "cards.json"
    cards_manifest_path.write_text(json.dumps(card_manifest, indent=2) + "\n", encoding="utf-8")
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    source_gate = {
        "passed": True, "dataset_id": "fresh-fixture", "manifest_sha256": dataset_digest,
        "database_sha256": database_sha256, "primary_source_fallback": False,
        "C_read_before_AB_freeze": False,
    }
    source_path = root / "source_gate.json"
    source_path.write_text(json.dumps(source_gate, indent=2) + "\n", encoding="utf-8")
    (root / "contracts").mkdir()
    (root / "development_agent").mkdir()
    (root / "validation").mkdir()
    development_contract = {
        "schema_version": 2,
        "run_id": "forward_fixture_baseline",
        "purpose": "research",
        "stage": "development",
        "start": hist_contract["development_start"],
        "end": hist_contract["validation_start"],
        "warmup_hours": hist_contract["warmup_hours"],
        "horizon_hours": hist_contract["horizon_hours"],
        "cards": [item["source_path"] for item in card_manifest],
        "universe": str(source_universe_path.resolve()),
        "dataset_manifest": os.path.relpath(source_manifest_path.resolve(), (root / "contracts").resolve()),
        "prior_data_use": hist_contract["prior_data_use"],
        "data_processing": hist_contract["data_processing"],
        "costs": hist_contract["costs"],
        "portfolio": hist_contract["portfolio"],
    }
    development_contract_path = root / "contracts" / "development.json"
    development_contract_path.write_text(json.dumps(development_contract, indent=2) + "\n", encoding="utf-8")
    baseline_root = root / "development_full_pool" / development_contract["run_id"]
    (baseline_root / "cards").mkdir(parents=True)
    (baseline_root / "contract.json").write_text(
        json.dumps(development_contract, indent=2) + "\n", encoding="utf-8"
    )
    baseline_card_manifest = []
    baseline_factors = []
    for index, (card, record) in enumerate(zip(snapshot.cards, card_manifest), start=1):
        filename = f"{index:03d}_{card['id']}.json"
        baseline_card_path = baseline_root / "cards" / filename
        baseline_card_path.write_bytes((root / record["snapshot_path"]).read_bytes())
        card_record = {
            "id": card["id"], "title": card["title"], "source_path": record["source_path"],
            "snapshot_path": f"cards/{filename}",
            "sha256": hashlib.sha256(baseline_card_path.read_bytes()).hexdigest(),
            "direction": card["direction"], "horizon_hours": card["horizon_hours"],
            "lookback_hours": card["lookback_hours"], "expression": card["expression"],
        }
        baseline_card_manifest.append(card_record)
        baseline_factors.append({
            "code": f"F{index}", "id": card["id"], "title": card["title"],
            "source_path": record["source_path"], "snapshot_path": f"cards/{filename}",
        })
    (baseline_root / "cards.json").write_text(
        json.dumps(baseline_card_manifest, indent=2) + "\n", encoding="utf-8"
    )
    baseline_result = {
        "engine": "deterministic_multifactor_v1",
        "model_called": False,
        "agent_used": False,
        "status": "engineering_complete",
        "run_id": development_contract["run_id"],
        "root": str(baseline_root.resolve()),
        "purpose": "research",
        "stage": "development",
        "forward_validation_started": False,
        "paper_started": False,
        "published": False,
        "factors": baseline_factors,
        "contract": "contract.json",
        "experiments": [],
        "stress_costs": None,
        "coverage_impact": {},
        "factor_diagnostics": {},
        "input_snapshot": "inputs/",
        "card_snapshot": "cards/",
    }
    baseline_path = baseline_root / "result.json"
    baseline_path.write_text(json.dumps(baseline_result, indent=2) + "\n", encoding="utf-8")
    agent_path = root / "development_agent" / "result.json"
    agent_path.write_text(json.dumps({"model_calls": 1}) + "\n", encoding="utf-8")

    selected_codes = [f"F{index}" for index, card in enumerate(snapshot.cards, start=1)
                      if card["id"] in selected_ids]
    ab_metrics = {"net_return": 0.08, "max_drawdown": 0.10,
                  "sharpe_ratio": 1.0, "total_turnover": 4.0,
                  "total_fees": 2.0, "total_slippage_cost": 1.0, "total_funding": -0.2}
    ab_stress = {"net_return": 0.04, "max_drawdown": 0.12,
                 "sharpe_ratio": 0.5, "total_turnover": 4.0,
                 "total_fees": 4.0, "total_slippage_cost": 2.0, "total_funding": -0.2}
    trial = {
        "trial_id": "agent-001", "arm": "agent", "selected_card_ids": list(selected_ids),
        "included_factors": selected_codes, "metrics": ab_metrics, "traded_bars": 10,
        "stress_costs": {"metrics": ab_stress}, "qualification_failures": [], "qualified_ab": True,
        "source_trial_path": "development_agent/experiment-001",
        "source_artifacts": {"ledger": "development_agent/experiment-001/ledger.csv"},
    }
    trial_path = root / "candidate-trials.json"
    trial_path.write_text(json.dumps([trial], indent=2) + "\n", encoding="utf-8")
    freeze = {
        "selected_trial_id": "agent-001", "selected_card_ids": list(selected_ids),
        "baseline_run_id": development_contract["run_id"],
        "selected_factors": selected_codes, "selected_ab_metrics": ab_metrics,
        "selected_ab_stress_metrics": ab_stress, "selected_ab_qualified": True,
        "selection_rule": candidate_selection, "selected_before_c_access": True,
        "c_data_accessed": False, "qualification_gates": gates,
    }
    freeze_path = root / "candidate-freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    freeze_digest = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    c_metrics = {"net_return": 0.05, "max_drawdown": 0.11,
                 "sharpe_ratio": 0.7, "total_turnover": 3.0,
                 "total_fees": 1.5, "total_slippage_cost": 0.8, "total_funding": -0.1}
    c_stress = {"net_return": c_stress_return, "max_drawdown": 0.13,
                "sharpe_ratio": 0.3, "total_turnover": 3.0,
                "total_fees": 3.0, "total_slippage_cost": 1.6, "total_funding": -0.1}
    if validation_qualified is None:
        validation_qualified = qualified
    c_failures = [] if validation_qualified else ["min_stress_return"]
    validation = {
        "selected_candidate_card_ids": list(selected_ids),
        "candidate": {"included_card_ids": list(selected_ids), "metrics": c_metrics},
        "candidate_stress_costs": {"metrics": c_stress}, "traded_bars": 8,
        "qualification_failures": c_failures, "qualification_gates": gates,
        "qualified_c": validation_qualified,
        "same_full_pool_common_mask": True, "agent_called_after_c_access": False,
        "candidate_freeze_sha256": freeze_digest,
    }
    validation_path = root / "validation" / "result.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    artifacts = {}
    for key, path in {
        "contract": contract_path,
        "source_gate": source_path,
        "dataset_manifest": dataset_path,
        "universe": universe_path,
        "cards_manifest": cards_manifest_path,
        "development_contract": development_contract_path,
        "development_baseline_result": baseline_path,
        "agent_session_result": agent_path,
        "candidate_trials": trial_path,
        "candidate_freeze": freeze_path,
        "internal_validation_result": validation_path,
    }.items():
        artifacts[key] = {"path": str(path.relative_to(root)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "engine": "multifactor_historical_v1",
        "status": "qualified_candidate_frozen" if qualified else "no_candidate_passed",
        "run_id": hist_contract["run_id"], "root": str(root.resolve()),
        "artifacts": artifacts,
        "full_pool_baseline": baseline_result,
        "agent_session": {"root": str((root / "development_agent").resolve()),
                          "fixed_results_exposed_to_agent": False},
        "stage_run_ids": {"development_baseline": development_contract["run_id"]},
        "dataset_provenance": {
            "dataset_id": "fresh-fixture",
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": dataset_digest,
            "database_path": str(root / "market_data.sqlite"),
            "database_sha256": database_sha256,
            "source_file_count": 0,
            "symbols": sorted(SYMBOLS),
            "spot_feature_missing_rows": 0,
            "source_causality": dataset_manifest["source_causality"],
            "source_causality_notes": dataset_manifest["source_causality_notes"],
            "historical_causality_certified": False,
            "repair_state": dataset_manifest["repair_state"],
            "excluded_documented_repair_rows": [],
            "funding_mark": dataset_manifest["funding_mark"],
            "primary_source_fallback": False,
        },
        "candidate_freeze": freeze, "internal_validation": validation,
        "candidate_freeze_sha256": freeze_digest,
        "selected_candidate": {"card_ids": list(selected_ids), "qualified_ab": qualified,
                               "qualified_c": qualified, "qualified": qualified},
        "selected_candidates": [list(selected_ids)], "trials": [trial], "candidates": [trial],
        "c_data_accessed_after_candidate_freeze": True,
        "c_results_returned_to_agent": False,
        "forward_readiness": {"ready_for_separate_forward_review": qualified,
                              "forward_validation_started": False,
                              "paper_started": False, "published": False},
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def _fixture(tmp_path, *, max_drawdown=0.15, selected_ids=None, mode="offline_event_replay"):
    start = pd.Timestamp("2026-01-01T03:00:00Z")
    seed_end = start - pd.Timedelta(hours=2)
    seed_index = pd.date_range(seed_end - pd.Timedelta(hours=2), periods=3, freq="h")
    panel_index = pd.MultiIndex.from_product(
        [seed_index, SYMBOLS], names=["timestamp", "symbol"]
    )
    values = pd.DataFrame(np.nan, index=panel_index, columns=INPUT_COLUMNS, dtype=float)
    availability = pd.DataFrame(index=panel_index, columns=sorted(REQUIRED_FIELDS), dtype=object)
    received = pd.Series(index=panel_index, dtype=object, name="received_at")
    frames = {}
    per_symbol = {symbol: index + 100.0 for index, symbol in enumerate(SYMBOLS)}
    for hour_position, timestamp in enumerate(seed_index):
        for index, symbol in enumerate(SYMBOLS):
            close = per_symbol[symbol] + hour_position * (index + 1) * 0.25
            values.loc[(timestamp, symbol), "perp_close"] = close
            values.loc[(timestamp, symbol), "mark_close"] = close
            values.loc[(timestamp, symbol), "spot_quote_volume"] = 1000.0 + index * 100 + hour_position
            values.loc[(timestamp, symbol), "spot_trades"] = 100 + index * 10 + hour_position
            ready = timestamp + pd.Timedelta(minutes=59, seconds=30)
            if mode == "forward":
                ready = timestamp + pd.Timedelta(hours=1)
            for field in REQUIRED_FIELDS:
                availability.loc[(timestamp, symbol), field] = ready
            received.loc[(timestamp, symbol)] = ready + pd.Timedelta(milliseconds=10)
    universe = pd.Series(True, index=panel_index, name="eligible", dtype=bool)
    panel = FactorInputPanel(values=values, universe=universe, diagnostics={"source": "offline fixture"})
    for index, symbol in enumerate(SYMBOLS):
        close = values.xs(symbol, level="symbol")["perp_close"]
        frames[symbol] = pd.DataFrame(
            {"open": close - 0.1, "close": close, "mark_close": close}, index=seed_index
        )
    seed = FreshSeedData(
        panel=panel,
        frames=frames,
        funding=pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"]),
        feature_available_at=availability,
        received_at=received,
        diagnostics={"source_causality": "fresh_collector_fixture"},
    )
    cards = [
        {
            "id": "f1",
            "title": "Perpetual close",
            "expression": "perp_close",
            "direction": 1,
            "fields": ["perp_close"],
            "lookback_hours": 0,
            "horizon_hours": 1,
            "card_snapshot": {"id": "f1", "formula": "perp_close"},
        },
        {
            "id": "f2",
            "title": "Close variant",
            "expression": "perp_close",
            "direction": -1,
            "fields": ["perp_close"],
            "lookback_hours": 0,
            "horizon_hours": 1,
            "card_snapshot": {"id": "f2", "formula": "perp_close", "direction": -1},
        },
        {
            "id": "f3",
            "title": "Spot activity",
            "expression": "add(spot_quote_volume, spot_trades)",
            "direction": 1,
            "fields": ["spot_quote_volume", "spot_trades"],
            "lookback_hours": 0,
            "horizon_hours": 1,
            "card_snapshot": {
                "id": "f3",
                "formula": "add(spot_quote_volume, spot_trades)",
                "direction": 1,
            },
        },
        {
            "id": "f4",
            "title": "Spot quote volume",
            "expression": "spot_quote_volume",
            "direction": 1,
            "fields": ["spot_quote_volume"],
            "lookback_hours": 0,
            "horizon_hours": 1,
            "card_snapshot": {"id": "f4", "formula": "spot_quote_volume", "direction": 1},
        },
    ]
    result = {
        "factors": [
            {"id": "f1", "code": "F1", "snapshot_path": "cards/f1.json"},
            {"id": "f2", "code": "F2", "snapshot_path": "cards/f2.json"},
            {"id": "f3", "code": "F3", "snapshot_path": "cards/f3.json"},
            {"id": "f4", "code": "F4", "snapshot_path": "cards/f4.json"},
        ]
    }
    snapshot = SimpleNamespace(cards=cards, result=result, records={})
    selected_ids = list(selected_ids or ["f1", "f4"])
    historical_result_path = _write_historical_qualification(
        tmp_path / "historical-result", snapshot, selected_ids
    )
    snapshot = _snapshot_for_history(snapshot, historical_result_path)
    session = freeze_forward_candidate(
        snapshot,
        historical_result_path,
        seed,
        tmp_path / "forward-session",
        start=start,
        mode=mode,
        max_drawdown=max_drawdown,
        observation_hours=720,
        minimum_rebalances=20,
    )
    return session, start, seed, snapshot, historical_result_path


def _feature_values(base):
    return {
        "perp_close": base,
        "mark_close": base,
        "spot_quote_volume": base * 10,
        "spot_trades": base,
    }


def _prices(*values):
    return dict(zip(SYMBOLS, values))


def _bar_close(session, start, timestamp, prices, *, late_by=pd.Timedelta(0),
               feature_delay=pd.Timedelta(0), live=False):
    close_time = timestamp + pd.Timedelta(hours=1)
    received_at = close_time - pd.Timedelta(milliseconds=100) + late_by
    outcomes = []
    for offset, symbol in enumerate(session.symbols):
        features = _feature_values(prices.get(symbol, 100.0 + offset * 3.0))
        feature_available = {
            field: close_time - pd.Timedelta(milliseconds=500) + (
                feature_delay if field in {"spot_quote_volume", "spot_trades"} else pd.Timedelta(0)
            )
            for field in REQUIRED_FIELDS
        }
        event = {
            "kind": "bar_close",
            "timestamp": timestamp,
            "symbol": symbol,
            "feature_values": features,
            "feature_available_at": feature_available,
        }
        if live:
            outcome = session.ingest_event(
                event, received_at=received_at + pd.Timedelta(milliseconds=offset)
            )
        else:
            outcome = session.replay_event(
                event, received_at=received_at + pd.Timedelta(milliseconds=offset)
            )
        outcomes.append(outcome)
    return outcomes


def _quote(session, execution_timestamp, symbol, price, *, quote_offset_ms=200, receive_offset_ms=300):
    quote_at = execution_timestamp + pd.Timedelta(milliseconds=quote_offset_ms)
    event = {
        "kind": "quote",
        "execution_timestamp": execution_timestamp,
        "symbol": symbol,
        "quote_at": quote_at,
        "price": price,
        "available_at": quote_at,
    }
    return session.replay_event(
        event,
        received_at=execution_timestamp + pd.Timedelta(milliseconds=receive_offset_ms),
    )


def test_freeze_requires_qualified_candidate_and_keeps_session_prepared(tmp_path):
    session, start, seed, snapshot, historical_result_path = _fixture(tmp_path)
    status = session.status()
    assert status["status"] == "prepared"
    assert status["forward_started"] is False
    assert status["paper_started"] is False
    assert status["live_bars"] == 0
    assert session.contract["observation"] == {
        "start": start.isoformat(),
        "end_exclusive": (start + pd.Timedelta(hours=720)).isoformat(),
        "duration_hours": 720,
        "minimum_rebalances": 20,
        "minimum_rebalances_basis": "complete scheduled signal decisions; independent of fill count",
    }
    assert session.contract["risk_stop"]["max_drawdown"] == 0.15
    assert {"spot_quote_volume", "spot_trades"}.issubset(session.required_fields)
    assert session.contract["execution_assumptions"]["reference"] == "first_observed_quote_after_signal_receipt"
    restored = ForwardSession.open(session.root)
    assert restored.state == session.state

    unqualified_history = _write_historical_qualification(
        tmp_path / "unqualified-history", snapshot, ["f1", "f4"], qualified=False
    )
    unqualified_snapshot = _snapshot_for_history(snapshot, unqualified_history)
    with pytest.raises(ValueError, match="no frozen qualified candidate"):
        freeze_forward_candidate(
            unqualified_snapshot,
            unqualified_history,
            seed,
            tmp_path / "unqualified",
            start=start,
            mode="offline_event_replay",
        )
    assert not (tmp_path / "unqualified").exists()

    freeze_path = historical_result_path.parent / "candidate-freeze.json"
    freeze_path.write_text(freeze_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch: candidate_freeze"):
        freeze_forward_candidate(
            snapshot,
            historical_result_path,
            seed,
            tmp_path / "tampered-history",
            start=start,
            mode="offline_event_replay",
        )
    assert not (tmp_path / "tampered-history").exists()

    failing_c_history = _write_historical_qualification(
        tmp_path / "failing-c-history", snapshot, ["f1", "f4"],
        qualified=True, c_stress_return=-0.01, validation_qualified=False,
    )
    failing_c_snapshot = _snapshot_for_history(snapshot, failing_c_history)
    with pytest.raises(ValueError, match="C validation artifact does not qualify"):
        freeze_forward_candidate(
            failing_c_snapshot,
            failing_c_history,
            seed,
            tmp_path / "failing-c",
            start=start,
            mode="offline_event_replay",
        )
    assert not (tmp_path / "failing-c").exists()


def test_qualified_full_pool_strategy_can_be_frozen(tmp_path):
    session, _start, _seed, _snapshot, _history = _fixture(
        tmp_path, selected_ids=["f1", "f2", "f3", "f4"]
    )
    assert session.contract["selected_card_ids"] == ["f1", "f2", "f3", "f4"]


def test_qualified_history_artifact_reference_cannot_escape_its_root(tmp_path):
    _session, start, seed, snapshot, historical_result_path = _fixture(tmp_path)
    result = json.loads(historical_result_path.read_text(encoding="utf-8"))
    result["artifacts"]["dataset_manifest"]["path"] = "../../outside/dataset_manifest.json"
    historical_result_path.write_text(json.dumps(result), encoding="utf-8")

    output = tmp_path / "escaped-history-forward"
    with pytest.raises(ValueError, match="historical artifact escapes its run root: dataset_manifest"):
        freeze_forward_candidate(
            snapshot,
            historical_result_path,
            seed,
            output,
            start=start,
            mode="offline_event_replay",
        )
    assert not output.exists()


def test_qualified_history_rejects_snapshot_with_changed_costs_or_portfolio(tmp_path):
    _session, start, seed, snapshot, historical_result_path = _fixture(tmp_path)
    changes = []
    changed_costs = snapshot.contract.as_dict()
    changed_costs["costs"]["fee_bps"] += 1.0
    changes.append(("costs", changed_costs))
    changed_portfolio = snapshot.contract.as_dict()
    changed_portfolio["portfolio"]["gross_exposure"] = 0.7
    changes.append(("portfolio", changed_portfolio))

    for name, contract_data in changes:
        output = tmp_path / f"changed-{name}-session"
        with pytest.raises(ValueError, match="development contract differs from the supplied baseline snapshot"):
            freeze_forward_candidate(
                _copy_snapshot(snapshot, contract_data=contract_data),
                historical_result_path,
                seed,
                output,
                start=start,
                mode="offline_event_replay",
            )
        assert not output.exists()

    wrong_baseline = dict(snapshot.result)
    wrong_baseline["root"] = str(tmp_path / "different-baseline-root")
    output = tmp_path / "changed-baseline-root-session"
    with pytest.raises(ValueError, match="supplied baseline snapshot result differs from the frozen historical baseline"):
        freeze_forward_candidate(
            _copy_snapshot(snapshot, result_data=wrong_baseline),
            historical_result_path,
            seed,
            output,
            start=start,
            mode="offline_event_replay",
        )
    assert not output.exists()


def test_qualified_history_rejects_changed_ten_symbol_pool(tmp_path):
    _session, start, seed, snapshot, historical_result_path = _fixture(tmp_path)
    changed_symbols = [*SYMBOLS[:-1], "ZZZUSDT"]
    output = tmp_path / "changed-pool-session"
    with pytest.raises(ValueError, match="supplied baseline snapshot market pool differs"):
        freeze_forward_candidate(
            _copy_snapshot(snapshot, frame_symbols=changed_symbols),
            historical_result_path,
            seed,
            output,
            start=start,
            mode="offline_event_replay",
        )
    assert not output.exists()


def test_qualified_history_rejects_seed_from_another_ten_symbol_pool(tmp_path):
    _session, start, seed, snapshot, historical_result_path = _fixture(tmp_path)
    old_symbol = SYMBOLS[-1]
    new_symbol = "ZZZUSDT"
    renamed_index = pd.MultiIndex.from_tuples(
        [(timestamp, new_symbol if symbol == old_symbol else symbol)
         for timestamp, symbol in seed.panel.values.index],
        names=["timestamp", "symbol"],
    )
    values = seed.panel.values.copy()
    values.index = renamed_index
    universe = seed.panel.universe.copy()
    universe.index = renamed_index
    availability = seed.feature_available_at.copy()
    availability.index = renamed_index
    received = seed.received_at.copy()
    received.index = renamed_index
    frames = dict(seed.frames)
    frames[new_symbol] = frames.pop(old_symbol)
    changed_seed = FreshSeedData(
        panel=FactorInputPanel(values=values, universe=universe, diagnostics=seed.panel.diagnostics),
        frames=frames,
        funding=seed.funding,
        feature_available_at=availability,
        received_at=received,
        diagnostics=seed.diagnostics,
    )
    output = tmp_path / "changed-seed-pool-session"
    with pytest.raises(ValueError, match="fresh seed symbol pool differs from the qualified historical universe"):
        freeze_forward_candidate(
            snapshot,
            historical_result_path,
            changed_seed,
            output,
            start=start,
            mode="offline_event_replay",
        )
    assert not output.exists()


def test_forward_mode_stays_prepared_without_explicit_activation(tmp_path):
    session, start, _seed, _snapshot, _history = _fixture(tmp_path, mode="forward")
    assert session.status()["status"] == "prepared"
    assert session.state["pending_orders"] == {}
    assert session.state["orders"] == []
    assert session.status()["forward_started"] is False
    assert session.status()["live_bars"] == 0
    assert session.state["seed_signal_timestamp"] == (start - pd.Timedelta(hours=2)).isoformat()
    with pytest.raises(ForwardSessionError, match="activation must follow seed receipt"):
        session.activate(start + pd.Timedelta(milliseconds=1))
    assert session.status()["forward_started"] is False
    assert session.state["event_sequence"] == 0
    with pytest.raises(ForwardSessionError, match="activate explicitly"):
        session.ingest_event(
            {"kind": "funding", "timestamp": start, "symbol": SYMBOLS[0],
             "funding_rate": 0.0, "mark_price": 100.0, "available_at": start},
            received_at=start,
        )

    signal_timestamp = start - pd.Timedelta(hours=1)
    close_time = start
    early_event = {
        "kind": "bar_close",
        "timestamp": signal_timestamp,
        "symbol": session.symbols[0],
        "feature_values": _feature_values(100.0),
        "feature_available_at": {field: close_time - pd.Timedelta(milliseconds=1)
                                  for field in REQUIRED_FIELDS},
    }
    with pytest.raises(ValueError, match="forward feature available_at predates its completed bar"):
        session._normalize_event(early_event, received_at=close_time + pd.Timedelta(milliseconds=100))
    late_event = dict(
        early_event,
        feature_available_at={field: close_time + pd.Timedelta(milliseconds=100)
                              for field in REQUIRED_FIELDS},
    )
    normalized, received, _logical_id, _fingerprint = session._normalize_event(
        late_event, received_at=close_time + pd.Timedelta(milliseconds=200)
    )
    assert received == close_time + pd.Timedelta(milliseconds=200)
    assert normalized["timestamp"] == signal_timestamp.isoformat()
    assert session.status()["forward_started"] is False


def test_offline_event_replay_cannot_activate_or_ingest_live_events(tmp_path):
    session, start, *_ = _fixture(tmp_path)
    with pytest.raises(ForwardSessionError, match="offline event-replay sessions cannot be activated"):
        session.activate(start)
    with pytest.raises(ForwardSessionError, match="offline event-replay sessions cannot ingest live events"):
        session.ingest_event(
            {"kind": "funding", "timestamp": start, "symbol": SYMBOLS[0],
             "funding_rate": 0.0, "mark_price": 100.0, "available_at": start},
            received_at=start,
        )
    assert session.status()["forward_started"] is False
    assert session.status()["live_bars"] == 0
    assert session.state["event_sequence"] == 0


def test_forward_seed_accepts_delayed_closed_bar_receipt_before_start(tmp_path):
    session, start, seed, snapshot, historical_result_path = _fixture(
        tmp_path, mode="forward"
    )
    seed_timestamp = start - pd.Timedelta(hours=2)
    for symbol in SYMBOLS:
        seed.feature_available_at.loc[(seed_timestamp, symbol), list(REQUIRED_FIELDS)] = (
            seed_timestamp + pd.Timedelta(hours=1, minutes=15)
        )
        seed.received_at.loc[(seed_timestamp, symbol)] = seed_timestamp + pd.Timedelta(
            hours=1, minutes=20
        )

    delayed_session = freeze_forward_candidate(
        snapshot,
        historical_result_path,
        seed,
        tmp_path / "delayed-seed-session",
        start=start,
        mode="forward",
    )
    assert delayed_session.status()["status"] == "prepared"
    assert delayed_session.status()["forward_started"] is False
    assert delayed_session.state["pending_orders"] == {}
    assert delayed_session.state["orders"] == []
    assert delayed_session.contract["seed"]["last_received_at"] == (
        seed_timestamp + pd.Timedelta(hours=1, minutes=20)
    ).isoformat()

    for symbol in SYMBOLS:
        seed.feature_available_at.loc[(seed_timestamp, symbol), list(REQUIRED_FIELDS)] = start
        seed.received_at.loc[(seed_timestamp, symbol)] = start
    boundary_session = freeze_forward_candidate(
        snapshot,
        historical_result_path,
        seed,
        tmp_path / "boundary-seed-session",
        start=start,
        mode="forward",
    )
    assert boundary_session.contract["seed"]["last_received_at"] == start.isoformat()
    assert boundary_session.state["pending_orders"] == {}
    assert boundary_session.state["orders"] == []
    assert boundary_session.status()["forward_started"] is False

    for symbol in SYMBOLS:
        seed.feature_available_at.loc[(seed_timestamp, symbol), list(REQUIRED_FIELDS)] = start + pd.Timedelta(milliseconds=1)
        seed.received_at.loc[(seed_timestamp, symbol)] = start + pd.Timedelta(milliseconds=1)
    stale_root = tmp_path / "stale-seed-session"
    with pytest.raises(ValueError, match="fresh seed rows must be received by the frozen start"):
        freeze_forward_candidate(
            snapshot,
            historical_result_path,
            seed,
            stale_root,
            start=start,
            mode="forward",
        )
    assert not stale_root.exists()


def test_forward_quote_is_only_used_after_received_signal_and_restart_is_identical(tmp_path):
    session, start, _seed, _snapshot, _candidate = _fixture(tmp_path)

    execution_timestamp = start
    before_fills = len(session.state["fills"])
    # This quote arrives before the first post-seed closed feature snapshot and
    # cannot become a historical fill for an order that does not exist yet.
    _quote(session, execution_timestamp, session.symbols[0], 90.0,
           quote_offset_ms=50, receive_offset_ms=60)
    close_outcomes = _bar_close(
        session,
        start,
        start - pd.Timedelta(hours=1),
        _prices(105.0, 112.0, 108.0),
        late_by=pd.Timedelta(milliseconds=300),
        feature_delay=pd.Timedelta(milliseconds=600),
    )
    assert close_outcomes[-1] == "signal_prepared"
    pending = session.state["pending_orders"][execution_timestamp.isoformat()]
    frozen_quantities = {symbol: pending[symbol]["target_quantity"] for symbol in session.symbols}
    assert len(session.state["fills"]) == before_fills

    # First quote after receipt executes at its observed price; order quantity
    # remains the signal-close quantity even though this quote is far different.
    for symbol in session.symbols:
        _quote(session, execution_timestamp, symbol, 1_000.0 + session.symbols.index(symbol),
               quote_offset_ms=500, receive_offset_ms=600)
    new_fills = session.state["fills"][before_fills:]
    for fill in new_fills:
        assert fill["quote_at"] > pending[fill["symbol"]]["signal_received_at"]
        assert fill["scheduled_open"] == execution_timestamp.isoformat()
        assert np.isclose(
            abs(fill["signed_quantity"]),
            abs(frozen_quantities[fill["symbol"]] - pending[fill["symbol"]]["current_quantity"]),
        )
        assert fill["quote_lag_seconds"] > 0.0

    # A repeated event is a no-op; the same logical quote ID with changed price fails.
    prior_state = session.state
    duplicate = {
        "kind": "quote",
        "execution_timestamp": execution_timestamp,
        "symbol": session.symbols[0],
        "quote_at": execution_timestamp + pd.Timedelta(milliseconds=500),
        "price": 1_000.0,
        "available_at": execution_timestamp + pd.Timedelta(milliseconds=500),
    }
    assert session.replay_event(duplicate, received_at=execution_timestamp + pd.Timedelta(milliseconds=700)) == "duplicate"
    assert session.state == prior_state
    with pytest.raises(ForwardSessionError, match="conflicting duplicate"):
        session.replay_event(dict(duplicate, price=1_001.0), received_at=execution_timestamp + pd.Timedelta(milliseconds=700))

    resumed = ForwardSession.open(session.root)
    assert resumed.state == session.state
    assert resumed.status()["forward_started"] is False
    assert resumed.status()["live_bars"] == 0
    assert resumed.status()["replay_bars"] == 1


def test_late_funding_is_audited_but_cannot_rewrite_previous_ledger(tmp_path):
    session, start, _seed, _snapshot, _candidate = _fixture(tmp_path)
    _bar_close(
        session,
        start,
        start - pd.Timedelta(hours=1),
        _prices(100.0, 110.0, 120.0),
        late_by=pd.Timedelta(milliseconds=300),
        feature_delay=pd.Timedelta(milliseconds=600),
    )
    for symbol in session.symbols:
        _quote(session, start, symbol, 100.0 + session.symbols.index(symbol),
               quote_offset_ms=400, receive_offset_ms=500)
    _bar_close(
        session,
        start,
        start,
        _prices(103.0, 111.0, 109.0),
    )
    ledger_before = [dict(row) for row in session.state["ledger"]]
    cash_before = session.state["account"]["cash"]
    funding_time = start + pd.Timedelta(minutes=30)
    event = {
        "kind": "funding",
        "timestamp": funding_time,
        "symbol": SYMBOLS[0],
        "funding_rate": 0.01,
        "mark_price": 103.0,
        "available_at": funding_time,
    }

    assert session.replay_event(event, received_at=start + pd.Timedelta(hours=1, seconds=1)) == "late_unapplied"
    assert session.state["ledger"] == ledger_before
    assert session.state["account"]["cash"] == cash_before
    assert session.state["funding_events"][-1]["applied"] is False
    assert session.status()["status"] == "halted_late_funding"
    assert ForwardSession.open(session.root).state == session.state


def test_missing_quote_expires_order_without_historical_open_fill(tmp_path):
    session, start, _seed, _snapshot, _candidate = _fixture(tmp_path)
    close_time = start + pd.Timedelta(hours=1)
    _bar_close(
        session,
        start,
        start - pd.Timedelta(hours=1),
        _prices(101.0, 111.0, 119.0),
        late_by=pd.Timedelta(milliseconds=300),
        feature_delay=pd.Timedelta(milliseconds=600),
    )
    _bar_close(
        session,
        start,
        start,
        _prices(102.0, 112.0, 118.0),
        late_by=pd.Timedelta(milliseconds=300),
        feature_delay=pd.Timedelta(milliseconds=600),
    )
    assert session.state["fills"] == []
    assert session.state["missed_executions"]
    assert session.state["missed_executions"][0]["execution_timestamp"] == start.isoformat()
    assert session.state["ledger"][-1]["market_timestamp"] == close_time.isoformat()
    assert all(quantity == 0.0 for quantity in session.state["account"]["quantities"].values())


def test_max_drawdown_stop_preserves_positions_without_forced_liquidation(tmp_path):
    session, start, _seed, _snapshot, _candidate = _fixture(tmp_path, max_drawdown=0.01)
    _bar_close(
        session, start, start - pd.Timedelta(hours=1),
        _prices(100.0, 110.0, 120.0),
        late_by=pd.Timedelta(milliseconds=300),
        feature_delay=pd.Timedelta(milliseconds=600),
    )
    for index, symbol in enumerate(session.symbols):
        _quote(session, start, symbol, 100.0 + index * 10.0,
               quote_offset_ms=400, receive_offset_ms=500)
    _bar_close(
        session,
        start,
        start,
        _prices(110.0, 110.0, 108.0),
    )

    assert session.status()["status"] == "halted_max_drawdown"
    assert session.state["account"]["drawdown"] > 0.01
    assert any(quantity != 0.0 for quantity in session.state["account"]["quantities"].values())
    assert len(session.state["fills"]) == 4
