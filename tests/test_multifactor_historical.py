import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from crypto_quant.research.strategy_research.multifactor_historical import (
    APPROVED_CARD_IDS,
    CANDIDATE_SELECTION,
    HistoricalContract,
    _candidate_failures,
    _select_ab_candidate,
    _stage_contract,
    _source_manifest_gate,
    _run_validation_stage,
    run_historical,
)


MODULE = "crypto_quant.research.strategy_research.multifactor_historical"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"]


def _contract(cards, universe="universe.csv", dataset_manifest="dataset/manifest.json", **changes):
    value = {
        "schema_version": 1,
        "run_id": "historical-test-001",
        "dataset_manifest": dataset_manifest,
        "cards": list(cards),
        "universe": universe,
        "horizon_hours": 24,
        "warmup_hours": 168,
        "development_start": "2022-08-01T00:00:00Z",
        "validation_start": "2025-08-01T00:00:00Z",
        "validation_end": "2026-08-01T00:00:00Z",
        "prior_data_use": "Historical A+B are development; C is previously exposed internal validation.",
        "data_processing": "Raw hourly archives only; funding marks use a declared completed 1m proxy.",
        "costs": {"initial_capital": 10000.0, "fee_bps": 10.0,
                  "slippage_bps": 5.0, "stress_multiplier": 2.0},
        "portfolio": {"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                      "max_asset_weight": 0.2, "rebalance_hours": 24, "margin_fraction": 0.1},
        "experiment_budget": 2,
        "context_bytes": 256000,
        "qualification_gates": {"min_net_return": 0.0, "max_drawdown": 0.15,
                                 "min_traded_bars": 1, "min_stress_return": 0.0},
        "candidate_selection": CANDIDATE_SELECTION,
    }
    value.update(changes)
    return value


class HistoricalContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cards = [str(self.root / f"{card_id}.json") for card_id in APPROVED_CARD_IDS]

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_frozen_contract_and_input_warmup(self):
        contract = HistoricalContract.from_dict(_contract(self.cards))
        self.assertEqual(contract.experiment_budget, 2)
        self.assertEqual(contract.context_bytes, 256000)
        self.assertEqual(contract.input_start.isoformat(), "2022-07-24T23:00:00+00:00")
        self.assertEqual(contract.as_dict()["candidate_selection"], CANDIDATE_SELECTION)

        changes = (
            {"horizon_hours": 4},
            {"warmup_hours": 167},
            {"experiment_budget": 3},
            {"context_bytes": 256001},
            {"costs": {"initial_capital": 10000.0, "fee_bps": 11.0,
                        "slippage_bps": 5.0, "stress_multiplier": 2.0}},
            {"portfolio": {"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                            "max_asset_weight": 0.2, "rebalance_hours": 12, "margin_fraction": 0.1}},
            {"qualification_gates": {"min_net_return": -0.01, "max_drawdown": 0.15,
                                      "min_traded_bars": 1, "min_stress_return": 0.0}},
            {"candidate_selection": "choose_after_C"},
            {"unknown_switch": True},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                HistoricalContract.from_dict(_contract(self.cards, **change))

        with self.assertRaises(ValueError):
            HistoricalContract.from_dict(_contract(self.cards[:3]))


class SharedDatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "isolated_dataset"
        self.dataset.mkdir()
        self.db = self.dataset / "market_data.sqlite"
        self.db.write_bytes(b"isolated bundle")
        self.manifest_path = self.dataset / "manifest.json"
        self.manifest_path.write_text("{}", encoding="utf-8")
        self.contract = HistoricalContract.from_dict(_contract(
            [str(self.root / f"{card_id}.json") for card_id in APPROVED_CARD_IDS],
        ))
        self.cards = [{"id": card_id} for card_id in APPROVED_CARD_IDS]
        self.universe = self.root / "universe.csv"
        self.universe.write_text("timestamp,symbol,eligible\n", encoding="utf-8")
        self.contracts_dir = self.root / "contracts"
        self.contracts_dir.mkdir()
        self.stage_contract = _stage_contract(
            self.contract, stage="development",
            card_paths=[Path(path) for path in self.contract.cards],
            universe_path=self.universe,
            manifest_path=self.manifest_path,
            stage_contract_dir=self.contracts_dir,
        )
        self.document = {
            "dataset_id": "multifactor_raw_20261002_v1",
            "database_signature": {"sha256": "a" * 64},
            "source_files": [{"source_kind": "official_public_download"}],
            "symbols": SYMBOLS,
            "spot_feature_missing_rows": 7,
            "source_causality": "raw_archives_with_declared_funding_proxy",
            "source_causality_notes": "Archive receipt timestamps are unavailable.",
            "historical_causality_certified": False,
            "repair_state": {"primary_source_fallback": False},
            "excluded_documented_repair_rows": [],
            "funding_mark": {"proxy_count": 1},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_historical_gate_delegates_to_shared_strict_manifest_parser(self):
        from crypto_quant.research.strategy_research.multifactor_data import ResearchDatasetManifest

        dataset = SimpleNamespace(document=self.document, path=self.manifest_path, database_path=self.db)
        with patch.object(ResearchDatasetManifest, "from_path", return_value=dataset) as shared_gate:
            manifest, database_path, provenance = _source_manifest_gate(
                self.manifest_path, self.stage_contract, self.contracts_dir,
            )
        parsed_contract = shared_gate.call_args.args[1]
        self.assertEqual(parsed_contract.purpose, "research")
        self.assertEqual(parsed_contract.stage, "development")
        self.assertEqual(parsed_contract.bounds[0].isoformat(), "2022-08-01T00:00:00+00:00")
        self.assertEqual(manifest, self.document)
        self.assertEqual(database_path, self.db)
        self.assertEqual(provenance["primary_source_fallback"], False)
        self.assertEqual(provenance["spot_feature_missing_rows"], 7)
        self.assertEqual(shared_gate.call_args.args[0], self.manifest_path)
        self.assertEqual(shared_gate.call_args.args[2], self.contracts_dir)

    def test_shared_source_failure_is_not_replaced_with_a_database_fallback(self):
        from crypto_quant.research.strategy_research.multifactor_data import ResearchDatasetManifest

        with patch.object(ResearchDatasetManifest, "from_path", side_effect=ValueError("bad archive hash")):
            with self.assertRaisesRegex(ValueError, "bad archive hash"):
                _source_manifest_gate(self.manifest_path, self.stage_contract, self.contracts_dir)

        primary_database = Path(__file__).resolve().parents[1] / "market_data" / "crypto_quant.sqlite"
        dataset = SimpleNamespace(document=self.document, path=self.manifest_path,
                                  database_path=primary_database)
        with patch.object(ResearchDatasetManifest, "from_path", return_value=dataset):
            with self.assertRaisesRegex(ValueError, "primary database"):
                _source_manifest_gate(self.manifest_path, self.stage_contract, self.contracts_dir)


class CandidateQualificationTests(unittest.TestCase):
    def setUp(self):
        self.cards = [f"/tmp/{card_id}.json" for card_id in APPROVED_CARD_IDS]
        self.contract = HistoricalContract.from_dict(_contract(self.cards))

    def test_gates_and_predeclared_candidate_tie_breaks(self):
        metrics = {"net_return": 0.0, "max_drawdown": -0.15}
        stress = {"net_return": 0.0}
        self.assertEqual(_candidate_failures(metrics, stress, 1, self.contract.qualification_gates), [])
        self.assertEqual(_candidate_failures({**metrics, "max_drawdown": -0.1501}, stress, 1,
                                             self.contract.qualification_gates), ["max_drawdown"])

        failing_best = {"trial_id": "agent-1", "selected_card_ids": ["b", "a"],
                        "arm": "agent", "source_trial_path": "agent_arm/1", "traded_bars": 2,
                        "included_factors": ["F1", "F2"],
                        "metrics": {"net_return": 0.20, "max_drawdown": -0.30},
                        "stress_costs": {"metrics": {"net_return": -0.2}}, "qualified_ab": False,
                        "qualification_failures": ["max_drawdown"]}
        passing_tie_a = {"trial_id": "fixed-1", "selected_card_ids": ["a", "c"],
                         "arm": "fixed", "source_trial_path": "fixed_arm/1", "traded_bars": 2,
                         "included_factors": ["F1", "F3"],
                         "metrics": {"net_return": 0.10, "max_drawdown": -0.10},
                         "stress_costs": {"metrics": {"net_return": 0.01}}, "qualified_ab": True,
                         "qualification_failures": []}
        passing_tie_b = {"trial_id": "agent-2", "selected_card_ids": ["a", "b"],
                         "arm": "agent", "source_trial_path": "agent_arm/2", "traded_bars": 2,
                         "included_factors": ["F1", "F2"],
                         "metrics": {"net_return": 0.10, "max_drawdown": -0.10},
                         "stress_costs": {"metrics": {"net_return": 0.01}}, "qualified_ab": True,
                         "qualification_failures": []}
        selected = _select_ab_candidate([failing_best, passing_tie_a, passing_tie_b])
        self.assertEqual(selected["selection_rule"], CANDIDATE_SELECTION)
        self.assertEqual(selected["selected_trial_id"], "agent-1")
        self.assertEqual(selected["selection_mode"], "best_ab_diagnostic_only")
        self.assertFalse(selected["selected_ab_qualified"])
        tied_qualified = _select_ab_candidate([passing_tie_a, passing_tie_b])
        self.assertEqual(tied_qualified["selected_trial_id"], "agent-2")

        full_pool = {"trial_id": "full-pool-equal-weight", "arm": "full_pool_equal_weight",
                     "selected_card_ids": list(APPROVED_CARD_IDS), "included_factors": ["F1", "F2", "F3", "F4"],
                     "source_trial_path": "experiments/equal_weight", "traded_bars": 20,
                     "metrics": {"net_return": 0.30, "max_drawdown": -0.05},
                     "stress_costs": {"metrics": {"net_return": 0.1}},
                     "qualified_ab": True, "qualification_failures": [], "experiment_budget_charge": 0}
        selected_with_control = _select_ab_candidate([passing_tie_a, full_pool])
        self.assertEqual(selected_with_control["selected_trial_id"], "full-pool-equal-weight")

    def test_no_ab_pass_chooses_best_for_c_diagnostics_without_marking_qualified(self):
        candidates = [
            {"trial_id": "agent-1", "selected_card_ids": ["c", "d"],
             "arm": "agent", "source_trial_path": "agent_arm/1", "traded_bars": 2,
             "included_factors": ["F3", "F4"],
             "metrics": {"net_return": 0.03, "max_drawdown": -0.2},
             "stress_costs": {"metrics": {"net_return": -0.1}}, "qualified_ab": False,
             "qualification_failures": ["max_drawdown"]},
            {"trial_id": "fixed-1", "selected_card_ids": ["a", "d"],
             "arm": "fixed", "source_trial_path": "fixed_arm/1", "traded_bars": 2,
             "included_factors": ["F1", "F4"],
             "metrics": {"net_return": -0.01, "max_drawdown": -0.04},
             "stress_costs": {"metrics": {"net_return": -0.2}}, "qualified_ab": False,
             "qualification_failures": ["min_net_return"]},
        ]
        selected = _select_ab_candidate(candidates)
        self.assertEqual(selected["selected_trial_id"], "agent-1")
        self.assertEqual(selected["selection_mode"], "best_ab_diagnostic_only")
        self.assertFalse(selected["selected_ab_qualified"])


class HistoricalOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "dataset").mkdir()
        self.db = self.root / "dataset/market_data.sqlite"
        self.db.write_bytes(b"independent source fixture")
        self.manifest_path = self.root / "dataset/manifest.json"
        self.cards_paths = []
        for card_id in APPROVED_CARD_IDS:
            path = self.root / f"{card_id}.json"
            path.write_text("{}", encoding="utf-8")
            self.cards_paths.append(str(path))
        self.universe = self.root / "universe.csv"
        self.universe.write_text("timestamp,symbol,eligible\n", encoding="utf-8")
        self.manifest_path.write_text("{}", encoding="utf-8")
        self.contract_path = self.root / "historical.json"
        self.contract_path.write_text(json.dumps(_contract(self.cards_paths, universe=str(self.universe),
                                                           dataset_manifest=str(self.manifest_path))),
                                      encoding="utf-8")
        self.provenance = {"dataset_id": "multifactor_raw_20261002_v1", "manifest_sha256": "a" * 64,
                           "database_sha256": "b" * 64, "symbols": SYMBOLS,
                           "source_causality": "raw_archives_with_declared_funding_proxy",
                           "historical_causality_certified": False}

    def tearDown(self):
        self.temp.cleanup()

    def test_freezes_ab_selection_before_c_loader_and_returns_no_forward_status(self):
        events = []
        baseline_root = self.root / "runs" / "historical-test-001" / "development_full_pool" / "ab-run"
        cards = [{"id": card_id, "title": f"Factor {index}"}
                 for index, card_id in enumerate(APPROVED_CARD_IDS, start=1)]
        baseline = {
            "status": "engineering_complete",
            "root": str(baseline_root),
            "run_id": "ab-run",
            "experiments": [{"name": "equal_weight", "included_factors": ["F1", "F2", "F3", "F4"],
                             "metrics": {"net_return": 0.02, "max_drawdown": -0.04,
                                         "total_fees": 1.0, "total_slippage_cost": 1.0,
                                         "total_funding": -1.0}}],
        }
        trials = [
            {"trial_id": "agent-1", "arm": "agent", "selected_card_ids": [APPROVED_CARD_IDS[0], APPROVED_CARD_IDS[1]],
             "included_factors": ["F1", "F2"], "traded_bars": 2,
             "metrics": {"net_return": 0.05, "max_drawdown": -0.05},
             "stress_costs": {"metrics": {"net_return": 0.01}}, "qualified_ab": True,
             "qualification_failures": [], "source_trial_path": "agent_arm/experiment-001", "source_artifacts": {}},
            {"trial_id": "fixed-1", "arm": "fixed", "selected_card_ids": [APPROVED_CARD_IDS[1], APPROVED_CARD_IDS[2]],
             "included_factors": ["F2", "F3"], "traded_bars": 2,
             "metrics": {"net_return": -0.01, "max_drawdown": -0.08},
             "stress_costs": {"metrics": {"net_return": -0.02}}, "qualified_ab": False,
             "qualification_failures": ["min_net_return"], "source_trial_path": "fixed_arm/experiment-001", "source_artifacts": {}},
        ]

        def fake_baseline(stage_contract_path, stage_output):
            events.append("ab")
            baseline_root.mkdir(parents=True)
            (baseline_root / "result.json").write_text(json.dumps(baseline), encoding="utf-8")
            return baseline

        def fake_agent(baseline_path, session_contract, session_output, model, *, model_mode, model_settings):
            events.append("agent")
            session_root = Path(session_output) / session_contract.run_id
            session_root.mkdir(parents=True)
            result = {"status": "engineering_complete", "root": str(session_root), "model_mode": model_mode,
                      "model_calls": 5, "sealed_initial_drop_results": True,
                      "fixed_results_exposed_to_agent": False, "independent_agent_value_established": False,
                      "paper_started": False, "forward_validation_started": False, "published": False,
                      "agent_experiments": [{}], "fixed_experiments": [{}, {}]}
            (session_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (session_root / "report.md").write_text("fixture", encoding="utf-8")
            return result

        def fake_c(stage_contract_path, stage_contract_data, manifest_path, cards_path, universe_path,
                   selected_card_ids, output_root, qualification_gates, candidate_freeze_sha256):
            events.append("c")
            freeze_path = Path(output_root).parent / "candidate-freeze.json"
            self.assertTrue(freeze_path.is_file())
            frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
            self.assertTrue(frozen["selected_before_c_access"])
            self.assertEqual(selected_card_ids, [APPROVED_CARD_IDS[0], APPROVED_CARD_IDS[1]])
            result = {
                "stage": "C_internal_validation",
                "selected_candidate_card_ids": selected_card_ids,
                "candidate_freeze_sha256": candidate_freeze_sha256,
                "candidate": {"metrics": {"net_return": 0.03, "max_drawdown": -0.05}},
                "equal_weight_control": {"metrics": {"net_return": -0.02, "max_drawdown": -0.1}},
                "candidate_stress_costs": {"metrics": {"net_return": 0.01}},
                "traded_bars": 3,
                "reference_equals_candidate": False,
                "candidate_artifact_ref": "experiments/selected_candidate",
                "equal_weight_artifact_ref": "experiments/equal_weight_control",
                "qualification_failures": [],
                "qualification_gates": qualification_gates,
                "qualified_c": True,
            }
            Path(output_root).mkdir(parents=True)
            (Path(output_root) / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return result

        with patch(f"{MODULE}.read_cards", return_value=cards), \
                patch(f"{MODULE}._source_manifest_gate",
                      return_value=({"symbols": SYMBOLS}, self.db, self.provenance)), \
                patch(f"{MODULE}._validate_universe", return_value=None), \
                patch(f"{MODULE}._run_development_baseline", side_effect=fake_baseline), \
                patch(f"{MODULE}.load_baseline_snapshot", return_value=SimpleNamespace(
                    contract=SimpleNamespace(run_id="ab-run"))), \
                patch(f"{MODULE}._collect_ab_candidates", return_value=trials), \
                patch(f"{MODULE}._run_validation_stage", side_effect=fake_c):
            result = run_historical(self.contract_path, self.root / "runs", object(),
                                    model_mode="replay", model_settings={}, agent_runner=fake_agent)

        self.assertEqual(events, ["ab", "agent", "c"])
        self.assertEqual(result["status"], "qualified_candidate_frozen")
        self.assertTrue(result["forward_readiness"]["ready_for_separate_forward_review"])
        self.assertFalse(result["forward_readiness"]["forward_validation_started"])
        self.assertFalse(result["c_results_returned_to_agent"])
        self.assertEqual(result["selected_candidate"]["card_ids"],
                         [APPROVED_CARD_IDS[0], APPROVED_CARD_IDS[1]])
        self.assertEqual(result["internal_validation"]["candidate_freeze_sha256"],
                         result["candidate_freeze_sha256"])
        self.assertEqual(result["selected_candidate"]["c_traded_bars"], 3)
        self.assertEqual(result["stage_run_ids"]["historical"], "historical-test-001")
        for key, artifact in result["artifacts"].items():
            artifact_path = Path(result["root"]) / artifact["path"]
            with self.subTest(artifact=key):
                self.assertTrue(artifact_path.is_file())
                self.assertEqual(artifact["sha256"], hashlib.sha256(artifact_path.read_bytes()).hexdigest())
        stage_contract = json.loads((Path(result["root"]) / "contracts/internal_validation.json").read_text())
        self.assertEqual(stage_contract["schema_version"], 2)
        self.assertEqual(stage_contract["purpose"], "research")
        self.assertFalse((Path(result["root"]) / "forward").exists())


class ValidationReferenceTests(unittest.TestCase):
    def test_full_pool_candidate_reuses_the_equal_control_and_stresses_once(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        cards = [str(root / f"{card_id}.json") for card_id in APPROVED_CARD_IDS]
        for card_path in cards:
            Path(card_path).write_text("{}", encoding="utf-8")
        historical = HistoricalContract.from_dict(_contract(
            cards, universe=str(root / "universe.csv"), dataset_manifest=str(root / "dataset/manifest.json"),
        ))
        contract_dir = root / "contracts"
        contract_dir.mkdir()
        manifest_path = root / "dataset" / "manifest.json"
        manifest_path.parent.mkdir()
        manifest_path.write_text("{}", encoding="utf-8")
        contract_data = _stage_contract(historical, stage="internal_validation",
                                        card_paths=[Path(p) for p in cards],
                                        universe_path=root / "universe.csv",
                                        manifest_path=manifest_path, stage_contract_dir=contract_dir)
        contract_path = contract_dir / "validation.json"
        contract_path.write_text(json.dumps(contract_data), encoding="utf-8")

        start = pd.Timestamp("2025-08-01T00:00:00Z")
        symbols = ["BTCUSDT", "ETHUSDT"]
        index = pd.MultiIndex.from_product(
            [pd.date_range(start - pd.Timedelta(hours=1), start + pd.Timedelta(hours=2), freq="h"), symbols],
            names=["timestamp", "symbol"],
        )
        factors = SimpleNamespace(
            standardized=pd.DataFrame(1.0, index=index, columns=APPROVED_CARD_IDS),
            common_mask=pd.Series(True, index=index),
            eligible=pd.Series(True, index=index),
        )
        inputs = SimpleNamespace(panel=object(), frames={}, funding=pd.DataFrame(),
                                 universe=pd.Series(dtype=bool), diagnostics={})
        cards_data = [{"id": card_id, "title": card_id} for card_id in APPROVED_CARD_IDS]
        account_calls, stress_calls = [], []
        metrics = {"net_return": 0.01, "max_drawdown": -0.05, "sharpe_ratio": 1.0,
                   "total_turnover": 1.0, "total_fees": 1.0, "total_slippage_cost": 1.0,
                   "total_funding": -0.1}

        def fake_run_one(name, kind, included, score, run_inputs, run_factors,
                         stage_contract, directory, sample, factor_codes):
            account_calls.append((name, tuple(included)))
            Path(directory).mkdir(parents=True)
            pd.DataFrame({"trade_notional": [100.0, 0.0]}).to_csv(Path(directory) / "ledger.csv", index=False)
            return {"name": name, "kind": kind, "included_factors": [factor_codes[item] for item in included],
                    "metrics": dict(metrics), "path": "experiments/equal_weight_control"}

        def fake_stress(experiment, score, run_inputs, run_factors, stage_contract, directory):
            stress_calls.append(experiment["name"])
            return {"multiplier": 2.0, "metrics": dict(metrics), "path": "experiments/stress"}

        output = root / "validation"
        with patch(f"{MODULE}.read_cards", return_value=cards_data), \
                patch("crypto_quant.research.strategy_research.multifactor_data.load_research_inputs",
                      return_value=inputs, create=True), \
                patch(f"{MODULE}.process_factors", return_value=factors), \
                patch(f"{MODULE}._snapshot_inputs"), \
                patch(f"{MODULE}._snapshot_factor_panels"), \
                patch(f"{MODULE}._run_one", side_effect=fake_run_one), \
                patch(f"{MODULE}._run_stress", side_effect=fake_stress):
            result = _run_validation_stage(contract_path, contract_data, manifest_path,
                                           [Path(path) for path in cards], root / "universe.csv",
                                           list(APPROVED_CARD_IDS), output,
                                           historical.qualification_gates, "a" * 64)

        self.assertEqual(account_calls, [("equal_weight_control", APPROVED_CARD_IDS)])
        self.assertEqual(stress_calls, ["equal_weight_control"])
        self.assertTrue(result["reference_equals_candidate"])
        self.assertEqual(result["candidate_artifact_ref"], result["equal_weight_artifact_ref"])
        self.assertTrue(result["qualified_c"])


if __name__ == "__main__":
    unittest.main()
