"""Real SQLite -> existing backtester -> recorded role loop; no network."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_quant.research.strategy_research.cli import ReplayModel
from crypto_quant.research.strategy_research.contracts import StrategyResearchContract
from crypto_quant.research.strategy_research.engine import load_segment, evaluate
from crypto_quant.research.strategy_research.workflow import StrategyResearch, latest_result, latest_validation_result, validate_run
from crypto_quant.strategies.relative_strength import run_relative_strength_backtest
from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.research.strategy_research.rule_strategy import generate_rule_targets
from crypto_quant.research.strategy_research.engine import execution_plan
from crypto_quant.research.factor_mining.contracts import digest
from crypto_quant.research.factor_mining.model import ModelReply

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/strategy_research"


class ResearchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "market.sqlite"
        raw = json.loads((EXAMPLES / "contract.json").read_text())
        raw.update(development_start="2026-01-05T00:00:00Z", validation_start="2026-01-08T00:00:00Z",
                   validation_end="2026-01-10T00:00:00Z")
        self.contract = StrategyResearchContract.from_dict(raw)
        self.idea = json.loads((EXAMPLES / "idea.json").read_text())
        self.replies = json.loads((EXAMPLES / "replay.json").read_text())
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE klines (symbol TEXT, interval TEXT, open_time INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL, close_time INTEGER,
                quote_volume REAL, trades INTEGER, taker_buy_base_volume REAL, taker_buy_quote_volume REAL)""")
            for i, t in enumerate(pd.date_range("2026-01-01", "2026-01-10", freq="h", tz="UTC")):
                for j, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
                    close = 100 * np.exp(.001 * i + .08 * np.sin(i / 10 + j * 2))
                    op = close * .999
                    ms = t.value // 1000000
                    conn.execute("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (symbol, "1h", ms, op, close * 1.002, op * .998, close, 100000,
                                  ms + 3599999, 10000000, 10, 50000, 5000000))

    def runner(self, replies=None, name="run"):
        return StrategyResearch(self.contract, self.idea, ReplayModel(replies or self.replies), self.db,
                                self.root / name, model_mode="replay")

    def rule_contract(self):
        raw = self.contract.as_dict()
        raw["task"]["strategy_family"] = "factor_rule_spot"
        raw["parameter_space"] = {"warmup_hours": 72}
        return StrategyResearchContract.from_dict(raw)

    def test_factor_combination_and_breakout_complete_development_and_validation(self):
        self.contract = self.rule_contract()
        for name in ("multi-factor", "breakout"):
            with self.subTest(strategy=name):
                replies = json.loads((EXAMPLES / "factor_rules" / f"{name}.replay.json").read_text())
                runner = self.runner(replies, name)
                self.assertEqual(runner.run()["status"], "retained_for_validation")
                self.assertFalse((runner.root / "validation-access.json").exists())
                targets = pd.read_csv(runner.root / "v0001/strategy-targets.csv", index_col=0)
                self.assertGreater(targets.sum().sum(), 0)
                self.assertLessEqual(targets.sum(axis=1).max(), replies[0]["response"]["parameters"]["allocation"]["gross_exposure"])
                self.assertTrue((runner.root / "v0001/rule-values.csv").exists())
                executions = pd.read_csv(runner.root / "v0001/strategy-executions.csv", index_col=0, parse_dates=[0])
                elapsed = (executions.index - pd.Timestamp(self.contract.development_start)) / pd.Timedelta(hours=1)
                interval = replies[0]["response"]["parameters"]["rebalance_hours"]
                self.assertTrue((executions.loc[elapsed % interval != 0, "trade_notional"] == 0).all())
                self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")
                request = json.loads(json.loads((runner.root / "model_calls/call-0001.request.json").read_text())[1]["content"])
                self.assertIn("strategy_definition_format", request)
                self.assertIn("allocation", request["modification_components"])

    def test_rule_targets_entry_exit_position_caps_and_causality(self):
        contract = self.rule_contract()
        frames = load_segment(self.db, contract, "development")
        start = pd.Timestamp(contract.development_start)
        index = frames["BTCUSDT"].index
        definition = {"signals": [], "score": "div(spot_close,spot_open)", "entry": "sub(div(spot_close,spot_open),1)",
                      "exit": "sub(1,div(spot_close,spot_open))", "allocation": {
                          "max_positions": 2, "gross_exposure": 0.8, "max_asset_weight": 0.3}, "rebalance_hours": 1}
        for symbol, frame in frames.items():
            frame["open"] = 100.0
            frame["close"] = 120.0 if symbol == "BTCUSDT" else 110.0
        frames["BTCUSDT"].loc[start, "close"] = 90.0
        targets, _ = generate_rule_targets(frames, definition, 72, start)
        self.assertEqual(targets["BTCUSDT"].loc[start - pd.Timedelta(hours=1)], 0.3)
        self.assertEqual(targets["ETHUSDT"].loc[start - pd.Timedelta(hours=1)], 0.3)
        self.assertEqual(targets["BTCUSDT"].loc[start], 0)
        self.assertEqual(targets["ETHUSDT"].loc[start], 0.3)
        cutoff = start + pd.Timedelta(hours=5)
        changed = {s: f.copy() for s, f in frames.items()}
        for frame in changed.values():
            frame.loc[frame.index > cutoff, "close"] = 1.0
        later, _ = generate_rule_targets(changed, definition, 72, start)
        for symbol in targets:
            pd.testing.assert_series_equal(targets[symbol].loc[:cutoff], later[symbol].loc[:cutoff])
        market = {s: f.loc[index >= start - pd.Timedelta(hours=1)].copy() for s, f in frames.items()}
        for frame in market.values():
            frame[["open", "high", "low", "close"]] = 100.0
        weights = {s: t.loc[market[s].index] for s, t in targets.items()}
        backtest = run_relative_strength_backtest(market, weights, config=BacktestConfig(
            initial_capital=10000, fee_bps=0, slippage_bps=0, min_trade_fraction=0))
        self.assertGreater(backtest.weights.loc[start, "trade_notional"], 0)
        self.assertGreater(backtest.weights.loc[start + pd.Timedelta(hours=1), "trade_notional"], 0)
        closed = backtest.trades.iloc[0]
        self.assertEqual(closed["symbol"], "BTCUSDT")
        self.assertEqual(closed["exit_time"], start + pd.Timedelta(hours=1))
        self.assertAlmostEqual(closed["quantity"], 30)
        definition.update(entry="1", exit="1")
        exited, _ = generate_rule_targets(frames, definition, 72, start)
        self.assertEqual(sum(t.sum() for t in exited.values()), 0)

    def test_rule_definition_rejects_unsupported_inputs_and_invalid_execution(self):
        contract = self.rule_contract()
        definition = json.loads((EXAMPLES / "factor_rules/multi-factor.replay.json").read_text())[0]["response"]["parameters"]
        for name, change in (
            ("unavailable_input", {"entry": "funding_rate"}),
            ("future", {"entry": "ts_return(spot_close,-1)"}),
            ("warmup", {"entry": "ts_return(spot_close,100)"}),
            ("allocation", {"allocation": {"max_positions": 2, "gross_exposure": 2, "max_asset_weight": 1}}),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    contract.parameters({**definition, **change})
        contract.parameters({**definition, "score": "spot_close", "entry": "spot_close"})

    def test_scheduled_rule_fill_waits_for_real_bar(self):
        contract = self.rule_contract()
        frames = load_segment(self.db, contract, "development")
        start = pd.Timestamp(contract.development_start)
        for frame in frames.values():
            frame.loc[start, "synthetic"] = True
        definition = {"signals": [], "score": "1", "entry": "1", "exit": "0", "allocation": {
            "max_positions": 1, "gross_exposure": 0.5, "max_asset_weight": 0.5}, "rebalance_hours": 24}
        root = self.root / "scheduled"
        evaluate(frames, definition, contract, "development", root)
        executions = pd.read_csv(root / "strategy-executions.csv", index_col=0, parse_dates=[0])
        self.assertEqual(executions.loc[start, "trade_notional"], 0)
        self.assertGreater(executions.loc[start + pd.Timedelta(hours=1), "trade_notional"], 0)
        self.assertTrue((executions.loc[start + pd.Timedelta(hours=2):start + pd.Timedelta(hours=23), "trade_notional"] == 0).all())

    def test_rule_optimizer_changes_only_authorized_component(self):
        self.contract = self.rule_contract()
        first = json.loads((EXAMPLES / "factor_rules/multi-factor.replay.json").read_text())
        first[3]["response"].update(action="optimize", selected_version=None, modification={
            "change_parameter": "allocation", "hypothesis": "Reduce exposure to limit losses.",
            "min_return_improvement": 0.01, "max_drawdown_increase": 0.0})
        second = copy.deepcopy(first)
        definition = second[0]["response"]["parameters"]
        definition["allocation"]["gross_exposure"] = 0.4
        plan = execution_plan(definition, self.contract)
        second[0]["response"]["calculation_meaning"] = plan
        calc = second[1]["response"]
        calc.update(version="v0002", plan_sha256=digest(plan), evidence_refs=["v0002-definition"])
        for reply in second[2:]:
            reply["response"]["evidence_refs"] = [s.replace("v0001", "v0002") for s in reply["response"]["evidence_refs"]]
        second[3]["response"].update(action="discard", modification=None)
        runner = self.runner(first + second, "rule-optimization")
        self.assertEqual(runner.run()["status"], "discarded")
        experiment = json.loads((runner.root / "development_records/v0002-experiment.json").read_text())["data"]
        self.assertIn("comparison", experiment)
        self.assertEqual(experiment["parameters"]["allocation"]["gross_exposure"], 0.4)

    def test_full_loop_versions_freeze_costs_and_holdout_isolation(self):
        runner = self.runner()
        original_load = load_segment
        stages = []
        def checked_load(db, contract, stage):
            stages.append(stage)
            if stage == "validation":
                self.assertTrue((runner.root / "frozen.json").exists())
                self.assertTrue((runner.root / "validation-access.json").exists())
            return original_load(db, contract, stage)
        with patch("crypto_quant.research.strategy_research.workflow.load_segment", side_effect=checked_load):
            result = runner.run()
            self.assertEqual(result["status"], "retained_for_validation")
            self.assertEqual(stages, ["development"])
            self.assertFalse((runner.root / "frozen.json").exists())
            self.assertFalse((runner.root / "validation-access.json").exists())
            development_result = (runner.root / "result.json").read_bytes()
            development_report = (runner.root / "report.md").read_bytes()
            validation = validate_run(runner.root, self.db)
        self.assertEqual(validation["status"], "engineering_complete")
        self.assertEqual((runner.root / "result.json").read_bytes(), development_result)
        self.assertEqual((runner.root / "report.md").read_bytes(), development_report)
        self.assertEqual(result["versions"], 2)
        self.assertEqual(stages, ["development", "validation"])
        self.assertFalse(result["paper_started"])
        definitions = json.loads((runner.root / "development_records/v0002-definition.json").read_text())["data"]
        self.assertEqual(definitions["parent"], "v0001")
        self.assertEqual(definitions["parameters"]["lookback"], 72)
        calculation = json.loads((runner.root / "development_records/v0002-calculation.json").read_text())["data"]
        self.assertTrue(calculation["consistent"])
        report = json.loads((runner.root / "development_records/v0001-experiment.json").read_text())["data"]
        self.assertGreater(report["results"]["strategy"]["fees"], 0)
        self.assertGreater(report["results"]["benchmark"]["fees"], 0)
        self.assertIn("comparison", json.loads((runner.root / "development_records/v0002-experiment.json").read_text())["data"])
        requests = sorted((runner.root / "model_calls").glob("*.request.json"))
        self.assertEqual(len(requests), 8)
        self.assertEqual([json.loads(json.loads(p.read_text())[1]["content"])["role"] for p in requests],
                         ["design", "calculate", "review", "decide"] * 2)
        for request in requests:
            payload = json.loads(json.loads(request.read_text())[1]["content"])
            self.assertFalse(any(r["data"].get("stage") == "validation" for r in payload["records"]))
        with self.assertRaises(FileExistsError):
            self.runner()

    def test_reviewed_execution_preserves_existing_numeric_results(self):
        runner = self.runner()
        runner.run()
        frames = load_segment(self.db, self.contract, "development")
        direct = evaluate(frames, self.replies[0]["response"]["parameters"], self.contract,
                          "development", self.root / "direct")
        recorded = json.loads((runner.root / "development_records/v0001-experiment.json").read_text())["data"]
        self.assertEqual(direct["results"], recorded["results"])
        for name in ("strategy", "benchmark", "stress"):
            for kind in ("targets", "executions", "equity", "closed_trades"):
                self.assertEqual((self.root / "direct" / f"{name}-{kind}.csv").read_bytes(),
                                 (runner.root / "v0001" / f"{name}-{kind}.csv").read_bytes())

    def test_unsupported_task_stops_before_data_and_model(self):
        for key, value in (("research_type", "event_study"), ("strategy_family", "liquidation_reversal"),
                           ("deliverable", "relationship_report")):
            with self.subTest(key=key):
                raw = self.contract.as_dict()
                raw["task"][key] = value
                runner = StrategyResearch(StrategyResearchContract.from_dict(raw), self.idea,
                                          ReplayModel(self.replies), self.root / "absent.sqlite",
                                          self.root / key, model_mode="replay")
                with patch("crypto_quant.research.strategy_research.workflow.load_segment") as loader:
                    result = runner.run()
                loader.assert_not_called()
                self.assertEqual(result["status"], "unsupported")
                self.assertEqual(result["task"][key], value)
                self.assertFalse((runner.root / "model_calls").exists())

    def test_semantic_mismatch_is_recorded_without_blocking_backtest(self):
        replies = copy.deepcopy(self.replies)
        replies[0]["response"]["calculation_meaning"]["execution"] = "execute at same bar close"
        calculation = replies[1]["response"]
        calculation["checks"]["execution"].update(matches=False,
                                                   reason="original definition would use same-close execution")
        calculation["consistent"] = False
        runner = self.runner(replies)
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        self.assertTrue((runner.root / "v0001/strategy-equity.csv").exists())
        self.assertTrue((runner.root / "development_records/v0001-calculation.json").exists())
        saved = json.loads((runner.root / "development_records/v0001-calculation.json").read_text())["data"]
        self.assertFalse(saved["consistent"])
        self.assertFalse((runner.root / "validation-access.json").exists())

    def test_calculation_opinions_do_not_block_execution_but_types_still_parse(self):
        for variant in ("version", "hash", "contradiction", "hypothesis"):
            with self.subTest(variant=variant):
                replies = copy.deepcopy(self.replies)
                calculation = replies[1]["response"]
                if variant == "version":
                    calculation["version"] = "v9999"
                elif variant == "hash":
                    calculation["plan_sha256"] = "wrong"
                elif variant == "hypothesis":
                    calculation["hypothesis_matches"] = False
                else:
                    calculation["checks"]["signal"]["matches"] = False
                runner = self.runner(replies, variant)
                self.assertEqual(runner.run()["status"], "retained_for_validation")
                self.assertTrue((runner.root / "v0001/strategy-equity.csv").exists())
        replies = copy.deepcopy(self.replies)
        replies[1]["response"]["checks"]["signal"]["matches"] = "true"
        replies[2:2] = [copy.deepcopy(replies[1]) for _ in range(3)]
        with self.assertRaises(ValueError):
            self.runner(replies, "invalid-boolean").run()

    def test_quote_variation_is_ignored_and_sources_are_bound_by_program(self):
        self.contract = self.rule_contract()
        replies = json.loads((EXAMPLES / "factor_rules/multi-factor.replay.json").read_text())
        replies[0]["response"]["extra_explanation"] = "ignored"
        replies[0]["response"]["calculation_meaning"]["extra_note"] = "ignored"
        calc = replies[1]["response"]
        # Actual RSI failure: model appended 'over entry' to its copied sentence.
        calc["checks"]["signal"]["execution_quote"] = "exit > 0 takes precedence over entry"
        calc["checks"]["signal"]["definition_quote"] = "a paraphrase"
        calc["checks"]["extra_note"] = "ignored"
        calc["sources"] = {"signal": "a forged source"}
        runner = self.runner(replies)
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        saved = json.loads((runner.root / "development_records/v0001-calculation.json").read_text())["data"]
        self.assertNotIn("execution_quote", saved["checks"]["signal"])
        self.assertEqual(saved["sources"]["signal"]["execution"],
                         {"record_id": "v0001-definition", "field_path": "execution_plan.signal"})
        raw = json.loads((runner.root / "model_calls/call-0002.response.json").read_text())
        self.assertIn("over entry", raw["text"])
        self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")

    def test_unknown_semantics_are_recorded_without_blocking_execution(self):
        for field in ("signal", "hypothesis"):
            with self.subTest(field=field):
                replies = copy.deepcopy(self.replies)
                calc = replies[1]["response"]
                calc["consistent"] = None
                if field == "hypothesis":
                    calc["hypothesis_matches"] = None
                else:
                    calc["checks"][field]["matches"] = None
                runner = self.runner(replies, field)
                self.assertEqual(runner.run()["status"], "retained_for_validation")
                self.assertTrue((runner.root / "v0001/strategy-equity.csv").exists())

    def test_semantic_feedback_no_longer_forces_redesign(self):
        self.contract = self.rule_contract()
        good = json.loads((EXAMPLES / "factor_rules/multi-factor.replay.json").read_text())
        good[1]["response"]["checks"]["execution"].update(matches=False, reason="same close conflicts with next open")
        good[1]["response"]["consistent"] = False
        runner = self.runner(good)
        self.assertEqual(runner.run()["selected_version"], "v0001")
        self.assertTrue((runner.root / "v0001/strategy-equity.csv").exists())
        self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")

    def test_malformed_calculation_is_corrected_once_with_error_feedback(self):
        self.contract = self.rule_contract()
        replies = json.loads((EXAMPLES / "factor_rules/multi-factor.replay.json").read_text())
        bad = copy.deepcopy(replies[1])
        bad["response"]["checks"]["signal"]["matches"] = "true"
        runner = self.runner(replies[:1] + [bad] + replies[1:])
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        self.assertEqual(runner.calls, 5)
        error = json.loads((runner.root / "model_calls/call-0002.validation-error.json").read_text())
        self.assertIn("bool_type", error["reason"])
        self.assertIn("signal", error["reason"])
        request = json.loads((runner.root / "model_calls/call-0003.request.json").read_text())
        self.assertEqual(request[-2]["role"], "assistant")
        self.assertIn(error["reason"], request[-1]["content"])
        self.assertEqual(json.loads(request[1]["content"])["role"], "calculate")

    def test_decision_missing_reason_and_invalid_action_are_corrected_together(self):
        replies = copy.deepcopy(self.replies)
        bad = copy.deepcopy(replies[3])
        bad["response"].pop("reason")
        bad["response"]["action"] = "decide"  # Actual MiMo failure on 2026-09-22.
        runner = self.runner(replies[:3] + [bad] + replies[3:])
        result = runner.run()
        self.assertEqual(result["status"], "retained_for_validation")
        error = json.loads((runner.root / "model_calls/call-0004.validation-error.json").read_text())
        self.assertIn('"action"', error["reason"])
        self.assertIn('"reason"', error["reason"])
        request = json.loads((runner.root / "model_calls/call-0005.request.json").read_text())
        self.assertIn(error["reason"], request[-1]["content"])
        self.assertTrue((runner.root / "development_records/v0002-experiment.json").exists())
        self.assertFalse((runner.root / "validation-access.json").exists())

    def test_proposal_extra_executable_field_is_corrected_before_capability_check(self):
        self.contract = self.rule_contract()
        replies = json.loads((EXAMPLES / "factor_rules/exploration.replay.json").read_text())
        bad = copy.deepcopy(replies[0])
        bad["response"]["parameters"]["definition"]["warmup_hours"] = 72
        runner = self.runner([bad] + replies, "proposal-correction")
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        error = json.loads((runner.root / "model_calls/call-0001.validation-error.json").read_text())
        self.assertIn("expected exact fields", error["reason"])
        self.assertTrue((runner.root / "development_records/v0001-proposal.json").exists())

    def test_json_and_required_field_errors_allow_three_corrections(self):
        for bad in ("not JSON", '{"action":"design"}'):
            with self.subTest(bad=bad):
                runner = self.runner(name=str(len(bad)))
                runner.data_catalog = {}
                model = unittest.mock.Mock()
                model.complete.side_effect = [ModelReply(bad, {}, "test")] * 3 + [
                    ModelReply(json.dumps(self.replies[0]["response"]), {}, "test")]
                runner.model = model
                self.assertEqual(runner._ask("design", {})["action"], "design")
                self.assertEqual(model.complete.call_count, 4)
                self.assertTrue((runner.root / "model_calls/call-0001.validation-error.json").exists())

    def test_gap_stops_after_design_before_backtest_and_preserves_reason(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE symbol='ETHUSDT' AND open_time=?",
                         (pd.Timestamp("2026-01-06T00:00:00Z").value // 1000000,))
        runner = self.runner()
        result = runner.run()
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIn("missing=1", result["reason"])
        self.assertTrue((runner.root / "model_calls/call-0001.response.json").exists())
        self.assertFalse((runner.root / "model_calls/call-0002.request.json").exists())
        self.assertFalse((runner.root / "v0001").exists())

    def test_holdout_gap_keeps_freeze_and_access_record(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE open_time=?", (pd.Timestamp("2026-01-09T00:00:00Z").value // 1000000,))
        runner = self.runner()
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        self.assertEqual(validate_run(runner.root, self.db)["status"], "insufficient_data")
        self.assertTrue((runner.root / "validation-access.json").exists())
        self.assertTrue((runner.root / "frozen.json").exists())

    def test_pause_and_discard_never_load_validation(self):
        for action in ("pause", "discard"):
            with self.subTest(action=action):
                replies = copy.deepcopy(self.replies[:4])
                replies[3]["response"].update(action=action, modification=None,
                    resume_condition="need new evidence" if action == "pause" else None)
                runner = self.runner(replies, action)
                result = runner.run()
                self.assertEqual(result["status"], "paused" if action == "pause" else "discarded")
                self.assertFalse((runner.root / "validation-access.json").exists())

    def test_empty_review_evidence_reaches_decision(self):
        for empty_fields in (("supporting_evidence",), ("counter_evidence",),
                             ("supporting_evidence", "counter_evidence")):
            with self.subTest(empty_fields=empty_fields):
                replies = copy.deepcopy(self.replies[:4])
                for name in empty_fields:
                    replies[2]["response"][name] = []
                replies[3]["response"].update(action="discard", modification=None)
                runner = self.runner(replies, "-".join(empty_fields))
                self.assertEqual(runner.run()["status"], "discarded")
                review = json.loads((runner.root / "development_records/v0001-review.json").read_text())["data"]
                for name in empty_fields:
                    self.assertEqual(review[name], [])
                self.assertEqual(runner.calls, 4)
                self.assertFalse((runner.root / "validation-access.json").exists())

    def test_review_evidence_still_requires_lists_of_nonempty_text(self):
        for name in ("supporting_evidence", "counter_evidence"):
            for index, value in enumerate((None, "evidence", [""], [123])):
                with self.subTest(name=name, value=value):
                    replies = copy.deepcopy(self.replies[:4])
                    replies[2]["response"][name] = value
                    runner = self.runner(replies[:3] + [copy.deepcopy(replies[2]) for _ in range(3)], f"{name}-{index}")
                    with self.assertRaises(ValueError):
                        runner.run()
                    self.assertEqual(runner.calls, 6)
                    error = json.loads((runner.root / "model_calls/call-0006.validation-error.json").read_text())
                    self.assertEqual(error["correction_remaining"], 0)
                    self.assertIn(name, error["reason"])
                    self.assertTrue((runner.root / "model_calls/call-0003.response.json").exists())

    def test_untrusted_outputs_fail_with_raw_reply_preserved(self):
        for variant in ("outside_space", "extra_rule"):
            with self.subTest(variant=variant):
                replies = copy.deepcopy(self.replies)
                if variant == "outside_space":
                    replies[0]["response"]["parameters"]["lookback"] = 999
                elif variant == "extra_rule":
                    replies[0]["response"]["parameters"]["stop_loss"] = .02
                runner = self.runner(replies, variant)
                with self.assertRaises(ValueError):
                    runner.run()
                self.assertEqual(json.loads((runner.root / "result.json").read_text())["status"], "error")
                self.assertTrue(list((runner.root / "model_calls").glob("*.response.json")))
                self.assertFalse((runner.root / "validation-access.json").exists())

    def test_duplicate_trial_and_modification_label_do_not_block_backtest(self):
        for variant in ("duplicate", "wrong_change"):
            with self.subTest(variant=variant):
                replies = copy.deepcopy(self.replies)
                if variant == "duplicate":
                    replies[4]["response"]["parameters"]["lookback"] = 24
                else:
                    replies[3]["response"]["modification"]["change_parameter"] = "spread_threshold"
                runner = self.runner(replies, variant)
                self.assertEqual(runner.run()["status"], "retained_for_validation")
                self.assertTrue((runner.root / "v0002/strategy-equity.csv").exists())

    def test_context_byte_setting_no_longer_blocks_model_call(self):
        raw = self.contract.as_dict()
        raw["context_bytes"] = 1
        runner = StrategyResearch(StrategyResearchContract.from_dict(raw), self.idea, ReplayModel(self.replies),
                                  self.db, self.root / "small", model_mode="replay")
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        self.assertTrue(list((runner.root / "model_calls").glob("*.request.json")))
        self.assertTrue(list((runner.root / "model_calls").glob("*.response.json")))

    def test_unsupported_idea_is_not_silently_rewritten(self):
        response = copy.deepcopy(self.replies[:1])
        response[0]["response"].update(action="unsupported", reason="requires liquidation strategy", parameters=None, calculation_meaning=None)
        runner = self.runner(response)
        self.assertEqual(runner.run()["status"], "unsupported")
        self.assertFalse((runner.root / "frozen.json").exists())

    def test_future_prices_cannot_change_earlier_targets_or_equity(self):
        frames = load_segment(self.db, self.contract, "development")
        changed = {s: f.copy() for s, f in frames.items()}
        cutoff = pd.Timestamp("2026-01-07T00:00:00Z")
        for frame in changed.values():
            frame.loc[cutoff:, ["open", "high", "low", "close"]] *= 2
        params = self.replies[0]["response"]["parameters"]
        for name, data in (("original", frames), ("changed", changed)):
            evaluate(data, params, self.contract, "development", self.root / name)
        for kind in ("targets", "equity"):
            original = pd.read_csv(self.root / "original" / f"strategy-{kind}.csv", index_col=0, parse_dates=True)
            modified = pd.read_csv(self.root / "changed" / f"strategy-{kind}.csv", index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(original.loc[original.index < cutoff], modified.loc[modified.index < cutoff])
        execution = pd.read_csv(self.root / "original/strategy-executions.csv", index_col=0, parse_dates=True)
        self.assertEqual(execution.index[0], pd.Timestamp(self.contract.development_start))
        targets = pd.read_csv(self.root / "original/strategy-targets.csv", index_col=0, parse_dates=True)
        np.testing.assert_allclose(execution.target_weight.to_numpy(), targets.sum(axis=1).iloc[:-1].to_numpy())

    def test_failed_modification_cannot_be_retained(self):
        replies = copy.deepcopy(self.replies)
        replies[3]["response"]["modification"]["min_return_improvement"] = 100
        bad = copy.deepcopy(replies[7])
        bad["response"]["selected_version"] = "v0002"
        replies.insert(7, bad)
        runner = self.runner(replies)
        self.assertEqual(runner.run()["selected_version"], "v0001")
        error = json.loads((runner.root / "model_calls/call-0008.validation-error.json").read_text())
        self.assertIn("failed its predeclared comparison", error["reason"])
        self.assertFalse((runner.root / "validation-access.json").exists())

    def test_resume_after_failed_model_call_keeps_prior_evidence_and_validation_boundary(self):
        runner = self.runner(self.replies[:5], "resume")
        with self.assertRaises(RuntimeError):
            runner.run()
        self.assertEqual(latest_result(runner.root)["status"], "error")
        self.assertFalse((runner.root / "code.json").exists())
        checkpoint_path = sorted((runner.root / "checkpoints").glob("*.json"))[-1]
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["code_sha256"] = "legacy-code-version"
        checkpoint["sha256"] = digest({key: value for key, value in checkpoint.items() if key != "sha256"})
        checkpoint_path.write_text(json.dumps(checkpoint))
        (runner.root / "code.json").write_text('{"legacy.py": "different-code-version"}')
        input_path = runner.root / "development_records/input.json"
        prior_input = json.loads(input_path.read_text())
        prior_input["data"]["note"] = "edited after the failed run"
        input_path.write_text(json.dumps(prior_input))
        first_records = {p.name: p.read_bytes() for p in (runner.root / "development_records").glob("*.json")}
        resumed = StrategyResearch.resume(runner.root, self.db, ReplayModel(self.replies, skip=5),
                                          model_mode="replay")
        self.assertEqual(resumed.run()["status"], "retained_for_validation")
        self.assertEqual(latest_result(runner.root)["selected_version"], "v0001")
        self.assertEqual(first_records, {name: (runner.root / "development_records" / name).read_bytes()
                                         for name in first_records})
        self.assertFalse((runner.root / "validation-access.json").exists())
        self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")

    def test_first_design_failure_has_a_resume_checkpoint(self):
        bad = copy.deepcopy(self.replies[0])
        bad["response"]["action"] = "unknown"
        runner = self.runner([bad], "early-resume")
        with self.assertRaises(RuntimeError):
            runner.run()
        self.assertEqual(latest_result(runner.root)["status"], "error")
        checkpoint = json.loads((runner.root / "checkpoints/000001.json").read_text())
        self.assertEqual(checkpoint["next_node"], "design")
        resumed = StrategyResearch.resume(runner.root, self.db, ReplayModel([bad] + self.replies, skip=1),
                                          model_mode="replay")
        self.assertEqual(resumed.run()["status"], "retained_for_validation")
        self.assertEqual(latest_result(runner.root)["selected_version"], "v0001")

    def test_data_change_before_validation_does_not_block_calculation(self):
        runner = self.runner()
        runner.run()
        original_load = load_segment
        def changed_load(db, contract, stage):
            frames = original_load(db, contract, stage)
            if stage == "validation":
                frames["BTCUSDT"].iloc[0, frames["BTCUSDT"].columns.get_loc("close")] *= 1.01
            return frames
        with patch("crypto_quant.research.strategy_research.workflow.load_segment", side_effect=changed_load):
            self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")

    def test_validation_can_be_repeated_and_keeps_prior_result(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                runner = self.runner(name=str(fail))
                runner.run()
                if fail:
                    with patch("crypto_quant.research.strategy_research.workflow.load_segment", side_effect=RuntimeError("broken")):
                        with self.assertRaisesRegex(RuntimeError, "broken"):
                            validate_run(runner.root, self.db)
                    self.assertEqual(json.loads((runner.root / "validation-result.json").read_text())["status"], "error")
                else:
                    validate_run(runner.root, self.db)
                self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")
                self.assertEqual(latest_validation_result(runner.root)["status"], "engineering_complete")
                self.assertTrue((runner.root / "validation-attempts/attempt-0001/validation-result.json").exists())

    def test_validation_ignores_prior_evidence_and_runtime_metadata_changes(self):
        for variant in ("contract", "definition", "data", "artifact", "checkpoint", "runtime"):
            with self.subTest(variant=variant):
                runner = self.runner(name=variant)
                runner.run()
                paths = {"contract": "contract.json", "definition": "development_records/v0001-definition.json",
                         "data": "development_data/BTCUSDT.csv", "artifact": "v0001/strategy-equity.csv",
                         "checkpoint": "development-complete.json"}
                if variant in paths:
                    path = runner.root / paths[variant]
                    if variant == "checkpoint":
                        data = json.loads(path.read_text())
                        data["sha256"] = "outdated-hash"
                        path.write_text(json.dumps(data))
                    else:
                        path.write_text(path.read_text() + " ")
                if variant == "runtime":
                    (runner.root / "environment.json").write_text('{"python":"old"}')
                self.assertEqual(validate_run(runner.root, self.db)["status"], "engineering_complete")

    def test_validate_rejects_discarded_candidate(self):
        replies = copy.deepcopy(self.replies[:4])
        replies[3]["response"].update(action="discard", modification=None)
        runner = self.runner(replies)
        runner.run()
        with self.assertRaisesRegex(ValueError, "no retained candidate"):
            validate_run(runner.root, self.db)

    def test_authorized_gap_uses_only_previous_complete_bars_and_preserves_source(self):
        gap = pd.Timestamp("2026-01-06T13:00:00Z")
        partial = gap - pd.Timedelta(hours=1)
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE open_time=?", (gap.value // 1000000,))
            conn.execute("UPDATE klines SET close_time=? WHERE open_time=?",
                         ((partial + pd.Timedelta(minutes=39)).value // 1000000, partial.value // 1000000))
        raw = self.contract.as_dict()
        raw["imputed_hours"] = [gap.isoformat()]
        contract = StrategyResearchContract.from_dict(raw)
        frames = load_segment(self.db, contract, "development")
        for frame in frames.values():
            self.assertTrue(frame.loc[gap, "synthetic"])
            self.assertTrue(frame.loc[partial, "shortened"])
            for field in ("open", "high", "low", "close"):
                self.assertEqual(frame.loc[gap, field], frame.loc[[gap-pd.Timedelta(hours=3), gap-pd.Timedelta(hours=2)], field].mean())
            self.assertEqual(frame.loc[gap, "quote_volume"], 0)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM klines WHERE open_time=?", (gap.value//1000000,)).fetchone()[0], 0)
            conn.execute("UPDATE klines SET open=open*2, high=high*2, low=low*2, close=close*2 WHERE open_time>?",
                         (gap.value // 1000000,))
        changed = load_segment(self.db, contract, "development")
        for symbol in frames:
            pd.testing.assert_frame_equal(frames[symbol].loc[:gap], changed[symbol].loc[:gap])
        with self.assertRaisesRegex(ValueError, "missing hours"):
            load_segment(self.db, self.contract, "development")

    def test_nontradable_hour_blocks_both_legs_without_changing_holdings(self):
        frames = load_segment(self.db, self.contract, "development")
        index = frames["BTCUSDT"].index[:5]
        market = {s: f.loc[index] for s, f in frames.items()}
        targets = {"BTCUSDT": pd.Series([1.,0.,0.,0.,0.], index=index),
                   "ETHUSDT": pd.Series([0.,1.,1.,1.,1.], index=index)}
        tradable = pd.Series([True,True,False,True,True], index=index)
        result = run_relative_strength_backtest(market, targets, config=BacktestConfig(), tradable=tradable)
        before, halted, after = result.weights.iloc[0], result.weights.iloc[1], result.weights.iloc[2]
        self.assertGreater(before.trade_notional, 0)
        self.assertEqual(halted.trade_notional, 0)
        self.assertEqual(halted.fee, 0)
        held_notional = before.executed_weight * result.equity.iloc[0]
        cash = result.equity.iloc[0] - held_notional
        units = held_notional / market["BTCUSDT"].loc[index[1], "close"]
        self.assertAlmostEqual(result.equity.iloc[1], cash + units * market["BTCUSDT"].loc[index[2], "close"])
        self.assertGreater(after.trade_notional, 0)

    def test_contract_rejects_bad_split_and_replay_as_research(self):
        raw = self.contract.as_dict()
        raw["validation_start"] = raw["development_start"]
        with self.assertRaises(ValueError):
            StrategyResearchContract.from_dict(raw)
        raw = self.contract.as_dict()
        raw["purpose"] = "research"
        with self.assertRaisesRegex(ValueError, "engineering"):
            StrategyResearch(StrategyResearchContract.from_dict(raw), self.idea, ReplayModel(self.replies), self.db,
                             self.root / "invalid", model_mode="replay")


if __name__ == "__main__":
    unittest.main()
