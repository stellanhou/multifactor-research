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
from crypto_quant.research.strategy_research.workflow import StrategyResearch

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
        self.assertEqual(result["status"], "engineering_complete")
        self.assertEqual(result["versions"], 2)
        self.assertEqual(stages, ["development", "validation"])
        self.assertFalse(result["paper_started"])
        definitions = json.loads((runner.root / "development_records/v0002-definition.json").read_text())["data"]
        self.assertEqual(definitions["parent"], "v0001")
        self.assertEqual(definitions["parameters"]["lookback"], 72)
        report = json.loads((runner.root / "development_records/v0001-experiment.json").read_text())["data"]
        self.assertGreater(report["results"]["strategy"]["fees"], 0)
        self.assertGreater(report["results"]["benchmark"]["fees"], 0)
        self.assertIn("comparison", json.loads((runner.root / "development_records/v0002-experiment.json").read_text())["data"])
        requests = sorted((runner.root / "model_calls").glob("*.request.json"))
        self.assertEqual(len(requests), 6)
        for request in requests:
            payload = json.loads(json.loads(request.read_text())[1]["content"])
            self.assertFalse(any(r["data"].get("stage") == "validation" for r in payload["records"]))
        with self.assertRaises(FileExistsError):
            self.runner()

    def test_gap_stops_before_model_and_preserves_reason(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE symbol='ETHUSDT' AND open_time=?",
                         (pd.Timestamp("2026-01-06T00:00:00Z").value // 1000000,))
        runner = self.runner()
        result = runner.run()
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIn("missing=1", result["reason"])
        self.assertFalse((runner.root / "model_calls").exists())

    def test_holdout_gap_keeps_freeze_and_access_record(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE open_time=?", (pd.Timestamp("2026-01-09T00:00:00Z").value // 1000000,))
        runner = self.runner()
        self.assertEqual(runner.run()["status"], "insufficient_data")
        self.assertTrue((runner.root / "validation-access.json").exists())
        self.assertTrue((runner.root / "frozen.json").exists())

    def test_pause_and_discard_never_load_validation(self):
        for action in ("pause", "discard"):
            with self.subTest(action=action):
                replies = copy.deepcopy(self.replies[:3])
                replies[2]["response"].update(action=action, modification=None,
                    resume_condition="need new evidence" if action == "pause" else None)
                runner = self.runner(replies, action)
                result = runner.run()
                self.assertEqual(result["status"], "paused" if action == "pause" else "discarded")
                self.assertFalse((runner.root / "validation-access.json").exists())

    def test_untrusted_outputs_fail_with_raw_reply_preserved(self):
        for variant in ("outside_space", "unknown_reference", "extra_rule", "duplicate", "wrong_change"):
            with self.subTest(variant=variant):
                replies = copy.deepcopy(self.replies)
                if variant == "outside_space":
                    replies[0]["response"]["parameters"]["lookback"] = 999
                elif variant == "unknown_reference":
                    replies[1]["response"]["evidence_refs"] = ["imaginary-experiment"]
                elif variant == "extra_rule":
                    replies[0]["response"]["parameters"]["stop_loss"] = .02
                elif variant == "duplicate":
                    replies[3]["response"]["parameters"]["lookback"] = 24
                else:
                    replies[2]["response"]["modification"]["change_parameter"] = "spread_threshold"
                runner = self.runner(replies, variant)
                with self.assertRaises(ValueError):
                    runner.run()
                self.assertEqual(json.loads((runner.root / "result.json").read_text())["status"], "error")
                self.assertTrue(list((runner.root / "model_calls").glob("*.response.json")))
                self.assertFalse((runner.root / "validation-access.json").exists())

    def test_context_limit_stops_without_truncation_or_model_call(self):
        raw = self.contract.as_dict()
        raw["context_bytes"] = 1
        runner = StrategyResearch(StrategyResearchContract.from_dict(raw), self.idea, ReplayModel(self.replies),
                                  self.db, self.root / "small", model_mode="replay")
        with self.assertRaisesRegex(ValueError, "no truncation"):
            runner.run()
        self.assertTrue(list((runner.root / "model_calls").glob("*.request.json")))
        self.assertFalse(list((runner.root / "model_calls").glob("*.response.json")))

    def test_unsupported_idea_is_not_silently_rewritten(self):
        response = copy.deepcopy(self.replies[:1])
        response[0]["response"].update(action="unsupported", reason="requires liquidation strategy", parameters=None)
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
        replies[2]["response"]["modification"]["min_return_improvement"] = 100
        replies[5]["response"]["selected_version"] = "v0002"
        runner = self.runner(replies)
        with self.assertRaisesRegex(ValueError, "failed its predeclared comparison"):
            runner.run()
        self.assertFalse((runner.root / "validation-access.json").exists())

    def test_data_change_before_validation_is_rejected(self):
        runner = self.runner()
        original_load = load_segment
        def changed_load(db, contract, stage):
            frames = original_load(db, contract, stage)
            if stage == "validation":
                frames["BTCUSDT"].iloc[0, frames["BTCUSDT"].columns.get_loc("close")] *= 1.01
            return frames
        with patch("crypto_quant.research.strategy_research.workflow.load_segment", side_effect=changed_load):
            with self.assertRaisesRegex(ValueError, "development data changed"):
                runner.run()

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
