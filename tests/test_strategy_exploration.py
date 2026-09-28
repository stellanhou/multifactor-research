"""A broad proposal must survive a narrow execution adapter without rewriting."""
import copy
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_strategy_research as fixtures
from crypto_quant.research.strategy_research.exploration import data_catalog


class ExplorationTests(unittest.TestCase):
    setUp = fixtures.ResearchTests.setUp
    runner = fixtures.ResearchTests.runner

    def proposal(self, **changes):
        replies = copy.deepcopy(self.replies)
        design = replies[0]["response"]
        definition = design["parameters"]
        design["action"] = "propose"
        design["parameters"] = {
            "family": self.contract.task["strategy_family"],
            "symbols": ["BTCUSDT", "ETHUSDT"], "required_fields": ["spot_close"],
            "required_capabilities": ["spot_long_cash", "next_hour_open", "relative_strength_rotation"],
            "definition": definition, **changes}
        return replies

    def test_executable_proposal_enters_existing_loop_and_preserves_original(self):
        replies = self.proposal()
        runner = self.runner(replies)
        self.assertEqual(runner.run()["status"], "retained_for_validation")
        saved = json.loads((runner.root / "development_records/v0001-proposal.json").read_text())["data"]
        self.assertEqual(saved, replies[0]["response"])
        self.assertTrue((runner.root / "v0001/strategy-equity.csv").exists())
        self.assertFalse((runner.root / "validation-access.json").exists())

    def test_new_execution_idea_preserved_without_loading_btc_eth_adapter(self):
        replies = self.proposal(family="cross_exchange_market_making", definition=None,
                                required_capabilities=["two_exchange_limit_orders"])
        runner = self.runner(replies)
        with patch("crypto_quant.research.strategy_research.workflow.load_segment") as load:
            result = runner.run()
        load.assert_not_called()
        self.assertEqual(result["status"], "missing_execution_capability")
        self.assertEqual(result["capability_check"]["missing_data"], [])
        saved = json.loads((runner.root / "development_records/v0001-proposal.json").read_text())["data"]
        self.assertEqual(saved, replies[0]["response"])
        self.assertIn("two_exchange_limit_orders", str(result))

    def test_missing_field_is_reported_separately_from_execution_gap(self):
        runner = self.runner(self.proposal(required_fields=["spot_close", "onchain_whale_flow"], definition=None))
        with patch("crypto_quant.research.strategy_research.workflow.evaluate") as evaluate:
            result = runner.run()
        evaluate.assert_not_called()
        self.assertEqual(result["status"], "missing_data")
        self.assertEqual(result["capability_check"]["missing_data"][0]["field"], "onchain_whale_flow")
        self.assertTrue(result["capability_check"]["missing_execution"])

    def test_present_altcoin_data_is_visible_but_not_replaced_with_btc(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO klines SELECT 'SOLUSDT',interval,open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base_volume,taker_buy_quote_volume FROM klines WHERE symbol='BTCUSDT'")
        runner = self.runner(self.proposal(symbols=["SOLUSDT"]))
        result = runner.run()
        self.assertEqual(result["status"], "missing_execution_capability")
        self.assertEqual(result["capability_check"]["missing_data"], [])
        self.assertIn("SOLUSDT", str(runner.data_catalog))
        self.assertFalse((runner.root / "development_data").exists())

    def test_catalog_does_not_expose_validation_only_symbols(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE klines SET symbol='FUTUREUSDT' WHERE open_time>=? AND symbol='ETHUSDT'",
                         (1767830400000,)) # 2026-01-08 UTC, validation boundary
        self.assertNotIn("FUTUREUSDT", str(data_catalog(self.db, self.contract)))

    def test_proposal_is_saved_before_exact_price_gap_stops_execution(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM klines WHERE symbol='ETHUSDT' AND open_time=1767657600000")
        runner = self.runner(self.proposal())
        self.assertEqual(runner.run()["status"], "insufficient_data")
        self.assertTrue((runner.root / "development_records/v0001-proposal.json").exists())
        self.assertTrue((runner.root / "development_records/v0001-data-gap.json").exists())
