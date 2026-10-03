import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_quant.research.strategy_research.multifactor_factors import FactorPanels
from crypto_quant.research.strategy_research.multifactor_workflow import run_baseline


MODULE = "crypto_quant.research.strategy_research.multifactor_workflow"


def _test_inputs(start: pd.Timestamp, end: pd.Timestamp):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    panel_index = pd.MultiIndex.from_product(
        [pd.date_range(start - pd.Timedelta(hours=2), end - pd.Timedelta(hours=1), freq="h"), symbols],
        names=["timestamp", "symbol"],
    )
    panel_values = pd.DataFrame({"perp_close": np.full(len(panel_index), 100.0)}, index=panel_index)
    universe = pd.Series(True, index=panel_index, name="eligible")
    panel = SimpleNamespace(values=panel_values, diagnostics={"panel_source": "fixture"})

    market_index = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1), freq="h")
    frames = {}
    symbol_offsets = {"BTCUSDT": 0.0, "ETHUSDT": 1.0, "SOLUSDT": 2.0}
    for symbol in symbols:
        offset = symbol_offsets[symbol]
        close = 100.0 + offset + np.arange(len(market_index), dtype=float) * (0.2 - offset * 0.03)
        frames[symbol] = pd.DataFrame({
            "open": close - 0.05,
            "close": close,
            "mark_close": close,
        }, index=market_index)
    funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
    inputs = SimpleNamespace(panel=panel, frames=frames, funding=funding, universe=universe,
                              diagnostics={"historical_causality_certified": False,
                                           "row_repair_mask": "unavailable"})

    standardized = pd.DataFrame(index=panel_index, columns=["f1", "f2"], dtype=float)
    rows = np.array([[1.0, 0.5], [0.0, -1.0], [-1.0, 0.5]])
    for timestamp in panel_index.get_level_values("timestamp").unique():
        loc = panel_index.get_level_values("timestamp") == timestamp
        standardized.loc[loc, :] = rows
    masked_time = start + pd.Timedelta(hours=1)
    common_mask = pd.Series(True, index=panel_index, name="common_mask")
    common_mask.loc[masked_time] = False
    standardized.loc[~common_mask, :] = np.nan
    masked_rows = panel_index.get_level_values("timestamp") == masked_time
    standardized.loc[masked_rows, "f1"] = rows[:, 0]
    raw = standardized.copy()
    valid_masks = raw.notna()
    eligible_rows = int(universe.sum())
    coverage = pd.DataFrame({
        "eligible_rows": [eligible_rows, eligible_rows],
        "valid_rows": [int(valid_masks.f1.sum()), int(valid_masks.f2.sum())],
        "missing_rows": [int((~valid_masks.f1).sum()), int((~valid_masks.f2).sum())],
        "coverage_ratio": [float(valid_masks.f1.mean()), float(valid_masks.f2.mean())],
    }, index=pd.Index(["f1", "f2"], name="factor_id"))
    factors = FactorPanels(
        raw=raw,
        standardized=standardized,
        eligible=universe,
        valid_masks=valid_masks,
        common_mask=common_mask,
        coverage=coverage,
        correlations=pd.DataFrame(columns=["timestamp", "factor_a", "factor_b", "spearman", "pairwise_n"]),
        correlation_summary=pd.DataFrame(columns=["valid_periods", "mean", "median", "mean_abs", "p10", "p90",
                                                  "positive_share"]),
        input_diagnostics={"panel_source": "fixture"},
    )
    return inputs, factors, masked_time


class MultifactorWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract_dir = self.root / "contract"
        self.contract_dir.mkdir()
        self.start = pd.Timestamp("2025-01-01T00:00:00Z")
        self.end = pd.Timestamp("2025-01-01T05:00:00Z")
        self.inputs, self.factors, self.masked_time = _test_inputs(self.start, self.end)
        self.cards = []
        card_paths = []
        for card_id in ("f1", "f2"):
            card_path = self.contract_dir / f"{card_id}.json"
            payload = {"source_type": "factor_mining", "id": card_id, "title": card_id}
            card_path.write_text(json.dumps(payload), encoding="utf-8")
            card_paths.append(str(card_path))
            self.cards.append({
                "id": card_id,
                "title": card_id,
                "expression": "perp_close",
                "direction": 1,
                "horizon_hours": 1,
                "lookback_hours": 1,
                "card_snapshot": payload,
                "path": str(card_path),
            })
        self.contract_path = self.contract_dir / "contract.json"
        self.contract_path.write_text(json.dumps({
            "schema_version": 1,
            "run_id": "workflow_test",
            "purpose": "engineering",
            "stage": "development",
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "warmup_hours": 2,
            "horizon_hours": 1,
            "cards": card_paths,
            "universe": "universe.csv",
            "prior_data_use": "fixture data, never independent validation",
            "data_processing": "no fills; use source values as stored",
            "costs": {"initial_capital": 10000.0, "fee_bps": 10.0,
                      "slippage_bps": 5.0, "stress_multiplier": 2.0},
            "portfolio": {"long_count": 1, "short_count": 1, "gross_exposure": 0.4,
                          "max_asset_weight": 0.2, "rebalance_hours": 1, "margin_fraction": 0.1},
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_contract_to_costed_baseline_and_ablation_artifacts(self):
        output = self.root / "runs"
        with patch(f"{MODULE}.read_cards", return_value=self.cards), \
                patch(f"{MODULE}.load_inputs", return_value=self.inputs) as load_inputs, \
                patch(f"{MODULE}.process_factors", return_value=self.factors), \
                patch(f"{MODULE}._code_version", return_value={"git_commit": "fixture", "source_sha256": {}}):
            result = run_baseline(self.contract_path, self.root / "fixture.sqlite", output)

        load_inputs.assert_called_once()
        self.assertEqual(load_inputs.call_args.args[2], self.contract_dir.resolve())
        self.assertEqual(result["engine"], "deterministic_multifactor_v1")
        self.assertFalse(result["model_called"])
        self.assertFalse(result["agent_used"])
        self.assertEqual(result["status"], "engineering_complete")
        self.assertFalse(result["forward_validation_started"])
        self.assertTrue(Path(result["root"]).is_dir())
        self.assertEqual([item["name"] for item in result["experiments"]],
                         ["single:F1", "single:F2", "equal_weight", "drop:F1", "drop:F2"])
        self.assertEqual([(factor["code"], factor["id"]) for factor in result["factors"]],
                         [("F1", "f1"), ("F2", "f2")])

        run_root = output / "workflow_test"
        contract_copy = json.loads((run_root / "contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract_copy["run_id"], "workflow_test")
        usage = json.loads((run_root / "data-usage.json").read_text(encoding="utf-8"))
        self.assertFalse(usage["agent_used"])
        self.assertFalse(usage["upstream_card_status_changed"])
        self.assertFalse(usage["forward_validation_started"])
        self.assertEqual(len(json.loads((run_root / "cards.json").read_text(encoding="utf-8"))), 2)
        self.assertTrue((run_root / "inputs/panel.values.csv").is_file())
        self.assertTrue((run_root / "inputs/data-provenance.json").is_file())
        self.assertTrue((run_root / "factor_panels/correlation_summary.csv").is_file())
        self.assertEqual(result["coverage_impact"]["account_signal_window_eligible_rows"], 15)
        self.assertEqual(result["coverage_impact"]["account_signal_window_common_rows"], 12)
        self.assertAlmostEqual(result["coverage_impact"]["account_shared_coverage_ratio"], 0.8)

        signal_time = self.start - pd.Timedelta(hours=1)
        expected_equal = {"BTCUSDT": 0.75, "ETHUSDT": -0.5, "SOLUSDT": -0.25}
        for experiment in result["experiments"]:
            directory = run_root / experiment["path"]
            for filename in ("signals.csv", "targets.csv", "orders.csv", "fills.csv",
                             "funding_events.csv", "positions.csv", "ledger.csv", "metrics.json"):
                self.assertTrue((directory / filename).is_file(), filename)
            signals = pd.read_csv(directory / "signals.csv", index_col=0, parse_dates=[0])
            self.assertNotIn(self.end - pd.Timedelta(hours=1), signals.index)
            self.assertTrue(signals.loc[self.masked_time].isna().all())
            if experiment["name"] == "equal_weight":
                for symbol, value in expected_equal.items():
                    self.assertAlmostEqual(float(signals.loc[signal_time, symbol]), value)
            if experiment["name"] == "drop:F1":
                self.assertAlmostEqual(float(signals.loc[signal_time, "BTCUSDT"]), 0.5)
            targets = pd.read_csv(directory / "targets.csv", index_col=0, parse_dates=[0])
            self.assertEqual(targets.index[0], signal_time)
            self.assertTrue((targets.loc[self.masked_time] == 0).all())
            self.assertIn("net_return_delta_vs_equal_weight", experiment)
            self.assertEqual(experiment["delta_vs_equal_weight"]["net_return"],
                             experiment["net_return_delta_vs_equal_weight"])

        self.assertIsNotNone(result["stress_costs"])
        self.assertEqual(result["stress_costs"]["applies_to"], "equal_weight")
        self.assertEqual([path.name for path in (run_root / "experiments").glob("*stress*")],
                         ["equal_weight_stress"])
        stress_dir = run_root / "experiments/equal_weight_stress"
        self.assertTrue((stress_dir / "signals.csv").is_file())
        self.assertTrue((stress_dir / "funding_events.csv").is_file())
        self.assertEqual(result["stress_costs"]["multiplier"], 2.0)
        self.assertEqual(result["coverage_impact"]["account_signal_window_end_exclusive"],
                         "2025-01-01T04:00:00+00:00")
        report = (run_root / "report.md").read_text(encoding="utf-8")
        self.assertIn("| F1 | `f1` | f1 |", report)
        self.assertIn("single:F1", report)
        self.assertIn("剩余因子在相同共同样本上重新等权", report)
        self.assertIn("最后一小时只用于账户收盘估值", report)

        duplicate_output = self.root / "duplicate_runs"
        with patch(f"{MODULE}.read_cards", return_value=self.cards), \
                patch(f"{MODULE}.load_inputs", return_value=self.inputs), \
                patch(f"{MODULE}.process_factors", return_value=self.factors), \
                patch(f"{MODULE}._code_version", return_value={"git_commit": "fixture", "source_sha256": {}}):
            duplicate = run_baseline(self.contract_path, self.root / "fixture.sqlite", duplicate_output)
        self.assertEqual(result["experiments"], duplicate["experiments"])
        self.assertEqual(result["coverage_impact"], duplicate["coverage_impact"])
        for experiment in result["experiments"]:
            first_dir = run_root / experiment["path"]
            second_dir = duplicate_output / "workflow_test" / experiment["path"]
            self.assertEqual((first_dir / "ledger.csv").read_bytes(), (second_dir / "ledger.csv").read_bytes())

        with self.assertRaises(FileExistsError):
            run_baseline(self.contract_path, self.root / "fixture.sqlite", output)


if __name__ == "__main__":
    unittest.main()
