import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from crypto_quant.research.strategy_research import multifactor_data
from crypto_quant.research.strategy_research.multifactor_account import _performance_metrics
from crypto_quant.research.strategy_research.multifactor_contracts import MultifactorContract
from crypto_quant.research.strategy_research.multifactor_evidence import load_baseline_snapshot


BASELINE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/strategy_research/multifactor_v1/multifactor-engineering-20261002"
)


def _copy_snapshot(tmp_path):
    root = tmp_path / "snapshot"
    shutil.copytree(BASELINE_ROOT, root)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["root"] = str(root.resolve())
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def test_load_baseline_snapshot_uses_frozen_inputs_and_builds_compact_evidence(monkeypatch):
    def no_database_reads(*args, **kwargs):
        pytest.fail("baseline snapshot loader must not access the live database")

    monkeypatch.setattr(multifactor_data, "load_inputs", no_database_reads)
    snapshot = load_baseline_snapshot(BASELINE_ROOT, for_research_agent=True)

    assert snapshot.contract.run_id == "multifactor-engineering-20261002"
    assert len(snapshot.cards) == 3
    assert snapshot.inputs.panel.values.shape == (5050, 35)
    assert len(snapshot.inputs.frames) == 10
    assert len(snapshot.inputs.funding) == 420
    assert snapshot.factors.standardized.shape == (5050, 3)
    assert set(snapshot.records) == {
        "data_usage", "portfolio_rules", "baseline_metrics", "coverage", "correlations",
        "card:F1", "card:F2", "card:F3", "strategy:equal_weight",
        "single:F1", "single:F2", "single:F3", "drop:F1", "drop:F2", "drop:F3",
    }
    assert snapshot.records["coverage"]["scope"] == "account signal window only"
    assert "factor_panel_common_rows_including_warmup" not in snapshot.records["coverage"]["data"]["summary"]
    assert snapshot.records["coverage"]["data"]["summary"]["account_signal_window_eligible_rows"] == 3360
    assert snapshot.records["correlations"]["data"]["scope"] == "full input panel, including prewarm rows"
    assert snapshot.records["data_usage"]["data"]["source_causality"] == "retrospective_or_unknown"
    assert snapshot.records["card:F1"]["data"]["formula"]["expression"] == snapshot.cards[0]["expression"]
    assert snapshot.records["strategy:equal_weight"]["data"]["metrics"] == next(
        item["metrics"] for item in snapshot.result["experiments"] if item["name"] == "equal_weight"
    )
    assert all(record["run_id"] == snapshot.contract.run_id for record in snapshot.records.values())


def test_research_agent_rejects_internal_validation_before_reading_factor_files(tmp_path):
    root = _copy_snapshot(tmp_path)
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(schema_version=2, purpose="research", dataset_manifest="data/manifest.json")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(purpose="research", stage="internal_validation")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "cards.json").unlink()

    with pytest.raises(ValueError, match="research Agent may load only a development-stage snapshot"):
        load_baseline_snapshot(root, for_research_agent=True)


def test_snapshot_loader_rejects_card_snapshot_hash_mismatch(tmp_path):
    root = _copy_snapshot(tmp_path)
    manifest = json.loads((root / "cards.json").read_text(encoding="utf-8"))
    card_path = root / manifest[0]["snapshot_path"]
    card_path.write_bytes(card_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="sha256 differs from manifest"):
        load_baseline_snapshot(root)


def test_snapshot_loader_rejects_fractional_coverage_counts(tmp_path):
    root = _copy_snapshot(tmp_path)
    path = root / "coverage_impact.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["account_raw_valid_rows"] = frame["account_raw_valid_rows"].astype(float)
    frame.loc[0, "account_raw_valid_rows"] = float(frame.loc[0, "account_raw_valid_rows"]) + 0.2
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="coverage count differs from recomputed values"):
        load_baseline_snapshot(root)


@pytest.mark.parametrize("relative_path", [
    "inputs/panel.values.csv",
    "inputs/market/ADAUSDT.csv",
])
def test_snapshot_loader_rejects_symlinked_inputs_outside_root(tmp_path, relative_path):
    root = _copy_snapshot(tmp_path)
    outside = tmp_path / "external-input.csv"
    outside.write_text("outside snapshot data\n", encoding="utf-8")
    target = root / relative_path
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="path escapes the snapshot root"):
        load_baseline_snapshot(root)


@pytest.mark.parametrize("tamper,expected_error", [
    ("equity", "ledger equity differs from cash and unrealized PnL"),
    ("cash", "ledger cash differs from realized PnL, fees, and funding"),
])
def test_snapshot_loader_rejects_broken_ledger_identities_even_with_synced_metrics(
    tmp_path, tamper, expected_error,
):
    root = _copy_snapshot(tmp_path)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    experiment = result["experiments"][0]
    ledger_path = root / experiment["artifacts"]["ledger"]
    ledger = pd.read_csv(ledger_path, float_precision="round_trip")
    if tamper == "equity":
        ledger.loc[0, "equity"] += 5000.0
    else:
        ledger.loc[0, "cash"] += 5000.0
        ledger.loc[0, "equity"] += 5000.0
    ledger.to_csv(ledger_path, index=False)

    metrics_path = root / experiment["artifacts"]["metrics"]
    contract = MultifactorContract.from_dict(json.loads((root / "contract.json").read_text(encoding="utf-8")))
    metrics = _performance_metrics(
        ledger=ledger,
        initial_capital=contract.costs["initial_capital"],
        start=contract.bounds[0],
        end=contract.bounds[1],
        fills=pd.read_csv(root / experiment["artifacts"]["fills"], float_precision="round_trip"),
        funding_events=pd.read_csv(root / experiment["artifacts"]["funding_events"], float_precision="round_trip"),
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    experiment["metrics"] = metrics
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_baseline_snapshot(root)


@pytest.mark.parametrize("field,value,error", [
    ("source_causality", "original_causal", "overstates historical causality"),
    ("historical_causality_certified", True, "overstates historical causality"),
    ("policy_id", "different-policy", "policy or actual database processing"),
])
def test_snapshot_loader_rejects_overstated_data_provenance(tmp_path, field, value, error):
    root = _copy_snapshot(tmp_path)
    path = root / "inputs/data-provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance[field] = value
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_baseline_snapshot(root)


@pytest.mark.parametrize("relative_path", [
    "factor_panels/standardized.csv",
    "factor_panels/common_mask.csv",
])
def test_snapshot_loader_rejects_factor_panel_changes(tmp_path, relative_path):
    root = _copy_snapshot(tmp_path)
    path = root / relative_path
    frame = pd.read_csv(path, float_precision="round_trip")
    if relative_path.endswith("standardized.csv"):
        frame.iloc[0, 2] = float(frame.iloc[0, 2]) + 0.25
        expected_error = "recomputed standardized factors differ"
    else:
        frame.loc[0, "common_mask"] = not bool(frame.loc[0, "common_mask"])
        expected_error = "recomputed common mask differs"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=expected_error):
        load_baseline_snapshot(root)


@pytest.mark.parametrize("tamper,expected_error", [
    ("coverage", "coverage count differs from recomputed values"),
    ("correlations", "correlation summary differs from recomputed factor panels"),
    ("membership", "experiment factor membership differs from the frozen cards"),
    ("sample", "sample account_signal_window_common_rows differs from recomputed values"),
    ("metrics", "annualized_return vs ledger differs from recomputed values"),
    ("ledger", "ledger cash differs from realized PnL, fees, and funding"),
    ("external_path", "metrics path differs from its experiment directory"),
])
def test_snapshot_loader_rejects_tampered_research_evidence(tmp_path, tamper, expected_error):
    root = _copy_snapshot(tmp_path)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    if tamper == "coverage":
        path = root / "coverage_impact.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        frame.loc[0, "account_raw_valid_rows"] = -7
        frame.to_csv(path, index=False)
    elif tamper == "correlations":
        path = root / "factor_panels/correlation_summary.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        frame.loc[0, "mean_abs"] = 99.0
        frame.to_csv(path, index=False)
    elif tamper == "membership":
        result["experiments"][0]["included_card_ids"] = ["unlisted-card"]
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif tamper == "sample":
        result["experiments"][0]["sample"]["account_signal_window_common_rows"] = 1
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif tamper == "metrics":
        experiment = result["experiments"][0]
        experiment["metrics"]["annualized_return"] = 99.0
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics_path = root / experiment["artifacts"]["metrics"]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["annualized_return"] = 99.0
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif tamper == "ledger":
        ledger_path = root / result["experiments"][0]["artifacts"]["ledger"]
        frame = pd.read_csv(ledger_path, float_precision="round_trip")
        frame.loc[0, "funding_cashflow"] += 0.1
        frame.to_csv(ledger_path, index=False)
    else:
        result["experiments"][0]["artifacts"]["metrics"] = "../../external.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_baseline_snapshot(root)
