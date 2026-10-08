from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research import multifactor_combination_research as research


def _empty_market_funding():
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "symbol": pd.Series(dtype=str),
        "funding_rate": pd.Series(dtype=float),
        "mark_price": pd.Series(dtype=float),
    })


def _calendar_fixture(*, missing_validation_label=True):
    anchor = pd.Timestamp("2024-01-01T23:00:00Z")
    fit_time = anchor + pd.Timedelta(hours=27 * research.WEEK_HOURS)
    timestamps = pd.date_range(anchor, fit_time, freq="h", tz="UTC")
    windows = research.search.calendar_windows(fit_time)
    labels = {}
    for source in windows.training[::3]:
        labels[pd.Timestamp(source)] = (
            np.array([[-1.0, 0.5], [0.0, -1.0], [1.0, 0.5]]),
            np.array([-0.01, 0.0, 0.01]),
        )
    missing_source = windows.validation[100]
    for source in windows.validation:
        if missing_validation_label and source == missing_source:
            continue
        labels[pd.Timestamp(source)] = (
            np.array([[-1.0, 0.5], [0.0, -1.0], [1.0, 0.5]]),
            np.array([-0.01, 0.0, 0.01]),
        )
    cube = np.tile(np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
                   (len(timestamps), 1, 1))
    if missing_validation_label:
        missing_position = timestamps.get_loc(missing_source)
        cube[missing_position] = np.nan
    frames = {symbol: pd.DataFrame(index=timestamps) for symbol in ("A", "B", "C")}
    prepared = SimpleNamespace(
        timestamps=timestamps,
        frames=frames,
        symbols=["A", "B", "C"],
        market_full=SimpleNamespace(
            index=timestamps,
            position={pd.Timestamp(timestamp): position
                      for position, timestamp in enumerate(timestamps)},
        ),
        dataset=labels,
        factor_cube=cube,
        factor_names=["factor_a", "factor_b"],
        market_funding=_empty_market_funding(),
        dev_contract=SimpleNamespace(horizon_hours=24),
    )
    return prepared, fit_time, windows, missing_source


def test_primary_selection_rule_freezes_the_nine_e2_e3_routes_and_gates():
    rule = research.primary_selection_rule()
    route_ids = [
        "E2_full_elastic_net",
        "E2_full_equal_weight",
        "E2_full_ridge",
        "E3_ga0_equal_weight",
        "E3_ga0_ridge",
        "E3_grid_equal_weight",
        "E3_grid_ridge",
        "E3_stepwise_equal_weight",
        "E3_stepwise_ridge",
    ]

    assert rule["eligible_experiments"] == ["E2", "E3"]
    assert rule["eligible_route_ids"] == route_ids
    assert rule["excluded_experiments"] == ["E1", "E4", "E5"]
    assert rule["admission"] == {
        "stage": "development",
        "base": {
            "net_return": {"operator": ">", "value": 0.0},
            "net_sharpe": {"operator": ">", "value": 0.0, "finite": True},
            "traded_bars": {"operator": ">", "value": 0},
            "max_drawdown": {"operator": "<=", "value": 0.15},
        },
        "stress": {"net_return": {"operator": ">=", "value": 0.0}},
    }
    assert rule["ranking"] == {
        "primary_metric": "development_base_net_sharpe",
        "primary_direction": "descending",
        "tie_break_field": "route_id",
        "tie_break_direction": "ascending_lexicographic",
        "fixed_route_identity_order": route_ids,
    }


def test_execution_profile_preserves_the_corrected_original_fixed_budget():
    assert research.EXECUTION_PROFILE == {
        "max_heavy_processes": 2,
        "blas_threads_per_process": 1,
        "candidate_batch_size": 1024,
        "execution_start_utc": "2026-10-03T15:46:35Z",
        "extension_cutoff_utc": "2026-10-03T21:01:35Z",
        "target_deadline_utc": "2026-10-03T21:46:35Z",
        "extension_cutoff_elapsed_seconds": 18_900,
        "target_total_elapsed_seconds": 21_600,
        "execution_start_correction": {
            "previously_recorded_start_utc": "2026-10-03T14:26:35Z",
            "corrected_start_utc": "2026-10-03T15:46:35Z",
            "basis": "goal.createdAt Unix timestamp 1791042395",
        },
    }


def test_run_final_accounts_finishes_bookkeeping_for_four_saved_outputs(tmp_path: Path, monkeypatch):
    route = next(row for row in research.route_definitions()
                 if row["route_id"] == "E1_stepwise_ridge_v21")
    fit_time = pd.Timestamp("2025-01-01T23:00:00Z")
    contract = {"routes": research.route_definitions()}
    (tmp_path / "run_contract.json").write_text(json.dumps(contract, ensure_ascii=False))

    manifest = []
    model_schedule = [{"fit_timestamp": fit_time}]
    for stage in ("development", "C"):
        for cost_path in ("base", "stress"):
            relative = Path("accounts") / route["route_id"] / stage / cost_path
            account_dir = tmp_path / relative
            account_dir.mkdir(parents=True)
            research._write_json(account_dir / "account.json", {
                "route_id": route["route_id"], "stage": stage,
                "cost_path": cost_path, "model_schedule": model_schedule,
            })
            manifest.append({
                "route_id": route["route_id"], "stage": stage,
                "cost_path": cost_path, "path": relative.as_posix(),
                "elapsed_seconds": 1.0,
            })
    (tmp_path / "account_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    state = {
        "statuses": {"E1": {"stage_accounts": 0, "outputs": [], "elapsed_seconds": 0.0}},
        "routes": {route["route_id"]: {"status": "complete"}},
        "account_count": 0,
    }
    (tmp_path / "machine_state.json").write_text(json.dumps(state, ensure_ascii=False))
    monkeypatch.setattr(research, "_fit_records", lambda *_args: [{"fit_timestamp": fit_time}])
    monkeypatch.setattr(research, "common_fit_times", lambda *_args, **_kwargs: [fit_time])
    monkeypatch.setattr(research, "run_stage_account",
                        lambda *_args, **_kwargs: pytest.fail("saved account outputs must be reused"))
    report_roots = []
    monkeypatch.setattr(research, "refresh_report_and_status",
                        lambda root: report_roots.append(Path(root)))

    prepared = SimpleNamespace(validation_contract=SimpleNamespace(bounds=(fit_time, fit_time + pd.Timedelta(hours=1))))
    records = research.run_final_accounts(prepared, tmp_path, route["route_id"])

    expected_keys = {
        (route["route_id"], stage, cost_path)
        for stage in ("development", "C") for cost_path in ("base", "stress")
    }
    assert {(row["route_id"], row["stage"], row["cost_path"]) for row in records} == expected_keys
    saved_manifest = json.loads((tmp_path / "account_manifest.json").read_text(encoding="utf-8"))
    assert {(row["route_id"], row["stage"], row["cost_path"]) for row in saved_manifest} == expected_keys
    updated_state = json.loads((tmp_path / "machine_state.json").read_text(encoding="utf-8"))
    assert updated_state["statuses"]["E1"]["stage_accounts"] == 4
    assert set(updated_state["statuses"]["E1"]["outputs"]) == {row["path"] for row in manifest}
    assert updated_state["routes"][route["route_id"]]["stage_accounts"] == 4
    assert updated_state["account_count"] == 4
    assert report_roots == [tmp_path]


def test_e1_route_definitions_and_dispatch_use_stepwise_8_factor_2048_proposals(monkeypatch):
    routes = [route for route in research.route_definitions()
              if route["experiment"] == "E1"]
    calls = []

    class Evaluator:
        method = "ridge"

    evaluator = Evaluator()
    context = SimpleNamespace(
        prepared=SimpleNamespace(factor_names=[f"f{i}" for i in range(12)]),
        evaluator_factory=lambda method: evaluator,
    )

    def fake_run_search(actual_evaluator, **kwargs):
        calls.append((actual_evaluator, kwargs))
        return {"selected": [0, 1, 2, 3, 4, 5, 6, 7]}

    monkeypatch.setattr(research.search, "run_search", fake_run_search)

    assert [route["validation_hours"] for route in routes] == [21 * 24, 90 * 24]
    assert all(route["algorithm"] == "stepwise" and route["capacity"] == 8
               and route["budget"] == 2048 and route["budget_unit"] == "feature_subset_proposal"
               for route in routes)
    for route in routes:
        result, returned_evaluator, selected = research._route_search(context, route, cluster=None)
        assert result["selected"] == list(range(8))
        assert returned_evaluator is evaluator
        assert selected == tuple(range(8))
    assert len(calls) == 2
    for _actual_evaluator, kwargs in calls:
        assert kwargs["algorithm"] == "stepwise"
        assert kwargs["groups"] == [] and kwargs["grid"] == []
        assert kwargs["budget"] == 2048
        assert kwargs["size"] == 8
        assert kwargs["proposal_budget"] is True


def test_common_weekly_schedule_keeps_fit_when_a_validation_source_label_is_missing(monkeypatch):
    prepared, fit_time, windows, missing_source = _calendar_fixture()

    fits = research.common_fit_times(
        prepared, start=fit_time, end=fit_time + pd.Timedelta(hours=1),
    )
    assert fits == [fit_time]

    monkeypatch.setattr(research, "_slice_market", lambda *_args: None)
    monkeypatch.setattr(research, "_scope_market_funding", lambda *_args: _empty_market_funding())
    context = research.build_window_context(prepared, fit_time)

    assert context.readiness_status == "ready"
    assert context.validation_label_periods == len(windows.validation) - 1
    assert context.validation_times.equals(windows.validation)
    missing_position = context.validation_times.get_loc(missing_source)
    assert not context.validation_complete[missing_position].any()
    assert context.validation_values.shape[0] == 2160


def test_common_schedule_skips_training_market_prefix_and_requires_full_validation_market():
    factor_start = pd.Timestamp("2022-07-24T23:00:00Z")
    market_start = pd.Timestamp("2022-07-31T23:00:00Z")
    first_bad_fit = pd.Timestamp("2023-01-29T23:00:00Z")
    first_full_fit = pd.Timestamp("2023-02-05T23:00:00Z")
    factor_times = pd.date_range(factor_start, first_full_fit, freq="h", tz="UTC")
    market_times = pd.date_range(market_start, first_full_fit, freq="h", tz="UTC")
    assert factor_times[0] == pd.Timestamp("2022-07-24T23:00:00Z")
    assert market_times[0] == pd.Timestamp("2022-07-31T23:00:00Z")
    prepared = SimpleNamespace(
        timestamps=factor_times,
        dev_contract=SimpleNamespace(horizon_hours=24),
        symbols=["A"],
        frames={"A": pd.DataFrame(index=market_times)},
        market_full=SimpleNamespace(
            index=market_times,
            position={pd.Timestamp(timestamp): position
                      for position, timestamp in enumerate(market_times)},
        ),
    )

    bad_window = research.search.calendar_windows(first_bad_fit)
    full_window = research.search.calendar_windows(first_full_fit)
    assert bad_window.training[0] == pd.Timestamp("2022-07-31T22:00:00Z")
    assert full_window.training[0] == pd.Timestamp("2022-08-07T22:00:00Z")
    fits = research.common_fit_times(
        prepared, validation_hours=research.VALIDATION_HOURS,
        start=first_bad_fit, end=first_full_fit + research.HOUR,
    )
    assert fits == [first_full_fit]
    assert fits[0] + research.HOUR == pd.Timestamp("2023-02-06T00:00:00Z")

    missing_validation_hour = full_window.validation[100]
    gap_market_times = market_times.delete(market_times.get_loc(missing_validation_hour))
    prepared_with_validation_gap = SimpleNamespace(
        timestamps=prepared.timestamps,
        symbols=prepared.symbols,
        frames={"A": pd.DataFrame(index=gap_market_times)},
        market_full=SimpleNamespace(
            index=gap_market_times,
            position={pd.Timestamp(timestamp): position
                      for position, timestamp in enumerate(gap_market_times)},
        ),
    )
    assert not research._full_calendar(prepared_with_validation_gap, full_window)


def test_context_records_cash_readiness_when_validation_labels_fall_below_existing_minimum(monkeypatch):
    prepared, fit_time, windows, _missing_source = _calendar_fixture(missing_validation_label=False)
    for source in windows.validation[research.MIN_VALIDATION_LABEL_PERIODS - 1:]:
        prepared.dataset.pop(pd.Timestamp(source), None)

    monkeypatch.setattr(research, "_slice_market", lambda *_args: None)
    monkeypatch.setattr(research, "_scope_market_funding", lambda *_args: _empty_market_funding())
    context = research.build_window_context(prepared, fit_time)

    assert context.readiness_status == "insufficient_matured_validation_periods"
    assert context.validation_label_periods == research.MIN_VALIDATION_LABEL_PERIODS - 1
    assert len(context.validation_times) == research.VALIDATION_HOURS


def test_e1_route_keeps_all_common_weekly_fits_through_c_for_rolling_evaluation(
        tmp_path: Path, monkeypatch):
    route = next(row for row in research.route_definitions()
                 if row["route_id"] == "E1_stepwise_ridge_v90")
    dev_end = pd.Timestamp("2025-08-01T00:00:00Z")
    fit_times = [dev_end - pd.Timedelta(days=14), dev_end - pd.Timedelta(days=7),
                 dev_end, dev_end + pd.Timedelta(days=7)]
    run_id = "multifactor-combination90-20261003-r3"
    contract = {"run_id": run_id, "routes": research.route_definitions(),
                "source_manifest_hashes": {}}
    root = tmp_path / "run3"
    root.mkdir()
    (root / "run_contract.json").write_text(json.dumps(contract, ensure_ascii=False))
    state = {"statuses": {
        "E0": {"status": "passed"},
        "E1": {"status": "pending", "elapsed_seconds": 0.0},
    }, "routes": {}}
    (root / "machine_state.json").write_text(research.json.dumps(state))
    prepared = SimpleNamespace(
        dev_contract=SimpleNamespace(horizon_hours=24, bounds=(dev_end - pd.Timedelta(days=1095), dev_end)),
        validation_contract=SimpleNamespace(bounds=(dev_end, dev_end + pd.Timedelta(days=365))),
        verify_source_hashes=lambda: None,
    )
    monkeypatch.setattr(research, "_module_hashes", lambda _root: {})
    evaluated = []

    def fake_evaluate(_prepared, _root, _contract_sha, _route, fit_time, **_kwargs):
        evaluated.append(fit_time)
        return {"status": "cash_rejected_by_common_admission_gate"}

    monkeypatch.setattr(research, "evaluate_window", fake_evaluate)

    result = research.run_route(prepared, root, route["route_id"], fit_times=fit_times)

    assert evaluated == fit_times
    assert result["status"] == "complete"
    assert result["expected_fit_count"] == len(fit_times)
    assert result["completed_fit_count"] == len(fit_times)
    project_root = Path(research.__file__).resolve().parents[4]
    assert shlex.split(result["resume_entrypoint"]) == [
        sys.executable,
        str(project_root / "experiments/strategy_research/combination90_20261003/run_combination90.py"),
        "--action", "route", "--run-root", str(root.resolve()),
        "--run-id", run_id, "--route", route["route_id"],
    ]


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_outer_account_target_rows_follow_frozen_global_r0_phase(tmp_path: Path, monkeypatch, horizon):
    anchor = pd.Timestamp("2025-01-01T23:00:00Z")
    start = anchor + pd.Timedelta(hours=1)
    end = start + pd.Timedelta(hours=72)
    timestamps = pd.date_range(anchor, end - pd.Timedelta(hours=2), freq="h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    factor_cube = np.tile(np.array([[2.0], [1.0], [-1.0], [-2.0]]),
                          (len(timestamps), 1, 1))
    missing_r0_position = timestamps.get_loc(anchor + pd.Timedelta(hours=24))
    factor_cube[missing_r0_position] = np.nan
    frame_index = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=1),
                                freq="h", tz="UTC")
    frames = {symbol: pd.DataFrame({"open": 100.0, "close": 100.0, "mark_close": 100.0},
                                   index=frame_index) for symbol in symbols}
    portfolio = {"long_count": 2, "short_count": 2, "gross_exposure": 0.8,
                 "max_asset_weight": 0.2, "margin_fraction": 0.1}
    costs = {"initial_capital": 10_000.0, "fee_bps": 5.0,
             "slippage_bps": 2.0, "stress_multiplier": 2.0}
    contract = SimpleNamespace(horizon_hours=horizon, bounds=(start, end), portfolio=portfolio, costs=costs)
    prepared = SimpleNamespace(
        timestamps=timestamps, factor_cube=factor_cube, symbols=symbols,
        factor_names=["factor"], frames=frames, funding=_empty_market_funding(),
        dev_contract=contract, source_freeze_reference_sha256="frozen-source",
        source_hashes={}, validation_contract=contract, source_root=tmp_path,
    )
    (tmp_path / "lifecycle.json").write_text(json.dumps({"events": []}))
    fit = {
        "fit_timestamp": anchor, "status": "active", "coefficient_vector": np.array([1.0]),
        "selected_factors": ["factor"], "coefficients": {"factor": 1.0},
        "selected_alpha": 0.1, "validation_net_sharpe": 0.2, "validation_net_return": 0.01,
        "windows": {
            "training_source_start": anchor - pd.Timedelta(days=180),
            "training_source_end": anchor - pd.Timedelta(days=90),
            "validation_source_start": anchor - pd.Timedelta(days=90),
            "validation_source_end": anchor - pd.Timedelta(hours=25),
            "validation_account_start": anchor - pd.Timedelta(days=90) + pd.Timedelta(hours=1),
            "validation_account_end_exclusive": anchor,
        },
    }
    captured = {}

    def fake_account_run(_frames, targets, _funding, **kwargs):
        captured["targets"] = targets.copy()
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            orders=pd.DataFrame(), fills=pd.DataFrame(), funding_events=pd.DataFrame(),
            positions=pd.DataFrame(), ledger=pd.DataFrame(), metrics={"net_return": 0.0},
        )

    monkeypatch.setattr(research.account, "run_perpetual_account", fake_account_run)
    route = {"route_id": "phase-test", "route_family": "test", "experiment": "E1",
             "method": "stepwise", "synthesis": "ridge", "seed": 0,
             "selection_window_days": 90, "primary_selection_eligible": False}

    research.run_stage_account(prepared, tmp_path, route, [fit], "development")

    target_times = captured["targets"].index
    assert len(target_times) == 72 // horizon
    assert target_times[0] == anchor
    assert all((timestamp - anchor) % pd.Timedelta(hours=horizon) == pd.Timedelta(0)
               for timestamp in target_times)
    missing_target = captured["targets"].loc[anchor + pd.Timedelta(hours=24)]
    assert (missing_target == 0.0).all()
    assert captured["kwargs"]["start"] == start

    if horizon == 1:
        return  # Every hourly fit lies on the hourly rebalance phase.
    misaligned_fit = {**fit, "fit_timestamp": anchor + pd.Timedelta(hours=1)}
    with pytest.raises(ValueError, match="original R0 anchor phase"):
        research.run_stage_account(
            prepared, tmp_path, {**route, "route_id": "phase-mismatch"},
            [misaligned_fit], "development",
        )


def test_evaluate_window_resumes_matching_checkpoint_without_rewriting_it(tmp_path: Path, monkeypatch):
    route = {"route_id": "grid-equal", "experiment": "E3", "algorithm": "grid",
             "synthesis": "equal_weight", "seed": 0}
    fit_time = pd.Timestamp("2025-01-01T23:00:00Z")
    prepared = SimpleNamespace(source_freeze_reference_sha256="source-freeze")
    root = tmp_path
    window = root / "routes" / route["route_id"] / "windows" / "20250101T230000Z"
    trials_path = window / "candidate_trials.json.gz"
    fit_path = window / "fit.json.gz"
    identity = research._checkpoint_identity("contract-hash", route, fit_time, prepared)
    saved = {"identity_sha256": research._hash_json(identity), "status": "complete"}
    research._write_gzip_json(trials_path, {"identity_sha256": saved["identity_sha256"]})
    research._write_gzip_json(fit_path, saved)
    before_trials, before_fit = trials_path.read_bytes(), fit_path.read_bytes()
    monkeypatch.setattr(research, "build_window_context",
                        lambda *_args, **_kwargs: pytest.fail("resume must not rerun a completed window"))

    result = research.evaluate_window(prepared, root, "contract-hash", route, fit_time)

    assert result == saved
    assert trials_path.read_bytes() == before_trials
    assert fit_path.read_bytes() == before_fit


def test_evaluate_window_refuses_partial_or_foreign_checkpoint(tmp_path: Path):
    route = {"route_id": "stepwise", "experiment": "E1", "algorithm": "stepwise",
             "synthesis": "ridge", "seed": 0}
    fit_time = pd.Timestamp("2025-01-01T23:00:00Z")
    prepared = SimpleNamespace(source_freeze_reference_sha256="source-freeze")
    root = tmp_path
    window = root / "routes" / route["route_id"] / "windows" / "20250101T230000Z"
    window.mkdir(parents=True)
    trials_path, fit_path = window / "candidate_trials.json.gz", window / "fit.json.gz"

    trials_path.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete route checkpoint"):
        research.evaluate_window(prepared, root, "contract-hash", route, fit_time)
    trials_path.unlink()

    identity = research._checkpoint_identity("other-contract", route, fit_time, prepared)
    research._write_gzip_json(trials_path, {"identity_sha256": research._hash_json(identity)})
    research._write_gzip_json(fit_path, {"identity_sha256": research._hash_json(identity)})
    with pytest.raises(ValueError, match="another frozen contract"):
        research.evaluate_window(prepared, root, "contract-hash", route, fit_time)


def test_evaluate_window_refuses_candidate_trials_from_another_checkpoint(tmp_path: Path):
    route = {"route_id": "grid-equal", "experiment": "E3", "algorithm": "grid",
             "synthesis": "equal_weight", "seed": 0}
    fit_time = pd.Timestamp("2025-01-01T23:00:00Z")
    prepared = SimpleNamespace(source_freeze_reference_sha256="source-freeze")
    root = tmp_path
    window = root / "routes" / route["route_id"] / "windows" / "20250101T230000Z"
    trials_path, fit_path = window / "candidate_trials.json.gz", window / "fit.json.gz"
    identity = research._checkpoint_identity("contract-hash", route, fit_time, prepared)
    identity_sha256 = research._hash_json(identity)
    research._write_gzip_json(trials_path, {"identity_sha256": "other-window"})
    research._write_gzip_json(fit_path, {"identity_sha256": identity_sha256, "status": "complete"})

    with pytest.raises(ValueError, match="another frozen contract"):
        research.evaluate_window(prepared, root, "contract-hash", route, fit_time)


def test_frozen_source_manifest_requires_every_declared_implementation_source(tmp_path: Path):
    required = [
        "docs/plans/多因子组合算法90天选优实验Plan_20261003.md",
        "scripts/multifactor_horizon_turnover.py",
        "scripts/multifactor_position_validation.py",
        "src/crypto_quant/research/strategy_research/multifactor_portfolio.py",
        "pyproject.toml",
        "uv.lock",
        "src/crypto_quant/research/strategy_research/multifactor_combination_research.py",
        "src/crypto_quant/research/strategy_research/multifactor_combination_search.py",
        "src/crypto_quant/research/strategy_research/multifactor_combination_report.py",
        "src/crypto_quant/research/strategy_research/multifactor_combination_verify.py",
        "src/crypto_quant/research/strategy_research/multifactor_selection.py",
        "src/crypto_quant/research/strategy_research/multifactor_net_sharpe.py",
        "src/crypto_quant/research/strategy_research/multifactor_account.py",
        "src/crypto_quant/research/strategy_research/multifactor_selection_research.py",
        "src/crypto_quant/research/strategy_research/multifactor_contracts.py",
        "src/crypto_quant/research/strategy_research/multifactor_rebalance.py",
        "experiments/strategy_research/combination90_20261003/run_combination90.py",
        "tests/test_multifactor_horizon_turnover.py",
        "tests/test_multifactor_combination_research.py",
        "tests/test_multifactor_combination_search.py",
        "tests/test_multifactor_combination_report.py",
        "tests/test_multifactor_combination_verify.py",
    ]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "src/crypto_quant/research/strategy_research/multifactor_combination_verify.py":
            continue
        path.write_text("frozen test source\n")

    with pytest.raises(FileNotFoundError, match="multifactor_combination_verify.py"):
        research._module_hashes(tmp_path)

    verifier = tmp_path / "src/crypto_quant/research/strategy_research/multifactor_combination_verify.py"
    verifier.write_text("frozen verifier source\n")
    verifier_test = tmp_path / "tests/test_multifactor_combination_verify.py"
    verifier_test.write_text("frozen verifier test\n")
    manifest = research._module_hashes(tmp_path)
    assert "src/crypto_quant/research/strategy_research/multifactor_combination_verify.py" in manifest


def test_route_score_frame_switches_weekly_and_preserves_rejected_update_as_cash():
    anchor = pd.Timestamp("2025-01-01T23:00:00Z")
    fit_times = [anchor, anchor + pd.Timedelta(hours=168), anchor + pd.Timedelta(hours=336)]
    end = anchor + pd.Timedelta(hours=400)
    signal_times = pd.date_range(anchor, end - pd.Timedelta(hours=2), freq="h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    per_symbol = np.array([[2.0], [1.0], [-1.0], [-2.0]])
    factor_cube = np.tile(per_symbol, (len(signal_times), 1, 1))
    prepared = SimpleNamespace(
        timestamps=signal_times,
        factor_cube=factor_cube,
        symbols=symbols,
        factor_names=["factor"],
        dev_contract=SimpleNamespace(horizon_hours=24, bounds=(anchor + pd.Timedelta(hours=1), end)),
    )

    def fit(timestamp, *, status, coefficient):
        return {
            "fit_timestamp": timestamp,
            "status": status,
            "coefficient_vector": None if coefficient is None else np.array([coefficient]),
            "selected_factors": [] if coefficient is None else ["factor"],
            "coefficients": None if coefficient is None else {"factor": coefficient},
            "selected_alpha": None,
            "validation_net_sharpe": None,
            "validation_net_return": None,
            "windows": {
                "training_source_start": timestamp - pd.Timedelta(days=180),
                "training_source_end": timestamp - pd.Timedelta(days=90),
                "validation_source_start": timestamp - pd.Timedelta(days=90),
                "validation_source_end": timestamp - pd.Timedelta(hours=25),
                "validation_account_start": timestamp - pd.Timedelta(days=90) + pd.Timedelta(hours=1),
                "validation_account_end_exclusive": timestamp,
            },
        }

    fits = [fit(fit_times[0], status="active", coefficient=1.0),
            fit(fit_times[1], status="cash_rejected_by_common_admission_gate", coefficient=None),
            fit(fit_times[2], status="active", coefficient=2.0)]

    scores, schedule = research._route_score_frame(
        prepared, fits, start=anchor + pd.Timedelta(hours=1), end=end,
    )

    assert scores.index.equals(signal_times)
    assert len(scores) == len(signal_times)
    assert scores.loc[anchor].tolist() == [2.0, 1.0, -1.0, -2.0]
    assert scores.loc[anchor + pd.Timedelta(hours=24)].tolist() == [2.0, 1.0, -1.0, -2.0]
    assert scores.loc[fit_times[1]].isna().all()
    assert scores.loc[fit_times[1] + pd.Timedelta(hours=24)].isna().all()
    assert scores.loc[fit_times[2]].tolist() == [4.0, 2.0, -2.0, -4.0]
    assert [row["fit_timestamp"] for row in schedule] == fit_times
    assert [row["status"] for row in schedule] == [
        "active", "cash_rejected_by_common_admission_gate", "active",
    ]
