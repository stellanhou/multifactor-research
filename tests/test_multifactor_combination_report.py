import json
import gzip

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_account import run_perpetual_account
from crypto_quant.research.strategy_research.multifactor_combination_report import (
    build_report,
    refresh_morning_report,
    _utc_series,
    _selected_scope,
    _window_comparison,
    verify_account,
)


INITIAL_CAPITAL = 10_000.0
DEV_START = pd.Timestamp("2025-07-31T16:00:00Z")
DEV_END = pd.Timestamp("2025-08-01T00:00:00Z")
C_START = pd.Timestamp("2025-08-01T00:00:00Z")
C_END = pd.Timestamp("2026-08-01T00:00:00Z")


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"E4": {"status": "passed"}, "E5": {"status": "partial"}}, (68, 17)),
        ({"E4": {"status": "partial"}, "E5": {"status": "pending"}}, (52, 13)),
    ],
)
def test_selected_scope_uses_runner_active_statuses(statuses, expected):
    assert _selected_scope({"statuses": statuses}, []) == expected


def _account(directory, *, cost_multiplier=1, stage="development", route="stepwise-ridge-90d",
             experiment="E1", primary_selection_eligible=False, slope_scale=1.0,
             delist_busdt=False):
    directory.mkdir(parents=True)
    start, end = ((DEV_START, DEV_END) if stage == "development" else (C_START, C_END))
    if delist_busdt:
        end = start + pd.Timedelta(hours=72)
    index = pd.date_range(start - pd.Timedelta(hours=1), end, freq="h")
    step = np.arange(len(index), dtype=float)
    slope = (0.2 if stage == "development" else 0.005) * slope_scale
    prices = {
        "AUSDT": (100 + slope * step).tolist(),
        "BUSDT": (100 - slope / 2 * step).tolist(),
    }
    frames = {
        symbol: pd.DataFrame({"open": values, "close": values, "mark_close": values}, index=index)
        for symbol, values in prices.items()
    }
    if delist_busdt:
        inactive = index >= start + pd.Timedelta(hours=25)
        frames["BUSDT"]["inactive"] = inactive
        frames["BUSDT"].loc[inactive, ["open", "close", "mark_close"]] = np.nan
        targets = pd.DataFrame({"AUSDT": [0.25, 0.25, 0.25], "BUSDT": [-0.25, 0.0, 0.0]},
                               index=index[[0, 24, 48]])
    else:
        targets = pd.DataFrame({"AUSDT": [0.25], "BUSDT": [-0.25]}, index=index[:1])
    funding = pd.DataFrame({
        "timestamp": [start + pd.Timedelta(milliseconds=1),
                      start + pd.Timedelta(minutes=30),
                      start + pd.Timedelta(hours=4, milliseconds=5)],
        "symbol": ["AUSDT", "BUSDT", "AUSDT"],
        "funding_rate": [0.001, 0.001, 0.001],
        "mark_price": [100.0, 100.0, 106.0],
    })
    fee_bps, slippage_bps = 5.0 * cost_multiplier, 2.0 * cost_multiplier
    fit = start - pd.Timedelta(hours=1)
    selection_end = fit - pd.Timedelta(hours=25)
    selection_start = selection_end - pd.Timedelta(days=90) + pd.Timedelta(hours=1)
    account = run_perpetual_account(
        frames, targets, funding, initial_capital=INITIAL_CAPITAL,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        start=start, end=end, margin_fraction=0.05,
    )
    for name in ("orders", "fills", "funding_events", "positions", "ledger"):
        getattr(account, name).to_csv(directory / f"{name}.csv")
    (directory / "metrics.json").write_text(json.dumps(account.metrics))
    metadata = {
        "route_id": route, "route_family": "stepwise-ridge-h8",
        "method": "stepwise", "synthesis": "ridge", "selection_window_days": 90,
        "seed": 0, "stage": stage, "experiment": experiment,
        "primary_selection_eligible": primary_selection_eligible, "cost_multiplier": cost_multiplier,
        "initial_capital": INITIAL_CAPITAL, "start": start.isoformat(), "end": end.isoformat(),
        "fee_bps": fee_bps, "slippage_bps": slippage_bps,
        "r0_anchor": (start - pd.Timedelta(hours=1)).isoformat(),
        "portfolio": {"gross_exposure": 0.5, "max_asset_weight": 0.25,
                      "rebalance_hours": 24},
        "lifecycle_events": [],
        "model_schedule": [{
            "fit_timestamp": fit.isoformat(),
            "selected_factors": ["factor_a", "factor_b"],
            "coefficients": {"factor_a": 0.6, "factor_b": -0.4},
            "alpha": 0.01, "regularization": 0.1,
            "validation_net_sharpe": 1.2, "validation_net_return": 0.03,
            "validation_status": "selected",
            "train_start": (selection_start - pd.Timedelta(days=90)).isoformat(),
            "train_end": selection_start.isoformat(),
            "selection_start": selection_start.isoformat(),
            "selection_end": selection_end.isoformat(),
            "validation_account_start": (selection_start + pd.Timedelta(hours=1)).isoformat(),
            "validation_account_end": fit.isoformat(),
            "account_score_start": start.isoformat(), "account_score_end": end.isoformat(),
        }],
    }
    (directory / "account.json").write_text(json.dumps(metadata))
    return account


def test_independent_reconciliation_includes_funding_and_ending_unrealized(tmp_path):
    directory = tmp_path / "account"
    account = _account(directory)
    metadata = json.loads((directory / "account.json").read_text())

    verified = verify_account(directory, metadata)

    assert verified["status"] == "passed"
    assert verified["fills"] == 2
    assert verified["funding_events"] == 3
    assert verified["average_gross_exposure"] > 0
    assert np.isclose(account.metrics["net_return"], account.metrics["final_equity"] / INITIAL_CAPITAL - 1)


def test_independent_reconciliation_fails_on_cash_ledger_corruption(tmp_path):
    directory = tmp_path / "account"
    _account(directory)
    metadata = json.loads((directory / "account.json").read_text())
    ledger = pd.read_csv(directory / "ledger.csv", index_col=0)
    ledger.loc[ledger.index[2], "cash"] += 1.0
    ledger.to_csv(directory / "ledger.csv")

    with pytest.raises(ValueError, match=r"cash conservation|cash \+ unrealized"):
        verify_account(directory, metadata)


def test_utc_series_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        _utc_series(pd.Series(["2026-10-04 00:00:00.001+00:00", "not-a-timestamp"]),
                    "funding_events.timestamp")


def test_inactive_asset_is_flat_and_has_no_post_settlement_fills(tmp_path):
    directory = tmp_path / "account"
    _account(directory, delist_busdt=True)
    metadata = json.loads((directory / "account.json").read_text())

    result = verify_account(directory, metadata)
    positions = pd.read_csv(directory / "positions.csv")
    positions["timestamp"] = pd.to_datetime(positions["timestamp"], utc=True)
    fills = pd.read_csv(directory / "fills.csv")
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    orders = pd.read_csv(directory / "orders.csv")
    inactive = positions.loc[(positions.symbol == "BUSDT") & positions.mark_price.isna()]
    inactive_orders = orders.loc[(orders.symbol == "BUSDT") & orders.signal_close.isna()]

    assert result["status"] == "passed"
    assert not inactive.empty
    assert inactive.quantity.eq(0.0).all()
    assert not inactive_orders.empty
    assert inactive_orders.target_weight.eq(0.0).all()
    assert inactive_orders.current_quantity.eq(0.0).all()
    assert inactive_orders.target_quantity.eq(0.0).all()
    assert inactive_orders.signed_order_quantity.eq(0.0).all()
    assert not (fills.loc[fills.symbol == "BUSDT", "timestamp"] >= inactive.timestamp.min() - pd.Timedelta(hours=1)).any()


def test_report_ranks_only_verified_development_by_net_sharpe(tmp_path):
    run_root = tmp_path / "combination90"
    records = []
    for stage in ("development", "C"):
        for multiplier, name in ((1, "base"), (2, "stress")):
            relative = f"accounts/stepwise-ridge-90d/{stage}/{name}"
            _account(run_root / relative, cost_multiplier=multiplier, stage=stage)
            records.append({"path": relative})

    result = build_report(run_root, records)

    assert result["status"] == "partial"
    assert result["expected_account_count"] == 44
    assert result["development_winner"] is None
    assert result["development_leader"] is None
    assert result["qualified_development_routes"] == 1
    assert result["experiment_progress"]["E1"]["status"] == "incomplete"
    assert (run_root / "morning_report.md").is_file()
    assert (run_root / "plots/equity_development.svg").is_file()
    assert (run_root / "plots/equity_C.svg").is_file()
    comparison = pd.read_csv(run_root / "comparison.csv")
    assert set(comparison["stage"]) == {"development", "C"}
    assert comparison["selection_source_bars"].eq(2160).all()
    assert comparison["internal_validation_account_hours"].eq(2183).all()
    development = comparison.loc[(comparison.stage == "development") & (comparison.cost == "base")].iloc[0]
    assert development.qualified
    assert not development.primary_selection_eligible
    assert pd.isna(development.selection_rank)
    monthly = pd.read_csv(run_root / "monthly.csv")
    contributions = pd.read_csv(run_root / "symbol_contribution.csv")
    for row in comparison.itertuples():
        account_pnl = row.final_equity - INITIAL_CAPITAL
        attributed = contributions.loc[(contributions.route == row.route)
                                       & (contributions.stage == row.stage)
                                       & (contributions.cost == row.cost), "net_contribution"].sum()
        assert np.isclose(attributed, account_pnl)
    assert not monthly.empty
    report = (run_root / "morning_report.md").read_text()
    assert "C 段是已经使用过的历史评价数据" in report
    assert "尚无 E2/E3 主选范围内的达标路线" in report
    assert "没有启动前向或 Paper 账户" in report
    assert refresh_morning_report(run_root)["status"] == "partial"


def test_paired_candidate_report_uses_same_subset_and_best_ridge_alpha(tmp_path):
    run_root = tmp_path / "combination90"
    records = []
    for stage in ("development", "C"):
        for multiplier, name in ((1, "base"), (2, "stress")):
            relative = f"accounts/stepwise-ridge-90d/{stage}/{name}"
            _account(run_root / relative, cost_multiplier=multiplier, stage=stage)
            records.append({"path": relative})
    window = "2025-01-01T00:00:00Z"
    candidates = {
        "equal-route": [{"identity": ["factor_a", "factor_b"], "route_family": "stepwise",
                         "experiment": "E3", "route_id": "equal-route", "selection_window_days": 90,
                         "seed": 0, "method": "stepwise", "synthesis": "equal_weight", "alpha": None,
                         "validation_net_sharpe": 0.2, "valid_objective": True},
                     {"identity": ["factor_a", "factor_b"], "route_family": "stepwise",
                      "experiment": "E3", "route_id": "equal-route", "selection_window_days": 90,
                      "seed": 0, "method": "stepwise", "synthesis": "equal_weight", "alpha": None,
                      "validation_net_sharpe": 0.9, "valid_objective": False}],
        "ridge-route": [{"identity": ["factor_a", "factor_b"], "route_family": "stepwise",
                         "experiment": "E3", "route_id": "ridge-route", "selection_window_days": 90,
                         "seed": 0, "method": "stepwise", "synthesis": "ridge", "alpha": alpha,
                         "validation_net_sharpe": score, "valid_objective": True}
                        for alpha, score in ((0.00001, 0.1), (0.001, 0.3), (0.1, 0.25), (10.0, 0.0))],
    }
    for route, candidate_rows in candidates.items():
        directory = run_root / "routes" / route / "windows" / window
        directory.mkdir(parents=True)
        with gzip.open(directory / "candidate_trials.json.gz", "wt", encoding="utf-8") as output:
            json.dump({"candidate_rows": candidate_rows}, output)

    result = build_report(run_root, records)

    assert result["paired_score_status"]["status"] == "passed"
    paired = pd.read_csv(run_root / "paired_score_summary.csv").iloc[0]
    assert paired["paired_subsets"] == 1
    assert paired["equal_net_sharpe_mean"] == 0.2
    assert paired["ridge_best_net_sharpe_mean"] == 0.3
    assert paired["ridge_minus_equal_mean"] == pytest.approx(0.1)
    assert paired["best_ridge_alpha"] == 0.001


def test_empty_batch_is_reported_as_not_run(tmp_path):
    result = build_report(tmp_path / "empty", [])

    assert result["status"] == "not_run"
    assert result["account_count"] == 0
    assert result["verification"]["status"] == "not_run"
    assert result["experiment_progress"]["E1"]["status"] == "incomplete"
    assert (tmp_path / "empty" / "morning_report.md").is_file()


def test_account_guard_failure_keeps_timestamp_and_stays_unqualified(tmp_path):
    run_root = tmp_path / "combination90"
    directory = run_root / "accounts/stepwise-primary/development/base"
    _account(directory, route="stepwise-primary", experiment="E3", primary_selection_eligible=True)
    metadata = json.loads((directory / "account.json").read_text())
    metadata["account_status"] = "failed_at_account_guard"
    (directory / "account.json").write_text(json.dumps(metadata))
    (directory / "account_failure.json").write_text(json.dumps({
        "timestamp": "2025-07-31T19:00:00Z",
        "reason": "sampled margin guard breached at 2025-07-31T19:00:00Z",
    }))

    result = build_report(run_root, [{"path": str(directory.relative_to(run_root))}])

    assert result["status"] == "failed"
    check = result["verification"]["accounts"][0]
    assert "sampled margin guard breached at 2025-07-31T19:00:00Z" in check["errors"][0]
    comparison = pd.read_csv(run_root / "comparison.csv").iloc[0]
    assert not comparison.verified
    assert not comparison.qualified
    assert refresh_morning_report(run_root)["status"] == "failed"


def test_high_sharpe_e5_control_cannot_rank_over_e3_primary(tmp_path):
    run_root = tmp_path / "combination90"
    records = []
    routes = [
        ("stepwise-primary", "E3", True, 1.0),
        ("ga-seed-1-control", "E5", False, 2.0),
    ]
    for route, experiment, eligible, slope_scale in routes:
        for stage in ("development", "C"):
            for multiplier, cost in ((1, "base"), (2, "stress")):
                relative = f"accounts/{route}/{stage}/{cost}"
                _account(run_root / relative, cost_multiplier=multiplier, stage=stage,
                         route=route, experiment=experiment,
                         primary_selection_eligible=eligible, slope_scale=slope_scale)
                records.append({"path": relative})

    result = build_report(run_root, records)
    comparison = pd.read_csv(run_root / "comparison.csv")
    development = comparison.loc[(comparison.stage == "development") & (comparison.cost == "base")].set_index("route")

    assert development.loc["ga-seed-1-control", "sharpe_ratio"] > development.loc["stepwise-primary", "sharpe_ratio"]
    assert development.loc["ga-seed-1-control", "qualified"]
    assert not development.loc["ga-seed-1-control", "primary_selection_eligible"]
    assert pd.isna(development.loc["ga-seed-1-control", "selection_rank"])
    assert development.loc["stepwise-primary", "selection_rank"] == 1
    assert result["development_leader"]["route"] == "stepwise-primary"
    assert result["development_winner"] is None


def test_window_pair_keeps_e1_out_of_e3_capacity_comparison():
    rows = pd.DataFrame([
        {"experiment": "E1", "route_family": "stepwise-ridge-h8", "method": "stepwise",
         "synthesis": "ridge", "selection_window_days": 21, "seed": 0, "stage": "development",
         "cost": "base", "route": "e1-21", "net_return": 0.1, "sharpe_ratio": 0.5,
         "selection_source_bars": 504, "internal_validation_account_hours": 527,
         "average_gross_exposure": 0.7, "verified": True, "route_complete": True},
        {"experiment": "E1", "route_family": "stepwise-ridge-h8", "method": "stepwise",
         "synthesis": "ridge", "selection_window_days": 90, "seed": 0, "stage": "development",
         "cost": "base", "route": "e1-90", "net_return": 0.2, "sharpe_ratio": 0.8,
         "selection_source_bars": 2160, "internal_validation_account_hours": 2183,
         "average_gross_exposure": 0.6, "verified": True, "route_complete": True},
        {"experiment": "E3", "route_family": "stepwise-ridge-h8", "method": "stepwise",
         "synthesis": "ridge", "selection_window_days": 90, "seed": 0, "stage": "development",
         "cost": "base", "route": "e3-90", "net_return": 0.3, "sharpe_ratio": 1.2,
         "selection_source_bars": 2160, "internal_validation_account_hours": 2183,
         "average_gross_exposure": 0.5, "verified": True, "route_complete": True},
    ])

    paired = _window_comparison(rows)

    assert len(paired) == 1
    assert paired.iloc[0].route_21d == "e1-21"
    assert paired.iloc[0].route_90d == "e1-90"
    assert paired.iloc[0].selection_source_bars_21d == 504
    assert paired.iloc[0].selection_source_bars_90d == 2160
    assert paired.iloc[0].internal_validation_account_hours_21d == 527
    assert paired.iloc[0].internal_validation_account_hours_90d == 2183
