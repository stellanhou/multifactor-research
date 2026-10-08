"""Compare graduated portfolio targets using a frozen combination90 model run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from crypto_quant.research.strategy_research import multifactor_account as account
from crypto_quant.research.strategy_research import multifactor_combination_report as audit
from crypto_quant.research.strategy_research import multifactor_portfolio as portfolio

HOUR = pd.Timedelta(hours=1)
ARMS = ("R0", "TAPER5", "EMA50", "SCORE", "HOLD4")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def freeze_input(path: Path, manifest: dict, expected: str | None = None) -> None:
    value = sha(path)
    if expected is not None and value != expected:
        raise ValueError(f"frozen input checksum mismatch: {path}")
    manifest[str(path)] = value


def daily_factors(source: Path, contract: dict, manifest: dict) -> pd.DataFrame:
    pieces = []
    for stage in ("development", "internal_validation"):
        relative = f"{stage}/h{contract['horizon_hours']}/factor_panels/standardized.csv"
        path = source / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = next(stream); digest.update(header); selected = [header]
            for line in stream:
                digest.update(line)
                if int(line[11:13]) % contract["horizon_hours"] == contract["horizon_hours"] - 1:
                    selected.append(line)
        if digest.hexdigest() != contract["source_hashes"][relative]:
            raise ValueError(f"frozen factor input changed: {path}")
        manifest[str(path)] = digest.hexdigest()
        frame = pd.read_csv(io.BytesIO(b"".join(selected)), float_precision="round_trip")
        frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
        pieces.append(frame.set_index(["timestamp", "symbol"]).sort_index())
    joined = pd.concat([pieces[0], pieces[1].loc[~pieces[1].index.isin(pieces[0].index)]]).sort_index()
    return joined.loc[:, sorted(contract["factor_order"])]


def load_market(source: Path, stage: str, start, end, contract, manifest):
    frames = {}
    for path in sorted((source / stage / "inputs/market").glob("*.csv")):
        freeze_input(path, manifest, contract["source_hashes"][str(path.relative_to(source))])
        frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frames[path.stem] = frame.reindex(pd.date_range(start-HOUR, end-HOUR, freq="h"))
    path = source / stage / "inputs/funding.csv"
    freeze_input(path, manifest, contract["source_hashes"][str(path.relative_to(source))])
    funding = pd.read_csv(path, float_precision="round_trip")
    funding.timestamp = pd.to_datetime(funding.timestamp, utc=True, format="mixed")
    return frames, funding.loc[(funding.timestamp >= start) & (funding.timestamp < end)].copy()


def raw_scores(panel, names, symbols, metadata):
    start, end = pd.Timestamp(metadata["start"]), pd.Timestamp(metadata["end"])
    times = pd.date_range(start-HOUR, end-2*HOUR, freq=f"{metadata['r0_rebalance_hours']}h")
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    if not index.isin(panel.index).all():
        raise ValueError("factor panel does not cover every decision timestamp and symbol")
    values = panel.reindex(index).to_numpy(copy=True).reshape(len(times), len(symbols), len(names))
    complete = np.isfinite(values).all(axis=2)
    complete[complete.sum(axis=1) < 3] = False
    values[~complete] = np.nan
    schedule = metadata["model_schedule"]
    fits = pd.DatetimeIndex([item["fit_timestamp"] for item in schedule])
    fit_rows = np.searchsorted(fits.asi8, times.asi8, side="right")-1
    if (fit_rows < 0).any():
        raise ValueError("an account decision precedes the first frozen model")
    result = np.full((len(times), len(symbols)), np.nan)
    for i, k in enumerate(fit_rows):
        model = schedule[k]
        if model["status"] == "active":
            coefficients = np.array([model["coefficients"][name] for name in names])
            result[i] = np.einsum("tsf,f->ts", values[i:i+1], coefficients, optimize=True)[0]
    return pd.DataFrame(result, index=times, columns=symbols)


def rank_reference(scores):
    # Reconstruct the fixed R0 definition independently of the new constructors.
    target = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for timestamp, row in scores.iterrows():
        available = row.dropna().index.tolist()
        longs = sorted(available, key=lambda symbol: (-row[symbol], symbol))[:2]
        shorts = sorted([s for s in available if s not in longs], key=lambda s: (row[s], s))[:2]
        target.loc[timestamp, longs] = .2
        target.loc[timestamp, shorts] = -.2
    return target


def write_account(path, result, metadata, verification, summary):
    final_path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    save(path / "account.json", metadata)
    save(path / "metrics.json", result.metrics)
    save(path / "verification.json", verification)
    for name, frame in (("ledger", result.ledger), ("fills", result.fills),
                        ("orders", result.orders), ("positions", result.positions),
                        ("funding_events", result.funding_events)):
        frame.to_csv(path / f"{name}.csv.gz", index=name == "ledger",
                     compression={"method": "gzip", "compresslevel": 1})
    save(path / "summary.json", summary)
    path.rename(final_path)


def retention_targets(scores, raw):
    previous = dict.fromkeys(scores.columns, 0.)
    policy = account.RebalancePolicy(2, 2, .8, .2, holding_rank=4)
    rows = []
    for timestamp, row in scores.iterrows():
        chosen = account._rank_decision(row, previous, policy, raw.loc[timestamp].to_dict())[0]
        rows.append(chosen)
        previous = chosen
    return pd.DataFrame(rows, index=scores.index, columns=scores.columns)


def run(source_run: Path, output: Path) -> None:
    source_run, output = source_run.resolve(), output.resolve()
    original = json.loads((source_run / "run_contract.json").read_text())
    if original["horizon_hours"] not in {1, 4, 24} or original["rebalance"] != "R0":
        raise ValueError("this paired study requires the frozen 1h/4h/24h R0 model run")
    if original["portfolio"] != {"gross_exposure": .8, "long_count": 2, "margin_fraction": .1,
                                 "max_asset_weight": .2, "rebalance_hours": original["horizon_hours"], "short_count": 2}:
        raise ValueError("source portfolio differs from the declared comparison")
    if original["costs"] != {"fee_bps": 10., "initial_capital": 10000., "slippage_bps": 5., "stress_multiplier": 2.}:
        raise ValueError("source costs differ from the declared comparison")
    routes = [item["route_id"] for item in original["routes"] if item["experiment"] in {"E2", "E3"}]
    if len(routes) != 9:
        raise ValueError("the paired comparison requires all nine E2/E3 routes")
    expected_accounts = len(routes) * len(ARMS) * 4
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, expected in manifest.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"frozen paired-study source changed: {name}")
    for path in [Path(__file__).resolve(), Path(account.__file__), Path(portfolio.__file__), Path(audit.__file__)]:
        freeze_input(path, manifest)
        destination = output / "execution_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    freeze_input(source_run / "run_contract.json", manifest)
    contract = {
        "source_run": str(source_run), "horizon_hours": original["horizon_hours"], "routes": routes, "expected_accounts": expected_accounts,
        "arms": {
            "R0": "Original top/bottom two, 20% per asset, native horizon execution.",
            "TAPER5": "Top/bottom five, each side 40% split 5:4:3:2:1; disjoint equal seat counts; missing outer seats and clipped weights remain cash.",
            "EMA50": "50% desired R0 target plus 50% prior EMA target, initialized at cash. Ordinary zero desired weights decay. Ineligible assets and model-cash rows exit immediately. No renormalization.",
            "HOLD4": "Retain top/bottom four, enter top/bottom two; native horizon cadence; no weight buffer.",
            "SCORE": "Center composite scores at cross-sectional mean. Allocate proportionally to positive/negative centered scores, redistributing individual 20% cap excess; equal leg budget min(40%, capacity of each side). Flat cross-section is cash."
        },
        "fixed": "Original factor values, weekly model coefficients, lists, model admission, native horizon decision timestamps, development/C dates, source market/funding and transaction costs.",
        "costs": original["costs"], "primary_metric": "paired account net Sharpe",
        "checks": "Account reconciliation; original target equality and full original ledger reproduction; caps; net/gross exposure; cash and costs; turnover and width. EMA can lower gross exposure and change neutrality after a forced asset exit; report both explicitly.",
        "historical_scope": "Previously exposed development and C history. No C-based winner, new holdout, forward or Paper run.",
        "acceptance": "Original development gates: positive return and finite positive Sharpe, actual fills, max drawdown <=15%, double-cost return >=0.",
        "no_adaptive_parameter_search": True,
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("paired-study output already contains a different contract")
    save(contract_path, contract)
    save(manifest_path, manifest)
    save(output / "state.json", {"status": "preparing", "completed_accounts": 0, "expected_accounts": expected_accounts})
    source = Path(original["source_run"])
    panel = daily_factors(source, original, manifest)
    names = sorted(original["factor_order"])
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    rows, checks = [], []
    for stage, source_stage in (("development", "development"), ("C", "internal_validation")):
        sample = json.loads((source_run / "accounts" / routes[0] / stage / "base/account.json").read_text())
        start, end = pd.Timestamp(sample["start"]), pd.Timestamp(sample["end"])
        frames, funding = load_market(source, source_stage, start, end, original, manifest)
        for route in routes:
            base = source_run / "accounts" / route / stage / "base"
            freeze_input(base / "account.json", manifest)
            meta = json.loads((base / "account.json").read_text())
            scores = raw_scores(panel, names, symbols, meta)
            raw = rank_reference(scores)
            freeze_input(base / "orders.csv", manifest)
            orders = pd.read_csv(base / "orders.csv", float_precision="round_trip")
            orders.signal_timestamp = pd.to_datetime(orders.signal_timestamp, utc=True)
            targets_saved = orders.pivot(index="signal_timestamp", columns="symbol", values="target_weight").reindex(index=raw.index, columns=symbols)
            np.testing.assert_array_equal(raw.to_numpy(), targets_saved.to_numpy())
            targets_by_arm = {
                "R0": raw,
                "HOLD4": retention_targets(scores, raw),
                "TAPER5": portfolio.generate_tapered_targets(scores, side_count=5, gross_exposure=.8, max_asset_weight=.2),
                "EMA50": portfolio.smooth_target_weights(raw, scores.notna(), alpha=.5),
                "SCORE": portfolio.generate_score_weighted_targets(scores, gross_exposure=.8, max_asset_weight=.2),
            }
            for arm, target in targets_by_arm.items():
                if not np.isfinite(target.to_numpy()).all() or (target.abs().sum(axis=1) > .8+1e-12).any() \
                        or (target.abs() > .2+1e-12).any().any():
                    raise ValueError("portfolio target violates finite-value or exposure limits")
                if (target.where(~scores.notna(), 0) != 0).any().any():
                    raise ValueError("portfolio target retains an ineligible asset")
                target_dir = output / "accounts" / route / stage / arm
                target_dir.mkdir(parents=True, exist_ok=True)
                target.to_csv(target_dir / "targets.csv")
                for cost, multiplier in (("base", 1.), ("stress", 2.)):
                    destination = target_dir / cost
                    summary_path = destination / "summary.json"
                    if summary_path.exists():
                        row = json.loads(summary_path.read_text())
                        verification = json.loads((destination / "verification.json").read_text())
                        if verification["status"] != "passed" or (row["route"], row["stage"], row["arm"], row["cost"]) != (route, stage, arm, cost):
                            raise ValueError("completed account checkpoint is unverified or mismatched")
                        rows.append(row)
                        checks.append({"route": route, "stage": stage, "arm": arm, "cost": cost, **verification})
                        continue
                    tick = time.monotonic()
                    result = account.run_perpetual_account(frames, target, funding, initial_capital=10000.,
                        fee_bps=10*multiplier, slippage_bps=5*multiplier, start=start, end=end, margin_fraction=.1)
                    metadata = {**meta, "primary_selection_eligible": False, "execution_variant": arm,
                        "cost_path": cost, "cost_multiplier": multiplier, "fee_bps": 10*multiplier,
                        "slippage_bps": 5*multiplier, "target_rows": len(target)}
                    tables = {"ledger": result.ledger, "positions": result.positions, "fills": result.fills,
                        "orders": result.orders, "funding": result.funding_events, "metrics": result.metrics}
                    destination = target_dir / cost
                    verification = audit.verify_account(destination, metadata, tables)
                    if verification["status"] != "passed":
                        raise ValueError(f"account reconciliation failed: {verification}")
                    original_error = 0.
                    if arm == "R0":
                        old = source_run / "accounts" / route / stage / cost / "ledger.csv"
                        freeze_input(old, manifest)
                        saved = pd.read_csv(old, index_col=0, float_precision="round_trip")
                        for column in result.ledger.select_dtypes(include="number"):
                            np.testing.assert_allclose(result.ledger[column], saved[column], rtol=2e-10, atol=1e-7, equal_nan=True)
                        original_error = float(np.max(np.abs(result.ledger.equity.to_numpy()-saved.equity.to_numpy())))
                    ledger, metrics = result.ledger, result.metrics
                    row = {"route": route, "stage": stage, "arm": arm, "cost": cost, **metrics,
                        "average_gross_exposure": float((ledger.gross_notional/ledger.equity).mean()),
                        "average_net_exposure": float((ledger.net_notional/ledger.equity).mean()),
                        "mean_absolute_net_exposure": float((ledger.net_notional/ledger.equity).abs().mean()),
                        "average_target_gross": float(target.abs().sum(axis=1).mean()),
                        "average_target_holdings": float(target.ne(0).sum(axis=1).mean()),
                        "position_time_fraction": float((ledger.gross_notional>0).mean()),
                        "fills": len(result.fills), "gross_pnl_including_funding": float(metrics["final_equity"]-10000+metrics["total_fees"]+metrics["total_slippage_cost"]),
                        "baseline_max_equity_error": original_error, "verification": "passed"}
                    save(manifest_path, manifest)
                    write_account(destination, result, metadata, verification, row)
                    save(destination / "runtime.json", {"seconds": time.monotonic()-tick})
                    rows.append(row); checks.append({"route": route, "stage": stage, "arm": arm, "cost": cost, **verification})
                    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
                    save(output / "state.json", {"status": "running", "completed_accounts": len(rows), "expected_accounts": expected_accounts})
                    print(f"Completed {len(rows)}/{expected_accounts} {route} {stage} {arm} {cost}: {metrics['net_return']:.2%} ({time.monotonic()-tick:.1f}s)", flush=True)
    for path, value in manifest.items():
        if sha(Path(path)) != value:
            raise ValueError(f"source changed during the experiment: {path}")
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    save(output / "source_manifest.json", manifest)
    save(output / "verification.json", {"status": "passed", "accounts": len(checks),
        "baseline_accounts": len(routes) * 4, "max_baseline_equity_error": max(row["baseline_max_equity_error"] for row in rows), "checks": checks})
    save(output / "state.json", {"status": "complete", "completed_accounts": len(rows), "expected_accounts": expected_accounts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run, args.output)
