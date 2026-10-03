from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from statistics import NormalDist

from crypto_quant.backtesting.backtest import BacktestConfig, bars_per_year
from crypto_quant.data_access.data import interval_to_ms, load_klines, validate_klines
from crypto_quant.research.integrity import verify_artifact_manifest
from crypto_quant.backtesting.metrics import performance_metrics
from crypto_quant.backtesting.portfolio import (
    PORTFOLIO_NO_TRADE_BAND,
    PORTFOLIO_STRATEGY,
    SLEEVE_WEIGHTS,
    run_portfolio_backtest,
)
from crypto_quant.research.provenance import build_source_manifest, spot_klines_digest
from crypto_quant.research.reporting import _sanitize_json, file_hash
from crypto_quant.strategies.strategies import strategy_registry
from crypto_quant.backtesting.validation import RISK_GATES, risk_gate


EULER_MASCHERONI = 0.5772156649015329
SURVIVAL_DSR_THRESHOLD = 0.95
WATCH_DSR_THRESHOLD = 0.80

# Pinned evidence prevents later studies from silently expanding the trial set.
SPOT_ROBUSTNESS_RUN = "20260823T085502Z_spot_robustness_study"
PORTFOLIO_NEIGHBORHOOD_RUN = "20260823T085526Z_donchian_portfolio_neighborhood"
DIVERSIFICATION_RUN = "20260823T085534Z_donchian_portfolio_diversification"
TRIAL_SOURCE_RUNS = (
    SPOT_ROBUSTNESS_RUN,
    PORTFOLIO_NEIGHBORHOOD_RUN,
    DIVERSIFICATION_RUN,
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pinned evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"pinned evidence is not an object: {path}")
    return payload


def load_pinned_trial_sharpes(
    output_root: Path,
    spot_klines_sha256: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load fixed OOS trials; the legacy input fingerprint is not compared."""
    runs_root = output_root / "runs"
    evidence: Dict[str, Any] = {}
    frames: List[pd.DataFrame] = []

    definitions = [
        (
            "spot_parameter_and_time_robustness",
            SPOT_ROBUSTNESS_RUN,
            "neighborhood_results.csv",
        ),
        (
            "portfolio_parameter_neighborhood",
            PORTFOLIO_NEIGHBORHOOD_RUN,
            "neighborhood_results.csv",
        ),
        (
            "portfolio_diversification_stability",
            DIVERSIFICATION_RUN,
            "allocation_scenarios.csv",
        ),
    ]
    for study, run_id, filename in definitions:
        directory = runs_root / run_id
        manifest = verify_artifact_manifest(directory)
        if manifest["status"] != "valid":
            raise ValueError(
                f"pinned evidence manifest is {manifest['status']}: {directory}"
            )
        payload = _read_json(directory / "results.json")
        recorded_hash = str(payload.get("database_sha256", ""))
        recorded_spot_hash = str(payload.get("spot_klines_sha256", ""))

        frame = pd.read_csv(directory / filename)
        if study == "spot_parameter_and_time_robustness":
            active = frame[frame["strategy"] != "buy_and_hold"].copy()
            active["trial_id"] = (
                active["symbol"].astype(str)
                + "|"
                + active["strategy"].astype(str)
                + "|"
                + active["params_canonical"].astype(str)
            )
            active = active.drop_duplicates("trial_id")
            value_field = "out_of_sample_sharpe"
        elif study == "portfolio_parameter_neighborhood":
            active = frame.copy()
            active["trial_id"] = "joint_portfolio|" + active[
                "variant_id"
            ].astype(str)
            value_field = "out_of_sample_sharpe_ratio"
        else:
            active = frame[~frame["is_declared"].astype(bool)].copy()
            active["trial_id"] = "sleeve_mix|" + active["scenario"].astype(str)
            value_field = "out_of_sample_sharpe_ratio"

        missing = {"trial_id", value_field} - set(active.columns)
        if missing:
            raise ValueError(f"pinned trial source {study} missing fields: {sorted(missing)}")
        output = pd.DataFrame(
            {
                "source": study,
                "source_run_id": run_id,
                "trial_id": active["trial_id"].astype(str),
                "oos_sharpe_annualized": pd.to_numeric(
                    active[value_field], errors="raise"
                ),
            }
        )
        if output["trial_id"].duplicated().any():
            raise ValueError(f"duplicate pinned trial identifier in {study}")
        evidence[study] = {
            "run_id": run_id,
            "database_sha256": recorded_hash,
            "spot_klines_sha256": recorded_spot_hash,
            "manifest_status": manifest["status"],
            "trial_count": int(len(output)),
        }
        frames.append(output)

    trials = pd.concat(frames, ignore_index=True)
    if trials.empty:
        raise ValueError("pinned multiple-testing evidence contains no trials")
    if not np.isfinite(trials["oos_sharpe_annualized"]).all():
        raise ValueError("pinned trial Sharpes must be finite")
    if trials["trial_id"].duplicated().any():
        raise ValueError("trial identifiers must be unique across pinned sources")
    return trials, evidence


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    observed_sharpe_annualized: float,
    benchmark_sharpe_annualized: float,
    bars_per_year_value: float,
) -> float:
    """Calculate PSR against a benchmark using the observed return moments."""
    if len(returns) < 3:
        raise ValueError("PSR needs at least three return observations")
    if bars_per_year_value <= 0 or not np.isfinite(returns).all():
        raise ValueError("PSR needs a positive frequency and finite returns")

    series = pd.Series(returns, dtype=float)
    observed = float(observed_sharpe_annualized) / math.sqrt(bars_per_year_value)
    benchmark = float(benchmark_sharpe_annualized) / math.sqrt(bars_per_year_value)
    skew = float(series.skew())
    kurtosis = float(series.kurtosis() + 3.0)
    denominator = 1.0 - skew * observed + ((kurtosis - 1.0) / 4.0) * observed**2
    if denominator <= 0.0:
        raise ValueError("PSR moment denominator is non-positive")
    statistic = (
        (observed - benchmark)
        * math.sqrt(len(returns) - 1.0)
        / math.sqrt(denominator)
    )
    return float(NormalDist().cdf(statistic))


def expected_maximum_sharpe_annualized(
    trial_sharpes_annualized: np.ndarray,
    bars_per_year_value: float,
) -> float:
    """Estimate the largest of N independent trial Sharpes (Lopez de Prado)."""
    values = np.asarray(trial_sharpes_annualized, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("expected-max calculation needs at least two finite trials")
    if bars_per_year_value <= 0:
        raise ValueError("bars per year must be positive")

    per_period_values = values / math.sqrt(bars_per_year_value)
    variance = float(np.var(per_period_values, ddof=1))
    n_trials = int(values.size)
    normal = NormalDist()
    first_tail = normal.inv_cdf(1.0 - 1.0 / n_trials)
    second_tail = normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    scale = (1.0 - EULER_MASCHERONI) * first_tail
    scale += EULER_MASCHERONI * second_tail
    return float(math.sqrt(variance) * scale * math.sqrt(bars_per_year_value))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trial_sharpes_annualized: np.ndarray,
    bars_per_year_value: float,
) -> Dict[str, float]:
    """Compute ordinary PSR and its penalty for the expected best of N trials."""
    series = pd.Series(returns, dtype=float)
    volatility = float(series.std(ddof=1))
    if volatility <= 0.0:
        raise ValueError("observed returns must have positive dispersion")
    observed_sharpe = float(series.mean() / volatility * math.sqrt(bars_per_year_value))
    threshold = expected_maximum_sharpe_annualized(
        trial_sharpes_annualized, bars_per_year_value
    )
    naive_psr = probabilistic_sharpe_ratio(
        returns=returns,
        observed_sharpe_annualized=observed_sharpe,
        benchmark_sharpe_annualized=0.0,
        bars_per_year_value=bars_per_year_value,
    )
    dsr = probabilistic_sharpe_ratio(
        returns=returns,
        observed_sharpe_annualized=observed_sharpe,
        benchmark_sharpe_annualized=threshold,
        bars_per_year_value=bars_per_year_value,
    )
    return {
        "observed_sharpe_annualized": observed_sharpe,
        "expected_max_sharpe_annualized": threshold,
        "naive_probabilistic_sharpe_vs_zero": naive_psr,
        "deflated_sharpe_probability": dsr,
    }


def classify_multiple_testing(dsr: float) -> str:
    if dsr >= SURVIVAL_DSR_THRESHOLD:
        return "multiple_testing_survives"
    if dsr >= WATCH_DSR_THRESHOLD:
        return "multiple_testing_watch"
    return "multiple_testing_not_established"


def _load_candidate_result(
    db_path: Path,
    symbols: List[str],
    interval: str,
    start: str,
    test_start: str,
) -> tuple[Any, Dict[str, Dict[str, Any]], float]:
    normalized_symbols = [symbol.upper() for symbol in symbols]
    if set(normalized_symbols) != set(SLEEVE_WEIGHTS):
        raise ValueError("current audit requires exactly BTCUSDT and ETHUSDT")

    spec = strategy_registry()[PORTFOLIO_STRATEGY]
    config = BacktestConfig(min_trade_fraction=PORTFOLIO_NO_TRADE_BAND)
    split_timestamp = pd.Timestamp(test_start, tz="UTC")
    raw_data: Dict[str, pd.DataFrame] = {}
    quality: Dict[str, Dict[str, Any]] = {}
    for raw_symbol in normalized_symbols:
        symbol = raw_symbol.upper()
        data = load_klines(db_path, symbol, interval=interval, start=start)
        validation = validate_klines(data, interval)
        quality[symbol] = {
            "interval": interval,
            "requested_start": start,
            "first_available": data.index[0].isoformat(),
            "last_available": data.index[-1].isoformat(),
            "validation": validation,
        }
        raw_data[symbol] = data

    common_index = raw_data[normalized_symbols[0]].index
    for symbol in normalized_symbols[1:]:
        common_index = common_index.intersection(raw_data[symbol].index)
    market_data = {
        symbol: raw_data[symbol].loc[common_index]
        for symbol in normalized_symbols
    }
    annualization = bars_per_year(common_index, interval_to_ms(interval) // 1000)
    parameters = dict(spec.parameters)
    parameters["bars_per_year"] = int(round(annualization))
    targets = {
        symbol: spec.generate(market_data[symbol], parameters)
        for symbol in normalized_symbols
    }
    result, _ = run_portfolio_backtest(
        market_data=market_data,
        target_weights=targets,
        sleeve_weights={
            symbol: SLEEVE_WEIGHTS[symbol] for symbol in normalized_symbols
        },
        config=config,
    )
    return result, quality, annualization


def _markdown_report(
    test_start: str,
    quality: Dict[str, Dict[str, Any]],
    evidence: Dict[str, Any],
    trial_summary: pd.DataFrame,
    observed: Dict[str, Any],
    diagnostics: Dict[str, float],
    status: str,
) -> str:
    display_observed = {
        "oos_years": observed["years"],
        "oos_return": observed["total_return"],
        "oos_sharpe": diagnostics["observed_sharpe_annualized"],
        "oos_sortino": observed["sortino_ratio"],
        "oos_max_drawdown": observed["max_drawdown"],
        "annualized_turnover": observed["turnover_annualized"],
    }
    lines = [
        "# Donchian Portfolio Multiple-Testing Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Fixed out-of-sample start: **{test_start}**",
        "",
        "## Protocol",
        "",
        "- The trial set is pinned to three sealed experiments, not chosen after",
        "  inspecting this correction.",
        "- It counts unique active spot rule/config/symbol OOS results, every joint",
        "  Donchian portfolio parameter neighbor, and every nondeclared sleeve mix.",
        "- The unchanged declared BTC/ETH 50/50 portfolio is replayed once from local",
        "  data for return-moment calculations.",
        "- Deflated Sharpe Ratio asks whether the observed Sharpe exceeds an",
        "  estimated best-of-N threshold, while adjusting for return skew and",
        "  kurtosis. It does not select or replace the declared strategy.",
        "- Predeclared triage: at least 0.95 survives, 0.80-0.95 is watch, and",
        "  below 0.80 is not established.",
        "",
        "## Data",
        "",
    ]
    for symbol, item in quality.items():
        lines.append(
            f"- **{symbol}**: {item['validation']['rows']} complete bars; "
            f"{item['validation']['missing_bars_by_span']} missing intervals."
        )
    lines.extend(
        [
            "",
            "## Pinned Trial Sources",
            "",
            pd.DataFrame(
                [
                    {
                        "source": name,
                        **values,
                    }
                    for name, values in evidence.items()
                ]
            ).to_markdown(index=False),
            "",
            trial_summary.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Selection-Adjusted Diagnostics",
            "",
            pd.DataFrame([display_observed]).to_markdown(
                index=False, floatfmt=".3f"
            ),
            "",
            f"- Expected-max Sharpe threshold: "
            f"**{diagnostics['expected_max_sharpe_annualized']:.3f}**.",
            f"- Ordinary PSR versus zero: "
            f"**{diagnostics['naive_probabilistic_sharpe_vs_zero']:.1%}**.",
            f"- Deflated Sharpe probability: "
            f"**{diagnostics['deflated_sharpe_probability']:.1%}**.",
            f"- Triage: **`{status}`**.",
            "",
            "## Limits",
            "",
            "- Treating correlated trials as independent is intentionally conservative;",
            "  it does not turn robustness checks into new strategy discoveries.",
            "- Execution-stress cells are diagnostics rather than promotion candidates,",
            "  so they are documented separately and not added to N.",
            "- DSR cannot repair lookahead bias, cost errors, short history, regime",
            "  dependence, liquidity constraints, or absent forward confirmation.",
            "- This statistical watch does not approve live execution or parameter changes.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_multiple_testing_audit(
    db_path: Path,
    output_root: Path,
    symbols: List[str],
    interval: str,
    start: str,
    test_start: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_donchian_portfolio_multiple_testing"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = build_source_manifest()
    database_sha256 = file_hash(db_path)
    spot_klines_sha256 = spot_klines_digest(db_path)
    trials, evidence = load_pinned_trial_sharpes(
        output_root, spot_klines_sha256
    )
    result, quality, annualization = _load_candidate_result(
        db_path=db_path,
        symbols=symbols,
        interval=interval,
        start=start,
        test_start=test_start,
    )
    split_timestamp = pd.Timestamp(test_start, tz="UTC")
    observed_metrics = performance_metrics(
        result, annualization, start=split_timestamp
    )
    gate = risk_gate(observed_metrics)
    oos_returns = (
        result.equity.loc[split_timestamp:].pct_change().dropna().to_numpy(dtype=float)
    )
    diagnostics = deflated_sharpe_ratio(
        returns=oos_returns,
        trial_sharpes_annualized=trials["oos_sharpe_annualized"].to_numpy(),
        bars_per_year_value=annualization,
    )
    status = classify_multiple_testing(
        diagnostics["deflated_sharpe_probability"]
    )

    trials.sort_values(["source", "trial_id"]).to_csv(
        run_dir / "trial_sharpes.csv", index=False
    )
    trial_summary = (
        trials.groupby("source", sort=True)["oos_sharpe_annualized"]
        .agg(trial_count="count", min_sharpe="min", median_sharpe="median", max_sharpe="max")
        .reset_index()
    )
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "donchian_portfolio_multiple_testing_audit",
        "strategy": PORTFOLIO_STRATEGY,
        "symbols": [symbol.upper() for symbol in symbols],
        "interval": interval,
        "requested_start": start,
        "fixed_test_start": test_start,
        "config": BacktestConfig(min_trade_fraction=PORTFOLIO_NO_TRADE_BAND).__dict__,
        "source_provenance": source_manifest,
        "risk_gates": RISK_GATES,
        "pinned_source_runs": list(TRIAL_SOURCE_RUNS),
        "source_evidence": evidence,
        "trial_count": int(len(trials)),
        "status": status,
        "fixed_oos_risk_gate": gate,
        "diagnostics": diagnostics,
        "observed_out_of_sample_metrics": observed_metrics,
        "limits": [
            "Independent-trial DSR is conservative when tested strategies overlap.",
            "The pinned trial set changes only through an explicit rebaseline.",
            "This audit does not alter the paper session or approve execution.",
        ],
        "database_sha256": database_sha256,
        "spot_klines_sha256": spot_klines_sha256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(payload), handle, indent=2, allow_nan=False)
    markdown = _markdown_report(
        test_start=test_start,
        quality=quality,
        evidence=evidence,
        trial_summary=trial_summary,
        observed=observed_metrics,
        diagnostics=diagnostics,
        status=status,
    )
    report_path = run_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    ledger_record = {
        "run_id": run_id,
        "created_at": payload["created_at"],
        "study": payload["study"],
        "database_sha256": database_sha256,
        "spot_klines_sha256": spot_klines_sha256,
        "symbols": payload["symbols"],
        "interval": interval,
        "fixed_test_start": test_start,
        "status": status,
        "oos_risk_status": gate["status"],
        "pinned_source_runs": list(TRIAL_SOURCE_RUNS),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "trial_count": int(len(trials)),
        "observed_oos_sharpe": diagnostics["observed_sharpe_annualized"],
        "expected_max_sharpe": diagnostics["expected_max_sharpe_annualized"],
        "deflated_sharpe_probability": diagnostics[
            "deflated_sharpe_probability"
        ],
        "report_path": str(report_path),
    }
    with (output_root / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_sanitize_json(ledger_record)) + "\n")
    return run_dir
