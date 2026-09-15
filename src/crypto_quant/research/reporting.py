from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from crypto_quant.backtesting.backtest import BacktestResult, bars_per_year, run_backtest
from crypto_quant.backtesting.config import BacktestConfig
from crypto_quant.data_access.data import interval_to_ms, load_klines, validate_klines
import numpy as np
from crypto_quant.backtesting.metrics import STANDARD_RISK_RETURN_FIELDS, performance_metrics
from crypto_quant.research.provenance import build_source_manifest
from crypto_quant.strategies.strategies import StrategySpec, strategy_registry
from crypto_quant.backtesting.validation import cross_symbol_risk_status, risk_gate


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not pd.isna(value) and not np.isfinite(value):
        return None
    raise TypeError(f"not serializable: {type(value)}")


def _sanitize_json(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_round(values: Iterable[float], digits: int = 6) -> List[float]:
    output = []
    for value in values:
        output.append(round(float(value), digits) if pd.notna(value) else None)
    return output


def _strategy_metrics_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    cross_symbol: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for symbol, by_strategy in results.items():
        for strategy_name, periods in by_strategy.items():
            row: Dict[str, Any] = {"symbol": symbol, "strategy": strategy_name}
            for period in ("full", "train", "out_of_sample"):
                metrics = periods[period]
                row.update(
                    {
                        f"{period}_annualized_return": metrics["annualized_return"],
                        f"{period}_sharpe": metrics["sharpe_ratio"],
                        f"{period}_sortino": metrics["sortino_ratio"],
                        f"{period}_max_drawdown": metrics["max_drawdown"],
                        f"{period}_annualized_volatility": metrics["annualized_volatility"],
                        f"{period}_win_rate": metrics["trade_win_rate"],
                        f"{period}_payoff_ratio": metrics["payoff_ratio"],
                        f"{period}_turnover": metrics["turnover_annualized"],
                        **{
                            field: metrics[field]
                            for field in STANDARD_RISK_RETURN_FIELDS
                        },
                    }
                )
            oos = periods["out_of_sample"]
            row.update(
                {
                    "trade_count_oos": oos["trade_count"],
                    "p95_participation_oos": oos["p95_trade_participation"],
                    "execution_feasible": oos["execution_feasible_at_research_size"],
                    "risk_status": periods["risk"]["status"],
                    "cross_symbol_status": cross_symbol[strategy_name],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _markdown_report(
    title: str,
    test_start: str,
    symbols: List[str],
    specs: Dict[str, StrategySpec],
    quality: Dict[str, Dict[str, Any]],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    table: pd.DataFrame,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Fixed out-of-sample start: **{test_start}**",
        "",
        "## Research Contract",
        "",
        "- Signals use information through bar close.",
        "- Orders execute at the next bar open.",
        "- Costs are 10 bps commission plus 5 bps slippage per side.",
        "- The same predeclared rules apply across symbols and periods.",
        "- No parameter search was performed inside this study.",
        "",
        "## Data Quality",
        "",
    ]
    for symbol in symbols:
        item = quality[symbol]
        lines.append(
            f"- **{symbol}**: {item['validation']['rows']} bars, "
            f"{item['validation']['missing_bars_by_span']} missing intervals by span, "
            f"{item['validation']['gap_locations']} gap locations."
        )

    lines.extend(["", "## Strategy Hypotheses", ""])
    for name, spec in specs.items():
        if name == "buy_and_hold":
            continue
        lines.append(f"### {name}")
        lines.append(f"- Family: {spec.family}")
        lines.append(f"- Hypothesis: {spec.hypothesis}")
        lines.append(f"- Why it might work: {spec.rationale}")
        lines.append(f"- Failure modes: {spec.failure_modes}")
        lines.append(f"- Parameters: `{spec.parameters}`")
        lines.append("")

    lines.extend(
        [
            "## Results",
            "",
            "`full` includes both periods. `train` ends immediately before the "
            "out-of-sample date. `out_of_sample` starts on that date. Win rate is "
            "based on realized reductions/exits; buy-and-hold therefore has no trade win rate.",
            "",
            table.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Decision Rule",
            "",
            "A strategy is only a research candidate when every risk gate passes. ",
            "The stricter cross-symbol status requires the same rule to pass on every evaluated asset. ",
            "Candidate status does not imply live deployment approval.",
            "",
            "## Next Iteration",
            "",
            "1. Preserve failed strategies in the ledger and record why they failed.",
            "2. Extend only the hypotheses that survive this fixed split.",
            "3. Add futures funding, open interest, breadth, and stress-period analysis before any execution work.",
            "4. Use walk-forward and parameter-neighborhood checks, not best-parameter selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_first_study(
    db_path: Path,
    output_root: Path,
    symbols: List[str],
    interval: str,
    start: str,
    test_start: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_first_spot_study"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = strategy_registry()
    quality: Dict[str, Dict[str, Any]] = {}
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    config = BacktestConfig()
    test_timestamp = pd.Timestamp(test_start, tz="UTC")

    for symbol in symbols:
        data = load_klines(db_path, symbol, interval, start=start)
        validation = validate_klines(data, interval)
        quality[symbol] = {
            "interval": interval,
            "requested_start": start,
            "first_available": data.index[0].isoformat(),
            "last_available": data.index[-1].isoformat(),
            "validation": validation,
        }
        annualization = bars_per_year(data.index, interval_to_ms(interval) // 1000)
        results[symbol] = {}
        for name, spec in specs.items():
            params = dict(spec.parameters)
            params["bars_per_year"] = int(round(annualization))
            targets = spec.generate(data, params)
            result = run_backtest(data, targets, config=config)
            full = performance_metrics(result, annualization)
            train = performance_metrics(
                result, annualization, end=test_timestamp - pd.Timedelta(seconds=1)
            )
            out_of_sample = performance_metrics(result, annualization, start=test_timestamp)
            gate = risk_gate(out_of_sample)
            results[symbol][name] = {
                "full": full,
                "train": train,
                "out_of_sample": out_of_sample,
                "risk": gate,
                "hypothesis": spec.hypothesis,
                "rationale": spec.rationale,
                "failure_modes": spec.failure_modes,
                "parameters": params,
            }

            pd.DataFrame(
                {
                    "equity": result.equity,
                    "benchmark_equity": result.benchmark_equity,
                    "target_weight": result.weights["target_weight"],
                    "executed_weight": result.weights["executed_weight"],
                    "turnover": result.weights["turnover"],
                }
            ).to_csv(run_dir / f"{symbol}_{name}_equity.csv")
            result.trades.to_csv(run_dir / f"{symbol}_{name}_trades.csv", index=False)

    cross_symbol = cross_symbol_risk_status(results)
    table = _strategy_metrics_table(results, cross_symbol)
    table.to_csv(run_dir / "summary.csv", index=False)
    source_manifest = build_source_manifest()
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": "first_spot_study",
        "symbols": symbols,
        "interval": interval,
        "requested_start": start,
        "test_start": test_start,
        "config": config.__dict__,
        "data_quality": quality,
        "results": results,
        "cross_symbol_risk_status": cross_symbol,
        "database_sha256": file_hash(db_path),
        "source_provenance": source_manifest,
    }
    strict_payload = _sanitize_json(payload)
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(strict_payload, handle, indent=2, allow_nan=False)

    markdown = _markdown_report(
        "First Binance Spot Study", test_start, symbols, specs, quality, results, table
    )
    report_path = run_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    ledger_path = output_root / "ledger.jsonl"
    ledger_record = {
        "run_id": run_id,
        "created_at": payload["created_at"],
        "study": payload["study"],
        "database_sha256": payload["database_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "symbols": symbols,
        "interval": interval,
        "test_start": test_start,
        "risk_status": {
            symbol: {
                strategy: values["risk"]["status"]
                for strategy, values in by_strategy.items()
            }
            for symbol, by_strategy in results.items()
        },
        "cross_symbol_risk_status": cross_symbol,
        "report_path": str(report_path),
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_record, default=_json_default) + "\n")
    return run_dir
