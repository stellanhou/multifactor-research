from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from crypto_quant.data_access.data import update_klines
from crypto_quant.data_access.data_quality import run_data_quality_report
from crypto_quant.research.agents import audit_data, run_research_agents
from crypto_quant.research.audit import run_system_audit
from crypto_quant.backtesting.capacity import run_portfolio_capacity_study
from crypto_quant.backtesting.capacity_replay import run_portfolio_capacity_replay_study
from crypto_quant.backtesting.diagnostics import run_strategy_autopsy
from crypto_quant.backtesting.portfolio import run_donchian_portfolio_study
from crypto_quant.backtesting.portfolio_robustness import run_portfolio_neighborhood_study
from crypto_quant.backtesting.portfolio_regimes import run_portfolio_regime_study
from crypto_quant.backtesting.portfolio_drawdowns import run_portfolio_drawdown_study
from crypto_quant.backtesting.portfolio_diversification import run_portfolio_diversification_study
from crypto_quant.backtesting.portfolio_margin_stress import run_portfolio_margin_stress
from crypto_quant.backtesting.portfolio_stress import run_portfolio_stress
from crypto_quant.data_access.futures import (
    BinanceFuturesClient,
    update_funding_mark_prices_from_archive,
    update_funding_from_archive,
    update_funding_from_rest,
)
from crypto_quant.backtesting.forward_review import run_forward_paper_review
from crypto_quant.research.integrity import seal_research_artifacts
from crypto_quant.backtesting.multiple_testing import run_multiple_testing_audit
from crypto_quant.data_access.open_interest import BinanceOpenInterestArchiveClient
from crypto_quant.strategies.open_interest_study import (
    export_rc01_pinned_trials,
    run_open_interest_confirmed_donchian_study,
)
from crypto_quant.backtesting.order_book_replay import run_order_book_replay_study
from crypto_quant.strategies.positioning_study import run_positioning_extremes_event_study
from crypto_quant.data_access.open_interest import (
    update_open_interest_from_archive,
)
from crypto_quant.research.integrity import enable_paper_append_manifest
from crypto_quant.research.reporting_funding import run_funding_study
from crypto_quant.execution.paper import initialize_paper_session, update_paper_session
from crypto_quant.execution.shadow import (
    ShadowSession,
    replay_shadow_events,
    run_public_stream,
)
from crypto_quant.execution.testnet import (
    RequestsTransport,
    TestnetClient,
    TestnetCredentials,
    TestnetRehearsal,
    TestnetSession,
)
from crypto_quant.execution.demo import DemoClient, DemoCredentials, DemoRehearsal, DemoSession
from crypto_quant.execution.futures_demo import (
    FuturesDemoSession,
    make_futures_demo_orchestrator,
)
from crypto_quant.execution.strategy_platform import MetricInputs, STRATEGY_FAMILIES, StrategyPlatform
from crypto_quant.execution.demo_platform import DemoExecutionOrchestrator, make_demo_orchestrator
from crypto_quant.data_access.data_requests import DataRequestStore, build_inventory_plan, build_dynamic_top50_plan, resume_inventory_plan, run_inventory_plan, run_top50_plan
from crypto_quant.data_access.futures_backfill import build_um_futures_core_plan, repair_partial_metrics, run_um_futures_core_plan, um_futures_core_status
from crypto_quant.data_access.market_data import MarketDataStore
from crypto_quant.features.factors import FactorEngine, factor_catalog
from crypto_quant.features.factor_evaluation import DEFAULT_FACTOR_UNIVERSE, FactorEvaluationConfig, run_factor_evaluation
from crypto_quant.strategies.positioning_contrarian import (
    PositioningContrarianConfig,
    run_positioning_contrarian_study,
)
from crypto_quant.data_access.liquidation_data import (
    align_liquidation_history,
    audit_liquidation_alignment,
    audit_liquidation_history,
    download_liquidation_history,
    liquidation_ingestion_status,
    prepare_liquidation_universe,
)
from crypto_quant.research.reporting import run_first_study
from crypto_quant.strategies.basket_trend import run_basket_trend_study
from crypto_quant.strategies.basket_vol_beta import run_basket_vol_beta_study
from crypto_quant.strategies.relative_strength import run_cross_sectional_study
from crypto_quant.strategies.risk_parity_vol_target import (
    run_risk_parity_study,
    run_risk_parity_v2_confirmatory_study,
)
from crypto_quant.backtesting.robustness import run_robustness_study
from crypto_quant.research.strategy_library import run_strategy_library_snapshot
from crypto_quant.backtesting.stress import run_execution_stress
from crypto_quant.backtesting.uncertainty import run_portfolio_bootstrap_study
from crypto_quant.backtesting.walk_forward import run_donchian_portfolio_walk_forward


DEFAULT_DB = Path("market_data/crypto_quant.sqlite")
DEFAULT_OUTPUT = Path("experiments")


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="crypto-quant",
        description="Personal crypto quant research system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from crypto_quant.research.factor_mining.cli import add_arguments as add_mining_arguments
    add_mining_arguments(subparsers.add_parser("factor-mine", help="因子挖掘、固定批次验证和创意入池"))

    update = subparsers.add_parser("update-data", help="download public Binance spot klines")
    update.add_argument(
        "--authorize-public-download",
        required=True,
        action="store_true",
        help="explicitly authorize this public-network download",
    )
    update.add_argument("--db", type=Path, default=DEFAULT_DB)
    update.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    update.add_argument("--interval", default="4h")
    update.add_argument("--start", default="2019-01-01")
    update.add_argument("--end", default=None)

    data_request_create = subparsers.add_parser(
        "data-request-create", help="create a deterministic public-data request"
    )
    data_request_create.add_argument("--spec-json", type=Path, default=None)
    data_request_create.add_argument("--strategy-version-id", default="")
    data_request_create.add_argument("--reason", default="")
    data_request_create.add_argument("--datasets-json", type=Path, default=None)
    data_request_create.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    data_request_create.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))

    data_request_status = subparsers.add_parser(
        "data-request-status", help="show one data request without downloading"
    )
    data_request_status.add_argument("--request-id", required=True)
    data_request_status.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    data_request_status.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))

    data_request_fulfill = subparsers.add_parser(
        "data-request-fulfill", help="fill only missing Binance public-data ranges"
    )
    data_request_fulfill.add_argument("--request-id", required=True)
    data_request_fulfill.add_argument("--db", type=Path, default=DEFAULT_DB)
    data_request_fulfill.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    data_request_fulfill.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))
    data_request_fulfill.add_argument("--authorize-public-download", action="store_true")

    research_resume = subparsers.add_parser(
        "research-resume-status", help="show whether a Codex research checkpoint can resume"
    )
    research_resume.add_argument("--checkpoint-id", default=None)
    research_resume.add_argument("--strategy-version-id", default=None)
    research_resume.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    research_resume.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))

    research_checkpoint = subparsers.add_parser(
        "research-checkpoint-create", help="save a deterministic Codex research resume checkpoint"
    )
    research_checkpoint.add_argument("--strategy-version-id", required=True)
    research_checkpoint.add_argument("--request-ids", default="")
    research_checkpoint.add_argument("--completed-steps", default="")
    research_checkpoint.add_argument("--next-action", default="run_candidate_backtest")
    research_checkpoint.add_argument("--stage", default="waiting_for_data")
    research_checkpoint.add_argument("--hypothesis", default="")
    research_checkpoint.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    research_checkpoint.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))

    data_request_consume = subparsers.add_parser(
        "data-request-consume", help="mark one ready request consumed"
    )
    data_request_consume.add_argument("--request-id", required=True)
    data_request_consume.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    data_request_consume.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))

    inventory_plan = subparsers.add_parser(
        "data-inventory-plan", help="write a no-download universe/backfill plan"
    )
    inventory_plan.add_argument("--db", type=Path, default=DEFAULT_DB)
    inventory_plan.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    inventory_plan.add_argument("--symbols", default="")
    inventory_plan.add_argument("--interval", default="1h")
    inventory_plan.add_argument("--top-n", type=int, default=50)
    inventory_plan.add_argument("--discover", action="store_true", help="discover current and historical symbols from official endpoints")
    inventory_plan.add_argument("--batch-size", type=int, default=50)
    inventory_plan.add_argument("--materialize", action="store_true", help="create DataRequestStore batch requests; does not download")
    inventory_plan.add_argument("--max-symbols", type=int, default=None)

    inventory_resume = subparsers.add_parser("data-inventory-resume", help="refresh batch statuses and select the next inventory batch")
    inventory_resume.add_argument("--plan-id", required=True)
    inventory_resume.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    inventory_run = subparsers.add_parser("data-inventory-run", help="execute uncompleted inventory batches sequentially")
    inventory_run.add_argument("--plan-id", required=True)
    inventory_run.add_argument("--db", type=Path, default=DEFAULT_DB)
    inventory_run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    inventory_run.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))
    inventory_run.add_argument("--max-batches", type=int, default=None)
    inventory_run.add_argument("--authorize-public-download", action="store_true")

    top50_plan = subparsers.add_parser("data-top50-plan", help="plan historical rolling quote-volume Top-N 5m requests")
    top50_plan.add_argument("--db", type=Path, default=DEFAULT_DB)
    top50_plan.add_argument("--inventory-plan", type=Path, required=True)
    top50_plan.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    top50_plan.add_argument("--top-n", type=int, default=50)
    top50_plan.add_argument("--window-days", type=int, default=30)
    top50_plan.add_argument("--step-days", type=int, default=30)
    top50_plan.add_argument("--materialize", action="store_true")

    top50_run = subparsers.add_parser("data-top50-run", help="execute uncompleted historical Top-N 5m requests sequentially")
    top50_run.add_argument("--plan-id", required=True)
    top50_run.add_argument("--db", type=Path, default=DEFAULT_DB)
    top50_run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    top50_run.add_argument("--raw-root", type=Path, default=Path("market_data/raw"))
    top50_run.add_argument("--max-requests", type=int, default=None)
    top50_run.add_argument("--authorize-public-download", action="store_true")

    futures_core_plan = subparsers.add_parser("futures-core-plan", help="discover all archived USDT-M perpetual core-data prefixes")
    futures_core_plan.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    futures_core_plan.add_argument("--symbols", default="")
    futures_core_plan.add_argument("--max-symbols", type=int, default=None)

    futures_core_run = subparsers.add_parser("futures-core-run", help="run or resume the USDT-M perpetual core-data backfill")
    futures_core_run.add_argument("--plan-id", required=True)
    futures_core_run.add_argument("--db", type=Path, default=DEFAULT_DB)
    futures_core_run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    futures_core_run.add_argument("--workers", type=int, default=8)
    futures_core_run.add_argument("--chunk-files", type=int, default=24)
    futures_core_run.add_argument("--max-tasks", type=int, default=None)
    futures_core_run.add_argument("--max-files", type=int, default=None)
    futures_core_run.add_argument("--authorize-public-download", action="store_true")

    futures_core_status = subparsers.add_parser("futures-core-status", help="show USDT-M perpetual core backfill progress")
    futures_core_status.add_argument("--plan-id", required=True)
    futures_core_status.add_argument("--db", type=Path, default=DEFAULT_DB)
    futures_core_status.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    futures_metrics_repair = subparsers.add_parser("futures-core-repair-metrics", help="reset metric archives whose optional fields were rejected")
    futures_metrics_repair.add_argument("--plan-id", required=True)
    futures_metrics_repair.add_argument("--db", type=Path, default=DEFAULT_DB)
    futures_metrics_repair.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    market_catalog = subparsers.add_parser(
        "market-data-catalog",
        help="show unified local spot and USD-M data coverage",
    )
    market_catalog.add_argument("--db", type=Path, default=DEFAULT_DB)
    market_catalog.add_argument(
        "--symbols",
        default="",
        help="optional comma-separated symbols for exact per-symbol coverage",
    )

    liquidation_plan = subparsers.add_parser(
        "liquidation-plan",
        help="freeze the local/provider liquidation symbol and time overlap",
    )
    liquidation_plan.add_argument("--db", type=Path, default=DEFAULT_DB)
    liquidation_plan.add_argument(
        "--raw-root", type=Path, default=Path("market_data/raw/liquidations")
    )

    liquidation_download = subparsers.add_parser(
        "liquidation-download",
        help="download the exact CryptoHFTData liquidation overlap",
    )
    liquidation_download.add_argument("--db", type=Path, default=DEFAULT_DB)
    liquidation_download.add_argument(
        "--raw-root", type=Path, default=Path("market_data/raw/liquidations")
    )
    liquidation_download.add_argument("--workers", type=int, default=16)
    liquidation_download.add_argument("--start-date", default=None)
    liquidation_download.add_argument("--end-date", default=None)
    liquidation_download.add_argument(
        "--api-key-env", default="CRYPTOHFTDATA_API_KEY"
    )
    liquidation_download.add_argument(
        "--authorize-public-download", action="store_true"
    )

    liquidation_align = subparsers.add_parser(
        "liquidation-align",
        help="align raw liquidations with local OHLCV, funding, and OI",
    )
    liquidation_align.add_argument("--db", type=Path, default=DEFAULT_DB)
    liquidation_align.add_argument(
        "--raw-root", type=Path, default=Path("market_data/raw/liquidations")
    )
    liquidation_align.add_argument(
        "--output",
        type=Path,
        default=Path("market_data/derived/liquidation_alignment"),
    )
    liquidation_status = subparsers.add_parser(
        "liquidation-status",
        help="show resumable liquidation download and alignment readiness",
    )
    liquidation_status.add_argument(
        "--raw-root", type=Path, default=Path("market_data/raw/liquidations")
    )
    liquidation_audit = subparsers.add_parser(
        "liquidation-audit",
        help="verify raw liquidation archives and normalized hourly hashes",
    )
    liquidation_audit.add_argument(
        "--raw-root", type=Path, default=Path("market_data/raw/liquidations")
    )
    liquidation_alignment_audit = subparsers.add_parser(
        "liquidation-alignment-audit",
        help="audit causal and structural invariants in aligned liquidations",
    )
    liquidation_alignment_audit.add_argument(
        "--output",
        type=Path,
        default=Path("market_data/derived/liquidation_alignment"),
    )

    subparsers.add_parser(
        "market-factor-catalog",
        help="show causal derived-factor definitions",
    )
    factor_sample = subparsers.add_parser(
        "market-factor-sample",
        help="build a compact local factor sample for one symbol",
    )
    factor_sample.add_argument("--db", type=Path, default=DEFAULT_DB)
    factor_sample.add_argument("--symbol", required=True)
    factor_sample.add_argument("--interval", default="1h")
    factor_sample.add_argument(
        "--base-market",
        choices=("spot", "usd_m_perpetual"),
        default="usd_m_perpetual",
    )
    factor_sample.add_argument("--start", default=None)
    factor_sample.add_argument("--end", default=None)
    factor_sample.add_argument("--warmup-days", type=int, default=30)
    factor_sample.add_argument("--tail", type=int, default=3)
    factor_sample.add_argument("--include-liquidations", action="store_true")

    factor_evaluate = subparsers.add_parser(
        "factor-evaluate",
        help="run exploratory cross-sectional factor evaluation",
    )
    factor_evaluate.add_argument("--db", type=Path, default=DEFAULT_DB)
    factor_evaluate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    factor_evaluate.add_argument("--symbols", default=",".join(DEFAULT_FACTOR_UNIVERSE))
    factor_evaluate.add_argument("--interval", default="1h")
    factor_evaluate.add_argument("--start", default="2023-01-01")
    factor_evaluate.add_argument("--test-start", default="2025-01-01")
    factor_evaluate.add_argument("--end", default="2026-07-31")
    factor_evaluate.add_argument(
        "--base-market",
        choices=("spot", "usd_m_perpetual"),
        default="usd_m_perpetual",
    )
    factor_evaluate.add_argument("--cost-bps", type=float, default=10.0)
    factor_evaluate.add_argument("--quantile", type=float, default=0.20)
    factor_evaluate.add_argument("--min-cross-section", type=int, default=5)

    positioning_contrarian = subparsers.add_parser(
        "positioning-contrarian-study",
        help="run the frozen top-trader positioning contrarian V1 development study",
    )
    positioning_contrarian.add_argument("--db", type=Path, default=DEFAULT_DB)
    positioning_contrarian.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    positioning_contrarian.add_argument(
        "--symbols", default=",".join(DEFAULT_FACTOR_UNIVERSE)
    )
    positioning_contrarian.add_argument("--interval", default="4h")
    positioning_contrarian.add_argument("--start", default="2023-01-01")
    positioning_contrarian.add_argument("--test-start", default="2025-01-01")
    positioning_contrarian.add_argument("--end", default="2026-07-31")
    positioning_contrarian.add_argument("--cost-bps", type=float, default=10.0)
    positioning_contrarian.add_argument("--min-cross-section", type=int, default=5)

    status = subparsers.add_parser("status", help="show local market-data coverage")
    status.add_argument("--db", type=Path, default=DEFAULT_DB)

    study = subparsers.add_parser("first-study", help="run the predeclared spot study")
    study.add_argument("--db", type=Path, default=DEFAULT_DB)
    study.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    study.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    study.add_argument("--interval", default="4h")
    study.add_argument("--start", default="2019-01-01")
    study.add_argument("--test-start", default="2023-01-01")

    strategy_library = subparsers.add_parser(
        "strategy-library",
        help="build a sealed-ready strategy lifecycle library snapshot",
    )
    strategy_library.add_argument("--db", type=Path, default=DEFAULT_DB)
    strategy_library.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    strategy_library.add_argument("--as-of", default=None)

    data_quality = subparsers.add_parser(
        "data-quality",
        help="audit local OHLCV, funding, derivatives, and cache quality",
    )
    data_quality.add_argument("--db", type=Path, default=DEFAULT_DB)
    data_quality.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    data_quality.add_argument("--as-of", default=None)
    data_quality.add_argument(
        "--raw-root",
        type=Path,
        default=Path("market_data/raw/funding"),
    )
    data_quality.add_argument(
        "--open-interest-raw-root",
        type=Path,
        default=Path("market_data/raw/open_interest"),
    )

    funding_update = subparsers.add_parser(
        "update-funding",
        help="download Binance USD-M perpetual funding history",
    )
    funding_update.add_argument("--db", type=Path, default=DEFAULT_DB)
    funding_update.add_argument(
        "--authorize-public-download",
        required=True,
        action="store_true",
        help="explicitly authorize this public-network download",
    )
    funding_update.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    funding_update.add_argument("--start", default="2019-09-01")
    funding_update.add_argument("--end", default=None)
    funding_update.add_argument(
        "--source",
        choices=["rest", "archive"],
        default="archive",
        help="REST needs fapi access; archive uses data.binance.vision monthly zips",
    )
    funding_update.add_argument(
        "--raw-root",
        type=Path,
        default=Path("market_data/raw/funding"),
    )

    funding_mark_update = subparsers.add_parser(
        "update-funding-mark-prices",
        help="download official mark-price klines and pair them to funding stamps",
    )
    funding_mark_update.add_argument("--db", type=Path, default=DEFAULT_DB)
    funding_mark_update.add_argument(
        "--authorize-public-download",
        required=True,
        action="store_true",
        help="explicitly authorize this public-network download",
    )
    funding_mark_update.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    funding_mark_update.add_argument("--start", default="2019-09-01")
    funding_mark_update.add_argument("--end", default=None)
    funding_mark_update.add_argument("--interval", default="1m")
    funding_mark_update.add_argument(
        "--raw-root",
        type=Path,
        default=Path("market_data/raw/funding"),
    )

    open_interest_update = subparsers.add_parser(
        "update-open-interest",
        help="download Binance USD-M public derivatives metrics archives",
    )
    open_interest_update.add_argument("--db", type=Path, default=DEFAULT_DB)
    open_interest_update.add_argument(
        "--authorize-public-download",
        required=True,
        action="store_true",
        help="explicitly authorize this public-network download",
    )
    open_interest_update.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    open_interest_update.add_argument("--period", default="5m")
    open_interest_update.add_argument("--start", default="2023-01-01")
    open_interest_update.add_argument("--end", default=None)
    open_interest_update.add_argument(
        "--raw-root",
        type=Path,
        default=Path("market_data/raw/open_interest"),
    )

    rc01_trial_export = subparsers.add_parser(
        "export-rc01-trials",
        help="freeze the RC-20260822-01 trial set before metric ingestion",
    )
    rc01_trial_export.add_argument("--db", type=Path, default=DEFAULT_DB)
    rc01_trial_export.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    oi_donchian_study = subparsers.add_parser(
        "oi-donchian-study",
        help="run preregistered contract RC-20260822-01",
    )
    oi_donchian_study.add_argument("--db", type=Path, default=DEFAULT_DB)
    oi_donchian_study.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    oi_donchian_study.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    oi_donchian_study.add_argument("--interval", default="4h")
    oi_donchian_study.add_argument("--start", default="2019-01-01")
    oi_donchian_study.add_argument("--test-start", default="2023-01-01")
    oi_donchian_study.add_argument("--period", default="5m")

    positioning_study = subparsers.add_parser(
        "positioning-event-study",
        help="run preregistered contract RC-20260822-02",
    )
    positioning_study.add_argument("--db", type=Path, default=DEFAULT_DB)
    positioning_study.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    positioning_study.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    positioning_study.add_argument("--interval", default="4h")
    positioning_study.add_argument("--start", default="2019-01-01")
    positioning_study.add_argument("--period", default="5m")

    order_book_replay = subparsers.add_parser(
        "replay-order-book",
        help="replay hypothetical requests against a locally captured L2 event file",
    )
    order_book_replay.add_argument("--events", type=Path, required=True)
    order_book_replay.add_argument("--requests", type=Path, required=True)
    order_book_replay.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    funding_study = subparsers.add_parser(
        "funding-study",
        help="run the predeclared delta-neutral carry study",
    )
    funding_study.add_argument("--db", type=Path, default=DEFAULT_DB)
    funding_study.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    funding_study.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    funding_study.add_argument("--interval", default="4h")
    funding_study.add_argument("--start", default="2019-09-01")
    funding_study.add_argument("--test-start", default="2023-01-01")

    robustness = subparsers.add_parser(
        "robustness-study",
        help="evaluate declared spot rules across parameter neighborhoods and time slices",
    )
    robustness.add_argument("--db", type=Path, default=DEFAULT_DB)
    robustness.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    robustness.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    robustness.add_argument("--interval", default="4h")
    robustness.add_argument("--start", default="2019-01-01")
    robustness.add_argument("--test-start", default="2023-01-01")

    execution_stress = subparsers.add_parser(
        "execution-stress",
        help="stress robust spot rules across costs, no-trade bands, and latency",
    )
    execution_stress.add_argument("--db", type=Path, default=DEFAULT_DB)
    execution_stress.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    execution_stress.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    execution_stress.add_argument("--interval", default="4h")
    execution_stress.add_argument("--start", default="2019-01-01")
    execution_stress.add_argument("--test-start", default="2023-01-01")

    relative_strength = subparsers.add_parser(
        "relative-strength-study",
        help="run the frozen BTC/ETH relative-strength rotation study",
    )
    relative_strength.add_argument("--db", type=Path, default=DEFAULT_DB)
    relative_strength.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    relative_strength.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    relative_strength.add_argument("--interval", default="4h")
    relative_strength.add_argument("--start", default="2019-01-01")
    relative_strength.add_argument("--test-start", default="2023-01-01")
    relative_strength.add_argument(
        "--code-tests-passed",
        action="store_true",
        help="attest that the dedicated relative-strength tests passed immediately before this run",
    )

    basket_trend = subparsers.add_parser(
        "basket-trend-study",
        help="run the frozen BTC/ETH basket trend volatility-target study",
    )
    basket_trend.add_argument("--db", type=Path, default=DEFAULT_DB)
    basket_trend.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    basket_trend.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    basket_trend.add_argument("--interval", default="4h")
    basket_trend.add_argument("--start", default="2019-01-01")
    basket_trend.add_argument("--test-start", default="2023-01-01")
    basket_trend.add_argument(
        "--code-tests-passed",
        action="store_true",
        help="attest that the dedicated basket-trend tests passed immediately before this run",
    )

    basket_vol_beta = subparsers.add_parser(
        "basket-vol-beta-study",
        help="run the frozen non-directional BTC/ETH basket volatility-target study",
    )
    basket_vol_beta.add_argument("--db", type=Path, default=DEFAULT_DB)
    basket_vol_beta.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    basket_vol_beta.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    basket_vol_beta.add_argument("--interval", default="4h")
    basket_vol_beta.add_argument("--start", default="2019-01-01")
    basket_vol_beta.add_argument("--test-start", default="2023-01-01")
    basket_vol_beta.add_argument(
        "--code-tests-passed",
        action="store_true",
        help="attest that the dedicated basket-vol-beta tests passed immediately before this run",
    )

    risk_parity = subparsers.add_parser(
        "risk-parity-vol-target-study",
        help="run the frozen BTC/ETH risk-parity volatility-target study",
    )
    risk_parity.add_argument("--db", type=Path, default=DEFAULT_DB)
    risk_parity.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    risk_parity.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    risk_parity.add_argument("--interval", default="4h")
    risk_parity.add_argument("--start", default="2019-01-01")
    risk_parity.add_argument("--test-start", default="2023-01-01")
    risk_parity.add_argument(
        "--code-tests-passed",
        action="store_true",
        help="attest that the dedicated risk-parity tests passed immediately before this run",
    )

    risk_parity_v2 = subparsers.add_parser(
        "risk-parity-vol-target-v2-study",
        help="run the frozen confirmatory 25%% risk-parity volatility-target study",
    )
    risk_parity_v2.add_argument("--db", type=Path, default=DEFAULT_DB)
    risk_parity_v2.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    risk_parity_v2.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    risk_parity_v2.add_argument("--interval", default="4h")
    risk_parity_v2.add_argument("--start", default="2019-01-01")
    risk_parity_v2.add_argument("--test-start", default="2023-01-01")
    risk_parity_v2.add_argument(
        "--code-tests-passed",
        action="store_true",
        help="attest that the dedicated risk-parity tests passed immediately before this run",
    )

    research_brief = subparsers.add_parser(
        "research-brief",
        help="audit data/evidence and generate a multi-agent research brief",
    )
    research_brief.add_argument("--db", type=Path, default=DEFAULT_DB)
    research_brief.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    research_brief.add_argument("--as-of", default=None)

    system_audit = subparsers.add_parser(
        "system-audit",
        help="audit ledger evidence, reports, and paper-session integrity",
    )
    system_audit.add_argument("--db", type=Path, default=DEFAULT_DB)
    system_audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    system_audit.add_argument("--as-of", default=None)

    seal_artifacts = subparsers.add_parser(
        "seal-artifacts",
        help="create SHA-256 manifests for existing research artifacts",
    )
    seal_artifacts.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    autopsy = subparsers.add_parser(
        "strategy-autopsy",
        help="analyze return concentration, regimes, costs, drawdowns, and trade episodes",
    )
    autopsy.add_argument("--db", type=Path, default=DEFAULT_DB)
    autopsy.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    autopsy.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    autopsy.add_argument("--interval", default="4h")
    autopsy.add_argument("--start", default="2019-01-01")
    autopsy.add_argument("--test-start", default="2023-01-01")
    autopsy.add_argument("--strategy", default="donchian_96_48")

    portfolio = subparsers.add_parser(
        "portfolio-study",
        help="evaluate the predeclared Donchian BTC/ETH equal-sleeve portfolio",
    )
    portfolio.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio.add_argument("--interval", default="4h")
    portfolio.add_argument("--start", default="2019-01-01")
    portfolio.add_argument("--test-start", default="2023-01-01")

    portfolio_stress = subparsers.add_parser(
        "portfolio-stress",
        help="stress the Donchian BTC/ETH portfolio across execution assumptions",
    )
    portfolio_stress.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_stress.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_stress.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_stress.add_argument("--interval", default="4h")
    portfolio_stress.add_argument("--start", default="2019-01-01")
    portfolio_stress.add_argument("--test-start", default="2023-01-01")

    portfolio_capacity = subparsers.add_parser(
        "portfolio-capacity",
        help="estimate ex ante liquidity capacity for the Donchian BTC/ETH portfolio",
    )
    portfolio_capacity.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_capacity.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_capacity.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_capacity.add_argument("--interval", default="4h")
    portfolio_capacity.add_argument("--start", default="2019-01-01")
    portfolio_capacity.add_argument("--test-start", default="2023-01-01")

    portfolio_margin_stress = subparsers.add_parser(
        "portfolio-margin-stress",
        help=(
            "run hypothetical cross-margin and liquidation stress for the "
            "Donchian BTC/ETH portfolio"
        ),
    )
    portfolio_margin_stress.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_margin_stress.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_margin_stress.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_margin_stress.add_argument("--interval", default="4h")
    portfolio_margin_stress.add_argument("--start", default="2019-01-01")
    portfolio_margin_stress.add_argument("--test-start", default="2023-01-01")

    portfolio_walk_forward = subparsers.add_parser(
        "portfolio-walk-forward",
        help="validate the unchanged Donchian portfolio across annual calendar folds",
    )
    portfolio_walk_forward.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_walk_forward.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_walk_forward.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_walk_forward.add_argument("--interval", default="4h")
    portfolio_walk_forward.add_argument("--start", default="2019-01-01")
    portfolio_walk_forward.add_argument("--test-start", default="2023-01-01")

    paper_init = subparsers.add_parser(
        "paper-init",
        help="initialize an append-only forward paper session from local history",
    )
    paper_init.add_argument("--db", type=Path, default=DEFAULT_DB)
    paper_init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    paper_init.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    paper_init.add_argument("--interval", default="4h")
    paper_init.add_argument("--start", default="2019-01-01")
    paper_init.add_argument("--initial-capital", type=float, default=10_000.0)

    paper_update = subparsers.add_parser(
        "paper-update",
        help="process only new complete local bars in a forward paper session",
    )
    paper_update.add_argument("--session", type=Path, required=True)
    paper_update.add_argument("--db", type=Path, default=DEFAULT_DB)

    paper_enable_append = subparsers.add_parser(
        "paper-enable-append",
        help="convert a sealed genesis paper session to append-safe auditing",
    )
    paper_enable_append.add_argument("--session", type=Path, required=True)

    paper_review = subparsers.add_parser(
        "paper-review",
        help="apply the predeclared read-only forward-paper review",
    )
    paper_review.add_argument("--session", type=Path, required=True)
    paper_review.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    shadow_init = subparsers.add_parser(
        "shadow-init",
        help="initialize an independent local realtime shadow session",
    )
    shadow_init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    shadow_init.add_argument("--db", type=Path, default=DEFAULT_DB)
    shadow_init.add_argument("--initial-capital", type=float, default=10_000.0)

    shadow_status = subparsers.add_parser(
        "shadow-status",
        help="inspect a shadow session without network access",
    )
    shadow_status.add_argument("--session", type=Path, required=True)
    shadow_status.add_argument("--db", type=Path, default=DEFAULT_DB)

    shadow_replay = subparsers.add_parser(
        "shadow-replay",
        help="replay a locally captured public event file without network access",
    )
    shadow_replay.add_argument("--session", type=Path, required=True)
    shadow_replay.add_argument("--events", type=Path, required=True)
    shadow_replay.add_argument("--db", type=Path, default=DEFAULT_DB)
    shadow_replay.add_argument("--received-at", default=None, help="explicit UTC receive time for events missing E")

    shadow_run = subparsers.add_parser(
        "shadow-run",
        help="run an authorized public stream or replay local events",
    )
    shadow_run.add_argument("--session", type=Path, required=True)
    shadow_run.add_argument("--events", type=Path, default=None)
    shadow_run.add_argument(
        "--authorize-public-stream",
        action="store_true",
        help="explicitly authorize the public Binance WebSocket (no account access)",
    )
    shadow_run.add_argument("--max-events", type=int, default=None)
    shadow_run.add_argument("--db", type=Path, default=DEFAULT_DB)
    shadow_run.add_argument("--received-at", default=None, help="explicit UTC receive time for offline events missing E")

    shadow_kill = subparsers.add_parser("shadow-kill", help="activate the offline shadow kill switch")
    shadow_kill.add_argument("--session", type=Path, required=True)
    shadow_kill.add_argument("--db", type=Path, default=DEFAULT_DB)

    testnet_init = subparsers.add_parser("testnet-init", help="initialize an offline Spot Testnet execution rehearsal")
    testnet_init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    testnet_status = subparsers.add_parser("testnet-status", help="show offline Spot Testnet rehearsal status")
    testnet_status.add_argument("--session", type=Path, required=True)

    testnet_kill = subparsers.add_parser("testnet-kill", help="persist the Spot Testnet rehearsal kill switch")
    testnet_kill.add_argument("--session", type=Path, required=True)

    testnet_reconcile = subparsers.add_parser("testnet-reconcile", help="reconcile a Spot Testnet rehearsal account and open orders")
    testnet_reconcile.add_argument("--session", type=Path, required=True)
    testnet_reconcile.add_argument("--authorize-testnet-orders", action="store_true", help="explicitly authorize testnet REST reconciliation")

    testnet_run = subparsers.add_parser("testnet-run", help="place one bounded Spot Testnet rehearsal order")
    testnet_run.add_argument("--session", type=Path, required=True)
    testnet_run.add_argument("--intent-id", required=True)
    testnet_run.add_argument("--side", choices=["BUY", "SELL"], default="BUY")
    testnet_run.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], required=True)
    testnet_run.add_argument("--quantity", required=True)
    testnet_run.add_argument("--price", required=True)
    testnet_run.add_argument("--authorize-testnet-orders", action="store_true", help="explicitly authorize testnet REST order rehearsal")

    demo_init = subparsers.add_parser("demo-init", help="initialize an offline Binance Spot Demo Mode simulation")
    demo_init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    demo_status = subparsers.add_parser("demo-status", help="show offline Binance Spot Demo Mode status")
    demo_status.add_argument("--session", type=Path, required=True)
    demo_kill = subparsers.add_parser("demo-kill", help="persist the Binance Spot Demo Mode kill switch")
    demo_kill.add_argument("--session", type=Path, required=True)
    demo_reconcile = subparsers.add_parser("demo-reconcile", help="reconcile a Binance Spot Demo Mode account")
    demo_reconcile.add_argument("--session", type=Path, required=True)
    demo_reconcile.add_argument("--authorize-demo-orders", action="store_true", help="explicitly authorize Demo Mode REST simulation")
    demo_run = subparsers.add_parser("demo-run", help="place one bounded Binance Spot Demo Mode simulation order")
    demo_run.add_argument("--session", type=Path, required=True)
    demo_run.add_argument("--intent-id", required=True)
    demo_run.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], required=True)
    demo_run.add_argument("--side", choices=["BUY", "SELL"], default="BUY")
    demo_run.add_argument("--quantity", required=True)
    demo_run.add_argument("--price", required=True)
    demo_run.add_argument("--authorize-demo-orders", action="store_true", help="explicitly authorize Demo Mode REST simulation")
    demo_quote = subparsers.add_parser("demo-quote", help="read one public Demo Mode best bid/ask quote")
    demo_quote.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], required=True)
    demo_quote.add_argument("--authorize-demo-market-data", action="store_true", help="explicitly authorize Demo public market-data access")
    demo_cancel = subparsers.add_parser("demo-cancel", help="cancel one owned non-terminal Demo order")
    demo_cancel.add_argument("--session", type=Path, required=True)
    demo_cancel.add_argument("--intent-id", required=True)
    demo_cancel.add_argument("--authorize-demo-orders", action="store_true", help="explicitly authorize Demo Mode REST cancellation")

    futures_demo_init = subparsers.add_parser(
        "futures-demo-init", help="initialize an offline USD-M Futures Demo forward session"
    )
    futures_demo_init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    futures_demo_status = subparsers.add_parser(
        "futures-demo-status", help="show an offline USD-M Futures Demo session"
    )
    futures_demo_status.add_argument("--session", type=Path, required=True)
    futures_demo_reconcile = subparsers.add_parser(
        "futures-demo-reconcile", help="reconcile USD-M Futures Demo account and positions"
    )
    futures_demo_reconcile.add_argument("--session", type=Path, required=True)
    futures_demo_snapshot = subparsers.add_parser(
        "futures-demo-snapshot", help="build the frozen current public positioning signal"
    )
    futures_demo_snapshot.add_argument("--session", type=Path, required=True)
    futures_demo_enable = subparsers.add_parser(
        "futures-demo-enable", help="enable the frozen USD-M Futures Demo cycle"
    )
    futures_demo_enable.add_argument("--session", type=Path, required=True)
    futures_demo_disable = subparsers.add_parser(
        "futures-demo-disable", help="disable the USD-M Futures Demo cycle"
    )
    futures_demo_disable.add_argument("--session", type=Path, required=True)
    futures_demo_run = subparsers.add_parser(
        "futures-demo-run-cycle", help="run one scheduled USD-M Futures Demo rebalance cycle"
    )
    futures_demo_run.add_argument("--session", type=Path, required=True)
    futures_demo_kill = subparsers.add_parser(
        "futures-demo-kill", help="disable USD-M Futures Demo and cancel open orders"
    )
    futures_demo_kill.add_argument("--session", type=Path, required=True)

    platform_init = subparsers.add_parser("platform-init", help="initialize an offline local strategy platform")
    platform_init.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_status = subparsers.add_parser("platform-status", help="show offline strategy platform status")
    platform_status.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_register = subparsers.add_parser("platform-register", help="register one local strategy version")
    platform_register.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_register.add_argument("--name", required=True); platform_register.add_argument("--hypothesis", required=True)
    platform_register.add_argument("--code-hash", required=True); platform_register.add_argument("--data-hash", required=True)
    platform_register.add_argument("--role", choices=["incumbent", "challenger", "candidate"], default="candidate")
    platform_register.add_argument("--parameters-json", type=Path, default=None)
    platform_register.add_argument("--primary-family", choices=sorted(STRATEGY_FAMILIES), default="unclassified")
    platform_register.add_argument("--secondary-tags", default="", help="comma-separated secondary tags (最多3)")
    platform_register.add_argument("--profit-mechanism", default="")
    platform_register.add_argument("--expected-regimes", default="")
    platform_register.add_argument("--failure-regimes", default="")
    platform_score = subparsers.add_parser("platform-score", help="score one local strategy version from raw metrics JSON")
    platform_score.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_score.add_argument("--version-id", required=True); platform_score.add_argument("--metrics-json", type=Path, required=True)
    platform_checkpoint = subparsers.add_parser("platform-checkpoint", help="write a deterministic strategy checkpoint")
    platform_checkpoint.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_checkpoint.add_argument("--version-id", required=True); platform_checkpoint.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "platform_checkpoints")
    platform_checkpoint.add_argument("--assumptions-json", type=Path, default=None)

    platform_demo_enable = subparsers.add_parser("platform-demo-enable", help="enable the Binance Spot Demo platform bridge")
    platform_demo_enable.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_demo_enable.add_argument("--session", type=Path, required=True)
    platform_demo_disable = subparsers.add_parser("platform-demo-disable", help="disable the Binance Spot Demo platform bridge")
    platform_demo_disable.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_demo_disable.add_argument("--session", type=Path, required=True)
    platform_demo_kill = subparsers.add_parser("platform-demo-kill", help="activate the Binance Spot Demo platform kill switch")
    platform_demo_kill.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_demo_kill.add_argument("--session", type=Path, required=True)
    platform_demo_status = subparsers.add_parser("platform-demo-status", help="show the Binance Spot Demo platform bridge status")
    platform_demo_status.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_demo_status.add_argument("--session", type=Path, required=True)
    platform_demo_cycle = subparsers.add_parser("platform-demo-run-cycle", help="run one finite Binance Spot Demo strategy cycle")
    platform_demo_cycle.add_argument("--platform", type=Path, default=DEFAULT_OUTPUT / "strategy_platform")
    platform_demo_cycle.add_argument("--session", type=Path, required=True)
    platform_demo_cycle.add_argument("--db", type=Path, default=DEFAULT_DB)

    portfolio_regimes = subparsers.add_parser(
        "portfolio-regimes",
        help="diagnose Donchian portfolio behavior across causal market states",
    )
    portfolio_regimes.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_regimes.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_regimes.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_regimes.add_argument("--interval", default="4h")
    portfolio_regimes.add_argument("--start", default="2019-01-01")
    portfolio_regimes.add_argument("--test-start", default="2023-01-01")

    portfolio_neighborhood = subparsers.add_parser(
        "portfolio-neighborhood",
        help="evaluate the predeclared Donchian neighborhood on the BTC/ETH portfolio",
    )
    portfolio_neighborhood.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_neighborhood.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_neighborhood.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_neighborhood.add_argument("--interval", default="4h")
    portfolio_neighborhood.add_argument("--start", default="2019-01-01")
    portfolio_neighborhood.add_argument("--test-start", default="2023-01-01")

    portfolio_bootstrap = subparsers.add_parser(
        "portfolio-bootstrap",
        help="run deterministic block-bootstrap uncertainty analysis for the portfolio",
    )
    portfolio_bootstrap.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_bootstrap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_bootstrap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_bootstrap.add_argument("--interval", default="4h")
    portfolio_bootstrap.add_argument("--start", default="2019-01-01")
    portfolio_bootstrap.add_argument("--test-start", default="2023-01-01")

    portfolio_capacity_replay = subparsers.add_parser(
        "portfolio-capacity-replay",
        help="replay constrained execution across account sizes and participation caps",
    )
    portfolio_capacity_replay.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_capacity_replay.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_capacity_replay.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_capacity_replay.add_argument("--interval", default="4h")
    portfolio_capacity_replay.add_argument("--start", default="2019-01-01")
    portfolio_capacity_replay.add_argument("--test-start", default="2023-01-01")

    portfolio_drawdowns = subparsers.add_parser(
        "portfolio-drawdowns",
        help="attribute every declared basket-drawdown episode for the portfolio",
    )
    portfolio_drawdowns.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_drawdowns.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_drawdowns.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_drawdowns.add_argument("--interval", default="4h")
    portfolio_drawdowns.add_argument("--start", default="2019-01-01")
    portfolio_drawdowns.add_argument("--test-start", default="2023-01-01")

    portfolio_diversification = subparsers.add_parser(
        "portfolio-diversification",
        help="test rolling diversification stability and predeclared sleeve mixes",
    )
    portfolio_diversification.add_argument("--db", type=Path, default=DEFAULT_DB)
    portfolio_diversification.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    portfolio_diversification.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    portfolio_diversification.add_argument("--interval", default="4h")
    portfolio_diversification.add_argument("--start", default="2019-01-01")
    portfolio_diversification.add_argument("--test-start", default="2023-01-01")

    multiple_testing = subparsers.add_parser(
        "multiple-testing-audit",
        help="apply a pinned Deflated Sharpe correction to the portfolio",
    )
    multiple_testing.add_argument("--db", type=Path, default=DEFAULT_DB)
    multiple_testing.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    multiple_testing.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    multiple_testing.add_argument("--interval", default="4h")
    multiple_testing.add_argument("--start", default="2019-01-01")
    multiple_testing.add_argument("--test-start", default="2023-01-01")

    args = parser.parse_args()
    if args.command == "factor-mine":
        from crypto_quant.research.factor_mining.cli import execute as execute_mining
        _print_json(execute_mining(args))
    elif args.command == "data-request-create":
        if args.spec_json:
            spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
        else:
            if not args.datasets_json:
                parser.error("data-request-create requires --spec-json or --datasets-json")
            spec = {
                "strategy_version_id": args.strategy_version_id,
                "reason": args.reason,
                "datasets": json.loads(args.datasets_json.read_text(encoding="utf-8")),
            }
        _print_json(DataRequestStore(args.root, args.raw_root).create(spec))
    elif args.command == "data-request-status":
        store = DataRequestStore(args.root, args.raw_root)
        _print_json(store.get(args.request_id))
    elif args.command == "data-request-fulfill":
        if not args.authorize_public_download and os.environ.get("AUTHORIZED_PUBLIC_DOWNLOAD") != "1":
            parser.error("data-request-fulfill requires --authorize-public-download or AUTHORIZED_PUBLIC_DOWNLOAD=1")
        store = DataRequestStore(args.root, args.raw_root)
        _print_json(store.fulfill(args.request_id, db_path=args.db, authorize=args.authorize_public_download))
    elif args.command == "research-resume-status":
        _print_json(DataRequestStore(args.root, args.raw_root).resume_status(args.checkpoint_id, args.strategy_version_id))
    elif args.command == "research-checkpoint-create":
        store = DataRequestStore(args.root, args.raw_root)
        _print_json(store.create_checkpoint(args.strategy_version_id, [item.strip() for item in args.request_ids.split(",") if item.strip()], [item.strip() for item in args.completed_steps.split(",") if item.strip()], args.next_action, stage=args.stage, hypothesis=args.hypothesis))
    elif args.command == "data-request-consume":
        _print_json(DataRequestStore(args.root, args.raw_root).consume(args.request_id))
    elif args.command == "data-inventory-plan":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        _print_json(build_inventory_plan(args.db, symbols=symbols, output_root=args.output, interval=args.interval, top_n=args.top_n, discover=args.discover, batch_size=args.batch_size, materialize=args.materialize, max_symbols=args.max_symbols))
    elif args.command == "data-inventory-resume":
        _print_json(resume_inventory_plan(args.plan_id, output_root=args.output))
    elif args.command == "data-inventory-run":
        if not args.authorize_public_download and os.environ.get("AUTHORIZED_PUBLIC_DOWNLOAD") != "1":
            parser.error("data-inventory-run requires --authorize-public-download or AUTHORIZED_PUBLIC_DOWNLOAD=1")
        _print_json(run_inventory_plan(args.plan_id, db_path=args.db, output_root=args.output, raw_root=args.raw_root, max_batches=args.max_batches, authorize=args.authorize_public_download))
    elif args.command == "data-top50-plan":
        _print_json(build_dynamic_top50_plan(args.db, args.inventory_plan, output_root=args.output, top_n=args.top_n, window_days=args.window_days, step_days=args.step_days, materialize=args.materialize))
    elif args.command == "data-top50-run":
        if not args.authorize_public_download and os.environ.get("AUTHORIZED_PUBLIC_DOWNLOAD") != "1":
            parser.error("data-top50-run requires --authorize-public-download or AUTHORIZED_PUBLIC_DOWNLOAD=1")
        _print_json(run_top50_plan(args.plan_id, db_path=args.db, output_root=args.output, raw_root=args.raw_root, max_requests=args.max_requests, authorize=args.authorize_public_download))
    elif args.command == "futures-core-plan":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        _print_json(build_um_futures_core_plan(output_root=args.output, symbols=symbols, max_symbols=args.max_symbols))
    elif args.command == "futures-core-run":
        if not args.authorize_public_download and os.environ.get("AUTHORIZED_PUBLIC_DOWNLOAD") != "1":
            parser.error("futures-core-run requires --authorize-public-download or AUTHORIZED_PUBLIC_DOWNLOAD=1")
        _print_json(run_um_futures_core_plan(args.plan_id, db_path=args.db, output_root=args.output, authorize=args.authorize_public_download, workers=args.workers, chunk_files=args.chunk_files, max_tasks=args.max_tasks, max_files=args.max_files))
    elif args.command == "futures-core-status":
        _print_json(um_futures_core_status(args.plan_id, db_path=args.db, output_root=args.output))
    elif args.command == "futures-core-repair-metrics":
        _print_json(repair_partial_metrics(args.plan_id, db_path=args.db, output_root=args.output))
    elif args.command == "market-data-catalog":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        _print_json(MarketDataStore(args.db).catalog(symbols or None))
    elif args.command == "liquidation-plan":
        _print_json(prepare_liquidation_universe(args.db, args.raw_root))
    elif args.command == "liquidation-download":
        authorized = args.authorize_public_download or os.environ.get(
            "AUTHORIZED_PUBLIC_DOWNLOAD"
        ) == "1"
        if not authorized:
            parser.error(
                "liquidation-download requires --authorize-public-download "
                "or AUTHORIZED_PUBLIC_DOWNLOAD=1"
            )
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(
                f"liquidation-download requires API key environment variable "
                f"{args.api_key_env}"
            )
        _print_json(
            download_liquidation_history(
                args.db,
                args.raw_root,
                api_key,
                workers=args.workers,
                authorize=authorized,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        )
    elif args.command == "liquidation-align":
        _print_json(align_liquidation_history(args.db, args.raw_root, args.output))
    elif args.command == "liquidation-status":
        _print_json(liquidation_ingestion_status(args.raw_root))
    elif args.command == "liquidation-audit":
        _print_json(audit_liquidation_history(args.raw_root))
    elif args.command == "liquidation-alignment-audit":
        _print_json(audit_liquidation_alignment(args.output))
    elif args.command == "market-factor-catalog":
        _print_json(factor_catalog())
    elif args.command == "market-factor-sample":
        engine = FactorEngine(MarketDataStore(args.db))
        frame = engine.load(
            args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            base_market=args.base_market,
            warmup_days=args.warmup_days,
            include_liquidations=args.include_liquidations,
        )
        _print_json(engine.snapshot(frame, tail=args.tail))
    elif args.command == "factor-evaluate":
        symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
        config = FactorEvaluationConfig(
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=args.test_start,
            end=args.end,
            base_market=args.base_market,
            cost_bps=args.cost_bps,
            quantile=args.quantile,
            min_cross_section=args.min_cross_section,
        )
        run_dir = run_factor_evaluation(args.db, args.output, config)
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "positioning-contrarian-study":
        symbols = tuple(
            item.strip().upper()
            for item in args.symbols.split(",")
            if item.strip()
        )
        config = PositioningContrarianConfig(
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=args.test_start,
            end=args.end,
            cost_bps=args.cost_bps,
            min_cross_section=args.min_cross_section,
        )
        run_dir = run_positioning_contrarian_study(args.db, args.output, config)
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "update-data":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        counts = update_klines(
            db_path=args.db,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
        _print_json({"status": "updated", "rows_received": counts})
    elif args.command == "status":
        if not args.db.exists():
            _print_json({"status": "empty", "database": str(args.db)})
        else:
            data_audit = audit_data(args.db, pd.Timestamp.now(tz="UTC"))
            source_statuses = {
                item["source_id"]: item["local_status"]
                for item in data_audit["source_registry"]
            }
            attention_required = (
                bool(data_audit["stale_data"])
                or data_audit["funding_status"] != "ready"
                or source_statuses.get("taker_flow_proxy") != "ready"
                or source_statuses.get("open_interest_history") != "ready"
                or source_statuses.get("long_short_positioning") != "ready"
            )
            _print_json(
                {
                    "status": (
                        "attention_required" if attention_required else "ready"
                    ),
                    "database": str(args.db),
                    "datasets": data_audit["spot_datasets"],
                    "funding_rates": data_audit["funding_datasets"],
                    "open_interest": data_audit["open_interest_datasets"],
                    "positioning": data_audit["positioning_datasets"],
                    "unified_market_data": data_audit["unified_market_data"],
                    "source_statuses": source_statuses,
                }
            )
    elif args.command == "first-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_first_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "strategy-library":
        run_dir = run_strategy_library_snapshot(
            output_root=args.output,
            db_path=args.db,
            as_of=args.as_of,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "data-quality":
        run_dir = run_data_quality_report(
            db_path=args.db,
            output_root=args.output,
            as_of=args.as_of,
            raw_root=args.raw_root,
            open_interest_raw_root=args.open_interest_raw_root,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "update-funding":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.source == "rest":
            counts = update_funding_from_rest(
                db_path=args.db,
                symbols=symbols,
                start=args.start,
                end=args.end,
                client=BinanceFuturesClient(),
            )
            _print_json({"status": "updated", "source": "rest", "symbols": counts})
        else:
            counts = update_funding_from_archive(
                db_path=args.db,
                symbols=symbols,
                start=args.start,
                end=args.end,
                raw_root=args.raw_root,
            )
            _print_json({"status": "updated", "source": "archive", "symbols": counts})
    elif args.command == "update-funding-mark-prices":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        counts = update_funding_mark_prices_from_archive(
            db_path=args.db,
            symbols=symbols,
            start=args.start,
            end=args.end,
            interval=args.interval,
            raw_root=args.raw_root,
        )
        _print_json({"status": "updated", "source": "mark_price_archive", "symbols": counts})
    elif args.command == "funding-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_funding_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "update-open-interest":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        counts = update_open_interest_from_archive(
            db_path=args.db,
            symbols=symbols,
            period=args.period,
            start=args.start,
            end=args.end,
            raw_root=args.raw_root,
            client=BinanceOpenInterestArchiveClient(),
        )
        _print_json({"status": "updated", "source": "metrics_archive", "symbols": counts})
    elif args.command == "export-rc01-trials":
        path = export_rc01_pinned_trials(db_path=args.db, output_root=args.output)
        _print_json({"status": "exported", "trial_directory": str(path)})
    elif args.command == "oi-donchian-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_open_interest_confirmed_donchian_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=args.test_start,
            period=args.period,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "positioning-event-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_positioning_extremes_event_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            period=args.period,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "replay-order-book":
        run_dir = run_order_book_replay_study(
            events_path=args.events,
            requests_path=args.requests,
            output_root=args.output,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "robustness-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_robustness_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "execution-stress":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_execution_stress(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "relative-strength-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_cross_sectional_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            code_tests_passed=args.code_tests_passed,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "basket-trend-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_basket_trend_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            code_tests_passed=args.code_tests_passed,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "basket-vol-beta-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_basket_vol_beta_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            code_tests_passed=args.code_tests_passed,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "risk-parity-vol-target-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_risk_parity_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            code_tests_passed=args.code_tests_passed,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "risk-parity-vol-target-v2-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_risk_parity_v2_confirmatory_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            code_tests_passed=args.code_tests_passed,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "research-brief":
        briefing = run_research_agents(
            db_path=args.db,
            output_root=args.output,
            as_of=args.as_of,
        )
        _print_json(
            {
                "status": "complete",
                "brief_directory": str(briefing.brief_dir),
                "strategy_decisions": {
                    item["strategy"]: item["decision"]
                    for item in briefing.payload["strategy_states"]
                },
            }
        )
    elif args.command == "system-audit":
        result = run_system_audit(
            db_path=args.db,
            output_root=args.output,
            as_of=args.as_of,
        )
        _print_json(
            {
                "status": result.status,
                "audit_directory": str(result.run_dir),
            }
        )
    elif args.command == "seal-artifacts":
        summary = seal_research_artifacts(output_root=args.output)
        _print_json(
            {
                "status": "sealed",
                "directories_scanned": summary["directories_scanned"],
                "sealed_now": summary["sealed_now"],
                "already_sealed": summary["already_sealed"],
            }
        )
    elif args.command == "strategy-autopsy":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_strategy_autopsy(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
            strategy_name=args.strategy,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-study":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_donchian_portfolio_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-stress":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_stress(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-capacity":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_capacity_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-margin-stress":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_margin_stress(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-walk-forward":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_donchian_portfolio_walk_forward(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "paper-init":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        session = initialize_paper_session(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            initial_capital=args.initial_capital,
        )
        _print_json({"status": "initialized", "session_directory": str(session)})
    elif args.command == "paper-update":
        session = update_paper_session(session_path=args.session, db_path=args.db)
        _print_json({"status": "updated", "session_directory": str(session)})
    elif args.command == "paper-enable-append":
        result = enable_paper_append_manifest(args.session)
        _print_json({"status": result["status"], "session_directory": str(args.session)})
    elif args.command == "paper-review":
        run_dir = run_forward_paper_review(
            session_path=args.session,
            output_root=args.output,
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "shadow-init":
        session = ShadowSession.initialize(
            output_root=args.output,
            initial_cash=args.initial_capital,
            db_path=args.db,
        )
        _print_json({"status": "initialized", "session_directory": str(session.path)})
    elif args.command == "shadow-status":
        session = ShadowSession.open(args.session, db_path=args.db)
        _print_json(session.status(db_path=args.db))
    elif args.command == "shadow-replay":
        _print_json(replay_shadow_events(args.session, args.events, db_path=args.db, received_at=args.received_at))
    elif args.command == "shadow-run":
        if args.events is not None:
            # Local replay is deliberately independent of network authorization.
            _print_json(replay_shadow_events(args.session, args.events, db_path=args.db, received_at=args.received_at if hasattr(args, "received_at") else None))
        else:
            if not args.authorize_public_stream:
                parser.error("shadow-run public mode requires --authorize-public-stream")
            _print_json(run_public_stream(
                args.session,
                authorize_public_stream=args.authorize_public_stream,
                max_events=args.max_events,
                db_path=args.db,
            ))
    elif args.command == "shadow-kill":
        session = ShadowSession.open(args.session, db_path=args.db)
        session.activate_kill_switch()
        _print_json(session.status(db_path=args.db))
    elif args.command == "testnet-init":
        session = TestnetSession.initialize(args.output)
        _print_json(session.status())
    elif args.command == "testnet-status":
        _print_json(TestnetSession.open(args.session).status())
    elif args.command == "testnet-kill":
        session = TestnetSession.open(args.session)
        session.activate_kill_switch()
        _print_json(session.status())
    elif args.command in {"testnet-reconcile", "testnet-run"}:
        if not args.authorize_testnet_orders:
            parser.error("testnet network mode requires --authorize-testnet-orders")
        if os.environ.get("AUTHORIZED_BINANCE_TESTNET") != "1":
            parser.error("testnet network mode requires AUTHORIZED_BINANCE_TESTNET=1")
        credentials = TestnetCredentials.from_env()
        client = TestnetClient(credentials, transport=RequestsTransport())
        session = TestnetSession.open(args.session)
        rehearsal = TestnetRehearsal(session, client)
        if args.command == "testnet-reconcile":
            _print_json(rehearsal.reconcile())
        else:
            rehearsal.reconcile()
            rehearsal.load_rules()
            order = rehearsal.place_limit_buy(args.intent_id, args.symbol, args.quantity, args.price) if args.side == "BUY" else rehearsal.place_limit_sell(args.intent_id, args.symbol, args.quantity, args.price)
            _print_json(order)
    elif args.command == "demo-init":
        session = DemoSession.initialize(args.output)
        _print_json(session.status())
    elif args.command == "demo-status":
        _print_json(DemoSession.open(args.session).status())
    elif args.command == "futures-demo-init":
        session = FuturesDemoSession.initialize(args.output)
        _print_json(session.status())
    elif args.command == "futures-demo-status":
        _print_json(FuturesDemoSession.open(args.session).status())
    elif args.command in {
        "futures-demo-reconcile",
        "futures-demo-snapshot",
        "futures-demo-enable",
        "futures-demo-disable",
        "futures-demo-run-cycle",
        "futures-demo-kill",
    }:
        if args.command != "futures-demo-disable":
            if os.environ.get("AUTHORIZED_BINANCE_FUTURES_DEMO") != "1":
                parser.error(
                    f"{args.command} requires AUTHORIZED_BINANCE_FUTURES_DEMO=1"
                )
        orchestrator = make_futures_demo_orchestrator(args.session)
        if args.command == "futures-demo-reconcile":
            _print_json(orchestrator.reconcile())
        elif args.command == "futures-demo-snapshot":
            _print_json(orchestrator.snapshot())
        elif args.command == "futures-demo-enable":
            _print_json(orchestrator.enable())
        elif args.command == "futures-demo-disable":
            _print_json(orchestrator.disable())
        elif args.command == "futures-demo-run-cycle":
            _print_json(orchestrator.run_cycle())
        else:
            _print_json(orchestrator.kill())
    elif args.command == "demo-kill":
        session = DemoSession.open(args.session)
        session.activate_kill_switch()
        _print_json(session.status())
    elif args.command == "demo-quote":
        if not args.authorize_demo_market_data:
            parser.error("Demo public market data requires --authorize-demo-market-data")
        if os.environ.get("AUTHORIZED_BINANCE_DEMO") != "1":
            parser.error("Demo public market data requires AUTHORIZED_BINANCE_DEMO=1")
        public_client = DemoClient(DemoCredentials("", ""), transport=RequestsTransport())
        _print_json(public_client.book_ticker(args.symbol))
    elif args.command == "demo-cancel":
        if not args.authorize_demo_orders:
            parser.error("Demo cancellation requires --authorize-demo-orders")
        if os.environ.get("AUTHORIZED_BINANCE_DEMO") != "1":
            parser.error("Demo cancellation requires AUTHORIZED_BINANCE_DEMO=1")
        client = DemoClient(DemoCredentials.from_env(), transport=RequestsTransport())
        session = DemoSession.open(args.session)
        _print_json(DemoRehearsal(session, client).cancel(args.intent_id))
    elif args.command == "platform-init":
        _print_json({"platform": str(StrategyPlatform.initialize(args.platform).path), "status": "initialized", "execution_enabled": False})
    elif args.command == "platform-status":
        platform = StrategyPlatform.open(args.platform)
        _print_json({"platform": str(platform.path), "event_sequence": platform.state["event_sequence"], "slots": platform.state["slots"], "queue": platform.state["queue"], "trial_count": platform.state["trial_count"], "execution_enabled": platform.state["execution_enabled"]})
    elif args.command == "platform-register":
        platform = StrategyPlatform.open(args.platform)
        parameters = json.loads(args.parameters_json.read_text(encoding="utf-8")) if args.parameters_json else {}
        secondary_tags = [tag.strip() for tag in args.secondary_tags.split(",") if tag.strip()]
        version = platform.register_version(args.name, args.hypothesis, parameters, args.code_hash, args.data_hash, role=args.role, primary_family=args.primary_family, secondary_tags=secondary_tags, profit_mechanism=args.profit_mechanism, expected_regimes=args.expected_regimes, failure_regimes=args.failure_regimes)
        _print_json(asdict(version))
    elif args.command == "platform-score":
        platform = StrategyPlatform.open(args.platform)
        metrics = MetricInputs(**json.loads(args.metrics_json.read_text(encoding="utf-8")))
        _print_json(platform.score_version_from_metrics(args.version_id, metrics))
    elif args.command == "platform-checkpoint":
        platform = StrategyPlatform.open(args.platform)
        assumptions = json.loads(args.assumptions_json.read_text(encoding="utf-8")) if args.assumptions_json else {}
        _print_json({"checkpoint": str(platform.checkpoint(args.version_id, args.output, assumptions=assumptions))})
    elif args.command in {"platform-demo-enable", "platform-demo-disable", "platform-demo-kill", "platform-demo-status", "platform-demo-run-cycle"}:
        orchestrator = make_demo_orchestrator(args.platform, args.session, args.db if hasattr(args, "db") else DEFAULT_DB)
        if args.command == "platform-demo-enable":
            _print_json(orchestrator.enable())
        elif args.command == "platform-demo-disable":
            _print_json(orchestrator.disable())
        elif args.command == "platform-demo-kill":
            _print_json(orchestrator.kill())
        elif args.command == "platform-demo-status":
            _print_json(orchestrator.status())
        else:
            _print_json(orchestrator.run_cycle())
    elif args.command in {"demo-reconcile", "demo-run"}:
        if not args.authorize_demo_orders:
            parser.error("Demo network mode requires --authorize-demo-orders")
        if os.environ.get("AUTHORIZED_BINANCE_DEMO") != "1":
            parser.error("Demo network mode requires AUTHORIZED_BINANCE_DEMO=1")
        client = DemoClient(DemoCredentials.from_env(), transport=RequestsTransport())
        session = DemoSession.open(args.session)
        rehearsal = DemoRehearsal(session, client)
        if args.command == "demo-reconcile":
            _print_json(rehearsal.reconcile())
        else:
            rehearsal.reconcile()
            rehearsal.load_rules()
            order = rehearsal.place_limit_buy(args.intent_id, args.symbol, args.quantity, args.price) if args.side == "BUY" else rehearsal.place_limit_sell(args.intent_id, args.symbol, args.quantity, args.price)
            _print_json(order)
    elif args.command == "portfolio-regimes":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_regime_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-neighborhood":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_neighborhood_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-bootstrap":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_bootstrap_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-capacity-replay":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_capacity_replay_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-drawdowns":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_drawdown_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "portfolio-diversification":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_portfolio_diversification_study(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})
    elif args.command == "multiple-testing-audit":
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        run_dir = run_multiple_testing_audit(
            db_path=args.db,
            output_root=args.output,
            symbols=symbols,
            interval=args.interval,
            start=args.start,
            test_start=pd.Timestamp(args.test_start).isoformat(),
        )
        _print_json({"status": "complete", "run_directory": str(run_dir)})


if __name__ == "__main__":
    main()
