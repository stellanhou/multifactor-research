"""FM-v6 factor-card loading and deterministic cross-sectional processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_quant.features.factor_expressions import compile_expression, evaluate_expression
from crypto_quant.features.factor_inputs import FactorInputPanel, validate_universe


@dataclass(frozen=True)
class FactorPanels:
    """Aligned raw, direction-standardized, and diagnostic factor panels."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    eligible: pd.Series
    valid_masks: pd.DataFrame
    common_mask: pd.Series
    coverage: pd.DataFrame
    correlations: pd.DataFrame
    correlation_summary: pd.DataFrame
    input_diagnostics: dict[str, Any]


def read_cards(
    paths: list[Path],
    horizon_hours: int,
    max_lookback_hours: int,
) -> list[dict[str, Any]]:
    """Load and validate FM-v6 cards for one common evaluation horizon.

    Card formula metadata is checked against the current restricted DSL before
    the card can enter a factor panel. The complete card is retained as a
    snapshot so later work does not depend on its original path remaining valid.
    """
    if type(horizon_hours) is not int or horizon_hours not in {1, 4, 24}:
        raise ValueError("horizon_hours must be one of 1, 4, or 24")
    if type(max_lookback_hours) is not int or max_lookback_hours < 0:
        raise ValueError("max_lookback_hours must be a non-negative integer")
    if not paths:
        raise ValueError("at least one FM-v6 card path is required")

    cards: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        card = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(card, dict):
            raise ValueError(f"card must contain a JSON object: {path}")
        if card.get("source_type") != "factor_mining":
            raise ValueError(f"card is not a factor-mining card: {path}")
        if card.get("status") != "research_idea":
            raise ValueError(f"card is not an admitted research idea: {path}")

        card_id = card.get("id")
        title = card.get("title")
        claim = card.get("original_claim")
        market = card.get("market_and_horizon")
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError(f"card id is missing: {path}")
        if card_id in seen_ids:
            raise ValueError(f"duplicate factor card id: {card_id}")
        seen_ids.add(card_id)
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"card title is missing: {path}")
        if not isinstance(claim, dict) or not isinstance(market, dict):
            raise ValueError(f"FM-v6 card fields are missing: {path}")

        direction = claim.get("direction")
        if type(direction) is not int or direction not in {-1, 1}:
            raise ValueError(f"card direction must be explicitly frozen to -1 or 1: {path}")
        passed_horizons = market.get("passed_horizons")
        if (not isinstance(passed_horizons, list) or not passed_horizons
                or any(type(value) is not int for value in passed_horizons)
                or len(passed_horizons) != len(set(passed_horizons))):
            raise ValueError(f"card passed_horizons must be a nonempty unique integer list: {path}")
        if horizon_hours not in passed_horizons:
            raise ValueError(f"card did not pass the requested {horizon_hours}h horizon: {path}")
        if card.get("b_validation_status") != "passed":
            raise ValueError(f"card B validation status is not passed: {path}")
        admission = card.get("admission_evidence")
        if not isinstance(admission, dict) or admission.get("eligible_for_idea_pool") is not True:
            raise ValueError(f"card is not eligible for the idea pool: {path}")
        if (market.get("venue") != "Binance" or market.get("market") != "USD-M perpetual"
                or market.get("inputs") != "1h"):
            raise ValueError(f"card market must be Binance USD-M perpetual with 1h inputs: {path}")

        formula = claim.get("formula")
        if not isinstance(formula, dict):
            raise ValueError(f"card formula metadata is missing: {path}")
        expression = formula.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"card formula expression is missing: {path}")
        compiled = compile_expression(expression)
        if formula.get("expanded_expression") != compiled.expanded_expression:
            raise ValueError(f"card expanded expression differs from the current DSL: {path}")
        if formula.get("fields") != list(compiled.fields):
            raise ValueError(f"card input fields differ from the current DSL: {path}")
        if formula.get("lookback_hours") != compiled.lookback_hours:
            raise ValueError(f"card lookback metadata differs from the current DSL: {path}")
        if compiled.lookback_hours > max_lookback_hours:
            raise ValueError(
                f"card lookback {compiled.lookback_hours}h exceeds configured maximum "
                f"{max_lookback_hours}h: {path}"
            )

        cards.append({
            "id": card_id,
            "title": title,
            "expression": expression,
            "direction": direction,
            "horizon_hours": horizon_hours,
            "passed_horizons": passed_horizons,
            "fields": list(compiled.fields),
            "lookback_hours": compiled.lookback_hours,
            "source": card.get("source"),
            "path": str(path),
            "card_snapshot": card,
        })
    return cards


def process_factors(
    panel: FactorInputPanel,
    cards: list[dict[str, Any]],
    *,
    include_correlations: bool = True,
) -> FactorPanels:
    """Calculate factor values, optionally including correlation diagnostics.

    When ``include_correlations`` is false, the correlation tables keep their
    established empty schemas to indicate that those diagnostics were not
    calculated; the standardized factor panels are still fully computed.
    """
    if type(include_correlations) is not bool:
        raise ValueError("include_correlations must be a boolean")
    if not cards:
        raise ValueError("at least one factor card is required")
    eligible = validate_universe(panel.universe)
    if not panel.values.index.equals(eligible.index):
        raise ValueError("factor input values and universe indices differ")

    card_ids = [card.get("id") for card in cards]
    if any(not isinstance(card_id, str) or not card_id for card_id in card_ids):
        raise ValueError("every factor card requires a nonempty id")
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("factor card ids must be unique")

    raw = pd.DataFrame(index=eligible.index)
    directions: dict[str, int] = {}
    for card in cards:
        card_id = card["id"]
        expression = card.get("expression")
        direction = card.get("direction")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"factor {card_id} has no expression")
        if type(direction) is not int or direction not in {-1, 1}:
            raise ValueError(f"factor {card_id} direction must be explicitly -1 or 1")
        compiled = compile_expression(expression)
        declared_fields = card.get("fields")
        declared_lookback = card.get("lookback_hours")
        if declared_fields is not None and declared_fields != list(compiled.fields):
            raise ValueError(f"factor {card_id} fields differ from its formula")
        if declared_lookback is not None and declared_lookback != compiled.lookback_hours:
            raise ValueError(f"factor {card_id} lookback differs from its formula")
        result = evaluate_expression(expression, panel)
        if result.definition["expanded_expression"] != compiled.expanded_expression:
            raise ValueError(f"factor {card_id} executed expression differs from compiled formula")
        if not result.values.index.equals(eligible.index):
            raise ValueError(f"factor {card_id} values differ from the canonical panel index")
        raw[card_id] = result.values
        directions[card_id] = direction

    valid_masks = pd.DataFrame(
        np.isfinite(raw.to_numpy(dtype=float)), index=raw.index, columns=raw.columns,
    )
    valid_masks = valid_masks.astype(bool).where(eligible, False)

    standardized = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns, dtype=float)
    for card in cards:
        card_id = card["id"]
        directed = (raw[card_id] * directions[card_id]).where(eligible)
        grouped = directed.groupby(level="timestamp", sort=False)
        means = grouped.transform("mean")
        stds = grouped.transform(lambda values: values.std(ddof=0))
        standardized[card_id] = (directed - means).div(stds.where(stds > 0))
    standardized = standardized.where(eligible)

    eligible_rows = int(eligible.sum())
    coverage_rows = []
    for card_id in raw.columns:
        valid_count = int(valid_masks[card_id].sum())
        coverage_rows.append({
            "factor_id": card_id,
            "eligible_rows": eligible_rows,
            "valid_rows": valid_count,
            "missing_rows": eligible_rows - valid_count,
            "coverage_ratio": valid_count / eligible_rows,
        })
    coverage = pd.DataFrame(coverage_rows).set_index("factor_id")

    common_mask = eligible & standardized.notna().all(axis=1)
    if include_correlations:
        correlations = _cross_section_correlations(standardized, eligible)
    else:
        correlations = pd.DataFrame(
            columns=["timestamp", "factor_a", "factor_b", "spearman", "pairwise_n"],
        )
    correlation_summary = _summarize_correlations(correlations)
    return FactorPanels(
        raw=raw,
        standardized=standardized,
        eligible=eligible,
        valid_masks=valid_masks,
        common_mask=common_mask.astype(bool),
        coverage=coverage,
        correlations=correlations,
        correlation_summary=correlation_summary,
        input_diagnostics=dict(panel.diagnostics),
    )


def _cross_section_correlations(values: pd.DataFrame, eligible: pd.Series) -> pd.DataFrame:
    factor_ids = list(values.columns)
    rows: list[dict[str, Any]] = []
    for timestamp, group in values.groupby(level="timestamp", sort=True):
        member = eligible.xs(timestamp, level="timestamp")
        cross_section = group.droplevel("timestamp").where(member, axis=0)
        for left_index, factor_a in enumerate(factor_ids):
            for factor_b in factor_ids[left_index + 1:]:
                pair = cross_section[[factor_a, factor_b]].replace([np.inf, -np.inf], np.nan).dropna()
                coefficient = (
                    pair[factor_a].rank(method="average").corr(
                        pair[factor_b].rank(method="average"), method="pearson",
                    )
                    if len(pair) >= 2 else np.nan
                )
                rows.append({
                    "timestamp": timestamp,
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "spearman": float(coefficient) if pd.notna(coefficient) else np.nan,
                    "pairwise_n": int(len(pair)),
                })
    return pd.DataFrame(rows, columns=["timestamp", "factor_a", "factor_b", "spearman", "pairwise_n"])


def _summarize_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    columns = ["factor_a", "factor_b", "valid_periods", "mean", "median", "mean_abs",
               "p10", "p90", "positive_share"]
    if correlations.empty:
        return pd.DataFrame(columns=columns).set_index(["factor_a", "factor_b"])
    rows = []
    for (factor_a, factor_b), group in correlations.groupby(["factor_a", "factor_b"], sort=True):
        valid = group["spearman"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "factor_a": factor_a,
            "factor_b": factor_b,
            "valid_periods": int(len(valid)),
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "median": float(valid.median()) if len(valid) else np.nan,
            "mean_abs": float(valid.abs().mean()) if len(valid) else np.nan,
            "p10": float(valid.quantile(0.10)) if len(valid) else np.nan,
            "p90": float(valid.quantile(0.90)) if len(valid) else np.nan,
            "positive_share": float((valid > 0).mean()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows).set_index(["factor_a", "factor_b"])
