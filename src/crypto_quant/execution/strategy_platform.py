"""Deterministic local strategy-slot platform.

This module is deliberately ordinary local Python. It has no agent, LLM,
network, broker, or exchange dependency. It provides isolated virtual
sub-ledgers, lifecycle evidence, a fixed V1 promotion score, and generated
checkpoint artifacts. Any future product adapter must be explicitly injected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


MAX_SLOTS = 10
INCUMBENT_SLOTS = 8
CHALLENGER_SLOTS = 2
INITIAL_VIRTUAL_CASH = 500.0
INITIAL_VIRTUAL_CASH_DECIMAL = Decimal("500")
PLATFORM_SCHEMA_VERSION = 1

# Strategy families describe the primary source of a strategy's signal.  They
# are metadata for research coverage and auditability, not admission gates or
# a direct source of score points.
STRATEGY_FAMILIES = frozenset(
    {
        "trend_momentum",
        "mean_reversion",
        "carry_term_structure",
        "relative_value_arbitrage",
        "liquidity_order_flow",
        "volatility_convexity",
        "information_event_structural",
        "beta_allocation_regime",
        "unclassified",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


class PlatformError(RuntimeError):
    pass


@dataclass
class VirtualOrder:
    order_id: str
    strategy_id: str
    product: str
    symbol: str
    side: str
    quantity: float
    price: float
    leverage: float = 1.0
    status: str = "OPEN"
    filled_quantity: float = 0.0
    filled_notional: float = 0.0
    margin_requirement: str = "0"


@dataclass
class Position:
    signed_qty: Decimal = Decimal("0")
    avg_entry: Decimal = Decimal("0")
    reserved_margin: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    def __float__(self) -> float:
        return float(self.signed_qty)

    def __lt__(self, other: Any) -> bool:
        return self.signed_qty < Decimal(str(other))

    def snapshot(self) -> Dict[str, str]:
        return {"signed_qty": str(self.signed_qty), "avg_entry": str(self.avg_entry), "reserved_margin": str(self.reserved_margin), "realized_pnl": str(self.realized_pnl), "unrealized_pnl": str(self.unrealized_pnl)}


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    kind: str
    multiplier: Decimal = Decimal("1")
    maintenance_margin_rate: Decimal = Decimal("0.005")


SPOT_INSTRUMENT = InstrumentSpec("spot", "spot", Decimal("1"), Decimal("0"))
LINEAR_INSTRUMENT = InstrumentSpec("linear_futures", "linear", Decimal("1"), Decimal("0.005"))


@dataclass
class VirtualSubledger:
    strategy_id: str
    cash: Decimal = INITIAL_VIRTUAL_CASH_DECIMAL
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, VirtualOrder] = field(default_factory=dict)
    realized_pnl: Decimal = Decimal("0")
    reserved_order_margin: Decimal = Decimal("0")
    fee_rate: Decimal = Decimal("0")
    maintenance_margin_rate: Decimal = Decimal("0.005")
    mark_prices: Dict[str, Decimal] = field(default_factory=dict)
    instruments: Dict[str, InstrumentSpec] = field(default_factory=dict)
    reserved_spot_sell_qty: Dict[str, Decimal] = field(default_factory=dict)

    @staticmethod
    def _d(value: Any, name: str) -> Decimal:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise PlatformError(f"{name} must be finite") from exc
        if not result.is_finite():
            raise PlatformError(f"{name} must be finite")
        return result

    def _position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position())

    def equity(self, marks: Optional[Mapping[str, Any]] = None) -> Decimal:
        if marks:
            self.mark_prices.update({key: self._d(value, "mark price") for key, value in marks.items()})
        equity = self.cash
        for symbol, position in self.positions.items():
            spec = self.instruments.get(symbol, LINEAR_INSTRUMENT)
            if spec.kind == "spot":
                equity += position.signed_qty * self.mark_prices.get(symbol, position.avg_entry)
            else:
                equity += position.unrealized_pnl
        return equity

    def free_collateral(self) -> Decimal:
        return self.equity() - sum((position.reserved_margin for position in self.positions.values()), Decimal("0")) - self.reserved_order_margin

    @staticmethod
    def _incremental_linear_margin(
        position_qty: Decimal,
        current_position_margin: Decimal,
        signed_order_qty: Decimal,
        price: Decimal,
        leverage: Decimal,
    ) -> Decimal:
        """Reserve margin only for the part of an order that increases exposure."""
        required_final_margin = (
            abs(position_qty + signed_order_qty) * price / leverage
        )
        return max(
            Decimal("0"), required_final_margin - current_position_margin
        )

    def submit_order(self, order_id: str, product: str, symbol: str, side: str, quantity: float, price: float, leverage: float = 1.0) -> VirtualOrder:
        if order_id in self.orders:
            old = self.orders[order_id]
            if (old.product, old.symbol, old.side, old.quantity, old.price, old.leverage) != (product, symbol, side.upper(), float(quantity), float(price), float(leverage)):
                raise PlatformError("duplicate virtual order id has conflicting parameters")
            return old
        qty = self._d(quantity, "quantity")
        px = self._d(price, "price")
        lev = self._d(leverage, "leverage")
        if qty <= 0 or px <= 0 or lev <= 0:
            raise PlatformError("virtual order quantity, price, and leverage must be finite and positive")
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise PlatformError("virtual order side must be BUY or SELL")
        product_key = product.lower()
        if product_key in {"spot", "cash"}:
            spec = SPOT_INSTRUMENT
        elif product_key in {"future", "futures", "linear", "linear_futures", "usdm"}:
            spec = LINEAR_INSTRUMENT
        else:
            raise PlatformError("only spot and linear futures virtual accounting are implemented; inverse/options are unsupported")
        self.instruments[symbol] = spec
        derivative = spec.kind == "linear"
        position = self._position(symbol)
        if not derivative and normalized_side == "SELL" and position.signed_qty <= 0:
            raise PlatformError("spot sell would create a short position")
        if not derivative and normalized_side == "SELL" and qty > position.signed_qty - self.reserved_spot_sell_qty.get(symbol, Decimal("0")):
            raise PlatformError("spot sell quantity exceeds available unreserved position")
        signed_order_qty = qty if normalized_side == "BUY" else -qty
        margin = (
            self._incremental_linear_margin(
                position.signed_qty,
                position.reserved_margin,
                signed_order_qty,
                px,
                lev,
            )
            if derivative
            else (qty * px if normalized_side == "BUY" else Decimal("0"))
        )
        fee = qty * px * self.fee_rate
        # Closing exposure must remain possible even when no free collateral is
        # left. Reserve fees only for the risk-increasing portion; closing fees
        # settle against wallet balance when the fill is applied.
        opening_fraction = (
            margin * lev / (qty * px) if derivative and qty * px else Decimal("0")
        )
        opening_fee = fee * opening_fraction if derivative else fee
        if margin + opening_fee > self.free_collateral() + Decimal("1e-12"):
            raise PlatformError("insufficient free collateral for virtual order")
        self.reserved_order_margin += margin
        if not derivative and normalized_side == "SELL":
            self.reserved_spot_sell_qty[symbol] = self.reserved_spot_sell_qty.get(symbol, Decimal("0")) + qty
        order = VirtualOrder(order_id, self.strategy_id, product, symbol, normalized_side, float(qty), float(px), float(lev), margin_requirement=str(margin))
        self.orders[order_id] = order
        return order

    def apply_fill(self, order_id: str, quantity: float, price: Optional[float] = None) -> VirtualOrder:
        if order_id not in self.orders:
            raise PlatformError("unknown virtual order")
        order = self.orders[order_id]
        fill_price = self._d(price if price is not None else order.price, "fill price")
        fill_qty = self._d(quantity, "fill quantity")
        remaining = self._d(order.quantity - order.filled_quantity, "remaining quantity")
        if fill_qty <= 0 or fill_qty > remaining + Decimal("1e-12"):
            raise PlatformError("invalid virtual fill quantity")
        notional = fill_qty * fill_price
        spec = self.instruments.get(order.symbol, SPOT_INSTRUMENT if order.product.lower() in {"spot", "cash"} else LINEAR_INSTRUMENT)
        derivative = spec.kind == "linear"
        fee = notional * self.fee_rate
        position = self._position(order.symbol)
        realized_before = position.realized_pnl
        old_qty = position.signed_qty
        signed_delta = fill_qty if order.side == "BUY" else -fill_qty
        closing = min(abs(old_qty), abs(signed_delta)) if old_qty != 0 and old_qty * signed_delta < 0 else Decimal("0")
        opening = abs(signed_delta) - closing
        current_order_margin = self._d(order.margin_requirement, "order margin")
        if not derivative and order.side == "SELL":
            self.reserved_spot_sell_qty[order.symbol] = max(Decimal("0"), self.reserved_spot_sell_qty.get(order.symbol, Decimal("0")) - fill_qty)
        if not derivative:
            if order.side == "SELL" and fill_qty > position.signed_qty:
                raise PlatformError("spot sell would create a short position")
            if order.side == "BUY":
                self.cash -= notional + fee
                position.avg_entry = ((position.avg_entry * position.signed_qty) + notional) / (position.signed_qty + fill_qty) if position.signed_qty else fill_price
                position.signed_qty += fill_qty
            else:
                self.cash += notional - fee
                position.realized_pnl += (fill_price - position.avg_entry) * fill_qty - fee
                position.signed_qty -= fill_qty
                if position.signed_qty == 0:
                    position.avg_entry = Decimal("0")
        else:
            if closing:
                direction = Decimal("1") if old_qty > 0 else Decimal("-1")
                pnl = direction * (fill_price - position.avg_entry) * closing
                released = position.reserved_margin * (closing / abs(old_qty))
                self.cash += pnl
                position.reserved_margin -= released
                position.realized_pnl += pnl
                position.signed_qty += direction * -closing
                signed_delta += direction * closing
                if position.signed_qty == 0:
                    position.avg_entry = Decimal("0")
            if opening:
                margin = opening * fill_price / self._d(order.leverage, "leverage")
                position.reserved_margin += margin
                position.avg_entry = fill_price if position.signed_qty == 0 else ((position.avg_entry * abs(position.signed_qty)) + (fill_price * abs(signed_delta))) / (abs(position.signed_qty) + abs(signed_delta))
                position.signed_qty += signed_delta
            self.cash -= fee
            position.realized_pnl -= fee
        remaining_after_fill = self._d(
            order.quantity - order.filled_quantity - float(fill_qty),
            "remaining quantity after fill",
        )
        if derivative:
            remaining_signed = (
                remaining_after_fill if order.side == "BUY" else -remaining_after_fill
            )
            remaining_margin = self._incremental_linear_margin(
                position.signed_qty,
                position.reserved_margin,
                remaining_signed,
                self._d(order.price, "order price"),
                self._d(order.leverage, "leverage"),
            )
        elif order.side == "BUY":
            remaining_margin = remaining_after_fill * self._d(order.price, "order price")
        else:
            remaining_margin = Decimal("0")
        self.reserved_order_margin = max(
            Decimal("0"),
            self.reserved_order_margin - current_order_margin + remaining_margin,
        )
        order.margin_requirement = str(remaining_margin)
        order.filled_quantity += float(fill_qty)
        order.filled_notional += float(notional)
        order.status = "FILLED" if order.filled_quantity >= order.quantity - 1e-12 else "PARTIALLY_FILLED"
        self.realized_pnl += position.realized_pnl - realized_before
        return order

    def apply_mark(self, symbol: str, price: Any) -> Dict[str, str]:
        px = self._d(price, "mark price")
        if px <= 0:
            raise PlatformError("mark price must be positive")
        self.mark_prices[symbol] = px
        position = self._position(symbol)
        position.unrealized_pnl = position.signed_qty * (px - position.avg_entry)
        maintenance = sum((abs(pos.signed_qty * self.mark_prices.get(sym, pos.avg_entry)) * self.instruments.get(sym, LINEAR_INSTRUMENT).maintenance_margin_rate for sym, pos in self.positions.items() if self.instruments.get(sym, LINEAR_INSTRUMENT).kind == "linear"), Decimal("0"))
        liquidated = False
        liquidation_symbols: List[str] = []
        if maintenance > 0 and self.equity() <= maintenance:
            aggregate_unrealized = sum((pos.unrealized_pnl for sym, pos in self.positions.items() if self.instruments.get(sym, LINEAR_INSTRUMENT).kind == "linear"), Decimal("0"))
            liquidation_fee = Decimal("0")
            self.cash = max(Decimal("0"), self.cash + aggregate_unrealized - liquidation_fee)
            for sym, pos in self.positions.items():
                if self.instruments.get(sym, LINEAR_INSTRUMENT).kind != "linear" or pos.signed_qty == 0:
                    continue
                liquidation_symbols.append(sym)
                pos.realized_pnl += pos.unrealized_pnl
                pos.signed_qty = Decimal("0"); pos.avg_entry = Decimal("0"); pos.reserved_margin = Decimal("0"); pos.unrealized_pnl = Decimal("0")
            remaining_spot_buy_margin = Decimal("0")
            for order in self.orders.values():
                if order.status not in {"OPEN", "PARTIALLY_FILLED"}:
                    continue
                spec = self.instruments.get(order.symbol, SPOT_INSTRUMENT)
                if spec.kind == "linear":
                    order.status = "LIQUIDATED"
                    order.margin_requirement = "0"
                elif order.side == "BUY":
                    remaining_spot_buy_margin += self._d(
                        order.margin_requirement, "spot order margin"
                    )
            self.reserved_order_margin = remaining_spot_buy_margin
            liquidated = True
            self.realized_pnl = sum((pos.realized_pnl for pos in self.positions.values()), Decimal("0"))
        return {"symbol": symbol, "equity": str(max(Decimal("0"), self.equity())), "free_collateral": str(max(Decimal("0"), self.free_collateral())), "liquidated": str(liquidated).lower(), "liquidation_symbols": liquidation_symbols}

    def apply_funding(self, symbol: str, rate: Any, mark_price: Any) -> Decimal:
        funding_rate = self._d(rate, "funding rate")
        px = self._d(mark_price, "funding mark")
        position = self._position(symbol)
        payment = position.signed_qty * px * funding_rate
        self.cash -= payment
        position.realized_pnl -= payment
        self.realized_pnl -= payment
        if symbol in self.mark_prices:
            self.apply_mark(symbol, self.mark_prices[symbol])
        return payment

    def cancel_virtual_order(self, order_id: str) -> VirtualOrder:
        if order_id not in self.orders:
            raise PlatformError("unknown virtual order")
        order = self.orders[order_id]
        if order.status in {"FILLED", "CANCELED", "LIQUIDATED"}:
            return order
        remaining = self._d(order.quantity - order.filled_quantity, "remaining quantity")
        self.reserved_order_margin = max(Decimal("0"), self.reserved_order_margin - self._d(order.margin_requirement, "order margin") * (remaining / self._d(order.quantity, "order quantity")))
        if self.instruments.get(order.symbol, SPOT_INSTRUMENT).kind == "spot" and order.side == "SELL":
            self.reserved_spot_sell_qty[order.symbol] = max(Decimal("0"), self.reserved_spot_sell_qty.get(order.symbol, Decimal("0")) - remaining)
        order.status = "CANCELED"
        return order

    def snapshot(self) -> Dict[str, Any]:
        return {"strategy_id": self.strategy_id, "cash": str(self.cash), "positions": {symbol: position.snapshot() for symbol, position in self.positions.items()}, "orders": {key: asdict(value) for key, value in self.orders.items()}, "realized_pnl": str(self.realized_pnl), "reserved_order_margin": str(self.reserved_order_margin), "reserved_spot_sell_qty": {key: str(value) for key, value in self.reserved_spot_sell_qty.items()}, "fee_rate": str(self.fee_rate), "instruments": {symbol: asdict(spec) for symbol, spec in self.instruments.items()}}


@dataclass
class StrategyVersion:
    strategy_id: str
    version_id: str
    experiment_id: str
    name: str
    parent_version_id: Optional[str]
    hypothesis: str
    parameters: Dict[str, Any]
    code_hash: str
    data_hash: str
    product_types: List[str]
    allows_short: bool
    status: str = "queued"
    trial_count: int = 1
    score: Optional[Dict[str, Any]] = None
    primary_family: str = "unclassified"
    secondary_tags: List[str] = field(default_factory=list)
    profit_mechanism: str = ""
    expected_regimes: str = ""
    failure_regimes: str = ""


@dataclass(frozen=True)
class PromotionScore:
    P: float
    R: float
    B: float
    X: float
    F: float
    C: float
    D: float
    hard_gate: bool = True
    hard_gate_failures: Tuple[str, ...] = ()
    dsr_raw: float = 1.0
    final_gate_failures: Tuple[str, ...] = ()
    V: float = 0.0

    @property
    def H(self) -> float:
        return 0.30 * self.P + 0.25 * self.R + 0.30 * self.B + 0.15 * self.X

    @property
    def G(self) -> float:
        return 0.35 * self.H + 0.40 * self.F + 0.25 * self.C

    @property
    def challenger_eligible(self) -> bool:
        return self.hard_gate and self.dsr_raw >= 0.80 and self.H >= 60.0

    @property
    def final_eligible(self) -> bool:
        return self.hard_gate and not self.final_gate_failures and self.dsr_raw >= 0.95 and self.H >= 70.0 and self.F >= 65.0 and self.C >= 60.0 and self.D >= 90.0 and self.G >= 70.0

    def as_dict(self) -> Dict[str, Any]:
        return {"P": self.P, "R": self.R, "B": self.B, "X": self.X, "F": self.F, "C": self.C, "V": self.V, "D": self.D, "H": self.H, "G": self.G, "dsr_raw": self.dsr_raw, "hard_gate": self.hard_gate, "hard_gate_failures": list(self.hard_gate_failures), "final_gate_failures": list(self.final_gate_failures), "challenger_eligible": self.challenger_eligible, "final_eligible": self.final_eligible}


@dataclass(frozen=True)
class MetricInputs:
    cagr: float
    profit_factor: float
    positive_month_share: float
    hac_sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    dsr: float
    wf_positive_share: float
    neighborhood_pass: float
    stress_pass: float
    p95_participation: float
    replay_fill_ratio: float
    slippage_budget_ratio: float
    forward_positive_probability: float
    forward_hac_sharpe: float
    forward_calmar: float
    forward_mdd: float
    delta_sharpe: float
    delta_cagr: float
    mdd_improvement: float
    correlation: float
    target_position_parity: float = 1.0
    reconciliation: float = 1.0
    state_recovery_no_duplicates: float = 1.0
    slippage_budget_adherence: float = 1.0
    uptime: float = 1.0
    hard_gate_failures: Tuple[str, ...] = ()
    data_integrity: bool = True
    causal_timing: bool = True
    code_tests: bool = True
    oos_total_return: float = 1.0
    oos_duration_months: float = 12.0
    independent_cycles: float = 20.0
    doubled_cost_return: float = 1.0
    no_insolvency: bool = True
    portfolio_mdd: float = 0.0
    added_mdd_degradation: float = 0.0
    # These are upstream-computed similarities to the most similar active
    # strategy.  None preserves the pre-diversity API for historical studies;
    # new callers should provide both values.  A single missing value is
    # treated as full overlap (1.0), so partial evidence earns no unearned
    # diversity credit.
    position_overlap: Optional[float] = None
    trade_overlap: Optional[float] = None


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if not math.isfinite(float(value)):
        raise PlatformError("metric inputs must be finite")
    return max(low, min(high, float(value)))


def N(value: float, lower: float, upper: float) -> float:
    if not all(math.isfinite(float(item)) for item in (value, lower, upper)) or upper <= lower:
        raise PlatformError("N bounds and value must be finite with upper > lower")
    return _bounded((float(value) - lower) / (upper - lower) * 100.0)


def L(value: float, good: float, bad: float) -> float:
    if not all(math.isfinite(float(item)) for item in (value, good, bad)) or bad <= good:
        raise PlatformError("L bounds and value must be finite with bad > good")
    return _bounded((bad - float(value)) / (bad - good) * 100.0)


def _similarity(value: Optional[float], *, missing: float = 1.0) -> float:
    """Return a finite [0, 1] overlap, conservatively for missing evidence."""
    actual = missing if value is None else value
    if not math.isfinite(float(actual)) or not 0.0 <= float(actual) <= 1.0:
        raise PlatformError("position/trade overlap must be finite in [0, 1]")
    return float(actual)


def diversity_value(
    correlation: float,
    position_overlap: Optional[float] = None,
    trade_overlap: Optional[float] = None,
) -> float:
    """Compute the 0-100 behavior diversity reward.

    ``correlation`` is the highest return correlation to the existing pool;
    negative correlation is useful diversification and is therefore clipped
    at zero rather than converted with ``abs``.  When both overlap inputs are
    absent, the known correlation is used as a conservative proxy for both
    overlaps (so unobserved behavior cannot be treated as extra difference).
    A partially supplied pair treats the missing component as full overlap.
    """
    if not math.isfinite(float(correlation)) or not -1.0 <= float(correlation) <= 1.0:
        raise PlatformError("return correlation must be finite in [-1, 1]")
    if position_overlap is None and trade_overlap is None:
        # Historical producers did not have these two upstream measurements.
        # Use the known correlation as the only similarity evidence rather
        # than inventing unobserved position/trade differences.
        position = trade = max(0.0, float(correlation))
    else:
        position = _similarity(position_overlap)
        trade = _similarity(trade_overlap)
    similarity = (
        0.60 * max(0.0, float(correlation))
        + 0.25 * position
        + 0.15 * trade
    )
    return _bounded(100.0 * (1.0 - similarity), 0.0, 100.0)


def score_metrics(metrics: MetricInputs) -> PromotionScore:
    """Map raw platform metrics to V1 without counting the same metric twice."""
    P = 0.50 * N(metrics.cagr, 0, .30) + 0.25 * N(metrics.profit_factor, 1, 1.75) + 0.25 * N(metrics.positive_month_share, .50, .70)
    R = 0.35 * N(metrics.hac_sharpe, .30, 1.5) + 0.25 * N(metrics.sortino, .5, 2) + 0.20 * N(metrics.calmar, .3, 1.5) + 0.20 * L(abs(metrics.max_drawdown), .10, .50)
    B = 0.35 * N(metrics.dsr, .80, .99) + 0.25 * N(metrics.wf_positive_share, .50, 1) + 0.20 * N(metrics.neighborhood_pass, .50, 1) + 0.20 * N(metrics.stress_pass, .50, 1)
    X = 0.40 * L(metrics.p95_participation, .001, .01) + 0.30 * N(metrics.replay_fill_ratio, .90, 1) + 0.30 * L(metrics.slippage_budget_ratio, .50, 1)
    F = 0.35 * N(metrics.forward_positive_probability, .50, .95) + 0.25 * N(metrics.forward_hac_sharpe, 0, 1.2) + 0.20 * N(metrics.forward_calmar, 0, 1) + 0.20 * L(abs(metrics.forward_mdd), .10, .40)
    V = diversity_value(metrics.correlation, metrics.position_overlap, metrics.trade_overlap)
    C = 0.35 * N(metrics.delta_sharpe, 0, .25) + 0.25 * N(metrics.delta_cagr, 0, .10) + 0.20 * N(metrics.mdd_improvement, 0, .10) + 0.20 * V
    D = 30.0 * _bounded(metrics.target_position_parity, 0, 1) + 25.0 * _bounded(metrics.reconciliation, 0, 1) + 20.0 * _bounded(metrics.state_recovery_no_duplicates, 0, 1) + 15.0 * _bounded(metrics.slippage_budget_adherence, 0, 1) + 10.0 * _bounded(metrics.uptime, 0, 1)
    failures = list(metrics.hard_gate_failures)
    base_checks = ((metrics.data_integrity, "data_integrity"), (metrics.causal_timing, "causal_timing"), (metrics.code_tests, "code_tests"), (metrics.oos_total_return > 0, "oos_total_return"), (metrics.oos_duration_months >= 12, "oos_duration"), (metrics.independent_cycles >= 20, "independent_cycles"), (metrics.hac_sharpe >= 0.30, "hac_sharpe"), (abs(metrics.max_drawdown) <= 0.50, "max_drawdown"), (metrics.doubled_cost_return > 0, "doubled_cost_return"), (metrics.p95_participation <= 0.01, "p95_participation"), (metrics.no_insolvency, "insolvency"), (metrics.dsr >= 0.80, "dsr"))
    base_failures = failures + [name for passed, name in base_checks if not passed]
    final_failures = base_failures + [name for passed, name in ((metrics.dsr >= 0.95, "dsr_final"), (abs(metrics.portfolio_mdd) <= 0.40, "portfolio_mdd"), (metrics.added_mdd_degradation <= 0.05, "added_mdd_degradation")) if not passed]
    return PromotionScore(P, R, B, X, F, C, D, hard_gate=not bool(base_failures), hard_gate_failures=tuple(sorted(set(base_failures))), dsr_raw=metrics.dsr, final_gate_failures=tuple(sorted(set(final_failures))), V=V)


class ExecutionAdapter(Protocol):
    def status(self) -> Mapping[str, Any]:
        ...

    def submit(self, order: VirtualOrder) -> Mapping[str, Any]:
        ...

    def cancel(self, order: VirtualOrder) -> Mapping[str, Any]:
        ...


class StrategyPlatform:
    """Ten isolated virtual strategy slots with append-only evidence."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state_path = self.path / "state.json"
        self.events_path = self.path / "events.jsonl"
        self.pending_path = self.path / "pending.json"
        self.state: Dict[str, Any] = {}
        self.fault_before_checkpoint_event = False
        self.fault_after_event_append = False

    @classmethod
    def initialize(cls, path: Path) -> "StrategyPlatform":
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {"schema_version": PLATFORM_SCHEMA_VERSION, "platform_type": "local_strategy_platform", "max_slots": MAX_SLOTS, "incumbent_capacity": INCUMBENT_SLOTS, "challenger_capacity": CHALLENGER_SLOTS, "initial_virtual_cash": INITIAL_VIRTUAL_CASH, "execution_enabled": False, "slots": {}, "queue": [], "versions": {}, "trial_count": 0, "event_sequence": 0, "last_event_hash": "genesis"}
        state["state_hash"] = _hash(state)
        _atomic_json(path / "state.json", state)
        (path / "events.jsonl").write_text("", encoding="utf-8")
        obj = cls(path)
        obj.state = state
        return obj

    @classmethod
    def open(cls, path: Path) -> "StrategyPlatform":
        obj = cls(path)
        obj._recover_pending()
        state = json.loads(obj.state_path.read_text(encoding="utf-8"))
        expected = state.pop("state_hash", None)
        if expected != _hash(state) or state.get("schema_version") != PLATFORM_SCHEMA_VERSION:
            raise PlatformError("strategy platform state verification failed")
        obj.state = {**state, "state_hash": expected}
        obj._verify_events()
        return obj

    def _recover_pending(self) -> None:
        if not self.pending_path.exists():
            return
        pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
        current = json.loads(self.state_path.read_text(encoding="utf-8"))
        row = pending.get("row"); next_state = pending.get("next_state")
        if not isinstance(row, Mapping) or not isinstance(next_state, Mapping):
            raise PlatformError("strategy platform pending journal malformed")
        if current.get("state_hash") != _hash({key: value for key, value in current.items() if key != "state_hash"}):
            raise PlatformError("strategy platform pending current state hash mismatch")
        expected = _hash({"sequence": row.get("sequence"), "previous_event_hash": row.get("previous_event_hash"), "event_type": row.get("event_type"), "payload": row.get("payload")})
        if row.get("event_hash") != expected:
            raise PlatformError("strategy platform pending event hash mismatch")
        rows = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        base = int(pending.get("base_sequence", -1))
        if len(rows) == base:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(row) + "\n")
                handle.flush(); os.fsync(handle.fileno())
        elif len(rows) != base + 1 or rows[-1] != row:
            raise PlatformError("strategy platform pending event mismatch")
        if next_state.get("state_hash") != _hash({key: value for key, value in next_state.items() if key != "state_hash"}):
            raise PlatformError("strategy platform pending next state hash mismatch")
        _atomic_json(self.state_path, next_state)
        self.pending_path.unlink()

    def _verify_events(self) -> None:
        previous = "genesis"
        rows = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for sequence, row in enumerate(rows, start=1):
            expected = _hash({"sequence": row.get("sequence"), "previous_event_hash": row.get("previous_event_hash"), "event_type": row.get("event_type"), "payload": row.get("payload")})
            if row.get("sequence") != sequence or row.get("previous_event_hash") != previous or row.get("event_hash") != expected:
                raise PlatformError("strategy platform event chain verification failed")
            previous = row["event_hash"]
        if len(rows) != self.state["event_sequence"] or previous != self.state["last_event_hash"]:
            raise PlatformError("strategy platform state/event mismatch")

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        sequence = int(self.state["event_sequence"]) + 1
        previous = self.state["last_event_hash"]
        safe_payload = json.loads(_canonical(dict(payload)))
        event_hash = _hash({"sequence": sequence, "previous_event_hash": previous, "event_type": event_type, "payload": safe_payload})
        row = {"sequence": sequence, "previous_event_hash": previous, "event_type": event_type, "payload": safe_payload, "event_hash": event_hash}
        next_state = dict(self.state)
        next_state["event_sequence"] = sequence
        next_state["last_event_hash"] = event_hash
        next_state["state_hash"] = _hash({key: value for key, value in next_state.items() if key != "state_hash"})
        _atomic_json(self.pending_path, {"base_sequence": sequence - 1, "row": row, "next_state": next_state})
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(row) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        if self.fault_after_event_append:
            raise PlatformError("fault injected after strategy event append")
        _atomic_json(self.state_path, next_state)
        self.pending_path.unlink()
        self.state = next_state

    def register_version(self, name: str, hypothesis: str, parameters: Mapping[str, Any], code_hash: str, data_hash: str, *, parent_version_id: Optional[str] = None, product_types: Optional[Iterable[str]] = None, allows_short: bool = False, role: str = "challenger", primary_family: str = "unclassified", secondary_tags: Optional[Iterable[str]] = None, profit_mechanism: str = "", expected_regimes: str = "", failure_regimes: str = "") -> StrategyVersion:
        if role not in {"incumbent", "challenger", "candidate"}:
            raise PlatformError("role must be incumbent, challenger, or candidate")
        if primary_family not in STRATEGY_FAMILIES:
            raise PlatformError(f"primary_family must be one of {sorted(STRATEGY_FAMILIES)}")
        if secondary_tags is None:
            normalized_tags: List[str] = []
        elif isinstance(secondary_tags, str):
            normalized_tags = [secondary_tags]
        else:
            normalized_tags = [str(tag) for tag in secondary_tags]
        if len(normalized_tags) > 3 or any(not tag.strip() for tag in normalized_tags):
            raise PlatformError("secondary_tags must contain at most 3 non-empty tags")
        if any(not isinstance(value, str) for value in (profit_mechanism, expected_regimes, failure_regimes)):
            raise PlatformError("strategy family descriptions must be strings")
        active = [item for item in self.state["versions"].values() if item.get("status") in {"incumbent", "challenger"}]
        if role != "candidate" and (len(active) >= MAX_SLOTS or sum(item.get("status") == "incumbent" for item in active) >= INCUMBENT_SLOTS and role == "incumbent" or sum(item.get("status") == "challenger" for item in active) >= CHALLENGER_SLOTS and role == "challenger"):
            raise PlatformError("strategy slot capacity is full")
        strategy_id = "st_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        if any(item.get("strategy_id") == strategy_id for item in active):
            raise PlatformError("strategy already has an active version; retire it before replacement")
        version_id = "sv_" + _hash({"strategy_id": strategy_id, "parent": parent_version_id, "parameters": dict(parameters), "code_hash": code_hash})[:20]
        experiment_id = "exp_" + _hash({"version_id": version_id, "data_hash": data_hash, "trial": self.state["trial_count"] + 1})[:20]
        version = StrategyVersion(strategy_id, version_id, experiment_id, name, parent_version_id, hypothesis, dict(parameters), code_hash, data_hash, list(product_types or ["spot"]), bool(allows_short), role, 1, None, primary_family, normalized_tags, profit_mechanism, expected_regimes, failure_regimes)
        self.state["versions"][version_id] = asdict(version)
        if role == "candidate":
            self.state["queue"].append(version_id)
        else:
            self.state["slots"][version_id] = {"role": role, "status": role, "strategy_id": strategy_id, "version_id": version_id, "cash": INITIAL_VIRTUAL_CASH, "ledger": VirtualSubledger(strategy_id).snapshot()}
        self.state["trial_count"] += 1
        self._append("version_registered", {"version": asdict(version), "role": role})
        return version

    def set_execution_enabled(self, enabled: bool, *, reason: str) -> None:
        """Persist the Demo-only execution control in the platform event chain.

        The flag is intentionally separate from strategy promotion.  A
        strategy can be a challenger while all exchange execution remains
        disabled, and enabling it never promotes a strategy.
        """
        if not isinstance(enabled, bool) or not str(reason).strip():
            raise PlatformError("execution control needs a boolean and a reason")
        if bool(self.state.get("execution_enabled")) == enabled:
            return
        self.state["execution_enabled"] = enabled
        self._append("execution_control", {"enabled": enabled, "reason": str(reason), "environment": "binance_spot_demo"})

    def disable_execution(self, *, reason: str) -> None:
        self.set_execution_enabled(False, reason=reason)

    def score_version(self, version_id: str, components: Mapping[str, float], *, hard_gate: bool = True, hard_gate_failures: Iterable[str] = (), dsr_raw: float = 1.0, final_gate_failures: Iterable[str] = (), diversity: float = 0.0) -> Dict[str, Any]:
        """Internal/testing override; public CLI entry is raw MetricInputs."""
        if version_id not in self.state["versions"]:
            raise PlatformError("unknown strategy version")
        required = {"P", "R", "B", "X", "F", "C", "D"}
        if set(components) != required:
            raise PlatformError(f"score must contain exactly {sorted(required)}")
        values = {key: float(value) for key, value in components.items()}
        if any(not math.isfinite(value) or value < 0 or value > 100 for value in values.values()):
            raise PlatformError("score components must be finite values in [0, 100]")
        if not math.isfinite(float(diversity)) or not 0.0 <= float(diversity) <= 100.0:
            raise PlatformError("diversity must be finite in [0, 100]")
        score = PromotionScore(**values, hard_gate=hard_gate, hard_gate_failures=tuple(hard_gate_failures), dsr_raw=dsr_raw, final_gate_failures=tuple(final_gate_failures), V=float(diversity)).as_dict()
        version = self.state["versions"][version_id]
        version["score"] = score
        current_challengers = sum(item.get("status") == "challenger" for item in self.state["versions"].values())
        if version.get("status") == "candidate":
            version["status"] = "challenger" if score["challenger_eligible"] and current_challengers < CHALLENGER_SLOTS else ("queued" if score["challenger_eligible"] else "rejected")
            if version["status"] == "challenger":
                self.state["queue"] = [item for item in self.state["queue"] if item != version_id]
                self.state["slots"][version_id] = {"role": "challenger", "status": "challenger", "strategy_id": version["strategy_id"], "version_id": version_id, "cash": INITIAL_VIRTUAL_CASH, "ledger": VirtualSubledger(version["strategy_id"]).snapshot()}
        elif version.get("status") == "challenger":
            version["status"] = "challenger"
            self.state["slots"].setdefault(version_id, {})["status"] = "challenger"
        else:
            # Incumbents remain incumbents when re-scored; a final score never
            # silently demotes or replaces an active version.
            version["status"] = "incumbent" if version.get("status") == "incumbent" else ("challenger" if score["challenger_eligible"] else "rejected")
            if version["status"] == "rejected":
                self.state["slots"].pop(version_id, None)
            elif version_id in self.state["slots"]:
                self.state["slots"][version_id]["status"] = version["status"]
        if version["status"] == "rejected":
            self.state["queue"] = [item for item in self.state["queue"] if item != version_id]
        self._append("version_scored", {"version_id": version_id, "score": score, "trial_count": self.state["trial_count"]})
        return score

    def retire_version(self, version_id: str, reason: str) -> None:
        version = self.state["versions"].get(version_id)
        if version is None:
            raise PlatformError("unknown strategy version")
        version["status"] = "retired"
        self.state["slots"].pop(version_id, None)
        self.state["queue"] = [item for item in self.state["queue"] if item != version_id]
        self._append("version_retired", {"version_id": version_id, "reason": reason})
        self.promote_next_candidate()

    def promote_next_candidate(self) -> Optional[str]:
        challengers = sum(item.get("status") == "challenger" for item in self.state["versions"].values())
        if challengers >= CHALLENGER_SLOTS:
            return None
        for version_id in list(self.state["queue"]):
            version = self.state["versions"].get(version_id, {})
            score = version.get("score") or {}
            if version.get("status") == "queued" and score.get("challenger_eligible"):
                version["status"] = "challenger"
                self.state["queue"].remove(version_id)
                self.state["slots"][version_id] = {"role": "challenger", "status": "challenger", "strategy_id": version["strategy_id"], "version_id": version_id, "cash": INITIAL_VIRTUAL_CASH, "ledger": VirtualSubledger(version["strategy_id"]).snapshot()}
                self._append("candidate_promoted", {"version_id": version_id, "role": "challenger"})
                return version_id
        return None

    def replace_incumbent(self, incumbent_version_id: str, challenger_version_id: str, reason: str, *, forward_days: int, forward_cycles: int) -> None:
        incumbent = self.state["versions"].get(incumbent_version_id)
        challenger = self.state["versions"].get(challenger_version_id)
        if not incumbent or not challenger or incumbent.get("status") != "incumbent" or challenger.get("status") != "challenger":
            raise PlatformError("replacement requires active incumbent and challenger")
        if not (challenger.get("score") or {}).get("final_eligible"):
            raise PlatformError("challenger does not meet final promotion thresholds")
        if (challenger.get("score") or {}).get("G", -1) < (incumbent.get("score") or {}).get("G", -1) + 5.0:
            raise PlatformError("challenger G must exceed incumbent G by at least 5 points")
        if forward_days < 90 or forward_cycles < 20:
            raise PlatformError("incumbent replacement requires 90 forward days and 20 forward cycles")
        incumbent["status"] = "retired"
        challenger["status"] = "incumbent"
        self.state["slots"].pop(incumbent_version_id, None)
        self.state["slots"][challenger_version_id]["role"] = "incumbent"
        self.state["slots"][challenger_version_id]["status"] = "incumbent"
        self._append("incumbent_replaced", {"parent_version_id": incumbent_version_id, "version_id": challenger_version_id, "reason": reason})

    def promote_challenger_to_open_incumbent(self, challenger_version_id: str, *, forward_days: int, forward_cycles: int) -> None:
        challenger = self.state["versions"].get(challenger_version_id)
        if not challenger or challenger.get("status") != "challenger":
            raise PlatformError("promotion requires an active challenger")
        if sum(item.get("status") == "incumbent" for item in self.state["versions"].values()) >= INCUMBENT_SLOTS:
            raise PlatformError("all incumbent slots are occupied")
        if not (challenger.get("score") or {}).get("final_eligible") or forward_days < 90 or forward_cycles < 20:
            raise PlatformError("open incumbent promotion requires final score and 90 forward days/20 cycles")
        challenger["status"] = "incumbent"
        self.state["slots"][challenger_version_id]["role"] = "incumbent"
        self.state["slots"][challenger_version_id]["status"] = "incumbent"
        self._append("challenger_promoted_to_open_incumbent", {"version_id": challenger_version_id, "forward_days": forward_days, "forward_cycles": forward_cycles})

    def score_version_from_metrics(self, version_id: str, metrics: MetricInputs) -> Dict[str, Any]:
        score = score_metrics(metrics).as_dict()
        return self.score_version(version_id, {key: score[key] for key in ("P", "R", "B", "X", "F", "C", "D")}, hard_gate=score["hard_gate"], hard_gate_failures=score["hard_gate_failures"], dsr_raw=score["dsr_raw"], final_gate_failures=score["final_gate_failures"], diversity=score["V"])

    def checkpoint(self, version_id: str, output_dir: Path, *, assumptions: Mapping[str, Any], artifacts: Iterable[str] = (), shadow_status: Mapping[str, Any] | None = None, demo_status: Mapping[str, Any] | None = None, next_action: str = "review") -> Path:
        if version_id not in self.state["versions"]:
            raise PlatformError("unknown strategy version")
        version = self.state["versions"][version_id]
        artifact_paths = [str(item) for item in artifacts]
        artifact_hashes = {}
        for item in artifact_paths:
            path = Path(item)
            if path.is_file():
                artifact_hashes[item] = hashlib.sha256(path.read_bytes()).hexdigest()
        shadow_payload = dict(shadow_status or {})
        demo_payload = dict(demo_status or {})
        shadow_research_evidence = bool(shadow_payload.get("research_evidence", False))
        historical_research_evidence = bool(
            assumptions.get("historical_research_evidence", False)
        )
        research_evidence = bool(shadow_research_evidence or historical_research_evidence)
        payload = {"checkpoint_schema": 1, "created_at": datetime.now(timezone.utc).isoformat(), "strategy": version, "assumptions": dict(assumptions), "score": version.get("score"), "trial_history_count": self.state["trial_count"], "trial_history": [{"version_id": key, "status": value.get("status"), "experiment_id": value.get("experiment_id")} for key, value in self.state["versions"].items()], "shadow_status": shadow_payload, "demo_status": demo_payload, "research_evidence_source": str(assumptions.get("research_evidence_source", "platform_structured_inputs")), "demo_research_evidence": False, "demo_execution_evidence_only": bool(demo_payload), "shadow_forward_evidence": shadow_research_evidence, "research_evidence": research_evidence, "next_action": next_action, "artifacts": artifact_paths, "artifact_hashes": artifact_hashes, "platform_event_sequence": self.state["event_sequence"]}
        run_dir = Path(output_dir) / f"{version_id}_checkpoint"
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "checkpoint.json", payload)
        lines = [f"# Strategy checkpoint: {version['name']}", "", "This is a deterministic platform artifact; it is not an LLM-authored conclusion.", "", f"- Version: `{version_id}`", f"- Decision status: `{version['status']}`", f"- Research evidence: `{str(payload['research_evidence']).lower()}`", f"- Demo execution evidence only: `{str(payload['demo_execution_evidence_only']).lower()}`", f"- Shadow forward evidence: `{str(payload['shadow_forward_evidence']).lower()}`", f"- Next action: {next_action}", "", "## Score", "", json.dumps(payload["score"], indent=2) if payload["score"] else "Not scored.", ""]
        (run_dir / "checkpoint.md").write_text("\n".join(lines), encoding="utf-8")
        if self.fault_before_checkpoint_event:
            shutil.rmtree(run_dir)
            raise PlatformError("fault injected before checkpoint event")
        self._append("checkpoint_written", {"version_id": version_id, "artifact": str(run_dir), "artifact_hash": _hash(payload)})
        return run_dir


__all__ = ["CHALLENGER_SLOTS", "INCUMBENT_SLOTS", "INITIAL_VIRTUAL_CASH", "MAX_SLOTS", "STRATEGY_FAMILIES", "InstrumentSpec", "L", "MetricInputs", "N", "PlatformError", "PromotionScore", "StrategyPlatform", "StrategyVersion", "VirtualOrder", "VirtualSubledger", "diversity_value", "score_metrics"]
