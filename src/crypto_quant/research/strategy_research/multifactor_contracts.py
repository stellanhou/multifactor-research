"""Frozen inputs for the deterministic USD-M multi-factor baseline."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import pandas as pd

from crypto_quant.research.factor_mining.contracts import identifier, require


def _exact(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), f"{label}: expected fields {names}")


def _positive_int(value, label):
    require(type(value) is int and value > 0, f"{label} must be a positive integer")


def _number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")


@dataclass(frozen=True)
class MultifactorContract:
    schema_version: int
    run_id: str
    purpose: str
    stage: str
    start: str
    end: str
    warmup_hours: int
    horizon_hours: int
    cards: list[str]
    universe: str
    prior_data_use: str
    data_processing: str
    costs: dict
    portfolio: dict

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), "multi-factor contract must be an object")
        schema_version = value.get("schema_version")
        require(type(schema_version) is int, "schema_version must be an integer")
        if schema_version == 1:
            require(cls is MultifactorContract, "schema_version 1 is not a research contract")
            _exact(value, MultifactorContract.__dataclass_fields__, "multi-factor contract v1")
            result = MultifactorContract(**value)
            _validate_common(result, expected_purpose="engineering")
            require(result.schema_version == 1, "unsupported schema_version")
            require(result.purpose == "engineering",
                    "v1 supports engineering replay; original row-level repair provenance is unavailable")
            return result
        if schema_version == 2:
            require(cls in {MultifactorContract, ResearchContract},
                    "schema_version 2 is reserved for ResearchContract")
            fields = set(MultifactorContract.__dataclass_fields__) | {"dataset_manifest"}
            _exact(value, fields, "multi-factor research contract v2")
            result = ResearchContract(**value)
            _validate_common(result, expected_purpose="research")
            require(result.schema_version == 2, "unsupported schema_version")
            require(result.purpose == "research", "schema_version 2 purpose must be research")
            require(isinstance(result.dataset_manifest, str) and result.dataset_manifest.strip(),
                    "research dataset_manifest path is required")
            require(not Path(result.dataset_manifest).is_absolute(),
                    "research dataset_manifest must be relative to the contract")
            return result
        raise ValueError(f"unsupported multi-factor contract schema_version: {schema_version}")

    def as_dict(self):
        return asdict(self)

    @property
    def bounds(self):
        return pd.Timestamp(self.start).tz_convert("UTC"), pd.Timestamp(self.end).tz_convert("UTC")

    @property
    def input_start(self):
        return self.bounds[0] - pd.Timedelta(hours=self.warmup_hours + 1)


@dataclass(frozen=True)
class ResearchContract(MultifactorContract):
    dataset_manifest: str


def _validate_common(result: MultifactorContract, *, expected_purpose: str) -> None:
    identifier(result.run_id)
    require(result.purpose == expected_purpose, f"contract purpose must be {expected_purpose}")
    require(result.stage in {"development", "internal_validation"}, "historical stages only")
    dates = [pd.Timestamp(result.start), pd.Timestamp(result.end)]
    require(all(t.tzinfo is not None and t == t.floor("h") for t in dates),
            "UTC-aware hourly boundaries required")
    require(dates[0] < dates[1] <= pd.Timestamp.now(tz="UTC"), "invalid historical interval")
    require(dates[1] - dates[0] >= pd.Timedelta(hours=3), "at least three account hours required")
    _positive_int(result.warmup_hours, "warmup_hours")
    require(type(result.horizon_hours) is int and result.horizon_hours in {1, 4, 24},
            "supported horizon: 1/4/24h")
    require(isinstance(result.cards, list) and len(result.cards) >= 2,
            "baseline comparison requires at least two cards")
    require(all(isinstance(path, str) and path.strip() for path in result.cards),
            "cards must be explicit paths")
    require(len(set(result.cards)) == len(result.cards), "duplicate card paths")
    for name in ("universe", "prior_data_use", "data_processing"):
        require(isinstance(getattr(result, name), str) and getattr(result, name).strip(),
                f"{name} is required")
    _exact(result.costs, ("initial_capital", "fee_bps", "slippage_bps", "stress_multiplier"), "costs")
    for name, item in result.costs.items():
        _number(item, name)
    require(result.costs["initial_capital"] > 0, "initial_capital must be positive")
    require(result.costs["stress_multiplier"] >= 1, "stress_multiplier must be at least one")
    require(all(0 <= result.costs[name] * result.costs["stress_multiplier"] < 10000
                for name in ("fee_bps", "slippage_bps")), "invalid base/stress transaction costs")
    _exact(result.portfolio, ("long_count", "short_count", "gross_exposure", "max_asset_weight",
                             "rebalance_hours", "margin_fraction"), "portfolio")
    for name in ("long_count", "short_count", "rebalance_hours"):
        _positive_int(result.portfolio[name], name)
    for name in ("gross_exposure", "max_asset_weight", "margin_fraction"):
        _number(result.portfolio[name], name)
        require(0 < result.portfolio[name] <= 1, f"{name} must be in (0,1]")
    require(result.portfolio["margin_fraction"] < 1, "margin_fraction must be below one")
    require(result.portfolio["gross_exposure"] / (2 * min(result.portfolio["long_count"],
                result.portfolio["short_count"])) <= result.portfolio["max_asset_weight"],
            "equal-weight legs would exceed max_asset_weight")
