"""Explicit research choices and validation of untrusted model responses."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .structured_output import validate_schema as model_fields


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def without_hash_metadata(value: Any) -> Any:
    """Compare saved evidence without treating historical hash fields as gates."""
    if isinstance(value, dict):
        return {key: without_hash_metadata(item) for key, item in value.items()
                if key != "sha256" and not key.endswith("_sha256")}
    if isinstance(value, list):
        return [without_hash_metadata(item) for item in value]
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: Any, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def identifier(value: Any) -> str:
    require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value)),
            "identifier must contain 1-80 letters, digits, underscores or hyphens")
    return value


def number(value: Any, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{name} must be finite")
    return float(value)


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _text_schema(description: str) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


def _array_schema(items: dict[str, Any], **limits: int) -> dict[str, Any]:
    return {"type": "array", "items": items, **limits}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}



def utc_hour(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(stamp.tz is not None and str(stamp.tz) == "UTC" and stamp == stamp.floor("h"),
            "research boundaries must be exact UTC hours")
    return stamp


@dataclass(frozen=True)
class ResearchSpec:
    run_id: str
    objective: str
    purpose: str
    a_start: str
    b_start: str
    c_start: str
    c_end: str
    universe_provenance: str
    data_usage_review: str
    label: str
    sample_hours: int
    groups: int
    min_symbols: int
    min_periods: int
    hac_lags: int
    confidence: float
    stage_hours: int
    rolling_periods: int
    fdr_method: str
    fdr_alpha: float
    min_abs_ic: float
    min_directional_spread: float
    min_stage_share: float
    max_repairs: int
    max_formula_nodes: int
    max_lookback_hours: int
    context_tokens: int
    output_tokens: int | None
    b_horizons: tuple[int, ...]
    admission_scheme: str = "plan3"
    plan3_tracks_gate: bool = False

    def __post_init__(self) -> None:
        identifier(self.run_id)
        for name in ("objective", "universe_provenance", "data_usage_review"):
            text(getattr(self, name), name)
        require(self.purpose in {"research", "engineering_check"}, "unknown purpose")
        a, b, c, end = [utc_hour(getattr(self, name)) for name in ("a_start", "b_start", "c_start", "c_end")]
        require(a < b < c < end, "expected a_start < b_start < c_start < c_end")
        require(self.label == "perp_next_open_24h", "supported label: perp_next_open_24h")
        positive = ("sample_hours", "groups", "min_symbols", "min_periods", "stage_hours",
                    "rolling_periods", "max_formula_nodes", "context_tokens")
        for name in positive:
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f"{name} must be a positive integer")
        for name in ("hac_lags", "max_repairs", "max_lookback_hours"):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 0, f"{name} must be nonnegative integer")
        require(24 % self.sample_hours == 0, "sample_hours must divide 24")
        require(self.groups >= 2 and self.min_symbols >= 2 * self.groups, "need at least two assets per group")
        require(self.hac_lags >= 24 // self.sample_hours - 1, "HAC lag must cover overlapping 24h returns")
        require(self.min_periods > self.hac_lags + 1 and self.rolling_periods >= 2, "insufficient statistical window")
        for name in ("confidence", "fdr_alpha"):
            require(0 < number(getattr(self, name), name) < 1, f"{name} must lie in (0,1)")
        require(self.fdr_method == "BH", "FM-v6 requires BH Rank IC correction")
        require(self.admission_scheme == "plan3", "FM-v6 requires admission_scheme=plan3")
        require(self.plan3_tracks_gate is False, "FM-v6 requires plan3_tracks_gate=false")
        require(isinstance(self.b_horizons, (list, tuple))
                and all(type(horizon) is int for horizon in self.b_horizons)
                and tuple(self.b_horizons) == (1, 4, 24),
                "FM-v6 b_horizons must be exactly [1, 4, 24]")
        object.__setattr__(self, "b_horizons", tuple(self.b_horizons))
        require(0 <= number(self.min_abs_ic, "min_abs_ic") <= 1, "invalid IC threshold")
        require(number(self.min_directional_spread, "min_directional_spread") >= 0, "spread threshold must be nonnegative")
        require(0 <= number(self.min_stage_share, "min_stage_share") <= 1, "invalid stage share")
        if self.output_tokens is not None:
            require(type(self.output_tokens) is int and self.output_tokens > 0, "output_tokens must be null or a positive integer")
            require(self.context_tokens > self.output_tokens, "context must reserve output capacity")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchSpec:
        require(isinstance(value, dict), "research contract must be an object")
        require({"fdr_method", "admission_scheme", "plan3_tracks_gate", "b_horizons"} <= value.keys(),
                "FM-v6 contract must explicitly declare fdr_method, admission_scheme, plan3_tracks_gate and b_horizons")
        require(isinstance(value["b_horizons"], list), "FM-v6 b_horizons must be a list")
        return cls(**{**value, "b_horizons": tuple(value["b_horizons"])})

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["b_horizons"] = list(self.b_horizons)
        return value

    def bounds(self, stage: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        require(stage in {"A", "B"}, "C is reserved for strategy internal validation; factor mining cannot read C")
        return (utc_hour(self.a_start), utc_hour(self.b_start)) if stage == "A" else (utc_hour(self.b_start), utc_hour(self.c_start))


def candidate(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "candidate must be an object")
    require(set(value) == {"name", "expression", "meaning", "hypothesis", "direction", "parent_id", "proposal_id", "change_reason"},
            "candidate fields do not match the declared schema")
    for name in ("name", "expression", "hypothesis", "change_reason"):
        text(value[name], name)
    require(isinstance(value["meaning"], str), "meaning must be text")
    require(type(value["direction"]) is int and value["direction"] in {-1, 1}, "declare fixed direction -1 or 1")
    for name in ("parent_id", "proposal_id"):
        if value[name] is not None:
            identifier(value[name])
    return value


def modification_plan(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "modification plan must be an object")
    expected = {"route_id", "control_id", "evidence_refs", "modification_task", "experiment_design",
                "restart_of", "new_evidence"}
    require(set(value) == expected, "modification plan fields do not match the declared schema")
    for name in ("route_id", "control_id"):
        identifier(value[name])
    refs = value["evidence_refs"]
    require(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs)),
            "modification evidence references must be a nonempty unique list")
    for ref in refs:
        identifier(ref)
    task = value["modification_task"]
    task_fields = {"core_hypothesis", "observed_problem", "modification_hypothesis", "change_target",
                   "fixed_components"}
    require(isinstance(task, dict) and set(task) == task_fields,
            "modification task fields do not match the declared schema")
    for name in task_fields:
        text(task[name], name)
    experiment = value["experiment_design"]
    base_experiment_fields = {"question", "metric", "min_improvement", "max_ic_loss", "expected_outcome",
                              "stop_condition", "pause_condition"}
    require(isinstance(experiment, dict), "experiment design must be an object")
    metric = experiment.get("metric")
    if metric == "rank_displacement":
        experiment_fields = base_experiment_fields | {"horizon_hours", "displacement_hours"}
    else:
        experiment_fields = base_experiment_fields
    require(set(experiment) == experiment_fields,
            "experiment design fields do not match the declared schema")
    for name in ("question", "expected_outcome", "stop_condition", "pause_condition"):
        text(experiment[name], name)
    require(metric in {"rank_ic", "directional_spread", "rank_displacement"},
            "primary improvement metric is not supported")
    if metric == "rank_displacement":
        for name in ("horizon_hours", "displacement_hours"):
            require(type(experiment[name]) is int and experiment[name] in {1, 4, 24},
                    f"{name} must be explicitly set to 1, 4 or 24")
    min_improvement = number(experiment["min_improvement"], "min_improvement")
    require(min_improvement > 0,
            "rank-displacement min_improvement must be a positive absolute D decrease"
            if metric == "rank_displacement" else "minimum improvement must be positive")
    require(number(experiment["max_ic_loss"], "max_ic_loss") >= 0,
            "maximum IC loss must be nonnegative")
    if value["restart_of"] is None:
        require(value["new_evidence"] is None, "new_evidence applies to a declared route restart")
    else:
        identifier(value["restart_of"])
        text(value["new_evidence"], "new evidence for restarting")
    return value
