"""Explicit research choices and validation of untrusted model responses."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


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


def model_fields(value: Any, schema: dict[str, Any], extras: list[str], pointer: str = "") -> Any:
    """Project model JSON onto declared fields, retaining types and missing-field errors.

    Supports the objects, arrays, scalar types and nullable schemas used by this
    workflow. Business constraints are checked by the existing callers.
    """
    if "anyOf" in schema:  # All unions in this workflow are nullable fields.
        if value is None:
            return None
        schema = next(option for option in schema["anyOf"] if option.get("type") != "null")
    kind = schema.get("type")
    where = pointer or "/"
    if kind == "object":
        require(isinstance(value, dict), f"{where} must be an object")
        properties = schema["properties"]
        extras.extend(pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                      for key in sorted(set(value) - set(properties)))
        missing = sorted(set(schema["required"]) - set(value))
        require(not missing, f"{where} missing required fields: {', '.join(missing)}")
        return {key: model_fields(value[key], field, extras,
                    pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
                for key, field in properties.items() if key in value}
    if kind == "array":
        require(isinstance(value, list), f"{where} must be an array")
        return [model_fields(item, schema["items"], extras, f"{pointer}/{index}")
                for index, item in enumerate(value)]
    if kind is not None:
        valid = {"string": isinstance(value, str), "boolean": type(value) is bool,
                 "integer": type(value) is int, "number": type(value) in (int, float), "null": value is None}
        require(valid[kind], f"{where} must be {kind}")
    return value


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
        require(self.fdr_method in {"BH", "BY"}, "fdr_method must be BH or BY")
        require(0 <= number(self.min_abs_ic, "min_abs_ic") <= 1, "invalid IC threshold")
        require(number(self.min_directional_spread, "min_directional_spread") >= 0, "spread threshold must be nonnegative")
        require(0 <= number(self.min_stage_share, "min_stage_share") <= 1, "invalid stage share")
        if self.output_tokens is not None:
            require(type(self.output_tokens) is int and self.output_tokens > 0, "output_tokens must be null or a positive integer")
            require(self.context_tokens > self.output_tokens, "context must reserve output capacity")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchSpec:
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bounds(self, stage: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        require(stage in {"A", "B"}, "C is reserved for final strategy testing")
        return (utc_hour(self.a_start), utc_hour(self.b_start)) if stage == "A" else (utc_hour(self.b_start), utc_hour(self.c_start))


def candidate(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "candidate must be an object")
    require(set(value) == {"name", "expression", "meaning", "hypothesis", "direction", "parent_id", "proposal_id", "change_reason"},
            "candidate fields do not match the declared schema")
    for name in ("name", "expression", "hypothesis", "change_reason"):
        text(value[name], name)
    require(isinstance(value["meaning"], str), "meaning must be text; an empty definition returns to ideation")
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
    experiment_fields = {"question", "metric", "min_improvement", "max_ic_loss", "expected_outcome",
                         "stop_condition", "pause_condition"}
    require(isinstance(experiment, dict) and set(experiment) == experiment_fields,
            "experiment design fields do not match the declared schema")
    for name in ("question", "expected_outcome", "stop_condition", "pause_condition"):
        text(experiment[name], name)
    require(experiment["metric"] in {"rank_ic", "directional_spread"},
            "primary improvement metric is not supported")
    require(number(experiment["min_improvement"], "min_improvement") > 0,
            "minimum improvement must be positive")
    require(number(experiment["max_ic_loss"], "max_ic_loss") >= 0,
            "maximum IC loss must be nonnegative")
    if value["restart_of"] is None:
        require(value["new_evidence"] is None, "new_evidence applies to a declared route restart")
    else:
        identifier(value["restart_of"])
        text(value["new_evidence"], "new evidence for restarting")
    return value
