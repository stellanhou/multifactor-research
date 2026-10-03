"""Strict schemas and boundary checks for the bounded multi-factor Agent loop."""

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from crypto_quant.research.factor_mining.contracts import identifier, require, text


def _exact_fields(value: Any, fields: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == fields, f"{label} fields do not match the declared schema")


def _text_list(value: Any, label: str, *, allow_empty: bool) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    if not allow_empty:
        require(bool(value), f"{label} must be nonempty")
    return [text(item, label) for item in value]


@dataclass(frozen=True)
class AgentSessionContract:
    """Frozen budget and input limit for one Agent-assisted run."""

    schema_version: int
    run_id: str
    experiment_budget: int
    context_bytes: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentSessionContract":
        _exact_fields(value, set(cls.__dataclass_fields__), "Agent session contract")
        result = cls(**value)
        require(type(result.schema_version) is int and result.schema_version == 1,
                "unsupported Agent session schema_version")
        identifier(result.run_id)
        require(type(result.experiment_budget) is int and result.experiment_budget > 0,
                "experiment_budget must be a positive integer")
        require(type(result.context_bytes) is int and result.context_bytes > 0,
                "context_bytes must be a positive integer")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def max_model_calls(self) -> int:
        """Initial review plus one design and one post-run review per experiment."""
        return 1 + 2 * self.experiment_budget


REVIEW_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "findings", "limitations"],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "pattern": r"\S"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "evidence_refs"],
                "properties": {
                    "claim": {"type": "string", "minLength": 1, "pattern": r"\S"},
                    "evidence_refs": {"type": "array", "items": {"type": "string", "minLength": 1,
                                                                       "pattern": r"\S"},
                                      "minItems": 1, "uniqueItems": True},
                },
            },
        },
        "limitations": {"type": "array", "items": {"type": "string", "minLength": 1,
                                                        "pattern": r"\S"}},
    },
}

DESIGN_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "hypothesis", "strategy_card_ids", "evidence_refs"],
    "properties": {
        "action": {"type": "string", "enum": ["experiment", "stop"]},
        "hypothesis": {"type": "string", "minLength": 1, "pattern": r"\S"},
        "strategy_card_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1,
                      "pattern": r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"},
            "uniqueItems": True,
            "description": "For experiment: a new subset with 2..N-1 full card IDs. For stop: an empty list.",
        },
        "evidence_refs": {"type": "array", "items": {"type": "string", "minLength": 1,
                                                          "pattern": r"\S"},
                          "minItems": 1, "uniqueItems": True},
    },
}


def _allowed_reference_set(allowed_refs: Iterable[str]) -> set[str]:
    require(isinstance(allowed_refs, (list, tuple, set, frozenset)),
            "allowed_refs must be a collection of record IDs")
    references = list(allowed_refs)
    require(all(isinstance(ref, str) and bool(ref.strip()) for ref in references),
            "allowed_refs must contain nonempty record IDs")
    require(len(references) == len(set(references)), "allowed_refs cannot repeat record IDs")
    return set(references)


def _check_references(references: list[str], allowed: set[str], label: str) -> None:
    unknown = sorted(set(references) - allowed)
    require(not unknown, f"{label} contains references outside the current evidence set: {unknown}")


def check_review(value: Any, allowed_refs: Iterable[str]) -> dict[str, Any]:
    """Validate exact review fields and bind every finding to supplied evidence IDs."""
    fields = {"summary", "findings", "limitations"}
    _exact_fields(value, fields, "review")
    summary = text(value["summary"], "summary")
    require(isinstance(value["findings"], list), "findings must be a list")
    limitations = _text_list(value["limitations"], "limitations", allow_empty=True)
    allowed = _allowed_reference_set(allowed_refs)

    findings = []
    for index, finding in enumerate(value["findings"]):
        _exact_fields(finding, {"claim", "evidence_refs"}, f"findings[{index}]")
        claim = text(finding["claim"], f"findings[{index}].claim")
        references = _text_list(finding["evidence_refs"], f"findings[{index}].evidence_refs",
                                allow_empty=False)
        require(len(references) == len(set(references)),
                f"findings[{index}].evidence_refs cannot repeat IDs")
        _check_references(references, allowed, f"findings[{index}].evidence_refs")
        findings.append({"claim": claim, "evidence_refs": references})

    return {"summary": summary, "findings": findings, "limitations": limitations}


def _id_collection(values: Any, label: str) -> list[str]:
    require(isinstance(values, (list, tuple, set, frozenset)), f"{label} must be an ID collection")
    result = [identifier(item) for item in values]
    require(len(result) == len(set(result)), f"{label} cannot contain duplicate IDs")
    return result


def _canonical_seen_subsets(seen_subsets: Iterable[Iterable[str]], pool: set[str]) -> set[tuple[str, ...]]:
    require(isinstance(seen_subsets, (list, tuple, set, frozenset)),
            "seen_subsets must be a collection of ID collections")
    canonical = set()
    for index, subset in enumerate(seen_subsets):
        ids = _id_collection(subset, f"seen_subsets[{index}]")
        require(set(ids) <= pool, f"seen_subsets[{index}] contains an ID outside the frozen card pool")
        canonical.add(tuple(sorted(ids)))
    return canonical


def check_design(
    value: Any,
    allowed_refs: Iterable[str],
    pool_ids: Iterable[str],
    seen_subsets: Iterable[Iterable[str]],
) -> dict[str, Any]:
    """Validate a stop or a new fixed-rule equal-weight card subset."""
    fields = {"action", "hypothesis", "strategy_card_ids", "evidence_refs"}
    _exact_fields(value, fields, "design")
    action = value["action"]
    require(isinstance(action, str) and action in {"experiment", "stop"},
            "design action must be experiment or stop")
    hypothesis = text(value["hypothesis"], "hypothesis")
    references = _text_list(value["evidence_refs"], "evidence_refs", allow_empty=False)
    require(len(references) == len(set(references)), "evidence_refs cannot repeat IDs")
    _check_references(references, _allowed_reference_set(allowed_refs), "evidence_refs")

    pool = _id_collection(pool_ids, "pool_ids")
    pool_set = set(pool)
    require(isinstance(value["strategy_card_ids"], list), "strategy_card_ids must be a list")
    selected = _id_collection(value["strategy_card_ids"], "strategy_card_ids")
    if action == "stop":
        require(not selected, "stop requires an empty strategy_card_ids list")
        return {"action": action, "hypothesis": hypothesis,
                "strategy_card_ids": [], "evidence_refs": references}

    require(set(selected) <= pool_set, "experiment selects a card outside the frozen pool")
    require(2 <= len(selected) < len(pool),
            "experiment subset must contain 2..N-1 cards from the frozen pool")
    canonical = tuple(sorted(selected))
    seen = _canonical_seen_subsets(seen_subsets, pool_set)
    require(canonical not in seen, "experiment subset has already been evaluated")
    return {"action": action, "hypothesis": hypothesis,
            "strategy_card_ids": list(canonical), "evidence_refs": references}
