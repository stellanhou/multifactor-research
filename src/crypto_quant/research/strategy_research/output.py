"""Typed role replies; execution parameters still use the strategy compiler."""
from typing import Annotated, Literal

from pydantic import Field, model_validator

from crypto_quant.research.factor_mining.structured_output import ReplyModel


Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Texts = Annotated[list[Text], Field(min_length=1)]


class Meaning(ReplyModel):
    inputs: Text
    signal: Text
    execution: Text
    position: Text
    costs: Text
    parameters: Text


class Design(ReplyModel):
    action: Literal["propose", "design", "unsupported"]
    reason: Text
    hypothesis: Text
    falsification_conditions: Texts
    calculation_meaning: Meaning | None
    # Route-dependent executable fields stay under the existing strict compiler.
    parameters: dict | None

    @model_validator(mode="after")
    def action_payload(self):
        if self.action == "unsupported":
            if self.parameters is not None or self.calculation_meaning is not None:
                raise ValueError("unsupported requires null parameters and calculation_meaning")
        elif self.parameters is None or self.calculation_meaning is None:
            raise ValueError("design/propose requires parameters and calculation_meaning")
        return self


class Check(ReplyModel):
    matches: bool | None
    reason: Text


class Checks(ReplyModel):
    inputs: Check
    signal: Check
    execution: Check
    position: Check
    costs: Check
    parameters: Check


class Calculation(ReplyModel):
    version: Text
    plan_sha256: Text
    consistent: bool | None
    checks: Checks
    hypothesis_matches: bool | None
    hypothesis_reason: Text
    evidence_refs: Texts

class Review(ReplyModel):
    summary: Text
    supporting_evidence: list[Text]
    counter_evidence: list[Text]
    limitations: Texts
    evidence_refs: Texts


class Modification(ReplyModel):
    # Unlike explanatory fields, extra executable task fields are not ignored.
    model_config = {"extra": "forbid"}
    change_parameter: Text
    hypothesis: Text
    min_return_improvement: Annotated[float, Field(gt=0)]
    max_drawdown_increase: Annotated[float, Field(ge=0)]


class Decision(ReplyModel):
    action: Literal["optimize", "retain", "pause", "discard"]
    reason: Text
    research_basis: Text
    modification_hypothesis: Text
    verifiable_improvement: Text
    attempt_value: Text
    evidence_refs: Texts
    selected_version: Text | None
    modification: Modification | None
    resume_condition: Text | None

    @model_validator(mode="after")
    def action_payload(self):
        errors = []
        for name, required_action in (("selected_version", "retain"), ("modification", "optimize"),
                                      ("resume_condition", "pause")):
            if (getattr(self, name) is not None) != (self.action == required_action):
                errors.append(f"{name} must be non-null only for action={required_action}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


ROLE_OUTPUTS = {"design": Design, "calculate": Calculation, "review": Review, "decide": Decision}
