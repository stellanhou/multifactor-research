"""Model protocol and shared response/error types for the current providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


from .contracts import require


def read_api_key(name: str, env_file: Path) -> str:
    """Environment takes precedence; the local file contains literal single-line values."""
    key = os.environ.get(name)
    if key is None and env_file.is_file():
        values = [line.partition("=")[2].strip() for line in env_file.read_text(encoding="utf-8").splitlines()
                  if line.partition("=")[0].strip() == name]
        require(len(values) <= 1, f"{env_file.name} contains duplicate {name} entries")
        if values:
            key = values[0]
            if key[:1] in {"'", '"'}:
                require(len(key) >= 2 and key[-1] == key[0], f"check the quotes around {name} in {env_file.name}")
                key = key[1:-1]
    require(bool(key) and not any(character.isspace() for character in key),
            f"fill {name} in {env_file.name} or set the environment variable before a real model run")
    return key


@dataclass
class ModelReply:
    text: str | None
    usage: dict[str, Any]
    model: str
    finish_reason: str = "stop"
    requested_reasoning_effort: str | None = None
    raw_response: dict[str, Any] | None = None


class ApiCallError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.retryable = retryable
        self.diagnostics = diagnostics


class JsonModel(Protocol):
    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None,
                 session_id: str) -> ModelReply: ...


FACTOR_ROLES = {"ideator", "calculator", "evaluator", "optimizer"}


class RoleModels:
    """Keep each factor role on its explicitly configured model, including repairs."""

    def __init__(self, models: dict[str, JsonModel]):
        require(set(models) == FACTOR_ROLES, "configure exactly the four factor roles")
        self.models = models

    def settings(self):
        return {"provider": "factor-role-models",
                "roles": {role: model.settings() for role, model in self.models.items()}}

    def complete(self, messages, *, max_output_tokens, session_id):
        role = json.loads(messages[1]["content"])["role"]
        require(role in self.models, "unknown factor model role")
        return self.models[role].complete(messages, max_output_tokens=max_output_tokens,
                                          session_id=session_id)
