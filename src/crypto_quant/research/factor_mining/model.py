"""One OpenCode Go transport attempt; the gateway budgets and records retries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from .contracts import require, text


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


class OpenCodeGoModel:
    """Direct Go API; key is supplied by the operator, never copied from auth files.

    Go docs: https://opencode.ai/docs/go/ (verified 2026-09-13).
    Kimi/GLM/DeepSeek/MiMo use chat; MiniMax/Qwen use messages.
    Protocol is explicit because model availability and routing can change.
    """

    def __init__(self, model: str, protocol: str, *, api_key_env: str, timeout_seconds: int,
                 reasoning_effort: str | None = None):
        self.model = text(model, "model")
        require(not model.startswith("opencode-go/"), "direct Go API uses the bare model ID")
        require(protocol in {"chat", "messages"}, "supported Go protocols are chat and messages")
        require(type(timeout_seconds) is int and timeout_seconds > 0, "timeout must be positive")
        require(reasoning_effort is None or (protocol == "chat" and reasoning_effort in {"low", "high", "max"}),
                "explicit reasoning effort currently supports chat models with low/high/max")
        self.protocol = protocol
        self.api_key_env = text(api_key_env, "api_key_env")
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None,
                 session_id: str) -> ModelReply:
        key = read_api_key(self.api_key_env, Path(".env"))
        headers = {"User-Agent": "crypto-quant-factor-mining/1.0", "x-opencode-session": session_id}
        payload: dict[str, Any] = {"model": self.model, "stream": False}
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.protocol == "chat":
            endpoint = "https://opencode.ai/zen/go/v1/chat/completions"
            headers["Authorization"] = f"Bearer {key}"
            payload["messages"] = messages
        else:
            endpoint = "https://opencode.ai/zen/go/v1/messages"
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            payload["system"] = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            payload["messages"] = [m for m in messages if m["role"] != "system"]
        try:
            response = requests.post(endpoint, json=payload, headers=headers,
                                     timeout=self.timeout_seconds, allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError) as exc:
            # Exception strings may contain request details. Persist only the failure type.
            raise ApiCallError(f"OpenCode Go transport failed: {type(exc).__name__}", retryable=True,
                               diagnostics={"error_type": type(exc).__name__}) from None
        if response.status_code != 200:
            raise ApiCallError(f"OpenCode Go returned HTTP {response.status_code}",
                               retryable=response.status_code in {408, 429} or 500 <= response.status_code < 600,
                               diagnostics={"http_status": response.status_code,
                                            "response_body": response.text.replace(key, "[REDACTED]")})
        try:
            body = response.json()
        except ValueError:
            raise ApiCallError("OpenCode Go returned invalid response JSON", retryable=True,
                               diagnostics={"http_status": 200, "response_body": response.text.replace(key, "[REDACTED]")}) from None
        # Preserve provider diagnostics, including empty answers and reasoning usage, without the key.
        def redact(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(key, "[REDACTED]")
            if isinstance(value, dict):
                return {redact(k): redact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v) for v in value]
            return value
        raw = body = redact(body)
        try:
            if self.protocol == "chat":
                choice = body["choices"][0]
                finish_reason = choice["finish_reason"]
                content = choice["message"]["content"]
            else:
                finish_reason = body["stop_reason"]
                content = "".join(part["text"] for part in body["content"] if part["type"] == "text")
            usage = body["usage"]
            require(content is None or isinstance(content, str), "unexpected content type")
            require(isinstance(usage, dict) and isinstance(finish_reason, str), "unexpected response metadata")
        except (KeyError, IndexError, TypeError, ValueError):
            raise ApiCallError("OpenCode Go response structure is invalid", retryable=True,
                               diagnostics={"http_status": 200, "raw_response": raw}) from None
        return ModelReply(content, usage, self.model, finish_reason, self.reasoning_effort, raw)


def available_go_models() -> dict[str, Any]:
    response = requests.get("https://opencode.ai/zen/go/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()
