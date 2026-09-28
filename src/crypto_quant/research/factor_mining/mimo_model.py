"""MiMo through LangChain ChatOpenAI, with bounded SDK transport retries."""
import json
from importlib.metadata import version
from pathlib import Path

import httpx
import openai
from langchain_openai import ChatOpenAI

from .contracts import require, text
from .model import ApiCallError, ModelReply, read_api_key
from crypto_quant.research.progress import ProgressLog


DEFAULT_MODEL = "mimo-v2.6-flash"
BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
PROVIDER = "xiaomi-token-plan"
MAX_RETRIES = 2


class MimoModel:
    def __init__(self, model=DEFAULT_MODEL, *, timeout_seconds=300, thinking="enabled"):
        self.model = text(model, "model")
        require(model.startswith("mimo-"), "MiMo provider requires a MiMo model")
        require(type(timeout_seconds) is int and timeout_seconds > 0, "timeout must be positive")
        require(thinking in {"enabled", "disabled"}, "invalid MiMo thinking mode")
        self.timeout_seconds, self.thinking = timeout_seconds, thinking

    def settings(self):
        return {"provider": PROVIDER, "model": self.model, "base_url": BASE_URL,
                "api_key_env": "MIMO_API_KEY", "timeout_seconds": self.timeout_seconds,
                "thinking": self.thinking, "transport": "langchain-openai",
                "max_retries": MAX_RETRIES,
                "langchain_openai_version": version("langchain-openai"),
                "openai_version": version("openai")}

    def _call(self, messages=None, *, max_output_tokens=None):
        key = read_api_key("MIMO_API_KEY", Path(".env"))
        responses = []
        progress = ProgressLog.current()
        attempts = 0

        def record_request(request):
            nonlocal attempts
            attempts += 1
            if progress is not None:
                progress.emit("started" if attempts == 1 else "retry", "mimo.http_attempt",
                              attempt=attempts, max_attempts=MAX_RETRIES + 1)

        def record_response(response):
            # Read inside the SDK send boundary so interrupted bodies are retryable.
            try:
                response.read()
            except httpx.HTTPError as exc:
                responses.append({"http_status": response.status_code, "error_type": type(exc).__name__})
                if progress is not None:
                    progress.emit("failed", "mimo.http_response", attempt=attempts,
                                  http_status=response.status_code, error_type=type(exc).__name__)
                raise
            raw = response.text.replace(key, "[REDACTED]")
            try:
                body = json.loads(raw)
            except ValueError:
                body = raw
            responses.append({"http_status": response.status_code, "body": body})
            if progress is not None:
                progress.emit("completed", "mimo.http_response", attempt=attempts,
                              http_status=response.status_code)

        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=False,
                          event_hooks={"request": [record_request], "response": [record_response]}) as http_client:
            llm = ChatOpenAI(model=self.model, api_key=key, base_url=BASE_URL,
                timeout=self.timeout_seconds, max_retries=MAX_RETRIES,
                http_client=http_client, use_responses_api=False, streaming=False,
                default_headers={"api-key": key, "User-Agent": "crypto-quant-research/1.0"},
                extra_body={"thinking": {"type": self.thinking}})
            try:
                if messages is None:
                    llm.root_client.models.list()
                else:
                    options = {"response_format": {"type": "json_object"}}
                    if max_output_tokens is not None:
                        options["max_completion_tokens"] = max_output_tokens
                    llm.invoke(messages, **options)
            except (openai.OpenAIError, ValueError) as exc:
                # The SDK has already retried transport failures; do not multiply
                # retries in AgentGateway or expose headers/keys in exception text.
                raise ApiCallError(f"MiMo LangChain call failed: {type(exc).__name__}",
                    retryable=False, diagnostics={"error_type": type(exc).__name__,
                    "sdk_max_retries": MAX_RETRIES, "responses": responses}) from None
        require(bool(responses), "MiMo SDK returned without an HTTP response")
        return responses[-1]["body"], responses

    def available_models(self):
        body, _ = self._call()
        return body

    def complete(self, messages, *, max_output_tokens, session_id):
        body, responses = self._call(messages, max_output_tokens=max_output_tokens)
        try:
            choice = body["choices"][0]
            content, finish, usage = choice["message"]["content"], choice["finish_reason"], body["usage"]
            require(isinstance(content, str) and bool(content.strip()), "empty MiMo reply")
            require(finish == "stop" and not choice["message"].get("tool_calls"), "incomplete MiMo reply")
            require(body["model"] == self.model, "MiMo returned a different model")
            require(isinstance(usage, dict), "missing MiMo usage")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ApiCallError(f"Invalid MiMo response: {exc}", retryable=False,
                               diagnostics={"raw_response": body, "responses": responses}) from None
        return ModelReply(content, usage, self.model, finish, raw_response={
            "settings": self.settings(), "session_id": session_id, "response": body,
            "responses": responses})
