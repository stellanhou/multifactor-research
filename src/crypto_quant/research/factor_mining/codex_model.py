"""Official Codex Python SDK adapter using the operator's ChatGPT subscription."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from openai_codex import CodexConfig
from openai_codex.async_client import AsyncCodexClient
from openai_codex.errors import CodexError, InternalRpcError
from openai_codex.generated.v2_all import ConfigReadResponse, TurnStatus

from .contracts import require, text
from .model import ApiCallError, ModelReply


DEFAULT_MODEL = "gpt-5.6-luna"
PROVIDER = "codex-sdk-chatgpt"


def _config(directory, codex_bin=None, *, service_tier=None):
    # These are process-local overrides: the user's Codex settings and login stay intact.
    # Research evidence is supplied explicitly. Codex may retain its global AGENTS.md;
    # project documents, personal memories and research-external tools are disabled.
    disabled = ("shell_tool", "unified_exec", "apply_patch_freeform", "view_image", "code_mode",
                "code_mode_host", "multi_agent", "multi_agent_v2", "apps", "plugins", "browser_use",
                "computer_use", "image_generation", "memories", "memory_tool", "hooks", "codex_hooks",
                "plugin_hooks", "skill_search", "tool_suggest", "workspace_dependencies", "goals")
    overrides = (
        'model_provider="openai"', 'forced_login_method="chatgpt"',
        'chatgpt_base_url="https://chatgpt.com/backend-api"',
        'agents.enabled=false', 'web_search="disabled"',
        'project_doc_max_bytes=0', 'skills.include_instructions=false',
        'features.skip_host_skill_discovery=true', 'developer_instructions=""',
        *(f"features.{name}=false" for name in disabled),
    )
    if service_tier is not None:
        overrides += ('features.fast_mode=true',)
    options = {"cwd": directory, "config_overrides": overrides,
               "env": {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}}
    if codex_bin is not None:
        options["codex_bin"] = codex_bin
    return CodexConfig(**options)


def _codex_binary(path):
    require(isinstance(path, (str, os.PathLike)), "codex_bin must be a filesystem path")
    raw_path = os.fspath(path)
    require(isinstance(raw_path, str) and bool(raw_path.strip()), "codex_bin must be a nonempty path")
    try:
        binary = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"codex_bin does not identify an existing file: {raw_path}") from exc
    require(binary.is_file() and os.access(binary, os.X_OK),
            "codex_bin must identify an executable file")
    try:
        result = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"could not read version from codex_bin: {binary}") from exc
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"(?<![\w.])(?:v)?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)(?![\w.])", output)
    require(match is not None, f"codex_bin --version returned no parseable version: {binary}")
    return str(binary), match.group(1)


async def _subscription(codex):
    await codex.start()
    await codex.initialize()
    account = (await codex.account_read()).account
    require(account is not None and account.root.type == "chatgpt",
            "Codex requires a ChatGPT subscription login; run codex login first (API keys are not used)")


async def _thread_config(codex):
    # Empty tables merge with user settings in Codex; explicitly disable each inherited
    # server/plugin before starting a thread. Never print or persist their configuration.
    effective = await codex.request("config/read", {"includeLayers": False}, response_model=ConfigReadResponse)
    config = effective.config.model_dump(mode="json")
    return {section: {name: {"enabled": False} for name in config.get(section, {})}
            for section in ("mcp_servers", "plugins")}


class CodexModel:
    def __init__(self, model=DEFAULT_MODEL, *, reasoning_effort="max", timeout_seconds=300,
                 codex_bin=None, service_tier=None):
        self.model = text(model, "model")
        require(reasoning_effort in {"low", "medium", "high", "xhigh", "max"}, "unsupported reasoning effort")
        require(type(timeout_seconds) is int and timeout_seconds > 0, "timeout must be positive")
        require(service_tier in {None, "priority"}, "supported service tier is priority (Fast)")
        self.service_tier = service_tier
        self.reasoning_effort, self.timeout_seconds = reasoning_effort, timeout_seconds
        self.codex_bin, self.codex_runtime_version = (
            _codex_binary(codex_bin) if codex_bin is not None else (None, None))

    def settings(self):
        settings = {"provider": PROVIDER, "model": self.model, "reasoning_effort": self.reasoning_effort,
                    "timeout_seconds": self.timeout_seconds, "sdk_version": version("openai-codex"),
                    "runtime_version": (self.codex_runtime_version if self.codex_bin is not None
                                        else version("openai-codex-cli-bin"))}
        if self.codex_bin is not None:
            settings["codex_bin"] = self.codex_bin
        if self.service_tier is not None:
            settings["service_tier"] = self.service_tier
        return settings

    def available_models(self):
        return asyncio.run(self._models())

    async def _models(self):
        with tempfile.TemporaryDirectory(prefix="quant-codex-") as directory:
            codex = AsyncCodexClient(_config(directory, self.codex_bin, service_tier=self.service_tier))
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await _subscription(codex)
                    catalog = await codex.model_list()
                    return {"provider": PROVIDER, "authentication": "chatgpt",
                            **catalog.model_dump(mode="json")}
            finally:
                await codex.close()

    def complete(self, messages, *, max_output_tokens, session_id):
        # Codex turns have no per-call output token cap. Never silently ignore a contract cap.
        require(max_output_tokens is None, "Codex SDK requires output_tokens=null; per-call output caps are unsupported")
        require(bool(messages) and messages[0]["role"] == "system"
                and all(m["role"] in {"user", "assistant"} and isinstance(m["content"], str)
                        for m in messages[1:]) and isinstance(messages[0]["content"], str)
                and messages[-1]["role"] == "user", "invalid Codex conversation")
        return asyncio.run(self._complete(messages, session_id))

    async def _complete(self, messages, session_id):
        trace = {"session_id": session_id, "settings": self.settings(), "events": [],
                 "generation_started": False}
        with tempfile.TemporaryDirectory(prefix="quant-codex-") as directory:
            codex = AsyncCodexClient(_config(directory, self.codex_bin, service_tier=self.service_tier))
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await _subscription(codex)
                    catalog = await codex.model_list()
                    selected = next((m for m in catalog.data if m.model == self.model), None)
                    require(selected is not None, "requested model is not in the Codex account catalog")
                    require(any(r.reasoning_effort.value == self.reasoning_effort
                                for r in selected.supported_reasoning_efforts),
                            "model does not support requested reasoning effort")
                    if self.service_tier is not None:
                        require(selected.service_tiers is not None
                                and any(tier.id == self.service_tier for tier in selected.service_tiers),
                                "model does not support requested Fast service tier")
                    # Fresh ephemeral threads isolate A/B, roles and corrections. The transcript
                    # is lossless; the SDK accepts turn text, not an arbitrary chat-message array.
                    instructions = (messages[0]["content"] + "\n\n"
                        "You are a bounded research role. Use only the supplied conversation. "
                        "Do not use tools, files, external sources or other sessions. "
                        "The user provides a JSON conversation transcript with user/assistant roles. "
                        "Answer its final user turn with only the requested JSON object.")
                    prompt = json.dumps(messages[1:], ensure_ascii=False)
                    trace["request"] = {"base_instructions": instructions, "input": prompt}
                    thread_options = {"model": self.model, "modelProvider": "openai",
                        "cwd": directory, "baseInstructions": instructions, "developerInstructions": "",
                        "ephemeral": True, "sandbox": "read-only", "approvalPolicy": "never",
                        "config": await _thread_config(codex)}
                    turn_options = {"effort": self.reasoning_effort}
                    if self.service_tier is not None:
                        thread_options["serviceTier"] = self.service_tier
                        turn_options["serviceTier"] = self.service_tier
                    started = await codex.thread_start(thread_options)
                    require(started.model == self.model and started.model_provider == "openai",
                            "Codex started with an unexpected model/provider")
                    if self.service_tier is not None:
                        require(started.service_tier == self.service_tier,
                                "Codex did not accept the requested Fast service tier")
                        trace["accepted_service_tier"] = started.service_tier
                    trace["thread_id"] = started.thread.id
                    trace["generation_started"] = True
                    turn = await codex.turn_start(started.thread.id, prompt, turn_options)
                    trace["turn_id"] = turn.turn.id
                    answer, usage, completed = None, None, None
                    while completed is None:
                        event = await codex.next_turn_notification(turn.turn.id)
                        payload = event.payload.model_dump(mode="json", by_alias=True)
                        trace["events"].append({"method": event.method, "params": payload})
                        if event.method in {"item/started", "item/completed"}:
                            item = payload["item"]
                            require(item["type"] in {"userMessage", "agentMessage", "reasoning"},
                                    "Codex attempted a tool or context compaction; research call rejected")
                            if event.method == "item/completed" and item["type"] == "agentMessage":
                                if item.get("phase") in {None, "final_answer"}:
                                    answer = item["text"]
                        elif event.method == "thread/tokenUsage/updated":
                            usage = payload["tokenUsage"]
                        elif event.method == "turn/completed":
                            completed = event.payload.turn
                    require(completed is not None, "Codex did not return a completed turn")
                    if completed.status != TurnStatus.completed:
                        raise ApiCallError("Codex turn did not complete", retryable=False, diagnostics=trace)
                    require(usage is not None, "Codex completed without token usage")
                    require(bool(answer and answer.strip()), "Codex completed without a final answer")
                    return ModelReply(answer, usage, self.model, "stop", self.reasoning_effort, trace)
            except (TimeoutError, CodexError, ValueError) as exc:
                # The gateway owns bounded retries. Preparation timeouts are safe to repeat;
                # once turn/start is attempted, generation may already have begun remotely.
                trace["error_type"] = type(exc).__name__
                trace["error"] = str(exc)
                retryable = not trace["generation_started"] and (
                    isinstance(exc, TimeoutError) or isinstance(exc, InternalRpcError)
                    and exc.message == "workspace routing discovery timed out")
                raise ApiCallError(f"Codex SDK call failed: {type(exc).__name__}: {exc}",
                                   retryable=retryable, diagnostics=trace) from exc
            finally:
                await codex.close()
