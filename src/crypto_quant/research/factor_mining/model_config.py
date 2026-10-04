"""Shared, explicit model selection for research and scan entry points."""
import json
from pathlib import Path

from .contracts import require
from .model import FACTOR_ROLES, RoleModels
from .codex_model import CodexModel, DEFAULT_MODEL as LUNA_MODEL, PROVIDER as CODEX_PROVIDER
from .mimo_model import MimoModel, DEFAULT_MODEL as MIMO_MODEL, PROVIDER as MIMO_PROVIDER


def add_model_arguments(parser, *, include_model=True, include_roles=False):
    parser.add_argument("--provider", choices=("mimo", "codex"), default="mimo")
    if include_model:
        parser.add_argument("--model", help="default: MiMo 2.6 Flash; codex provider: Luna")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="Codex only; default max")
    parser.add_argument("--codex-bin", help="Codex only; use this explicit Codex executable")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), help="MiMo only; default enabled")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    if include_roles:
        parser.add_argument("--role-models", type=Path,
                            help="Codex only; JSON model and reasoning_effort for each of the four factor roles")


def model_from_args(args):
    if getattr(args, "role_models", None) is not None:
        require(args.provider == "codex", "role models require --provider codex")
        require(args.model is None and args.reasoning_effort is None and args.thinking is None,
                "role models declare their own model and reasoning_effort; omit shared model/thinking flags")
        roles = json.loads(args.role_models.read_text(encoding="utf-8"))
        require(isinstance(roles, dict) and set(roles) == FACTOR_ROLES,
                "role-models JSON must configure exactly the four factor roles")
        models = {}
        for role, settings in roles.items():
            require(isinstance(settings, dict)
                    and {"model", "reasoning_effort"} <= set(settings)
                    and set(settings) <= {"model", "reasoning_effort", "service_tier"},
                    "each role requires model and reasoning_effort; service_tier is optional")
            models[role] = CodexModel(**settings, timeout_seconds=args.timeout_seconds,
                                      codex_bin=args.codex_bin)
        return RoleModels(models)
    if args.provider == "mimo":
        require(args.reasoning_effort is None, "MiMo uses --thinking, not --reasoning-effort")
        require(args.codex_bin is None, "--codex-bin is only supported by Codex")
        return MimoModel(args.model or MIMO_MODEL, timeout_seconds=args.timeout_seconds,
                         thinking=args.thinking or "enabled")
    require(args.provider == "codex", "unknown model provider")
    require(args.thinking is None, "--thinking is only supported by MiMo")
    return CodexModel(args.model or LUNA_MODEL, timeout_seconds=args.timeout_seconds,
                      reasoning_effort=args.reasoning_effort or "max", codex_bin=args.codex_bin)


def model_from_settings(settings):
    if settings["provider"] == "factor-role-models":
        require(set(settings) == {"provider", "roles"}
                and isinstance(settings["roles"], dict)
                and set(settings["roles"]) == FACTOR_ROLES,
                "saved role models must configure exactly the four factor roles")
        require(all(role["provider"] == CODEX_PROVIDER for role in settings["roles"].values()),
                "factor role models require Codex settings")
        model = RoleModels({role: model_from_settings(value)
                            for role, value in settings["roles"].items()})
    elif settings["provider"] == MIMO_PROVIDER:
        model = MimoModel(settings["model"], timeout_seconds=settings["timeout_seconds"],
                          thinking=settings["thinking"])
    else:
        require(settings["provider"] == CODEX_PROVIDER, "unsupported saved model provider")
        codex_bin = {"codex_bin": settings["codex_bin"]} if "codex_bin" in settings else {}
        service_tier = {"service_tier": settings["service_tier"]} if "service_tier" in settings else {}
        model = CodexModel(settings["model"], timeout_seconds=settings["timeout_seconds"],
                           reasoning_effort=settings["reasoning_effort"], **codex_bin, **service_tier)
    require(model.settings() == settings, "saved model configuration or runtime changed; create a new Goal")
    return model
