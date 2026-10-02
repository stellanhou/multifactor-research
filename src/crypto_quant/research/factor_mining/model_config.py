"""Shared, explicit model selection for research and scan entry points."""
from .contracts import require
from .codex_model import CodexModel, DEFAULT_MODEL as LUNA_MODEL, PROVIDER as CODEX_PROVIDER
from .mimo_model import MimoModel, DEFAULT_MODEL as MIMO_MODEL, PROVIDER as MIMO_PROVIDER


def add_model_arguments(parser, *, include_model=True):
    parser.add_argument("--provider", choices=("mimo", "codex"), default="mimo")
    if include_model:
        parser.add_argument("--model", help="default: MiMo 2.6 Flash; codex provider: Luna")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="Codex only; default max")
    parser.add_argument("--codex-bin", help="Codex only; use this explicit Codex executable")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), help="MiMo only; default enabled")
    parser.add_argument("--timeout-seconds", type=int, default=300)


def model_from_args(args):
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
    if settings["provider"] == MIMO_PROVIDER:
        model = MimoModel(settings["model"], timeout_seconds=settings["timeout_seconds"],
                          thinking=settings["thinking"])
    else:
        require(settings["provider"] == CODEX_PROVIDER, "unsupported saved model provider")
        codex_bin = {"codex_bin": settings["codex_bin"]} if "codex_bin" in settings else {}
        model = CodexModel(settings["model"], timeout_seconds=settings["timeout_seconds"],
                           reasoning_effort=settings["reasoning_effort"], **codex_bin)
    require(model.settings() == settings, "saved model configuration or runtime changed; create a new Goal")
    return model
