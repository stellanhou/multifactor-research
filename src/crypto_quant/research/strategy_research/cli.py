"""Explicit replay/live entry points; replay never calls the model service."""
import json
from pathlib import Path

from crypto_quant.research.factor_mining.contracts import dumps, require
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.codex_model import (
    DEFAULT_MODEL, add_model_arguments, model_from_args)
from .contracts import StrategyResearchContract
from .workflow import StrategyResearch


class ReplayModel:
    def __init__(self, replies: list):
        require(isinstance(replies, list) and bool(replies), "replay must be a nonempty array")
        self.replies = iter(replies)

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        item = next(self.replies)  # Exhaustion is an error, never a live-model fallback.
        require(item["role"] == request["role"], "replay role differs from requested role")
        return ModelReply(dumps(item["response"]), {"engineering_replay": True}, "fixed-replay")


def add_arguments(parser):
    commands = parser.add_subparsers(dest="research_action", required=True)
    run = commands.add_parser("run", help="start a new idea-driven research run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--idea", type=Path, required=True)
    run.add_argument("--db", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("experiments/strategy_research"))
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--model", default=DEFAULT_MODEL)
    add_model_arguments(run, include_model=False)
    status = commands.add_parser("status", help="read saved terminal state")
    status.add_argument("--run", type=Path, required=True)


def execute(args):
    if args.research_action == "status":
        return json.loads((args.run / "result.json").read_text(encoding="utf-8"))
    contract = StrategyResearchContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
    idea = json.loads(args.idea.read_text(encoding="utf-8"))
    if args.replay:
        model = ReplayModel(json.loads(args.replay.read_text(encoding="utf-8")))
    else:
        model = model_from_args(args)
    return StrategyResearch(contract, idea, model, args.db, args.output,
                            model_mode="replay" if args.replay else "live").run()
