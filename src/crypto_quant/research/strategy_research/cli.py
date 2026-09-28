"""Explicit replay/live entry points; replay never calls the model service."""
import json
from pathlib import Path

from crypto_quant.research.factor_mining.contracts import dumps, require
from crypto_quant.research.factor_mining.model import ModelReply
from crypto_quant.research.factor_mining.model_config import add_model_arguments, model_from_args
from .contracts import StrategyResearchContract
from .workflow import StrategyResearch, latest_result, latest_validation_result, validate_run
from .engine import DataGap, load_segment, save_snapshot
from crypto_quant.research.data_policy import strategy_usage
from crypto_quant.research.factor_mining.records import write_json
from crypto_quant.research.progress import ProgressLog


class ReplayModel:
    def __init__(self, replies: list, *, skip: int = 0):
        require(isinstance(replies, list) and bool(replies), "replay must be a nonempty array")
        self.replies = iter(replies[skip:])

    def complete(self, messages, *, max_output_tokens, session_id):
        request = json.loads(messages[1]["content"])
        item = next(self.replies)  # Exhaustion is an error, never a live-model fallback.
        require(item["role"] == request["role"], "replay role differs from requested role")
        return ModelReply(dumps(item["response"]), {"engineering_replay": True}, "fixed-replay")


def add_arguments(parser):
    commands = parser.add_subparsers(dest="research_action", required=True)
    run = commands.add_parser("run", help="develop on A+B; never load C internal validation")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--idea", type=Path, required=True)
    run.add_argument("--db", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("experiments/strategy_research"))
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--model")
    add_model_arguments(run, include_model=False)
    resume = commands.add_parser("resume", help="continue development from a saved stage checkpoint")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--db", type=Path, required=True)
    resume_mode = resume.add_mutually_exclusive_group()
    resume_mode.add_argument("--replay", type=Path)
    resume_mode.add_argument("--model")
    add_model_arguments(resume, include_model=False)
    validate = commands.add_parser("validate", help="run C internal validation; repeated attempts are saved separately")
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--db", type=Path, required=True)
    status = commands.add_parser("status", help="read saved terminal state")
    status.add_argument("--run", type=Path, required=True)
    check = commands.add_parser("check-data", help="audit and snapshot one declared segment; no model or returns")
    check.add_argument("--contract", type=Path, required=True)
    check.add_argument("--db", type=Path, required=True)
    check.add_argument("--stage", choices=("development", "validation"), required=True)
    check.add_argument("--output", type=Path, required=True)


def execute(args):
    if args.research_action == "check-data":
        contract = StrategyResearchContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
        usage = {**strategy_usage(contract), "source_database": str(args.db.resolve())}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "contract.json", contract.as_dict())
        result = {"stage": args.stage, "data_usage": usage, "model_called": False,
                  "returns_computed": False}
        try:
            frames = load_segment(args.db, contract, args.stage)
            result.update(status="data_ready", snapshot=save_snapshot(frames, args.output / "data"))
        except DataGap as exc:
            result.update(status="insufficient_data", reason=str(exc))
        write_json(args.output / "result.json", result)
        return result
    if args.research_action == "status":
        if (args.run / "result.json").exists() or list(args.run.glob("resume-*.result.json")):
            result = latest_result(args.run)
        else:
            checkpoints = sorted((args.run / "checkpoints").glob("*.json"))
            require(bool(checkpoints), "run has no saved status or checkpoint")
            checkpoint = json.loads(checkpoints[-1].read_text())
            result = {"status": "unfinished", "next_node": checkpoint["next_node"],
                      "run_id": args.run.name}
        validation = latest_validation_result(args.run)
        if validation is not None:
            result["validation_stage"] = validation
        return result
    if args.research_action == "validate":
        with ProgressLog.for_run(args.run).span("strategy.validation", heartbeat=True):
            return validate_run(args.run, args.db)
    if args.research_action == "resume":
        if args.replay:
            replies = json.loads(args.replay.read_text(encoding="utf-8"))
            consumed = len(list((args.run / "model_calls").glob("*.response.json")))
            model = ReplayModel(replies, skip=consumed)
        else:
            model = model_from_args(args)
        research = StrategyResearch.resume(args.run, args.db, model,
                                           model_mode="replay" if args.replay else "live")
        with research.progress.span("strategy.development_resume"):
            return research.run()
    contract = StrategyResearchContract.from_dict(json.loads(args.contract.read_text(encoding="utf-8")))
    strategy_usage(contract)  # Reject incompatible research dates before creating a model client.
    idea = json.loads(args.idea.read_text(encoding="utf-8"))
    if args.replay:
        model = ReplayModel(json.loads(args.replay.read_text(encoding="utf-8")))
    else:
        model = model_from_args(args)
    research = StrategyResearch(contract, idea, model, args.db, args.output,
                                model_mode="replay" if args.replay else "live")
    with research.progress.span("strategy.development"):
        return research.run()
