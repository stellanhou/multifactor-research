"""Refit the nine 90-day search routes, then compare native-cadence portfolios."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_quant.research.strategy_research import multifactor_combination_research as research


def run(horizon: int, output: Path) -> None:
    output = output.resolve()
    source = ROOT / "experiments/strategy_research/selection30_20261003/source_inputs"
    routes = [r for r in research.route_definitions() if r["experiment"] in {"E2", "E3"}]
    prepared = research.prepare_inputs(source, horizon=horizon)
    profile = {"max_heavy_processes": 2, "blas_threads_per_process": 1,
               "candidate_batch_size": 256, "scope": "E2/E3 native-horizon refits and paired turnover"}
    frozen = research.freeze_run(prepared, output / "models", run_id=f"h{horizon}-combination90-20261004",
                                 route_ids=[r["route_id"] for r in routes], execution_profile=profile)
    model_root = output / "models"
    contract_sha = frozen["run_contract_sha256"]
    state_path = output / "state.json"
    research._write_json(state_path, {"status": "calibrating", "horizon_hours": horizon,
                                     "expected_model_routes": 9, "expected_paired_accounts": 180})
    e0_path = model_root / "verification/E0/e0_calibration.json"
    if not e0_path.exists():
        research.e0_calibration(prepared, model_root, batch_candidates=256)
    research.mark_e0_passed(model_root, e0_path)
    research._write_json(state_path, {"status": "refitting", "horizon_hours": horizon,
                                     "expected_model_routes": 9, "expected_paired_accounts": 180})
    expected_fits = research.common_fit_times(prepared, end=prepared.validation_contract.bounds[1]-research.HOUR)
    manifest_path = model_root / "account_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest_keys = {(r["route_id"], r["stage"], r["cost_path"]) for r in manifest}
    for route in routes:
        print(f"h{horizon} refitting {route['route_id']}: {len(expected_fits)} weekly fits", flush=True)
        research.run_route(prepared, model_root, route["route_id"], batch_candidates=256)
        fits = research._fit_records(model_root, route, prepared, contract_sha)
        if [research.pd.Timestamp(f["fit_timestamp"]) for f in fits] != expected_fits:
            raise ValueError("completed model checkpoints differ from the full frozen weekly schedule")
        for stage in ("development", "C"):
            for stress in (False, True):
                record = research.run_stage_account(prepared, model_root, route, fits, stage, stress=stress)
                if record.get("status") == "failed_at_account_guard":
                    raise ValueError(f"baseline account failed: {record}")
                key = (record["route_id"], record["stage"], record["cost_path"])
                if key not in manifest_keys:
                    manifest.append(record)
                    manifest_keys.add(key)
                research._write_json(manifest_path, manifest)
        print(f"h{horizon} completed model and four baseline accounts: {route['route_id']}", flush=True)
    prepared.verify_source_hashes()
    del prepared
    gc.collect()
    research._write_json(state_path, {"status": "paired_accounts", "horizon_hours": horizon,
                                     "completed_model_routes": 9, "expected_paired_accounts": 180})
    paired = output / "paired"
    if not (paired / "verification.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/multifactor_position_validation.py"),
                        "--source-run", str(model_root), "--output", str(paired)], check=True)
    verification = json.loads((paired / "verification.json").read_text())
    if verification["status"] != "passed" or verification["accounts"] != 180:
        raise ValueError("paired study is incomplete or unverified")
    research._write_json(state_path, {"status": "complete", "horizon_hours": horizon,
                                     "completed_model_routes": 9, "verified_paired_accounts": 180})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=[1, 4], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.horizon, args.output)
