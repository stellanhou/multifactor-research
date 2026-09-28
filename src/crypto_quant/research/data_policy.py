"""Shared research read boundaries; raw market storage is purpose-neutral."""
import pandas as pd

POLICY_ID = "research-data-20260922"
A_START = "2022-08-01T00:00:00Z"
B_START = "2024-08-01T00:00:00Z"
C_START = "2025-08-01T00:00:00Z"
C_END = "2026-08-01T00:00:00Z"
PRIMARY_DATA_PROCESSING = ("market_data/crypto_quant.sqlite: database-wide interior price/volume, metrics and "
    "funding gaps were interpolated in place on 2026-09-22; funding marks may be estimated from hourly mark prices. "
    "Boundary gaps were subsequently filled with the mean of the nearest two available values on the same side; "
    "fields with fewer than two values remain missing. These are retrospective estimates, not original causal observations. "
    "See 小时行情插值处理记录.md; raw timestamps do not enforce interpolation availability.")


def _check_dates(actual, expected):
    if tuple(pd.Timestamp(t) for t in actual) != tuple(pd.Timestamp(t) for t in expected):
        raise ValueError(f"{POLICY_ID}: research dates must be {expected}; "
                         "custom samples require explicit engineering purpose")


def strategy_usage(contract):
    engineering = contract.purpose == "engineering"
    return {"policy_id": POLICY_ID, "purpose": contract.purpose,
            "primary_database_processing": PRIMARY_DATA_PROCESSING,
            "development": {"use": "engineering_development" if engineering else "strategy_development_AB",
                            "start": contract.development_start, "end_exclusive": contract.validation_start},
            "validation": {"use": "engineering_validation" if engineering else "strategy_internal_validation_C",
                           "start": contract.validation_start, "end_exclusive": contract.validation_end},
            "prior_data_use": contract.prior_data_use,
            "independence": "historical development/internal validation; C has prior exposure; not final testing",
            "final_test": "future simulated-account observations after strategy freeze and test activation",
            "final_test_started": False}


def strategy_bounds(contract, stage):
    if stage not in {"development", "validation"}:
        raise ValueError("historical strategy reads support development/internal validation only; final test is forward")
    use = strategy_usage(contract)[stage]
    return pd.Timestamp(use["start"]), pd.Timestamp(use["end_exclusive"])


def factor_usage(spec, stage):
    if stage not in {"A", "B"}:
        raise ValueError("factor mining may read A/B only; C belongs to strategy internal validation")
    if spec.purpose == "research":
        _check_dates((spec.a_start, spec.b_start, spec.c_start, spec.c_end),
                     (A_START, B_START, C_START, C_END))
    elif spec.purpose != "engineering_check":
        raise ValueError("unknown factor data purpose")
    start, end = spec.bounds(stage)
    return {"policy_id": POLICY_ID, "purpose": spec.purpose,
            "primary_database_processing": PRIMARY_DATA_PROCESSING,
            "use": ("factor_development_A" if stage == "A" else "factor_candidate_validation_B")
                   if spec.purpose == "research" else f"engineering_factor_{stage}",
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "warmup_start": (start-pd.Timedelta(hours=spec.max_lookback_hours)).isoformat(),
            "prior_data_use": spec.data_usage_review, "final_test_started": False}


def policy_catalog():
    return {"policy_id": POLICY_ID, "timezone": "UTC",
            "primary_database_processing": PRIMARY_DATA_PROCESSING, "intervals": "left closed, right open",
            "factor_development": [A_START, B_START], "factor_validation": [B_START, C_START],
            "strategy_development": [A_START, C_START], "strategy_internal_validation": [C_START, C_END],
            "final_test": "simulated account after freeze; no historical final-test endpoint",
            "raw_data": "collection, coverage audit and legacy diagnostics are not formal research admission"}
