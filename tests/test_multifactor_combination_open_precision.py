import numpy as np
import pandas as pd
import pytest

from crypto_quant.research.strategy_research.multifactor_combination_research import (
    _audit_label_account_open_consistency,
)


def _source_files():
    return {
        stage: {
            "factor_label_open": f"{stage}/inputs/panel.values.csv#perp_open",
            "account_market_open": f"{stage}/inputs/market/{{symbol}}.csv#open",
        }
        for stage in ("development", "internal_validation")
    }


def test_shib_reciprocal_multiply_and_division_are_one_ulp_and_keep_lineage():
    timestamp = pd.Timestamp("2023-07-01T15:00:00Z")
    native_open = np.float64(0.007632)
    label_open = np.array([[native_open * (1.0 / 1000.0)]], dtype=np.float64)
    account_open = np.array([[native_open / 1000]], dtype=np.float64)
    index = pd.DatetimeIndex([timestamp])
    columns = pd.Index(["SHIBUSDT"])

    result = _audit_label_account_open_consistency(
        pd.DataFrame(label_open, index=index, columns=columns),
        pd.DataFrame(account_open, index=index, columns=columns),
        source_files_by_stage=_source_files(),
        development_timestamps={timestamp},
    )

    assert result["finite_comparisons"] == 1
    assert result["different_values"] == 1
    assert result["max_ulp_distance"] == result["accepted_max_ulp_distance"] == 1
    assert result["mismatch_ulp_histogram"] == {"1": 1}
    assert result["mismatch_by_symbol"] == {"SHIBUSDT": 1}
    example = result["examples"][0]
    assert example["stage"] == "development"
    assert example["factor_label_source"] == "development/inputs/panel.values.csv#perp_open"
    assert example["account_market_source"] == "development/inputs/market/SHIBUSDT.csv#open"

    two_steps_low = np.nextafter(account_open, 0.0)
    assert int(label_open.view(np.uint64)[0, 0] - two_steps_low.view(np.uint64)[0, 0]) == 2
    with pytest.raises(ValueError, match="more than one binary64 ULP"):
        _audit_label_account_open_consistency(
            pd.DataFrame(label_open, index=index, columns=columns),
            pd.DataFrame(two_steps_low, index=index, columns=columns),
            source_files_by_stage=_source_files(),
            development_timestamps={timestamp},
        )
