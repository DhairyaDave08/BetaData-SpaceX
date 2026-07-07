"""
Sanity check: rolling reliability features must never use information
from the current row or any future row.
"""

import pandas as pd
import sys
sys.path.append(".")
from src.feature_engineering import add_rolling_reliability_features


def test_no_leakage_simple_case():
    df = pd.DataFrame({
        "launch_date": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01", "2000-04-01"]),
        "rocket_family": ["Falcon 9", "Falcon 9", "Falcon 9", "Falcon 9"],
        "launch_site": ["A", "A", "A", "A"],
        "country": ["USA", "USA", "USA", "USA"],
        "mission_success": [1, 0, 1, 1],
    })

    result = add_rolling_reliability_features(df)

    # First launch: no prior history -> prior_flights == 0
    assert result.loc[0, "vehicle_prior_flights"] == 0

    # Second launch: exactly 1 prior flight, which succeeded -> prior success rate == 1.0
    assert result.loc[1, "vehicle_prior_flights"] == 1
    assert result.loc[1, "vehicle_prior_success_rate"] == 1.0

    # Third launch: 2 priors (success, failure) -> prior success rate == 0.5
    assert result.loc[2, "vehicle_prior_flights"] == 2
    assert abs(result.loc[2, "vehicle_prior_success_rate"] - 0.5) < 1e-9

    # Fourth launch: 3 priors (success, failure, success) -> prior success rate == 2/3
    assert result.loc[3, "vehicle_prior_flights"] == 3
    assert abs(result.loc[3, "vehicle_prior_success_rate"] - (2/3)) < 1e-9

    print("PASSED: no-leakage rolling feature test")


if __name__ == "__main__":
    test_no_leakage_simple_case()
