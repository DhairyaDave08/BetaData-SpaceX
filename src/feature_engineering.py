"""
Feature engineering for space mission risk model.
Core principle: every feature for a launch at time T must only use
information available strictly BEFORE T (no future leakage).
"""

import pandas as pd
import numpy as np


# --- Static payload capacity lookup (public knowledge; dataset has no payload mass column) ---
# Approximate max LEO payload capacity in kg. Documented explicitly as an assumption.
ROCKET_PAYLOAD_CAPACITY_KG = {
    "Falcon 9": 22800, "Falcon Heavy": 63800, "Atlas V": 18850,
    "Delta IV": 28790, "Delta II": 6100, "Soyuz": 7020, "Proton": 23000,
    "Ariane 5": 21000, "Ariane 6": 21600, "Long March 2": 9200,
    "Long March 3": 11500, "Long March 5": 25000, "PSLV": 3800,
    "GSLV": 5000, "H-IIA": 10000, "H-IIB": 19000, "Electron": 300,
    "New Shepard": 0, "Space Shuttle": 27500, "Titan": 21640,
    "Cosmos": 1400, "Vostok": 4730, "Zenit": 13740,
    "Tsyklon": 4000, "Molniya": 1600, "Voskhod": 5700,
}
DEFAULT_PAYLOAD_CAPACITY = int(np.median(list(ROCKET_PAYLOAD_CAPACITY_KG.values())))


def add_payload_capacity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def lookup_capacity(rocket_family):
        for key, val in ROCKET_PAYLOAD_CAPACITY_KG.items():
            if key.lower() in str(rocket_family).lower():
                return val
        return DEFAULT_PAYLOAD_CAPACITY

    df["payload_capacity_kg"] = df["rocket_family"].apply(lookup_capacity)
    df["payload_capacity_known"] = df["rocket_family"].apply(
        lambda x: any(k.lower() in str(x).lower() for k in ROCKET_PAYLOAD_CAPACITY_KG)
    )
    return df


def add_rolling_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    LEAK-FREE rolling features. df must already be sorted by launch_date ascending.
    For each launch, stats use ONLY strictly prior launches of that group.
    """
    df = df.copy()
    assert df["launch_date"].is_monotonic_increasing, "Sort by launch_date first!"

    global_rate = df["mission_success"].mean()

    def leak_free_rolling(group_col: str, prefix: str):
        grp = df.groupby(group_col)["mission_success"]
        df[f"{prefix}_prior_flights"] = grp.cumcount()
        df[f"{prefix}_prior_success_rate"] = (
            grp.apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
        df[f"{prefix}_prior_success_rate"] = df[f"{prefix}_prior_success_rate"].fillna(global_rate)

    leak_free_rolling("rocket_family", "vehicle")
    leak_free_rolling("launch_site", "site")
    leak_free_rolling("country", "country")

    # Vehicle maturity: days since this rocket family's first-ever launch
    first_launch = df.groupby("rocket_family")["launch_date"].transform("min")
    df["vehicle_age_days"] = (df["launch_date"] - first_launch).dt.days

    return df


def group_rare_categories(df: pd.DataFrame, col: str, min_count: int = 10, other_label: str = "other") -> pd.DataFrame:
    """Groups low-frequency categories into 'other' to prevent overfitting."""
    df = df.copy()
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[f"{col}_grouped"] = df[col].where(~df[col].isin(rare), other_label)
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = add_payload_capacity(df)
    df = add_rolling_reliability_features(df)
    df = group_rare_categories(df, "rocket_family", min_count=10)
    df = group_rare_categories(df, "launch_site", min_count=10)
    df = group_rare_categories(df, "country", min_count=10)

    feature_cols = [
        "launch_date", "year", "decade", "mission_success",
        "rocket_family_grouped", "launch_site_grouped", "country_grouped",
        "payload_capacity_kg", "payload_capacity_known",
        "vehicle_prior_flights", "vehicle_prior_success_rate",
        "site_prior_flights", "site_prior_success_rate",
        "country_prior_flights", "country_prior_success_rate",
        "vehicle_age_days",
    ]
    return df[feature_cols]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/launches_clean.csv", parse_dates=["launch_date"])
    df = df.sort_values("launch_date").reset_index(drop=True)

    features = build_feature_set(df)
    features.to_csv("data/processed/features.csv", index=False)

    print(features.shape)
    print(features.head(10))
    print("\nMissing values:\n", features.isna().sum())
    print("\nGrouped rocket families:", features["rocket_family_grouped"].nunique())
    print("Grouped sites:", features["launch_site_grouped"].nunique())
    print("Grouped countries:", features["country_grouped"].nunique())
