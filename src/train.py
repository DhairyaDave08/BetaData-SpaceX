

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


CATEGORICAL_COLS = ["rocket_family_grouped", "launch_site_grouped", "country_grouped"]

NUMERIC_COLS = [
    "payload_capacity_kg", "payload_capacity_known",
    "vehicle_prior_flights", "vehicle_prior_success_rate",
    "site_prior_flights", "site_prior_success_rate",
    "country_prior_flights", "country_prior_success_rate",
    "vehicle_age_days", "decade",
    "weather_available", "wind_speed_max_kmh", "temp_max_c", "precipitation_mm",
]


def era_stratified_split(df: pd.DataFrame, test_size: float = 0.2):
    
    df = df.sort_values("launch_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train: {train_df['launch_date'].min().date()} to {train_df['launch_date'].max().date()} "
          f"({len(train_df)} rows, {train_df['mission_success'].mean():.1%} success rate)")
    print(f"Test:  {test_df['launch_date'].min().date()} to {test_df['launch_date'].max().date()} "
          f"({len(test_df)} rows, {test_df['mission_success'].mean():.1%} success rate)")

    return train_df, test_df


def build_logistic_pipeline() -> Pipeline:
   
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, CATEGORICAL_COLS),
        ("num", numeric_transformer, NUMERIC_COLS),
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    return pipeline


def get_X_y(df: pd.DataFrame):
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df["mission_success"]
    return X, y
