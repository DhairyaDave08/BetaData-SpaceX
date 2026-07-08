import pandas as pd
import numpy as np
import joblib
import json
import os

from src.train import CATEGORICAL_COLS, NUMERIC_COLS

MODEL_PATH = "models/model.pkl"
METADATA_PATH = "models/model_metadata.json"


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Did you upload models/model.pkl?")
    return joblib.load(path)


def load_metadata(path: str = METADATA_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found at {path}. Did you upload models/model_metadata.json?")
    with open(path, "r") as f:
        return json.load(f)


FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def _prepare_input(input_dict: dict) -> pd.DataFrame:
    """
    Converts a dict of raw inputs (from a form, simulator, or batch row)
    into a single-row DataFrame matching the model's expected columns.
    Missing fields are filled with sensible defaults so partial inputs
    (e.g., a what-if simulator that doesn't set weather) still work.
    """
    defaults = {
        "rocket_family_grouped": "other",
        "launch_site_grouped": "other",
        "country_grouped": "other",
        "payload_capacity_kg": 9000,       # dataset median-ish fallback
        "payload_capacity_known": False,
        "vehicle_prior_flights": 0,
        "vehicle_prior_success_rate": 0.90,  # global historical rate
        "site_prior_flights": 0,
        "site_prior_success_rate": 0.90,
        "country_prior_flights": 0,
        "country_prior_success_rate": 0.90,
        "vehicle_age_days": 0,
        "decade": 2020,
        "weather_available": False,
        "wind_speed_max_kmh": np.nan,
        "temp_max_c": np.nan,
        "precipitation_mm": np.nan,
    }
    row = {**defaults, **input_dict}
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_single(input_dict: dict, model=None, metadata=None) -> dict:
    """
    Scores one mission. Returns calibrated probability, a 0-100 risk score,
    a risk band, and the binary label using the chosen operating threshold.
    """
    if model is None:
        model = load_model()
    if metadata is None:
        metadata = load_metadata()

    X = _prepare_input(input_dict)
    success_prob = float(model.predict_proba(X)[:, 1][0])
    failure_prob = 1 - success_prob

    risk_score = round(failure_prob * 100, 1)  # 0-100, higher = riskier

    failure_threshold = metadata.get("chosen_threshold_failure_prob", 0.24)
    predicted_label = "Failure" if failure_prob >= failure_threshold else "Success"

    if risk_score < 10:
        risk_band = "Low"
    elif risk_score < 25:
        risk_band = "Moderate"
    elif risk_score < 50:
        risk_band = "High"
    else:
        risk_band = "Critical"

    return {
        "success_probability": round(success_prob, 4),
        "failure_probability": round(failure_prob, 4),
        "risk_score_0_100": risk_score,
        "risk_band": risk_band,
        "predicted_label": predicted_label,
        "input_used": X.iloc[0].to_dict(),
    }


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Scores multiple missions at once — used for the historical analytics view."""
    if model is None:
        model = load_model()

    X = df[FEATURE_COLS]
    success_probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["success_probability"] = success_probs
    result["risk_score_0_100"] = ((1 - success_probs) * 100).round(1)
    return result


if __name__ == "__main__":
    # Quick manual test
    example = {
        "rocket_family_grouped": "Falcon 9",
        "launch_site_grouped": "SLC-40",
        "country_grouped": "USA",
        "vehicle_prior_flights": 150,
        "vehicle_prior_success_rate": 0.98,
        "vehicle_age_days": 3600,
        "decade": 2020,
    }
    result = predict_single(example)
    print(json.dumps(result, indent=2, default=str))
