"""
SHAP-based explainability for individual mission predictions.
Uses a model-agnostic Permutation explainer since the final model is a
CalibratedClassifierCV wrapping a full sklearn Pipeline (preprocessing +
Random Forest) — not a raw tree model, so TreeExplainer doesn't apply here.
"""

import pandas as pd
import numpy as np
import shap

from src.predict import FEATURE_COLS, load_model


def build_explainer(model, background_df: pd.DataFrame, sample_size: int = 50):
    """
    Builds a SHAP explainer around the model's predict_proba function.
    background_df should be a sample of realistic training rows (used as
    the reference distribution for permutation-based attribution).
    Kept small (default 50 rows) since KernelExplainer/Permutation scales
    with background size and this needs to run interactively in a dashboard.
    """
    background = background_df[FEATURE_COLS].sample(
        min(sample_size, len(background_df)), random_state=42
    )

    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=FEATURE_COLS)
        return model.predict_proba(X_df)[:, 1]  # probability of SUCCESS

    explainer = shap.Explainer(predict_fn, background, algorithm="permutation")
    return explainer


FEATURE_LABELS = {
    "rocket_family_grouped": "Rocket family",
    "launch_site_grouped": "Launch site",
    "country_grouped": "Country/agency",
    "payload_capacity_kg": "Payload capacity",
    "payload_capacity_known": "Payload capacity is a known value",
    "vehicle_prior_flights": "Rocket family's prior flight count",
    "vehicle_prior_success_rate": "Rocket family's historical success rate",
    "site_prior_flights": "Launch site's prior flight count",
    "site_prior_success_rate": "Launch site's historical success rate",
    "country_prior_flights": "Country/agency's prior flight count",
    "country_prior_success_rate": "Country/agency's historical success rate",
    "vehicle_age_days": "Days since this rocket family's first flight",
    "decade": "Decade of launch",
    "weather_available": "Weather data available",
    "wind_speed_max_kmh": "Max wind speed on launch day",
    "temp_max_c": "Max temperature on launch day",
    "precipitation_mm": "Precipitation on launch day",
}


def explain_instance(explainer, instance_df: pd.DataFrame, top_k: int = 5) -> list:
    """
    Returns the top_k features driving this instance's SUCCESS probability,
    translated into plain-language reasons for a non-technical investigator.
    Positive SHAP value = pushes toward success; negative = pushes toward failure.
    """
    shap_values = explainer(instance_df[FEATURE_COLS])
    values = shap_values.values[0]
    features = FEATURE_COLS

    contributions = sorted(
        zip(features, values, instance_df[FEATURE_COLS].iloc[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_k]

    explanations = []
    for feat, val, raw_value in contributions:
        direction = "increases" if val > 0 else "decreases"
        label = FEATURE_LABELS.get(feat, feat)
        explanations.append({
            "feature": feat,
            "label": label,
            "value": raw_value,
            "shap_value": round(float(val), 4),
            "direction": direction,
            "plain_language": f"{label} = {raw_value} ({direction} predicted success probability)",
        })

    return explanations


if __name__ == "__main__":
    model = load_model()
    features = pd.read_csv("data/processed/features.csv", parse_dates=["launch_date"])

    explainer = build_explainer(model, features)

    test_instance = features[FEATURE_COLS].iloc[[0]]
    explanations = explain_instance(explainer, test_instance)

    for e in explanations:
        print(e["plain_language"])
