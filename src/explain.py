

import pandas as pd
import numpy as np
import shap

from src.predict import FEATURE_COLS, load_model


CATEGORICAL_COLS = ["rocket_family_grouped", "launch_site_grouped", "country_grouped"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

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


def _build_category_maps(reference_df: pd.DataFrame) -> dict:
    """Builds a stable category -> integer code mapping per categorical column."""
    maps = {}
    for col in CATEGORICAL_COLS:
        categories = sorted(reference_df[col].astype(str).unique().tolist())
        if "other" not in categories:
            categories.append("other")
        maps[col] = {cat: i for i, cat in enumerate(categories)}
    return maps


def _encode(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Converts categorical string columns to integer codes. Numeric columns pass through."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        code_map = category_maps[col]
        df[col] = df[col].astype(str).map(lambda v: code_map.get(v, code_map["other"]))
    return df[FEATURE_COLS].astype(float)


def _decode(encoded_row: np.ndarray, category_maps: dict) -> pd.DataFrame:
    """Converts an encoded numeric row back into a real DataFrame the model can score."""
    row_dict = dict(zip(FEATURE_COLS, encoded_row))
    for col in CATEGORICAL_COLS:
        code = int(round(row_dict[col]))
        reverse_map = {v: k for k, v in category_maps[col].items()}
        row_dict[col] = reverse_map.get(code, "other")
    return pd.DataFrame([row_dict])[FEATURE_COLS]


def build_explainer(model, background_df: pd.DataFrame, sample_size: int = 50):
    """
    Builds a SHAP explainer around the model's predict_proba function,
    operating entirely in numeric-encoded space to avoid SHAP's known
    issue with mixed string/numeric columns in the permutation masker.
    """
    category_maps = _build_category_maps(background_df)

    background_sample = background_df[FEATURE_COLS].sample(
        min(sample_size, len(background_df)), random_state=42
    )
    background_encoded = _encode(background_sample, category_maps)

    def predict_fn(X_encoded):
        X_encoded = np.atleast_2d(X_encoded)
        rows = [_decode(row, category_maps) for row in X_encoded]
        X_decoded = pd.concat(rows, ignore_index=True)
        return model.predict_proba(X_decoded)[:, 1]

    explainer = shap.Explainer(predict_fn, background_encoded, algorithm="permutation")
    explainer._category_maps = category_maps  # stash for use in explain_instance
    return explainer


def explain_instance(explainer, instance_df: pd.DataFrame, top_k: int = 5) -> list:
    """
    Returns the top_k features driving this instance's SUCCESS probability,
    translated into plain-language reasons. Positive SHAP value = pushes
    toward success; negative = pushes toward failure.
    """
    category_maps = explainer._category_maps
    instance_encoded = _encode(instance_df, category_maps)

    shap_values = explainer(instance_encoded)
    values = shap_values.values[0]

    contributions = sorted(
        zip(FEATURE_COLS, values, instance_df[FEATURE_COLS].iloc[0]),
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
