"""
Space Mission Risk Analytics — Streamlit Dashboard.

Two views:
1. Historical Analytics — success rates sliced by vehicle, site, decade, payload class.
2. What-If Simulator — user picks inputs, sees live calibrated risk score + SHAP explanation.

Run locally: streamlit run dashboard/app.py
(Requires: models/model.pkl, models/model_metadata.json, data/processed/features.csv
 all present in the cloned repo — this dashboard does NOT retrain or refetch anything.)
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, load_metadata, FEATURE_COLS
from src.explain import build_explainer
from dashboard.components import historical_view, whatif_simulator


st.set_page_config(
    page_title="Space Mission Risk Analytics",
    page_icon="🚀",
    layout="wide",
)


@st.cache_resource
def get_model_and_metadata():
    model = load_model("models/model.pkl")
    metadata = load_metadata("models/model_metadata.json")
    return model, metadata


@st.cache_data
def get_features():
    return pd.read_csv("data/processed/features.csv", parse_dates=["launch_date"])


@st.cache_resource
def get_explainer(_model, _features):
    return build_explainer(_model, _features, sample_size=20)


def main():
    st.title("🚀 Space Mission Risk Analytics")
    st.caption(
        "Predicts launch success probability using historical vehicle, site, and "
        "weather data — with calibrated risk scores and per-mission SHAP explanations."
    )

    try:
        model, metadata = get_model_and_metadata()
        features = get_features()
        explainer = get_explainer(model, features)
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}")
        st.info(
            "Make sure `models/model.pkl`, `models/model_metadata.json`, and "
            "`data/processed/features.csv` are present in your cloned repo."
        )
        return

    tab1, tab2 = st.tabs(["📊 Historical Analytics", "🎛️ What-If Simulator"])

    with tab1:
        historical_view.render(features)

    with tab2:
        whatif_simulator.render(model, metadata, features, explainer)

    st.divider()
    st.caption(
        f"Model: Calibrated Random Forest · Test ROC-AUC: {metadata.get('roc_auc', 'N/A'):.3f} · "
        f"Brier score: {metadata.get('brier_score', 'N/A'):.4f} · "
        "Dataset: Kaggle 'All Space Missions from 1957' + Open-Meteo historical weather"
    )


if __name__ == "__main__":
    main()
