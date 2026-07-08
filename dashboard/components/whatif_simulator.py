"""
What-if simulator: user tweaks rocket family, site, prior flights, and
weather, sees a live calibrated risk score plus a SHAP explanation.
Satisfies Objective #7.
"""

import streamlit as st
import pandas as pd

from src.predict import predict_single
from src.explain import explain_instance
from dashboard.components.shap_display import render_shap_explanation


def render(model, metadata, features: pd.DataFrame, explainer):
    st.subheader("What-If Mission Simulator")
    st.caption("Adjust the inputs below to see how launch conditions affect predicted risk.")

    rocket_options = sorted(features["rocket_family_grouped"].unique().tolist())
    site_options = sorted(features["launch_site_grouped"].unique().tolist())
    country_options = sorted(features["country_grouped"].unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        rocket_family = st.selectbox("Rocket family", rocket_options,
                                      index=rocket_options.index("other") if "other" in rocket_options else 0)
        vehicle_prior_flights = st.slider("Rocket family's prior flights", 0, 300, 50)
        vehicle_prior_success_rate = st.slider("Rocket family's historical success rate", 0.0, 1.0, 0.90, 0.01)
        vehicle_age_days = st.slider("Days since this rocket's first flight", 0, 20000, 3000)

    with col2:
        launch_site = st.selectbox("Launch site", site_options,
                                    index=site_options.index("other") if "other" in site_options else 0)
        site_prior_flights = st.slider("Site's prior flights", 0, 500, 100)
        site_prior_success_rate = st.slider("Site's historical success rate", 0.0, 1.0, 0.90, 0.01)

    with col3:
        country = st.selectbox("Country / agency", country_options,
                                index=country_options.index("other") if "other" in country_options else 0)
        decade = st.selectbox("Decade", [1960, 1970, 1980, 1990, 2000, 2010, 2020], index=6)
        payload_capacity_kg = st.slider("Payload capacity (kg)", 0, 65000, 10000, step=500)

    st.markdown("**Weather conditions (optional)**")
    weather_known = st.checkbox("Include weather in this prediction", value=False)
    wind_speed = temp_max = precipitation = None
    if weather_known:
        wcol1, wcol2, wcol3 = st.columns(3)
        wind_speed = wcol1.slider("Max wind speed (km/h)", 0, 100, 20)
        temp_max = wcol2.slider("Max temperature (°C)", -20, 45, 25)
        precipitation = wcol3.slider("Precipitation (mm)", 0, 100, 0)

    if st.button("Predict Mission Risk", type="primary"):
        input_dict = {
            "rocket_family_grouped": rocket_family,
            "launch_site_grouped": launch_site,
            "country_grouped": country,
            "payload_capacity_kg": payload_capacity_kg,
            "payload_capacity_known": True,
            "vehicle_prior_flights": vehicle_prior_flights,
            "vehicle_prior_success_rate": vehicle_prior_success_rate,
            "site_prior_flights": site_prior_flights,
            "site_prior_success_rate": site_prior_success_rate,
            "country_prior_flights": site_prior_flights,  # reasonable proxy if not separately tracked
            "country_prior_success_rate": site_prior_success_rate,
            "vehicle_age_days": vehicle_age_days,
            "decade": decade,
            "weather_available": weather_known,
            "wind_speed_max_kmh": wind_speed if weather_known else None,
            "temp_max_c": temp_max if weather_known else None,
            "precipitation_mm": precipitation if weather_known else None,
        }

        result = predict_single(input_dict, model=model, metadata=metadata)

        st.divider()
        _render_result(result)

        instance_df = pd.DataFrame([result["input_used"]])
        explanations = explain_instance(explainer, instance_df, top_k=6)
        render_shap_explanation(explanations)


def _render_result(result: dict):
    band_colors = {
        "Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴",
    }
    icon = band_colors.get(result["risk_band"], "⚪")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted success probability", f"{result['success_probability']:.1%}")
    c2.metric("Risk score (0-100)", f"{result['risk_score_0_100']}")
    c3.metric("Risk band", f"{icon} {result['risk_band']}")
