"""
Renders per-mission SHAP explanations as a readable, color-coded list —
built for an investigator/mission-planner audience, not a data scientist.
Satisfies Objective #5's "human-readable explanation" requirement.
"""

import streamlit as st


def render_shap_explanation(explanations: list):
    st.markdown("**What's driving this prediction:**")

    for e in explanations:
        if e["direction"] == "increases":
            color = "green"
            arrow = "⬆️"
            verb = "increasing"
        else:
            color = "red"
            arrow = "⬇️"
            verb = "decreasing"

        st.markdown(
            f"{arrow} :{color}[**{e['label']}**] = `{e['value']}` "
            f"— {verb} predicted success probability "
            f"(impact: {abs(e['shap_value']):.3f})"
        )
