"""
Historical analytics view: success rates sliced by vehicle, site, decade,
and payload class. Satisfies Objective #6.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render(features: pd.DataFrame):
    st.subheader("Historical Success Rates")

    col1, col2 = st.columns(2)

    with col1:
        _plot_by_decade(features)

    with col2:
        _plot_by_country(features)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        _plot_by_rocket_family(features)

    with col4:
        _plot_by_payload_class(features)

    st.divider()
    _summary_table(features)


def _plot_by_decade(df: pd.DataFrame):
    st.markdown("**By Decade**")
    agg = df.groupby("decade")["mission_success"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= 5]  # avoid noisy tiny-sample decades

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(agg["decade"].astype(str), agg["mean"], color="#2563eb")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_by_country(df: pd.DataFrame):
    st.markdown("**By Country / Agency**")
    agg = df.groupby("country_grouped")["mission_success"].agg(["mean", "count"])
    agg = agg[agg["count"] >= 10].sort_values("count", ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(agg.index[::-1], agg["mean"][::-1], color="#059669")
    ax.set_xlabel("Success rate")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_by_rocket_family(df: pd.DataFrame):
    st.markdown("**By Rocket Family (top 10 by volume)**")
    agg = df.groupby("rocket_family_grouped")["mission_success"].agg(["mean", "count"])
    agg = agg[agg.index != "other"].sort_values("count", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(agg.index[::-1], agg["mean"][::-1], color="#dc2626")
    ax.set_xlabel("Success rate")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_by_payload_class(df: pd.DataFrame):
    st.markdown("**By Payload Capacity Class**")
    bins = [0, 3000, 10000, 25000, float("inf")]
    labels = ["Small (<3t)", "Medium (3-10t)", "Large (10-25t)", "Heavy (25t+)"]
    df = df.copy()
    df["payload_class"] = pd.cut(df["payload_capacity_kg"], bins=bins, labels=labels)

    agg = df.groupby("payload_class", observed=True)["mission_success"].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(agg.index.astype(str), agg["mean"], color="#7c3aed")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _summary_table(df: pd.DataFrame):
    st.markdown("**Dataset Summary**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total launches", f"{len(df):,}")
    c2.metric("Overall success rate", f"{df['mission_success'].mean():.1%}")
    c3.metric("Rocket families tracked", df["rocket_family_grouped"].nunique())
    c4.metric("Launches with real weather data", f"{df['weather_available'].sum():,}")
