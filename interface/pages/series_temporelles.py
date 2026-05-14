"""Page 3 — Séries temporelles : exploration interactive heure par heure."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_hourly, SEGMENTS_ORDER, SEUIL_NO2_UE

st.set_page_config(page_title="Séries temporelles", page_icon="📈", layout="wide")
st.title("📈 Séries temporelles — Pollution")

df = load_hourly()

# === Filtres ===
col1, col2, col3 = st.columns(3)

segments = col1.multiselect(
    "Segments à afficher",
    options=SEGMENTS_ORDER,
    default=["Chap-Bagn", "Berc-Ital"],
)

polluants = col2.multiselect(
    "Polluants",
    options=["NO2", "PM10", "PM25"],
    default=["NO2"],
)

granularite = col3.radio(
    "Granularité",
    options=["Horaire", "Journalière", "Hebdomadaire"],
    horizontal=True,
)

# Période
date_min = df["DATE"].min().date()
date_max = df["DATE"].max().date()
periode = st.date_input(
    "Période",
    value=(pd.Timestamp("2024-03-01").date(), pd.Timestamp("2024-03-31").date()),
    min_value=date_min,
    max_value=date_max,
)

# Filtre
if not segments or not polluants:
    st.warning("Sélectionnez au moins un segment et un polluant.")
    st.stop()

mask = df["segment"].isin(segments)
if isinstance(periode, tuple) and len(periode) == 2:
    d1, d2 = periode
    mask &= (df["DATE"] >= pd.Timestamp(d1)) & (df["DATE"] <= pd.Timestamp(d2))
df_filt = df[mask].copy()

# Agrégation selon la granularité
if granularite == "Horaire":
    df_plot = df_filt
    x_col = "time"
elif granularite == "Journalière":
    df_plot = df_filt.groupby(["DATE", "segment"])[polluants].mean().reset_index()
    x_col = "DATE"
else:  # Hebdomadaire
    df_filt["semaine"] = df_filt["time"].dt.to_period("W").dt.start_time
    df_plot = df_filt.groupby(["semaine", "segment"])[polluants].mean().reset_index()
    x_col = "semaine"

# === Graphique ===
st.subheader(f"Évolution {granularite.lower()}")

fig = go.Figure()
for seg in segments:
    sub = df_plot[df_plot["segment"] == seg].sort_values(x_col)
    for pol in polluants:
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[pol],
                name=f"{seg} — {pol}",
                mode="lines",
                line=dict(width=1.5),
            )
        )

if "NO2" in polluants:
    fig.add_hline(
        y=SEUIL_NO2_UE,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Limite UE NO2 ({SEUIL_NO2_UE} µg/m³)",
    )

fig.update_layout(
    yaxis_title="Concentration (µg/m³)",
    xaxis_title="",
    height=500,
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# === Statistiques rapides ===
st.subheader("Statistiques sur la période sélectionnée")
stats = df_filt.groupby("segment")[polluants].agg(["mean", "max", "min"]).round(1)
st.dataframe(stats, use_container_width=True)
