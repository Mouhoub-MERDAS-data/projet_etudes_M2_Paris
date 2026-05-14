"""Page 4 — Affluence transport : validations FER journalières."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_daily

st.set_page_config(page_title="Affluence transport", page_icon="🚇", layout="wide")
st.title("🚇 Affluence du transport ferré francilien")

df = load_daily()

# === KPIs catégoriels ===
st.subheader("Répartition annuelle par type de titre")

categories = [
    "VALD_NAVIGO",
    "VALD_IMAGINE_R",
    "VALD_SOLIDARITE",
    "VALD_AUTRES",
    "VALD_AMETHYSTE",
    "VALD_COURTS",
    "VALD_NON_DEFINI",
]
labels = [
    "Navigo",
    "Imagine R",
    "Solidarité",
    "Autres",
    "Améthyste",
    "Courts",
    "Non défini",
]

totals = [df[c].sum() / 1e6 for c in categories]

cols = st.columns(4)
cols[0].metric("Total validations 2024", f"{sum(totals):.0f} M")
cols[1].metric("Navigo (pendulaires)", f"{df['VALD_NAVIGO'].sum()/1e6:.0f} M")
cols[2].metric("Imagine R (étudiants)", f"{df['VALD_IMAGINE_R'].sum()/1e6:.0f} M")
cols[3].metric("Améthyste (seniors)", f"{df['VALD_AMETHYSTE'].sum()/1e6:.1f} M")

# === Donut chart ===
fig_donut = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=totals,
            hole=0.4,
            textinfo="label+percent",
        )
    ]
)
fig_donut.update_layout(height=400, showlegend=False)
st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# === Évolution temporelle ===
st.subheader("Évolution journalière des validations")

choix_cat = st.multiselect(
    "Catégories à afficher",
    options=labels,
    default=["Navigo", "Imagine R"],
)
label_to_col = dict(zip(labels, categories))

fig = go.Figure()
for lab in choix_cat:
    col = label_to_col[lab]
    fig.add_trace(
        go.Scatter(
            x=df["DATE"],
            y=df[col] / 1e6,
            name=lab,
            mode="lines",
            line=dict(width=1.2),
        )
    )

# Marqueurs sur les jours de JO
jo_dates = df[df["JO"] == 1]["DATE"]
if len(jo_dates) > 0:
    fig.add_vrect(
        x0=jo_dates.min(),
        x1=jo_dates.max(),
        fillcolor="gold",
        opacity=0.15,
        annotation_text="JO 2024",
        annotation_position="top left",
        line_width=0,
    )

fig.update_layout(
    yaxis_title="Validations (millions)",
    xaxis_title="",
    height=450,
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# === Profil hebdomadaire ===
st.subheader("Profil hebdomadaire moyen — Navigo")

jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
df["JOUR_NUM"] = df["DATE"].dt.dayofweek

# Distinguer fériés et non-fériés
prof = df[df["JOUR_FERIE"] == 0].groupby("JOUR_NUM")["VALD_NAVIGO"].mean()
prof_ferie = df[df["JOUR_FERIE"] == 1].groupby("JOUR_NUM")["VALD_NAVIGO"].mean()

fig_w = go.Figure()
fig_w.add_trace(
    go.Bar(
        x=jours,
        y=[prof.get(i, 0) / 1e6 for i in range(7)],
        name="Jour ordinaire",
        marker_color="#1f77b4",
    )
)
fig_w.add_trace(
    go.Bar(
        x=jours,
        y=[prof_ferie.get(i, 0) / 1e6 for i in range(7)],
        name="Jour férié",
        marker_color="#d62728",
    )
)
fig_w.update_layout(
    yaxis_title="Validations Navigo (millions)",
    height=400,
    barmode="group",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_w, use_container_width=True)

st.caption(
    "L'écart entre semaine (≈3 M) et weekend (≈1,7 M) reflète l'usage **pendulaire** "
    "du Navigo. Les jours fériés en semaine font chuter l'affluence au niveau du weekend."
)
