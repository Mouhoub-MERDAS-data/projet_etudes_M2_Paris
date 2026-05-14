"""Page — Vue d\'ensemble de la pollution 2024."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import SEGMENTS_ORDER, SEUIL_NO2_UE, color_for_no2, categorie_no2, load_hourly

st.set_page_config(page_title="Vue d\'ensemble", page_icon="📊", layout="wide")
st.title("📊 Vue d\'ensemble — Pollution 2024")
st.markdown("Synthèse de la qualité de l\'air sur le boulevard périphérique parisien (Airparif 2024).")

df = load_hourly()

# KPIs globaux
no2_mean = df["NO2"].mean()
no2_max  = df["NO2"].max()
pct_dep  = (df["NO2"] > SEUIL_NO2_UE).mean() * 100
n_seg    = df["segment"].nunique()

c1,c2,c3,c4 = st.columns(4)
c1.metric("NO₂ moyen 2024",    f"{no2_mean:.1f} µg/m³")
c2.metric("NO₂ max 2024",      f"{no2_max:.1f} µg/m³")
c3.metric("Heures > seuil UE", f"{pct_dep:.1f} %")
c4.metric("Segments couverts", f"{n_seg} / 8")

st.divider()

# Profil horaire moyen
st.subheader("Profil horaire moyen — NO₂ par segment")
hourly = df.groupby(["HEURE","segment"])["NO2"].mean().reset_index()
fig1 = px.line(hourly, x="HEURE", y="NO2", color="segment",
    labels={"NO2":"NO₂ (µg/m³)","HEURE":"Heure","segment":"Segment"}, height=400)
fig1.add_hline(y=SEUIL_NO2_UE, line_dash="dash", line_color="red",
    annotation_text="Seuil UE 40 µg/m³")
fig1.update_layout(legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig1, use_container_width=True)

col_a, col_b = st.columns(2)

# Pattern hebdomadaire
with col_a:
    st.subheader("Pattern hebdomadaire")
    dow_map = {0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"}
    df["DOW_NAME"] = df["JOUR_SEMAINE"].map(dow_map) if "JOUR_SEMAINE" in df.columns else df["time"].dt.dayofweek.map(dow_map)
    dow_order = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    weekly = df.groupby("DOW_NAME")["NO2"].mean().reindex(dow_order)
    fig2 = go.Figure(go.Bar(x=weekly.index, y=weekly.values,
        marker_color=[color_for_no2(v) for v in weekly.values],
        text=[f"{v:.1f}" for v in weekly.values], textposition="outside"))
    fig2.add_hline(y=SEUIL_NO2_UE, line_dash="dash", line_color="red")
    fig2.update_layout(yaxis_title="NO₂ (µg/m³)", height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# Distribution par polluant
with col_b:
    st.subheader("Distribution des polluants")
    poll_sel = st.selectbox("Polluant", ["NO2","PM10","PM25"], key="dist_poll")
    fig3 = px.histogram(df, x=poll_sel, nbins=60, color_discrete_sequence=["#1f77b4"],
        labels={poll_sel:f"{poll_sel} (µg/m³)"}, height=350)
    seuil_map = {"NO2":40,"PM10":40,"PM25":25}
    fig3.add_vline(x=seuil_map[poll_sel], line_dash="dash", line_color="red",
        annotation_text=f"Seuil {seuil_map[poll_sel]} µg/m³")
    st.plotly_chart(fig3, use_container_width=True)

# Heatmap heure × mois
st.subheader("Heatmap NO₂ — Heure × Mois")
seg_h = st.selectbox("Segment", SEGMENTS_ORDER, key="heat_seg")
df_seg = df[df["segment"]==seg_h].copy()
df_seg["MOIS_N"] = df_seg["time"].dt.month
heat = df_seg.groupby(["HEURE","MOIS_N"])["NO2"].mean().unstack()
mois_labels = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
               7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
heat.columns = [mois_labels.get(c,c) for c in heat.columns]
fig4 = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
    labels={"x":"Mois","y":"Heure","color":"NO₂ µg/m³"},
    title=f"Segment {seg_h}", height=400)
st.plotly_chart(fig4, use_container_width=True)

# Trend mensuelle
st.subheader("Évolution mensuelle du NO₂")
df["MOIS_N"] = df["time"].dt.month
monthly = df.groupby(["MOIS_N","segment"])["NO2"].mean().reset_index()
monthly["Mois"] = monthly["MOIS_N"].map(mois_labels)
fig5 = px.line(monthly, x="MOIS_N", y="NO2", color="segment",
    labels={"NO2":"NO₂ (µg/m³)","MOIS_N":"Mois"}, height=400)
fig5.update_xaxes(tickvals=list(range(1,13)),
    ticktext=[mois_labels[i] for i in range(1,13)])
fig5.add_hline(y=SEUIL_NO2_UE, line_dash="dash", line_color="red")
fig5.update_layout(legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig5, use_container_width=True)