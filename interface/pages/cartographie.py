"""Page — Cartographie du périphérique parisien."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    SEGMENTS_COORDS, SEGMENTS_ORDER, SEUIL_NO2_UE,
    color_for_no2, categorie_no2, load_hourly,
)

st.set_page_config(page_title="Cartographie", page_icon="🗺️", layout="wide")
st.title("🗺️ Cartographie — Pollution NO₂ sur le périphérique")
st.markdown("Visualisation de la concentration moyenne de NO₂ par segment du périphérique parisien — données Airparif 2024.")

df = load_hourly()

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    mois_options = {"Toute l\'année": None,"Janvier":1,"Février":2,"Mars":3,"Avril":4,"Mai":5,"Juin":6,"Juillet":7,"Août":8,"Septembre":9,"Octobre":10,"Novembre":11,"Décembre":12}
    mois_sel = st.selectbox("Mois", list(mois_options.keys()))
with col_f2:
    periode = st.radio("Période", ["24h","Heures de pointe (7-9h, 17-19h)","Nuit (22h-6h)"], horizontal=True)
with col_f3:
    polluant = st.selectbox("Polluant", ["NO2","PM10","PM25"])

df_f = df.copy()
if mois_options[mois_sel]:
    df_f = df_f[df_f["time"].dt.month == mois_options[mois_sel]]
if periode == "Heures de pointe (7-9h, 17-19h)":
    df_f = df_f[df_f["HEURE"].isin([7,8,9,17,18,19])]
elif periode == "Nuit (22h-6h)":
    df_f = df_f[df_f["HEURE"].isin([22,23,0,1,2,3,4,5,6])]

seg_means = df_f.groupby("segment")[polluant].mean().round(2)

st.subheader(f"Carte des segments — {polluant} moyen")

fig = go.Figure()
theta = np.linspace(0, 2*np.pi, 300)
fig.add_trace(go.Scatter(x=0.5+0.42*np.cos(theta), y=0.5+0.42*np.sin(theta),
    mode="lines", line=dict(color="#CCCCCC",width=3), showlegend=False, hoverinfo="skip"))

for seg in SEGMENTS_ORDER:
    x, y = SEGMENTS_COORDS[seg]
    val = seg_means.get(seg, 0)
    color = color_for_no2(val)
    categ = categorie_no2(val)
    fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
        marker=dict(size=38,color=color,line=dict(color="white",width=2)),
        text=[f"{val:.0f}"], textposition="middle center",
        textfont=dict(color="white",size=11,family="Arial Black"),
        name=seg,
        hovertemplate=f"<b>{seg}</b><br>{polluant}: {val:.2f} µg/m³<br>Qualité: {categ}<br>Seuil UE 40: {'⚠️ DÉPASSÉ' if val>40 else '✅ OK'}<extra></extra>",
        showlegend=False))
    ox = 0.10 if x>0.5 else -0.10
    oy = 0.07 if y>0.5 else -0.07
    fig.add_annotation(x=x+ox,y=y+oy,text=seg,showarrow=False,
        font=dict(size=10,color="#333"),bgcolor="rgba(255,255,255,0.8)",borderpad=2)

for label,color in [("< 20 µg/m³ — Bon","#2ca02c"),("20-40 µg/m³ — Modéré","#ffd700"),
                     ("40-60 µg/m³ — Dégradé","#ff7f0e"),("> 60 µg/m³ — Mauvais","#d62728")]:
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=12,color=color),name=label))

fig.add_annotation(x=0.5,y=0.5,text="<b>Paris</b>",showarrow=False,font=dict(size=14,color="#666"))
fig.update_layout(height=600,xaxis=dict(range=[0,1],visible=False),
    yaxis=dict(range=[0,1],visible=False,scaleanchor="x"),
    legend=dict(orientation="h",y=-0.08),margin=dict(l=10,r=10,t=20,b=10),
    plot_bgcolor="rgba(240,248,255,0.5)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Tableau — Concentration par segment")
rows = []
for seg in SEGMENTS_ORDER:
    val = seg_means.get(seg, float("nan"))
    rows.append({"Segment":seg, f"{polluant} moyen (µg/m³)":round(val,2),
                 "Qualité":categorie_no2(val), "Seuil UE 40":"⚠️ Dépassé" if val>40 else "✅ OK"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Comparaison des segments")
vals_bar = [seg_means.get(s,0) for s in SEGMENTS_ORDER]
fig_bar = go.Figure(go.Bar(x=SEGMENTS_ORDER, y=vals_bar,
    marker_color=[color_for_no2(v) for v in vals_bar],
    text=[f"{v:.1f}" for v in vals_bar], textposition="outside"))
fig_bar.add_hline(y=40, line_dash="dash", line_color="red",
    annotation_text="Seuil UE 40 µg/m³", annotation_position="top right")
fig_bar.update_layout(yaxis_title=f"{polluant} moyen (µg/m³)", height=400, showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)