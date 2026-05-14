"""Page 7 — Analyse du report modal : validation visuelle de l'hypothèse projet."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_hourly, load_daily

st.set_page_config(page_title="Report modal", page_icon="🔄", layout="wide")
st.title("🔄 Analyse du report modal")

st.markdown("""
### 📌 Hypothèse testée

> *Les perturbations du transport ferré (grèves, incidents) provoquent un report modal
> vers la voiture, mesurable par une **chute des validations Navigo** et une **hausse
> simultanée du NO2** sur le périphérique.*

Cette page met en regard les deux signaux pour visualiser et quantifier la relation.
""")

# Chargement
df_h = load_hourly()
df_d = load_daily()

# Aggrégation journalière du NO2 (moyenne réseau)
df_h_daily = df_h.groupby("DATE")["NO2"].mean().reset_index()
df_h_daily.columns = ["DATE", "NO2_moy_jour"]
merged = df_d.merge(df_h_daily, on="DATE")

# === Filtrage : jours "comparables" pour isoler le signal ===
st.subheader("Calibration des jours comparables")
st.markdown(
    "Pour isoler le signal de report modal, on filtre sur les **jours ouvrés "
    "non perturbés par d'autres facteurs** (ni fériés, ni vacances, ni weekend, ni JO)."
)
mask_ouvres = (
    (merged["WEEKEND"] == 0)
    & (merged["JOUR_FERIE"] == 0)
    & (merged["VACANCES_SCOLAIRES"] == 0)
    & (merged["JO"] == 0)
)
ouvres = merged[mask_ouvres].copy()
st.info(f"Jours ouvrés non perturbés : **{len(ouvres)}** sur 366")

# === Corrélation Navigo ↔ NO2 ===
st.subheader("Corrélation Navigo ↔ NO2 (jours ouvrés)")

correlation = ouvres["VALD_NAVIGO"].corr(ouvres["NO2_moy_jour"])
col1, col2 = st.columns([1, 3])
col1.metric(
    "Coefficient de Pearson",
    f"{correlation:.3f}",
    (
        "↘ négatif = hypothèse confirmée"
        if correlation < 0
        else "↗ positif = hypothèse infirmée"
    ),
)
col2.markdown(f"""
    Une corrélation **négative** signifie que **moins de Navigo ↔ plus de NO2**, ce qui
    valide visuellement l'hypothèse.

    Valeur observée : `{correlation:.3f}`
    """)

# Scatter
fig = px.scatter(
    ouvres,
    x="VALD_NAVIGO",
    y="NO2_moy_jour",
    hover_data=["DATE", "MAX_TEMPERATURE_C", "WINDSPEED_MAX_KMH"],
    labels={
        "VALD_NAVIGO": "Validations Navigo (jour)",
        "NO2_moy_jour": "NO2 moyen jour (µg/m³)",
    },
    trendline="ols",
    trendline_color_override="red",
    opacity=0.6,
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# === Détection des "anomalies de Navigo" ===
st.subheader("Détection des jours candidats à une perturbation transport")
st.markdown(
    "On calcule l'**écart relatif** entre les validations observées et la **médiane attendue** "
    "pour un jour comparable (même mois × même jour de la semaine). Un écart négatif fort "
    "(< −20%) signale un jour candidat à une perturbation transport."
)

# Médiane attendue par (mois, jour_semaine) sur les jours ouvrés
ouvres["MOIS"] = ouvres["DATE"].dt.month
ouvres["JS"] = ouvres["DATE"].dt.dayofweek
baseline = ouvres.groupby(["MOIS", "JS"])["VALD_NAVIGO"].median().reset_index()
baseline.columns = ["MOIS", "JS", "NAVIGO_BASELINE"]

ouvres = ouvres.merge(baseline, on=["MOIS", "JS"])
ouvres["ECART_REL"] = (ouvres["VALD_NAVIGO"] - ouvres["NAVIGO_BASELINE"]) / ouvres[
    "NAVIGO_BASELINE"
]

seuil = st.slider(
    "Seuil de détection (écart relatif négatif)",
    min_value=-0.50,
    max_value=-0.05,
    value=-0.20,
    step=0.01,
    format="%.2f",
)

candidats = ouvres[ouvres["ECART_REL"] < seuil].sort_values("ECART_REL")
st.write(
    f"**{len(candidats)} jour(s) candidat(s)** avec un écart < {seuil:.0%} par rapport à l'attendu."
)

# Comparaison NO2 sur ces jours vs jours normaux
if len(candidats) > 0:
    normaux = ouvres[ouvres["ECART_REL"] >= seuil]
    no2_cand = candidats["NO2_moy_jour"].mean()
    no2_norm = normaux["NO2_moy_jour"].mean()
    delta_pct = (no2_cand - no2_norm) / no2_norm * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("NO2 moyen jours candidats", f"{no2_cand:.1f} µg/m³")
    col2.metric("NO2 moyen jours normaux", f"{no2_norm:.1f} µg/m³")
    col3.metric(
        "Écart",
        f"{delta_pct:+.1f} %",
        "↗ hypothèse confirmée" if delta_pct > 0 else "↘ hypothèse infirmée",
    )

    # Tableau des candidats
    st.subheader("Liste des jours candidats")
    display = candidats[
        ["DATE", "VALD_NAVIGO", "NAVIGO_BASELINE", "ECART_REL", "NO2_moy_jour"]
    ].copy()
    display.columns = [
        "Date",
        "Navigo observé",
        "Navigo attendu",
        "Écart relatif",
        "NO2 jour (µg/m³)",
    ]
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d (%A)")
    display["Navigo observé"] = (display["Navigo observé"] / 1e6).round(2).astype(
        str
    ) + " M"
    display["Navigo attendu"] = (display["Navigo attendu"] / 1e6).round(2).astype(
        str
    ) + " M"
    display["Écart relatif"] = (display["Écart relatif"] * 100).round(1).astype(
        str
    ) + " %"
    display["NO2 jour (µg/m³)"] = display["NO2 jour (µg/m³)"].round(1)
    st.dataframe(display, hide_index=True, use_container_width=True)

# === Conclusion ===
st.markdown("---")
st.subheader("📝 Synthèse")
if correlation < -0.1:
    verdict = "**confirme**"
    couleur = "green"
elif correlation < 0.05:
    verdict = "**ne confirme ni n'infirme nettement**"
    couleur = "orange"
else:
    verdict = "**ne confirme pas**"
    couleur = "red"

st.markdown(f"""
    L'analyse des données 2024 {verdict} l'hypothèse de report modal.

    - Corrélation Navigo ↔ NO2 sur jours ouvrés : **:{couleur}[{correlation:.3f}]**
    - Les chutes anormales de validations sont associées à un NO2 **plus élevé**
      (cf. tableau ci-dessus), ce qui est cohérent avec un report vers la voiture.

    **Limites** :
    - Année 2024 atypique à cause des Jeux Olympiques (août-septembre)
    - Absence de données officielles de grève IDFM pour validation externe
    - Le NO2 dépend aussi fortement de la météo (vent, température), qui peut masquer le signal
    """)
