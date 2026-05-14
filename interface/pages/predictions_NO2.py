"""Page 5 — Prédictions NO2 : résultats du modèle hybride ARIMAX + LSTM."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle, joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_hourly, SEGMENTS_ORDER
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prédictions NO2", page_icon="🤖", layout="wide")
st.title("🤖 Prédictions NO2 — Modèle hybride ARIMAX + LSTM")

st.markdown("""
Cette page présente les **prédictions du modèle hybride** entraîné dans le notebook
`modelisation_hybride.ipynb`. Le modèle combine :
- **ARIMAX** : dynamique linéaire + effet des variables exogènes (météo, calendrier, validations)
- **LSTM sur résidus** : patterns non-linéaires que l'ARIMAX rate
""")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "models"

EXOG_FEATURES = [
    "HEURE_SIN",
    "HEURE_COS",
    "WEEKEND",
    "JOUR_FERIE",
    "VACANCES_SCOLAIRES",
    "JO",
    "JOUR_NON_OUVRE",
    "JOUR_PERTURBE",
    "TEMP_AVG_C",
    "WINDSPEED_MAX_KMH",
    "PRECIP_TOTAL_DAY_MM",
    "HUMIDITY_MAX_PERCENT",
    "PRESSURE_MAX_MB",
    "CLOUDCOVER_AVG_PERCENT",
    "VALD_NAVIGO",
]
LOOK_BACK = 24


@st.cache_data
def predict_hybrid(segment_name):
    """Recharge les modèles et renvoie les prédictions sur le test set nov-déc."""
    prefix = f"hybrid_NO2_{segment_name.replace('-', '_')}"
    arimax_path = MODELS_DIR / f"{prefix}_arimax.pkl"
    lstm_path = MODELS_DIR / f"{prefix}_lstm.keras"

    if not arimax_path.exists() or not lstm_path.exists():
        return None

    with open(arimax_path, "rb") as f:
        arimax = pickle.load(f)
    lstm = load_model(lstm_path, compile=False)
    scaler_exog = joblib.load(MODELS_DIR / f"{prefix}_scaler_exog.joblib")
    scaler_res = joblib.load(MODELS_DIR / f"{prefix}_scaler_res.joblib")

    df = load_hourly()
    seg = df[df["segment"] == segment_name].sort_values("time").reset_index(drop=True)
    train = seg[seg["time"] < "2024-09-01"].copy()
    test = seg[seg["time"] >= "2024-11-01"].copy()

    exog_te = pd.DataFrame(
        scaler_exog.transform(test[EXOG_FEATURES]),
        columns=EXOG_FEATURES,
        index=test.index,
    )
    pred_te_arimax = arimax.forecast(steps=len(test), exog=exog_te.values)

    pred_tr_arimax = arimax.fittedvalues
    res_te = test["NO2"].values - pred_te_arimax
    res_te_s = scaler_res.transform(res_te.reshape(-1, 1)).flatten()

    X = np.array(
        [res_te_s[i : i + LOOK_BACK] for i in range(len(res_te_s) - LOOK_BACK)]
    ).reshape(-1, LOOK_BACK, 1)
    pred_res_s = lstm.predict(X, verbose=0).flatten()
    pred_res = scaler_res.inverse_transform(pred_res_s.reshape(-1, 1)).flatten()

    return {
        "time": test["time"].values[LOOK_BACK:],
        "obs": test["NO2"].values[LOOK_BACK:],
        "arimax": pred_te_arimax[LOOK_BACK:],
        "hybrid": pred_te_arimax[LOOK_BACK:] + pred_res,
    }


def safe_metrics(y, yp):
    return {
        "RMSE": np.sqrt(mean_squared_error(y, yp)),
        "MAE": mean_absolute_error(y, yp),
        "R²": r2_score(y, yp),
    }


# === Vérification que les modèles existent ===
sample_path = (
    MODELS_DIR / f"hybrid_NO2_{SEGMENTS_ORDER[0].replace('-', '_')}_arimax.pkl"
)
if not sample_path.exists():
    st.error(
        f"Aucun modèle trouvé dans `{MODELS_DIR}`. "
        f"Exécutez d'abord le notebook `modelisation_hybride.ipynb`."
    )
    st.stop()

# === Sélection du segment ===
segment = st.selectbox("Segment", SEGMENTS_ORDER)

with st.spinner("Chargement et prédiction..."):
    preds = predict_hybrid(segment)

if preds is None:
    st.error(f"Modèle manquant pour {segment}. Exécutez `modelisation_hybride.ipynb`.")
    st.stop()

# === Métriques ===
st.subheader(f"Performance — {segment}")
col1, col2, col3 = st.columns(3)
m_hybrid = safe_metrics(preds["obs"], preds["hybrid"])
m_arimax = safe_metrics(preds["obs"], preds["arimax"])

col1.metric(
    "RMSE hybride",
    f"{m_hybrid['RMSE']:.2f} µg/m³",
    f"{m_hybrid['RMSE'] - m_arimax['RMSE']:+.2f} vs ARIMAX seul",
    delta_color="inverse",
)
col2.metric(
    "MAE hybride",
    f"{m_hybrid['MAE']:.2f} µg/m³",
    f"{m_hybrid['MAE'] - m_arimax['MAE']:+.2f} vs ARIMAX seul",
    delta_color="inverse",
)
col3.metric(
    "R² hybride",
    f"{m_hybrid['R²']:.3f}",
    f"{m_hybrid['R²'] - m_arimax['R²']:+.3f} vs ARIMAX seul",
)

# === Visualisation ===
st.subheader("Prédictions sur le test set (novembre-décembre 2024)")

dates = pd.to_datetime(preds["time"])
choix_periode = st.radio(
    "Fenêtre",
    options=["Première semaine", "Deux semaines", "Tout le test (2 mois)"],
    horizontal=True,
)
limites = {"Première semaine": 7, "Deux semaines": 14, "Tout le test (2 mois)": None}
limite_jours = limites[choix_periode]

if limite_jours:
    cut = dates[0] + pd.Timedelta(days=limite_jours)
    mask = dates < cut
else:
    mask = np.ones(len(dates), dtype=bool)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates[mask],
        y=preds["obs"][mask],
        name="Observé",
        line=dict(color="black", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates[mask],
        y=preds["arimax"][mask],
        name="ARIMAX seul",
        line=dict(color="#1f77b4", width=1.2, dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates[mask],
        y=preds["hybrid"][mask],
        name="Hybride ARIMAX+LSTM",
        line=dict(color="#d62728", width=1.5),
    )
)
fig.update_layout(
    yaxis_title="NO2 (µg/m³)",
    height=500,
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# === Scatter prédit vs observé ===
st.subheader("Précision du modèle hybride")
fig_sc = go.Figure()
fig_sc.add_trace(
    go.Scatter(
        x=preds["obs"],
        y=preds["hybrid"],
        mode="markers",
        marker=dict(size=4, opacity=0.4, color="#d62728"),
        name="Prédictions",
    )
)
lim = max(preds["obs"].max(), preds["hybrid"].max())
fig_sc.add_trace(
    go.Scatter(
        x=[0, lim],
        y=[0, lim],
        mode="lines",
        line=dict(dash="dash", color="black"),
        name="Prédiction parfaite",
    )
)
fig_sc.update_layout(
    xaxis_title="NO2 observé (µg/m³)",
    yaxis_title="NO2 prédit (µg/m³)",
    height=500,
    yaxis=dict(scaleanchor="x"),
)
st.plotly_chart(fig_sc, use_container_width=True)
