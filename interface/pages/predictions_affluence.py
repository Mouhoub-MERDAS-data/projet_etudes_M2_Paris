"""Page 6 — Prédictions affluence Navigo : modèle SARIMAX + LSTM journalier."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle, joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_daily
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prédictions affluence", page_icon="🚆", layout="wide")
st.title("🚆 Prédictions affluence — Modèle SARIMAX + LSTM")

st.markdown("""
Prédiction du **volume journalier de validations Navigo** sur le réseau ferré francilien.
Architecture similaire au modèle NO2, adaptée au journalier :
- **SARIMAX(1,0,1)(1,1,1,7)** : capture la saisonnalité hebdomadaire (cycle Lun-Dim)
- **LSTM sur résidus** : look-back 7 jours
""")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "models"
PREFIX = "navigo_daily"

EXOG_FEATURES = [
    "JOUR_FERIE",
    "VACANCES_SCOLAIRES",
    "JO",
    "JOP",
    "JOUR_NON_OUVRE",
    "JOUR_PERTURBE",
    "TEMP_AVG_C",
    "WINDSPEED_MAX_KMH",
    "PRECIP_TOTAL_DAY_MM",
    "HUMIDITY_MAX_PERCENT",
    "PRESSURE_MAX_MB",
    "CLOUDCOVER_AVG_PERCENT",
]
LOOK_BACK = 7


@st.cache_data
def predict_navigo():
    sarimax_path = MODELS_DIR / f"{PREFIX}_sarimax.pkl"
    lstm_path = MODELS_DIR / f"{PREFIX}_lstm.keras"

    if not sarimax_path.exists() or not lstm_path.exists():
        return None

    with open(sarimax_path, "rb") as f:
        sarimax = pickle.load(f)
    lstm = load_model(lstm_path, compile=False)
    scaler_exog = joblib.load(MODELS_DIR / f"{PREFIX}_scaler_exog.joblib")
    scaler_res = joblib.load(MODELS_DIR / f"{PREFIX}_scaler_res.joblib")

    df = load_daily().sort_values("DATE").reset_index(drop=True)
    train = df[df["DATE"] < "2024-09-01"]
    test = df[df["DATE"] >= "2024-11-01"]

    ex_te = pd.DataFrame(
        scaler_exog.transform(test[EXOG_FEATURES]),
        columns=EXOG_FEATURES,
        index=test.index,
    )
    pred_te_sarimax = sarimax.forecast(steps=len(test), exog=ex_te.values)

    res_te = test["VALD_NAVIGO"].values - pred_te_sarimax
    res_te_s = scaler_res.transform(res_te.reshape(-1, 1)).flatten()

    X = np.array(
        [res_te_s[i : i + LOOK_BACK] for i in range(len(res_te_s) - LOOK_BACK)]
    ).reshape(-1, LOOK_BACK, 1)
    pred_res_s = lstm.predict(X, verbose=0).flatten()
    pred_res = scaler_res.inverse_transform(pred_res_s.reshape(-1, 1)).flatten()

    # Baseline : moyenne par jour de la semaine sur le train
    bow = train.groupby(train["DATE"].dt.dayofweek)["VALD_NAVIGO"].mean()
    base = test["DATE"].dt.dayofweek.map(bow).values

    return {
        "dates": test["DATE"].values[LOOK_BACK:],
        "obs": test["VALD_NAVIGO"].values[LOOK_BACK:],
        "sarimax": pred_te_sarimax[LOOK_BACK:],
        "hybrid": pred_te_sarimax[LOOK_BACK:] + pred_res,
        "base": base[LOOK_BACK:],
    }


if not (MODELS_DIR / f"{PREFIX}_sarimax.pkl").exists():
    st.error(
        f"Modèle Navigo non trouvé dans `{MODELS_DIR}`. "
        f"Exécutez d'abord `prediction_affluence_navigo.ipynb`."
    )
    st.stop()

with st.spinner("Chargement des prédictions..."):
    preds = predict_navigo()


# === Métriques ===
def met(y, yp):
    return (
        np.sqrt(mean_squared_error(y, yp)) / 1e6,
        mean_absolute_error(y, yp) / 1e6,
        r2_score(y, yp),
        np.mean(np.abs((y - yp) / y)) * 100,
    )


rmse_h, mae_h, r2_h, mape_h = met(preds["obs"], preds["hybrid"])
rmse_s, mae_s, r2_s, mape_s = met(preds["obs"], preds["sarimax"])
rmse_b, mae_b, r2_b, mape_b = met(preds["obs"], preds["base"])

st.subheader("Performance comparative")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "RMSE hybride",
    f"{rmse_h:.2f} M",
    f"vs baseline : {rmse_h-rmse_b:+.2f} M",
    delta_color="inverse",
)
col2.metric(
    "MAE hybride",
    f"{mae_h:.2f} M",
    f"vs baseline : {mae_h-mae_b:+.2f} M",
    delta_color="inverse",
)
col3.metric("R² hybride", f"{r2_h:.3f}", f"vs baseline : {r2_h-r2_b:+.3f}")
col4.metric(
    "MAPE hybride",
    f"{mape_h:.1f} %",
    f"vs baseline : {mape_h-mape_b:+.1f} %",
    delta_color="inverse",
)

# Tableau comparatif
tab = pd.DataFrame(
    {
        "Modèle": [
            "Baseline (moyenne jour-semaine)",
            "SARIMAX",
            "Hybride SARIMAX + LSTM",
        ],
        "RMSE (M)": [rmse_b, rmse_s, rmse_h],
        "MAE (M)": [mae_b, mae_s, mae_h],
        "R²": [r2_b, r2_s, r2_h],
        "MAPE (%)": [mape_b, mape_s, mape_h],
    }
).round(3)
st.dataframe(tab, hide_index=True, use_container_width=True)

# === Graphique principal ===
st.subheader("Prédictions sur le test set (nov-déc 2024)")

dates = pd.to_datetime(preds["dates"])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates, y=preds["obs"] / 1e6, name="Observé", line=dict(color="black", width=2)
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=preds["base"] / 1e6,
        name="Baseline",
        line=dict(color="gray", width=1, dash="dot"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=preds["sarimax"] / 1e6,
        name="SARIMAX seul",
        line=dict(color="#1f77b4", width=1.2, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=preds["hybrid"] / 1e6,
        name="Hybride SARIMAX+LSTM",
        line=dict(color="#d62728", width=1.5),
    )
)
fig.update_layout(
    yaxis_title="Validations Navigo (millions)",
    xaxis_title="",
    height=500,
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# === Distribution des erreurs ===
st.subheader("Distribution des erreurs résiduelles (modèle hybride)")
errors = (preds["obs"] - preds["hybrid"]) / 1e6
fig_err = go.Figure()
fig_err.add_trace(
    go.Histogram(
        x=errors,
        nbinsx=20,
        marker_color="#d62728",
        marker_line=dict(color="black", width=1),
    )
)
fig_err.add_vline(
    x=0, line_dash="dash", line_color="green", annotation_text="Erreur nulle"
)
fig_err.update_layout(
    xaxis_title="Erreur de prédiction (millions de validations)",
    yaxis_title="Nombre de jours",
    height=350,
)
st.plotly_chart(fig_err, use_container_width=True)
