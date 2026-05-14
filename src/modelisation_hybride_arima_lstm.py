"""
Projet : Optimisation de la mobilité urbaine - M2 Big Data & IA
Modèle : Hybride ARIMAX + LSTM multi-segments (Couverture totale Paris)
"""

import os
import pandas as pd
import numpy as np
import pickle
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings("ignore")

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
]


def load_data_segment(filepath, segment_cible):
    df = pd.read_csv(filepath)
    df = df[df["segment"] == segment_cible]
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def run_arimax(train_target, test_target, train_exog, test_exog, order=(2, 1, 1)):
    model = ARIMA(train_target, exog=train_exog, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_target), exog=test_exog)
    residuals = test_target.copy()
    residuals.iloc[:] = test_target.values - predictions.values
    return pd.Series(predictions.values, index=test_target.index), residuals, model_fit


def run_lstm_on_residuals(residuals, look_back=12):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    X, Y = [], []
    for i in range(len(res_scaled) - look_back):
        X.append(res_scaled[i : (i + look_back), 0])
        Y.append(res_scaled[i + look_back, 0])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X, Y, epochs=5, batch_size=32, verbose=0
    )  # Epochs réduits pour rapidité multi-segments

    pred_res_scaled = model.predict(X, verbose=0)
    pred_res = scaler.inverse_transform(pred_res_scaled)
    return pred_res, model, scaler


def run_pipeline(df, target_col, segment_name, exog_cols, look_back=12):
    print(f"   🔎 Cible : {target_col}")
    MODELS_DIR = "src/models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    current_exog = exog_cols.copy()
    if target_col == "NO2" and "VALD_NAVIGO" in df.columns:
        current_exog.append("VALD_NAVIGO")

    data = df[[target_col] + [c for c in current_exog if c in df.columns]].dropna()
    split_idx = int(len(data) * 0.8)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    scaler_exog = StandardScaler()
    actual_exog = [c for c in current_exog if c in data.columns]
    train_exog = pd.DataFrame(
        scaler_exog.fit_transform(train[actual_exog]),
        index=train.index,
        columns=actual_exog,
    )
    test_exog = pd.DataFrame(
        scaler_exog.transform(test[actual_exog]), index=test.index, columns=actual_exog
    )

    preds_arima, test_res, model_arima = run_arimax(
        train[target_col], test[target_col], train_exog, test_exog
    )
    pred_res_lstm, model_lstm, scaler_res = run_lstm_on_residuals(
        test_res, look_back=look_back
    )

    # SAUVEGARDE AVEC NOM DU SEGMENT
    suffix = f"{target_col}_{segment_name}"
    with open(f"{MODELS_DIR}/arimax_{suffix}.pkl", "wb") as f:
        pickle.dump(model_arima, f)
    model_lstm.save(f"{MODELS_DIR}/lstm_{suffix}.keras")
    joblib.dump(scaler_exog, f"{MODELS_DIR}/scaler_exog_{suffix}.joblib")
    joblib.dump(scaler_res, f"{MODELS_DIR}/scaler_res_{suffix}.joblib")

    print(f"      ✅ Modèles sauvegardés pour {segment_name} ({target_col})")


def main():
    DATA_PATH = "data/processed/dataset_entrainement_2024.csv"
    if not os.path.exists(DATA_PATH):
        return

    df_raw = pd.read_csv(DATA_PATH)
    segments = df_raw["segment"].unique()
    print(f"🚀 Début de l'entraînement global sur {len(segments)} segments...")

    for seg in segments:
        print(f"\n📍 TRAITEMENT SEGMENT : {seg}")
        df = load_data_segment(DATA_PATH, seg)
        run_pipeline(df, "NO2", seg, EXOG_FEATURES)
        run_pipeline(df, "VALD_NAVIGO", seg, EXOG_FEATURES)

    print("\n🎉 TOUS LES SEGMENTS DE PARIS SONT MODÉLISÉS !")


if __name__ == "__main__":
    main()
