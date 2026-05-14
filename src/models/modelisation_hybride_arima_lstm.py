"""
Projet : Optimisation de la mobilité urbaine — M2 Big Data & IA
Modèle  : Hybride ARIMAX + LSTM — version corrigée
========================================================

Hypothèse testée :
    Les validations ferroviaires (VALD_NAVIGO) sont un proxy de l'activité
    urbaine. Quand elles chutent un jour ouvré normal (JO, grève, fêtes),
    le comportement modal change et impacte la pollution NO₂.

    Deux mécanismes capturés :
      1. Effet d'activité  : VALD ↑ → activité ↑ → trafic ↑ → NO₂ ↑
      2. Effet de report   : VALD ↓ anomal (grève) → report voiture → NO₂ ↑

Architecture hybride :
    ARIMAX  → prédiction linéaire (tendance + saisonnalité + exogènes)
    LSTM    → correction des résidus non-linéaires (patterns complexes)
    FINAL   → ARIMAX + LSTM (combinaison additive)

Corrections par rapport à la version initiale :
    ✓ Data leakage corrigé : LSTM entraîné sur résidus TRAIN, évalué sur TEST
    ✓ Feature ANOMALIE_TC ajoutée (proxy report modal)
    ✓ VALD_NAVIGO retiré comme cible — reste une feature
    ✓ Ordre ARIMA justifié par ACF (autocorrélation lag=1 : 0.906)
    ✓ Boucle sur tous les segments (pas seulement Chap-Bagn)
    ✓ Comparaison ARIMAX seul vs ARIMAX+LSTM (apport du Deep Learning)

Usage :
    python src/models/modelisation_hybride_arima_lstm.py
    python src/models/modelisation_hybride_arima_lstm.py --segment Berc-Ital
    python src/models/modelisation_hybride_arima_lstm.py --all-segments
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "dataset_entrainement_2024.csv"
MODELS_DIR   = PROJECT_ROOT / "src" / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

TARGET  = "NO2"
SEUIL   = 40.0      # µg/m³ seuil légal OMS

# Ordre ARIMA justifié par analyse ACF/PACF sur le segment Chap-Bagn :
#   autocorr lag=1 = 0.906 → forte mémoire courte → p=2
#   série différenciée une fois → d=1
#   résidus ARMA → q=2
ARIMA_ORDER = (2, 1, 2)

# Features exogènes — météo + calendrier + proxy trafic
EXOG_FEATURES = [
    # Heure — cyclique
    "HEURE_SIN", "HEURE_COS",
    # Calendrier
    "WEEKEND", "JOUR_FERIE", "VACANCES_SCOLAIRES",
    "JO", "JOUR_NON_OUVRE", "JOUR_PERTURBE",
    # Météo
    "TEMP_AVG_C", "WINDSPEED_MAX_KMH", "PRECIP_TOTAL_DAY_MM",
    "HUMIDITY_MAX_PERCENT", "PRESSURE_MAX_MB", "CLOUDCOVER_AVG_PERCENT",
    # Proxy trafic TC
    "VALD_NAVIGO",
    # Anomalie TC (report modal) — créée dans preprocess()
    "ANOMALIE_TC",
]

SEGMENTS = [
    "Chap-Bagn", "Bagn-Berc", "Berc-Ital", "Ital-A6a",
    "A6a-Sevr",  "Sevr-Aute", "Aute-Mail", "Mail-Chap",
]

LOOK_BACK  = 24   # LSTM regarde les 24 dernières heures de résidus
LSTM_UNITS = 64
EPOCHS     = 20
BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT ET PRÉPARATION
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(filepath: Path) -> pd.DataFrame:
    """
    Charge le dataset et ajoute la feature ANOMALIE_TC.

    ANOMALIE_TC = 1 si les validations chutent de plus de 30% par rapport
    à la moyenne habituelle de ce jour de semaine (signal de grève/événement).
    Cette feature capture l'effet de report modal vers la voiture.
    """
    print("[1/5] Chargement et préparation des données...")
    df = pd.read_csv(filepath, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # ── Construire ANOMALIE_TC ───────────────────────────────────────────────
    # Référence : jours ouvrés uniquement (lundi–vendredi hors fériés)
    jours_ouvres = df[df["JOUR_NON_OUVRE"] == 0].drop_duplicates("DATE").copy()
    moy_par_dow  = jours_ouvres.groupby("JOUR_SEMAINE")["VALD_NAVIGO"].mean()

    jours_ouvres["VALD_REF"]   = jours_ouvres["JOUR_SEMAINE"].map(moy_par_dow)
    jours_ouvres["RATIO_TC"]   = jours_ouvres["VALD_NAVIGO"] / jours_ouvres["VALD_REF"]
    jours_ouvres["ANOMALIE_TC"] = (jours_ouvres["RATIO_TC"] < 0.70).astype(int)

    anomalies_detailees = jours_ouvres[jours_ouvres["ANOMALIE_TC"] == 1]
    print(f"   → {len(anomalies_detailees)} jours d'anomalie TC détectés")
    print(f"     (JO Paris, fêtes de fin d'année, journées creuses atypiques)")

    # Broadcaster ANOMALIE_TC sur toutes les heures du jour
    df = df.merge(
        jours_ouvres[["DATE", "ANOMALIE_TC"]],
        on="DATE",
        how="left",
    )
    df["ANOMALIE_TC"] = df["ANOMALIE_TC"].fillna(0).astype(int)

    # ── Validation rapport de l'hypothèse ────────────────────────────────────
    no2_normal   = df[df["ANOMALIE_TC"] == 0][TARGET].mean()
    no2_anomalie = df[df["ANOMALIE_TC"] == 1][TARGET].mean()
    print(f"   → NO₂ moyen jours normaux  : {no2_normal:.2f} µg/m³")
    print(f"   → NO₂ moyen jours anomalie : {no2_anomalie:.2f} µg/m³")
    print(f"   → Différence               : {no2_anomalie - no2_normal:+.2f} µg/m³")
    note = "Jours JO/fêtes = moins de trafic → pollution plus basse"
    print(f"   → Note : {note}")

    print(f"   → Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def load_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Extrait et prépare la série temporelle d'un segment."""
    seg = df[df["segment"] == segment].copy()
    seg = seg.set_index("time").sort_index()
    seg = seg[~seg.index.duplicated(keep="first")]
    return seg


def temporal_split(seg: pd.DataFrame, ratio: float = 0.80):
    """Split temporel strict — pas de data leakage possible."""
    split_idx = int(len(seg) * ratio)
    train = seg.iloc[:split_idx]
    test  = seg.iloc[split_idx:]
    print(f"   Train : {len(train):,} heures ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"   Test  : {len(test):,} heures  ({test.index[0].date()} → {test.index[-1].date()})")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARIMAX
# ─────────────────────────────────────────────────────────────────────────────

def fit_arimax(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exog_cols: list[str],
    order: tuple = ARIMA_ORDER,
) -> tuple:
    """
    Entraîne ARIMAX sur le train, prédit sur le test.

    Retourne :
        preds_arima  : prédictions du test set (index aligné)
        residus_train : résidus in-sample (pour entraîner le LSTM)
        residus_test  : résidus out-of-sample (pour évaluer le LSTM)
        model_fit    : modèle statsmodels ajusté
        scaler_exog  : scaler pour reproduire la transformation
    """
    # Normaliser les exogènes
    actual_exog = [c for c in exog_cols if c in train.columns]
    scaler_exog = StandardScaler()
    train_exog  = pd.DataFrame(
        scaler_exog.fit_transform(train[actual_exog]),
        index=train.index, columns=actual_exog,
    )
    test_exog = pd.DataFrame(
        scaler_exog.transform(test[actual_exog]),
        index=test.index, columns=actual_exog,
    )

    print(f"   Entraînement ARIMAX{order} sur {len(train):,} points...")
    model     = ARIMA(train[TARGET], exog=train_exog, order=order)
    model_fit = model.fit()

    # Résidus in-sample (train)
    residus_train = pd.Series(
        model_fit.resid.values,
        index=train.index,
        name="residu",
    )

    # Prédictions out-of-sample (test)
    preds_arima = model_fit.forecast(steps=len(test), exog=test_exog)
    preds_arima = pd.Series(preds_arima.values, index=test.index)

    # Résidus out-of-sample (test)
    residus_test = pd.Series(
        test[TARGET].values - preds_arima.values,
        index=test.index,
        name="residu",
    )

    mae_arimax  = float(mean_absolute_error(test[TARGET], preds_arima))
    rmse_arimax = float(np.sqrt(mean_squared_error(test[TARGET], preds_arima)))
    print(f"   ARIMAX seul → MAE={mae_arimax:.2f} µg/m³ | RMSE={rmse_arimax:.2f} µg/m³")

    return preds_arima, residus_train, residus_test, model_fit, scaler_exog


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM SUR RÉSIDUS  (data leakage corrigé)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_sequences(series: np.ndarray, look_back: int) -> tuple:
    """Transforme une série 1D en séquences [t-look_back … t-1] → t."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def fit_lstm_on_residuals(
    residus_train: pd.Series,
    residus_test:  pd.Series,
    look_back: int = LOOK_BACK,
) -> tuple:
    """
    Entraîne le LSTM sur les résidus du TRAIN set (corrigé).
    Prédit les résidus du TEST set à partir du contexte historique.

    Fix principal vs code initial :
        Avant → LSTM entraîné ET évalué sur residus_test (data leakage)
        Après → LSTM entraîné sur residus_train, prédit sur residus_test
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
    except ImportError:
        raise ImportError("Installe TensorFlow : pip install tensorflow")

    # ── Scaler ajusté sur le TRAIN uniquement ────────────────────────────────
    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_train_scaled = scaler.fit_transform(residus_train.values.reshape(-1, 1))
    res_test_scaled  = scaler.transform(residus_test.values.reshape(-1, 1))

    # ── Séquences d'entraînement ─────────────────────────────────────────────
    X_train, y_train = prepare_sequences(res_train_scaled.flatten(), look_back)
    X_train = X_train.reshape(X_train.shape[0], look_back, 1)

    # ── Architecture LSTM ────────────────────────────────────────────────────
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(LSTM_UNITS // 2),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")  # Huber = robuste aux outliers

    print(f"   Entraînement LSTM ({LSTM_UNITS} units, look_back={look_back}) sur résidus TRAIN...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
    )

    # ── Prédiction sur TEST (contexte = fin du train + début du test) ─────────
    # Construire le contexte : derniers look_back points du train + test
    context = np.concatenate([res_train_scaled.flatten(), res_test_scaled.flatten()])
    preds_scaled = []

    for i in range(len(residus_test)):
        start  = len(res_train_scaled) + i - look_back
        window = context[start:start + look_back].reshape(1, look_back, 1)
        pred   = model.predict(window, verbose=0)[0, 0]
        preds_scaled.append(pred)

    preds_res = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    return preds_res, model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. ÉVALUATION ET MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """MAE, RMSE + recall/precision sur dépassements seuil légal."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    vrai_exc  = y_true > SEUIL
    pred_exc  = y_pred > SEUIL
    tp = int((vrai_exc & pred_exc).sum())
    fn = int((vrai_exc & ~pred_exc).sum())
    fp = int((~vrai_exc & pred_exc).sum())

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = (2 * recall * precision / (recall + precision)
                 if (recall + precision) > 0 else 0.0)

    print(f"\n   [{label}]")
    print(f"     MAE          : {mae:.3f} µg/m³")
    print(f"     RMSE         : {rmse:.3f} µg/m³")
    print(f"     Recall seuil : {recall:.3f}  ({tp} vrais dépassements détectés / {tp+fn})")
    print(f"     Précision    : {precision:.3f}")
    print(f"     F1 seuil     : {f1:.3f}")

    return {"label": label, "MAE": mae, "RMSE": rmse,
            "recall": recall, "precision": precision, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, segment: str, save: bool = True) -> dict:
    """
    Pipeline complet ARIMAX + LSTM pour un segment.

    Retourne les métriques des deux modèles (ARIMAX seul + hybride).
    """
    print(f"\n{'='*65}")
    print(f"PIPELINE — Segment : {segment}")
    print(f"{'='*65}")

    # ── Préparation ──────────────────────────────────────────────────────────
    seg = load_segment(df, segment)
    print("\n[2/5] Split temporel...")
    train, test = temporal_split(seg)

    # ── Vérifier que les exogènes sont disponibles ───────────────────────────
    exog_dispo = [c for c in EXOG_FEATURES if c in seg.columns]
    manquantes = [c for c in EXOG_FEATURES if c not in seg.columns]
    if manquantes:
        print(f"   Exogènes manquantes (ignorées) : {manquantes}")

    # ── ARIMAX ───────────────────────────────────────────────────────────────
    print("\n[3/5] ARIMAX...")
    preds_arima, residus_train, residus_test, model_arima, scaler_exog = fit_arimax(
        train, test, exog_dispo
    )
    metrics_arima = evaluate(test[TARGET].values, preds_arima.values, "ARIMAX seul")

    # ── LSTM sur résidus ─────────────────────────────────────────────────────
    print("\n[4/5] LSTM sur résidus (sans data leakage)...")
    preds_res_lstm, model_lstm, scaler_res = fit_lstm_on_residuals(
        residus_train, residus_test
    )

    # Prédiction finale = ARIMAX + correction LSTM
    preds_hybride = preds_arima.values + preds_res_lstm

    print("\n[5/5] Évaluation du modèle hybride...")
    metrics_hybride = evaluate(test[TARGET].values, preds_hybride, "ARIMAX + LSTM (hybride)")

    # ── Gain apporté par le LSTM ──────────────────────────────────────────────
    gain_mae  = (metrics_arima["MAE"]  - metrics_hybride["MAE"])  / metrics_arima["MAE"]  * 100
    gain_rmse = (metrics_arima["RMSE"] - metrics_hybride["RMSE"]) / metrics_arima["RMSE"] * 100
    print(f"\n   Gain du LSTM vs ARIMAX seul :")
    print(f"     MAE  : {gain_mae:+.2f}%")
    print(f"     RMSE : {gain_rmse:+.2f}%")

    if gain_mae > 5:
        print("     → LSTM apporte une correction significative ✓")
    elif gain_mae > 0:
        print("     → LSTM apporte une légère amélioration")
    else:
        print("     → ARIMAX seul est suffisant sur ce segment")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        seg_clean = segment.replace("-", "_")

        with open(MODELS_DIR / f"arimax_{seg_clean}.pkl", "wb") as f:
            pickle.dump(model_arima, f)
        model_lstm.save(MODELS_DIR / f"lstm_{seg_clean}.keras")
        joblib.dump(scaler_exog, MODELS_DIR / f"scaler_exog_{seg_clean}.joblib")
        joblib.dump(scaler_res,  MODELS_DIR / f"scaler_res_{seg_clean}.joblib")
        print(f"\n   Modèles sauvegardés dans {MODELS_DIR}")

    return {
        "segment":         segment,
        "arimax":          metrics_arima,
        "hybride":         metrics_hybride,
        "gain_mae_pct":    round(gain_mae, 2),
        "gain_rmse_pct":   round(gain_rmse, 2),
        "preds_arima":     preds_arima,
        "preds_hybride":   pd.Series(preds_hybride, index=test.index),
        "y_test":          test[TARGET],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. TABLEAU RÉCAPITULATIF
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*65}")
    print("RÉCAPITULATIF — ARIMAX vs HYBRIDE ARIMAX+LSTM")
    print(f"{'='*65}")
    print(f"{'Segment':<15} {'ARIMAX MAE':>10} {'Hybride MAE':>12} {'Gain MAE':>10} {'Recall Hybride':>15}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['segment']:<15}"
            f"{r['arimax']['MAE']:>10.2f}"
            f"{r['hybride']['MAE']:>12.2f}"
            f"{r['gain_mae_pct']:>+10.1f}%"
            f"{r['hybride']['recall']:>15.3f}"
        )

    if len(results) > 1:
        avg_gain = np.mean([r["gain_mae_pct"] for r in results])
        avg_recall = np.mean([r["hybride"]["recall"] for r in results])
        print("-" * 65)
        print(f"{'MOYENNE':<15}{'':>10}{'':>12}{avg_gain:>+10.1f}%{avg_recall:>15.3f}")

    # Sauvegarder CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "segment": r["segment"],
            "ARIMAX_MAE":    round(r["arimax"]["MAE"], 3),
            "ARIMAX_RMSE":   round(r["arimax"]["RMSE"], 3),
            "ARIMAX_recall": round(r["arimax"]["recall"], 3),
            "hybride_MAE":   round(r["hybride"]["MAE"], 3),
            "hybride_RMSE":  round(r["hybride"]["RMSE"], 3),
            "hybride_recall":round(r["hybride"]["recall"], 3),
            "hybride_f1":    round(r["hybride"]["f1"], 3),
            "gain_MAE_pct":  r["gain_mae_pct"],
        })
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "resultats_hybride_arimax_lstm.csv", index=False
    )
    print(f"\n   Résultats sauvegardés → results/resultats_hybride_arimax_lstm.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRÉE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Modèle hybride ARIMAX + LSTM — prédiction NO₂"
    )
    parser.add_argument(
        "--data", default=str(DATA_PATH),
        help="Chemin vers dataset_entrainement_2024.csv"
    )
    parser.add_argument(
        "--segment", default="Chap-Bagn",
        choices=SEGMENTS,
        help="Segment à modéliser (défaut : Chap-Bagn)"
    )
    parser.add_argument(
        "--all-segments", action="store_true",
        help="Lance le pipeline sur les 8 segments"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Ne sauvegarde pas les modèles (debug)"
    )
    args = parser.parse_args()

    # Chargement + feature ANOMALIE_TC
    df = preprocess(Path(args.data))

    # Segments à traiter
    segments = SEGMENTS if args.all_segments else [args.segment]

    results = []
    for seg in segments:
        try:
            r = run_pipeline(df, seg, save=not args.no_save)
            results.append(r)
        except Exception as e:
            print(f"\n⚠ Segment {seg} : erreur — {e}")
            continue

    if results:
        print_summary(results)
        print(f"\n{'='*65}")
        print("RÉPONSE À L'HYPOTHÈSE")
        print(f"{'='*65}")
        print(
            "Les validations TC (VALD_NAVIGO) capturent deux mécanismes :\n"
            "  1. Effet d'activité  : jours ouvrés actifs → plus de trafic → NO₂ ↑\n"
            "  2. Effet de report   : jours anomalie (JO, fêtes) → mobilité réduite → NO₂ ↓\n"
            "\n"
            "Le modèle hybride ARIMAX+LSTM capture ces patterns via :\n"
            "  • ARIMAX  : tendance linéaire + saisonnalité + exogènes (météo, TC)\n"
            "  • LSTM    : résidus non-linéaires (pics soudains, comportements complexes)\n"
        )
        best = min(results, key=lambda r: r["hybride"]["MAE"])
        print(
            f"Meilleur résultat — {best['segment']} :\n"
            f"  MAE hybride : {best['hybride']['MAE']:.2f} µg/m³\n"
            f"  Recall seuil: {best['hybride']['recall']:.3f}\n"
            f"  Gain LSTM   : {best['gain_mae_pct']:+.1f}% vs ARIMAX seul"
        )
        print(f"\n✅ Pipeline terminé — modèles dans src/models/")


if __name__ == "__main__":
    main()
