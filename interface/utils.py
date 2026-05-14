"""
Utilitaires partagés entre les pages : chargement des données + ordre des segments.
Les fonctions sont mises en cache pour éviter de relire les CSV à chaque navigation.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# Chemins résolus depuis dashboard/ vers data/processed/
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Ordre géographique des 8 segments (cohérent avec les notebooks)
SEGMENTS_ORDER = [
    "Chap-Bagn",
    "Bagn-Berc",
    "Berc-Ital",
    "Ital-A6a",
    "A6a-Sevr",
    "Sevr-Aute",
    "Aute-Mail",
    "Mail-Chap",
]

# Coordonnées schématiques du périphérique (cercle, segments dans le sens horaire)
# Utilisées pour le diagramme Plotly de la page 2.
SEGMENTS_COORDS = {
    "Chap-Bagn": (0.50, 0.95),  # nord
    "Bagn-Berc": (0.85, 0.78),
    "Berc-Ital": (0.95, 0.40),  # est
    "Ital-A6a": (0.78, 0.10),
    "A6a-Sevr": (0.40, 0.05),  # sud
    "Sevr-Aute": (0.10, 0.20),
    "Aute-Mail": (0.05, 0.55),  # ouest
    "Mail-Chap": (0.20, 0.88),
}

# Seuils OMS / UE pour le NO2 (µg/m³, moyenne annuelle)
SEUIL_NO2_OMS = 10
SEUIL_NO2_UE = 40


@st.cache_data
def load_hourly():
    """Charge le dataset d'entraînement complet (70 272 lignes horaires)."""
    df = pd.read_csv(DATA_DIR / "dataset_entrainement_2024.csv")
    df["time"] = pd.to_datetime(df["time"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


@st.cache_data
def load_daily():
    """Charge le dataset journalier (366 lignes)."""
    df = pd.read_csv(DATA_DIR / "dataset_journalier_2024.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def color_for_no2(value):
    """Code couleur pour un niveau de NO2 (µg/m³)."""
    if value < 20:
        return "#2ca02c"  # vert
    if value < 40:
        return "#ffd700"  # jaune (limite UE annuelle)
    if value < 60:
        return "#ff7f0e"  # orange
    return "#d62728"  # rouge


def categorie_no2(value):
    """Étiquette qualitative d'un niveau de NO2."""
    if value < 20:
        return "Bon"
    if value < 40:
        return "Modéré"
    if value < 60:
        return "Dégradé"
    return "Mauvais"
