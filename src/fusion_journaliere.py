"""
Fusion journalière du dataset Smart Mobility Paris.

Combine sur la clé DATE (YYYY-MM-DD) :
  - meteo_paris_2024_clean.csv        (1 ligne / jour)
  - evenements_paris_2024.csv         (1 ligne / jour)
  - validations_fer_2024.csv          (agrégé : 1 ligne / jour, ventilé par catégorie)

Les données HORAIRES (NO2, PM10, PM25) ne sont PAS touchées ici — elles seront
broadcast sur ce dataset journalier dans une étape ultérieure.
"""

import pandas as pd
from pathlib import Path

# === Chemins ===
# Résolus par rapport à l'emplacement du script (src/), donc lancement
# possible depuis n'importe quel dossier.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = DATA_DIR / "dataset_journalier_2024.csv"


# === 1. Validations FER : pivot par catégorie + total ===
print("[1/4] Agrégation des validations FER...")

fer = pd.read_csv(
    DATA_DIR / "validations_fer_2024.csv",
    usecols=["JOUR", "CATEGORIE_TITRE", "NB_VALD"],
)

# Pivot : une colonne par catégorie de titre
fer_pivot = (
    fer.pivot_table(
        index="JOUR",
        columns="CATEGORIE_TITRE",
        values="NB_VALD",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)

# Renommage propre des colonnes (snake_case sans accent)
rename_map = {
    "Forfait Navigo": "VALD_NAVIGO",
    "Imagine R": "VALD_IMAGINE_R",
    "Contrat Solidarité Transport": "VALD_SOLIDARITE",
    "Autres titres": "VALD_AUTRES",
    "NON DEFINI": "VALD_NON_DEFINI",
    "Amethyste": "VALD_AMETHYSTE",
    "Forfaits courts": "VALD_COURTS",
}
fer_pivot = fer_pivot.rename(columns=rename_map)
fer_pivot = fer_pivot.rename(columns={"JOUR": "DATE"})

# Total réseau / jour
vald_cols = [c for c in fer_pivot.columns if c.startswith("VALD_")]
fer_pivot["VALD_TOTAL"] = fer_pivot[vald_cols].sum(axis=1)

print(f"   → {len(fer_pivot)} jours, {len(vald_cols)} catégories + total")


# === 2. Météo ===
print("[2/4] Lecture météo...")
meteo = pd.read_csv(DATA_DIR / "meteo_paris_2024_clean.csv")

# Les colonnes temporelles MOIS / JOUR_SEMAINE / WEEKEND existent aussi dans
# evenements (avec en plus NOM_JOUR, NUM_SEMAINE, JOUR_ANNEE).
# On les garde côté evenements pour éviter les doublons.
meteo = meteo.drop(columns=["MOIS", "JOUR_SEMAINE", "WEEKEND"])
print(f"   → {len(meteo)} jours, {len(meteo.columns)-1} variables météo")


# === 3. Événements ===
print("[3/4] Lecture événements...")
events = pd.read_csv(DATA_DIR / "evenements_paris_2024.csv")
print(f"   → {len(events)} jours, {len(events.columns)-1} variables événementielles")


# === 4. Fusion sur DATE ===
print("[4/4] Fusion sur DATE (YYYY-MM-DD)...")

# Sécurité : harmoniser le type DATE (string YYYY-MM-DD partout)
for df in (meteo, events, fer_pivot):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")

dataset = (
    events
    .merge(meteo,     on="DATE", how="left", validate="one_to_one")
    .merge(fer_pivot, on="DATE", how="left", validate="one_to_one")
)

# Tri chronologique
dataset = dataset.sort_values("DATE").reset_index(drop=True)

# === Sauvegarde + contrôles ===
dataset.to_csv(OUT_PATH, index=False)

print(f"\n✓ Fichier écrit : {OUT_PATH}")
print(f"  Dimensions  : {dataset.shape[0]} lignes × {dataset.shape[1]} colonnes")
print(f"  Période     : {dataset['DATE'].min()} → {dataset['DATE'].max()}")
print(f"  NaN totaux  : {dataset.isna().sum().sum()}")

print("\n  Colonnes :")
for c in dataset.columns:
    print(f"    - {c}")

print("\n  Aperçu :")
print(dataset.head(3).to_string())
