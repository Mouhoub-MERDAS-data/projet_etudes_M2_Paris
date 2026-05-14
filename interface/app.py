"""
Dashboard Smart Mobility Paris — Point d'entrée

Lancer depuis la racine du projet :
    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Smart Mobility Paris",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === En-tête ===
st.title("🚦 Smart Mobility Paris")
st.subheader("Tableau de bord d'aide à la décision — Pollution & affluence transport")

st.markdown("""
Bienvenue sur le tableau de bord d'analyse et de prédiction de la mobilité urbaine
parisienne. Cet outil croise les données de pollution du périphérique (Airparif), de
fréquentation des transports en commun (validations Navigo) et de contexte (météo,
calendrier, événements) pour aider à la prise de décision opérationnelle.

---

### 🧭 Navigation

Utilisez le menu latéral pour accéder aux différentes vues :

1. **🏠 Vue d'ensemble** — KPIs globaux et chiffres clés 2024
2. **🗺️ Cartographie du périph** — diagramme des 8 segments et niveaux moyens
3. **📈 Séries temporelles** — exploration de la pollution heure par heure
4. **🚇 Affluence transport** — validations Navigo et patterns hebdomadaires
5. **🤖 Prédictions NO2** — résultats du modèle hybride ARIMAX + LSTM
6. **🚆 Prédictions affluence** — résultats du modèle SARIMAX + LSTM Navigo
7. **🔄 Analyse report modal** — validation de l'hypothèse projet

---

### 📌 Hypothèse projet

> **Les perturbations du transport ferré (grèves, incidents) provoquent un report
> modal vers la voiture, mesurable par une chute des validations Navigo et une
> hausse simultanée du NO2 sur le périphérique.**

Cet outil permet de visualiser cette relation et de la quantifier.

---

### 🧱 Sources de données

- **Pollution** : Airparif (NO2, PM10, PM25 horaires sur 8 segments du périph, 2024)
- **Transport** : Île-de-France Mobilités (validations FER journalières par catégorie)
- **Météo** : Visual Crossing (températures, vent, précipitations)
- **Calendrier** : jours fériés, vacances scolaires, JO/JOP 2024
""")

st.markdown("---")
st.caption(
    "Projet M2 Big Data & IA · Données 2024 · "
    "Modèles : XGBoost, ARIMAX+LSTM, SARIMAX+LSTM"
)
