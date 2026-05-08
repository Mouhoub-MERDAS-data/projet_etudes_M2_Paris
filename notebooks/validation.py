
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


# Chaque fichier a un encodage différent, on le précise
df_s1 = pd.read_csv('data/2024_S1_NB_FER.txt', sep='\t', encoding='iso-8859-1', dtype=str)
df_t3 = pd.read_csv('data/2024_T3_NB_FER.txt', sep='\t', encoding='utf-8', dtype=str)
df_t4 = pd.read_csv('data/2024_T4_NB_FER.txt', sep='\t', encoding='utf-8', dtype=str)

print(f"S1 : {len(df_s1)} lignes")
print(f"T3 : {len(df_t3)} lignes")
print(f"T4 : {len(df_t4)} lignes")

# Fusion
df = pd.concat([df_s1, df_t3, df_t4], ignore_index=True)
print(f"Total : {len(df)} lignes")


# S1 = '01/01/2024', T3 = '01/07/24', T4 = '2024-10-01'
df['JOUR'] = pd.to_datetime(df['JOUR'], format='mixed', dayfirst=True, errors='coerce')

#  NB_VALD : enlever les espaces ('2 093' -> 2093) puis convertir en int ---
df['NB_VALD'] = df['NB_VALD'].str.replace(' ', '').str.replace('\xa0', '')
df['NB_VALD'] = pd.to_numeric(df['NB_VALD'], errors='coerce')

#  Suppression des lignes avec des valeurs invalides ---
df = df.dropna(subset=['JOUR', 'NB_VALD'])
df['NB_VALD'] = df['NB_VALD'].astype(int)

# Ajout de variables temporelles utiles ---
df['mois'] = df['JOUR'].dt.month
df['jour_semaine'] = df['JOUR'].dt.day_name()
df['weekend'] = df['JOUR'].dt.dayofweek >= 5

# Sauvegarde du fichier propre ---
df.to_csv('validations_fer_2024.csv', index=False)
print(f"\n✅ Fichier propre sauvegardé : {len(df)} lignes")


# 3. EXPLORATION


print("\n--- APERÇU ---")
print(df.head())

print("\n--- INFOS ---")
print(df.info())

print(f"\nPériode : {df['JOUR'].min().date()} -> {df['JOUR'].max().date()}")
print(f"Arrêts uniques : {df['CODE_STIF_ARRET'].nunique()}")
print(f"Total validations : {df['NB_VALD'].sum():,}")


# --- Top 10 des arrêts les plus fréquentés ---
print("\n--- TOP 10 DES ARRÊTS ---")
top_arrets = df.groupby('LIBELLE_ARRET')['NB_VALD'].sum().sort_values(ascending=False).head(10)
print(top_arrets)

plt.figure(figsize=(12, 6))
top_arrets.plot(kind='barh', color='teal')
plt.title('Top 10 des arrêts les plus fréquentés en 2024')
plt.xlabel('Nombre total de validations')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- Évolution mensuelle ---
print("\n--- VALIDATIONS PAR MOIS ---")
par_mois = df.groupby('mois')['NB_VALD'].sum()
print(par_mois)

plt.figure(figsize=(10, 5))
par_mois.plot(kind='bar', color='steelblue')
plt.title('Validations totales par mois en 2024')
plt.xlabel('Mois')
plt.ylabel('Nombre de validations')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# --- Répartition par catégorie de titre ---
print("\n--- VALIDATIONS PAR CATÉGORIE ---")
par_cat = df.groupby('CATEGORIE_TITRE')['NB_VALD'].sum().sort_values(ascending=False)
print(par_cat)

plt.figure(figsize=(10, 5))
par_cat.plot(kind='bar', color='coral')
plt.title('Répartition par catégorie de titre')
plt.ylabel('Nombre de validations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- Comparaison semaine vs week-end ---
print("\n--- SEMAINE vs WEEK-END ---")
par_jour = df.groupby('jour_semaine')['NB_VALD'].sum()
# Ordre des jours
ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
par_jour = par_jour.reindex(ordre)
print(par_jour)

plt.figure(figsize=(10, 5))
par_jour.plot(kind='bar', color='purple')
plt.title('Validations par jour de la semaine')
plt.ylabel('Nombre de validations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()