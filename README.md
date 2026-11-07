<<<<<<< HEAD
# Alaotra Rain V5

Application Streamlit interactive pour l'analyse des précipitations, la comparaison de modèles Machine Learning et l'aide à la décision rizicole dans la région Alaotra-Mangoro.

## Nouveautés V5

- Logo officiel de la région au-dessus du titre **ALAOTRA RAIN**.
- Import CSV obligatoire avant toute analyse.
- Aucun traitement ou graphique n'est exécuté avant l'importation.
- Interface sombre inspirée de la maquette : vert, bleu, cartes, KPI et panneaux dynamiques.
- Thème sombre / clair.
- Vue d'ensemble avec KPI, courbes annuelles, cycle mensuel, carte et heatmap.
- Climatologie mensuelle.
- Analyse spatiale avec animation temporelle.
- Entraînement dynamique Random Forest + XGBoost depuis le CSV importé.
- Prévision de l'année suivante ou d'une année future.
- Conseils rizicoles et interprétations dynamiques.
- Export du dataset normalisé et des prévisions.

## Installation

```bash
cd projet_prevision_pluie_alaotra_v5
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Formats CSV acceptés

### Format long recommandé

```text
date,year,month,latitude,longitude,precipitation_mm_month
2000-01-01,2000,1,-18.0,48.125,122.02
```

### Format mensuel large

Colonnes :

```text
PARAMETER,YEAR,LAT,LON,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN
```

L'application détecte automatiquement la structure et la transforme en format exploitable.

## Important

Les prévisions restent expérimentales et doivent être complétées avec les observations de terrain, les services techniques agricoles et les bulletins météorologiques officiels.

## V6 — prévisions multi-années corrigées

La prévision est désormais réellement récursive entre les années. Si les observations se terminent en 2025 et que l'utilisateur choisit 2028, l'application calcule 2026, puis utilise ces prédictions pour 2027, puis celles de 2027 pour 2028. L'interface affiche également une comparaison des années intermédiaires et un avertissement d'incertitude qui augmente avec l'horizon.
=======
# ALAOTRA-RAIN-M2
>>>>>>> 7fa1c6fc4d40e0be739f7608b561dbcc94deea87
