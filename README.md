# Seattle Building Energy Prediction API

Service REST de Machine Learning permettant de prédire la consommation énergétique annuelle totale météo-normalisée (*Site Energy Use WN*, en kBtu) des bâtiments non résidentiels de la ville de Seattle[cite: 1, 2].

Projet développé avec BentoML, conteneurisé via Docker et déployé en mode serverless sur Google Cloud Run.

---

## 📌 Architecture du projet

- `data/2016_Building_Energy_Benchmarking.csv` : Données réelles de Seattle
- `notebooks/exploration_et_modelisation.ipynb` : Analyse exploratoire et GridSearchCV
- `service.py` : Service BentoML (validation Pydantic, inférence)
- `save_model.py` : Script d'entraînement et de sérialisation
- `model.joblib` : Artefact du pipeline scikit-learn entraîné
- `test_api.py` : Suite de tests automatisés (pytest)
- `Dockerfile` : Image conteneur pour le déploiement Cloud
- `requirements.txt` : Dépendances Python
- `.gcloudignore` : Fichiers exclus du build GCP
- `README.md` : Documentation technique

---

## ⚙️ Architecture Technique & Modélisation

- **Framework API** : BentoML 1.4+ (serveur ASGI haute performance).
- **Validation des schémas** : Pydantic v2 (contrôle strict des types, plages de valeurs physiques et coordonnées géographiques de Seattle).
- **Modèle ML** : Pipeline scikit-learn composé de :
  - Imputation des valeurs manquantes (`SimpleImputer`).
  - Normalisation standard (`StandardScaler`) et encodage catégoriel (`OneHotEncoder`).
  - Régresseur `GradientBoostingRegressor` optimisé par `GridSearchCV` (`learning_rate=0.05`, `n_estimators=200`, `max_depth=3`, `subsample=0.7`, `min_samples_split=5`).
- **Cible transformée** : Logarithmique (`np.log1p`) à l'entraînement pour corriger l'asymétrie de distribution, recalculée en exponentielle (`np.expm1`) lors de l'inférence pour restituer la valeur physique en kBtu[cite: 1, 2].

---

## 🚀 Installation & Exécution en local

### 1. Prérequis
- Python 3.10 ou supérieur
- pip et git

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. LEntraînement et génération de l'artefact
```
python save_model.py
```

### 4. Lancement du serveur BentoML en local
```bash
bentoml serve service:SeattleEnergyService --reload --port 3000
```

L'interface interactive Swagger UI est accessible sur : http://localhost:3000

---

## 📡 Spécification de l'API (Endpoints)

### POST /predict
Prédit la consommation énergétique d'un bâtiment.

Exemple de corps de requête (JSON) :
```
{
  "PropertyGFATotal": 88434.0,
  "PropertyGFAParking": 0.0,
  "LargestPropertyUseTypeGFA": 88434.0,
  "NumberofBuildings": 1.0,
  "NumberofFloors": 12,
  "BuildingAge": 89.0,
  "Latitude": 47.612,
  "Longitude": -122.337,
  "PrimaryPropertyType_Clean": "Hotel"
}
```

Exemple de réponse (200 OK) :
```
{
  "predicted_energy_log": 14.916,
  "predicted_energy_kbtu": 3005778.29,
  "status": "success"
}
```
Règles de validation des données (Pydantic) :
- PropertyGFATotal : Float > 0 (Surface totale du bâtiment en sq ft)
- PropertyGFAParking : Float >= 0 (Surface parking en sq ft)
- LargestPropertyUseTypeGFA : Float > 0 (Surface de l'usage principal)
- NumberofBuildings : Float >= 1.0 (Nombre de bâtiments)
- NumberofFloors : Integer entre 1 et 120 (Nombre d'étages)
- BuildingAge : Float entre 0 et 200 (Âge du bâtiment en années)
- Latitude : Float entre 47.4 et 47.8 (Zone Seattle)
- Longitude : Float entre -122.5 et -122.2 (Zone Seattle)
- PrimaryPropertyType_Clean : String requis (ex. Hotel, Large Office)

---

## 🧪 Tests Automatisés (pytest)

Une suite de tests automatisés valide la cohérence des prédictions ainsi que le rejet des données aberrantes :
```
pytest test_api.py -v
```
---

## ☁️ Déploiement sur Google Cloud Run

L'application est déployée en architecture Serverless sur Google Cloud Run (facturation à la requête, scale-to-zero automatique).

### Déploiement :
```
gcloud run deploy seattle-energy-service --source . --region europe-west1 --port 3000 --memory 1Gi --allow-unauthenticated --quiet
```

### Suppression du service :
```
gcloud run services delete seattle-energy-service --region europe-west1 --quiet
```