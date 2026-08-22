# Seattle Building Energy Prediction API

Service REST de Machine Learning permettant de prédire la consommation énergétique annuelle totale (Site Energy Use, en kBtu) des bâtiments non résidentiels de la ville de Seattle.

Projet développé avec BentoML, conteneurisé via Docker et déployé en mode serverless sur Google Cloud Run.

---

## 📌 Architecture du projet

- service.py : Service BentoML (API, validation Pydantic, inférence)
- Dockerfile : Fichier de conteneurisation pour Cloud Run
- .gcloudignore : Fichiers exclus du build GCP
- test_api.py : Tests unitaires et d'intégration (pytest)
- requirements.txt : Dépendances Python du projet
- README.md : Documentation technique

---

## ⚙️ Architecture Technique & Modélisation

- Framework API : BentoML 1.4+ (serveur ASGI haute performance).
- Validation des schémas : Pydantic v2 (contrôle strict des types, plages de valeurs et coordonnées géographiques).
- Modèle ML : Pipeline scikit-learn composé de :
  - Imputation des valeurs manquantes (SimpleImputer).
  - Normalisation standard (StandardScaler) et encodage catégoriel (OneHotEncoder).
  - Régresseur GradientBoostingRegressor.
- Cible transformée : Logarithmique (log(1 + y)) à l'entraînement, recalculée en exponentielle (exp(y) - 1) pour l'inférence en kBtu.

---

## 🚀 Installation & Exécution en local

### 1. Prérequis
- Python 3.10 ou supérieur
- pip

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancement du serveur BentoML en local
```bash
bentoml serve service:SeattleEnergyService --reload --port 3000
```

L'interface interactive Swagger UI est accessible sur : http://localhost:3000

---

## 📡 Spécification de l'API (Endpoints)

### POST /predict
Prédit la consommation énergétique d'un bâtiment.
```
Exemple de corps de requête (JSON) :
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

## 🧪 Tests Unitaires (pytest)

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