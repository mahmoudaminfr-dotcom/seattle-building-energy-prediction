import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("1. Chargement et nettoyage des données de Seattle...")
data_path = "data/2016_Building_Energy_Benchmarking.csv"
df = pd.read_csv(data_path)

# Filtrage strict non-résidentiel (1 668 lignes)
non_residential_types = [
    "NonResidential",
    "SPS-District K-12",
    "Nonresidential COS",
    "Campus",
    "Nonresidential WA",
]
df_clean = df[df["BuildingType"].isin(non_residential_types)].copy()

# Filtrage de la cible météo-normalisée : SiteEnergyUseWN(kBtu) > 0 (1 640 lignes)
target_col = "SiteEnergyUseWN(kBtu)"
df_clean = df_clean[
    df_clean[target_col].notnull() & (df_clean[target_col] > 0)
].copy()

print(
    f"Nombre de lignes conservées pour l'entraînement : {len(df_clean)} (attendu : 1640)"
)

# Feature engineering
df_clean["BuildingAge"] = 2016 - df_clean["YearBuilt"]
df_clean["PrimaryPropertyType_Clean"] = df_clean["PrimaryPropertyType"].fillna(
    "Unknown"
)
df_clean["PropertyGFAParking"] = df_clean["PropertyGFAParking"].fillna(0.0)
df_clean["LargestPropertyUseTypeGFA"] = df_clean[
    "LargestPropertyUseTypeGFA"
].fillna(df_clean["PropertyGFATotal"])

df_clean["Parking_Ratio"] = (
    df_clean["PropertyGFAParking"] / df_clean["PropertyGFATotal"]
)
df_clean["MainUse_Ratio"] = (
    df_clean["LargestPropertyUseTypeGFA"] / df_clean["PropertyGFATotal"]
)

# Définition des features
features_num = [
    "PropertyGFATotal",
    "PropertyGFAParking",
    "LargestPropertyUseTypeGFA",
    "NumberofBuildings",
    "NumberofFloors",
    "BuildingAge",
    "Latitude",
    "Longitude",
    "Parking_Ratio",
    "MainUse_Ratio",
]
features_cat = ["PrimaryPropertyType_Clean"]
feature_names = features_num + features_cat

X = df_clean[feature_names]
y_log = np.log1p(df_clean[target_col])

# Pipeline de modélisation
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, features_num),
        ("cat", cat_pipeline, features_cat),
    ]
)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=5,
                n_estimators=200,
                subsample=0.7,
                random_state=42,
            ),
        ),
    ]
)

# Entraînement et sérialisation
print(
    "2. Entraînement du Gradient Boosting avec les hyperparamètres du GridSearchCV..."
)
model_pipeline.fit(X, y_log)

joblib.dump(model_pipeline, "model.joblib")
print("3. Modèle 'model.joblib' régénéré avec succès !")