import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import bentoml


class BuildingInput(BaseModel):
    PropertyGFATotal: float = Field(
        ..., gt=0, description="Surface totale (sq ft)"
    )
    PropertyGFAParking: float = Field(
        0.0, ge=0, description="Surface parking (sq ft)"
    )
    LargestPropertyUseTypeGFA: float = Field(
        ..., gt=0, description="Surface usage principal (sq ft)"
    )
    NumberofBuildings: float = Field(
        1.0, ge=1.0, description="Nombre de batiments"
    )
    NumberofFloors: int = Field(
        ..., ge=1, le=120, description="Nombre d etages"
    )
    BuildingAge: float = Field(..., ge=0, le=200, description="Age du batiment")
    Latitude: float = Field(
        ..., ge=47.4, le=47.8, description="Latitude (Seattle)"
    )
    Longitude: float = Field(
        ..., ge=-122.5, le=-122.2, description="Longitude (Seattle)"
    )
    PrimaryPropertyType_Clean: str = Field(
        ..., description="Type d usage principal"
    )


class PredictionOutput(BaseModel):
    predicted_energy_log: float
    predicted_energy_kbtu: float
    status: str


@bentoml.service(
    name="seattle_energy_predictor",
    resources={"cpu": "1"},
)
class SeattleEnergyService:

    def __init__(self):
        # 1. Chargement de l'artefact entraîné (issu de save_model.py)
        self.pipeline = joblib.load("model.joblib")

        # 2. Liste des colonnes attendues par le pipeline scikit-learn
        self.feature_names = [
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
            "PrimaryPropertyType_Clean",
        ]

    @bentoml.api
    def predict(self, data: BuildingInput) -> PredictionOutput:
        input_data = data.model_dump()

        # Calcul des features d'ingénierie
        input_data["Parking_Ratio"] = (
            input_data["PropertyGFAParking"] / input_data["PropertyGFATotal"]
        )
        input_data["MainUse_Ratio"] = (
            input_data["LargestPropertyUseTypeGFA"]
            / input_data["PropertyGFATotal"]
        )

        # Mise en forme DataFrame et prédiction
        df_input = pd.DataFrame([input_data])[self.feature_names]
        pred_log = float(self.pipeline.predict(df_input)[0])
        pred_kbtu = float(np.expm1(pred_log))

        return PredictionOutput(
            predicted_energy_log=round(pred_log, 4),
            predicted_energy_kbtu=round(pred_kbtu, 2),
            status="success",
        )