import bentoml
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient
from service import BuildingInput, SeattleEnergyService


@pytest.fixture(scope="module")
def service_instance():
    """Initialise une instance du service BentoML pour les tests unitaires."""
    return SeattleEnergyService()


def test_valid_prediction(service_instance):
    """Vérifie qu'une entrée valide produit une prédiction cohérente."""
    valid_data = BuildingInput(
        PropertyGFATotal=88434.0,
        PropertyGFAParking=0.0,
        LargestPropertyUseTypeGFA=88434.0,
        NumberofBuildings=1.0,
        NumberofFloors=12,
        BuildingAge=89.0,
        Latitude=47.612,
        Longitude=-122.337,
        PrimaryPropertyType_Clean="Hotel",
    )

    result = service_instance.predict(valid_data)

    assert result.status == "success"
    assert result.predicted_energy_log > 0
    assert result.predicted_energy_kbtu > 0
    assert isinstance(result.predicted_energy_kbtu, float)


def test_invalid_negative_surface():
    """Vérifie que Pydantic bloque les surfaces négatives ou nulles."""
    with pytest.raises(ValidationError):
        BuildingInput(
            PropertyGFATotal=-500.0,
            PropertyGFAParking=0.0,
            LargestPropertyUseTypeGFA=500.0,
            NumberofBuildings=1.0,
            NumberofFloors=2,
            BuildingAge=10.0,
            Latitude=47.60,
            Longitude=-122.33,
            PrimaryPropertyType_Clean="Office",
        )


def test_invalid_seattle_coordinates():
    """Vérifie que Pydantic bloque les coordonnées en dehors de Seattle."""
    with pytest.raises(ValidationError):
        BuildingInput(
            PropertyGFATotal=10000.0,
            PropertyGFAParking=0.0,
            LargestPropertyUseTypeGFA=10000.0,
            NumberofBuildings=1.0,
            NumberofFloors=2,
            BuildingAge=10.0,
            Latitude=48.8566,
            Longitude=2.3522,
            PrimaryPropertyType_Clean="Office",
        )


def test_invalid_building_floors():
    """Vérifie que Pydantic bloque un nombre d'étages aberrant."""
    with pytest.raises(ValidationError):
        BuildingInput(
            PropertyGFATotal=10000.0,
            PropertyGFAParking=0.0,
            LargestPropertyUseTypeGFA=10000.0,
            NumberofBuildings=1.0,
            NumberofFloors=250,
            BuildingAge=10.0,
            Latitude=47.60,
            Longitude=-122.33,
            PrimaryPropertyType_Clean="Office",
        )


def test_http_endpoint_prediction():
    """Test d'intégration : effectue un véritable appel HTTP POST sur l'endpoint /predict via ASGI."""
    svc = bentoml.load("service:SeattleEnergyService")
    app = svc.to_asgi()

    # Le context manager 'with' active les événements de démarrage (lifespan) de BentoML
    with TestClient(app) as client:
        payload = {
            "data": {
                "PropertyGFATotal": 88434.0,
                "PropertyGFAParking": 0.0,
                "LargestPropertyUseTypeGFA": 88434.0,
                "NumberofBuildings": 1.0,
                "NumberofFloors": 12,
                "BuildingAge": 89.0,
                "Latitude": 47.612,
                "Longitude": -122.337,
                "PrimaryPropertyType_Clean": "Hotel",
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "predicted_energy_kbtu" in data
        assert "predicted_energy_log" in data
        assert data["predicted_energy_kbtu"] > 0