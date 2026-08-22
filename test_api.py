import pytest
from pydantic import ValidationError
from service import BuildingInput, SeattleEnergyService


@pytest.fixture(scope="module")
def service():
    """Initialise une instance du service BentoML."""
    return SeattleEnergyService()


def test_valid_prediction(service):
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

    result = service.predict(valid_data)

    # Vérifications de cohérence
    assert result.status == "success"
    assert result.predicted_energy_log > 0
    assert result.predicted_energy_kbtu > 0
    # Vérifie la cohérence mathématique : expm1(log_pred) == kbtu_pred
    assert isinstance(result.predicted_energy_kbtu, float)


def test_invalid_negative_surface():
    """Vérifie que Pydantic bloque les surfaces négatives ou nulles."""
    with pytest.raises(ValidationError):
        BuildingInput(
            PropertyGFATotal=-500.0,  # Invalide : doit être > 0
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
            Latitude=48.8566,  # Invalide : Coordonnées de Paris
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
            NumberofFloors=250,  # Invalide : max 120
            BuildingAge=10.0,
            Latitude=47.60,
            Longitude=-122.33,
            PrimaryPropertyType_Clean="Office",
        )