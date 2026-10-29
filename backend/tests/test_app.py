import pytest
from backend.app.main import (
    app,
    calculate_status_and_alert,
    compute_error_percentage_from_probs,
    health,
    model_info,
    predict_endpoint,
    PredictRequest,
    get_alerts,
    record_alert,
    update_alert,
    AlertCreate,
    AlertUpdate,
    dataset_sample
)


def test_status_values_and_boundaries():
    """
    Test exact threshold rules:
    - 0%–25%, inclusive: No Error
    - Greater than 25% and up to 75%, inclusive: Normal Alert
    - Greater than 75%: Warning Alert
    """
    # 0% - 25% inclusive -> No Error
    assert calculate_status_and_alert(0.0) == ("No Error", None)
    assert calculate_status_and_alert(10.0) == ("No Error", None)
    assert calculate_status_and_alert(24.9) == ("No Error", None)
    assert calculate_status_and_alert(25.0) == ("No Error", None)  # Exact boundary: 25% is No Error

    # > 25% up to 75% inclusive -> Normal Alert
    assert calculate_status_and_alert(25.01) == ("Normal Alert", "Normal Alert")
    assert calculate_status_and_alert(50.0) == ("Normal Alert", "Normal Alert")
    assert calculate_status_and_alert(74.9) == ("Normal Alert", "Normal Alert")
    assert calculate_status_and_alert(75.0) == ("Normal Alert", "Normal Alert")  # Exact boundary: 75% is Normal Alert

    # > 75% -> Warning Alert
    assert calculate_status_and_alert(75.01) == ("Warning Alert", "Warning Alert")
    assert calculate_status_and_alert(90.0) == ("Warning Alert", "Warning Alert")
    assert calculate_status_and_alert(100.0) == ("Warning Alert", "Warning Alert")


def test_health_endpoint():
    data = health()
    assert data["status"] == "ok"
    assert data["dataset_available"] is True
    assert data["model_available"] is True
    assert "Freight train" in data["model_classes"]
    assert "No train" in data["model_classes"]
    assert "Passenger train" in data["model_classes"]
    assert len(data["model_features"]) == 20


def test_model_endpoint():
    data = model_info()
    assert data["available"] is True
    assert len(data["features"]) == 20
    assert len(data["feature_importance"]) > 0


def test_predict_with_row_indices():
    # Row 0: No train -> <= 25% (No Error)
    p0 = predict_endpoint(PredictRequest(row_index=0))
    assert p0.prediction == "No train"
    assert p0.error_percentage <= 25.0
    assert p0.status == "No Error"
    assert p0.alert_type is None
    assert p0.source == "Real Prediction"

    # Row 1000: Passenger train -> > 25% and <= 75% (Normal Alert)
    p1 = predict_endpoint(PredictRequest(row_index=1000))
    assert p1.prediction == "Passenger train"
    assert 25.0 < p1.error_percentage <= 75.0
    assert p1.status == "Normal Alert"
    assert p1.alert_type == "Normal Alert"

    # Row 2000: Freight train -> > 75% (Warning Alert)
    p2 = predict_endpoint(PredictRequest(row_index=2000))
    assert p2.prediction == "Freight train"
    assert p2.error_percentage > 75.0
    assert p2.status == "Warning Alert"
    assert p2.alert_type == "Warning Alert"


def test_predict_with_features_dict():
    samples = dataset_sample()
    assert len(samples) > 0
    first_feat = samples[0]["features"]
    p = predict_endpoint(PredictRequest(features=first_feat))
    assert p.prediction in ["No train", "Passenger train", "Freight train"]
    assert isinstance(p.error_percentage, (int, float))


def test_predict_missing_features():
    with pytest.raises(Exception):
        predict_endpoint(PredictRequest(features={"Distance from Node A (m)": 500}))


def test_alert_lifecycle():
    # Record simulation alert
    created = record_alert(
        AlertCreate(
            severity="Normal Alert",
            error_percentage=50.0,
            source="Simulation",
            reason="Simulation Scenario 2 triggered Normal Alert",
            node_number=1,
        )
    )
    assert created["id"] is not None
    assert created["severity"] == "Normal Alert"
    assert created["source"] == "Simulation"
    alert_id = created["id"]

    # Verify alert appears in alert history
    alerts = get_alerts()
    assert any(a["id"] == alert_id for a in alerts)

    # Update alert status
    updated = update_alert(alert_id, AlertUpdate(status="RESOLVED"))
    assert updated["status"] == "RESOLVED"
