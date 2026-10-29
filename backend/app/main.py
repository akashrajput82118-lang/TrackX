from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(
    os.getenv("DATA_PATH", str(ROOT / "data" / "railway_1km_measurements.csv"))
)
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ROOT / "trained_models" / "train_condition_model.joblib"))
)
DB_PATH = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'railway_simulation.db'}")

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False} if DB_PATH.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class SimNode(Base):
    __tablename__ = "simulation_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String(36), index=True)
    node_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="WAITING_FOR_PERMISSION")
    row_index: Mapped[int] = mapped_column(Integer)
    input_json: Mapped[str] = mapped_column(Text, default="{}")
    prediction: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_key: Mapped[str] = mapped_column(String(100), unique=True, default=lambda: str(uuid.uuid4()))
    node_number: Mapped[int] = mapped_column(Integer, default=0)
    severity: Mapped[str] = mapped_column(String(32))  # "Normal Alert" or "Warning Alert"
    error_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="Real Prediction")  # "Simulation" or "Real Prediction"
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )


Base.metadata.create_all(engine)

# Seamless SQLite column migrations for existing databases
try:
    with engine.connect() as conn:
        existing_alert_cols = [
            c[1] for c in conn.exec_driver_sql("PRAGMA table_info(alerts)").fetchall()
        ]
        if existing_alert_cols:
            if "error_percentage" not in existing_alert_cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN error_percentage FLOAT DEFAULT 0.0")
            if "source" not in existing_alert_cols:
                conn.exec_driver_sql("ALTER TABLE alerts ADD COLUMN source VARCHAR(32) DEFAULT 'Real Prediction'")

        existing_sim_cols = [
            c[1] for c in conn.exec_driver_sql("PRAGMA table_info(simulation_nodes)").fetchall()
        ]
        if existing_sim_cols:
            if "error_percentage" not in existing_sim_cols:
                conn.exec_driver_sql("ALTER TABLE simulation_nodes ADD COLUMN error_percentage FLOAT")
        conn.commit()
except Exception as migration_exc:
    print(f"Database schema migration note: {migration_exc}")

app = FastAPI(
    title="TrackX Railway Monitoring API",
    description="Backend API for ML inference, condition monitoring, error percentage tracking, and alert simulation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory dataset and model references loaded once on startup
df: pd.DataFrame = pd.DataFrame()
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"Error loading CSV at {DATA_PATH}: {exc}")

loaded_model: Any = None
feature_names: List[str] = []
model_classes: List[str] = []
model_error: Optional[str] = None


def load_model():
    """Load model once from disk and inspect feature requirements and classes."""
    global loaded_model, feature_names, model_classes, model_error
    if not MODEL_PATH.exists():
        model_error = f"Trained model not found at {MODEL_PATH}. Run training script or set MODEL_PATH."
        loaded_model = None
        return

    try:
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict) and "model" in saved:
            loaded_model = saved["model"]
            feature_names = list(
                saved.get("features")
                or getattr(loaded_model, "feature_names_in_", [])
            )
            model_classes = [
                str(x) for x in saved.get("classes", getattr(loaded_model, "classes_", []))
            ]
        else:
            loaded_model = saved
            feature_names = list(getattr(loaded_model, "feature_names_in_", []))
            model_classes = [str(x) for x in getattr(loaded_model, "classes_", [])]

        if not feature_names and not df.empty:
            excluded = {"Train condition", "Crack-vicinity label"}
            feature_names = [c for c in df.columns if c not in excluded]

        model_error = None
        print(f"Model successfully loaded from {MODEL_PATH}. Features: {len(feature_names)}, Classes: {model_classes}")
    except Exception as exc:
        model_error = f"Could not load model: {exc}"
        loaded_model = None
        print(f"Failed to load model: {exc}")


load_model()


def calculate_status_and_alert(error_pct: float) -> tuple[str, Optional[str]]:
    """
    Exact threshold rules:
    - 0%–25%, inclusive: 'No Error'. Show a green indicator. Do not send an alert.
    - Greater than 25% and up to 75%, inclusive: 'Normal Alert'. Show an amber indicator and trigger a normal alert.
    - Greater than 75%: 'Warning Alert'. Show a red indicator and trigger a warning alert.

    Boundary rules:
    - Exactly 25% -> 'No Error'
    - Exactly 75% -> 'Normal Alert'
    - Values > 75% -> 'Warning Alert'
    """
    if error_pct <= 25.0:
        return "No Error", None
    elif error_pct <= 75.0:
        return "Normal Alert", "Normal Alert"
    else:
        return "Warning Alert", "Warning Alert"


def compute_error_percentage_from_probs(
    probs: Dict[str, float], predicted_label: str
) -> tuple[float, str]:
    """
    Calculate error percentage based on model probability distribution:
    - If classes match ['No train', 'Passenger train', 'Freight train']:
      'No train' is normal baseline (0% error).
      'Passenger train' is moderate error (50% scale).
      'Freight train' is severe error (100% scale).
      Error percentage = (P(Passenger train) * 50) + (P(Freight train) * 100)
    - If defective/fault class exists, use P(defective) * 100
    - If baseline/normal class exists, use (1 - P(normal)) * 100
    - If no probabilities, map predicted label: baseline->10%, moderate->50%, severe->90%
    """
    if not probs:
        label_lower = predicted_label.lower()
        if any(w in label_lower for w in ["no train", "normal", "healthy", "safe"]):
            return 10.0, "Mapped from discrete baseline prediction"
        elif any(w in label_lower for w in ["passenger", "moderate", "warning", "medium"]):
            return 50.0, "Mapped from discrete moderate condition prediction"
        else:
            return 90.0, "Mapped from discrete severe condition prediction"

    classes_lower = {k.lower(): (k, v) for k, v in probs.items()}

    # Check for railway 3-class train condition model
    if "no train" in classes_lower and ("passenger train" in classes_lower or "freight train" in classes_lower):
        p_passenger = classes_lower.get("passenger train", ("", 0.0))[1]
        p_freight = classes_lower.get("freight train", ("", 0.0))[1]
        pct = (p_passenger * 50.0) + (p_freight * 100.0)
        return round(float(pct), 1), "Calibrated probability weighting: P(Passenger)*50% + P(Freight)*100%"

    # Check for direct defect class
    for def_key in ["defect", "defective", "fault", "crack", "error", "damaged", "1"]:
        if def_key in classes_lower:
            pct = classes_lower[def_key][1] * 100.0
            return round(float(pct), 1), f"Probability of defective class '{classes_lower[def_key][0]}'"

    # Check for baseline normal class
    for norm_key in ["normal", "no error", "ok", "healthy", "0"]:
        if norm_key in classes_lower:
            pct = (1.0 - classes_lower[norm_key][1]) * 100.0
            return round(float(pct), 1), f"Complementary probability: 100% - P('{classes_lower[norm_key][0]}')"

    # Fallback to max non-first probability
    first_cls = list(probs.keys())[0]
    pct = (1.0 - probs[first_cls]) * 100.0
    return round(float(pct), 1), f"Multi-class divergence from baseline '{first_cls}'"


def row_features(row_index: int) -> tuple[Dict[str, Any], pd.Series]:
    if df.empty:
        raise HTTPException(503, "Dataset not found. Configure DATA_PATH.")
    if row_index < 0 or row_index >= len(df):
        raise HTTPException(400, f"Dataset row index {row_index} out of range [0, {len(df)-1}].")
    row = df.iloc[row_index]
    excluded = {"Train condition", "Crack-vicinity label"}
    cols = feature_names or [c for c in df.columns if c not in excluded]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(503, f"Model feature columns are missing from dataset: {missing}")

    feat_dict = {}
    for c in cols:
        val = row[c]
        feat_dict[c] = None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)
    return feat_dict, row


# Pydantic Schemas
class PredictRequest(BaseModel):
    row_index: Optional[int] = Field(None, description="Zero-based index of row from railway dataset")
    features: Optional[Dict[str, Any]] = Field(None, description="Feature dictionary matching model inputs")


class PredictResponse(BaseModel):
    prediction: str
    error_percentage: float
    status: str
    alert_type: Optional[str] = None
    probabilities: Dict[str, float]
    features: Dict[str, Any]
    calculation_method: str
    source: str = "Real Prediction"
    timestamp: str
    row_index: Optional[int] = None
    reference_label: Optional[str] = None


class AlertCreate(BaseModel):
    severity: str = Field(..., description="'Normal Alert' or 'Warning Alert'")
    error_percentage: float
    source: str = Field("Simulation", description="'Simulation' or 'Real Prediction'")
    reason: str
    node_number: Optional[int] = None


class AlertUpdate(BaseModel):
    status: str


class StartRequest(BaseModel):
    row_indices: Optional[List[int]] = None


# Endpoints
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_available": not df.empty,
        "dataset_rows": len(df),
        "model_available": loaded_model is not None,
        "model_error": model_error,
        "model_classes": model_classes,
        "model_features": feature_names,
    }


@app.get("/api/model")
def model_info():
    importance = []
    if loaded_model is not None and hasattr(loaded_model, "feature_importances_"):
        importance = [
            {"feature": name, "importance": round(float(val), 5)}
            for name, val in zip(feature_names, loaded_model.feature_importances_)
        ]
        importance.sort(key=lambda x: x["importance"], reverse=True)
    return {
        "available": loaded_model is not None,
        "error": model_error,
        "features": feature_names,
        "classes": model_classes,
        "feature_importance": importance,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if loaded_model is None:
        raise HTTPException(
            503,
            detail=model_error
            or "Trained ML model is not available. Ensure train_condition_model.joblib exists.",
        )

    row = None
    reference_label = None

    if req.row_index is not None:
        feat_dict, row = row_features(req.row_index)
        reference_label = str(row.get("Train condition", "Unknown"))
    elif req.features is not None:
        feat_dict = req.features
    else:
        # Default to first row of dataset if neither provided
        feat_dict, row = row_features(0)
        reference_label = str(row.get("Train condition", "Unknown"))

    # Validate all required features are present
    missing = [c for c in feature_names if c not in feat_dict or feat_dict[c] is None]
    if missing:
        raise HTTPException(
            422,
            detail=f"Missing required model features: {missing[:5]} (total missing: {len(missing)})",
        )

    # Build input dataframe ensuring correct column order
    X = pd.DataFrame([[feat_dict[c] for c in feature_names]], columns=feature_names)

    try:
        pred = str(loaded_model.predict(X)[0])
        probabilities: Dict[str, float] = {}
        if hasattr(loaded_model, "predict_proba"):
            raw_probs = loaded_model.predict_proba(X)[0]
            probabilities = {
                str(k): round(float(v), 4)
                for k, v in zip(loaded_model.classes_, raw_probs)
            }

        error_pct, calc_method = compute_error_percentage_from_probs(probabilities, pred)
        status, alert_type = calculate_status_and_alert(error_pct)
        iso_time = datetime.now(timezone.utc).isoformat()

        # If alert triggered, store in local alert history
        if alert_type:
            with SessionLocal() as db:
                db.add(
                    Alert(
                        node_number=0,
                        severity=alert_type,
                        error_percentage=error_pct,
                        source="Real Prediction",
                        reason=f"Model prediction '{pred}' triggered {alert_type} with error percentage {error_pct}%.",
                    )
                )
                db.commit()

        return PredictResponse(
            prediction=pred,
            error_percentage=error_pct,
            status=status,
            alert_type=alert_type,
            probabilities=probabilities,
            features=feat_dict,
            calculation_method=calc_method,
            source="Real Prediction",
            timestamp=iso_time,
            row_index=req.row_index,
            reference_label=reference_label,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, detail=f"Prediction failed: {exc}")


@app.get("/api/alerts")
def get_alerts():
    with SessionLocal() as db:
        rows = db.query(Alert).order_by(desc(Alert.created_at)).limit(50).all()
        return [
            {
                "id": a.id,
                "severity": a.severity,
                "error_percentage": a.error_percentage,
                "source": a.source,
                "reason": a.reason,
                "status": a.status,
                "node_number": a.node_number,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in rows
        ]


@app.post("/api/alerts")
def record_alert(alert_in: AlertCreate):
    with SessionLocal() as db:
        new_alert = Alert(
            severity=alert_in.severity,
            error_percentage=alert_in.error_percentage,
            source=alert_in.source,
            reason=alert_in.reason,
            node_number=alert_in.node_number or 0,
            status="OPEN",
        )
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)
        return {
            "id": new_alert.id,
            "severity": new_alert.severity,
            "error_percentage": new_alert.error_percentage,
            "source": new_alert.source,
            "reason": new_alert.reason,
            "status": new_alert.status,
            "created_at": new_alert.created_at.isoformat(),
        }


@app.patch("/api/alerts/{alert_id}")
def update_alert(alert_id: int, req: AlertUpdate):
    if req.status not in ("OPEN", "ACKNOWLEDGED", "RESOLVED"):
        raise HTTPException(400, "Invalid alert status.")
    with SessionLocal() as db:
        alert = db.query(Alert).filter_by(id=alert_id).first()
        if not alert:
            raise HTTPException(404, "Alert not found.")
        alert.status = req.status
        db.commit()
        return {"id": alert.id, "status": alert.status}


@app.delete("/api/alerts")
def clear_alerts():
    with SessionLocal() as db:
        db.query(Alert).delete()
        db.commit()
    return {"message": "Alert history cleared"}


@app.get("/api/dataset/preview")
def dataset_preview():
    if df.empty:
        return {"rows": [], "columns": [], "row_count": 0}
    clean_cols = list(df.columns)
    sample_rows = df.head(25).fillna("").to_dict(orient="records")
    return {"columns": clean_cols, "rows": sample_rows, "row_count": len(df)}


@app.get("/api/dataset/sample")
def dataset_sample():
    """Return a diverse sample of rows for testing low, moderate, and high error predictions."""
    if df.empty:
        return []
    samples = []
    # Try to pick rows from each train condition
    for cond in ["No train", "Passenger train", "Freight train"]:
        matches = df[df["Train condition"] == cond]
        if not matches.empty:
            idx = int(matches.index[0])
            feat, row = row_features(idx)
            samples.append({
                "row_index": idx,
                "condition": cond,
                "features": feat
            })
    return samples


# Node simulation endpoints (preserving existing 3-node simulation functionality)
def node_dict(n: SimNode):
    return {
        "node_number": n.node_number,
        "status": n.status,
        "row_index": n.row_index,
        "inputs": __import__("json").loads(n.input_json or "{}"),
        "prediction": n.prediction,
        "error_percentage": n.error_percentage,
        "error": n.error,
        "timestamp": n.timestamp.isoformat() if n.timestamp else None,
    }


def get_simulation(sid: str):
    with SessionLocal() as db:
        nodes = (
            db.query(SimNode)
            .filter_by(simulation_id=sid)
            .order_by(SimNode.node_number)
            .all()
        )
        if not nodes:
            raise HTTPException(404, "Simulation not found.")
        return {"simulation_id": sid, "nodes": [node_dict(n) for n in nodes]}


@app.post("/api/simulations")
def create_simulation(req: StartRequest):
    if df.empty:
        raise HTTPException(503, "Dataset unavailable.")
    rows = req.row_indices or [0, 1000, 2000]
    if len(rows) != 3 or any(i < 0 or i >= len(df) for i in rows):
        raise HTTPException(400, "Provide exactly three valid dataset row indices.")
    sid = str(uuid.uuid4())
    with SessionLocal() as db:
        for n, ridx in enumerate(rows, start=1):
            db.add(SimNode(simulation_id=sid, node_number=n, row_index=ridx))
        db.commit()
    return get_simulation(sid)


@app.get("/api/simulations/{sid}")
def simulation_status(sid: str):
    return get_simulation(sid)


@app.post("/api/simulations/{sid}/nodes/{node_number}/run")
def run_node(sid: str, node_number: int):
    if node_number not in (1, 2, 3):
        raise HTTPException(404, "Node must be 1, 2, or 3.")
    with SessionLocal() as db:
        nodes = (
            db.query(SimNode)
            .filter_by(simulation_id=sid)
            .order_by(SimNode.node_number)
            .all()
        )
        if not nodes:
            raise HTTPException(404, "Simulation not found.")
        current = next((n for n in nodes if n.node_number == node_number), None)
        if current is None:
            raise HTTPException(404, "Node not found.")
        if current.status == "COMPLETED":
            raise HTTPException(409, "Node already completed. Reset simulation to run again.")
        if current.status == "RUNNING":
            raise HTTPException(409, "Node is already running.")

        for previous in nodes:
            if previous.node_number < node_number and previous.status != "COMPLETED":
                raise HTTPException(
                    409,
                    f"Node {node_number} is blocked until Node {previous.node_number} completes.",
                )

        current.status = "RUNNING"
        db.commit()

        try:
            if loaded_model is None:
                raise HTTPException(503, model_error or "Model is not configured.")
            feat_dict, row = row_features(current.row_index)
            X = pd.DataFrame([[feat_dict[c] for c in feature_names]], columns=feature_names)
            pred = str(loaded_model.predict(X)[0])
            probs = {}
            if hasattr(loaded_model, "predict_proba"):
                probs = {
                    str(k): round(float(v), 4)
                    for k, v in zip(loaded_model.classes_, loaded_model.predict_proba(X)[0])
                }
            error_pct, _ = compute_error_percentage_from_probs(probs, pred)
            status, alert_type = calculate_status_and_alert(error_pct)

            current.input_json = __import__("json").dumps(feat_dict, default=str)
            current.prediction = pred
            current.error_percentage = error_pct
            current.status = "COMPLETED"
            current.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            current.error = None
            db.commit()

            if alert_type:
                db.add(
                    Alert(
                        node_number=node_number,
                        severity=alert_type,
                        error_percentage=error_pct,
                        source="Real Prediction",
                        reason=f"Node {node_number} model prediction '{pred}' triggered {alert_type} ({error_pct}% error).",
                    )
                )
                db.commit()

            return {
                "node": node_dict(current),
                "probabilities": probs,
                "error_percentage": error_pct,
                "status": status,
                "alert_type": alert_type,
                "reference_label": str(row.get("Train condition", "Unknown")),
            }
        except HTTPException as exc:
            current.status = "FAILED"
            current.error = str(exc.detail)
            db.commit()
            raise
        except Exception as exc:
            current.status = "FAILED"
            current.error = str(exc)
            db.commit()
            raise HTTPException(500, detail=f"Node execution failed: {exc}")


@app.post("/api/simulations/{sid}/reset")
def reset_simulation(sid: str):
    with SessionLocal() as db:
        nodes = db.query(SimNode).filter_by(simulation_id=sid).all()
        if not nodes:
            raise HTTPException(404, "Simulation not found.")
        for n in nodes:
            n.status = "WAITING_FOR_PERMISSION"
            n.prediction = None
            n.error_percentage = None
            n.error = None
            n.input_json = "{}"
            n.timestamp = None
        db.commit()
    return get_simulation(sid)


@app.get("/api/map")
def map_data():
    return {
        "city": os.getenv("MAP_CITY", "Delhi Corridor (TrackX Network)"),
        "demo_locations": True,
        "nodes": [
            {"id": 1, "name": "Node 1", "lat": 28.6139, "lng": 77.2090, "status": "WAITING_FOR_PERMISSION"},
            {"id": 2, "name": "Node 2", "lat": 28.6280, "lng": 77.2180, "status": "WAITING_FOR_PERMISSION"},
            {"id": 3, "name": "Node 3", "lat": 28.6400, "lng": 77.2300, "status": "WAITING_FOR_PERMISSION"},
        ],
        "stations": [
            {"id": "demo-station-1", "name": "Central Terminal", "lat": 28.6200, "lng": 77.2150, "demo": True},
            {"id": "demo-station-2", "name": "North Junction", "lat": 28.6350, "lng": 77.2250, "demo": True},
        ],
    }
