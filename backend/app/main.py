from __future__ import annotations
import os, math, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", str(ROOT / "data" / "railway_1km_measurements.csv")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "trained_models" / "train_condition_model.joblib")))
DB_PATH = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{ROOT / 'railway_simulation.db'}"
)

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False}
    if DB_PATH.startswith("sqlite")
    else {}
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase): pass
class SimNode(Base):
    __tablename__ = "simulation_nodes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String(36), index=True)
    node_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="WAITING_FOR_PERMISSION")
    row_index: Mapped[int] = mapped_column(Integer)
    input_json: Mapped[str] = mapped_column(Text, default="{}")
    prediction: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_key: Mapped[str] = mapped_column(String(100), unique=True)
    node_number: Mapped[int] = mapped_column(Integer)
    severity: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

Base.metadata.create_all(engine)
app = FastAPI(title="Railway Track Monitoring API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
loaded_model: Any = None
feature_names: list[str] = []
model_classes: list[str] = []
model_error: str | None = None

def load_model():
    global loaded_model, feature_names, model_classes, model_error
    if not MODEL_PATH.exists():
        model_error = f"Trained model not found at {MODEL_PATH}. Copy your existing .joblib model here or set MODEL_PATH."
        return
    try:
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict) and "model" in saved:
            loaded_model = saved["model"]
            feature_names = list(saved.get("features") or getattr(loaded_model, "feature_names_in_", []))
            model_classes = [str(x) for x in saved.get("classes", getattr(loaded_model, "classes_", []))]
        else:
            loaded_model = saved
            feature_names = list(getattr(loaded_model, "feature_names_in_", []))
            model_classes = [str(x) for x in getattr(loaded_model, "classes_", [])]
        if not feature_names and not df.empty:
            model_error = "Model loaded but no feature metadata is available. Save a model dictionary with a 'features' list."
            loaded_model = None
    except Exception as exc:
        model_error = f"Could not load model: {exc}"
        loaded_model = None
load_model()

def now_iso(): return datetime.now(timezone.utc).isoformat()
def row_features(row_index: int):
    if df.empty: raise HTTPException(503, "Dataset not found. Configure DATA_PATH.")
    if row_index < 0 or row_index >= len(df): raise HTTPException(400, "Dataset row index out of range.")
    row = df.iloc[row_index]
    excluded = {"Train condition", "Crack-vicinity label"}
    cols = feature_names or [c for c in df.columns if c not in excluded]
    missing = [c for c in cols if c not in df.columns]
    if missing: raise HTTPException(503, f"Model feature columns are missing from dataset: {missing}")
    return {c: (None if pd.isna(row[c]) else row[c].item() if hasattr(row[c], "item") else row[c]) for c in cols}, row

def predict(row_index: int):
    if loaded_model is None:
        raise HTTPException(503, model_error or "Model is not configured.")
    features, row = row_features(row_index)
    X = pd.DataFrame([[features[c] for c in feature_names]], columns=feature_names)
    try:
        result = str(loaded_model.predict(X)[0])
        probabilities = {}
        if hasattr(loaded_model, "predict_proba"):
            probabilities = {str(k): round(float(v), 4) for k, v in zip(loaded_model.classes_, loaded_model.predict_proba(X)[0])}
        return result, probabilities, features, row
    except Exception as exc:
        raise HTTPException(422, f"Model prediction failed: {exc}")

class StartRequest(BaseModel):
    row_indices: list[int] | None = None
class AlertUpdate(BaseModel):
    status: str

@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_available": not df.empty, "dataset_rows": len(df), "model_available": loaded_model is not None, "model_error": model_error, "model_classes": model_classes, "model_features": feature_names}

@app.get("/api/model")
def model_info():
    importance = []
    if loaded_model is not None and hasattr(loaded_model, "feature_importances_"):
        importance = [{"feature": name, "importance": round(float(val), 5)} for name, val in zip(feature_names, loaded_model.feature_importances_)]
        importance.sort(key=lambda x: x["importance"], reverse=True)
    return {"available": loaded_model is not None, "error": model_error, "features": feature_names, "classes": model_classes, "feature_importance": importance}

@app.post("/api/simulations")
def create_simulation(req: StartRequest):
    if df.empty: raise HTTPException(503, "Dataset unavailable.")
    rows = req.row_indices or [0, 1, 2]
    if len(rows) != 3 or any(i < 0 or i >= len(df) for i in rows):
        raise HTTPException(400, "Provide exactly three valid dataset row indices.")
    sid = str(uuid.uuid4())
    with SessionLocal() as db:
        for n, ridx in enumerate(rows, start=1):
            db.add(SimNode(simulation_id=sid, node_number=n, row_index=ridx))
        db.commit()
    return get_simulation(sid)

def get_simulation(sid: str):
    with SessionLocal() as db:
        nodes = db.query(SimNode).filter_by(simulation_id=sid).order_by(SimNode.node_number).all()
        if not nodes: raise HTTPException(404, "Simulation not found.")
        return {"simulation_id": sid, "nodes": [node_dict(n) for n in nodes]}

def node_dict(n):
    return {"node_number": n.node_number, "status": n.status, "row_index": n.row_index, "inputs": __import__("json").loads(n.input_json or "{}"), "prediction": n.prediction, "error": n.error, "timestamp": n.timestamp.isoformat() if n.timestamp else None}

@app.get("/api/simulations/{sid}")
def simulation_status(sid: str): return get_simulation(sid)

@app.post("/api/simulations/{sid}/nodes/{node_number}/run")
def run_node(sid: str, node_number: int):
    if node_number not in (1, 2, 3): raise HTTPException(404, "Node must be 1, 2, or 3.")
    with SessionLocal() as db:
        nodes = db.query(SimNode).filter_by(simulation_id=sid).order_by(SimNode.node_number).all()
        if not nodes: raise HTTPException(404, "Simulation not found.")
        current = next((n for n in nodes if n.node_number == node_number), None)
        if current is None: raise HTTPException(404, "Node not found.")
        if current.status == "COMPLETED": raise HTTPException(409, "Node already completed. Reset simulation to run again.")
        if current.status == "RUNNING": raise HTTPException(409, "Node is already running.")
        # Enforce explicit, sequential authorization server-side.
        for previous in nodes:
            if previous.node_number < node_number and previous.status != "COMPLETED":
                raise HTTPException(409, f"Node {node_number} is blocked until Node {previous.node_number} completes.")
        current.status = "RUNNING"; db.commit()
        try:
            pred, probs, features, row = predict(current.row_index)
            current.input_json = __import__("json").dumps(features, default=str)
            current.prediction = pred
            current.status = "COMPLETED"
            current.timestamp = datetime.utcnow()
            current.error = None
            db.commit()
            # A train class is not a defect prediction; no automatic defect alert is inferred from it.
            return {"node": node_dict(current), "probabilities": probs, "reference_label": str(row.get("Train condition", "Unknown")), "crack_vicinity_label": str(row.get("Crack-vicinity label", "Unavailable")), "alert_note": "Train-condition output is not evidence of a track defect. Configure verified track coordinates and a validated defect model/rule for risk alerts."}
        except HTTPException as exc:
            current.status = "FAILED"; current.error = str(exc.detail); db.commit()
            raise
        except Exception as exc:
            current.status = "FAILED"; current.error = str(exc); db.commit()
            raise HTTPException(500, "Node execution failed.")

@app.post("/api/simulations/{sid}/reset")
def reset_simulation(sid: str):
    with SessionLocal() as db:
        nodes = db.query(SimNode).filter_by(simulation_id=sid).all()
        if not nodes: raise HTTPException(404, "Simulation not found.")
        for n in nodes:
            n.status="WAITING_FOR_PERMISSION"; n.prediction=None; n.error=None; n.input_json="{}"; n.timestamp=None
        db.commit()
    return get_simulation(sid)

@app.get("/api/alerts")
def get_alerts():
    with SessionLocal() as db:
        rows = db.query(Alert).order_by(Alert.created_at.desc()).all()
        return [{"id": a.id, "severity": a.severity, "reason": a.reason, "status": a.status, "node_number": a.node_number, "created_at": a.created_at.isoformat()} for a in rows]

@app.patch("/api/alerts/{alert_id}")
def update_alert(alert_id: int, req: AlertUpdate):
    if req.status not in ("OPEN", "ACKNOWLEDGED", "RESOLVED"): raise HTTPException(400, "Invalid alert status.")
    with SessionLocal() as db:
        alert = db.query(Alert).filter_by(id=alert_id).first()
        if not alert: raise HTTPException(404, "Alert not found.")
        alert.status=req.status; db.commit()
        return {"id": alert.id, "status": alert.status}

@app.get("/api/dataset/preview")
def dataset_preview():
    if df.empty: return {"rows": [], "columns": []}
    return {"columns": list(df.columns), "rows": df.head(20).fillna("").to_dict(orient="records"), "row_count": len(df)}

@app.get("/api/map")
def map_data():
    # Explicitly demo schematic positions; replace with verified coordinates before real use.
    return {"city": os.getenv("MAP_CITY", "Delhi (demo view)"), "demo_locations": True,
      "nodes": [
        {"id": 1, "name": "Node 1", "lat": 28.6139, "lng": 77.2090, "status": "WAITING_FOR_PERMISSION"},
        {"id": 2, "name": "Node 2", "lat": 28.6280, "lng": 77.2180, "status": "WAITING_FOR_PERMISSION"},
        {"id": 3, "name": "Node 3", "lat": 28.6400, "lng": 77.2300, "status": "WAITING_FOR_PERMISSION"}],
      "stations": [
        {"id": "demo-station-1", "name": "Demo Station A", "lat": 28.6200, "lng": 77.2150, "demo": True},
        {"id": "demo-station-2", "name": "Demo Station B", "lat": 28.6350, "lng": 77.2250, "demo": True}]}
