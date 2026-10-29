# Railway Track Monitoring & 3-Node Simulation

A React + FastAPI + SQLite starter application for the railway monitoring project. It supports a three-node, manually authorized simulation, dataset-backed node parameters, an interactive OpenStreetMap map, and station proximity alerts.

## Important model note
The CSV dataset is included, but your previously trained `.joblib` model was not available in the uploaded files when this project was generated. The backend will report `model_available: false` until you copy your existing model into `trained_models/` or configure `MODEL_PATH`. It will **not invent predictions**. The frontend will still run and show this status clearly.

The CSV uses `Train condition` as its target. A train-condition prediction is not proof of a track defect. The app keeps defect alerts separate and labels manually configured demo alerts as simulated.

## Folder layout
- `frontend/`: React + Vite UI
- `backend/app/`: FastAPI API, model integration, SQLite persistence
- `data/railway_1km_measurements.csv`: supplied dataset
- `trained_models/`: copy your existing model here

## 1. Backend (Windows PowerShell)
From the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item .\data\railway_1km_measurements.csv .\backend\railway_1km_measurements.csv
$env:DATA_PATH = "$PWD\data\railway_1km_measurements.csv"
# Optional: set this to your existing trained model file
$env:MODEL_PATH = "$PWD\trained_models\train_condition_model.joblib"
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

If Python 3.14 is not installed, use `py -0p` and substitute an installed version such as `py -3.13`.

API docs: http://127.0.0.1:8000/docs

## 2. Frontend
Open a second terminal in the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, usually http://localhost:5173.

## Model file format
The backend supports either:
1. a scikit-learn estimator saved directly with `joblib.dump(model, path)`, or
2. a dictionary containing `model` and optionally `features` and `classes`, like:
   `{"model": model, "features": feature_names, "classes": model.classes_.tolist()}`.

The API verifies the saved feature list against the CSV before prediction. If your model expects a different target or feature schema, configure/adapt the backend service instead of silently changing feature order.

## Map
The map uses OpenStreetMap tiles and a configurable default center (Delhi only as a demo view, not as your current location). Station locations shown in the starter are explicitly marked as demo locations. Replace them with verified station coordinates before using them for anything beyond demonstration.

## Tests
```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
```

This is a simulation/demo application, not a railway safety system. It does not connect to real sensors or send alerts to railway authorities.
