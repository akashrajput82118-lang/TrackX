# TrackX — Railway Track Monitoring, ML Model Integration & Simulation

TrackX is a modern, light-themed full-stack railway track monitoring platform built with **FastAPI**, **React (Vite)**, **scikit-learn**, and **SQLite**. It features continuous error-percentage defect assessment, calibrated ML inference, an automated 3-scenario testing simulation, and sequential 3-node monitoring.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │            TrackX Frontend            │
                      │     (React 18 + Vite + Leaflet)       │
                      │       Light-Themed UI @ :5173         │
                      └──────────────────┬────────────────────┘
                                         │
                   REST API Calls        │  Environment variable:
             (GET /health, POST /predict)│  VITE_API_BASE_URL
                                         ▼
                      ┌───────────────────────────────────────┐
                      │          FastAPI Backend              │
                      │          (Python 3.14 @ :8000)        │
                      └───────────┬───────────────┬───────────┘
                                  │               │
               Loads once         │               │  SQLite persistence
               on startup         ▼               ▼  (Alerts, SimNodes)
                     ┌──────────────────┐   ┌───────────────────────────┐
                     │  Trained Model   │   │   railway_simulation.db   │
                     │  (RandomForest)  │   │  (Schema auto-migrated)   │
                     │  .joblib 2.4 MB  │   └───────────────────────────┘
                     └─────────┬────────┘
                               │ Trained from
                               ▼
                     ┌───────────────────────────────────┐
                     │ data/railway_1km_measurements.csv │
                     │   (3,000 records, 20 features)    │
                     └───────────────────────────────────┘
```

---

## Key Features

1. **Continuous Error Percentage Monitoring**:
   - Replaced old three-condition classification UI with a prominent **Error Percentage (XX%)** display and visual progress gauge.
   - **Threshold Rules**:
     - **0%–25% (inclusive)**: `No Error` (Green indicator, no alert).
     - **> 25% and &le; 75% (inclusive)**: `Normal Alert` (Amber indicator, triggers normal alert).
     - **> 75%**: `Warning Alert` (Red indicator, triggers warning alert).
   - Exact boundary handling: `25.0%` is `No Error`; `75.0%` is `Normal Alert`; `> 75.0%` is `Warning Alert`.

2. **Automated 1-Button Simulation Cycle**:
   - Prominent **Start Simulation** button triggers an automatic continuous cycle with **exactly 10 seconds** between scenarios:
     - **Scenario 1 — Low Error**: `10%` &rarr; `No Error` (Green, no alert) &rarr; wait 10s.
     - **Scenario 2 — Moderate Error**: `50%` &rarr; `Normal Alert` (Amber, triggers normal alert) &rarr; wait 10s.
     - **Scenario 3 — High Error**: `90%` &rarr; `Warning Alert` (Red, triggers warning alert) &rarr; wait 10s.
     - Cycles back to Scenario 1 until **Stop Simulation** is clicked.
   - One managed timer with cleanup on unmount; prevents duplicate timers; displays scenario timestamps and countdown.

3. **Separation of Modes**:
   - **Simulation Mode**: Clearly marked simulated data for testing UI and thresholds.
   - **Real Prediction Mode**: Passes genuine 20-feature measurements from `data/railway_1km_measurements.csv` to the trained Random Forest model.

4. **Trained ML Model & Calibrated Error Percentage**:
   - Model artifact: `trained_models/train_condition_model.joblib` (96.3% accuracy, 100 estimators).
   - Loaded once into memory on backend startup.
   - Computes error percentage from class probability distribution:
     $$\text{Error Percentage} = (P(\text{Passenger train}) \times 50\%) + (P(\text{Freight train}) \times 100\%)$$
   - Documented, transparent calculation reflecting heavy axle load and track stress.

5. **Alert System & History**:
   - Real-time audit log recording timestamp, error percentage, severity, source (Simulation vs Real Prediction), and resolution status.
   - Browser notification support with graceful fallback when denied.

6. **Clean Light Theme**:
   - Crisp white/light canvas (`#f8fafc`), subtle borders (`#e2e8f0`), dark readable typography (`#0f172a`), blue accents (`#2563eb`), and accessible contrast across all sections.

---

## Quickstart (Windows PowerShell)

### Prerequisites
- **Python 3.10+** (Python 3.14 tested)
- **Node.js 18+** & npm (Node v22 tested)

### 1. Install Dependencies & Train/Verify Model
Open PowerShell in the project root:

```powershell
# 1. Install backend dependencies
python -m pip install -r backend/requirements.txt

# 2. Train and serialize the Random Forest model (if not already trained)
python trackXModel/train_model.py

# 3. Install frontend dependencies
npm.cmd install
```

### 2. Run the Backend API
In terminal 1:

```powershell
python -m uvicorn backend.app.main:app --port 8000 --host 127.0.0.1 --reload
```
- API Base URL: **http://127.0.0.1:8000**
- Interactive Swagger Docs: **http://127.0.0.1:8000/docs**
- Health Check: **http://127.0.0.1:8000/api/health**

### 3. Run the Frontend Website
In terminal 2:

```powershell
npm.cmd run dev
```
- Website URL: **http://127.0.0.1:5173**

---

## Running Automated Backend Tests

```powershell
python -m pytest backend/tests/test_app.py -v
```

All 7 test cases verify:
1. Exact threshold boundaries (`25.0%` = No Error, `75.0%` = Normal Alert, `> 75%` = Warning Alert).
2. Health and model availability.
3. Model metadata and feature importance.
4. Real predictions with dataset rows.
5. Real predictions with custom feature dictionaries.
6. Validation for missing input features (HTTP 422).
7. Full alert lifecycle (recording, fetching, resolving).

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Backend and model availability, dataset rows, classes, features. |
| `GET` | `/api/model` | Feature list, classes, and global feature importance. |
| `POST` | `/api/predict` | Executes inference with `row_index` or feature dictionary. Returns `error_percentage`, `status`, and probabilities. |
| `GET` | `/api/alerts` | Returns recent alerts ordered by timestamp descending. |
| `POST` | `/api/alerts` | Logs a new alert (used by simulation or manual alerts). |
| `PATCH` | `/api/alerts/{id}` | Updates alert status (`OPEN`, `ACKNOWLEDGED`, `RESOLVED`). |
| `DELETE` | `/api/alerts` | Clears all recorded alerts. |
| `GET` | `/api/dataset/preview` | Returns table columns and sample rows from the CSV dataset. |
| `GET` | `/api/dataset/sample` | Returns diverse test rows matching condition classes. |
| `POST` | `/api/simulations` | Creates a sequential 3-node simulation session. |
| `POST` | `/api/simulations/{id}/nodes/{n}/run` | Sequentially executes node inference with server-side authorization check. |
| `POST` | `/api/simulations/{id}/reset` | Resets a 3-node simulation session. |
| `GET` | `/api/map` | Node and station geo coordinates for Leaflet corridor map. |
