
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "trained_models" / "train_condition_model.joblib"
DATA_PATH = ROOT / "data" / "railway_1km_measurements.csv"

st.set_page_config(
    page_title="TrackX ML Dashboard",
    page_icon="🚆",
    layout="wide"
)

st.title("🚆 TrackX - Railway Track Monitoring")
st.write("AI-based Railway Track Condition Prediction")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

# Load trained model
try:
    payload = load_model()
    model = payload["model"]
    features = payload["features"]

    st.success("ML Model Loaded Successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Dataset information
df = load_data()

if df is not None:
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(features))

    with st.expander("View Dataset"):
        st.dataframe(df.head(20), use_container_width=True)

st.divider()

# Prediction section
st.subheader("Predict Track Condition")

st.write("Enter sensor measurements:")

defaults = {}

if df is not None:
    for feature in features:
        if feature in df.columns:
            defaults[feature] = pd.to_numeric(
                df[feature], errors="coerce"
            ).median()

with st.form("prediction_form"):

    values = {}

    columns = st.columns(2)

    for i, feature in enumerate(features):

        with columns[i % 2]:
            default = defaults.get(feature, 0.0)

            values[feature] = st.number_input(
                feature,
                value=float(default),
                format="%.4f"
            )

    submit = st.form_submit_button("Predict Track Condition")

if submit:

    input_data = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Condition",
        str(prediction)
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        result = pd.DataFrame({
            "Condition": model.classes_,
            "Probability": probabilities
        })

        st.write("Prediction Probabilities")

        st.bar_chart(
            result.set_index("Condition")
        )

        st.metric(
            "Prediction Confidence",
            f"{max(probabilities) * 100:.2f}%"
        )

        st.caption(
            "Model confidence is not the same as a measured physical defect percentage."
        )

st.divider()

st.caption("TrackX - AI Railway Monitoring System")
