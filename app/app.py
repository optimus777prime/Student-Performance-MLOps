from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "student_model.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    return model, encoders


model, encoders = load_artifacts()

st.title("🎓 Student Performance Predictor")
st.caption("Estimate whether a student is likely to pass based on the supplied details and scores.")
st.divider()

st.sidebar.header("📋 Student Details")
gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
race = st.sidebar.selectbox("Race / Ethnicity", encoders["race/ethnicity"].classes_)
education = st.sidebar.selectbox(
    "Parental Education", encoders["parental level of education"].classes_
)
lunch = st.sidebar.selectbox("Lunch", encoders["lunch"].classes_)
prep = st.sidebar.selectbox(
    "Test Preparation", encoders["test preparation course"].classes_
)
math = st.sidebar.slider("Math Score", 0, 100, 50)
reading = st.sidebar.slider("Reading Score", 0, 100, 50)
writing = st.sidebar.slider("Writing Score", 0, 100, 50)

st.subheader("📊 Student Score Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Math", math)
col2.metric("Reading", reading)
col3.metric("Writing", writing)

average = (math + reading + writing) / 3
st.metric("Average Score", f"{average:.2f}")

student_details = pd.DataFrame(
    {
        "Gender": [gender],
        "Race / Ethnicity": [race],
        "Parental Education": [education],
        "Lunch": [lunch],
        "Test Preparation": [prep],
    }
)
st.subheader("📋 Student Details")
st.dataframe(student_details, use_container_width=True, hide_index=True)

score_df = pd.DataFrame(
    {"Subject": ["Math", "Reading", "Writing"], "Score": [math, reading, writing]}
)
st.subheader("📈 Score Visualization")
st.bar_chart(score_df.set_index("Subject"))

if st.button("Predict", type="primary"):
    data = pd.DataFrame(
        {
            "gender": [encoders["gender"].transform([gender])[0]],
            "race/ethnicity": [encoders["race/ethnicity"].transform([race])[0]],
            "parental level of education": [
                encoders["parental level of education"].transform([education])[0]
            ],
            "lunch": [encoders["lunch"].transform([lunch])[0]],
            "test preparation course": [
                encoders["test preparation course"].transform([prep])[0]
            ],
            "math score": [math],
            "reading score": [reading],
            "writing score": [writing],
        }
    )

    prediction = model.predict(data)
    probabilities = model.predict_proba(data)[0]
    confidence = probabilities.max() * 100
    result = encoders["result"].inverse_transform(prediction)[0]

    if result == "Pass":
        st.success("🎉 Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

    st.progress(int(confidence))
    st.write(f"### Prediction confidence: {confidence:.2f}%")
