import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/student_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("### Predict whether a student will **PASS** or **FAIL**")
st.markdown("---")

# User Inputs
st.sidebar.header("📋 Student Details")

gender = st.sidebar.selectbox(
    "Gender",
    encoders["gender"].classes_
)

race = st.sidebar.selectbox(
    "Race / Ethnicity",
    encoders["race/ethnicity"].classes_
)

education = st.sidebar.selectbox(
    "Parental Education",
    encoders["parental level of education"].classes_
)

lunch = st.sidebar.selectbox(
    "Lunch",
    encoders["lunch"].classes_
)

prep = st.sidebar.selectbox(
    "Test Preparation",
    encoders["test preparation course"].classes_
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

st.metric("Average Score", round(average, 2))

if st.button("Predict"):

    data = pd.DataFrame({
        "gender": [encoders["gender"].transform([gender])[0]],
        "race/ethnicity": [encoders["race/ethnicity"].transform([race])[0]],
        "parental level of education": [encoders["parental level of education"].transform([education])[0]],
        "lunch": [encoders["lunch"].transform([lunch])[0]],
        "test preparation course": [encoders["test preparation course"].transform([prep])[0]],
        "math score": [math],
        "reading score": [reading],
        "writing score": [writing]
    })

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0]
    confidence = max(probability) * 100

    result = encoders["result"].inverse_transform(prediction)

    if result[0] == "Pass":
        st.success("🎉 Student is likely to PASS")
    else:
        st.error("❌ Student is likely to FAIL")

    st.progress(confidence / 100)
    st.write(f"### Confidence: {confidence:.2f}%")

    st.subheader("📋 Student Details")

st.write(f"**Gender:** {gender}")
st.write(f"**Race:** {race}")
st.write(f"**Parental Education:** {education}")
st.write(f"**Lunch:** {lunch}")
st.write(f"**Test Preparation:** {prep}")

score_df = pd.DataFrame({
    "Subject": ["Math", "Reading", "Writing"],
    "Score": [math, reading, writing]
})

st.subheader("📊 Scores")
st.dataframe(score_df, use_container_width=True)

st.subheader("📈 Score Visualization")
st.bar_chart(score_df.set_index("Subject"))