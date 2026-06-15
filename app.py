import streamlit as st
import joblib
import numpy as np
import os

# Debug Information
st.write("Current Folder:", os.getcwd())
st.write("Available Files:", os.listdir())

# Load Model and Encoders
try:
    model = joblib.load("salary_model.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    education_encoder = joblib.load("education_encoder.pkl")
    job_encoder = joblib.load("job_encoder.pkl")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Title
st.title("💰 Salary Prediction App")

st.write("Enter employee details to predict salary")

# Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

education = st.selectbox(
    "Education Level",
    education_encoder.classes_
)

job = st.selectbox(
    "Job Title",
    job_encoder.classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.0
)

# Prediction Button
if st.button("Predict Salary"):

    gender_encoded = gender_encoder.transform([gender])[0]

    education_encoded = education_encoder.transform([education])[0]

    job_encoded = job_encoder.transform([job])[0]

    input_data = np.array([
        [
            age,
            gender_encoded,
            education_encoded,
            job_encoded,
            experience
        ]
    ])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: ₹ {prediction[0]:,.0f}"
    )