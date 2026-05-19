import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🧠", layout="centered")

st.title("🧠 Diabetes Prediction using SVM")
st.markdown("Fill the patient details below to predict diabetes risk.")

# ---------------------------
# LOAD DATASET
# ---------------------------
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------------------
# SCALING + MODEL TRAINING
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel="linear", probability=True)
model.fit(X_scaled, y)

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("Patient Details")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

# ---------------------------
# PREDICTION INPUT
# ---------------------------
input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

input_scaled = scaler.transform(input_data)

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("🔍 Predict Diabetes"):

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error("🔴 High Risk of Diabetes Detected")
    else:
        st.success("🟢 No Diabetes Detected")

    st.write(f"Probability of Diabetes: **{probability:.2f}**")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and SVM")