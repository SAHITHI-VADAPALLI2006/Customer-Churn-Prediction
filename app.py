# app.py
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model and scaler
# -------------------------------
rf = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("Customer Churn Prediction App")
st.write("Please enter customer details below:")

# -------------------------------
# Input form
# -------------------------------
with st.form(key='churn_form'):
    # Binary features
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

    # Categorical features
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    # Numeric features with range hints
    tenure = st.number_input("Tenure (months) [Range: 0 - 72]", min_value=0, max_value=72, value=0)
    MonthlyCharges = st.number_input("Monthly Charges ($) [Range: 18.0 - 120.0]", min_value=18.0, max_value=120.0, value=50.0)
    TotalCharges = st.number_input("Total Charges ($) [Range: 18.0 - 8000.0]", min_value=18.0, max_value=8000.0, value=500.0)

    submit_button = st.form_submit_button(label="Predict Churn")

# -------------------------------
# Mapping input to model format
# -------------------------------
if submit_button:
    input_dict = {
        "gender": 1 if gender == "Female" else 0,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "MultipleLines": 1 if MultipleLines == "Yes" else 0,
        "OnlineSecurity": 1 if OnlineSecurity == "Yes" else 0,
        "OnlineBackup": 1 if OnlineBackup == "Yes" else 0,
        "DeviceProtection": 1 if DeviceProtection == "Yes" else 0,
        "TechSupport": 1 if TechSupport == "Yes" else 0,
        "StreamingTV": 1 if StreamingTV == "Yes" else 0,
        "StreamingMovies": 1 if StreamingMovies == "Yes" else 0,
        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
        "Contract": {"Month-to-month":0, "One year":1, "Two year":2}[Contract],
        "InternetService": {"DSL":0, "Fiber optic":1, "No":2}[InternetService],
        "PaymentMethod": {"Electronic check":0, "Mailed check":1, "Bank transfer":2, "Credit card":3}[PaymentMethod],
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Convert to DataFrame
    X_input = pd.DataFrame([input_dict])

    # 🔑 Ensure correct column order for scaler
    X_input = X_input[scaler.feature_names_in_]

    # Scale numeric features
    X_scaled = scaler.transform(X_input)

    # Predict
    prediction = rf.predict(X_scaled)[0]
    probability = rf.predict_proba(X_scaled)[0][1]

    # Display result
    st.subheader("Prediction Result")
    st.write("Churn:", "Yes" if prediction == 1 else "No")
    st.write(f"Probability of Churn: {probability*100:.2f}%")
