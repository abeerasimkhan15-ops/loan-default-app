
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('loan_model.pkl')
with open('feature_names.json') as f:
    features = json.load(f)

st.title("🏦 Loan Default Prediction")
st.write("Enter applicant details to predict default risk")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income", 10000, 500000, 50000)
loan_amount = st.number_input("Loan Amount", 1000, 200000, 10000)
credit_score = st.slider("Credit Score", 300, 850, 650)
interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 8.0)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
months_employed = st.number_input("Months Employed", 0, 300, 24)
num_credit_lines = st.slider("Number of Credit Lines", 1, 10, 3)
education = st.selectbox("Education", [0, 1, 2, 3])
employment_type = st.selectbox("Employment Type", [0, 1, 2, 3])
marital_status = st.selectbox("Marital Status", [0, 1, 2])
has_mortgage = st.selectbox("Has Mortgage", [0, 1])
has_dependents = st.selectbox("Has Dependents", [0, 1])
loan_purpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4])
has_cosigner = st.selectbox("Has Co-Signer", [0, 1])

loan_to_income = loan_amount / income
income_per_month = income / 12
credit_per_age = credit_score / age
employment_stability = months_employed / 12

input_data = pd.DataFrame([[age, income, loan_amount, credit_score,
    months_employed, num_credit_lines, interest_rate, loan_term,
    dti_ratio, education, employment_type, marital_status,
    has_mortgage, has_dependents, loan_purpose, has_cosigner,
    loan_to_income, income_per_month, credit_per_age, employment_stability]],
    columns=features)

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    st.subheader("Result")
    if prob > 0.5:
        st.error(f"⚠️ High Default Risk: {prob*100:.1f}%")
    else:
        st.success(f"✅ Low Default Risk: {prob*100:.1f}%")
    
    st.subheader("🔍 Explanation (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data.iloc[0],
        feature_names=features
    ), show=False)
    st.pyplot(fig)
