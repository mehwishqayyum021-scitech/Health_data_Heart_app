import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. Page Setup & Sidebar Research Credits
st.set_page_config(page_title="Heart Risk CDSS", layout="wide")

with st.sidebar:
    st.title("Clinical Metadata")
    st.info("""
    **Methodology:** Uses **MICE** (IterativeImputer) to handle missing values. 
    **Researcher:** Mehwish Qayyum, Pharmacist.
    **Data:** UCI Heart Disease Dataset (via Redwan Karim Sony).
    """)
    st.warning("**Disclaimer:** For educational/research use only. Not for clinical diagnosis.")

# 2. Load the Model & Create Imputer Live
@st.cache_resource
def load_assets():
    # Load model and column list
    model = joblib.load('heart_model1.joblib')
    model_columns = joblib.load('model_columns1.joblib')
    # Create the imputer LIVE to avoid version errors
    train_df = pd.read_csv('train_data.csv')
    imputer = IterativeImputer(random_state=42, max_iter=10)
    imputer.fit(train_df)
    return model, imputer, model_columns

try:
    model, mice_imputer, model_columns = load_assets()
except Exception as e:
    st.error(f"Configuration Error: {e}. Please ensure 'train_data.csv' is uploaded to GitHub.")
    st.stop()

st.title("Heart Disease Risk Predictor")
st.write("Professional Clinical Decision Support Tool (MICE Imputed)")

# 3. Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("Resting BP (mm Hg)", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", [True, False])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t wave abnormality"])
    thalch = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Induced Angina", [True, False])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 0.0)

# 4. Prediction Logic
if st.button("Analyze Risk"):
    # Create initial dataframe from inputs
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak
    }])

    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # Reindex to match training columns
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    final_input = final_input.astype(float)

    # Apply MICE imputation (Live)
    final_input_imputed = mice_imputer.transform(final_input)

    # Predict
    prediction = model.predict(final_input_imputed)
    probability = model.predict_proba(final_input_imputed)[0][1]

    # Display Results
    st.divider()
    if prediction[0] == 1:
        st.error(f"High Risk Detected (Probability: {probability:.2%})")
        st.info("Medical recommendation: Further clinical investigation required.")
    else:
        st.success(f"Low Risk Detected (Probability: {probability:.2%})")
