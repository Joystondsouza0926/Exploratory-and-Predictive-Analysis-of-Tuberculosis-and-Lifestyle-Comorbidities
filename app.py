# ==========================================
# FILE: app.py
# ==========================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NTEP Smart Predictor", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; background-color: #008080; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD THE BEST MODEL ---
# This loads the model you saved in the Jupyter Notebook
try:
    model = joblib.load('best_tb_model.pkl')
    st.sidebar.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error("Model file 'best_tb_model.pkl' not found. Please run the notebook first.")
    st.stop()

st.title("🏥 NTEP: Tuberculosis Treatment Outcome Predictor")
st.markdown("### Powered by Machine Learning (Best Performing Model)")

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Patient Demographics")
    age = st.slider("Age", 15, 90, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    residence = st.selectbox("Residence", ["Urban", "Rural"])
    
    st.header("Clinical Markers")
    bmi = st.number_input("BMI (Baseline)", 10.0, 40.0, 18.5)
    diabetes = st.radio("Diabetes Status", ["No", "Yes"])
    hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.4)
    
    st.header("Lifestyle Habits")
    smoking = st.selectbox("Smoking Status", ["Never", "Ex", "Current"])
    alcohol = st.selectbox("Alcohol Frequency", ["None", "Rare", "Weekly", "Daily"])

# --- 4. DATA PREPROCESSING ---
def preprocess_input(age, gender, residence, bmi, diabetes, hba1c, smoking, alcohol):
    # 1. Feature Engineering: Syndemic Burden
    burden = 0
    if diabetes == "Yes": burden += 1
    if smoking == "Current": burden += 1
    if alcohol == "Daily": burden += 1
    
    # 2. Encoding (Must match the Training logic exactly)
    # Gender: Male=1, Female=0
    gender_enc = 1 if gender == "Male" else 0
    
    # Residence: Urban=1, Rural=0
    res_enc = 1 if residence == "Urban" else 0
    
    # Diabetes: Yes=1, No=0
    diab_enc = 1 if diabetes == "Yes" else 0
    
    # Smoking: Never=2, Ex=0, Current=1 (Check your LabelEncoder mapping!)
    # Assumption based on alphabetical: Current=0, Ex=1, Never=2? 
    # To be safe, we map manually based on typical LabelEncoder behavior sorted alphabetically
    # Alphabetical: Current(0), Ex(1), Never(2)
    smoke_map = {"Current": 0, "Ex": 1, "Never": 2}
    
    # Alcohol: Daily(0), None(1), Rare(2), Weekly(3)
    alc_map = {"Daily": 0, "None": 1, "Rare": 2, "Weekly": 3}
    
    # Create DataFrame
    # Note: State_Zone is missing here. If the model was trained with State_Zone, 
    # we must provide it (usually 0s for generic prediction) or remove it from training.
    # Here we assume we pass dummy 0s for zones to avoid errors.
    
    data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_enc],
        'Residence': [res_enc],
        'BMI_Baseline': [bmi],
        'Diabetes_Status': [diab_enc],
        'HbA1c_Level': [hba1c],
        'Smoking_Status': [smoke_map[smoking]],
        'Alcohol_Frequency': [alc_map[alcohol]],
        'Syndemic_Burden': [burden],
        'State_Zone_North': [0],
        'State_Zone_South': [0],
        'State_Zone_West': [0]
    })
    return data

# --- 5. PREDICTION ---
if st.button("Predict Outcome"):
    input_df = preprocess_input(age, gender, residence, bmi, diabetes, hba1c, smoking, alcohol)
    
    # Get Probability
    prediction_prob = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Success)
    prediction_class = model.predict(input_df)[0]
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Probability of Cure")
        if prediction_prob >= 0.7:
            st.metric(label="Success Rate", value=f"{prediction_prob*100:.2f}%", delta="High Chance")
        elif prediction_prob >= 0.5:
            st.metric(label="Success Rate", value=f"{prediction_prob*100:.2f}%", delta="Moderate", delta_color="off")
        else:
            st.metric(label="Success Rate", value=f"{prediction_prob*100:.2f}%", delta="-High Risk", delta_color="inverse")
            
    with col2:
        st.subheader("Clinical Recommendation")
        if prediction_prob < 0.5:
            st.error("⚠️ HIGH RISK PATIENT")
            st.write("**Action Plan:**")
            st.write("- Initiate Directly Observed Therapy (DOTS) Plus.")
            st.write("- Mandatory Diabetes Management (Endocrinology referral).")
            st.write("- Substance abuse counseling required.")
        else:
            st.success("✅ STANDARD REGIMEN")
            st.write("Patient likely to respond well to standard First-Line treatment.")