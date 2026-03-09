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
            
# --- 6. BATCH PREDICTION (FILE UPLOAD) ---
st.divider()
st.header("📂 Batch Clinical Processor")
st.markdown("Upload patient records for automated risk stratification and clinical advice.")

uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type="csv")

if uploaded_file is not None:
    try:
        # Load the uploaded dataset
        batch_df = pd.read_csv(uploaded_file)
        
        # Required columns for the model
        required_cols = ['Age', 'Gender', 'Residence', 'BMI_Baseline', 'Diabetes_Status', 
                         'HbA1c_Level', 'Smoking_Status', 'Alcohol_Frequency']
        
        if all(col in batch_df.columns for col in required_cols):
            processed_rows = []
            advices = []
            
            for index, row in batch_df.iterrows():
                # Process each row through our training logic
                p_data = preprocess_input(
                    row['Age'], row['Gender'], row['Residence'], 
                    row['BMI_Baseline'], 
                    "Yes" if row['Diabetes_Status'] == 1 else "No", 
                    row['HbA1c_Level'], row['Smoking_Status'], row['Alcohol_Frequency']
                )
                processed_rows.append(p_data)
                
                # Logic for Clinical Advice
                if row['HbA1c_Level'] > 7.0:
                    advice = "Critical: Prioritize Glycemic Control."
                elif row['BMI_Baseline'] < 18.5:
                    advice = "Nutritional Intervention Required."
                elif row['Smoking_Status'] == "Current" or row['Alcohol_Frequency'] == "Daily":
                    advice = "Behavioral Counseling & DOTS Plus."
                else:
                    advice = "Routine Monitoring (Standard Regimen)."
                advices.append(advice)
            
            # Combine and Predict
            final_batch_df = pd.concat(processed_rows, ignore_index=True)
            batch_probs = model.predict_proba(final_batch_df)[:, 1]
            
            # Enrich the DataFrame
            batch_df['Success_Rate_%'] = (batch_probs * 100).round(2)
            batch_df['Risk_Level'] = ["Low" if p >= 0.7 else "Moderate" if p >= 0.5 else "High" for p in batch_probs]
            batch_df['Clinical_Suggestion'] = advices
            
            # UI: Results Summary
            st.success(f"Batch Analysis Complete: {len(batch_df)} Patients Screened.")
            
            # UI: Styled Dataframe
            st.dataframe(batch_df[['Age', 'Gender', 'Success_Rate_%', 'Risk_Level', 'Clinical_Suggestion']])
            
            # Download Section
            csv_output = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Clinical Report (CSV)",
                data=csv_output,
                file_name="NTEP_Smart_Predictor_Report.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Missing Columns! Required: {required_cols}")
            
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")