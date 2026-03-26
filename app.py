import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import uuid

# Set page config
st.set_page_config(
    page_title="NTEP Smart Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #151522 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #232336;
        border-right: 1px solid #383854;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* Metrics Core Elements */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Cards and Containers */
    div[data-testid="stVerticalBlock"] div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background: rgba(43, 43, 60, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a2a40;
        border-radius: 8px;
    }
    
    /* Input Fields */
    .stSelectbox div[data-baseweb="select"], .stNumberInput > div > div > input, .stSlider > div[data-baseweb="slider"] {
        background-color: rgba(255,255,255,0.05);
        color: white;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Resources ---
@st.cache_resource
def load_model():
    model_path = 'best_tb_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it is in the same directory as app.py.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data
def load_data():
    data_path = 'TB.csv'
    if not os.path.exists(data_path):
        st.warning(f"Warning: Dataset '{data_path}' not found for EDA. Place it in the app directory.")
        return None
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

model = load_model()
df = load_data()


# --- Feature Engineering Pipeline ---
def preprocess_input(inputs):
    """
    Transforms raw user inputs into the exact feature array format required by the model:
    ['Age', 'Gender', 'Residence', 'BMI_Baseline', 'Diabetes_Status', 'HbA1c_Level', 
     'Smoking_Status', 'Alcohol_Frequency', 'Syndemic_Burden', 
     'State_Zone_North', 'State_Zone_South', 'State_Zone_West']
    """
    
    # Extract inputs
    age = inputs['Age']
    gender_raw = inputs['Gender']
    residence_raw = inputs['Residence']
    bmi = inputs['BMI_Baseline']
    diabetes_raw = inputs['Diabetes_Status']
    hba1c = inputs['HbA1c_Level']
    smoking_raw = inputs['Smoking_Status']
    alcohol_raw = inputs['Alcohol_Frequency']
    state_zone_raw = inputs['State_Zone']

    # 1. Diabetes mapping (Yes=1, No=0)
    diabetes_status = 1 if diabetes_raw == 'Yes' else 0

    # 2. Impute HbA1c
    if hba1c == 0.0:
        hba1c = 8.13 if diabetes_status == 1 else 5.40

    # 3. Syndemic Burden Calculation (0-3)
    smoking_burden = 1 if smoking_raw == 'Current' else 0
    alcohol_burden = 1 if alcohol_raw == 'Daily' else 0
    syndemic_burden = diabetes_status + smoking_burden + alcohol_burden

    # 4. Label Encoding
    # Assuming standard alphabetical/order-of-appearance mapping similar to LabelEncoder
    # Note: For strict adherence to the notebook, we need to map based on how the training data was encoded.
    # Usually: Female:0, Male:1; Rural:0, Urban:1.
    # We will implement reasonable defaults matching standard alphabetical LabelEncoder behavior if unknown.
    gender_map = {'Female': 0, 'Male': 1, 'Transgender/Other': 2}
    gender = gender_map.get(gender_raw, 0)
    
    residence_map = {'Rural': 0, 'Urban': 1, 'Slum': 2}
    residence = residence_map.get(residence_raw, 0)
    
    smoking_map = {'Current': 0, 'Former': 1, 'Never': 2}
    smoking = smoking_map.get(smoking_raw, 2)
    
    alcohol_map = {'Daily': 0, 'Occasional': 1, 'Never': 2, 'NaN': 3} # Based on standard label encoder behavior
    alcohol = alcohol_map.get(alcohol_raw, 2)

    # 5. One-Hot Encoding for State_Zone (North, South, West)
    # The training model dropped the first (likely East).
    state_zone_north = 1 if state_zone_raw == 'North' else 0
    state_zone_south = 1 if state_zone_raw == 'South' else 0
    state_zone_west = 1 if state_zone_raw == 'West' else 0

    features = np.array([[
        age, 
        gender, 
        residence, 
        bmi, 
        diabetes_status, 
        hba1c, 
        smoking, 
        alcohol, 
        syndemic_burden, 
        state_zone_north, 
        state_zone_south, 
        state_zone_west
    ]])
    
    return features, syndemic_burden

def preprocess_batch(df_input):
    df = df_input.copy()
    
    expected_cols = ['Age', 'Gender', 'Residence', 'BMI_Baseline', 'Diabetes_Status', 'HbA1c_Level', 'Smoking_Status', 'Alcohol_Frequency']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
    df['Diabetes_Status_num'] = df['Diabetes_Status'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' or x == 1 else 0)
    
    def impute_hba1c(row):
        h = row['HbA1c_Level']
        if pd.isna(h) or h == 0.0:
            return 8.13 if row['Diabetes_Status_num'] == 1 else 5.40
        return h
    df['HbA1c_Level_num'] = df.apply(impute_hba1c, axis=1)
    
    df['smoking_burden'] = df['Smoking_Status'].apply(lambda x: 1 if str(x).strip().lower() == 'current' else 0)
    df['alcohol_burden'] = df['Alcohol_Frequency'].apply(lambda x: 1 if str(x).strip().lower() == 'daily' else 0)
    df['Syndemic_Burden'] = df['Diabetes_Status_num'] + df['smoking_burden'] + df['alcohol_burden']
    
    gender_map = {'Female': 0, 'Male': 1, 'Transgender/Other': 2, 'female': 0, 'male': 1, 'transgender/other': 2}
    df['Gender_num'] = df['Gender'].map(gender_map).fillna(0)
    
    residence_map = {'Rural': 0, 'Urban': 1, 'Slum': 2, 'rural': 0, 'urban': 1, 'slum': 2}
    df['Residence_num'] = df['Residence'].map(residence_map).fillna(0)
    
    smoking_map = {'Current': 0, 'Former': 1, 'Never': 2, 'current': 0, 'former': 1, 'never': 2}
    df['Smoking_Status_num'] = df['Smoking_Status'].map(smoking_map).fillna(2)
    
    alcohol_map = {'Daily': 0, 'Occasional': 1, 'Never': 2, 'daily': 0, 'occasional': 1, 'never': 2}
    df['Alcohol_Frequency_num'] = df['Alcohol_Frequency'].map(alcohol_map).fillna(2)
    
    df['State_Zone_North'] = 0
    df['State_Zone_South'] = 0
    df['State_Zone_West'] = 0
    
    features = df[['Age', 'Gender_num', 'Residence_num', 'BMI_Baseline', 'Diabetes_Status_num', 'HbA1c_Level_num', 
                   'Smoking_Status_num', 'Alcohol_Frequency_num', 'Syndemic_Burden', 
                   'State_Zone_North', 'State_Zone_South', 'State_Zone_West']]
    
    features = features.rename(columns={
        'Gender_num': 'Gender',
        'Residence_num': 'Residence',
        'Diabetes_Status_num': 'Diabetes_Status',
        'HbA1c_Level_num': 'HbA1c_Level',
        'Smoking_Status_num': 'Smoking_Status',
        'Alcohol_Frequency_num': 'Alcohol_Frequency'
    })
    
    return features


# --- Sidebar Navigation ---
st.sidebar.image("Molbio w tagline. JPG.jpg", use_container_width=True)
st.sidebar.title("TB Risk Predictor")
st.sidebar.caption("👨‍💻 Developed by **Joyston Jose D'souza**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Clinical Triage", "Interactive EDA", "Clinical Protocols"])
st.sidebar.markdown("---")

persistent_file = "patient_records.csv"
if os.path.exists(persistent_file):
    st.sidebar.markdown("### 💾 Patient Database")
    with open(persistent_file, "rb") as f:
        st.sidebar.download_button(
            label="Download All Records (CSV)",
            data=f,
            file_name="all_patient_records.csv",
            mime="text/csv",
            use_container_width=True
        )
    if st.sidebar.button("🗑️ Delete All Records", use_container_width=True):
        os.remove(persistent_file)
        st.rerun()
    st.sidebar.markdown("---")

st.sidebar.caption("Revolutionizing TB care with predictive analytics and dynamic clinical insights.")


# ==========================================
# PAGE 1: Clinical Triage & Prediction
# ==========================================
if page == "Clinical Triage":
    st.title("Exploratory and Predictive Analysis of Tuberculosis and Lifestyle Comorbidities")
    st.markdown("Evaluate patient risk through a single entry below or upload a batch CSV for bulk predictions.")
    
    tab1, tab2 = st.tabs(["Single Patient Triage", "Batch Prediction Upload"])
    
    with tab1:
        with st.container():
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Demographic details")
                age = st.number_input("Age (Years) *", min_value=15, max_value=90, value=None, placeholder="Required")
                gender = st.selectbox("Gender", ["Male", "Female", "Transgender/Other"])
                residence = st.selectbox("Residence Type", ["Rural", "Urban", "Slum"])
                
                st.markdown("### Clinical Baseline")
                bmi = st.slider("Baseline BMI", min_value=10.0, max_value=50.0, value=20.5, step=0.1)
                st.caption("ℹ️ **Body Mass Index (BMI)**: A measure of body fat based on height and weight. BMI < 18.5 indicates undernutrition, which is a major risk factor for TB.")
                
            with col2:
                st.markdown("### Syndemic Factors (Comorbidities)")
                diabetes = st.selectbox("Diabetes Status", ["No", "Yes"])
                hba1c = st.slider("HbA1c Level (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1)
                st.caption("ℹ️ Leave at 0 to auto-estimate based on diabetes status.")
                smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
                alcohol = st.selectbox("Alcohol Frequency", ["Never", "Occasional", "Daily"])
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_btn = st.button("Generate Prediction & Compute Risk", use_container_width=True)
        
        if predict_btn:
            if age is None:
                st.error("Please enter a valid age before generating a prediction.")
            elif model is None:
                st.error("Prediction unavailable because the model file could not be loaded.")
            else:
                inputs = {
                    'Age': age,
                    'Gender': gender,
                    'Residence': residence,
                    'State_Zone': 'East',
                    'BMI_Baseline': bmi,
                    'Diabetes_Status': diabetes,
                    'HbA1c_Level': hba1c,
                    'Smoking_Status': smoking,
                    'Alcohol_Frequency': alcohol
                }
                
                # Preprocess and calculate factors
                features, syndemic_score = preprocess_input(inputs)
                
                # Make prediction
                try:
                    prediction = model.predict(features)[0]
                    proba = model.predict_proba(features)[0]
                    
                    # Update Session State for Clinical Protocols Page
                    st.session_state['last_inputs'] = inputs
                    st.session_state['syndemic_score'] = syndemic_score
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prognostic Results")
                    
                    res_col1, res_col2 = st.columns([2, 1])
                    
                    with res_col1:
                        # Model Output
                        if prediction == 1:
                            st.success("🟢 **Forecast: Success / Cured / Completed**")
                            st.info(f"Model Confidence: {proba[1]*100:.1f}%")
                            st.progress(float(proba[1]))
                        else:
                            st.error("🔴 **Forecast: Poor Outcome / Failed / Lost to Follow-up**")
                            st.warning(f"Model Confidence: {proba[0]*100:.1f}%")
                            st.progress(float(proba[0]))
                    
                    with res_col2:
                        # Syndemic Metric Display
                        score_color = ""
                        if syndemic_score == 0:
                            score_color = "normal"
                            state = "Low Risk"
                        elif syndemic_score == 1:
                            score_color = "off"
                            state = "Moderate"
                        else:
                            score_color = "inverse"
                            state = "High Burden"
                            
                        st.metric(label="Syndemic Burden Score", value=f"{syndemic_score} / 3", delta=state, delta_color=score_color)
                    
                    st.markdown("Navigate to the **Clinical Protocols** tab on the sidebar to view personalized actionable recommendations based on these results.")
                    

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    
                st.markdown("---")
                st.markdown("---")
                # --- Generate Patient ID ---
                persistent_file = "patient_records.csv"
                patient_id = "TB-0001"
                if os.path.exists(persistent_file):
                    try:
                        existing_df = pd.read_csv(persistent_file)
                        if not existing_df.empty and 'Patient_ID' in existing_df.columns:
                            last_id = existing_df['Patient_ID'].iloc[-1]
                            if str(last_id).startswith("TB-"):
                                num_part = int(str(last_id).split("-")[1])
                                patient_id = f"TB-{num_part + 1:04d}"
                            else:
                                patient_id = f"TB-{len(existing_df) + 1:04d}"
                    except Exception:
                        pass

                # --- Build and Save Record ---
                st.session_state['last_patient_id'] = patient_id
                record_inputs = inputs.copy()
                if 'State_Zone' in record_inputs:
                    del record_inputs['State_Zone']

                record = {'Patient_ID': patient_id}
                record.update(record_inputs)
                record['Syndemic_Burden_Score'] = f"{syndemic_score} / 3"
                record['Prediction_Result'] = "Success / Cured" if prediction == 1 else "Poor Outcome / Failed"
                record['Confidence'] = f"{proba[1]*100:.1f}%" if prediction == 1 else f"{proba[0]*100:.1f}%"

                record_df = pd.DataFrame([record])

                if os.path.exists(persistent_file):
                    record_df.to_csv(persistent_file, mode='a', header=False, index=False)
                else:
                    record_df.to_csv(persistent_file, index=False)

                csv_report = record_df.to_csv(index=False).encode('utf-8')



                st.markdown(f"### 💾 Patient Record (ID: {patient_id})")
                st.success("Record generated and database updated successfully. You can download the individual prediction below.")
                st.download_button(
                    label="Download Individual Report (CSV)",
                    data=csv_report,
                    file_name=f"{patient_id}_triage_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )


    # ---- Always-visible All Patient Records table (inside tab1 only) ----
    with tab1:
        def build_proto_report(row):
            """Generate a clinical protocols report text for a given patient record row."""
            lines = [
                "==========================================",
                "     SMART PREDICTOR CLINICAL REPORT   ",
                f"   Patient ID: {row.get('Patient_ID', 'N/A')}",
                "==========================================\n",
                f"Patient Age: {row.get('Age', 'N/A')}",
                f"Patient Gender: {row.get('Gender', 'N/A')}",
                f"Baseline BMI: {row.get('BMI_Baseline', 'N/A')}",
                f"Diabetes Status: {row.get('Diabetes_Status', 'N/A')}",
                f"Smoking Status: {row.get('Smoking_Status', 'N/A')}",
                f"Alcohol Frequency: {row.get('Alcohol_Frequency', 'N/A')}",
                f"Syndemic Burden Score: {row.get('Syndemic_Burden_Score', 'N/A')}\n",
                "------------------------------------------",
                "TARGETED PROTOCOLS & INTERVENTIONS:",
                "------------------------------------------",
            ]
            has_proto = False
            try:
                if str(row.get('Smoking_Status', '')).strip() == 'Current':
                    has_proto = True
                    lines.append("\n[TOBACCO DEPENDENCY]")
                    lines.append("Protocol: Immediate referral for pulmonary counseling and cessation therapy.")
                hba1c_val = float(row.get('HbA1c_Level', 0) or 0)
                if hba1c_val > 6.5 or str(row.get('Diabetes_Status', '')).strip() == 'Yes':
                    has_proto = True
                    lines.append("\n[GLYCEMIC DYSREGULATION]")
                    lines.append("Protocol: Endocrinology consultation required. Initiate fasting glucose tracking.")
                if str(row.get('Alcohol_Frequency', '')).strip() == 'Daily':
                    has_proto = True
                    lines.append("\n[ALCOHOL CONSUMPTION RISK]")
                    lines.append("Protocol: Assess for hepatic strain. Increase frequency of LFTs.")
                bmi_val = float(row.get('BMI_Baseline', 99) or 99)
                if bmi_val < 18.5:
                    has_proto = True
                    lines.append("\n[UNDERNUTRITION]")
                    lines.append("Protocol: Initiate direct nutritional supplementation pathway.")
            except Exception:
                pass
            if not has_proto:
                lines.append("\n[STANDARD PROTOCOL]")
                lines.append("No severe comorbidities detected. Proceed with standard DOTS protocol.")
            return "\n".join(lines)

        persistent_file_check = "patient_records.csv"
        if os.path.exists(persistent_file_check):
            try:
                all_records_df = pd.read_csv(persistent_file_check)
                if not all_records_df.empty:
                    st.markdown("---")
                    st.markdown("### 🗂️ All Patient Records")
                    data_cols = list(all_records_df.columns)
                    header_cols = st.columns(len(data_cols) + 1)
                    for i, col_name in enumerate(data_cols):
                        header_cols[i].markdown(f"**{col_name}**")
                    header_cols[-1].markdown("**Clinical Report**")
                    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)
                    for idx, row in all_records_df.iterrows():
                        row_cols = st.columns(len(data_cols) + 1)
                        for i, col_name in enumerate(data_cols):
                            row_cols[i].write(str(row[col_name]))
                        proto_txt = build_proto_report(row.to_dict())
                        pid = str(row.get('Patient_ID', 'patient'))
                        row_cols[-1].download_button(
                            label="📄",
                            data=proto_txt,
                            file_name=f"{pid}_clinical_protocols.txt",
                            mime="text/plain",
                            key=f"dl_main_{pid}_{idx}"
                        )
            except Exception:
                pass

    with tab2:

        st.markdown("### Batch Prediction Upload")
        st.markdown("Upload a **CSV or Excel** file containing patient data to generate predictions in bulk. Required columns: `Age`, `Gender`, `Residence`, `BMI_Baseline`, `Diabetes_Status`, `HbA1c_Level`, `Smoking_Status`, `Alcohol_Frequency`.")
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name.lower()
                if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                    batch_df = pd.read_excel(uploaded_file)
                else:
                    batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                if st.button("Generate Batch Predictions", use_container_width=True):
                    if model is None:
                        st.error("Prediction unavailable because the model file could not be loaded.")
                    else:
                        with st.spinner("Processing data and generating predictions..."):
                            features_df = preprocess_batch(batch_df)
                            predictions = model.predict(features_df)
                            probas = model.predict_proba(features_df)
                            
                            batch_df['Prediction'] = ['Success / Cured' if p == 1 else 'Poor Outcome' for p in predictions]
                            batch_df['Model Confidence'] = [f"{prob[1]*100:.1f}%" if p == 1 else f"{prob[0]*100:.1f}%" for p, prob in zip(predictions, probas)]
                            
                            # Remove State_Zone if it was accidentally included in the upload
                            if 'State_Zone' in batch_df.columns:
                                batch_df = batch_df.drop(columns=['State_Zone'])
                                
                            st.success("Batch predictions generated successfully!")
                            st.dataframe(batch_df)
                            
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name='batch_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")


# ==========================================
# PAGE 2: Interactive EDA
# ==========================================
elif page == "Interactive EDA":
    st.title("Interactive Exploratory Data Analysis")
    st.markdown("Visualize the underlying dynamics of the training cohort.")
    
    if df is None:
        st.warning("Please place `TB.csv` in the application root directory to enable visualizations.")
    else:
        # Pre-process DF for visualization purposes
        if 'Treatment_Outcome' in df.columns:
            success_outcomes = ['Cured', 'Completed']
            df['Outcome_Binary'] = df['Treatment_Outcome'].apply(lambda x: 1 if x in success_outcomes else 0)
            df['Outcome_Label'] = df['Outcome_Binary'].map({1: 'Success', 0: 'Poor Outcome'})
            
        st.markdown("### 1. Treatment Outcome Distribution")
        view_col1, view_col2 = st.columns([3, 1])
        
        with view_col1:
            fig1 = px.histogram(df, x='Treatment_Outcome', color='Outcome_Label', 
                                color_discrete_map={'Success': '#00f2fe', 'Poor Outcome': '#ff4b4b'},
                                title='Historical Distribution of TB Outcomes')
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig1, use_container_width=True)
            
        with view_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            with st.expander("ℹ️ Dataset Insight", expanded=True):
                st.write("The original dataset was highly imbalanced:")
                st.write("- **Success:** ~78%")
                st.write("- **Failure/Poor:** ~21%")
                st.write("Synthetic Minority Over-sampling Technique (SMOTE) was utilized during Gradient Boosting model training to syntheticize minority cases and stabilize predictions.")
                
        st.markdown("---")
        
        st.markdown("### 2. Machine Learning Model Insights")
        with st.expander("🧠 About the Predictive Model", expanded=True):
            st.markdown("""
            **Model Architecture:** Gradient Boosting Classifier
            
            **Why this model?**
            - **Non-linear Relationships:** TB treatment outcomes rely on complex, non-linear interactions between demographic and clinical factors (such as BMI, age, and HbA1c levels), which tree-based models excel at capturing.
            - **Robust to Outliers:** Gradient Boosting handles outliers in clinical datasets better than linear models.
            - **High Performance on Structured Data:** Ensembles of decision trees are historically the most performant algorithms for tabular/structured clinical datasets.
            - **Imbalanced Data:** When paired with SMOTE (Synthetic Minority Over-sampling Technique) to handle the 78/21 imbalanced outcome class, the model successfully maximizes recall for high-risk patients without sacrificing overall accuracy.
            """)
            
        st.markdown("---")

        st.markdown("### 3. Metabolic Variables vs. Treatment Outcome")
        # Ensure HbA1c is clean for visualization
        df_clean = df.copy()
        mean_diab = df_clean[df_clean['Diabetes_Status'] == 1]['HbA1c_Level'].mean()
        mean_nondiab = df_clean[df_clean['Diabetes_Status'] == 0]['HbA1c_Level'].mean()
        df_clean.loc[(df_clean['Diabetes_Status'] == 1) & (df_clean['HbA1c_Level'].isna()), 'HbA1c_Level'] = mean_diab
        df_clean.loc[(df_clean['Diabetes_Status'] == 0) & (df_clean['HbA1c_Level'].isna()), 'HbA1c_Level'] = mean_nondiab
        
        fig2 = px.scatter(df_clean, x='BMI_Baseline', y='HbA1c_Level', color='Outcome_Label',
                          opacity=0.6,
                          color_discrete_map={'Success': '#00f2fe', 'Poor Outcome': '#ff4b4b'},
                          title="BMI vs. HbA1c Highlighted by Outcome")
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 3. Syndemic Factors Correlation Heatmap")
        
        # Prepare correlation matrix data
        corr_cols = ['Diabetes_Status', 'Outcome_Binary']
        if 'Smoking_Status' in df.columns:
            df_clean['Smoking_Mapped'] = df_clean['Smoking_Status'].apply(lambda x: 1 if x == 'Current' else 0)
            corr_cols.append('Smoking_Mapped')
        if 'Alcohol_Frequency' in df.columns:
            df_clean['Alcohol_Mapped'] = df_clean['Alcohol_Frequency'].apply(lambda x: 1 if x == 'Daily' else 0)
            corr_cols.append('Alcohol_Mapped')
            
        corr_matrix = df_clean[corr_cols].corr()
        
        fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale='RdBu_r', 
                         title='Correlation of Comorbidities with Successful Outcome')
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # --- State Zone Visualization ---
        if 'State_Zone' in df.columns and 'Outcome_Label' in df.columns:
            st.markdown("### 4. TB Outcomes by State Zone")

            zone_col1, zone_col2 = st.columns([3, 2])

            with zone_col1:
                zone_outcome = df.groupby(['State_Zone', 'Outcome_Label']).size().reset_index(name='Count')
                fig_zone_bar = px.bar(
                    zone_outcome,
                    x='State_Zone', y='Count', color='Outcome_Label',
                    barmode='group',
                    color_discrete_map={'Success': '#00f2fe', 'Poor Outcome': '#ff4b4b'},
                    title='Treatment Outcome Distribution by State Zone',
                    labels={'State_Zone': 'State Zone', 'Count': 'Number of Patients'}
                )
                fig_zone_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), legend_title_text='Outcome'
                )
                st.plotly_chart(fig_zone_bar, use_container_width=True)

            with zone_col2:
                zone_counts = df['State_Zone'].value_counts().reset_index()
                zone_counts.columns = ['State_Zone', 'Count']
                fig_zone_pie = px.pie(
                    zone_counts,
                    names='State_Zone', values='Count',
                    title='Patient Share by Zone',
                    color_discrete_sequence=px.colors.sequential.Plasma_r
                )
                fig_zone_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(bgcolor='rgba(0,0,0,0)')
                )
                fig_zone_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_zone_pie, use_container_width=True)


# ==========================================
# PAGE 3: Actionable Clinical Protocols
# ==========================================
elif page == "Clinical Protocols":
    st.title("Actionable Clinical Protocols")
    st.markdown("Targeted interventions based on the patient's individual comorbidity profiles.")
    
    if 'last_inputs' not in st.session_state:
        st.info("⚠️ Please generate a prediction on the **Clinical Triage** page first to view personalized protocols.")
    else:
        inputs = st.session_state['last_inputs']
        score = st.session_state.get('syndemic_score', 0)
        
        patient_id_display = st.session_state.get('last_patient_id', 'N/A')
        st.subheader(f"Patient Summary Profile — ID: {patient_id_display}")
        st.write(f"**Burden Score:** {score}/3 | **BMI:** {inputs['BMI_Baseline']} | **Age:** {inputs['Age']}")
        
        st.markdown("### Recommended Detailed Interventions")
        
        has_warnings = False
        report_lines = [
            "==========================================",
            "   NTEP SMART PREDICTOR CLINICAL REPORT   ",
            "==========================================\n",
            f"Patient Age: {inputs['Age']}",
            f"Patient Gender: {inputs['Gender']}",
            f"Baseline BMI: {inputs['BMI_Baseline']}",
            f"Syndemic Burden Score: {score}/3\n",
            "------------------------------------------",
            "TARGETED PROTOCOLS & INTERVENTIONS:",
            "------------------------------------------"
        ]
        
        # Rule 1: Smoking
        if inputs['Smoking_Status'] == 'Current':
            has_warnings = True
            st.error("🚭 **Tobacco Dependency Identified**")
            msg = (
                "> **Protocol:** Immediate referral for pulmonary counseling and cessation therapy.\n\n"
                "**Clinical Justification:** Tobacco use significantly increases the risk of mortality and delayed sputum conversion in TB patients. "
                "Nicotine suppresses macrophage function, the primary immune defense against Mycobacterium tuberculosis.\n\n"
                "**Action Plan:**\n"
                "- Enlist patient in DOTS-plus smoking cessation program.\n"
                "- Prescribe Nicotine Replacement Therapy (NRT) if medically appropriate.\n"
                "- Schedule weekly behavioral counseling check-ins for the first month of treatment."
            )
            st.write(msg)
            report_lines.append("\n[TOBACCO DEPENDENCY]")
            report_lines.append(msg.replace("> **Protocol:**", "Protocol:").replace("**", ""))
        
        # Rule 2: Diabetes / HbA1c
        hba1c = inputs['HbA1c_Level']
        diabetes = inputs['Diabetes_Status']
        if (hba1c > 0 and hba1c > 6.5) or diabetes == 'Yes':
            has_warnings = True
            st.warning("🩸 **Glycemic Control Fluctuation**")
            msg = (
                "> **Protocol:** Endocrinology consultation required. Initiate fasting glucose tracking.\n\n"
                "**Clinical Justification:** Poorly controlled diabetes suppresses the Th1 immune response and requires aggressive concurrent management alongside standard DOTS therapy. "
                "Co-morbidity increases the risk of TB relapse and treatment failure.\n\n"
                "**Action Plan:**\n"
                "- Prescribe concurrent Metformin/Insulin titration schedule as recommended by an endocrinologist.\n"
                "- Bi-weekly fasting blood glucose and HbA1c testing every 3 months.\n"
                "- Dietary consultation emphasizing low-glycemic index foods."
            )
            st.write(msg)
            report_lines.append("\n[GLYCEMIC DYSREGULATION]")
            report_lines.append(msg.replace("> **Protocol:**", "Protocol:").replace("**", ""))
            
        # Rule 3: Alcohol
        if inputs['Alcohol_Frequency'] == 'Daily':
            has_warnings = True
            st.warning("🍻 **High-Risk Alcohol Consumption**")
            msg = (
                "> **Protocol:** Assess for hepatic strain. Increase frequency of Liver Function Tests (LFTs) and refer to an addiction counselor.\n\n"
                "**Clinical Justification:** High risk for drug-induced liver injury (DILI) due to first-line TB medications (Isoniazid, Rifampin, Pyrazinamide) combined with alcohol hepatotoxicity. "
                "Alcoholism often correlates with poor treatment adherence and malnutrition.\n\n"
                "**Action Plan:**\n"
                "- Baseline LFTs immediately and re-check every 2 weeks for the intensive phase.\n"
                "- Mandatory psychiatric evaluation for relapse prevention.\n"
                "- Ensure DOTS provider incorporates daily wellness checks."
            )
            st.write(msg)
            report_lines.append("\n[ALCOHOL CONSUMPTION RISK]")
            report_lines.append(msg.replace("> **Protocol:**", "Protocol:").replace("**", ""))
            
        # Rule 4: BMI/Nutrition
        if inputs['BMI_Baseline'] < 18.5:
            has_warnings = True
            st.info("🥗 **Undernutrition / Cachexia**")
            msg = (
                "> **Protocol:** Initiate direct nutritional supplementation pathway (e.g., Nikshay Poshan Yojana). Provide high-protein, calorie-dense diets.\n\n"
                "**Clinical Justification:** Undernutrition is the most prominent reversible risk factor for TB mortality. Low BMI reduces drug bioavailability and impairs cellular immunity.\n\n"
                "**Action Plan:**\n"
                "- Immediate provision of macro-nutrient rations (e.g., groundnut, pulses, milk powder).\n"
                "- Target weight gain of at least 5% of body weight in the first two months.\n"
                "- Register patient under regional direct benefit transfer (DBT) schemes for nutritional support."
            )
            st.write(msg)
            report_lines.append("\n[UNDERNUTRITION]")
            report_lines.append(msg.replace("> **Protocol:**", "Protocol:").replace("**", ""))

        if not has_warnings:
            st.success("✅ **Standard Protocol Sufficient**")
            msg = (
                "> The patient does not possess any immediate severe lifestyle comorbidities based on the intake. Proceed with standard direct observation, monthly weight checks, and scheduled sputum smears."
            )
            st.write(msg)
            report_lines.append("\n[STANDARD PROTOCOL]")
            report_lines.append("No severe lifestyle comorbidities detected. Proceed with standard DOTS protocol.")

        st.markdown("---")
        report_text = "\n".join(report_lines)
        st.download_button(
            label="Download Clinical Protocols Report (.txt)",
            data=report_text,
            file_name="clinical_protocols_report.txt",
            mime="text/plain",
            use_container_width=True
        )

