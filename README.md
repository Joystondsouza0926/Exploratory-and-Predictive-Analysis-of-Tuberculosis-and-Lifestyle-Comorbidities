# 🩺 TB Risk Predictor

**Exploratory and Predictive Analysis of Tuberculosis and Lifestyle Comorbidities**

A Streamlit-based clinical decision support tool that predicts TB treatment outcomes and generates personalised clinical protocol reports using a Gradient Boosting machine learning model.

> Developed by **Joyston Jose D'souza**

---

## 📋 Features

- **Single Patient Triage** — Enter patient demographics and comorbidity data to get an instant TB treatment outcome prediction with confidence score and download the individual triage report as a CSV
- **All Patient Records Table** — Every submitted patient is logged with a unique ID (TB-0001, TB-0002, …) and displayed in a persistent table with per-row **📄 Clinical Protocols Report** download
- **Batch Prediction Upload** — Upload a CSV or Excel file to run bulk predictions on multiple patients at once
- **Interactive EDA** — Visualise the training dataset with:
  - Treatment outcome distribution
  - BMI vs HbA1c scatter plot by outcome
  - Syndemic factors correlation heatmap
  - TB outcomes by State Zone (bar + pie charts)
- **Clinical Protocols Page** — Personalised intervention recommendations based on the last patient's comorbidities (tobacco, diabetes, alcohol, BMI)
- **Patient Database Management** — Download all records as CSV or delete all records from the sidebar

---

## 🗂️ Project Structure

```
├── app.py                  # Main Streamlit application
├── best_tb_model.pkl       # Trained Gradient Boosting model
├── TB.csv                  # Training dataset (for EDA)
├── patient_records.csv     # Auto-generated persistent patient records
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Joystondsouza0926/Exploratory-and-Predictive-Analysis-of-Tuberculosis-and-Lifestyle-Comorbidities.git
cd Exploratory-and-Predictive-Analysis-of-Tuberculosis-and-Lifestyle-Comorbidities
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 Model Details

| Attribute | Detail |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Imbalance Handling | SMOTE (Synthetic Minority Oversampling) |
| Input Features | Age, Gender, Residence, BMI, Diabetes Status, HbA1c, Smoking Status, Alcohol Frequency, State Zone |
| Output | Binary: Success/Cured (1) or Poor Outcome/Failed (0) |

---

## 📥 Batch Upload Format

The batch CSV/Excel file must contain these columns:

| Column | Values |
|---|---|
| `Age` | Integer (15–90) |
| `Gender` | Male / Female / Transgender/Other |
| `Residence` | Rural / Urban / Slum |
| `BMI_Baseline` | Float |
| `Diabetes_Status` | Yes / No |
| `HbA1c_Level` | Float (0 = auto-impute) |
| `Smoking_Status` | Never / Former / Current |
| `Alcohol_Frequency` | Never / Occasional / Daily |

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.32.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.3.0 |
| joblib | ≥ 1.3.0 |
| plotly | ≥ 5.18.0 |
| openpyxl | ≥ 3.1.0 |

---

## 📄 License

This project is developed for academic and research purposes under the NTEP (National Tuberculosis Elimination Programme) context.
