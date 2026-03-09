# Exploratory and Predictive Analysis of Tuberculosis and Lifestyle Comorbidities

## Overview

This project provides a comprehensive analysis of tuberculosis (TB) treatment outcomes with a focus on lifestyle comorbidities and predictive modeling. It includes exploratory data analysis, machine learning model development, and an interactive web application for clinical decision support.

## Features

### 📊 Exploratory Data Analysis
- Comprehensive analysis of TB patient data
- Visualization of demographic and clinical patterns
- Correlation analysis between lifestyle factors and treatment outcomes
- Missing data imputation and preprocessing

### 🤖 Predictive Modeling
- Binary classification of treatment success/failure
- Multiple machine learning algorithms (Random Forest, Gradient Boosting, etc.)
- Feature importance analysis
- Model evaluation and validation

### 🏥 Clinical Decision Support App
- Interactive Streamlit web application
- Patient risk assessment tool
- Personalized clinical recommendations
- Dashboard for data visualization
- Geographic analysis by state zones

## Dataset

The analysis uses a TB patient dataset (`TB.csv`) containing:
- Demographic information (age, gender, residence, state zone)
- Clinical markers (BMI, diabetes status, HbA1c levels, HIV status)
- Lifestyle factors (smoking, alcohol consumption, cooking fuel)
- Treatment outcomes and drug resistance patterns

## Project Structure

```
├── analytics.ipynb          # Jupyter notebook with EDA and modeling
├── app.py                   # Streamlit web application
├── TB.csv                   # Tuberculosis patient dataset
└── README.md               # Project documentation
```

## Installation

1. Clone the repository
2. Install required Python packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn seaborn matplotlib plotly joblib imbalanced-learn
   ```

## Usage

### Running the Analysis Notebook
```bash
jupyter notebook analytics.ipynb
```

### Running the Web Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Key Findings

- Lifestyle comorbidities significantly impact TB treatment outcomes
- Diabetes and smoking are major risk factors for poor prognosis
- Machine learning models achieve ~89% accuracy in predicting treatment success
- Nutritional status (BMI) is a critical predictor of recovery

## Clinical Recommendations

The application provides evidence-based recommendations for:
- Treatment protocol intensification based on risk scores
- Comorbidity management (diabetes, smoking cessation)
- Nutritional support and lifestyle counseling
- Monitoring frequency adjustments

## Technologies Used

- **Python** for data analysis and modeling
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for machine learning
- **Seaborn & Matplotlib** for visualization
- **Streamlit** for web application
- **Plotly** for interactive charts
- **Joblib** for model serialization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## License

This project is intended for educational and research purposes in tuberculosis management and public health.