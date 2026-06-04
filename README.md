# Health Insurance Claim Anomaly Detector

**Unsupervised anomaly detection for suspicious health insurance claims (upcoding) using Isolation Forest, SHAP explanations, and a Streamlit dashboard.**

---

## Overview

This project detects **anomalous health insurance claims** that may indicate fraud (upcoding). It uses an **unsupervised Isolation Forest** model trained on synthetic healthcare claims data. The system provides:

- **Ranked list of suspicious claims** (most anomalous first)
- **SHAP explanations** to understand *why* a claim was flagged
- **Interactive Streamlit dashboard** for auditors to upload CSV files and explore results

The project is designed as a proof‑of‑concept of my capstone project for an AI Accelerator Program by AI4SID.

---
## Dataset

**Source:** [Apex Synthetic Healthcare Fraud Dataset (10k)](https://www.kaggle.com/datasets/apexsyntheticdata/synthetic-healthcare-fraud-dataset-10k) (Kaggle)

- **10,000 rows**, 15 features
- Injected fraud types: `Upcoding`, `Phantom_Billing`, `Unbundling`
- Hidden labels: `is_anomaly` (0/1) and `anomaly_type` (used only for evaluation)

> The dataset is synthetic – no real patient or provider information is used.

---

## Methodology

1. **Preprocessing**  
   - Label encoding for categorical features (`cpt_procedure_code`, `icd10_diagnosis_code`, `provider_id`)  
   - Standard scaling for numerical features (`billed_amount`, `patient_age`, `length_of_stay`)

2. **Unsupervised Model**  
   - **Isolation Forest** (`n_estimators=100`, `contamination=0.05`)  
   - No fraud labels used during training

3. **Explainability**  
   - **SHAP (TreeExplainer)** to identify which features drove the anomaly score for each claim

4. **Deployment**  
   - **Streamlit dashboard** – upload CSV → view ranked anomalies → click on a claim to see SHAP bar chart

---

## Dependencies

This project requires the following Python packages. All versions are compatible with Python 3.11+.

| Library | Version | Purpose |
|--- | --- | --- | 
| pandas | 3.0.3 | Data loading, manipulation, and preprocessing | 
| numpy | 2.4.6 | Numerical operations and array handling | 
| scikit-learn | 1.8.0 | Isolation Forest model, StandardScaler, LabelEncoder | 
| matplotlib | 3.10.9 | Core plotting (SHAP visualisations, report charts) | 
| seaborn | 0.13.2 | Statistical data visualisation for EDA | 
| shap | 0.50.0 | SHAP explainability for model predictions | 
| streamlit | 1.57.0 | Interactive fraud detection dashboard | 
| python-dotenv | 1.0.0 | Load Kaggle API keys from .env file | 
| kagglehub | 1.0.0 | Download datasets directly from Kaggle | 
| jupyter | optional | Run exploratory notebooks | 



