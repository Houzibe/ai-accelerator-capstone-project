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

Library&emsp;&emsp;&emsp;&emsp;Version&emsp;&emsp;&emsp;&emsp;Purpose<br>
pandas&emsp;&emsp;&emsp;&emsp;3.0.3&emsp;&emsp;&emsp;&emsp;Data loading, manipulation, and preprocessing<br>
numpy&emsp;&emsp;&emsp;&emsp;2.4.6&emsp;&emsp;&emsp;&emsp;tNumerical operations and array handling<br>
scikit-learn&emsp;&emsp;&emsp;&emsp;1.8.0&emsp;&emsp;&emsp;&emsp;Isolation Forest model, StandardScaler, LabelEncoder<br>
matplotlib&emsp;&emsp;&emsp;&emsp;3.10.9&emsp;&emsp;&emsp;&emsp;Core plotting (SHAP visualisations, report charts)<br>
seaborn&emsp;&emsp;&emsp;&emsp;0.13.2&emsp;&emsp;&emsp;&emsp;Statistical data visualisation for EDA<br>
shap&emsp;&emsp;&emsp;&emsp;0.50.0&emsp;&emsp;&emsp;&emsp;SHAP explainability for model predictions<br>
streamlit&emsp;&emsp;&emsp;&emsp;1.57.0&emsp;&emsp;&emsp;&emsp;Interactive fraud detection dashboard<br>
python-dotenv&emsp;&emsp;&emsp;&emsp;1.0.0&emsp;&emsp;&emsp;&emsp;Load Kaggle API keys from .env file<br>
kagglehub&emsp;&emsp;&emsp;&emsp;1.0.0&emsp;&emsp;&emsp;&emsp;Download datasets directly from Kaggle<br>
jupyter&emsp;&emsp;&emsp;&emsp;optional&emsp;&emsp;&emsp;&emsp;Run exploratory notebooks<br>



