import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#------------------------------------------------------------
# 1. Page Configuration
#------------------------------------------------------------
st.set_page_config(
    page_title="Train Isolation Forest Model",
    layout="wide",
)
st.title("🩺 Health Insurance Claim Anomaly Detector")
st.markdown("Upload a claims CSV file to identify suspicious claims (upcoding)")

#------------------------------------------------------------
# 2. Load the trained model and preprocessors
#------------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessors():
    """
    Load the trained Isolation Forest, feature names, encoders, and scaler.
    """
    try:
        with open("../models/iso_forest.pkl", "rb") as f:
            model = pickle.load(f)
        with open("../models/feature_names.pkl", "rb") as f:
            feature_name = pickle.load(f)
        with open("../models/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("../models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, feature_name, encoders, scaler
    except FileNotFoundError:
        st.warning("Pre-trained model not found. Training a basic Isolation Forest on sample data")
        X_train = np.random.randn(1000, 5)
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_train)
        feature_names = ["cpt_procedure_code", "icd10_diagnosis_code", "provider_npi", "patient_id","billed_amount"]
        encoders = {}
        scaler = None
        return model, feature_names, encoders, scaler
    

model, feature_names, encoders, scaler = load_model_and_preprocessors()

#------------------------------------------------------------
# 3. Define a function to preprocess uploaded data
#------------------------------------------------------------
def preprocess_claims(df_raw, encoders, scaler, feature_names):
    """
    Convert raw uploaded CSV into the feature matrix expected by the model.
    """
    print(feature_names)
    df =  df_raw.copy()
    #select only the features used during training
    required_cols = ["cpt_procedure_code", "icd10_diagnosis_code", "provider_npi", "patient_id", "billed_amount"]
    
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None
    
    #Encode  categorical columns using the saved encoders
    for col in ["cpt_procedure_code", "icd10_diagnosis_code", "provider_npi", "patient_id"]:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    # Numerical features: scale if scaler provided
    num_cols = ['billed_amount']
    X = df[required_cols].copy()
    if scaler is not None:
        X[num_cols] = scaler.transform(X[num_cols])
    
    # Ensure column order matches training
    X = X[feature_names]
    return X

#------------------------------------------------------------
# 4. Main dashboard logic
#------------------------------------------------------------
upload_file = st.file_uploader("Upload Claims CSV", type=["csv"])

if upload_file is not None:
    df_claims = pd.read_csv(upload_file)
    st.write(f"Loaded {len(df_claims)} claims")

    #Preprocess the claims
    X_processed = preprocess_claims(df_claims, encoders, scaler, feature_names)
    if X_processed is None:
        st.stop()

    #Predict anomalies
    with st.spinner("Running anomaly detection..."):
        preds = model.predict(X_processed) 
        scores = model.score_samples(X_processed)
    
    # Add results to Dataframe
    df_claims["is_anomaly"] = (preds == -1).astype(int)  # 1 for anomaly, 0 for normal
    df_claims["anomaly_score"] = scores

    #Sort by anomaly score
    df_sorted = df_claims.sort_values(by="anomaly_score").reset_index(drop=True)

    #Show summary statistics
    n_anomalies = df_sorted["is_anomaly"].sum()
    st.metric("🚨 Suspicious Claims Found", f"{n_anomalies} / {len(df_claims)}")

#------------------------------------------------------------
# 5. Display flagged claims in a table
#------------------------------------------------------------
    st.subheader("Flagged Claims (most suspicious first)")

    #Select colymns to show in the table
    display_cols = ["claim_id", "cpt_procedure_code", "icd10_diagnosis_code", "billed_amount", "anomaly_score"] if "claim_id" in df_sorted.columns else ['cpt_procedure_code', 'icd10_diagnosis_code', 'billed_amount', 'anomaly_score']

    #Use st.dataframe with clickable raws
    st.dataframe(df_sorted[display_cols].head(50), use_container_width=True)
    
#------------------------------------------------------------
# 6. Click on a claim to see SHAP explanations
#------------------------------------------------------------
    st.subheader("🔍 Click on any claim index to see why it was flagged")
    selected_idx = st.number_input("Enter claim row index (0-based)", min_value=0, max_value=len(df_sorted)-1, step=1)

    if st.button("Show SHAP Explanation"):
        claim = df_sorted.iloc[selected_idx]
        st.write(f"**Selected Claim (Index {selected_idx})**")
        st.json(claim[display_cols].to_dict())
        
        # Get the feature vector for this claim
        X_claim = X_processed.iloc[[selected_idx]]  # keep as DataFrame
        
        # SHAP explanation
        with st.spinner("Generating explanation..."):
            # Use TreeExplainer for Isolation Forest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_claim)
        
        # Convert to 1D array (shap_values shape: (1, n_features))
        shap_vals = shap_values[0]
        
        # Create a horizontal bar chart of SHAP values
        fig, ax = plt.subplots(figsize=(8, 5))
        features = feature_names
        y_pos = np.arange(len(features))
        colors = ['red' if v > 0 else 'green' for v in shap_vals]
        ax.barh(y_pos, shap_vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("SHAP value (positive = pushes toward anomaly)")
        ax.set_title("Why this claim was flagged")
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        st.pyplot(fig)
        
        # Optional: show actual feature values
        st.write("**Actual feature values (after preprocessing):**")
        st.dataframe(X_claim)
        
        st.info("Positive SHAP values (red) mean that feature contributed to making the claim look suspicious. Negative (green) pushed it toward normal.")
