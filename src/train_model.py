import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score
import kagglehub as kh
import os
import dotenv
import pickle

#------------------------------------------------------------
# 1. Load the dataset
#------------------------------------------------------------
#Load environment variables
dotenv.load_dotenv()

#Download public synthetic upcoding health insurance claims dataset
handle = "apexsyntheticdata/synthetic-healthcare-fraud-dataset-10k"
path =  kh.dataset_download(handle)

#Access downloaded dataset
for file in os.listdir(path):
    if(file.endswith(".csv")):
        csv_read = os.path.join(path, file)
        df = pd.read_csv(csv_read)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

#------------------------------------------------------------
# 2. Separate features and hidden labels
#------------------------------------------------------------
X_raw = df.drop(["is_anomaly", "anomaly_type"], axis=1)
y_true = df["is_anomaly"]

#------------------------------------------------------------
# 3. Define features columns
#------------------------------------------------------------
categorical_cols = ["cpt_procedure_code", "icd10_diagnosis_code", "provider_npi", "patient_id"]
numerical_cols = ["billed_amount"]

#Ensure all required columns are present
categorical_cols = [col for col in categorical_cols if col in X_raw.columns]
numerical_cols = [col for col in numerical_cols if col in X_raw.columns]

#------------------------------------------------------------
# 4. Encode categorical features
#------------------------------------------------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le

#------------------------------------------------------------
# 5. Scale numerical features
#------------------------------------------------------------
scaler = StandardScaler()
X_raw[numerical_cols] = scaler.fit_transform(X_raw[numerical_cols])

#------------------------------------------------------------
# 6. Prepare final feature matrix
#------------------------------------------------------------
feature_names = categorical_cols + numerical_cols
X = X_raw[feature_names]

#------------------------------------------------------------
# 7. Train Isolation Forest model
#------------------------------------------------------------
iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, verbose=1)

print("Training Isolation Forest...")
iso_model.fit(X)

#------------------------------------------------------------
# 8. Save the model & Preprocessors
#------------------------------------------------------------
with open("../models/iso_forest.pkl", "wb") as f:
    pickle.dump(iso_model, f)

with open("../models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("../models/encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

#------------------------------------------------------------
# 9. Evaluate the model
#------------------------------------------------------------
y_pred = iso_model.predict(X)
y_pred_binary = (y_pred == -1).astype(int)

precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)

print("-"*70)
print(f"✅ Training completed. | Precision: {precision:.3f} | Recall: {recall:.3f}")
print("-"*70)