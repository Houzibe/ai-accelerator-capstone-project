import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score
import kagglehub as kh
import os
import dotenv

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

print(df.shape)
print(df.head())