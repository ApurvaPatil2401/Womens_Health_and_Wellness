import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler  

# Define model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_DIR = os.path.join(BASE_DIR, "../models/")  
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def preprocess_data(file_path=None, input_data=None, train=True):
    if train:
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame([input_data])

    df.columns = df.columns.str.strip()

    # Drop unnecessary columns
    df.drop(columns=["Sl. No", "Patient File No."], inplace=True, errors="ignore")

    # Convert categorical values from Y/N to 1/0
    categorical_cols = ["Reg.Exercise(Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Skin darkening (Y/N)", "Fast food (Y/N)"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace({"Y": 1, "N": 0})
    feature_names = [
        "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
        "Reg.Exercise(Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
        "Skin darkening (Y/N)", "Fast food (Y/N)"
    ]

    # Check for missing features
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    df = df[feature_names]

    # Create new features
    df["BMI*FastFood"] = df["BMI"] * df["Fast food (Y/N)"]
    df["Exercise*HairLoss"] = df["Reg.Exercise(Y/N)"] * df["Hair loss(Y/N)"]

    if train:
        # Train & fit scaler correctly
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)  

        # Save trained scaler CORRECTLY (not as a dict)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f" Scaler trained and saved to {SCALER_PATH}")

    else:
        # Load existing scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)  
        df_scaled = scaler.transform(df) 
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    df_scaled.dropna(inplace=True)  

    return df_scaled
