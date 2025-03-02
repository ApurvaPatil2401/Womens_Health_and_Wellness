import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(columns=["Sl. No", "Patient File No."], inplace=True)
    df = df.map(lambda x: str(x).replace(".", "", 1) if isinstance(x, str) else x)

    df = df.apply(pd.to_numeric, errors="coerce")  # Convert to float, setting errors to NaN


    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical features
    encoder = LabelEncoder()
    categorical_cols = ["Skin darkening (Y/N)", "hair growth(Y/N)", "Weight gain(Y/N)", 
                        "Pimples(Y/N)", "Hair loss(Y/N)", "Reg.Exercise(Y/N)", 
                        "Fast food (Y/N)", "Pregnant(Y/N)"]
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
