import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------
# Reusable Preprocessing Function
# -------------------------------
def preprocess_dataset(df, target_col=None):
    """
    Preprocess dataset for both training (synthetic CSV) and real sensor input.
    - Replaces 0 with NaN (for biologically impossible values)
    - Converts to numeric
    - Fills missing values with column mean
    - Encodes disease labels if available
    """
    df = df.copy()

    # Replace impossible 0 values with NaN
    df.replace(0, np.nan, inplace=True)

    # Convert all non-target columns to numeric
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing values with column mean
    for col in df.columns:
        if col != target_col:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

    # Encode target if available
    label_encoder = None
    if target_col and target_col in df.columns:
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col].astype(str))

    return df, label_encoder


# -------------------------------
# Train Model on Synthetic Dataset
# -------------------------------
def train_model():
    try:
        df = pd.read_csv("synthetic_master_dataset.csv")
        print("\n✅ Synthetic dataset loaded successfully")

        df_processed, label_encoder = preprocess_dataset(df, target_col="disease")

        X = df_processed.drop(columns=["disease"])
        y = df_processed["disease"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Save model + encoder
        joblib.dump(model, "disease_model.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        joblib.dump(imputer, "imputer.pkl")  # important!

        print("\n✅ Model and label encoder saved successfully")

    except FileNotFoundError:
        print("\n⚠ synthetic_disease_dataset.csv not found. Please generate or place it here.")


# -------------------------------
# Predict Disease from Sensor Data
# -------------------------------
def predict_from_sensor(sensor_input):

    # Load saved model, encoder, imputer
    model = joblib.load("disease_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    imputer = joblib.load("imputer.pkl")
    # Get the list of feature names from the imputer
    expected_features = imputer.feature_names_in_
    
    # Build row with all expected features
    row = []
    for col in expected_features:
        val = sensor_input.get(col, np.nan)
        if isinstance(val, list) and len(val) > 0:
            val = val[0]
        row.append(val)

    df_processed = pd.DataFrame([row], columns=expected_features)

    # Impute missing values
    df_processed = pd.DataFrame(
        imputer.transform(df_processed),
        columns=expected_features
    )

    prediction = model.predict(df_processed)[0]
    disease = label_encoder.inverse_transform([prediction])[0]

    # Get confidence score
    proba = model.predict_proba(df_processed)[0]
    confidence = np.max(proba)

    print(f"Predicted Disease: {disease}")
    return disease, confidence

# -------------------------------
# Run Training + Example Test
# -------------------------------
if __name__ == "__main__":
    # Step 1: Train Model
    train_model()
    # Step 2: Test with example sensor input
    sensor_input = {
    "Glucose": [100],
    "Cholesterol": [200],
    "Hemoglobin": [15],
    "Platelets": [140],
    # add other params if available...
    }
    predict_from_sensor(sensor_input)


    
