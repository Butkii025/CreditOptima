"""
Core data/ML logic for the Loan Approval Predictor.

This mirrors the preprocessing + modeling steps from the original notebook:
1. Impute missing values (mean for numeric, most-frequent for categorical)
2. Feature engineering: DTI_Ratio_sq, Credit_Score_sq (raw DTI_Ratio / Credit_Score dropped)
3. One-hot encode categorical columns (drop first, ignore unknown)
4. Standard-scale numeric columns
5. Train Logistic Regression, KNN, and Naive Bayes; compare metrics
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TARGET_COL = "Loan_Approved"
ID_COL = "Applicant_ID"

# Categorical columns that get one-hot encoded (includes Education_Level,
# which is binary so drop='first' behaves like a 0/1 label encoding)
CATEGORICAL_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

# Raw numeric columns as they appear in the CSV (before feature engineering)
RAW_NUMERIC_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression,
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB,
}


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add DTI_Ratio_sq / Credit_Score_sq and drop the raw columns, matching the notebook."""
    df = df.copy()
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df = df.drop(columns=["Credit_Score", "DTI_Ratio"])
    return df


def fit_pipeline(df: pd.DataFrame, random_state: int = 42):
    """
    Fit imputers / encoder / scaler / models on the full dataset and return an
    'artifacts' dict that the app can reuse for both evaluation and live predictions.
    """
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Rows with a missing target can't be used for training/evaluation
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    numeric_cols = [c for c in RAW_NUMERIC_COLS if c in df.columns]

    # 1. Impute
    num_imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Target
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)

    # 2. Feature engineering
    df = _engineer_features(df)
    engineered_numeric_cols = [c for c in numeric_cols if c not in ("Credit_Score", "DTI_Ratio")]
    engineered_numeric_cols += ["DTI_Ratio_sq", "Credit_Score_sq"]

    # 3. One-hot encode categoricals
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

    X = pd.concat([df[engineered_numeric_cols], encoded_df], axis=1)
    y = df[TARGET_COL]

    feature_order = X.columns.tolist()

    # 4. Train/test split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train all models, collect metrics
    models = {}
    metrics = {}
    for name, ctor in MODEL_REGISTRY.items():
        model = ctor()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        models[name] = model
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
        }

    artifacts = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ohe": ohe,
        "scaler": scaler,
        "models": models,
        "metrics": metrics,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "engineered_numeric_cols": engineered_numeric_cols,
        "feature_order": feature_order,
    }
    return artifacts


def predict_single(raw_input: dict, artifacts: dict, model_name: str):
    """
    raw_input: dict of {column_name: value} using the ORIGINAL raw column names
    (Applicant_Income, Credit_Score, DTI_Ratio, Employment_Status, ...).
    Returns (prediction_label, probability_of_approval).
    """
    row = pd.DataFrame([raw_input])

    numeric_cols = artifacts["numeric_cols"]
    categorical_cols = artifacts["categorical_cols"]

    row[numeric_cols] = artifacts["num_imputer"].transform(row[numeric_cols])
    row[categorical_cols] = artifacts["cat_imputer"].transform(row[categorical_cols])

    row = _engineer_features(row)

    encoded = artifacts["ohe"].transform(row[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=artifacts["ohe"].get_feature_names_out(categorical_cols), index=row.index
    )

    X_row = pd.concat([row[artifacts["engineered_numeric_cols"]], encoded_df], axis=1)
    X_row = X_row.reindex(columns=artifacts["feature_order"], fill_value=0)

    X_scaled = artifacts["scaler"].transform(X_row)

    model = artifacts["models"][model_name]
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]  # probability of class "Yes"/1

    label = "Approved" if pred == 1 else "Not Approved"
    return label, proba
