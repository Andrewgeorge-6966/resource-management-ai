import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "mental_health_workplace_survey.csv"
TARGET_COL = "BurnoutRisk"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR))).resolve()
OUTPUT_PATH = MODEL_DIR / "burnout_risk_pipeline.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    data = df.copy()
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    # Drop target and BurnoutLevel so the model never sees it (fixes data leakage)
    cols_to_drop = [TARGET_COL] + [c for c in ["BurnoutLevel"] if c in data.columns]
    X = data.drop(columns=cols_to_drop)
    y = data[TARGET_COL]

    # EmployeeID: keep in X for display, but do not use as a model feature
    numeric_features = [
        c for c in X.select_dtypes(include=[np.number]).columns.tolist() if c != "EmployeeID"
    ]
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # A strong, stable baseline classifier
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Threshold tuning to maximize F1 on the test split (same idea as the notebook)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_f1 = 0.0
    best_threshold = 0.5
    for thresh in np.arange(0.20, 0.55, 0.02):
        y_pred_th = (y_proba >= thresh).astype(int)
        f1_th = f1_score(y_test, y_pred_th, zero_division=0)
        if f1_th >= best_f1:
            best_f1 = float(f1_th)
            best_threshold = float(thresh)

    # Store deployment metadata directly on the pipeline so the API can use it
    pipeline.best_threshold_ = best_threshold
    pipeline.positive_class_ = 1
    pipeline.required_features_ = list(X.columns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved burnout pipeline to {OUTPUT_PATH}")
    print(f"best_threshold_={best_threshold:.2f} best_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
