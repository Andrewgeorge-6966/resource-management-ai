import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "project_risk_raw_dataset.csv"
TARGET_COL = "Risk_Level"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR))).resolve()
OUTPUT_PATH = MODEL_DIR / "risk_prediction_model.pkl"

# Keep the same feature list shape your API/model was using before
FEATURES = [
    "Project_Type",
    "Team_Size",
    "Project_Budget_USD",
    "Estimated_Timeline_Months",
    "Complexity_Score",
    "Team_Experience_Level",
    "External_Dependencies_Count",
    "Project_Phase",
    "Team_Turnover_Rate",
    "Technology_Familiarity",
    "Stakeholder_Engagement_Level",
    "Budget_Utilization_Rate",
    "Resource_Availability",
    "Priority_Level",
    "Previous_Delivery_Success_Rate",
    "Project_Manager_Experience",
    "Org_Process_Maturity",
    "Key_Stakeholder_Availability",
    "Resource_Contention_Level",
    "Industry_Volatility",
    "Change_Control_Maturity",
    "Risk_Management_Maturity",
    "Project_Start_Month",
    "Current_Phase_Duration_Months",
]

RISK_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Basic cleanup
    df = df.dropna(subset=[TARGET_COL])
    df = df.copy()

    y = df[TARGET_COL].map(RISK_MAP)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), TARGET_COL].unique().tolist())
        raise ValueError(f"Unknown Risk_Level values found: {unknown}")
    y = y.astype(int)

    X = df[FEATURES].copy()

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in FEATURES if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    # Store metadata for the API
    pipeline.required_features_ = FEATURES

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved risk prediction pipeline to {OUTPUT_PATH}")
    print("classes_:", getattr(pipeline, "classes_", None))


if __name__ == "__main__":
    main()
