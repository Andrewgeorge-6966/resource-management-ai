from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "project_risk_raw_dataset.csv"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR))).resolve()
OUTPUT_PATH = MODEL_DIR / "cost_overrun_model.joblib"
TARGET_COL = "Cost_Overrun"
FEATURES = [
    "Team_Size",
    "Project_Budget_USD",
    "Estimated_Timeline_Months",
    "Complexity_Score",
    "Stakeholder_Count",
    "Past_Similar_Projects",
    "External_Dependencies_Count",
    "Change_Request_Frequency",
    "Project_Phase",
    "Team_Experience_Level",
    "Historical_Risk_Incidents",
    "Technology_Familiarity",
    "Schedule_Pressure",
    "Market_Volatility",
    "Resource_Availability",
    "Priority_Level",
    "Previous_Delivery_Success_Rate",
    "Resource_Contention_Level",
    "Project_Start_Month",
    "Current_Phase_Duration_Months",
]


def build_cost_overrun_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df[TARGET_COL] = (df["Budget_Utilization_Rate"] > 1.0).astype(int)
    return df


def main() -> None:
    df = build_cost_overrun_dataset()
    x = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    cat_cols = x.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [col for col in FEATURES if col not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(x, y)
    model.required_features_ = list(FEATURES)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved cost overrun model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
