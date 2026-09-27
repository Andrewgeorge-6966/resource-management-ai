from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
TASK_DATA_PATH = PROJECT_DIR / "CESAW_task_fact.csv"
WBS_DATA_PATH = PROJECT_DIR / "wbs_parent.csv"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR))).resolve()
MODEL_PATH = MODEL_DIR / "task_estimation_model.joblib"
TASK_FEATURES = [
    "project_key",
    "person_key",
    "team_key",
    "wbs_element_key",
    "plan_item_key",
    "task_plan_time_minutes",
    "phase_key",
    "phase_short_name",
    "process_name",
    "parent_wbs_element_key",
    "start_month",
    "start_day_of_week",
    "start_hour",
]
TARGET_COL = "task_actual_time_minutes"


def build_task_dataset() -> pd.DataFrame:
    task_df = pd.read_csv(TASK_DATA_PATH)
    wbs_df = pd.read_csv(WBS_DATA_PATH)
    df = task_df.merge(wbs_df, on=["project_key", "wbs_element_key"], how="left")
    start_dt = pd.to_datetime(df["task_actual_start_date"], errors="coerce")
    df["start_month"] = start_dt.dt.month.fillna(0).astype(int)
    df["start_day_of_week"] = start_dt.dt.dayofweek.fillna(0).astype(int)
    df["start_hour"] = start_dt.dt.hour.fillna(0).astype(int)
    df["parent_wbs_element_key"] = df["parent_wbs_element_key"].fillna(0)
    df["phase_key"] = df["phase_key"].astype(str)
    df["phase_short_name"] = df["phase_short_name"].astype(str)
    df["process_name"] = df["process_name"].astype(str)
    df = df[df["task_plan_time_minutes"].notna()].copy()
    return df


def main() -> None:
    df = build_task_dataset()
    x = df[TASK_FEATURES].copy()
    y = df[TARGET_COL].astype(float)

    cat_cols = ["phase_key", "phase_short_name", "process_name"]
    num_cols = [col for col in TASK_FEATURES if col not in cat_cols]

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
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(x, y)
    model.required_features_ = list(TASK_FEATURES)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved task estimation model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
