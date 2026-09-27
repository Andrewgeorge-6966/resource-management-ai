from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Extended_Employee_Performance_and_Productivity_Data.csv"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_DIR))).resolve()
MODEL_PATH = MODEL_DIR / "resource_assignment_model.joblib"
FEATURES = [
    "Training_Hours",
    "Projects_Handled",
    "Overtime_Hours",
    "Sick_Days",
    "Remote_Work_Frequency",
    "Performance_Score",
]


def _minmax(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    scale = max(maximum - minimum, 1e-9)
    return (series - minimum) / scale


def build_resource_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    for column in FEATURES:
        df[column] = _minmax(df[column].astype(float))
    df["Historical_Task_Success"] = (
        0.5 * df["Performance_Score"]
        + 0.3 * df["Training_Hours"]
        + 0.2 * df["Projects_Handled"]
    )
    return df


def main() -> None:
    df = build_resource_dataset()
    x = df[FEATURES]
    y = df["Historical_Task_Success"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    model.required_features_ = list(FEATURES)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved resource recommendation model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
