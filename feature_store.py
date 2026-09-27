from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from schemas import ResourceRecommendationRequest, TaskEstimationRequest
from train_burnout_model import (
    OUTPUT_PATH as BURNOUT_MODEL_PATH,
    main as train_burnout_model,
)
from train_cost_overrun_model import (
    FEATURES as COST_OVERRUN_FEATURES,
    OUTPUT_PATH as COST_OVERRUN_MODEL_PATH,
    main as train_cost_overrun_model,
)
from train_resource_recommendation_model import (
    FEATURES as RESOURCE_FEATURES,
    MODEL_PATH as RESOURCE_MODEL_PATH,
    build_resource_dataset,
    main as train_resource_model,
)
from train_risk_model import (
    FEATURES as RISK_FEATURES,
    OUTPUT_PATH as RISK_MODEL_PATH,
    main as train_risk_model,
)
from train_task_estimation_model import (
    MODEL_PATH as TASK_ESTIMATION_MODEL_PATH,
    TASK_FEATURES,
    build_task_dataset,
    main as train_task_estimation_model,
)


PROJECT_DIR = Path(__file__).resolve().parent
RESOURCE_DATA_PATH = PROJECT_DIR / "Extended_Employee_Performance_and_Productivity_Data.csv"


@dataclass
class FeatureStore:
    risk_model: Any
    burnout_model: Any
    cost_overrun_model: Any
    task_estimation_model: Any
    resource_model: Any
    resource_df: pd.DataFrame
    task_df: pd.DataFrame
    risk_features: List[str]
    burnout_features: List[str]
    burnout_threshold: float | None
    cost_overrun_features: List[str]
    task_estimation_features: List[str]
    resource_features: List[str]
    risk_label_map: Dict[int, str]

    def feature_summary(self) -> Dict[str, Dict[str, Any]]:
        return {
            item["key"]: {
                "ready": item["ready"],
                "artifact": item["artifact"],
            }
            for item in self.feature_catalog()
        }

    def feature_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "risk_prediction",
                "ready": self.risk_model is not None,
                "artifact": RISK_MODEL_PATH.name,
                "required_features": self.risk_features,
            },
            {
                "key": "burnout_prediction",
                "ready": self.burnout_model is not None,
                "artifact": BURNOUT_MODEL_PATH.name,
                "required_features": self.burnout_features,
            },
            {
                "key": "cost_overrun_prediction",
                "ready": self.cost_overrun_model is not None,
                "artifact": COST_OVERRUN_MODEL_PATH.name,
                "required_features": self.cost_overrun_features,
            },
            {
                "key": "task_estimation",
                "ready": self.task_estimation_model is not None,
                "artifact": TASK_ESTIMATION_MODEL_PATH.name,
                "required_features": self.task_estimation_features,
            },
            {
                "key": "resource_recommendation",
                "ready": self.resource_model is not None,
                "artifact": RESOURCE_MODEL_PATH.name,
                "required_features": self.resource_features,
            },
        ]

    def predict_task_estimation(self, request: TaskEstimationRequest) -> Dict[str, Any]:
        start_date = pd.to_datetime(request.task_start_date) if request.task_start_date else pd.Timestamp.utcnow()
        row = {
            "project_key": request.project_key,
            "person_key": request.person_key,
            "team_key": request.team_key,
            "wbs_element_key": request.wbs_element_key,
            "plan_item_key": request.plan_item_key,
            "task_plan_time_minutes": request.task_plan_time_minutes,
            "phase_key": request.phase_key,
            "phase_short_name": request.phase_short_name,
            "process_name": request.process_name,
            "parent_wbs_element_key": request.parent_wbs_element_key,
            "start_month": start_date.month,
            "start_day_of_week": start_date.dayofweek,
            "start_hour": start_date.hour,
        }

        input_df = pd.DataFrame([row])
        predicted_minutes = float(self.task_estimation_model.predict(input_df)[0])
        baseline_ratio = float(predicted_minutes / request.task_plan_time_minutes)
        matching_tasks = self.task_df[
            (self.task_df["phase_short_name"] == request.phase_short_name)
            & (self.task_df["process_name"] == request.process_name)
        ]
        historical_median = (
            float(matching_tasks["task_actual_time_minutes"].median())
            if not matching_tasks.empty
            else None
        )
        return {
            "predicted_task_time_minutes": round(predicted_minutes, 2),
            "predicted_task_time_hours": round(predicted_minutes / 60.0, 2),
            "planned_task_time_minutes": request.task_plan_time_minutes,
            "estimated_overrun_minutes": round(predicted_minutes - request.task_plan_time_minutes, 2),
            "estimated_overrun_ratio": round(baseline_ratio, 4),
            "historical_median_minutes_for_phase_process": historical_median,
            "schedule_risk": "High" if baseline_ratio >= 1.5 else "Medium" if baseline_ratio >= 1.1 else "Low",
        }

    def recommend_resources(self, request: ResourceRecommendationRequest) -> Dict[str, Any]:
        df = self.resource_df.copy()

        if request.department:
            df = df[df["Department"].astype(str).str.lower() == request.department.lower()]
        if request.job_title:
            df = df[df["Job_Title"].astype(str).str.lower() == request.job_title.lower()]
        if request.max_work_hours_per_week is not None:
            df = df[df["Work_Hours_Per_Week"] <= request.max_work_hours_per_week]
        if df.empty:
            return {"recommendations": [], "count": 0}

        df["ML_Success"] = self.resource_model.predict(df[self.resource_features])
        df["Skill_Match"] = 1 - (df["Training_Hours"] - request.required_skill_level).abs()
        df["Experience_Match"] = 1 - (df["Projects_Handled"] - request.required_experience_level).abs()
        df["Performance_Match"] = 1 - (df["Performance_Score"] - request.required_performance_level).abs()
        df["Profile_Score"] = (
            0.5 * df["Skill_Match"].clip(0, 1)
            + 0.3 * df["Experience_Match"].clip(0, 1)
            + 0.2 * df["Performance_Match"].clip(0, 1)
        )
        df["Availability_Score"] = (
            1
            - (
                0.4 * df["Projects_Handled"]
                + 0.3 * df["Overtime_Hours"]
                + 0.3 * df["Sick_Days"]
            )
        ).clip(0, 1)
        df["Historical_Performance"] = (
            0.6 * df["Performance_Score"] + 0.4 * df["Projects_Handled"]
        ).clip(0, 1)
        df["Final_Grade"] = (
            0.30 * df["Profile_Score"]
            + 0.25 * df["Availability_Score"]
            + 0.25 * df["Historical_Performance"]
            + 0.20 * df["ML_Success"]
        )
        min_grade = float(df["Final_Grade"].min())
        max_grade = float(df["Final_Grade"].max())
        scale = max(max_grade - min_grade, 1e-9)
        df["Final_Grade"] = ((df["Final_Grade"] - min_grade) / scale) * 100.0
        top = df.sort_values("Final_Grade", ascending=False).head(request.top_n)

        records = top[
            [
                "Employee_ID",
                "Department",
                "Job_Title",
                "Final_Grade",
                "Profile_Score",
                "Availability_Score",
                "Historical_Performance",
                "ML_Success",
            ]
        ].to_dict(orient="records")
        return {"count": len(records), "recommendations": records}


def _load_or_train_model(path: Path, trainer: Any) -> Any:
    if not path.exists():
        trainer()

    try:
        return joblib.load(path)
    except Exception:
        trainer()
        return joblib.load(path)


def _normalize_feature_list(model: Any, fallback: List[str]) -> List[str]:
    if hasattr(model, "required_features_"):
        return list(getattr(model, "required_features_"))
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    return fallback


_STORE: FeatureStore | None = None


def get_feature_store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        risk_model = _load_or_train_model(RISK_MODEL_PATH, train_risk_model)
        burnout_model = _load_or_train_model(BURNOUT_MODEL_PATH, train_burnout_model)
        cost_overrun_model = _load_or_train_model(COST_OVERRUN_MODEL_PATH, train_cost_overrun_model)
        task_estimation_model = _load_or_train_model(TASK_ESTIMATION_MODEL_PATH, train_task_estimation_model)
        resource_model = _load_or_train_model(RESOURCE_MODEL_PATH, train_resource_model)
        resource_df = build_resource_dataset()
        task_df = build_task_dataset()

        _STORE = FeatureStore(
            risk_model=risk_model,
            burnout_model=burnout_model,
            cost_overrun_model=cost_overrun_model,
            task_estimation_model=task_estimation_model,
            resource_model=resource_model,
            resource_df=resource_df,
            task_df=task_df,
            risk_features=_normalize_feature_list(risk_model, list(RISK_FEATURES)),
            burnout_features=_normalize_feature_list(burnout_model, []),
            burnout_threshold=getattr(burnout_model, "best_threshold_", None),
            cost_overrun_features=_normalize_feature_list(cost_overrun_model, list(COST_OVERRUN_FEATURES)),
            task_estimation_features=list(TASK_FEATURES),
            resource_features=list(RESOURCE_FEATURES),
            risk_label_map={0: "Low", 1: "Medium", 2: "High", 3: "Critical"},
        )
    return _STORE


def refresh_feature_store() -> FeatureStore:
    global _STORE
    _STORE = None
    return get_feature_store()
