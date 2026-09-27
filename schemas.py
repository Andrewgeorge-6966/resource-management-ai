from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    features: Dict[str, Any]


class BurnoutPredictionRequest(BaseModel):
    features: Dict[str, Any]


class CostOverrunPredictionRequest(BaseModel):
    features: Dict[str, Any]


class TaskEstimationRequest(BaseModel):
    project_key: int = 0
    person_key: int = 0
    team_key: int = 0
    wbs_element_key: int = 0
    plan_item_key: int = 0
    task_plan_time_minutes: float = Field(..., gt=0)
    phase_key: str = "unknown"
    phase_short_name: str = "unknown"
    process_name: str = "unknown"
    parent_wbs_element_key: float = 0.0
    task_start_date: Optional[str] = None


class ResourceRecommendationRequest(BaseModel):
    required_skill_level: float = Field(..., ge=0.0, le=1.0)
    required_experience_level: float = Field(..., ge=0.0, le=1.0)
    required_performance_level: float = Field(..., ge=0.0, le=1.0)
    top_n: int = Field(default=5, ge=1, le=50)
    department: Optional[str] = None
    job_title: Optional[str] = None
    max_work_hours_per_week: Optional[float] = Field(default=None, gt=0)
