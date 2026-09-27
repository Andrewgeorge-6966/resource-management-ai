#!/usr/bin/env python3
"""Run a live smoke test against the local Resource Management AI API."""

from __future__ import annotations

import json
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 30


def request(method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.request(
        method,
        f"{BASE_URL}{endpoint}",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    print(f"PASS {method} {endpoint}")
    return body


def main() -> None:
    health = request("GET", "/health")
    assert health["status"] == "ok"

    features = request("GET", "/features")
    assert len(features) == 5
    assert all(feature["ready"] for feature in features)

    risk = request(
        "POST",
        "/predict/risk",
        {
            "features": {
                "Project_Type": "Infrastructure",
                "Team_Size": 8,
                "Project_Budget_USD": 500000,
                "Estimated_Timeline_Months": 12,
                "Complexity_Score": 7,
                "Team_Experience_Level": "High",
                "External_Dependencies_Count": 4,
                "Project_Phase": "Execution",
                "Team_Turnover_Rate": 0.1,
                "Technology_Familiarity": "Familiar",
                "Stakeholder_Engagement_Level": 0.8,
                "Budget_Utilization_Rate": 0.9,
                "Resource_Availability": 0.7,
                "Priority_Level": "High",
                "Previous_Delivery_Success_Rate": 0.75,
                "Project_Manager_Experience": 8,
                "Org_Process_Maturity": 0.7,
                "Key_Stakeholder_Availability": 0.8,
                "Resource_Contention_Level": "Medium",
                "Industry_Volatility": 0.3,
                "Change_Control_Maturity": 0.7,
                "Risk_Management_Maturity": 0.8,
                "Project_Start_Month": 4,
                "Current_Phase_Duration_Months": 2,
            }
        },
    )
    assert risk["prediction_label"] in {"Low", "Medium", "High", "Critical"}

    burnout = request(
        "POST",
        "/predict/burnout",
        {
            "features": {
                "EmployeeID": 10001,
                "Age": 32,
                "Gender": "Male",
                "Country": "Egypt",
                "JobRole": "Software Engineer",
                "Department": "IT",
                "YearsAtCompany": 4,
                "WorkHoursPerWeek": 45,
                "RemoteWork": "Hybrid",
                "JobSatisfaction": 6.5,
                "StressLevel": 5.0,
                "ProductivityScore": 7.0,
                "SleepHours": 7.0,
                "PhysicalActivityHrs": 3.0,
                "CommuteTime": 30,
                "HasMentalHealthSupport": "Yes",
                "ManagerSupportScore": 7.0,
                "HasTherapyAccess": "No",
                "MentalHealthDaysOff": 1,
                "SalaryRange": "60K-80K",
                "WorkLifeBalanceScore": 6.0,
                "TeamSize": 8,
                "CareerGrowthScore": 7.0,
            }
        },
    )
    assert "positive_probability" in burnout

    cost = request(
        "POST",
        "/predict/cost-overrun",
        {
            "features": {
                "Team_Size": 10,
                "Project_Budget_USD": 750000,
                "Estimated_Timeline_Months": 18,
                "Complexity_Score": 7.5,
                "Stakeholder_Count": 12,
                "Past_Similar_Projects": 3,
                "External_Dependencies_Count": 4,
                "Change_Request_Frequency": 1.2,
                "Project_Phase": "Execution",
                "Team_Experience_Level": "Senior",
                "Historical_Risk_Incidents": 2,
                "Technology_Familiarity": "Familiar",
                "Schedule_Pressure": 0.7,
                "Market_Volatility": 0.4,
                "Resource_Availability": 0.75,
                "Priority_Level": "High",
                "Previous_Delivery_Success_Rate": 0.8,
                "Resource_Contention_Level": "Medium",
                "Project_Start_Month": 5,
                "Current_Phase_Duration_Months": 4,
            }
        },
    )
    assert cost["overrun_risk_band"] in {"Low", "Medium", "High"}

    task = request(
        "POST",
        "/predict/task-estimation",
        {
            "task_plan_time_minutes": 120,
            "phase_key": "337",
            "phase_short_name": "DOC-Insp",
            "process_name": "Process-D",
        },
    )
    assert task["predicted_task_time_minutes"] >= 0

    recommendations = request(
        "POST",
        "/recommend/resources",
        {
            "required_skill_level": 0.7,
            "required_experience_level": 0.6,
            "required_performance_level": 0.8,
            "top_n": 3,
        },
    )
    assert recommendations["count"] == 3

    print(json.dumps({"status": "all smoke tests passed", "features": 5}, indent=2))


if __name__ == "__main__":
    main()
