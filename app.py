from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from feature_store import get_feature_store, refresh_feature_store
from schemas import (
    BurnoutPredictionRequest,
    CostOverrunPredictionRequest,
    ResourceRecommendationRequest,
    RiskPredictionRequest,
    TaskEstimationRequest,
)


APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Resource Management AI API", version="2.0.0")


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_native(v) for v in value]
    if isinstance(value, tuple):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _predict_classifier(
    model: Any,
    payload: Dict[str, Any],
    *,
    required_features: List[str],
    label_map: Dict[int, str] | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    missing_features = [feature for feature in required_features if feature not in payload]
    extra_features = sorted([feature for feature in payload if feature not in required_features])
    row = {feature: payload.get(feature) for feature in required_features}
    input_df = pd.DataFrame([row])

    try:
        probabilities = model.predict_proba(input_df)[0].tolist() if hasattr(model, "predict_proba") else None
        classes = getattr(model, "classes_", None)
        prediction_raw = model.predict(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    prediction = _native(prediction_raw)
    response: Dict[str, Any] = {
        "prediction": prediction,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "required_features": required_features,
    }

    if label_map:
        response["prediction_label"] = label_map.get(int(prediction), str(prediction))

    if classes is not None and probabilities is not None:
        raw_classes = classes.tolist() if hasattr(classes, "tolist") else list(classes)
        native_classes = [_native(item) for item in raw_classes]
        response["classes"] = native_classes
        response["probabilities"] = probabilities
        response["class_probabilities"] = {
            label_map.get(int(cls), str(cls)) if label_map else str(cls): float(probabilities[idx])
            for idx, cls in enumerate(native_classes)
        }

        if len(native_classes) == 2:
            positive_class = 1 if 1 in native_classes else native_classes[-1]
            positive_index = native_classes.index(positive_class)
            positive_probability = float(probabilities[positive_index])
            response["positive_class"] = positive_class
            response["positive_probability"] = positive_probability
            effective_threshold = threshold if threshold is not None else getattr(model, "best_threshold_", None)
            if effective_threshold is not None:
                response["threshold"] = float(effective_threshold)
                response["prediction_at_threshold"] = int(positive_probability >= float(effective_threshold))

    return _native(response)


@app.on_event("startup")
def startup() -> None:
    get_feature_store()


@app.get("/")
def read_root() -> Dict[str, Any]:
    store = get_feature_store()
    return {
        "message": "Resource Management AI API is running.",
        "features": store.feature_summary(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    store = get_feature_store()
    return {
        "status": "ok",
        "features": store.feature_summary(),
    }


@app.get("/features")
def list_features() -> List[Dict[str, Any]]:
    return get_feature_store().feature_catalog()


@app.post("/reload-models")
def reload_models() -> Dict[str, Any]:
    store = refresh_feature_store()
    return {
        "status": "reloaded",
        "features": store.feature_summary(),
    }


@app.post("/predict/risk")
def predict_risk(request: RiskPredictionRequest) -> Dict[str, Any]:
    store = get_feature_store()
    return _predict_classifier(
        store.risk_model,
        request.features,
        required_features=store.risk_features,
        label_map=store.risk_label_map,
    )


@app.post("/predict/burnout")
def predict_burnout(request: BurnoutPredictionRequest) -> Dict[str, Any]:
    store = get_feature_store()
    return _predict_classifier(
        store.burnout_model,
        request.features,
        required_features=store.burnout_features,
        threshold=store.burnout_threshold,
    )


@app.post("/predict/cost-overrun")
def predict_cost_overrun(request: CostOverrunPredictionRequest) -> Dict[str, Any]:
    store = get_feature_store()
    prediction = _predict_classifier(
        store.cost_overrun_model,
        request.features,
        required_features=store.cost_overrun_features,
    )

    positive_probability = prediction.get("positive_probability")
    if positive_probability is not None:
        if positive_probability >= 0.75:
            band = "High"
        elif positive_probability >= 0.45:
            band = "Medium"
        else:
            band = "Low"
        prediction["overrun_risk_band"] = band

    prediction["feature"] = "cost_overrun_prediction"
    return prediction


@app.post("/predict/task-estimation")
def predict_task_estimation(request: TaskEstimationRequest) -> Dict[str, Any]:
    store = get_feature_store()
    return _native(store.predict_task_estimation(request))


@app.post("/recommend/resources")
def recommend_resources(request: ResourceRecommendationRequest) -> Dict[str, Any]:
    store = get_feature_store()
    return _native(store.recommend_resources(request))


@app.get("/models")
def list_models() -> List[Dict[str, Any]]:
    store = get_feature_store()
    return store.feature_catalog()
