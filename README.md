# Resource Management AI API

An end-to-end machine-learning service that exposes five resource- and project-management capabilities through one FastAPI application: delivery-risk classification, employee burnout-risk prediction, cost-overrun prediction, task-duration estimation, and resource recommendation.

> **Internship project:** I developed this project during my internship in the **AI Engineering section**. My work focused on translating resource-management requirements into reproducible ML pipelines, packaging the models behind validated API contracts, and making the complete service deployable with Docker.

## Highlights

- Serves five ML capabilities from one versioned FastAPI application
- Uses reusable scikit-learn and XGBoost preprocessing/training pipelines
- Generates missing model artifacts automatically from the bundled datasets
- Returns probabilities, risk bands, ranking scores, and input diagnostics
- Includes interactive OpenAPI documentation and a live API smoke-test script
- Supports local Python execution and containerized deployment

## AI Capabilities

| Capability | Approach | Endpoint | Output |
| --- | --- | --- | --- |
| Delivery risk | XGBoost multiclass classification | `POST /predict/risk` | Risk class and class probabilities |
| Burnout risk | Random Forest binary classification | `POST /predict/burnout` | Probability, threshold, and prediction |
| Cost overrun | Random Forest binary classification | `POST /predict/cost-overrun` | Probability and Low/Medium/High band |
| Task estimation | Random Forest regression | `POST /predict/task-estimation` | Predicted duration and schedule-risk band |
| Resource recommendation | Random Forest regression plus weighted ranking | `POST /recommend/resources` | Ranked employee-resource candidates |

The API also provides `GET /health`, `GET /features`, `GET /models`, and `POST /reload-models`.

## Architecture

```text
Client or project-management backend
                |
                | JSON over HTTP
                v
          FastAPI service
                |
      +---------+----------+
      |                    |
Pydantic validation   FeatureStore
                           |
              +------------+-------------+
              |                          |
       trained pipelines          ranking logic
              |
     bundled tabular datasets
```

## Technology Stack

| Area | Technologies |
| --- | --- |
| API | Python, FastAPI, Uvicorn, Pydantic |
| Data | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, joblib |
| Deployment | Docker, Docker Compose |
| Validation | Live HTTP smoke tests, OpenAPI/Swagger UI |

## Project Structure

```text
resource-management-ai/
├── app.py                              # API routes and prediction responses
├── schemas.py                          # Request validation models
├── feature_store.py                    # Model loading and inference orchestration
├── bootstrap_models.py                 # Builds any missing model artifacts
├── train_*.py                          # Reproducible training pipelines
├── test_api.py                         # Live endpoint smoke tests
├── *.ipynb                             # Model experiments and comparisons
├── *.csv                               # Bundled demonstration/training datasets
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Generated `.joblib` and `.pkl` artifacts are intentionally excluded from Git. Running the bootstrap command recreates them locally.

## Getting Started

### Prerequisites

- Python 3.11 recommended
- `pip`
- Docker Desktop, optional

### Local setup

```bash
git clone https://github.com/Andrewgeorge-6966/resource-management-ai.git
cd resource-management-ai

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python bootstrap_models.py
uvicorn app:app --reload
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

No API keys or credentials are required. Real environment files are ignored; `.env.example` documents the optional `MODEL_DIR` setting for shell or container use.

### Docker

```bash
docker compose up --build
```

The container installs the dependencies, builds any missing model artifacts, starts the API on port `8000`, and exposes a health check.

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/recommend/resources \
  -H "Content-Type: application/json" \
  -d '{
    "required_skill_level": 0.7,
    "required_experience_level": 0.6,
    "required_performance_level": 0.8,
    "top_n": 5
  }'
```

Example response shape:

```json
{
  "count": 5,
  "recommendations": [
    {
      "Employee_ID": 123,
      "Department": "IT",
      "Job_Title": "Developer",
      "Final_Grade": 96.4,
      "Profile_Score": 0.91,
      "Availability_Score": 0.82,
      "Historical_Performance": 0.88,
      "ML_Success": 0.9
    }
  ]
}
```

## Validation

With the API running in one terminal, execute the smoke suite in another:

```bash
python test_api.py
```

The script checks service health, model readiness, all five prediction/recommendation endpoints, and their essential response contracts.

## Data and Model Notes

- The checked-in datasets use numeric or project-style identifiers and contain no names, email addresses, passwords, or API credentials.
- Models are trained for demonstration and portfolio use; the repository does not claim production benchmark performance.
- Training scripts preserve preprocessing with each estimator to keep inference consistent.
- The burnout feature is decision support only and must not be treated as a medical assessment or used as the sole basis for employment decisions.

## Production Considerations

Before production deployment, add authentication, rate limiting, audit logging, monitoring, model/version registry support, data-drift checks, bias evaluation, retraining controls, and organization-specific validation. Restrict or remove the model-reload endpoint in an internet-facing deployment.

## License

No open-source license is included because the original project license identified a product owner that has not been confirmed for this portfolio publication. The repository is shared for portfolio review; all rights remain with their respective owners.
