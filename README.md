# IT Job Salary Prediction

Machine learning project for predicting IT job salaries based on job title, seniority, location and technology stack.

The project combines machine learning with API serving, benchmarking and monitoring utilities.

The system exposes three interfaces:
* REST API (FastAPI)
* CLI tools
* Interactive UI (Streamlit)

# Business Problem

IT salaries vary significantly, depending on many interacting factors:
* seniority level,
* technology stack,
* specialization,
* location,
* market demand.

The objective of this project is to:
* identify key factors influencing salary levels,
* build a model predicting salary,
* evaluate model performance and interpret feature importance (supported for Random Forest models),
* expose the model through production-oriented interfaces.

In addition to model training, the project includes API serving, benchmarking and monitoring components.


# Quick Demo

```bash
git clone https://github.com/edytuj/it_job_salary_prediction.git
cd it_job_salary_prediction

conda create -n salary python=3.12
conda activate salary

pip install -r requirements.txt
pip install -e .

predict-salary \
  --title "python developer" \
  --skills "python,aws" \
  --city "Warszawa" \
  --seniority "mid"
```


# Example Prediction

Input:

```json
{
  "title": "python developer",
  "skills": ["python", "aws"],
  "city": "Warszawa",
  "seniority": "mid"
}
```

Example response:

```json
{
  "prediction": "20 000 PLN",
  "range": ["18 000 PLN", "22 000 PLN"],
  "uncertainty": "1 000 PLN",
  "confidence_absolute": "high",
  "confidence_relative": "medium",
  "method": "rf_variance"
}
```

# Key Features

## Machine Learning

* Salary prediction using multiple regression models
* Unified preprocessing pipeline across API, CLI and UI
* Multiple model support:
  * HistGradientBoostingRegressor
  * RandomForestRegressor
  * Ridge Regression
* Confidence estimation for predictions
* Model comparison and benchmarking

## Engineering & Infrastructure

* FastAPI-based REST API
* Streamlit UI
* Command-line prediction interface
* Benchmark CLI for inference latency testing
* Prometheus metrics exposure
* Health-check endpoint and monitoring script
* Centralized configuration and model loading
* Structured logging
* Docker support
* Automated tests
* Ruff-based linting and formatting


# Architecture

The repository follows a layered architecture separating interfaces, business logic and infrastructure concerns.

```text
┌───────────────────────────────────────┐
│           Interfaces Layer            │
│                                       │
│      FastAPI / CLI / Streamlit        │
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│          Prediction Layer             │
│                                       │
│ preprocessing / inference /           │
│ uncertainty estimation                │
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│       Model & Infrastructure          │
│                                       │
│ model loading / config / paths /      │
│ benchmarking / monitoring             │
└───────────────────────────────────────┘
```


# Project Structure

```text
src/
│
├── api/                 # FastAPI application
├── benchmark/           # Benchmark CLI and reporting
├── config/              # Centralized configuration
├── model/               # Model loading and training
├── prediction/          # Prediction logic
├── preprocessing/       # Data preprocessing (cleaning, features extraction, custom sklearn transformers)
├── scraping/            # Source page scraping for data
└── utils/               # Shared utilities

scripts/
├── health_check.sh      # Operational health-check script

models/                  # Trained models

data/                    # Datasets and processed artifacts

tests/                   # Automated tests

streamlit_app.py         # Streamlit UI
```

# Installation

## 1. Clone repository

```bash
git clone https://github.com/edytuj/it_job_salary_prediction.git
cd it_job_salary_prediction
```

## 2. Create environment

```bash
conda create -n salary python=3.12
conda activate salary
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install project in editable mode for local development

```bash
pip install -e .[dev]
```


# Usage

## 1. CLI Prediction

Run salary prediction directly from terminal:

```bash
predict-salary \
  --title "python developer" \
  --skills "python,aws,docker" \
  --city "Warszawa" \
  --seniority "mid"
```

## 2. REST API

Start FastAPI server:

```bash
uvicorn src.api.app:app --reload
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Example request:

```json
POST /predict
{
  "title": "python developer",
  "skills": ["python", "aws"],
  "city": "Warszawa",
  "seniority": "mid"
}
```

## 3. Streamlit UI

Run UI:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The UI requires the API server to be running.

## 4. Benchmark CLI

Run inference benchmark:

```bash
benchmark run --model hgb --runs 100
```

Show aggregated benchmark results:

```bash
benchmark show
```

The benchmark measures:
* cold-start latency,
* warm inference latency,
* latency variability,
* model-specific inference performance.

Results are stored in CSV format for comparison across runs.


# Docker

## Docker Compose (recommended)

```bash
docker-compose up --build
```

## Manual Docker commands

Build image:

```bash
docker build -t salary-app .
```

Run API:

```bash
docker run -d -p 8000:8000 --name api salary-app
```

Run Streamlit UI:

```bash
docker run -d -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  --name ui \
  salary-app \
  streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```


# Monitoring

The API exposes Prometheus-compatible metrics for:
* request count,
* request latency,
* model inference duration.

The metrics are exposed through the /metrics endpoint.

Operational monitoring includes:
* `/health` endpoint for basic service availability,
* `/ready` endpoint verifying model readiness,
* lightweight shell health-check script,
* structured logging.


# Machine Learning Pipeline

## Dataset

The dataset contains IT job offers scraped from publicly available sources.

Raw job data includes information such as:
* job title,
* seniority,
* city/location,
* required technologies,
* salary range.

## Preprocessing

The preprocessing pipeline includes:
* text cleanup and normalization,
* transliteration,
* missing value handling,
* categorical encoding,
* salary range averaging,
* technology stack processing,
* skill count extraction.


## Model Training and Evaluation

The project compares several regression models trained on approximately 2600 IT job offers.

The training pipeline includes:
* train/test split,
* cross-validation,
* hyperparameter tuning with GridSearchCV,
* comparison against a median-based baseline model.

The evaluated models include:
* Ridge Regression,
* RandomForestRegressor,
* HistGradientBoostingRegressor.

Hyperparameter tuning was performed separately for each model family.

### Evaluation Metrics

The project evaluates models using both:
* MAE (Mean Absolute Error),
* R² score.

MAE was treated as the primary optimization metric during hyperparameter tuning because it provides an interpretable estimate of average salary prediction error in PLN.

R² was used as a complementary metric for overall model fit.

### Results Summary

The median baseline achieved MAE of approximately 4900 PLN.

The best-performing model was HistGradientBoostingRegressor with:
* MAE ≈ 4680 PLN,
* R² ≈ 0.09,
* the best cross-validation performance among tested models.

The project also compares inference latency and uncertainty estimation characteristics across models.


## Uncertainty Estimation

The project supports two uncertainty estimation approaches.

### Variance-based uncertainty

Used for ensemble models such as Random Forest.

Prediction uncertainty is estimated using variance across individual trees.

This produces input-dependent uncertainty estimates that vary between predictions.

### MAE-based fallback estimation

For models without native uncertainty support, the system uses validation MAE as fallback error estimate.

This provides a simple and consistent uncertainty approximation across models, although the estimate is global rather than input-specific.

## Confidence Estimation

Two complementary confidence measures are exposed.

### Absolute confidence

Based on prediction spread.

Indicates how wide the predicted salary range is.

### Relative confidence

Based on uncertainty normalized by predicted salary value.

This helps distinguish between predictions with similar absolute uncertainty but different salary scales.

## Model Management

Model loading is centralized through a shared `get_model()` function used by the API, CLI and benchmarking tools.

Loaded models are cached in memory to avoid repeated disk loading and reduce inference startup overhead.

The application supports both:
* local model loading,
* automatic model download from GitHub Releases.

This allows running the project both locally and inside Docker containers without manually managing model artifacts.

Model selection is configurable through application settings.


# Benchmarking

The repository includes a dedicated benchmarking subsystem.

Goals:
* compare model inference latency,
* measure cold vs warm startup performance,
* observe latency stability,
* support model selection decisions.

Benchmark results are persisted to CSV files for historical comparison.


# Testing

The project includes tests for:
* API endpoints,
* preprocessing,
* prediction logic,
* transformers,
* benchmark reporting.

Run tests:

```bash
pytest
```

# Code Quality

The repository uses:
* Ruff for linting and import organization,
* type hints,
* structured logging,
* pytest-based tests,
* GitHub Actions CI for automated test execution and Docker build validation on pushes to the main branch.


# Dataset and Model Versioning

The project uses DVC for lightweight dataset and model artifact versioning.


# Future Improvements

Potential future directions:
* probabilistic models,
* quantile regression,
* model calibration,
* cloud deployment,
* experiment tracking,
* automated retraining pipeline.


# Tech Stack

## Machine Learning

* Scikit-learn
* Pandas
* NumPy

## Backend & Interfaces

* FastAPI
* Streamlit

## Tooling

* Pytest
* Ruff
* Docker
* Prometheus
* DVC


# Author

Agnieszka Zambrzycka
