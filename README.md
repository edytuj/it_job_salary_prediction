# IT Job Salary Prediction

Machine learning project predicting IT job salaries based on role, experience level, location and tech stack.
The goal of the project was to explore salary determinants in the IT market and build a predictive regression model.

The project provides multiple interfaces:
- CLI tool
- REST API (FastAPI)
- Interactive UI (Streamlit)

## Business Problem

IT salaries vary significantly depending on experience, location and technology stack.
The objective of this project is to:

- identify key factors influencing salary levels,
- build a model predicting salary,
- evaluate model performance and interpret feature importance.

## Features

- Salary prediction using trained ML models
- Confidence estimation:
    - absolute (based on prediction range)
    - relative (based on uncertainty)
- Support for multiple model types (e.g. Random Forest, Gradient Boosting)
- Benchmarking for model performance (latency)
- Unified model loading across API, CLI, and UI

## Project Structure

src/
  |
  - api/ # FastAPI application
  - model/ # model loading (centralized)
  - prediction/ # prediction logic
  - utils/ # utilities (paths, formatting, etc.)
scripts/
  |
  - benchmark.py # performance benchmarking
  - tests/ # tests for API, preprocessing procesures, pipeline, prediction, transformers
 models/ # trained models
 data/ # datasets
 streamlit_app.py # UI

## Installation

1. Clone the repository:

```bash
git clone edytuj/it_job_salary_prediction
cd it_job_salary_prediction
```

2. Create and activate environment with conda:

```bash
conda create -n salary python=3.12
conda activate salary
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the project in editable mode:

```bash
pip install -e .
```

This allows you to:
- run CLI commands (e.g. predict-salary)
- import project modules without modifying PYTHONPATH
- develop the project without reinstalling after changes


## Docker
1. Build image:
```bash
docker build -t salary-api .
```
2. Run container:
```bash
docker run -p 8000:8000 salary-api
```

## Usage

The project exposes three ways to interact with the model:

1. CLI

Run prediction directly from the command line:

```bash
predict-salary \ --title "python developer" \ --skills "python,aws,docker" \ --city "Warszawa" \ --seniority "mid"
```

2. API (FastAPI)
Start the server:

```bash
uvicorn src.api.app:app --reload
```

Interactive documentation (Swagger):
http://localhost:8000/docs

Examplary request:
```json
POST /predict
{
  "title": "python developer",
  "skills": ["python", "aws"],
  "city": "Warszawa",
  "seniority": "mid"
}
```

Examplary response:
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

3. UI (Streamlit)

Run interactive interface:

```bash
streamlit run streamlit_app.py
```

Then open:
http://localhost:8501

4. Benchmark

Measure model inference performance:

```bash
python scripts/benchmark.py run
```

View saved results:

```bash
python scripts/benchmark.py show
```

Benchmark includes:
- cold start latency
- warm inference latency
- results stored in CSV for comparison

## Dataset

The dataset contains job offers scraped from publicly available sources and includes:

- Job title
- Experience level
- Location
- Required technologies
- Salary range

Data preprocessing included:
- cleaning missing values,
- encoding categorical variables,
- feature engineering.

## Model and Uncertainty

The system supports two approaches:
1. Model-based (for Random Forest)
- uses variance across trees,
- provides statistical uncertainty estimate

2. Fallback (for other models)
- uses validation MAE as error proxy
- ensures consistent uncertainty estimation

## Confidence Estimation

Two types of confidence are computed:

- absolute — based on prediction spread
- relative — based on normalized uncertainty

## Testing

Run tests:

```bash
pytest
```

## Architecture Overview

The project follows a layered architecture that separates concerns between model logic, infrastructure, and application interfaces.

           |        Interfaces          │
           |     (CLI / API / UI)       │
           
                        │

           │     Prediction Layer       │
           │     (business logic)       │
                        
                        |

           │   Model & Infrastructure   │
           │   (model loading, paths)   │

## Desing Decisions:

- Centralized model loading
    - single source of truth (get_model)
    - shared across API, CLI, UI
- Path management
    - all paths resolved relative to project root
- Benchmarking
    - cold vs warm inference measurement
    - results persisted for comparison
- Uncertainty handling
    - data-driven fallback using MAE instead of fixed values

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Pytest

## Future Improvements
- model registry (multiple models via config)
- deployment (Docker/cloud
- better uncertainty estimation (quantiles/calibration)
- monitoring and logging

## Demo

```bash
git clone edytuj/it_job_salary_prediction
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

## Author

Agnieszka Zambrzycka
