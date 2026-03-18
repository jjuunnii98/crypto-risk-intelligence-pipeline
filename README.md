# Crypto Risk Intelligence Pipeline

## Overview
This project is a production-oriented machine learning pipeline for crypto market risk intelligence.

It is designed to simulate a real-world data and ML system that:
- ingests crypto market data
- preprocesses time-series data
- builds risk-related features
- trains predictive models
- serves risk predictions through an API

The project focuses on transforming raw market signals into risk-aware analytical outputs.

---

## Objective
To build an end-to-end crypto risk intelligence system that integrates:
- market data ingestion
- preprocessing for time-series analysis
- feature engineering for risk signals
- predictive modeling for risk scoring
- API-based inference

This project is structured to demonstrate practical ML engineering and data pipeline design.

---

## Key Features
- End-to-end pipeline for crypto market risk intelligence
- Data ingestion from exchange APIs or local market datasets
- Time-series preprocessing and feature engineering
- Risk-oriented predictive modeling
- FastAPI-based REST API for inference
- Modular and production-ready project structure
- Reproducible experimentation via Jupyter notebooks

---

## Project Structure

```text
src/
├── api/            # FastAPI endpoints
├── data/           # data ingestion, loading, preprocessing
├── features/       # feature engineering
├── models/         # risk model implementation
├── pipelines/      # data, training, and inference pipelines
├── services/       # business logic layer
└── utils/          # utilities (config, IO, etc.)
```

---

## Data Flow

```
Exchange API / Market Data
→ Data Ingestion
→ Preprocessing
→ Feature Engineering
→ Risk Model
→ Risk Score / Prediction
→ API Response
```

---

## API Endpoints

Health Check

GET /health

Response:
```
{
  "status": "ok"
}
```

---

## Risk Prediction

POST /predict-risk

Request:
```
{
  "features": {
    "open": 65000,
    "high": 65500,
    "low": 64000,
    "close": 64500,
    "volume": 1234.56
  }
}
```
Response:
```
{
  "risk_score": 0.82,
  "label": "High Risk"
}
```

---

## Tech Stack
	•	Python
	•	pandas / numpy
	•	scikit-learn
	•	FastAPI
	•	Pydantic
	•	PyYAML
	•	pytest
	•	matplotlib / seaborn

---

## Notebooks
	•	01_eda.ipynb
→ Exploratory Data Analysis for crypto market data
	•	02_risk_model_experiments.ipynb
→ Risk model experimentation and baseline modeling

---

## Configuration

Configuration is managed via:
```
configs/config.yaml
```
Includes:
	•	data paths
	•	market symbols / intervals
	•	model settings
	•	artifact paths

---

## Installation
```
git clone https://github.com/jjuunnii98/crypto-risk-intelligence-pipeline.git
cd crypto-risk-intelligence-pipeline

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run API
```
uvicorn src.api.main:app --reload
```

Access:
```
http://127.0.0.1:8000/docs
```

---

## Testing
```
pytest
```

---

## Future Improvements
	•	Real-time exchange API integration
	•	Advanced volatility and drawdown risk features
	•	LSTM / XGBoost-based risk modeling
	•	Docker-based deployment
	•	CI/CD integration
	•	Live monitoring dashboard for risk intelligence

---

## Author

Junyeong Song

AI Engineer | Machine Learning Systems | Risk Intelligence | Crypto AI