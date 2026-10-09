# Customer Churn Prediction API

A machine learning project that predicts whether a telecommunications customer is likely to leave a service provider. The model is served through a REST API built with FastAPI and packaged with Docker.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI and Uvicorn
- Joblib
- Docker

## Model Performance
Evaluated on a held-out test dataset:
- Accuracy: 76.0%
- ROC-AUC: 0.813
- Recall for churned customers: 64%

Predictions are estimates, not guarantees.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic API information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

Interactive API documentation: http://127.0.0.1:8000/docs

## Run with Docker

Build the image:

`docker build -t customer-churn-api .`

Start the container:

`docker run -d --name churn-api -p 8000:8000 customer-churn-api`

Stop the container:

`docker stop churn-api`

## Project Files
- `main.py` — FastAPI application
- `churn_model.joblib` — trained model pipeline
- `train_model.py` — model training and evaluation
- `clean_data.py` — data cleaning
- `inspect_data.py` — data inspection
- `requirements.txt` — dependencies
- `Dockerfile` — container configuration

## Dataset
Uses the IBM Telco Customer Churn dataset, containing customer demographics, account details, subscribed services, and churn labels.

## Limitations
Model performance may vary on new data. Predictions should be validated before being used for real business decisions.
