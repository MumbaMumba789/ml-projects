# ML Projects

A collection of machine-learning projects built with Python, scikit-learn, FastAPI, and Docker.

## Projects

### 1. Iris ML API

A machine-learning API that predicts an Iris flower species from four measurements.

**Folder:** [`iris-ml-api/`](./iris-ml-api)

Run locally from the project folder:

```powershell
cd iris-ml-api
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test predictions.

Run with Docker from `iris-ml-api/`:

```powershell
docker build -t iris-ml-api .
docker run --name iris-api -p 8000:8000 iris-ml-api
```

### 2. Customer Churn Prediction API

A Dockerized FastAPI service that predicts customer churn.

**Folder:** [`customer-churn-api/`](./customer-churn-api)

See the [Customer Churn API README](./customer-churn-api/README.md) for setup, endpoints, and evaluation details.

## Repository structure

- `iris-ml-api/` — Iris classification API, model, dependencies, and Dockerfile.
- `customer-churn-api/` — customer churn prediction API and trained model.

