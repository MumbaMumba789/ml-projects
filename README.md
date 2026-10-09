# ML Projects

Machine learning projects and deployment experiments.

## Project 1: Iris ML API

A machine learning API built with Python, scikit-learn, FastAPI, and Docker.

### Run locally

Install dependencies:

`ash
pip install -r requirements.txt
`

Start the API:

`ash
uvicorn main:app --reload
`

Open http://127.0.0.1:8000/docs to test predictions.

### Run with Docker

Build the image:

`ash
docker build -t iris-ml-api .
`

Run the container:

`ash
docker run --name iris-api -p 8000:8000 iris-ml-api
`

The /predict endpoint accepts four Iris measurements and returns a predicted flower species.

## Security

Environment variables and local secrets are excluded from version control.
