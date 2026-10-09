from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from sklearn.datasets import load_iris 
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the trained model
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
model = joblib.load(MODEL_PATH)


# Input data for the model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: IrisInput):
    logger.info("Received prediction request")
    
    prediction = model.predict([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    iris = load_iris()
    class_names = iris.target_names

    return {
        "prediction": class_names[int(prediction[0])]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}