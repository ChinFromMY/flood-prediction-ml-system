import os
import pickle
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# load model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_prod.pkl")
LOG_PATH = os.path.join(BASE_DIR, "flood.api.log")

# logging
logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)

# load ML model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    logging.critical(f"Model file not found at {MODEL_PATH}")
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# FastAPI 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# endpoint for CI
@app.get("/health")
def health():
    return{"status": "ok"}


# input
class FloodData(BaseModel):
    Max_Temp: float
    Min_Temp: float
    Rainfall: float
    Relative_Humidity: float
    Wind_Speed: float
    Cloud_Coverage: float


# input validation
    @field_validator("Max_Temp")
    @classmethod
    def max_temp_range(cls, v):
        if not -50 <= v <= 50:
            raise ValueError("Max_Temp must be between -50 and 50 °C")
        return v

    @field_validator("Min_Temp")
    @classmethod
    def min_temp_range(cls, v):
        if not -50 <= v <= 50:
            raise ValueError("Min_Temp must be between -50 and 50 °C")
        return v

    @field_validator("Rainfall")
    @classmethod
    def rainfall_range(cls, v):
        if v < 0:
            raise ValueError("Rainfall cannot be negative")
        return v

    @field_validator("Relative_Humidity")
    @classmethod
    def humidity_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Relative_Humidity must be 0-100%")
        return v

    @field_validator("Wind_Speed")
    @classmethod
    def wind_speed_range(cls, v):
        if v < 0:
            raise ValueError("Wind_Speed cannot be negative")
        return v

    @field_validator("Cloud_Coverage")
    @classmethod
    def cloud_coverage_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Cloud_Coverage must be 0-100%")
        return v


# prediction endpoint
@app.post("/predict-flood/")
def predict_flood(data: FloodData, request: Request):
    input_data = [[
        data.Max_Temp,
        data.Min_Temp,
        data.Rainfall,
        data.Relative_Humidity,
        data.Wind_Speed,
        data.Cloud_Coverage,
    ]]

    try:
        # Predict probability if the model supports it
        if hasattr(model, "predict_proba"):
            flood_probability = model.predict_proba(input_data)[0, 1]
        else:
            flood_probability = float(model.predict(input_data)[0])

        # Threshold for binary prediction
        PREDICTION_THRESHOLD = 0.35
        prediction = int(flood_probability >= PREDICTION_THRESHOLD)

        logging.info(
            f"Request from{request.client.host} | Input: {input_data[0]} |"
            f"Prediction: {prediction} | Probability: {flood_probability:.4f}"
        )

        return {
            "flood_prediction": prediction,
            "probability": round(flood_probability, 4)
        }

    except Exception as e:

        logging.error(
            f"Prediction failed for input {input_data[0]} | Error: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


    
