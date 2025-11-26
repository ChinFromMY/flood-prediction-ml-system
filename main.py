import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

with open('./random_forest_final_model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FloodData(BaseModel):
    Max_Temp: float
    Min_Temp: float
    Rainfall: float
    Relative_Humidity: float
    Wind_Speed: float
    Cloud_Coverage: float

@app.post("/predict-flood/")
def predict_flood(data: FloodData):

    input_data = [[
        data.Max_Temp,
        data.Min_Temp,
        data.Rainfall,
        data.Relative_Humidity,
        data.Wind_Speed,
        data.Cloud_Coverage,
    ]]

    flood_probability = model.predict_proba(input_data)[0, 1]

    PREDICTION_THRESHOLD = 0.35

    if flood_probability >= PREDICTION_THRESHOLD:
        prediction = 1 #flood
    else:
        prediction = 0 #no flood

    return {"flood_prediction": int(prediction),
            "probability": float(flood_probability)}

    
