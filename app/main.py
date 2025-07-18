import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware


model_path = os.path.join(os.path.dirname(__file__), "model", "titanic_pipeline.pkl")
model = joblib.load(model_path)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ou ["*"] si tu veux autoriser toutes origines (moins sécurisé)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int


@app.post("/Passenger")

async def predict_surval(item: Passenger):
    df = pd.DataFrame([item.dict()])
    prediction = model.predict(df)
    survived =bool(prediction[0])
    return {"prediction": int(prediction[0]),
             "message": "Survived" if survived else "Did not survive"}