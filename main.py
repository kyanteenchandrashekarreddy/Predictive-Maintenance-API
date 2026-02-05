import joblib
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from data_processor import DataProcessor

# 1. Define Input Schema
class SensorData(BaseModel):
    temperature: Union[float, str]
    vibration: Union[float, str]
    pressure: Union[float, str]

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

# 2. Enable CORS for your index.html dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
processor = None
metrics_data = {}

# 3. Database Initialization
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (timestamp TEXT, temperature TEXT, vibration TEXT, pressure TEXT, prediction TEXT, probability REAL)''')
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    global model, processor, metrics_data
    init_db()
    try:
        if os.path.exists("model.joblib") and os.path.exists("processor.joblib"):
            print("Loading artifacts...")
            model = joblib.load("model.joblib")
            processor = joblib.load("processor.joblib")
            if os.path.exists("metrics.json"):
                with open("metrics.json", "r") as f:
                    metrics_data = json.load(f)
            print("Artifacts loaded successfully.")
        else:
            print("Warning: Model artifacts not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.post("/predict")
def predict(data: SensorData):
    global model, processor
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess and Predict
        processed_data = processor.transform(df)
        prediction = model.predict(processed_data)
        probability = float(model.predict_proba(processed_data)[0][1])
        
        result = "Machine Failure" if prediction[0] == 1 else "Normal"
        
        # Log to Database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   str(data.temperature), str(data.vibration), str(data.pressure), 
                   result, probability))
        conn.commit()
        conn.close()
        
        return {
            "prediction": result,
            "probability_of_failure": probability,
            "input_received": input_dict
        }
    except Exception as e:
        # This catches errors like "ERR" being handled or math errors
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    return JSONResponse(content=metrics_data)

@app.get("/metrics/confusion_matrix")
def get_confusion_matrix_image():
    if os.path.exists("confusion_matrix.png"):
        return FileResponse("confusion_matrix.png", media_type="image/png")
    return JSONResponse(content={"error": "Confusion matrix image not found."}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)