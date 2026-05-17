from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import os

# Ensure the parent directory is in the path to import ml_backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_backend import (
    load_model, prepare_features, predict_price, 
    generate_recommendation, get_available_states, get_state_crops
)

app = FastAPI(
    title="Crop Price Prediction API",
    description="Real-time REST API for predicting agricultural crop prices",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    crop_name: str
    state: str = "Tamil Nadu"
    current_price: float
    month: Optional[int] = None
    year: Optional[int] = None

class PredictionResponse(BaseModel):
    crop_name: str
    state: str
    predicted_price: float
    recommendation: Dict[str, Any]
    metadata: Dict[str, Any]

@app.get("/")
def read_root():
    return {
        "name": "Crop Price Prediction API",
        "status": "active",
        "docs_url": "/docs"
    }

@app.get("/states")
def get_states():
    return {"states": get_available_states()}

@app.get("/crops/{state}")
def get_crops(state: str):
    crops = get_state_crops(state)
    if not crops:
        raise HTTPException(status_code=404, detail="State not found or no crops available")
    return {"state": state, "crops": crops}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Use current date if not provided
        month = request.month if request.month else datetime.now().month
        year = request.year if request.year else datetime.now().year
        
        # Load model (this is cached via lru_cache in ml_backend)
        model = load_model(request.crop_name, request.state)
        
        # Prepare features array
        features = prepare_features(request.current_price, month, year, request.crop_name)
        
        # Make prediction
        predicted_price = predict_price(model, features)
        
        # Generate insight
        recommendation = generate_recommendation(request.current_price, predicted_price)
        
        return {
            "crop_name": request.crop_name,
            "state": request.state,
            "predicted_price": round(predicted_price, 2),
            "recommendation": recommendation,
            "metadata": {
                "target_month": month,
                "target_year": year,
                "base_price": request.current_price
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
