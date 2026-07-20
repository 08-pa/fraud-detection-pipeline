from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from typing import Optional

from .models import Transaction, PredictionResponse, HealthResponse
from .utils import load_model, prepare_features, get_prediction

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection using XGBoost",
    version="1.0.0"
)

# Load model and scaler
try:
    model, scaler = load_model()
    model_loaded = True
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    model, scaler = None, None
    model_loaded = False
    print(f"❌ Error loading model: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: Transaction,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold")
):
    """
    Predict if a transaction is fraud
    
    - **threshold**: Adjust sensitivity (0.3 = catch more fraud, 0.7 = fewer false alarms)
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict
        data = transaction.dict()
        
        # Prepare features
        features = prepare_features(data)
        
        # Get prediction
        pred, proba = get_prediction(model, scaler, features)
        
        # Apply threshold
        is_fraud = proba >= threshold
        
        return PredictionResponse(
            prediction=pred,
            fraud_probability=float(proba),
            threshold_used=threshold,
            is_fraud=is_fraud
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(
    transactions: list[Transaction],
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Predict multiple transactions at once
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for transaction in transactions:
            data = transaction.dict()
            features = prepare_features(data)
            pred, proba = get_prediction(model, scaler, features)
            
            results.append({
                "prediction": pred,
                "fraud_probability": float(proba),
                "threshold_used": threshold,
                "is_fraud": proba >= threshold
            })
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features():
    """Get the list of expected features"""
    return {
        "features": [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount'
        ],
        "engineered_features": [
            'hour_of_day', 'is_night', 'log_amount', 'amount_zscore',
            'V14_V17', 'V3_V10'
        ],
        "total_features": 36
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)