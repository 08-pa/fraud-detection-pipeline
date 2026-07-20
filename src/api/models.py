from pydantic import BaseModel
from typing import Optional

class Transaction(BaseModel):
    """Transaction data for prediction"""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    # Optional engineered features (will be computed if not provided)
    hour_of_day: Optional[float] = None
    is_night: Optional[float] = None
    is_weekend: Optional[float] = None  # NEW!
    log_amount: Optional[float] = None
    amount_zscore: Optional[float] = None
    V14_V17: Optional[float] = None
    V3_V10: Optional[float] = None

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int
    fraud_probability: float
    threshold_used: float
    is_fraud: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str = "1.0.0"