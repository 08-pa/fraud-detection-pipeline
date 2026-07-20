import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

def load_model():
    """Load the best model and scaler"""
    model_path = ROOT_DIR / 'models' / 'best_model.pkl'
    scaler_path = ROOT_DIR / 'models' / 'scaler.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✅ Model loaded! Expects {scaler.n_features_in_} features")
    return model, scaler

def prepare_features(data: dict):
    """Convert input dict to feature vector with engineered features"""
    
    # Get all values with defaults (convert None to 0)
    Time = float(data.get('Time') or 0)
    Amount = float(data.get('Amount') or 0)
    
    # V1 to V28
    V1 = float(data.get('V1') or 0)
    V2 = float(data.get('V2') or 0)
    V3 = float(data.get('V3') or 0)
    V4 = float(data.get('V4') or 0)
    V5 = float(data.get('V5') or 0)
    V6 = float(data.get('V6') or 0)
    V7 = float(data.get('V7') or 0)
    V8 = float(data.get('V8') or 0)
    V9 = float(data.get('V9') or 0)
    V10 = float(data.get('V10') or 0)
    V11 = float(data.get('V11') or 0)
    V12 = float(data.get('V12') or 0)
    V13 = float(data.get('V13') or 0)
    V14 = float(data.get('V14') or 0)
    V15 = float(data.get('V15') or 0)
    V16 = float(data.get('V16') or 0)
    V17 = float(data.get('V17') or 0)
    V18 = float(data.get('V18') or 0)
    V19 = float(data.get('V19') or 0)
    V20 = float(data.get('V20') or 0)
    V21 = float(data.get('V21') or 0)
    V22 = float(data.get('V22') or 0)
    V23 = float(data.get('V23') or 0)
    V24 = float(data.get('V24') or 0)
    V25 = float(data.get('V25') or 0)
    V26 = float(data.get('V26') or 0)
    V27 = float(data.get('V27') or 0)
    V28 = float(data.get('V28') or 0)
    
    # ============================================
    # ENGINEERED FEATURES (7 of them)
    # ============================================
    
    # 1. hour_of_day
    hour_of_day = data.get('hour_of_day')
    if hour_of_day is None:
        hour_of_day = float((Time / 3600) % 24)
    else:
        hour_of_day = float(hour_of_day)
    
    # 2. is_night
    is_night = data.get('is_night')
    if is_night is None:
        is_night = 1.0 if 0 <= hour_of_day < 6 else 0.0
    else:
        is_night = float(is_night)
    
    # 3. is_weekend (NEW - THIS WAS MISSING!)
    is_weekend = data.get('is_weekend')
    if is_weekend is None:
        # Calculate day of week from Time
        # Time is in seconds, so convert to days and get day of week
        days = Time / (3600 * 24)
        day_of_week = days % 7
        # Weekend = Saturday (5) or Sunday (6)
        is_weekend = 1.0 if day_of_week in [5, 6] else 0.0
    else:
        is_weekend = float(is_weekend)
    
    # 4. log_amount
    log_amount = data.get('log_amount')
    if log_amount is None:
        log_amount = float(np.log1p(Amount))
    else:
        log_amount = float(log_amount)
    
    # 5. amount_zscore
    amount_zscore = data.get('amount_zscore')
    if amount_zscore is None:
        # Using approximate dataset stats
        amount_zscore = float(abs((Amount - 88.35) / 250.12))
    else:
        amount_zscore = float(amount_zscore)
    
    # 6. V14_V17 (interaction)
    V14_V17 = data.get('V14_V17')
    if V14_V17 is None:
        V14_V17 = float(V14 * V17)
    else:
        V14_V17 = float(V14_V17)
    
    # 7. V3_V10 (interaction)
    V3_V10 = data.get('V3_V10')
    if V3_V10 is None:
        V3_V10 = float(V3 * V10)
    else:
        V3_V10 = float(V3_V10)
    
    # ============================================
    # CREATE FEATURE VECTOR - ALL 37 FEATURES
    # ============================================
    # Order must match exactly what scaler expects!
    # ['Time', 'V1', 'V2', ..., 'V28', 'Amount', 
    #  'hour_of_day', 'is_night', 'is_weekend', 
    #  'log_amount', 'amount_zscore', 'V14_V17', 'V3_V10']
    
    features = np.array([
        # Original features (30)
        Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
        V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
        V21, V22, V23, V24, V25, V26, V27, V28, Amount,
        # Engineered features (7) - ORDER MATTERS!
        hour_of_day, is_night, is_weekend, log_amount, 
        amount_zscore, V14_V17, V3_V10
    ]).reshape(1, -1)
    
    print(f"✅ Created features shape: {features.shape}")
    return features

def get_prediction(model, scaler, features):
    """Get prediction and probability"""
    try:
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get probability
        proba = model.predict_proba(features_scaled)[0][1]
        
        # Get prediction (0 or 1)
        pred = 1 if proba >= 0.5 else 0
        
        return pred, proba
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        print(f"   Features shape: {features.shape}")
        raise