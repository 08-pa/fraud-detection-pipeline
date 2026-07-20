import requests
import json
import pandas as pd
import numpy as np

print("="*60)
print("🧪 TESTING FRAUD DETECTION API WITH REAL DATA")
print("="*60)

# API URL
url = "http://localhost:8000/predict"

# Load the actual dataset to get real transactions
print("\n📂 Loading real transactions from dataset...")
df = pd.read_csv('data/creditcard_engineered.csv')

# Get one real fraud transaction
fraud_df = df[df['Class'] == 1]
real_fraud = fraud_df.iloc[0].to_dict()

# Get one real normal transaction
normal_df = df[df['Class'] == 0]
real_normal = normal_df.iloc[0].to_dict()

print(f"   ✅ Found {len(fraud_df)} fraud transactions")
print(f"   ✅ Found {len(normal_df)} normal transactions")

# Function to convert to API format
def convert_to_api_format(row_dict):
    """Convert DataFrame row to API format"""
    return {
        "Time": float(row_dict['Time']),
        "V1": float(row_dict['V1']),
        "V2": float(row_dict['V2']),
        "V3": float(row_dict['V3']),
        "V4": float(row_dict['V4']),
        "V5": float(row_dict['V5']),
        "V6": float(row_dict['V6']),
        "V7": float(row_dict['V7']),
        "V8": float(row_dict['V8']),
        "V9": float(row_dict['V9']),
        "V10": float(row_dict['V10']),
        "V11": float(row_dict['V11']),
        "V12": float(row_dict['V12']),
        "V13": float(row_dict['V13']),
        "V14": float(row_dict['V14']),
        "V15": float(row_dict['V15']),
        "V16": float(row_dict['V16']),
        "V17": float(row_dict['V17']),
        "V18": float(row_dict['V18']),
        "V19": float(row_dict['V19']),
        "V20": float(row_dict['V20']),
        "V21": float(row_dict['V21']),
        "V22": float(row_dict['V22']),
        "V23": float(row_dict['V23']),
        "V24": float(row_dict['V24']),
        "V25": float(row_dict['V25']),
        "V26": float(row_dict['V26']),
        "V27": float(row_dict['V27']),
        "V28": float(row_dict['V28']),
        "Amount": float(row_dict['Amount'])
    }

# Test transactions
test_cases = [
    ("🚨 REAL FRAUD (from dataset)", convert_to_api_format(real_fraud)),
    ("✅ REAL NORMAL (from dataset)", convert_to_api_format(real_normal)),
    ("🔴 HIGH RISK (simulated)", {
        "Time": 1000.0,
        "V1": -1.23,
        "V2": 0.45,
        "V3": -0.67,
        "V4": 0.89,
        "V5": -0.12,
        "V6": 0.34,
        "V7": -0.56,
        "V8": 0.78,
        "V9": -0.90,
        "V10": 0.12,
        "V11": -0.34,
        "V12": 0.56,
        "V13": -0.78,
        "V14": 1.20,
        "V15": -0.45,
        "V16": 0.67,
        "V17": 1.10,
        "V18": -0.89,
        "V19": 0.23,
        "V20": -0.45,
        "V21": 0.67,
        "V22": -0.78,
        "V23": 0.89,
        "V24": -0.12,
        "V25": 0.34,
        "V26": -0.56,
        "V27": 0.78,
        "V28": -0.90,
        "Amount": 1500.0
    })
]

print("\n" + "="*60)

for name, transaction in test_cases:
    print(f"\n📊 Testing: {name}")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{url}?threshold=0.5",
            json=transaction,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            fraud_prob = result['fraud_probability']
            is_fraud = result['is_fraud']
            
            print(f"   Prediction: {'🚨 FRAUD' if is_fraud else '✅ NORMAL'}")
            print(f"   Fraud Probability: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
            print(f"   Threshold Used: {result['threshold_used']}")
            
            # Show key features for context
            if 'V14' in transaction and 'V17' in transaction:
                print(f"   V14: {transaction['V14']:.2f}, V17: {transaction['V17']:.2f}")
            print(f"   Amount: ${transaction['Amount']:.2f}")
            
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection Error: API is not running!")
        print("   Please start the API first: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ API Test Complete!")
print("="*60)