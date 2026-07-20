import requests
import pandas as pd

print("="*60)
print("🎯 TESTING DIFFERENT THRESHOLDS")
print("="*60)

# Load a real fraud transaction
df = pd.read_csv('data/creditcard_engineered.csv')
fraud_df = df[df['Class'] == 1]
real_fraud = fraud_df.iloc[0].to_dict()

# Convert to API format
transaction = {
    "Time": float(real_fraud['Time']),
    "V1": float(real_fraud['V1']),
    "V2": float(real_fraud['V2']),
    "V3": float(real_fraud['V3']),
    "V4": float(real_fraud['V4']),
    "V5": float(real_fraud['V5']),
    "V6": float(real_fraud['V6']),
    "V7": float(real_fraud['V7']),
    "V8": float(real_fraud['V8']),
    "V9": float(real_fraud['V9']),
    "V10": float(real_fraud['V10']),
    "V11": float(real_fraud['V11']),
    "V12": float(real_fraud['V12']),
    "V13": float(real_fraud['V13']),
    "V14": float(real_fraud['V14']),
    "V15": float(real_fraud['V15']),
    "V16": float(real_fraud['V16']),
    "V17": float(real_fraud['V17']),
    "V18": float(real_fraud['V18']),
    "V19": float(real_fraud['V19']),
    "V20": float(real_fraud['V20']),
    "V21": float(real_fraud['V21']),
    "V22": float(real_fraud['V22']),
    "V23": float(real_fraud['V23']),
    "V24": float(real_fraud['V24']),
    "V25": float(real_fraud['V25']),
    "V26": float(real_fraud['V26']),
    "V27": float(real_fraud['V27']),
    "V28": float(real_fraud['V28']),
    "Amount": float(real_fraud['Amount'])
}

# Test different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]

print("\n📊 Real Fraud Transaction Analysis")
print("-" * 40)
print(f"   Amount: ${transaction['Amount']:.2f}")
print(f"   V14: {transaction['V14']:.2f}, V17: {transaction['V17']:.2f}")

print("\n📊 Testing Different Thresholds:")
print("-" * 40)

for threshold in thresholds:
    try:
        response = requests.post(
            f"http://localhost:8000/predict?threshold={threshold}",
            json=transaction,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            proba = result['fraud_probability']
            is_fraud = result['is_fraud']
            
            status = "🚨 FRAUD" if is_fraud else "✅ NORMAL"
            print(f"   Threshold {threshold}: {status} (Probability: {proba:.3f})")
        else:
            print(f"   ❌ Error at threshold {threshold}: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error at threshold {threshold}: {e}")

print("\n" + "="*60)
print("✅ Threshold Test Complete!")
print("="*60)