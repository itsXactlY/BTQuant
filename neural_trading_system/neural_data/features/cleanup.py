import pickle
import numpy as np
import glob
from pathlib import Path

# Find feature cache
cache_files = list(Path('.').rglob('features_faee3f17ea44d4f8.pkl'))
if not cache_files:
    print("❌ No feature cache found!")
    print("Run from BTQuant root directory")
    exit(1)

cache_path = str(cache_files[0])
print(f"\n📁 Found cache: {cache_path}")

# Load
print("Loading cache...")
with open(cache_path, 'rb') as f:
    data = pickle.load(f)

features = data['features']
returns = data['returns']

print(f"\n📊 Original Data:")
print(f"   Features shape: {features.shape}")
print(f"   Features NaN: {np.isnan(features).sum():,}")
print(f"   Features Inf: {np.isinf(features).sum():,}")
print(f"   Returns NaN: {np.isnan(returns).sum():,}")
print(f"   Returns Inf: {np.isinf(returns).sum():,}")

# CLEAN IT
print("\n🧹 Cleaning data...")
features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
returns_clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\n✅ Cleaned Data:")
print(f"   Features NaN: {np.isnan(features_clean).sum():,}")
print(f"   Features Inf: {np.isinf(features_clean).sum():,}")
print(f"   Returns NaN: {np.isnan(returns_clean).sum():,}")
print(f"   Returns Inf: {np.isinf(returns_clean).sum():,}")

# Save cleaned version
data['features'] = features_clean
data['returns'] = returns_clean

cleaned_path = cache_path.replace('.pkl', '_CLEANED.pkl')
with open(cleaned_path, 'wb') as f:
    pickle.dump(data, f)

print(f"\n💾 Saved: {cleaned_path}")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print(f"\n1. Edit neural_pipeline.py")
print(f"2. Change cache path to: {cleaned_path}")
print(f"3. Delete old checkpoints: rm neural_trading_system/checkpoint*.pt")
print(f"4. Restart training")
print("\n" + "="*80)