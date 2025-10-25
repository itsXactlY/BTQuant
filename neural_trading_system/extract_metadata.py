import pandas as pd

# Load your training data
df = pd.read_parquet('neural_data/BTC_1h_2017-01-01_2024-12-31_neural_data.parquet')
df = df[(df['datetime'] >= '2017-01-01') & (df['datetime'] < '2024-01-01')]

# Calculate returns
returns = df['close'].pct_change().dropna()
return_scale = returns.std()

print(f"Estimated return_scale: {return_scale:.6f}")