import polars as pl
df = pl.read_parquet('exports/zerolag_sine_momentum_intermediates.parquet')
print(df.head())
print(df.select(['bar', 'HMA_wma_half', 'HMA_wma_full', 'HMA_raw_hma']))
print(df.columns)#

import matplotlib.pyplot as plt
# Plot specific intermediates
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot HMA components
df_pd = df.to_pandas()
axes[0].plot(df_pd['bar'], df_pd['HMA_wma_half'], label='WMA Half')
axes[0].plot(df_pd['bar'], df_pd['HMA_wma_full'], label='WMA Full')
axes[0].plot(df_pd['bar'], df_pd['HMA_raw_hma'], label='Raw HMA')
axes[0].legend()
axes[0].set_title('HMA Components')

# Plot ATR components
axes[1].plot(df_pd['bar'], df_pd['ATR_tr'], label='True Range')
axes[1].plot(df_pd['bar'], df_pd['ATR_atr'], label='ATR')
axes[1].legend()
axes[1].set_title('ATR Components')

# Plot momentum
axes[2].plot(df_pd['bar'], df_pd['Momentum_momentum'])
axes[2].set_title('Momentum')

plt.tight_layout()
plt.savefig('intermediates_analysis.png')
plt.show()