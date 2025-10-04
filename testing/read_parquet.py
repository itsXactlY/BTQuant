import polars as pl
df = pl.read_parquet('exports/zerolag_sine_momentum_intermediates.parquet')
print(df.head())
print(df.select(['bar', 'HMA_wma_half', 'HMA_wma_full', 'HMA_raw_hma']))
print(df.columns)