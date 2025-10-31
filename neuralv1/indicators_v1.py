
import polars as pl
import numpy as np
import talib
from rich.console import Console

console = Console()

CORE_INDICATORS = ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'CCI', 'Stoch_K', 'Stoch_D',
                   'WillR', 'MOM', 'ROC', 'OBV', 'MFI', 'CMO', 'SAR']

def compute_core_indicators(df: pl.DataFrame) -> pl.DataFrame:

    arrays = {col: df[col].to_numpy().astype(np.float64) for col in ['open', 'high', 'low', 'close', 'volume']}

    rsi = talib.RSI(arrays['close'], timeperiod=14)
    macd, signal, _ = talib.MACD(arrays['close'])
    atr = talib.ATR(arrays['high'], arrays['low'], arrays['close'])
    adx = talib.ADX(arrays['high'], arrays['low'], arrays['close'])
    cci = talib.CCI(arrays['high'], arrays['low'], arrays['close'])
    stoch_k, stoch_d = talib.STOCH(arrays['high'], arrays['low'], arrays['close'])
    willr = talib.WILLR(arrays['high'], arrays['low'], arrays['close'])
    mom = talib.MOM(arrays['close'], timeperiod=10)
    roc = talib.ROC(arrays['close'], timeperiod=10)
    obv = talib.OBV(arrays['close'], arrays['volume'])
    mfi = talib.MFI(arrays['high'], arrays['low'], arrays['close'], arrays['volume'])
    cmo = talib.CMO(arrays['close'])
    sar = talib.SAR(arrays['high'], arrays['low'])

    df_out = df.clone()
    ind_dict = {
        'RSI': rsi, 'MACD': macd, 'MACD_signal': signal, 'ATR': atr, 'ADX': adx,
        'CCI': cci, 'Stoch_K': stoch_k, 'Stoch_D': stoch_d, 'WillR': willr,
        'MOM': mom, 'ROC': roc, 'OBV': obv, 'MFI': mfi, 'CMO': cmo, 'SAR': sar
    }

    for name, arr in ind_dict.items():
        df_out = df_out.with_columns(pl.Series(name, np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0), dtype=pl.Float64))

    console.print(f"✅ Computed {len(CORE_INDICATORS)} core indicators (vectorized)")
    return df_out
