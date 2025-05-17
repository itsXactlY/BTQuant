from backtrader import backtest
from backtrader.strategies.QQE_Hullband_VolumeOsc import QQE_Example as _strategy
from backtrader.strategies.base import CustomPandasData

import pandas as pd
data = '../candles/BTCUSDT_1min.csv' # Example CSV file path

my_data_frame = pd.read_csv(data, index_col=0, parse_dates=True)
my_data_frame = my_data_frame.sort_index()

start = "2024-06-01"
end = "2025-01-08"
df = my_data_frame.loc[start:end]
data = CustomPandasData(dataname=df)

if __name__ == '__main__':
    try:
        backtest(_strategy, 
                data,
                init_cash=1000, 
                backtest=True, 
                plot=True)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
