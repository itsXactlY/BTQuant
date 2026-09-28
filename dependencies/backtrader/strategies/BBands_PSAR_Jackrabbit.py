import sys
sys.path.append('/home/JackrabbitRelay/Base/Library')
from JackrabbitRelay import TechnicalAnalysis as JRRta

def Strategy(ta, exchangeName, account, asset, direction):
    # === Persistent Range Tracking ===
    if not hasattr(Strategy, 'HighestHigh') or not hasattr(Strategy, 'LowestLow'):
        Strategy.HighestHigh, Strategy.LowestLow = float('-inf'), float('inf')
        for bar in JRRta(exchangeName, account, asset, 'MAX', 5000).GetOHLCV():
            if bar[2] > Strategy.HighestHigh: Strategy.HighestHigh = bar[2]
            if bar[3] < Strategy.LowestLow: Strategy.LowestLow = bar[3]

    # Update with current bar
    row = ta.LastRow()
    Strategy.HighestHigh = max(Strategy.HighestHigh, row[2])
    Strategy.LowestLow = min(Strategy.LowestLow, row[3])

    # === Column Indices ===
    O, H, L, C, V = 1, 2, 3, 4, 5  # OHLCV
    MA, BB_U, BB_L, PSAR = 8, 12, 13, 14  # SMA, Bollinger Bands, PSAR
    BUY_SIG, SELL_SIG = 18, 19

    # === Calculate Indicators ===
    ta.AddColumn(Strategy.HighestHigh)
    ta.AddColumn(Strategy.LowestLow)
    ta.SMA(C, 20)
    ta.BollingerBands(MA, 20, 2)
    ta.PSAR()
    row = ta.LastRow()

    # === Signal Generation ===
    if any(row[i] is None for i in [BB_U, BB_L, PSAR]):
        ta.AddColumn(0)
        ta.AddColumn(0)
        return ta, BUY_SIG, SELL_SIG, -1

    if direction == 1:  # LONG
        buy = 1 if row[C] <= row[BB_L] and row[C] > row[PSAR] else 0
        sell = 1 if (row[C] >= row[BB_U] or row[C] <= row[PSAR]) else 0
    elif direction == -1:  # SHORT
        buy = -1 if row[C] >= row[BB_U] and row[C] < row[PSAR] else 0
        sell = -1 if (row[C] <= row[BB_L] or row[C] >= row[PSAR]) else 0
    else:
        buy, sell = 0, 0

    ta.AddColumn(buy)
    ta.AddColumn(sell)
    return ta, BUY_SIG, SELL_SIG, -1