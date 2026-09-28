import sys
sys.path.append('/home/JackrabbitRelay2/Base/Library')
from JackrabbitRelay import TechnicalAnalysis as JRRta

def Strategy(ta, exchangeName, account, asset, direction):
    """PotatoHarvester Pro: EMA center + ATR volatility breakout channels.
    Long: close crosses above bottom band. Short: close crosses below top band.
    Signal: signed level (+N for long band N, -N for short band N, 0 = no signal).
    """
    O, H, L, C, V = 1, 2, 3, 4, 5
    centre_period, atr_period, levels = 21, 21, 10
    atr_mult = [3.0 + 2.0 * i for i in range(levels)]  # [3,5,7,...21]

    ta.EMA(C, centre_period)                    # col 6: centre EMA
    ta.ATR(H, L, C, atr_period, smooth_func=ta.EMA)  # col 7:TR, 8:ATR

    row = ta.LastRow()
    centre, atr = (row[6] if len(row) > 6 else None), (row[8] if len(row) > 8 else None)

    if centre is None or atr is None:
        for _ in range(4 * levels + 2): ta.AddColumn(None)
        return ta, -1, -1, -1

    band_top, band_bot = [], []
    for m in atr_mult:
        ta.AddColumn(centre + atr * m)  # top band
        ta.AddColumn(centre - atr * m)  # bottom band
        band_top.append(len(ta.LastRow()) - 2)
        band_bot.append(len(ta.LastRow()) - 1)

    buy_cross, sell_cross = [], []
    for b in band_bot: ta.Cross(C, b); buy_cross.append(len(ta.LastRow()) - 1)
    for t in band_top: ta.Cross(t, C); sell_cross.append(len(ta.LastRow()) - 1)

    row = ta.LastRow()
    buy_level = next((i + 1 for i, c in enumerate(buy_cross) if len(row) > c and row[c] != 0), 0)
    sell_level = next((-(i + 1) for i, c in enumerate(sell_cross) if len(row) > c and row[c] != 0), 0)

    if direction == 1:
        ta.AddColumn(buy_level), ta.AddColumn(sell_level)
    elif direction == -1:
        ta.AddColumn(sell_level), ta.AddColumn(buy_level)
    else:
        ta.AddColumn(0), ta.AddColumn(0)

    return ta, len(ta.LastRow()) - 2, len(ta.LastRow()) - 1, -1