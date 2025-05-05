# from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
# from backtrader.strategies.base import CustomSQN, CustomPandasData
import backtrader as bt
import pandas as pd
from pprint import pprint
# from backtrader.stores import binance_store# , bitget_store, mexc_store, pancakeswap_store
import datetime as dt
from backtrader.dontcommit import *


#########

# Quantstats
# import quantstats_lumi as quantstats

import datetime as dt
class PandasData(bt.feeds.PandasData):
    lines = ('open', 'high', 'low', 'close', 'volume')
    params = (('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'))
######### 
