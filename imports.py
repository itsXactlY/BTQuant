from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer, SQN
import backtrader as bt
import pandas as pd
from pprint import pprint
from pairs import *
from live_strategys.live_functions import MyPandasData, running_backtest

#########

# Quantstats
import quantstats
import datetime as dt
class PandasData(bt.feeds.PandasData):
    lines = ('open', 'high', 'low', 'close', 'volume')
    params = (('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'))
######### 
