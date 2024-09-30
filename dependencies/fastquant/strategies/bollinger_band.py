#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Import standard library
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

# Import modules
import backtrader as bt

# Import from package
from fastquant.strategies.base import BaseStrategy

class BBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy
    Simple implementation of backtrader BBands strategy reference: https://community.backtrader.com/topic/122/bband-strategy/2

    Parameters
    ----------
    period : int
        Period used as basis in calculating the moving average and standard deviation
    devfactor : int
        The number of standard deviations from the moving average to derive the upper and lower bands

    TODO: Study this strategy closer based on the above reference. Current implementation is naive.
    """

    params = (
        ("period", 20),  # period for the fast moving average
        ("devfactor", 2.0),
    )

    def __init__(self):
        # Initialize global variables
        super().__init__()
        # Strategy level variables
        self.period = self.params.period
        self.devfactor = self.params.devfactor

        if self.strategy_logging:
            print("===Strategy level arguments===")
            print("period :", self.period)
            print("devfactor :", self.devfactor)
        bbands = bt.ind.BBands(period=self.period, devfactor=self.devfactor)
        self.mid = bbands.mid
        self.top = bbands.top
        self.bot = bbands.bot
        self.DCA = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.dataclose[0] < self.bot:
                if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                        self.sizes.append(self.amount)
                        # self.load_trade_data()
                        self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                        self.calc_averages()
                        self.buy_executed = True
                        self.conditions_checked = True
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.dataclose[0] > self.top:
            if self.buy_executed and self.data.close[0] >= self.take_profit_price:
                average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0

                # Avoid selling at a loss or below the take profit price
                if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                    print(
                        f"| - Avoiding sell at a loss or below take profit. "
                        f"| - Current close price: {self.data.close[0]:.12f}, "
                        f"| - Average entry price: {average_entry_price:.12f}, "
                        f"| - Take profit price: {self.take_profit_price:.12f}"
                    )
                    self.conditions_checked = True
                    return

                if self.params.backtest == False:
                    self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                elif self.params.backtest == True:
                    self.close()

                self.reset_position_state()
                self.buy_executed = False
                self.conditions_checked = True
