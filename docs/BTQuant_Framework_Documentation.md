# BTQuant Framework Documentation

This document provides a comprehensive overview of the BTQuant framework, including its strategies, indicators, and core components. The framework is designed to provide transparency, flexibility, and robustness in trading operations.

## Table of Contents
- [Introduction](#introduction)
- [Core Components](#core-components)
- [Strategies](#strategies)
  - [Aligator SuperTrend](#aligator-supertrend)
  - [MACD ADX](#macd-adx)
  - [Nearest Neighbors Rational Quadratic Kernel](#nearest-neighbors-rational-quadratic-kernel)
  - [Order Chain Kioseff Trading](#order-chain-kioseff-trading)
  - [QQE Hullband Volume Oscillator](#qqe-hullband-volume-oscillator)
  - [SMA Cross MESAdaptive Prime](#sma-cross-mesadaptive-prime)
  - [ST RSX ASI](#st-rsx-asi)
  - [Staged Convergence Strategy](#staged-convergence-strategy)
  - [SuperTrend Scalp](#supertrend-scalp)
  - [Vumanchu A](#vumanchu-a)
  - [Vumanchu B](#vumanchu-b)
  - [Pancakeswap DCA Marketmaker](#pancakeswap-dca-marketmaker)
  - [Pancakeswap Orders](#pancakeswap-orders)
- [Indicators](#indicators)
  - [Accumulative Swing Index (ASI)](#accumulative-swing-index-asi)
  - [Adaptive Cyber Cycle](#adaptive-cyber-cycle)
  - [Chaikin Money Flow](#chaikin-money-flow)
  - [SuperTrend](#supertrend)
  - [Vumanchu Market Cipher](#vumanchu-market-cipher)
  - [Order Chain Indicator](#order-chain-indicator)
  - [Volume Oscillator](#volume-oscillator)
  - [QQE Indicator](#qqe-indicator)
  - [RSX](#rsx)
  - [Mesa Adaptive Moving Average (MAMA)](#mesa-adaptive-moving-average-mama)
  - [Hull Moving Average (HMA)](#hull-moving-average-hma)
  - [Zero Lag](#zero-lag)
  - [Sine Weighted Moving Average](#sine-weighted-moving-average)
  - [Fast QQE](#fast-qqe)
  - [Fast Volume Oscillator](#fast-volume-oscillator)
- [Conclusion](#conclusion)

## Introduction

BTQuant is a comprehensive trading framework built on top of the Backtrader library. It provides a wide range of strategies and indicators designed to enhance trading operations with transparency, flexibility, and robustness. The framework includes tools for backtesting, live trading, and performance analysis.

## Core Components

### BaseStrategy

The `BaseStrategy` class is the foundation of all strategies in BTQuant. It provides core functionality for order management, position tracking, and performance metrics. Key features include:

- **Order Management**: Unified order creation and tracking through the `OrderTracker` class.
- **Position Tracking**: Detailed tracking of entry prices, sizes, and take-profit levels.
- **Performance Metrics**: Calculation of metrics such as total P&L, win rate, and final value.
- **Debugging and Logging**: Colored console output for different types of messages.
- **Data Capture**: Optional data capture for transparency and debugging purposes.

### OrderTracker

The `OrderTracker` class is responsible for tracking individual orders and positions. It provides detailed information about each order, including entry prices, sizes, and timestamps. This information is persisted in CSV files for transparency and auditability.

### Transparency Patch

The `TransparencyPatch` class ensures transparency in indicator calculations by logging or exposing internal calculations of indicators. This allows users to see exactly how indicators are calculated and what values they produce.

## Strategies

### Aligator SuperTrend

**File**: `Aligator_supertrend.py`

The Aligator SuperTrend strategy combines the Williams Alligator indicator with the SuperTrend indicator to identify trends and generate buy/sell signals. Key features include:

- **Indicators**: Uses Williams Alligator and SuperTrend indicators.
- **Entry Conditions**: Buys when both indicators are bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold.
- **Exit Conditions**: Sells when the take-profit price is reached.

### MACD ADX

**File**: `MACD_ADX.py`

The MACD ADX strategy uses the MACD and ADX indicators to identify trends and momentum. Key features include:

- **Indicators**: Uses MACD, ADX, and various EMAs.
- **Entry Conditions**: Buys when the MACD and ADX indicators are bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold.
- **Exit Conditions**: Sells when the take-profit price is reached or when the trailing stop is hit.

### Nearest Neighbors Rational Quadratic Kernel

**File**: `NearestNeighbors_RationalQuadraticKernel.py`

The Nearest Neighbors Rational Quadratic Kernel strategy uses machine learning to identify trading signals. Key features include:

- **Indicators**: Uses RSI, Williams %R, CCI, ADX, and a custom Rational Quadratic Kernel indicator.
- **Entry Conditions**: Buys when the ML signal is positive.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the ML signal is positive.
- **Exit Conditions**: Sells when the take-profit price is reached.

### Order Chain Kioseff Trading

**File**: `Order_Chain_Kioseff_Trading.py`

The Order Chain Kioseff Trading strategy uses the Order Chain indicator to identify trading signals. Key features include:

- **Indicators**: Uses the Order Chain indicator.
- **Entry Conditions**: Buys when the Order Chain indicator exceeds a specified threshold.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the Order Chain indicator exceeds the threshold.
- **Exit Conditions**: Sells when the take-profit price is reached.

### QQE Hullband Volume Oscillator

**File**: `QQE_Hullband_VolumeOsc.py`

The QQE Hullband Volume Oscillator strategy combines the QQE indicator, Hull Moving Average, and Volume Oscillator to identify trading signals. Key features include:

- **Indicators**: Uses QQE, Hull Moving Average, EMA, and Volume Oscillator.
- **Entry Conditions**: Buys when the QQE indicator is positive, the price is below the Hull Moving Average, and the Volume Oscillator is below the short EMA.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### SMA Cross MESAdaptive Prime

**File**: `SMA_Cross_MESAdaptive_Prime.py`

The SMA Cross MESAdaptive Prime strategy uses Simple Moving Averages and the MESAdaptive Moving Average to identify trends. Key features include:

- **Indicators**: Uses SMA and MAMA indicators.
- **Entry Conditions**: Buys when the SMA crossover is positive and the MAMA is bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### ST RSX ASI

**File**: `ST_RSX_ASI.py`

The ST RSX ASI strategy combines the SuperTrend, RSX, and Accumulative Swing Index indicators to identify trading signals. Key features include:

- **Indicators**: Uses SuperTrend, RSX, and ASI indicators.
- **Entry Conditions**: Buys when the SuperTrend is bullish, the ASI is increasing, and the RSX is oversold.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached or when the trailing stop is hit.

### Staged Convergence Strategy

**File**: `StagedConvergenceStrategy.py`

The Staged Convergence Strategy uses multiple SMAs to identify trends and generate buy/sell signals. Key features include:

- **Indicators**: Uses multiple SMAs.
- **Entry Conditions**: Buys when all SMAs are bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### SuperTrend Scalp

**File**: `SuperTrend_Scalp.py`

The SuperTrend Scalp strategy uses the SuperTrend indicator and ADX to identify trends and generate buy/sell signals. Key features include:

- **Indicators**: Uses SuperTrend, ADX, and DI indicators.
- **Entry Conditions**: Buys when the ADX is strong, the DI is bearish, and the SuperTrend is bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### Vumanchu A

**File**: `Vumanchu_A.py`

The Vumanchu A strategy uses the Vumanchu Market Cipher A indicator to identify trading signals. Key features include:

- **Indicators**: Uses the Vumanchu Market Cipher A indicator.
- **Entry Conditions**: Buys when the Vumanchu Market Cipher A indicator is bullish.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### Vumanchu B

**File**: `Vumanchu_B.py`

The Vumanchu B strategy uses the Vumanchu Market Cipher B indicator to identify trading signals. Key features include:

- **Indicators**: Uses the Vumanchu Market Cipher B indicator.
- **Entry Conditions**: Buys when the Vumanchu Market Cipher B indicator is bullish and oversold.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold and the entry conditions are met.
- **Exit Conditions**: Sells when the take-profit price is reached.

### Pancakeswap DCA Marketmaker

**File**: `pancakeswap_dca_marketmaker.py`

The Pancakeswap DCA Marketmaker strategy is designed for trading on PancakeSwap. Key features include:

- **Indicators**: Uses basic price tracking.
- **Entry Conditions**: Buys at the current price.
- **DCA Conditions**: Adds to positions when the price drops below a specified threshold.
- **Exit Conditions**: Sells when the take-profit price is reached.

### Pancakeswap Orders

**File**: `pancakeswap_orders.py`

The Pancakeswap Orders strategy provides functionality for executing orders on PancakeSwap. Key features include:

- **Order Execution**: Executes buy and sell orders on PancakeSwap.
- **Token Management**: Manages token balances and allowances.
- **Gas Management**: Handles gas prices and transaction fees.

## Indicators

### Accumulative Swing Index (ASI)

**File**: `AccumulativeSwingIndex.py`

The Accumulative Swing Index (ASI) measures the cumulative swing of market prices to identify trends. It is used in various strategies to confirm trends and generate signals.

### Adaptive Cyber Cycle

**File**: `AdaptiveCyberCycle.py`

The Adaptive Cyber Cycle indicator adapts to market conditions to provide accurate cycle analysis. It is used to identify trends and generate trading signals.

### Chaikin Money Flow

**File**: `ChaikinMoneyFlow.py`

The Chaikin Money Flow indicator measures the flow of money into or out of a security. It is used to identify trends and confirm trading signals.

### SuperTrend

**File**: `SuperTrend.py`

The SuperTrend indicator identifies trends and provides buy/sell signals based on volatility. It is widely used in various strategies for trend identification.

### Vumanchu Market Cipher

**File**: `VumanchuMarketCipher_A.py`, `VumanchuMarketCipher_B.py`

The Vumanchu Market Cipher indicators combine multiple signals to identify market trends. They are used in the Vumanchu A and Vumanchu B strategies to generate trading signals.

### Order Chain Indicator

**File**: `OrderChain.py`

The Order Chain indicator tracks market orders and volume flow to identify trading signals. It is used in the Order Chain Kioseff Trading strategy.

### Volume Oscillator

**File**: `VolumeOscillator.py`

The Volume Oscillator measures the difference between short-term and long-term volume EMAs to identify trends. It is used in various strategies to confirm signals.

### QQE Indicator

**File**: `QQE.py`

The QQE (Quantitative Qualitative Estimation) indicator combines RSI and volatility measures to generate trading signals. It is used in the QQE Hullband Volume Oscillator strategy.

### RSX

**File**: `RSX.py`

The RSX (Relative Strength Index) indicator measures the strength of price movements to identify overbought and oversold conditions. It is used in various strategies to generate signals.

### Mesa Adaptive Moving Average (MAMA)

**File**: `MesaAdaptiveMovingAverage.py`

The Mesa Adaptive Moving Average (MAMA) adapts to market conditions to provide accurate trend analysis. It is used in the SMA Cross MESAdaptive Prime strategy.

### Hull Moving Average (HMA)

**File**: `HullMovingAverage.py`

The Hull Moving Average (HMA) reduces lag and provides accurate trend analysis. It is used in various strategies to identify trends.

### Zero Lag

**File**: `ZeroLag.py`

The Zero Lag indicator reduces lag in trend analysis by combining multiple moving averages. It is used in various strategies to identify trends.

### Sine Weighted Moving Average

**File**: `SineWeightedMA.py`

The Sine Weighted Moving Average applies sine weights to reduce lag and provide accurate trend analysis. It is used in various strategies to identify trends.

### Fast QQE

**File**: `FastQQE.py`

The Fast QQE indicator is a lightweight version of the QQE indicator, designed for faster calculations. It is used in various strategies to generate signals.

### Fast Volume Oscillator

**File**: `FastVolOsc.py`

The Fast Volume Oscillator is a lightweight version of the Volume Oscillator, designed for faster calculations. It is used in various strategies to confirm signals.

## Conclusion

The BTQuant framework provides a comprehensive set of strategies and indicators designed to enhance trading operations. The framework's transparency, flexibility, and robustness make it a powerful tool for both backtesting and live trading. By leveraging the core components and strategies, users can build and customize their trading systems to suit their specific needs.