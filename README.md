Work in Progress. If u came that far: https://discord.gg/Y7uBxmRg3Z

# BTQuant Framework

## Overview

This framework is designed to provide robust and efficient tools for backtesting, forward trading, and live trading. Leveraging the power of Backtrader, this framework integrates seamlessly with JackRabbitRelay to mimic over 400 exchanges for forward trading, and for live trading via a 1-second websocket data feed.

## Features

### Backtesting
- Utilizes **Backtrader** for comprehensive backtesting of trading strategies.
- Test and validate your strategies with historical data to ensure robustness and reliability.

### Forward Trading
- Powered by **Backtrader** in conjunction with **JackRabbitRelay**.
- Mimic over 400 exchanges, enabling thorough forward testing and simulation of real-world trading conditions.
- We act as our own broker, providing full control and customization.

### Live Trading
- Integrated with **Binance Websocket** (for now) for real-time data.
- Supports 1-second websocket data feeds for precise and timely trading decisions.
- Implements real-time dollar cost averaging.
- Requires as few as 50 lines of code to get started with your own trading Strategy.

## Getting Started

### Prerequisites
- Python 3.7+
- (Custom-)Backtrader
- JackRabbitRelay

### Installation


License
=======

[MIT](https://choosealicense.com/licenses/mit)
