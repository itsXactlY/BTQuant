# BTQuant Frequently Asked Questions

## General Questions

### What is BTQuant?

BTQuant is an institutional-grade algorithmic trading framework built on Backtrader. It provides advanced features for market data ingestion, strategy development, backtesting, and live trading with support for multiple exchanges and SQL Server data storage.

### Is BTQuant free to use?

Yes, BTQuant is open-source and free to use. It's built on the open-source Backtrader framework with additional enterprise-grade features.

### What programming language is BTQuant written in?

BTQuant is written in Python 3.12+ and extends the Backtrader framework. It also includes C++ components for high-performance market data ingestion.

### What operating systems does BTQuant support?

BTQuant is designed for Linux systems (Ubuntu 20.04+, CentOS 8+, Arch Linux, or equivalent). Windows and macOS support is limited and not officially recommended for production use.

### How does BTQuant differ from other trading frameworks?

BTQuant stands out with:
- **Institutional-grade data spine**: C++ market data ingestion with SQL Server storage
- **Multi-exchange support**: Native WebSocket feeds and CCXT integration
- **Advanced strategy framework**: Built-in DCA, risk management, and live trading
- **Enterprise features**: Zero latency CPU cyle friendly Hotspine data collection cache, CCAPI Websocket/FIX/REST order engine, JackRabbitRelay integration, and Web3 support

## Installation and Setup

### Do I need programming experience to use BTQuant?

Yes, BTQuant requires Python programming knowledge. You should be comfortable with:
- Python syntax and object-oriented programming
- Working with APIs and data structures
- Basic understanding of financial markets and trading concepts

### What are the minimum system requirements?

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, Arch Linux)
- **Python**: 3.12 or 3.13
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 20GB free disk space
- **Network**: Stable internet connection

### Can I install BTQuant on Windows?

While possible, Windows installation is not recommended for production use due to:
- Limited C++ compilation support
- Performance limitations with SQL Server
- Better Linux compatibility for exchange APIs

### Do I need SQL Server to use BTQuant?

SQL Server is recommended for full functionality but not strictly required. You can use:
- **With SQL Server**: Full data storage, Hotspine integration, enterprise features
- **Without SQL Server**: Limited to CCXT data fetching and basic backtesting, rely on "slower" Custom written python Websocket endpoints

### How long does installation take?

Typical installation time:
- **Automated installer**: 10-45 minutes
- **Manual installation**: 30-120 minutes
- **First-time setup**: Additional 10-20 minutes for configuration

## Data and Market Feeds

### Which exchanges does BTQuant support?

**CCXT-compatible exchanges** (100+ exchanges):
- Binance, Bybit, Kraken, Coinbase Pro, KuCoin, and many more

**Native Tickdata WebSocket feeds**:
- Binance, Bitget, MEXC (optimized performance)

**Web3 support**:
- PancakeSwap (Binance Smart Chain)

### How do I get historical data?

BTQuant provides multiple data sources:
```python
# CCXT data (recommended for beginners)
from backtrader.utils.ccxt_data import get_crypto_data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')

# SQL Server data (for advanced users)
from backtrader.feeds.mssql_crypto import get_database_data
data = get_database_data('BTC', '2024-01-01', '2024-01-31', '1h', 'USDT')
```

### What timeframes are supported?

BTQuant supports all standard timeframes:
- **Seconds**: 1s, 5s, 15s, 30s
- **Minutes**: 1m, 5m, 15m, 30m
- **Hours**: 1h, 2h, 4h, 6h, 12h
- **Days**: 1d, 3d, 7d
- **Weeks**: 1w
- **Months**: 1M

### Can I use my own data source?

Yes, BTQuant supports custom data feeds:
```python
import pandas as pd
from backtrader.feeds.polarfeed import PolarsData

# Load your data
df = pd.read_csv('my_data.csv')
data = PolarsData(dataname=df)
```

### How much historical data can BTQuant handle?

BTQuant can handle large datasets:
- **Memory**: Efficient data loading with caching
- **Storage**: SQL Server can store billions of records
- **Performance**: Optimized for high-frequency data

## Strategy Development

### Do I need to be a professional trader to use BTQuant?

No, BTQuant is designed for both beginners and professionals. Start with:
- **Simple strategies**: Moving average crossovers
- **Examples**: Use provided strategy templates
- **Backtesting**: Test strategies before live trading

### How do I create my first strategy?

Start with the quickstart guide:
```python
from backtrader.strategies.base import BaseStrategy

class MyFirstStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=20)
    
    def buy_or_short_condition(self):
        if not self.buy_executed and self.data.close[0] > self.sma[0]:
            self.create_order('BUY')
            return True
        return False
```

### Can I use machine learning with BTQuant?

Yes, BTQuant supports machine learning integration:
```python
from sklearn.ensemble import RandomForestClassifier

class MLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()
        # Train model with historical data
    
    def next(self):
        # Extract features
        features = self._extract_features()
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Trade based on prediction
        if prediction == 1:
            self.create_order('BUY')
```

### How do I optimize my strategy parameters?

Use the built-in optimization tools:
```python
from backtrader.utils.backtest import optimize_backtest

results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    sma_period=[10, 20, 30],
    rsi_period=[14, 20, 28],
    max_workers=4
)
```

### Can I backtest multiple strategies at once?

Yes, use bulk backtesting:
```python
from backtrader.utils.backtest import bulk_backtest

results = bulk_backtest(
    strategy=MyStrategy,
    coins=['BTC', 'ETH', 'ADA', 'SOL'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='1h',
    max_workers=4
)
```

## Live Trading

### Can I use BTQuant for live trading?

Yes, BTQuant supports live trading with:
- **Multiple exchanges**: Binance, Bybit, Kraken, etc.
- **Risk management**: Built-in stop-loss and position sizing
- **Alerts**: Discord and Telegram notifications
- **Monitoring**: Real-time performance tracking

### Is live trading safe with BTQuant?

BTQuant includes safety features:
- **Paper trading**: Test strategies before going live
- **Risk limits**: Maximum position sizes and stop-losses
- **Error handling**: Graceful failure recovery
- **Monitoring**: Real-time alerts and logging

### How do I start live trading?

1. **Paper trade first**: Test with simulated orders
2. **Start small**: Use minimal capital initially
3. **Monitor closely**: Watch performance and errors
4. **Scale gradually**: Increase position sizes over time

### Can I trade multiple exchanges simultaneously?

Yes, BTQuant supports multi-exchange trading:
```python
# Configure multiple exchanges
exchanges = {
    'binance': {'api_key': '...', 'secret': '...'},
    'bybit': {'api_key': '...', 'secret': '...'},
    'kraken': {'api_key': '...', 'secret': '...'}
}

# Trade across exchanges
for exchange_name, config in exchanges.items():
    # Execute trades on each exchange
    pass
```

### What about slippage and fees?

BTQuant accounts for trading costs:
```python
# Configure commission and slippage
cerebro.broker.setcommission(commission=0.00075)  # 0.075%
cerebro.broker.set_slippage_perc(perc=0.0005)     # 0.05% slippage
```

## Performance and Scalability

### How fast is BTQuant?

Performance depends on your setup:
- **Backtesting**: 1000+ bars/second on modern hardware
- **Live trading**: Real-time order execution
- **Data ingestion**: 100,000+ trades/second with Hotspine

### Can BTQuant handle high-frequency trading?

Yes, BTQuant supports HFT with:
- **C++ market data ingestion**: Microsecond precision
- **Optimized data storage**: SQL Server with proper indexing
- **Low-latency execution**: Direct exchange API integration

### How much memory does BTQuant use?

Memory usage varies:
- **Simple strategies**: 100-500 MB
- **Complex strategies**: 1-4 GB
- **Large datasets**: 4-16 GB (with proper caching)

### Can I run BTQuant on a VPS?

Yes, BTQuant works well on VPS:
- **Recommended**: 4+ CPU cores, 8+ GB RAM
- **Operating system**: Ubuntu 20.04 LTS
- **Network**: Low-latency connection to exchanges

## Data Management

### How do I backup my data?

SQL Server backup strategies:
```bash
# Full database backup
sqlcmd -S localhost -U SA -P "YourStrong!Passw0rd" -Q "BACKUP DATABASE BinanceData TO DISK = '/backup/BinanceData.bak'"

# Automated backups with cron
0 2 * * * /path/to/backup_script.sh
```

### Can I export my trading results?

Yes, multiple export formats:
```python
# Export to CSV
results.to_csv('trading_results.csv')

# Export to JSON
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# QuantStats HTML report
from backtrader.utils.backtest import backtest
backtest(strategy, data, quantstats=True)  # Generates HTML report
```

### How do I clean up old data?

Data management utilities:
```python
# Delete old data from SQL Server
import pyodbc
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
cursor.execute("DELETE FROM trades WHERE timestamp < DATEADD(day, -30, GETDATE())")
conn.commit()
conn.close()
```

## Troubleshooting

### My strategy isn't working, what should I do?

1. **Check data**: Ensure data is loading correctly
2. **Enable debug mode**: Add `debug=True` to strategy parameters
3. **Review logs**: Check console output for errors
4. **Test incrementally**: Add features one at a time
5. **Use examples**: Start with working examples

### I'm getting "No data available" errors, how do I fix this?

Common causes and solutions:
- **Invalid symbol**: Check symbol format (e.g., 'BTC/USDT')
- **Date range**: Ensure dates are valid and within exchange history
- **Exchange issues**: Try different exchange or timeframe
- **Rate limits**: Add delays between requests

### How do I debug strategy logic?

Use debugging techniques:
```python
class DebugStrategy(BaseStrategy):
    params = (('debug', True),)
    
    def next(self):
        if self.p.debug:
            print(f"Price: {self.data.close[0]}")
            print(f"Position: {self.position.size}")
            print(f"Indicators: {self.sma[0]}")
        
        super().next()
```

## Legal and Compliance

### Is algorithmic trading legal?

Yes, algorithmic trading is legal in most jurisdictions, but:
- **Check local regulations**: Some countries have restrictions
- **Exchange rules**: Follow exchange terms of service
- **Tax implications**: Consult tax professional for reporting

### Do I need a license to use BTQuant?

No license required for personal use. For commercial use:
- **Review licensing**: Check BTQuant and Backtrader licenses
- **Consult legal**: Verify compliance with local laws
- **Exchange agreements**: Follow exchange API terms

### Is my data secure with BTQuant?

BTQuant security practices:
- **Local storage**: Data stored on your infrastructure
- **API keys**: Never transmitted to third parties
- **Encryption**: Use encrypted connections where possible
- **Access control**: Limit access to sensitive files

## Support and Community

### Where can I get help?

Support options:
- **Documentation**: Comprehensive guides and examples
- **GitHub issues**: Report bugs and feature requests
- **Community**: Join discussions and share knowledge
- **Professional support**: Available for enterprise users

### How do I contribute to BTQuant?

Contribution guidelines:
1. **Fork the repository**: Create your own copy
2. **Follow coding standards**: Use consistent style and documentation
3. **Test thoroughly**: Ensure changes don't break existing functionality
4. **Submit pull request**: Describe changes and reasoning

### Can I request new features?

Yes, feature requests are welcome:
- **GitHub issues**: Submit detailed feature requests
- **Community discussion**: Gather feedback from other users
- **Implementation**: Consider implementing the feature yourself

## Advanced Topics

### Can I use BTQuant with Docker?

No, Docker support and any questions related to BTQuand and Docker will and must be ignored.
Squeeze all into containers, make all 10 times more complexer and harder to debug wont happen, ever.

### How do I deploy BTQuant in production?

Production deployment checklist:
- **Infrastructure**: Reliable servers with backup power
- **Monitoring**: 24/7 monitoring and alerting
- **Security**: Firewall, VPN, and access controls
- **Backup**: Regular data and configuration backups
- **Testing**: Staging environment for testing changes

### Can I integrate BTQuant with other systems?

Yes, BTQuant provides integration points:
- **REST APIs**: Custom endpoints for external systems
- **Message queues**: RabbitMQ, Redis for real-time updates
- **Databases**: Direct SQL Server integration
- **Monitoring**: Prometheus, Grafana for metrics

This FAQ covers the most common questions about BTQuant. For more detailed information, refer to the specific documentation sections or seek community support.