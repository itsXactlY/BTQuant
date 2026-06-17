# Quantower Complete Extraction — Every Page, Every Feature

## PAGES SCRAPED

### Main Site (quantower.com)
1. Homepage — platform overview
2. /assets-and-brokers-features — 40+ panels, 60+ connections
3. /charting-and-analytics-features — chart types, volume tools, drawings
4. /orders-execution-features — order types, chart trading, DOM
5. /options-trading-features — options desk, Greeks, What-If
6. /interface-features — workspaces, groups, linking, themes
7. /pricing — $70/mo all-in-one, extensions, packages
8. /connections — 60+ brokers/exchanges/data feeds
9. /release-notes — latest v1.146.11 (June 13, 2026)
10. /blog — 80+ articles from 2018-2026
11. /b2b — partnership types (7 categories)
12. /faq — 30+ FAQ items
13. /referral — 15% referral program
14. /dom-surface — DOM Surface product page
15. /volumeanalysistools — Volume Analysis product page
16. /tpoprofile — TPO Profile product page
17. /advancedfeatures — Advanced Features page
18. /contact-us — contact form

### Documentation (help.quantower.com)
19. Analytics Panels — Chart, Watchlist, T&S, Price Statistic, DOM Surface, Option Analytics, TPO
20. Trading Panels — Chart Trading, Order Entry, DOM Trader, Copy Trading, Market Depth, Simulator, Market Replay, FX Cell
21. Portfolio Panels — Positions, Working Orders, Trades, Orders History, Synthetic Symbols, Historical Symbols
22. Informational Panels — Account Info, Event Log, News, etc.
23. Customization — Localization
24. Quantower Algo — C# API development

### API Reference (api.quantower.com)
25. Core — TradingPlatform.BusinessLayer.Core
26. Connections — TradingPlatform.BusinessLayer.Connection
27. History — TradingPlatform.BusinessLayer.HistoricalData
28. BusinessObjects — TradingPlatform.BusinessLayer

---

## COMPLETE PANEL LIST (40+)

### Analytics Panels
| # | Panel | Description | Category |
|---|-------|-------------|----------|
| 1 | Chart | Multi-type charting with overlays, indicators, drawings | Core |
| 2 | Watchlist | Symbol watchlist with real-time data | Core |
| 3 | Time & Sales | Real-time trade tape | Core |
| 4 | Price Statistic | Volume distribution per price | Volume |
| 5 | DOM Surface | Order book heatmap | Volume |
| 6 | Option Analytics | Options chain with Greeks | Options |
| 7 | TPO Profile | Market Profile / TPO chart | Volume |
| 8 | Forward Curve | Futures forward curve | Analytics |
| 9 | Quote Board | Multi-symbol quote grid | Analytics |
| 10 | Market Heatmap | Cross-asset heatmap | Analytics |
| 11 | Stat Matrix | Statistical matrix | Analytics |
| 12 | Symbol Info | Symbol specifications | Info |
| 13 | Sessions Manager | Trading session management | Misc |
| 14 | Exchange Times | Exchange trading hours | Misc |
| 15 | History Export | Historical data export | Misc |

### Trading Panels
| # | Panel | Description | Category |
|---|-------|-------------|----------|
| 16 | DOM Trader | Price ladder trading | Core |
| 17 | Market Depth | Order book depth | Core |
| 18 | Order Entry | Order placement | Core |
| 19 | Multiple OE | Multi-symbol order entry | Core |
| 20 | Copy Trading | Copy trading between accounts | Core |
| 21 | Strategies Manager | Algo strategy management | Algo |
| 22 | Trading Simulator | Paper trading | Simulator |
| 23 | Market Replay | Historical replay | Simulator |
| 24 | FX Cell | Best bid/ask with market order | FX |

### Portfolio Panels
| # | Panel | Description | Category |
|---|-------|-------------|----------|
| 25 | Positions | Open positions with P&L | Core |
| 26 | Working Orders | Pending orders | Core |
| 27 | Trades | Trade history | Core |
| 28 | Orders History | Order history | Core |
| 29 | Synthetic Symbols | Custom spreads | Analytics |
| 30 | Historical Symbols | Import historical data | Data |

### Information Panels
| # | Panel | Description | Category |
|---|-------|-------------|----------|
| 31 | Account Info | Balance, equity, margin | Core |
| 32 | Account Perform | Performance analytics | Analytics |
| 33 | Crypto Balances | Crypto portfolio | Crypto |
| 34 | Event Log | System events | Info |
| 35 | News | Market news | Info |
| 36 | Alerts Log | Alert history | Info |
| 37 | Report | Trading reports | Analytics |

### Miscellaneous Panels
| # | Panel | Description | Category |
|---|-------|-------------|----------|
| 38 | Browser | Embedded web browser | Misc |
| 39 | Support Chat | Support chat | Misc |
| 40 | Backup Manager | Config backup | Misc |
| 41 | Symbols Mapping | Symbol mapping | Misc |

---

## CHART TYPES (12)
1. Candlestick
2. Line
3. Bar
4. Heiken-Ashi
5. Renko
6. Kagi
7. Point & Figure
8. Range Bars
9. Line Break
10. Volume Bars
11. Tick Bars
12. Reversal Bars

## TECHNICAL INDICATORS (100+)

### Moving Averages
- SMA, EMA, DEMA, TEMA, WMA, HMA, KAMA
- McGinley Dynamic, Modified Moving Average
- Pivot Point Moving Average, Regression Line
- Guppy Multiple Moving Average
- Trend Breakout System

### Oscillators
- RSI, MACD, CCI, Williams %R
- Awesome Oscillator, Accelerator Oscillator
- Momentum, Rate of Change, Stochastic
- Balance of Power, Delta Divergence Reversal
- Relative Spread Strength

### Trend
- ADX, DMI, Ichimoku Cloud, ZigZag
- Bionic Candle

### Volatility
- ATR, Standard Deviation, Bollinger Bands
- Keltner Channel, Donchian Channel
- Price Channel, Moving Average Envelope

### Volume
- Volume Profile, VWAP, Anchored VWAP
- OBV, Chaikin Money Flow, Accum/Dist
- Depth of Bid/Ask, Delta Flow, Delta Rotation
- Level2, Abnormal Volume, Abnormal Trades
- Volume Impulse, COT High/Low

### Channels
- Range Marker, Fair Value Gap (FVG)
- High Low, Round Numbers, Highest High, Lowest Low
- Bollinger Bands Flat

### Custom/Advanced
- Power Trades
- DOM Surface
- Cluster Chart
- TPO Profile
- Session Separators

## DRAWING TOOLS
- Trends & Channels
- Geometry tools
- Fibonacci & Gann
- Harmonic patterns
- Text & Comments
- Custom Profiles
- Elliot Wave
- Circle
- Quick Ruler

## VOLUME ANALYSIS TOOLS
- Cluster Chart (Footprint)
  - Up to 4 columns
  - Imbalance zones
  - Multiple coloring modes
  - Delta chart
- Volume Profiles
  - Right, Left, Custom, Step
  - POC, Value Area, VAH, VAL
- Time Statistics
  - 20+ data types
  - Delta, Volume, Trades
- Time Histogram
- Historical Time & Sales
- VWAP / Anchored VWAP
- Power Trades Scanner
- Volume Impact indicator
- Dynamic VPOC indicator

## OPTIONS FEATURES
- Options Desk (Calls/Puts)
- Options Analyzer
- Greeks (Delta, Gamma, Vega, Theta, Rho)
- What-If Scenario Analysis
- Payoff Chart
- Profile Overlays
- Volatility Smile/Skew
- Option Strategies

## ORDER TYPES
- Market Order
- Limit Order
- Stop Order
- Stop Limit
- OCO (One Cancels Other)
- Trailing Stop
- TIF: GTD, GTC, IOC, FOK, Day

## TRADING FEATURES
- Chart Trading (click to place orders)
- Mouse Trading (drag to set price)
- Hot Buttons for fast execution
- DOM Trading (price ladder)
- Order Placing Strategies
- Local SL/TP
- Breakeven Offset
- Copy Trading (Parent/Child accounts)
- Trading Simulator (100 accounts max)
- Market Replay (History Player)

## CONNECTIONS (60+)
### Crypto Exchanges
Binance, Binance US, Binance Futures, ByBit, OKX, Kraken, KuCoin, Bitfinex, BitMEX, Gate.io, Huobi, HitBTC, CoinMetro, Crypto.com, WOO X, RabbitX, Delta Exchange, Deribit

### Futures Brokers
Rithmic, CQG, CTS, AMP, Interactive Brokers, Tradier, DxTrade, Quantower Trader

### Prop Trading
Topstep, Apex Trader Funding, Bulenox, Earn2Trade, Elite Trader Funding, Leeloo Trading, OneUpTrader, Take Profit Trader, TickTick Trader, Phidias Propfirm, Phoenix Trader Funding, Funded Futures Network, Evalu8trading

### FX Brokers
OANDA, FXCM, Pepperstone, IC Markets, FxPro, LMAX, cTrader, Kimura Trading, Purple Trading, AXIORY, TOPFX, TradePro, TradersWay, Tradeview

### Data Feeds
dxFeed, Barchart, Polygon.io, AlphaVantage, iQFEED, Quotemedia, Intrinio, MetaStock

### Stocks
Interactive Brokers, Alpaca, Tradier, LYNX

## INTERFACE FEATURES
- Custom Workspaces (unlimited)
- Groups & Binds
- Panel Linking (by color)
- Panel Templates
- 5 Color Themes:
  - Dark Blue (default)
  - Dark Autumn
  - Dark Forest
  - Dark Gold
  - Light Water
- Multiple Monitor Support
- Per-panel Settings

## PRICING
- All-in-One: $70/month (30% off annual)
- Extensions available separately:
  - DOM Surface
  - Power Trades
  - Volume Analysis
  - TPO Profile
  - Advanced Features
  - Option Trading
  - Multi-asset Package
  - Crypto Package

## API (C#)
- Core: TradingPlatform.BusinessLayer.Core
- Connections: TradingPlatform.BusinessLayer.Connection
- History: TradingPlatform.BusinessLayer.HistoricalData
- BusinessObjects: TradingPlatform.BusinessLayer
- Custom indicators, strategies, broker integrations

## LATEST FEATURES (v1.146.11)
- Cluster chart label hiding
- Heiken-Ashi smoothing options
- DOM Trader position bar enhancements
- Volume analysis tools improvements
- Power trades scanner with Telegram alerts
- Trading simulator 100 accounts
- DOM Surface total bids/asks
- Market depth coloring modes
- Strategy manager template folders
- Hotkeys for cancelling orders
