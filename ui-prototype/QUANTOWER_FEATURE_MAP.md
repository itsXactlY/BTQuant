# Quantower Feature Map — Full Extraction

## What Quantower IS

Multi-asset, multi-connect trading platform. C# based. Windows native.
40+ trading/analytical panels. 60+ broker/exchange/data feed connections.
Open C# API for custom plugins, indicators, algo-strategies, broker integrations.

---

## ALL QUANTOWER PANELS (40+)

### Analytics
| Abbr | Panel | Description |
|------|-------|-------------|
| Cht | **Chart** | Multi-type charting (candle, line, bar, Heiken-Ashi, Kagi, Renko, Point&Figure, Line Break, Tick, Range) |
| T&S | **Time & Sales** | Real-time trade tape with filters |
| FwC | **Forward Curve** | Futures forward curve analysis |
| PrS | **Price Statistic** | Price statistics and distribution |
| DSf | **DOM Surface** | Depth-of-market heatmap visualization |
| TPO | **TPO Profile Chart** | Market profile / TPO analysis |
| OAn | **Option Analyzer** | Options chain analysis with Greeks |
| MH | **Market Heatmap** | Cross-asset heatmap |
| QB | **Quote Board** | Multi-symbol quote grid |
| SMx | **Stat Matrix** | Statistical matrix analysis |
| SMp | **Symbols Mapping** | Symbol mapping across connections |
| ExT | **Exchange Times** | Exchange trading hours/sessions |
| SeM | **Sessions Manager** | Trading session management |
| HEx | **History Export** | Historical data export |

### Trading
| Abbr | Panel | Description |
|------|-------|-------------|
| DOM | **DOM Trader** | Price ladder trading (Market, Limit, Stop from DOM) |
| MD | **Market Depth** | Order book depth visualization |
| OE | **Order Entry** | Order entry panel |
| MOE | **Multiple OE** | Multi-symbol order entry |
| StM | **Strategies Manager** | Algo strategy management |
| CTr | **Copy Trading** | Copy trading panel |
| TrS | **Trading Simulator** | Paper trading / simulation |

### Portfolio
| Abbr | Panel | Description |
|------|-------|-------------|
| Pos | **Positions** | Open positions with P&L |
| Ord | **Working Orders** | Pending orders management |
| Trd | **Trades** | Trade history |
| OrH | **Orders History** | Order history |
| Acc | **Account Info** | Account balance, equity, margin |
| Prf | **Account Perform** | Account performance analytics |
| CrB | **Crypto Balances** | Crypto portfolio balances |
| BaM | **Backup Manager** | Configuration backup |

### Information
| Abbr | Panel | Description |
|------|-------|-------------|
| Sym | **Symbol Info** | Symbol details and specifications |
| News | **News** | Market news feed |
| Rep | **Report** | Trading reports |
| AlrL | **Alerts Log** | Alert history |
| EvL | **Event Log** | System event log |
| HiS | **Historical Symbols** | Historical symbol data |
| WWW | **Browser** | Embedded web browser |
| HLP | **Support Chat** | Support chat |
| MaR | **Market Replay** | Historical market replay |

---

## CHARTING & ANALYTICS DEEP DIVE

### Chart Types
- Candlestick, Line, Bar
- Heiken-Ashi, Tick chart, Range Bars
- Kagi, Renko, Point & Figure, Line Break
- Time-based chart with custom periods

### Volume Analysis Tools
- **Cluster Chart** — Price + Volume + Time + Order Flow on single chart
- **Volume Profile** — POC, Value Area, various time periods
- **Historical Time & Sales** — Previous trades on selected candle
- **Time Statistic & Histogram** — Delta, Volume, Number of Trades per bar

### VWAP
- Volume Weighted Average Price
- Any period / session
- Benchmark price evaluation

### TPO Profile
- Market profile analysis
- TPO Point of Control, Value Area, Singles
- Split & Merge profiles

### DOM Surface
- Order flow heatmap
- Liquidity visualization per price level
- Limit order density analysis
- Heatmap mode for liquidity changes

### Chart Overlays
- Multiple assets on same chart
- Absolute price scales
- Correlation analysis

### Drawing Tools
- Trends & Channels
- Geometry tools
- Fibonacci & Gann
- Harmonic patterns
- Text & Comments
- Channels

### Indicators (Categories)
- Moving Averages (SMA, EMA, WMA, etc.)
- Trend indicators
- Volatility indicators
- Volume indicators
- Oscillators

---

## ORDER EXECUTION

### Order Types
- **MARKET ORDER** — Instant execution at current price
- **LIMIT ORDER** — Execution at specified or better price
- **STOP ORDER** — Becomes market order at trigger price
- **TIF** — GTD, GTC, IOC, FOK, Day

### Chart Trading
- Submit orders directly from chart
- Drag to change prices
- Cancel or execute by market price
- Hot Buttons for fast trading
- Mouse trading mode

### DOM Trading
- Place/modify/execute from price ladder
- Market, Stop, Limit orders from DOM

### Crypto Trading
- Crypto Balances panel
- One-click currency conversion
- Percentage slider for order sizing

### Trading Simulation
- Real-time market data simulation
- Full trading simulation on non-trading connections

---

## FLEXIBLE INTERFACE

### Panel System
- Each panel is independent, movable
- Panels organized in groups/binds/workspaces
- Save panel configs as templates
- Per-panel settings and defaults
- Multiple monitor support

### Workspaces
- Save/restore workspace layouts
- Quick-switch between workspaces
- Custom panel arrangements

---

## CONNECTIONS (60+)

### Asset Classes
- Cryptocurrency (Binance, ByBit, OKX, Kraken, etc.)
- Futures (Rithmic, CQG, CTS, etc.)
- Stocks (Interactive Brokers, Alpaca, etc.)
- Options (IB, CBOE, etc.)
- Currency/FX (OANDA, Pepperstone, etc.)

### Connection Types
- Crypto-exchange
- Broker
- Data feed
- Prop-trading (Topstep, Apex, etc.)
- Technology

### Simultaneous Connection
- Connect to multiple brokers/exchanges at once
- Compare data from multiple sources
- Combine data into synthetic symbols
- Trade across multiple brokers

### Synthetic Symbols
- Create synthetic instruments with multiple legs
- Exchange-listed spreads
- Custom spread definitions

---

## OPEN C# API

### Custom Development
- **Algo-strategies** — Automated trading algorithms
- **Technical indicators** — Custom indicators
- **Trading assistants** — Decision support tools
- **Broker integrations** — Third-party vendor integration

---

## WHAT BTQUANT ALREADY HAS vs QUANTOWER

| Quantower Feature | BTQuant Equivalent | Status |
|-------------------|-------------------|--------|
| **Chart (candle/line/bar)** | canvas chart in quantower.html | ✅ Implemented |
| **SMA/EMA/VWAP** | SMA 20, EMA 50, VWAP overlays | ✅ Implemented |
| **RSI** | RSI 14 in subchart | ✅ Implemented |
| **Bollinger Bands** | BB 20/2 | ✅ Implemented |
| **MACD** | MACD header value | ⚠️ Header only |
| **Drawing tools** | Trendline, H-line | ✅ Implemented |
| **Order Book** | 10-level depth | ✅ Implemented |
| **Trade Tape** | Live trades | ✅ Implemented |
| **Watchlist** | 12-symbol watchlist | ✅ Implemented |
| **Positions** | Mock positions panel | ✅ Implemented |
| **Strategy Runner** | 10 strategies with start/stop | ✅ Implemented |
| **Price Alerts** | Above/below alerts | ✅ Implemented |
| **Layout System** | 1/2/4 panel layouts | ✅ Implemented |
| **Keyboard Shortcuts** | 13 shortcuts | ✅ Implemented |
| **Cluster Chart** | NOT implemented | ❌ |
| **Volume Profile** | NOT implemented | ❌ |
| **TPO Profile** | NOT implemented | ❌ |
| **DOM Surface/Heatmap** | NOT implemented | ❌ |
| **Forward Curve** | NOT implemented | ❌ |
| **Option Chain/Greeks** | NOT implemented | ❌ |
| **Market Heatmap** | NOT implemented | ❌ |
| **Copy Trading** | NOT implemented | ❌ |
| **Trading Simulator** | NOT implemented | ❌ |
| **Account Info Panel** | NOT implemented | ❌ |
| **Crypto Balances** | NOT implemented | ❌ |
| **News Feed** | NOT implemented | ❌ |
| **Market Replay** | NOT implemented | ❌ |
| **Historical Data Export** | NOT implemented | ❌ |
| **Multiple Order Types** | NOT implemented | ❌ |
| **Chart Trading** | NOT implemented | ❌ |
| **DOM Trading** | NOT implemented | ❌ |
| **Synthetic Symbols** | NOT implemented | ❌ |
| **Multi-Broker Connection** | NOT implemented | ❌ |
| **Panel Binding/Grouping** | NOT implemented | ❌ |
| **Workspace Save/Restore** | NOT implemented | ❌ |
| **Open API** | MCP adapter exists | ⚠️ Different paradigm |
| **Heiken-Ashi** | NOT implemented | ❌ |
| **Kagi/Renko/P&F** | NOT implemented | ❌ |
| **Harmonic Patterns** | NOT implemented | ❌ |
| **Fibonacci Drawing** | Button exists, no logic | ⚠️ Stub |
| **Gann Tools** | NOT implemented | ❌ |
| **Volume Oscillators** | NOT implemented | ❌ |
| **Accumulation/Distribution** | NOT implemented | ❌ |
| **OBV** | NOT implemented | ❌ |
| **CMF** | NOT implemented | ❌ |
| **Session Manager** | NOT implemented | ❌ |
| **Exchange Times** | NOT implemented | ❌ |
| **Backup Manager** | NOT implemented | ❌ |
| **Symbol Info** | NOT implemented | ❌ |

---

## WHAT BTQUANT HAS THAT QUANTOWER DOES NOT

| BTQuant Feature | Description |
|----------------|-------------|
| **HotSpine SHM** | Shared memory real-time data pipeline |
| **C++ Manipulation Detectors** | 5 detectors (stop_hunt, spoofing, etc.) |
| **BTQ Vulkan Render Engine** | C++ rendering panels |
| **Backtrader Fork** | Modified Backtrader with 24 strategies |
| **Ehlers Indicators** | MAMA/FAMA, CyberCycle, etc. |
| **Autonomous Agency** | AI-driven strategy generation (MiMo-V2-Flash) |
| **Neural Trading Pipeline** | Transformer + CNN-LSTM + RL |
| **MCP Adapter** | 33 tools via JSON-RPC |
| **PULSE Macro Integration** | Macro trend mapping |
| **Live Deployer** | Sandbox-gated deployment |
| **BigBrainCentral** | MSSQL market data storage |
| **Research-Spine** | Evolutionary strategy research |
| **QuantStats Integration** | Performance analytics |
| **Transparency Patch** | Indicator audit trail |
