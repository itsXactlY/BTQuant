# BTQuant Vulkan Trading Dashboard - Complete Redesign Specification

## Project Overview
Transform the existing BTQuant Vulkan trading dashboard into a professional, institutional-grade quantitative trading platform. The current implementation uses Vulkan for rendering, Dear ImGui for UI, and shared memory for data feeds. This redesign adds professional charting, multi-symbol support, advanced order book visualization, and comprehensive trading tools.

## Current State Analysis
**Existing Features:**
- Basic line chart with auto-scaling
- Simple position & P&L display
- Strategy control panel with start/stop
- Basic order entry (BTC/USDT only)
- Time & Sales tape with delta calculation
- Minimal order book display (~10 levels)

**Critical Missing Features:**
- No menu bar or navigation system
- No proper candlestick charts
- Cannot view multiple symbols simultaneously
- No timeframe selection
- No technical indicators
- No drawing tools
- Limited order book depth (needs 20+ levels)
- No watchlist or symbol search

---

## 1. MAC-STYLE MENU BAR SYSTEM

### Top-Level Menu Structure
Create a professional menu bar at the very top of the application with these menus:

**File | Markets | Charts | Strategies | Tools | Window | Help**

### Menu Specifications

#### **File Menu**
- New Workspace
- Load Workspace...
- Save Workspace / Save Workspace As...
- Import Strategy Configuration
- Export Trade History (CSV/JSON)
- Settings / Preferences
- Exit Application

#### **Markets Menu** ⭐ PRIMARY NAVIGATION
This is the main way users select coins to trade.

**Quick Select Section** (Top favorites with hotkeys):
- BTC/USDT (Ctrl+1)
- ETH/USDT (Ctrl+2)  
- SOL/USDT (Ctrl+3)
- DOT/USDT (Ctrl+4)
- [Additional 6 configurable favorites]

**Search Markets...** (Ctrl+M):
- Opens modal dialog with search input
- Live fuzzy search filtering as user types
- Shows: Symbol, Last Price, 24h Change %, 24h Volume
- Arrow keys to navigate, Enter to select
- Opens chart for selected symbol

**By Exchange** submenu:
- Binance Futures → [List all available symbols]
- Bybit → [List all available symbols]  
- OKX → [List all available symbols]
- [Other connected exchanges]

**By Category** submenu:
- DeFi Tokens
- Layer 1 Blockchains
- Layer 2 Solutions
- Meme Coins
- Top 20 by 24h Volume
- Top Gainers (24h)
- Top Losers (24h)

**Manage Watchlist**:
- Add Current Symbol to Favorites
- Create Custom Watchlist
- Edit Watchlists
- Import/Export Watchlist (JSON)

#### **Charts Menu** ⭐ MULTI-CHART MANAGEMENT
**New Chart Window** (Ctrl+N):
- Opens symbol selector dialog
- Creates new chart in tab or grid slot

**Chart Layouts** submenu:
- Single Chart (Full Width)
- 2 Charts - Horizontal Split
- 2 Charts - Vertical Split  
- 4 Charts Grid (2x2)
- 6 Charts Grid (2x3)
- 9 Charts Grid (3x3)
- Custom Layout Editor

**Active Charts** (Dynamic list of open charts):
- □ BTC/USDT - 15m (Ctrl+F1)
- □ ETH/USDT - 5m (Ctrl+F2)
- ☑ SOL/USDT - 1h (Ctrl+F3) ← Currently focused
- [Additional charts...]
- Checkboxes toggle visibility
- Hotkeys switch focus

**Chart Settings** submenu:
- Default Timeframe (1m, 5m, 15m, 1h, etc.)
- Default Chart Type (Candlestick, Line, Area, etc.)
- Auto-scale Price Axis
- Sync Crosshair Across Charts
- Theme / Color Scheme
- Grid Line Settings

#### **Strategies Menu**
- Run Strategy...
- Stop All Strategies (Emergency Kill)
- Pause/Resume Strategy
- Strategy Performance Report
- Backtest Manager
- Strategy Editor/IDE
- Import Strategy File
- Export Strategy Configuration

#### **Tools Menu**
- Market Screener (Filter by technical conditions)
- Correlation Matrix (Multi-symbol correlation)
- Volatility Surface Viewer
- Liquidation Heatmap
- Funding Rate History Chart
- Alert Manager
- Trade Journal/Notes
- Economic Calendar
- Position Calculator

#### **Window Menu**
- Minimize All Panels
- Restore Default Layout
- **Show/Hide Panels** submenu:
  - ☑ Order Book
  - ☑ Time & Sales Tape
  - ☑ Positions & P&L
  - ☑ Order Entry
  - □ System Logs
  - ☑ Strategy Control
  - ☑ Watchlist
- Save Current Layout As...
- Manage Saved Layouts
- Full Screen (F11)

#### **Help Menu**
- Keyboard Shortcuts Reference
- User Guide / Documentation
- API Connection Status
- Check for Updates
- Report Bug / Send Feedback
- About BTQuant

### Menu Bar Implementation Notes
- Always visible at top of window
- Dark theme consistent with rest of UI
- Keyboard navigation (Alt+F for File menu, etc.)
- Recently used items at top of relevant menus
- Disabled menu items shown in gray
- Tooltips on hover for complex menu items

---

## 2. PROFESSIONAL CANDLESTICK CHART SYSTEM

Replace the current line chart with a fully-featured professional trading chart.

### Core Chart Components

#### **Candlestick Rendering**
- **OHLC Candles**: 
  - Bullish (close > open): Green body (#00FF88) with green wick
  - Bearish (close < open): Red body (#FF4444) with red wick
  - Doji (close ≈ open): Thin line or small body
  - Configurable body width (50-90% of bar width)

- **Chart Types** (switchable):
  - Candlesticks (standard OHLC)
  - Hollow Candles (hollow when bullish, filled when bearish)
  - Heikin-Ashi (smoothed candles for trend identification)
  - Line Chart (close prices only)
  - Area Chart (filled area under line)
  - Renko Bricks (price-based, time-independent)
  - OHLC Bars (traditional bar chart)

#### **Volume Sub-Panel**
- Below price chart, typically 20-30% of chart height
- Volume bars colored by price action:
  - Green if close > open
  - Red if close < open
- Overlay: Volume Moving Average (20 period default)
- Separable y-axis from price chart
- Hoverable to show exact volume values

#### **Axes & Grid**
- **Price Axis** (right side):
  - Auto-scaling with 10% padding
  - Manual scale lock option
  - Log scale toggle
  - Price labels at grid lines
  - Current price highlighted with horizontal line
  
- **Time Axis** (bottom):
  - Adaptive labels (show seconds/minutes/hours/days based on zoom)
  - Vertical grid lines at major intervals
  - Current time marker

- **Grid Lines**:
  - Horizontal: Every major price level
  - Vertical: Every major time interval  
  - Subtle dark gray (#333333) color
  - Toggle on/off

#### **Timeframe Selector**
Display prominent button bar above chart:
```
[1s] [5s] [15s] [30s] [1m] [5m] [15m] [30m] [1h] [4h] [12h] [1D] [1W] [1M] [Custom...]
```

- Active timeframe: Highlighted with accent color (#00A8FF)
- Custom input: User can type "3m", "2h", "6h", etc.
- Keyboard shortcuts: 1-9 keys for quick timeframe switch
- Smooth data loading transition when switching

### Interactivity Features

#### **Zoom & Pan**
- **Mouse Wheel**: Zoom in/out horizontally (time axis)
- **Shift + Mouse Wheel**: Zoom vertically (price axis)
- **Ctrl + Mouse Wheel**: Zoom both axes proportionally
- **Middle Mouse Button + Drag**: Pan chart in any direction
- **Arrow Keys**: Pan left/right/up/down
- **Double Click**: Auto-fit to visible data
- **Home Key**: Jump to most recent candle
- **End Key**: Jump to oldest loaded candle

#### **Crosshair & Info Display**
- **Crosshair Lines**: Vertical and horizontal lines following mouse
- **Info Box** (floating tooltip near crosshair):
  - Time: 2024-01-15 14:32:00
  - Open: 96,905.20
  - High: 96,920.50
  - Low: 96,890.00
  - Close: 96,910.30
  - Volume: 145.23 BTC
  - Change: +5.10 (+0.005%)

- **Price Label on Y-Axis**: Shows price at crosshair horizontal
- **Time Label on X-Axis**: Shows time at crosshair vertical
- **Snap to Candle**: Option to snap crosshair to nearest OHLC

#### **Range Selection**
- **Click + Drag**: Select time range
- **Selection Overlay**: Semi-transparent highlight
- **Selection Stats Box**:
  - Start Time → End Time
  - Duration: 2h 30m (150 bars)
  - Price Change: +245.80 (+0.25%)
  - High: 97,100.00
  - Low: 96,820.00
  - Volume: 1,234.56 BTC
- **Actions**: Zoom to Selection, Export Data, Run Backtest on Range

### Technical Indicators

#### **Overlay Indicators** (drawn on price chart)
**Moving Averages**:
- SMA (Simple Moving Average): 7, 25, 50, 99, 200 periods
- EMA (Exponential Moving Average): 9, 21, 55, 200 periods
- WMA (Weighted Moving Average)
- Each MA: Configurable color, line width, period
- Legend showing current values

**Bollinger Bands**:
- Period: 20 (default, configurable)
- Standard Deviations: 2 (default)
- Middle band: SMA
- Upper/Lower bands: ±2σ from middle
- Semi-transparent fill between bands (#00A8FF at 20% opacity)

**VWAP** (Volume Weighted Average Price):
- Session VWAP line (resets daily/weekly/custom)
- VWAP Standard Deviation Bands (±1σ, ±2σ, ±3σ)
- Different line style (dashed or dotted)

**Pivot Points**:
- Types: Traditional, Fibonacci, Camarilla, Woodie's
- Shows: R3, R2, R1, PP (Pivot), S1, S2, S3
- Horizontal lines with labels

**Ichimoku Cloud**:
- Tenkan-sen, Kijun-sen, Senkou Span A & B
- Cloud fill (green/red based on Span A/B relationship)

**Parabolic SAR**:
- Dots above/below candles indicating trend direction

#### **Sub-Panel Indicators** (separate panels below chart)
**RSI** (Relative Strength Index):
- Period: 14 (default)
- Overbought line: 70 (red dashed)
- Oversold line: 30 (green dashed)
- Midline: 50 (gray dashed)
- Color-coded line: Green above 50, Red below 50
- Divergence detection highlighting

**MACD** (Moving Average Convergence Divergence):
- Fast: 12, Slow: 26, Signal: 9
- MACD Line (blue), Signal Line (orange)
- Histogram bars (green/red based on sign)
- Zero line

**Stochastic Oscillator**:
- %K and %D lines
- Period: 14, Smoothing: 3
- Overbought (80) / Oversold (20) zones

**ATR** (Average True Range):
- Period: 14
- Line chart showing volatility
- Useful for stop-loss placement

**OBV** (On-Balance Volume):
- Cumulative volume line
- Divergence with price indicates potential reversals

**Money Flow Index (MFI)**:
- Volume-weighted RSI
- Period: 14
- Overbought (80) / Oversold (20)

**Indicator Management**:
- **Add Indicator Button** → Opens indicator library
- **Indicator Settings**: Click indicator name to edit parameters
- **Show/Hide**: Toggle visibility without removing
- **Remove**: Delete indicator from chart
- **Reorder**: Drag-and-drop to change sub-panel order

### Drawing Tools

#### **Toolbar** (left side of chart or top)
```
┌────────────────┐
│ →  Cursor      │ (Default selection mode)
│ ╱  Trendline   │
│ ─  Horizontal  │
│ │  Vertical    │
│ ▭  Rectangle   │
│ ○  Circle      │
│ φ  Fibonacci   │ (Retracement, Extension, Fan, Arc)
│ ⚡ Pitchfork   │
│ 📐 Triangle    │
│ ✏  Text/Note   │
│ 🗑  Delete      │
└────────────────┘
```

#### **Drawing Features**
- **Trendlines**: Click two points to draw, extends infinitely or to screen edge
- **Horizontal Line**: Support/Resistance levels, snaps to candle high/low
- **Vertical Line**: Mark important time events
- **Rectangle**: Highlight consolidation zones, channels
- **Fibonacci Retracement**: 
  - Click swing low → swing high (or reverse)
  - Shows: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
  - Configurable levels and colors
- **Text Annotations**: Add notes at specific price/time coordinates
- **Drawing Properties**:
  - Line color picker
  - Line width (1-5px)
  - Line style: Solid, Dashed, Dotted
  - Extend left/right/both/none
  - Lock/Unlock (prevent accidental movement)

#### **Drawing Management**
- **Selection**: Click on drawing to select (shows handles)
- **Edit**: Drag endpoints to adjust, drag middle to move
- **Copy/Paste**: Duplicate drawings
- **Delete**: Delete key or trash icon
- **Layer Order**: Bring to front / Send to back
- **Persistence**: All drawings saved with workspace
- **Alerts on Drawings**: Right-click horizontal line → "Create Alert at this Price"

### Trade Markers

Display executed trades directly on the chart:

- **Entry Markers**:
  - Long: Green triangle pointing up ▲ at entry price
  - Short: Red triangle pointing down ▼ at entry price
  - Size-based scaling: Larger markers for larger positions
  
- **Exit Markers**:
  - Cross symbol × at exit price
  - Color: Green if profitable, Red if loss
  
- **Position Lines**:
  - Horizontal line from entry to exit
  - Solid if still open, dashed if closed
  - Color: Position side (green long, red short)
  
- **P&L Labels**:
  - Floating label near exit marker
  - Shows: +$1,234.56 (+2.34%) or -$567.89 (-1.12%)
  
- **Stop Loss / Take Profit**:
  - Dashed horizontal lines at SL/TP levels
  - Red for SL, Green for TP
  - Updates in real-time as price moves
  
- **Hover Details** (when hovering over trade marker):
  - Entry Time: 2024-01-15 10:32:15
  - Entry Price: $96,500.00
  - Exit Time: 2024-01-15 12:45:30
  - Exit Price: $97,250.00
  - Size: 0.5 BTC
  - P&L: +$375.00 (+0.78%)
  - Duration: 2h 13m 15s
  - Fees: $15.30

### Chart Settings Panel

**Accessible via**: Charts Menu → Chart Settings or Right-click on chart → Settings

**Appearance**:
- Color Scheme: Dark (default), Light, Custom
- Candle Colors: Bullish, Bearish, Doji
- Background Color & Gradient
- Grid Line Color & Style
- Crosshair Color & Width

**Behavior**:
- Auto-scale: On/Off/Visible Range Only
- Scale Mode: Linear / Logarithmic
- Show Volume: Yes/No, Height %
- Snap Crosshair to OHLC: On/Off
- Sync Crosshair Across Charts: On/Off

**Performance**:
- Max Candles in Memory: 10,000 (prevents lag)
- Render Optimization: Level of Detail (reduce detail when zoomed out)
- GPU Acceleration: On/Off

**Data**:
- Data Source: Live/Historical
- Preload Bars: 500/1000/2000/5000
- Auto-refresh Interval: 100ms/500ms/1s

---

## 3. ADVANCED ORDER BOOK WIDGET (20 LEVELS)

Replace the existing order book with a professional, high-information-density display showing deep market liquidity.

### Layout Structure
```
┌─────────────────────────────────────────────────────┐
│         ORDER BOOK - BTC/USDT                [⚙]    │
├─────────────────────────────────────────────────────┤
│ PRICE       SIZE (BTC)    TOTAL      ORDERS    [██] │ ← Headers
├─────────────────────────────────────────────────────┤
│                     ASK SIDE                        │
│ ████████ 96,920.00  0.4200   12.45   [18]          │ ← ASK 20
│ ███████  96,915.50  0.3800   12.07   [15]          │
│ ██████   96,910.20  0.5100   11.69   [22]          │
│ █████    96,905.80  0.2900   11.18   [11]          │
│ ████     96,901.10  0.4500   10.89   [19]          │
│    ...   (15 more ASK levels)                      │
├─────────────────────────────────────────────────────┤
│      SPREAD: $0.50 (0.0005%) | MID: $96,909.90     │ ← Midpoint
├─────────────────────────────────────────────────────┤
│                     BID SIDE                        │
│ 96,909.40  0.6200   10.44   [24]   ██████          │ ← BID 1
│ 96,905.30  0.4500    9.82   [19]   ███████         │
│ 96,900.00  0.3100    9.37   [12]   ████████        │
│ 96,895.70  0.5800    9.06   [26]   █████           │
│ 96,891.20  0.3700    8.48   [14]   ████            │
│    ...   (15 more BID levels)                      │
└─────────────────────────────────────────────────────┘
```

### Display Columns

#### **PRICE**
- Formatted with appropriate decimals (e.g., 96,909.40)
- Color: Red for asks, Green for bids
- Font: Monospace for alignment
- Best bid/ask: Highlighted or bold

#### **SIZE (BTC)**
- Individual order size at this price level
- Aggregated if multiple orders at same price
- Format: Up to 4 decimals (0.4200)
- Large orders (>threshold): Bold or highlighted

#### **TOTAL (Cumulative)**
- Running sum from best bid/ask outward
- Shows total liquidity at this level and better
- Helps identify depth

#### **ORDERS**
- Number of individual orders at this price level
- Format: [18] in brackets
- Many small orders vs one large = different market structure

#### **Depth Visualization Bars**
- Horizontal bars extending from center
- Width proportional to SIZE at that level
- Gradient fill: Darker = more liquidity
- Max bar width = level with most size
- Semi-transparent to not obscure text

### Visual Features

#### **Color Coding**
- **Ask Side**: Red gradient
  - Base: #FF4444
  - Darker red for larger sizes
  - Transparency: 30-60% based on relative size
  
- **Bid Side**: Green gradient
  - Base: #00FF88
  - Darker green for larger sizes
  - Transparency: 30-60% based on relative size

- **Spread Area**: Gray or neutral (#444444)

#### **Live Update Animations**
- **Flash Effect**: When order size changes
  - Brief bright overlay (200ms duration)
  - Fade out smoothly
  
- **Price Level Shifts**: When best bid/ask moves
  - Smooth scroll/shift animation (100ms)
  - All levels move up/down together
  
- **New Orders**: Fade-in effect (150ms)
- **Removed Orders**: Fade-out effect (150ms)

#### **Special Order Highlighting**

**Whale Orders** (Size > 10 BTC or configurable threshold):
- 🐋 Emoji or icon next to size
- Bold text
- Brighter color
- Optional sound alert

**Iceberg Orders** (Suspected):
- Identified by: Many small orders appearing at same price repeatedly
- "+" symbol indicating hidden size
- Different text style (italic or underlined)

**Order Imbalance**:
- If bid total >> ask total (or reverse): Background tint
- Ratio indicator: "Bid:Ask = 2.3:1"

### Aggregation & Grouping

**Group By Price Levels** (Settings dropdown):
- 0.01 (High precision, noisy)
- 0.10 (Default for BTC/USDT)
- 1.00 (Simplified view)
- 10.00 (Very simplified, for high prices)
- Custom input

When grouped, sum all orders within that price band.

### Interaction Features

#### **Click Actions**
- **Left Click on Price**: Pre-fill order entry with this price
- **Right Click on Level**: Context menu
  - Copy Price
  - Set Alert at Price
  - Place Limit Order Here
  - View Order Details (if single order)

#### **Hover Tooltips**
When hovering over a price level, show detailed popup:
```
┌─────────────────────────────┐
│ Price: $96,905.30           │
│ Total Size: 0.4500 BTC      │
│ # Orders: 19                │
│ Avg Order Size: 0.0237 BTC  │
│ Cumulative: 9.82 BTC        │
│ % of Total Depth: 8.3%      │
└─────────────────────────────┘
```

#### **Scrolling**
- Mouse wheel to scroll through deeper levels (beyond 20)
- Show indicator: "Viewing levels 1-20 of 150" 
- Quick jump buttons: "Top", "Best Bid/Ask", "Bottom"

### Order Book Settings (⚙ Icon)

**Display Options**:
- Number of Levels: 5 / 10 / 20 / 50 / All
- Show Cumulative Column: On/Off
- Show Order Count: On/Off
- Show Depth Bars: On/Off

**Aggregation**:
- Group By: 0.01 / 0.10 / 1.00 / 10.00 / Custom
- Aggregate Type: Sum / Average

**Alerts**:
- Whale Alert Threshold: [10.00] BTC
- Enable Sound Alerts: On/Off
- Enable Visual Flash: On/Off

**Colors**:
- Bid Color: [Green Picker]
- Ask Color: [Red Picker]
- Depth Bar Opacity: [Slider 30-90%]

### Performance Considerations
- **Update Rate**: Max 100ms (10 updates/sec) to prevent flicker
- **Buffering**: Batch small updates, apply once per frame
- **Culling**: Only render visible rows (virtualized list)
- **Precision**: Round sizes to 4 decimals to reduce noise

---

## 4. MULTI-CHART SYSTEM

Enable users to view and trade multiple cryptocurrency pairs simultaneously with synchronized or independent chart configurations.

### Chart Window Management

#### **Chart Instance Architecture**
Each chart is a self-contained instance with:
- Independent symbol & exchange
- Independent timeframe
- Independent indicator set
- Independent drawings & annotations
- Independent zoom/pan state
- Independent color scheme (optional)
- Shared or independent crosshair (toggle)

#### **Chart Organization Methods**

**1. Tab-Based System** (Browser-style)
```
┌───────────────────────────────────────────────────────────┐
│ [BTC/USDT 15m] [ETH/USDT 5m] [SOL/USDT 1h] [+]     [×]   │ ← Tab Bar
├───────────────────────────────────────────────────────────┤
│                                                           │
│         [Currently Active Chart Displayed Here]           │
│                                                           │
│                   (Full chart area)                       │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Tab Features**:
- Active tab: Highlighted with accent color
- Inactive tabs: Dimmed
- Close button (×) on each tab (except if last one)
- Add button (+) to create new chart
- Reorderable: Drag-and-drop tabs
- Middle-click to close
- Ctrl+Tab / Ctrl+Shift+Tab to cycle
- Right-click tab → Context menu:
  - Rename Tab
  - Duplicate Chart
  - Close Tab
  - Close Other Tabs
  - Close Tabs to Right

**2. Grid Layout System**

**Preset Grids**:
- **1×1**: Single chart (full size)
- **1×2**: Two charts side-by-side (vertical split)
- **2×1**: Two charts stacked (horizontal split)
- **2×2**: Four charts in grid
- **2×3**: Six charts grid
- **3×3**: Nine charts grid
- **Custom**: User-defined layout

**Grid Example (2×2)**:
```
┌─────────────────────┬─────────────────────┐
│  BTC/USDT 15m       │  ETH/USDT 5m        │
│  [Candlestick]      │  [Candlestick]      │
│  [Indicators]       │  [Indicators]       │
│                     │                     │
├─────────────────────┼─────────────────────┤
│  SOL/USDT 1h        │  DOGE/USDT 15m      │
│  [Candlestick]      │  [Candlestick]      │
│  [Indicators]       │  [Indicators]       │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

**Grid Features**:
- **Resizable Dividers**: Drag borders between charts to resize
- **Min/Max Size**: Each chart has minimum size (300×200px)
- **Focus Indicator**: Active chart has subtle border highlight
- **Swap Charts**: Drag chart header to swap positions
- **Remove Chart**: × button in chart header
- **Add Chart**: + button in empty grid slot

**3. Floating Windows** (Multi-Monitor Support)
- Detach chart to separate OS window
- Right-click chart header → "Detach to Window"
- Independent window with full chart functionality
- Can move to second monitor
- "Attach to Main Window" to return

### Chart Creation Workflow

#### **Method 1: Menu-Driven**
1. User clicks: `Charts → New Chart Window` or presses `Ctrl+N`
2. **Symbol Selector Dialog** appears:
```
   ┌────────────────────────────────────────────┐
   │  Select Market for New Chart               │
   ├────────────────────────────────────────────┤
   │  Search: [btc________]  🔍                 │
   │                                            │
   │  Results:                                  │
   │  • BTC/USDT    $96,910  +1.25%  Binance   │
   │  • BTC/USD     $96,905  +1.20%  Coinbase  │
   │  • BTCDOM      55.2%    -0.50%  Index     │
   │                                            │
   │  Recent:                                   │
   │  • ETH/USDT    $3,420   -0.82%            │
   │  • SOL/USDT    $145.80  +5.12%            │
   │                                            │
   │         [Open Chart]      [Cancel]         │
   └────────────────────────────────────────────┘
```
3. User selects symbol and clicks "Open Chart"
4. Chart appears in:
   - New tab (if tab mode)
   - Next available grid slot (if grid mode)
   - Current active position (if replacing)

#### **Method 2: Quick-Add from Watchlist**
- Right-click symbol in watchlist → "Open in New Chart"
- Double-click symbol in watchlist → Opens chart

#### **Method 3: Keyboard Shortcut**
- `Ctrl+N` → Opens symbol selector
- `Ctrl+1-9` → Switches to chart 1-9
- `Ctrl+W` → Closes current chart
- `Ctrl+Shift+T` → Reopens last closed chart

### Chart Synchronization Features

#### **Synchronized Crosshair**
Toggle: Charts Menu → Sync Crosshair Across Charts

When enabled:
- Moving mouse on one chart shows crosshair on ALL charts
- Crosshair time coordinate syncs (same timestamp on all)
- Crosshair price shows for each chart's own price scale
- Useful for comparing correlations in real-time

#### **Synchronized Time Range**
Toggle: Charts Menu → Sync Time Range

When enabled:
- Zooming/panning on one chart applies to ALL charts
- All charts show same time period
- Price scales remain independent
- Useful for multi-asset analysis

#### **Synchronized Scale**
Toggle: Charts Menu → Sync Price Scale

When enabled:
- All charts use same Y-axis range (percentage-based)
- Useful for comparing relative performance
- Not useful if assets have vastly different prices

### Chart Header Bar

Each chart has a compact header showing:
┌─────────────────────────────────────────────────────┐
│ BTC/USDT  $96,910 (+1.25%)  15m  [📊] [⚙] [×]      │
└─────────────────────────────────────────────────────┘

**Header Elements**:
- **Symbol**: BTC/USDT (click to change)
- **Current Price**: Real-time price with color (green/red)
- **Change %**: 24h change
- **Timeframe**: Current timeframe (click to change)
- **Chart Type Icon**: Candlestick/Line/etc. (click to change)
- **Settings Icon**: Chart-specific settings
- **Close Icon**: Remove this chart

**Header Interactions**:
- Click Symbol → Opens symbol selector to switch
- Click Timeframe → Opens timeframe menu
- Drag Header → Move chart to different grid position
- Right-click Header → Context menu

### Layout Presets & Workspace Management

#### **Save Layout**
- `Window → Save Current Layout As...`
- Saves:
  - Grid structure (1×1, 2×2, etc.)
  - Each chart's symbol, timeframe, indicators
  - Window sizes and positions
  - All drawings and annotations
  - Panel visibility states
- Stored as JSON file

#### **Load Layout**
- `Window → Load Layout...`
- Quick Load: Dropdown in toolbar with recent layouts
- Layouts stored in: `~/.btquant/layouts/`

#### **Default Layouts**
- **Trading**: 2×2 grid (BTC, ETH, SOL, DOGE) + orderbook + positions

Analysis: Single large chart + multiple indicators
Monitoring: 3×3 grid with all major pairs, minimal UI
Scalping: Single chart + order book + time&sales, max screen space
Auto-Save

Current layout auto-saved every 5 minutes
On application close: "Save current layout?"
Crash recovery: Restore last auto-saved layout
Performance Optimization for Multi-Chart
Lazy Rendering: Only render visible charts (not tabs in background)
Reduced Update Rate: Background charts update at 1 FPS vs 60 FPS for active
Shared Data: Same symbol data shared across charts (no duplicate API calls)
GPU Instancing: Use GPU instancing to render multiple charts efficiently
Memory Limits: Automatically reduce history length if >6 charts open
Indicator Caching: Cache indicator calculations, reuse across charts
5. SYMBOL WATCHLIST PANELProvide quick overview and navigation for multiple trading pairs.Layout & PlacementOption 1: Sidebar (Left or Right, 200-300px wide, collapsible)
Option 2: Bottom Panel (Full width, 150-200px tall, collapsible)Watchlist Display┌─────────────────────────────────────────────────────────┐
│  WATCHLIST                            [+ Add] [⚙]       │
├──────────────────────────────────────────────────────────┤
│ Symbol       Last      Chg %    Vol (24h)    Chart      │
├──────────────────────────────────────────────────────────┤
│ BTC/USDT    96,910    +1.25%    2.5B        [────╱─]   │ ← Sparkline
│ ETH/USDT     3,420    -0.82%    980M        [─╱──╲─]   │
│ SOL/USDT    145.80    +5.12%    450M        [╱────╱]   │
│ DOT/USDT      7.85    +0.15%     85M        [──────]    │
│ DOGE/USDT   0.0832    +2.34%    120M        [╱─╱──]    │
│ MATIC/USDT   0.954    -1.20%     65M        [╲──╲─]    │
│    ...                                                  │
└──────────────────────────────────────────────────────────┘ColumnsSymbol

Trading pair name (e.g., BTC/USDT)
Bold or highlighted if currently active in main chart
Icon/logo of base asset (optional)
Last Price

Current price with appropriate decimals
Color-coded: Green if up, Red if down (vs last update)
Flash animation on update
Change %

24-hour percentage change
Color: Green (positive), Red (negative), Gray (0%)
Format: +1.25% or -0.82%
Volume (24h)

Formatted: 2.5B (billions), 980M (millions), 65K (thousands)
Optional: Switch to quote currency volume vs base currency
Mini Chart (Sparkline)

Tiny line chart (last 24h price action)
50×20 pixels approximately
Same color as Change % (green/red)
No axes or labels, just trend visualization
FeaturesSorting
Click column headers to sort:

Symbol (A-Z, Z-A)
Last Price (High-Low, Low-High)
Change % (Top Gainers, Top Losers)
Volume (High-Low, Low-High)
Default sort: User's custom order (drag-and-drop)Search/Filter

Search box at top: Filter symbols as you type
Quick Filters:

All
Favorites (⭐)
Gainers (+)
Losers (-)
High Volume


Interactions

Single Click: Highlight row, show details
Double Click: Open chart in main area (or new tab)
Right Click: Context menu

Open in New Chart
Add to Favorites (⭐)
Set Price Alert
View on Exchange Website
Remove from Watchlist


Drag-and-Drop

Drag symbol → Drop on chart area → Switches chart to that symbol
Drag symbol → Drop on grid slot → Opens new chart
Multi-Select

Ctrl+Click to select multiple
Bulk actions: Add all to favorites, Open all in grid
Watchlist ManagementMultiple Watchlists

Tabs at top: "Main", "DeFi", "Memes", "Scalping"
Create new list: + button
Each list stored separately
Switch between lists with tabs
Add Symbol
Click [+ Add] button:
┌──────────────────────────────┐
│  Add Symbol to Watchlist     │
├──────────────────────────────┤
│  Search: [eth_____] 🔍       │
│                              │
│  • ETH/USDT    Binance       │
│  • ETH/USD     Coinbase      │
│  • ETHBTC      Binance       │
│                              │
│     [Add]        [Cancel]    │
└──────────────────────────────┘Bulk Import/Export

Import: Load watchlist from JSON/CSV file
Export: Save current watchlist for backup/sharing
Format example:

json  {
    "name": "My Watchlist",
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
  }Sync Across Devices (Optional)

Save watchlists to cloud/account
Auto-sync when opening app on different machine
Watchlist Settings (⚙)
Update Frequency: 1s / 5s / 10s / Manual
Show Sparklines: On/Off
Price Decimal Places: Auto / 2 / 4 / 8
Volume Format: Base Currency / Quote Currency
Row Height: Compact / Normal / Large
Font Size: Small / Medium / Large
6. ENHANCED ORDER ENTRY PANELUpgrade the existing basic order entry to support advanced order types and rapid execution.Layout┌──────────────────────────────────────────────────────┐
│  ORDER ENTRY - BTC/USDT          [Quick Trade: OFF]  │
├──────────────────────────────────────────────────────┤
│  ┌──────────────┬──────────────┐                     │
│  │   BUY        │   SELL       │ ← Side Tabs         │
│  └──────────────┴──────────────┘                     │
│                                                      │
│  Order Type: [Limit ▼]                              │
│                                                      │
│  Price:      [96,910.00______]  (Last: 96,910.50)  │
│                                                      │
│  Quantity:   [0.500__________]  BTC                 │
│              [25%] [50%] [75%] [100%] ← Quick %     │
│              ≈ $48,455.00 (Notional)                │
│                                                      │
│  ☐ Post-Only    ☐ Reduce-Only    ☐ Iceberg         │
│                                                      │
│  [ Place BUY Order ]  ← Big Green Button            │
│                                                      │
│  Available Balance: $85,000.00                      │
│  Margin Used: $12,455.00 (14.6%)                    │
│  Est. Fees: $24.23                                  │
└──────────────────────────────────────────────────────┘Order Type SelectorDropdown with:

Market: Execute immediately at best available price
Limit: Execute only at specified price or better
Stop Market: Trigger market order when price hits stop
Stop Limit: Trigger limit order when price hits stop
Trailing Stop: Stop that trails price by fixed amount or %
TWAP (Time-Weighted Average Price): Split order over time
Iceberg: Show only small portion, hide full size
Post-Only: Only maker orders, cancel if would take
Conditional Fields Based on Type:

Limit: Price field required
Stop Market/Limit: Stop Price + Limit Price (if stop-limit)
Trailing Stop: Trail Amount ($ or %), Trail Type
TWAP: Duration (minutes), Slice Size
Iceberg: Visible Quantity, Hidden Total
Quantity InputMain Input: Text field for precise entry
Quick Percentage Buttons: 25% / 50% / 75% / 100% of available balance
Notional Display: Shows USD equivalent below quantity
Keyboard Shortcuts:

Up/Down arrows: Increment/decrement by 0.01
Shift+Up/Down: By 0.1
Ctrl+Up/Down: By 1.0
Side SelectionTwo Large Tabs:

BUY Tab: Green background when active
SELL Tab: Red background when active
Clicking switches side, updates button color
Advanced Options (Checkboxes)Post-Only:

Order only added to book (maker)
Canceled if would execute immediately (taker)
Ensures maker fee rebates
Reduce-Only:

Can only reduce existing position
Cannot open new position or increase size
Useful for closing positions safely
Iceberg:

Shows small visible quantity on order book
Full size hidden, replenished as filled
Prevents market impact for large orders
Time-In-Force (TIF) (Dropdown):

GTC (Good-Till-Cancel): Remains until filled or canceled
IOC (Immediate-Or-Cancel): Fill immediately, cancel rest
FOK (Fill-Or-Kill): Fill entire order immediately or cancel all
Order ButtonLarge, Prominent Button:

BUY side: Bright green (#00FF88), "PLACE BUY ORDER"
SELL side: Bright red (#FF4444), "PLACE SELL ORDER"
On click: Submits order, shows confirmation toast
Keyboard shortcut: Ctrl+Enter (configurable to just Enter)
Pre-Trade InformationAbove Button:

Available Balance: Cash available for trading
Margin Used: Current margin utilization
Est. Fees: Calculated from order size × fee rate
Max Position Size: Based on risk limits
Risk Warnings (if applicable):

"This order will use 85% of available margin" (yellow)
"This order exceeds max position size" (red, disable button)
"Insufficient balance" (red, disable button)
Quick Trade Mode (Toggle)When Enabled:

Removes confirmation dialogs
One-click order execution
Hotkeys: B for buy, S for sell at current best bid/ask
Useful for scalping, dangerous for beginners
Prominent indicator when active (e.g., red border)
Order Confirmation Dialog (When Quick Trade OFF)┌─────────────────────────────────────────┐
│  Confirm Order                          │
├─────────────────────────────────────────┤
│  Side:         BUY                      │
│  Symbol:       BTC/USDT                 │
│  Type:         Limit                    │
│  Price:        $96,910.00               │
│  Quantity:     0.500 BTC                │
│  Notional:     $48,455.00               │
│  Est. Fee:     $24.23 (0.05%)           │
│                                         │
│  ⚠ This order uses 45% of your margin  │
│                                         │
│     [Confirm]         [Cancel]          │
└─────────────────────────────────────────┘7. ACTIVE ORDERS PANELDisplay and manage all open orders with full control.Layout┌────────────────────────────────────────────────────────────────────────────┐
│  ACTIVE ORDERS (3)                              [Cancel All] [Refresh]     │
├────────────────────────────────────────────────────────────────────────────┤
│ ID      Time     Symbol    Side  Type   Price      Filled / Total  Status │
├────────────────────────────────────────────────────────────────────────────┤
│ 12345  14:32:15 BTC/USDT  BUY   Limit  96,900.00  0.000 / 0.500   Open   │ [Edit] [Cancel]
│ 12344  14:28:42 ETH/USDT  SELL  Limit   3,425.00  0.200 / 1.000   Partial │ [Edit] [Cancel]
│ 12343  14:15:10 SOL/USDT  BUY   Stop  145.50 (T) 0.000 / 2.000   Pending │ [Edit] [Cancel]
└────────────────────────────────────────────────────────────────────────────┘ColumnsOrder ID: Unique identifier (click to copy)
Time: When order was placed
Symbol: Trading pair
Side: BUY (green) / SELL (red)
Type: Market, Limit, Stop, etc.
Price: Order price (or "Market", or stop trigger price)
Filled / Total: Quantity filled so far / Total quantity
Status:

Open (not filled)
Partial (partially filled)
Pending (stop not triggered yet)
Filled (fully filled, shown briefly then moved to history)
Canceled (shown briefly)
Rejected (shown in red, with error tooltip)
ActionsPer-Order Buttons:

[Edit]: Opens edit dialog

Change price (for limit orders)
Change quantity (reduce only)
Change stop trigger


[Cancel]: Cancel this order

Shows confirmation for large orders
No confirmation if Quick Trade mode


Bulk Actions:

[Cancel All]: Cancel all open orders

Confirmation dialog: "Cancel all 3 orders?"
Option to filter: "Cancel all BUY orders only"


[Refresh]: Manually refresh order status (usually auto-updates)Filters & SortingFilter Bar:

All Orders / Buy Only / Sell Only
By Symbol: Dropdown to filter by specific pair
By Type: Limit / Stop / All
Sort: Click column headers to sortOrder Update NotificationsVisual Indicators:

Order filled (partial or complete): Brief green flash on row
Order canceled: Brief red flash, then fade out
Order rejected: Red background, stays visible with error icon
Sound Alerts (optional, configurable):

Order filled: Success beep
Order canceled: Warning beep
Order rejected: Error sound
8. POSITIONS & P&L PANELEnhanced position tracking with detailed metrics and risk management.Layout┌─────────────────────────────────────────────────────────────────────────────────┐
│  POSITIONS & P&L                                            [Close All Positions]│
├─────────────────────────────────────────────────────────────────────────────────┤
│ Symbol     Side  Entry      Current     Size   Unrealized P&L  ROE %   Liq.    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ BTC/USDT   LONG  96,500.00  96,910.00   0.500  +$205.00        +4.2%  $92,300 │ [Close]
│ ETH/USDT   SHORT  3,450.00   3,420.00   1.000  +$30.00         +0.9%   $3,580  │ [Close]
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  ACCOUNT SUMMARY                                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Total Equity:        $105,235.00    (+$235.00 / +0.22% today)                 │
│  Available Balance:    $85,000.00                                              │
│  Margin Used:          $12,455.00    (11.8% of equity)                         │
│  Unrealized P&L:          +$235.00                                             │
│                                                                                 │
│  Daily P&L:            +$1,245.00    (+1.18%)                                  │
│  Weekly P&L:           +$3,890.00    (+3.70%)                                  │
│  Monthly P&L:         +$12,450.00   (+11.84%)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘Position Table ColumnsSymbol: Trading pair
Side: LONG (green text) / SHORT (red text)
Entry Price: Average entry price if multiple fills
Current Price: Live market price
Size: Position size in base currency
Unrealized P&L:

$ amount and color (green/red)
Calculated: (Current - Entry) × Size × (1 if long, -1 if short)
ROE %: Return on Equity as percentage
(Unrealized P&L / Margin Used) × 100
Liquidation Price: Price at which position will be force-closed
Duration: How long position has been open (hover to see)
Position Actions[Close] Button per position:

Immediately closes position at market
Shows confirmation dialog (unless Quick Trade mode)
[Close All Positions] (top right):

Closes ALL open positions
Requires confirmation
Shows total P&L that will be realized
Right-click on Position → Context menu:

Add Stop Loss
Add Take Profit
Adjust Position Size
View Position History
Close Position
Account Summary MetricsTotal Equity:

Cash + Unrealized P&L
Today's change in $ and %
Available Balance: Cash not used as marginMargin Used:

Amount locked for open positions
Percentage of total equity
Warning color if >80%
Unrealized P&L: Sum of all position P&LsPeriod P&L:

Daily: Today's realized + unrealized change
Weekly: Last 7 days
Monthly: Last 30 days
Color-coded: Green (profit), Red (loss)
Equity Curve Chart (Optional, Below Summary)Small line chart showing equity over time:

X-axis: Time (1h, 1d, 1w, 1m)
Y-axis: Equity value
Line: Equity curve
Shaded area: Drawdown from peak (in red)
Shows visual performance trend
Risk Metrics (Expandable Section)Win Rate: % of profitable trades
Profit Factor: Gross profit / Gross loss
Average Win: Avg $ per winning trade
Average Loss: Avg $ per losing trade
Max Drawdown: Largest peak-to-trough decline
Sharpe Ratio: Risk-adjusted return metric9. TIME & SALES TAPEReal-time trade feed showing market microstructure.Layout┌──────────────────────────────────────────────────────┐
│  TIME & SALES - BTC/USDT               [⚙] [||]     │
├──────────────────────────────────────────────────────┤
│  Time       Price       Size     Side   Agr   Delta │
├──────────────────────────────────────────────────────┤
│ 14:32:45   96,910.50   0.1250   BUY    ▲    +0.125 │ ← Most recent
│ 14:32:44   96,910.00   0.0850   SELL   ▼    +0.040 │
│ 14:32:44   96,909.80   0.2100   BUY    ▲    +0.250 │
│ 14:32:43   96,909.50   0.0400   SELL   ▼    +0.210 │
│ 14:32:42   96,910.00   1.5000   BUY    ▲    +1.710 │ ← Large trade (bold)
│ 14:32:41   96,909.00   0.0950   SELL   ▼    +0.210 │
│    ...                                               │
│ 14:32:35   96,908.50   0.1100   BUY    ▲    +0.305 │
└──────────────────────────────────────────────────────┘

Cumulative Delta: +53.45 BTC (Last 5min)ColumnsTime: HH:MM:SS
Price: Trade execution price

Color: Green if > previous, Red if < previous
Size: Trade size
Side:
BUY (green): Aggressive buy (market buy hit ask)
SELL (red): Aggressive sell (market sell hit bid)
Aggressive Indicator:
▲: Taker was buyer (bullish aggression)
▼: Taker was seller (bearish aggression)
Delta:
Running cumulative: (Buy Volume - Sell Volume)
Resets every configurable period (1min, 5min, etc.)
FeaturesAuto-Scroll:

On by default, new trades appear at top
[||] Pause button to freeze scroll for analysis
Large Trade Highlighting:

Trades > threshold (e.g., 1 BTC): Bold text, brighter color
Optional sound alert
Configurable threshold in settings
Cumulative Delta Indicator:

Shows below tape
Positive (green): More buying pressure
Negative (red): More selling pressure
Helps identify order flow imbalance
Trade Clustering:

Multiple trades in <1s grouped visually
Shows total size: "3 trades, 1.45 BTC total"
Settings (⚙)Display:

Show Last N Trades: 50 / 100 / 200
Large Trade Threshold: [1.0] BTC
Highlight Large Trades: On/Off
Play Sound on Large Trade: On/Off
Delta:

Delta Period: 1min / 5min / 15min / Session
Show Delta Column: On/Off
Show Cumulative Indicator: On/Off
Filtering:

Min Trade Size: Only show trades > [0.01] BTC
Hide Duplicate Prices: Group same-price trades
10. SYSTEM LOGS PANELDeveloper-friendly logging for debugging and monitoring.Layout (Collapsible Bottom Panel)┌──────────────────────────────────────────────────────────────────────────┐
│  SYSTEM LOGS                    [Clear] [Export] [▲ Collapse]            │
├──────────────────────────────────────────────────────────────────────────┤
│  Level: [All ▼]  Component: [All ▼]  Search: [____]  ☐ Auto-scroll     │
├──────────────────────────────────────────────────────────────────────────┤
│ [14:32:45] [INFO]  [MARKET_DATA] BTC/USDT: Received order book update   │
│ [14:32:44] [WARN]  [ORDER] Order 12345 partially filled: 0.2/1.0 BTC    │
│ [14:32:40] [INFO]  [STRATEGY] Grid_Scalper: RUNNING, P&L: +$1250.21     │
│ [14:32:35] [ERROR] [WEBSOCKET] Connection lost, reconnecting... (3/5)   │
│ [14:32:30] [INFO]  [MARKET_DATA] SOL/USDT: Subscribed to trades stream  │
│    ...                                                                   │
└──────────────────────────────────────────────────────────────────────────┘Log Entry Format[Timestamp] [Level] [Component] MessageLevels (with colors):

INFO (Gray): Normal operations
WARN (Yellow): Warnings, not critical
ERROR (Red): Errors requiring attention
DEBUG (Cyan): Detailed debug info (only if debug mode)
Components:

MARKET_DATA: Data feed events
ORDER: Order placement, fills, cancellations
STRATEGY: Strategy execution events
SYSTEM: Application startup, shutdown
WEBSOCKET: Connection status
DATABASE: Data storage operations
UI: User interface events
ControlsLevel Filter: Dropdown to show only specific levels
Component Filter: Dropdown to show only specific components
Search: Text search in log messages
Auto-scroll: Toggle to auto-scroll to bottom as new logs arrive
[Clear]: Clear all logs from display
[Export]: Save logs to file (TXT or JSON)Log Persistence
Logs saved to: ~/.btquant/logs/btquant_YYYYMMDD.log
Daily rotation: New file each day
Keep last 30 days of logs
Searchable from UI: "Load Historical Logs"
11. GLOBAL HOTKEY SYSTEMComprehensive keyboard shortcuts for power users.Navigation & Windows

Ctrl+M → Open Market Search
Ctrl+N → New Chart Window
Ctrl+W → Close Active Chart
Ctrl+Tab → Next Chart
Ctrl+Shift+Tab → Previous Chart
Ctrl+1-9 → Switch to Chart 1-9
F11 → Toggle Fullscreen
Alt+1-9 → Switch to Workspace 1-9
Trading

B → Quick Buy (opens order entry, focused on price)
S → Quick Sell (opens order entry, focused on price)
Ctrl+Enter → Confirm & Place Order
Esc → Cancel Order Entry / Close Dialog
Ctrl+Shift+X → Close All Positions (with confirmation)
Ctrl+Shift+C → Cancel All Orders (with confirmation)
Chart Operations

Space → Pause/Resume Live Data
+ / - → Zoom In/Out
Arrow Keys → Pan Chart
Home → Jump to Latest Candle
End → Jump to Oldest Candle
L → Toggle Line Drawing Tool
H → Toggle Horizontal Line Tool
T → Toggle Text Annotation Tool
Delete → Remove Selected Drawing
Ctrl+Z → Undo Last Drawing
Ctrl+Y → Redo Drawing
Timeframes (Quick Switch)

1 → 1-minute
2 → 5-minute
3 → 15-minute
4 → 1-hour
5 → 4-hour
6 → 1-day
Panels

Ctrl+Shift+O → Toggle Order Book
Ctrl+Shift+P → Toggle Positions Panel
Ctrl+Shift+L → Toggle System Logs
Ctrl+Shift+W → Toggle Watchlist
12. ALERT SYSTEMPrice and indicator-based alerts with notifications.Alert TypesPrice Alerts:

Price crosses above/below specific level
E.g., "Alert me when BTC > $100,000"
Indicator Alerts:

RSI > 70 or < 30
MACD crosses signal line
Price crosses moving average
Order Alerts:

Order filled (partial or complete)
Order rejected
Position liquidation warning (near liquidation price)
Alert CreationMethod 1: Right-click on Chart

Right-click on price level → "Create Alert Here"
Opens alert dialog with price pre-filled
Method 2: Alert Manager

Tools Menu → Alert Manager → [+ New Alert]
Alert Dialog:
book + positions

Analysis: Single large chart + multiple indicators
Monitoring: 3×3 grid with all major pairs, minimal UI
Scalping: Single chart + order book + time&sales, max screen space

Auto-Save

Current layout auto-saved every 5 minutes
On application close: "Save current layout?"
Crash recovery: Restore last auto-saved layout

Performance Optimization for Multi-Chart

Lazy Rendering: Only render visible charts (not tabs in background)
Reduced Update Rate: Background charts update at 1 FPS vs 60 FPS for active
Shared Data: Same symbol data shared across charts (no duplicate API calls)
GPU Instancing: Use GPU instancing to render multiple charts efficiently
Memory Limits: Automatically reduce history length if >6 charts open
Indicator Caching: Cache indicator calculations, reuse across charts


5. SYMBOL WATCHLIST PANEL
Provide quick overview and navigation for multiple trading pairs.
Layout & Placement
Option 1: Sidebar (Left or Right, 200-300px wide, collapsible)
Option 2: Bottom Panel (Full width, 150-200px tall, collapsible)
Watchlist Display
┌─────────────────────────────────────────────────────────┐
│  WATCHLIST                            [+ Add] [⚙]       │
├──────────────────────────────────────────────────────────┤
│ Symbol       Last      Chg %    Vol (24h)    Chart      │
├──────────────────────────────────────────────────────────┤
│ BTC/USDT    96,910    +1.25%    2.5B        [────╱─]   │ ← Sparkline
│ ETH/USDT     3,420    -0.82%    980M        [─╱──╲─]   │
│ SOL/USDT    145.80    +5.12%    450M        [╱────╱]   │
│ DOT/USDT      7.85    +0.15%     85M        [──────]    │
│ DOGE/USDT   0.0832    +2.34%    120M        [╱─╱──]    │
│ MATIC/USDT   0.954    -1.20%     65M        [╲──╲─]    │
│    ...                                                  │
└──────────────────────────────────────────────────────────┘
Columns
Symbol

Trading pair name (e.g., BTC/USDT)
Bold or highlighted if currently active in main chart
Icon/logo of base asset (optional)

Last Price

Current price with appropriate decimals
Color-coded: Green if up, Red if down (vs last update)
Flash animation on update

Change %

24-hour percentage change
Color: Green (positive), Red (negative), Gray (0%)
Format: +1.25% or -0.82%

Volume (24h)

Formatted: 2.5B (billions), 980M (millions), 65K (thousands)
Optional: Switch to quote currency volume vs base currency

Mini Chart (Sparkline)

Tiny line chart (last 24h price action)
50×20 pixels approximately
Same color as Change % (green/red)
No axes or labels, just trend visualization

Features
Sorting
Click column headers to sort:

Symbol (A-Z, Z-A)
Last Price (High-Low, Low-High)
Change % (Top Gainers, Top Losers)
Volume (High-Low, Low-High)

Default sort: User's custom order (drag-and-drop)
Search/Filter

Search box at top: Filter symbols as you type
Quick Filters:

All
Favorites (⭐)
Gainers (+)
Losers (-)
High Volume



Interactions

Single Click: Highlight row, show details
Double Click: Open chart in main area (or new tab)
Right Click: Context menu

Open in New Chart
Add to Favorites (⭐)
Set Price Alert
View on Exchange Website
Remove from Watchlist



Drag-and-Drop

Drag symbol → Drop on chart area → Switches chart to that symbol
Drag symbol → Drop on grid slot → Opens new chart

Multi-Select

Ctrl+Click to select multiple
Bulk actions: Add all to favorites, Open all in grid

Watchlist Management
Multiple Watchlists

Tabs at top: "Main", "DeFi", "Memes", "Scalping"
Create new list: + button
Each list stored separately
Switch between lists with tabs

Add Symbol
Click [+ Add] button:
┌──────────────────────────────┐
│  Add Symbol to Watchlist     │
├──────────────────────────────┤
│  Search: [eth_____] 🔍       │
│                              │
│  • ETH/USDT    Binance       │
│  • ETH/USD     Coinbase      │
│  • ETHBTC      Binance       │
│                              │
│     [Add]        [Cancel]    │
└──────────────────────────────┘
Bulk Import/Export

Import: Load watchlist from JSON/CSV file
Export: Save current watchlist for backup/sharing
Format example:

json  {
    "name": "My Watchlist",
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
  }
```

#### **Sync Across Devices** (Optional)
- Save watchlists to cloud/account
- Auto-sync when opening app on different machine

### Watchlist Settings (⚙)

- **Update Frequency**: 1s / 5s / 10s / Manual
- **Show Sparklines**: On/Off
- **Price Decimal Places**: Auto / 2 / 4 / 8
- **Volume Format**: Base Currency / Quote Currency
- **Row Height**: Compact / Normal / Large
- **Font Size**: Small / Medium / Large

---

## 6. ENHANCED ORDER ENTRY PANEL

Upgrade the existing basic order entry to support advanced order types and rapid execution.

### Layout
```
┌──────────────────────────────────────────────────────┐
│  ORDER ENTRY - BTC/USDT          [Quick Trade: OFF]  │
├──────────────────────────────────────────────────────┤
│  ┌──────────────┬──────────────┐                     │
│  │   BUY        │   SELL       │ ← Side Tabs         │
│  └──────────────┴──────────────┘                     │
│                                                      │
│  Order Type: [Limit ▼]                              │
│                                                      │
│  Price:      [96,910.00______]  (Last: 96,910.50)  │
│                                                      │
│  Quantity:   [0.500__________]  BTC                 │
│              [25%] [50%] [75%] [100%] ← Quick %     │
│              ≈ $48,455.00 (Notional)                │
│                                                      │
│  ☐ Post-Only    ☐ Reduce-Only    ☐ Iceberg         │
│                                                      │
│  [ Place BUY Order ]  ← Big Green Button            │
│                                                      │
│  Available Balance: $85,000.00                      │
│  Margin Used: $12,455.00 (14.6%)                    │
│  Est. Fees: $24.23                                  │
└──────────────────────────────────────────────────────┘
```

### Order Type Selector

**Dropdown with:**
- **Market**: Execute immediately at best available price
- **Limit**: Execute only at specified price or better
- **Stop Market**: Trigger market order when price hits stop
- **Stop Limit**: Trigger limit order when price hits stop
- **Trailing Stop**: Stop that trails price by fixed amount or %
- **TWAP** (Time-Weighted Average Price): Split order over time
- **Iceberg**: Show only small portion, hide full size
- **Post-Only**: Only maker orders, cancel if would take

**Conditional Fields Based on Type:**
- **Limit**: Price field required
- **Stop Market/Limit**: Stop Price + Limit Price (if stop-limit)
- **Trailing Stop**: Trail Amount ($ or %), Trail Type
- **TWAP**: Duration (minutes), Slice Size
- **Iceberg**: Visible Quantity, Hidden Total

### Quantity Input

**Main Input**: Text field for precise entry
**Quick Percentage Buttons**: 25% / 50% / 75% / 100% of available balance
**Notional Display**: Shows USD equivalent below quantity
**Keyboard Shortcuts**: 
- Up/Down arrows: Increment/decrement by 0.01
- Shift+Up/Down: By 0.1
- Ctrl+Up/Down: By 1.0

### Side Selection

**Two Large Tabs**:
- **BUY Tab**: Green background when active
- **SELL Tab**: Red background when active
- Clicking switches side, updates button color

### Advanced Options (Checkboxes)

**Post-Only**:
- Order only added to book (maker)
- Canceled if would execute immediately (taker)
- Ensures maker fee rebates

**Reduce-Only**:
- Can only reduce existing position
- Cannot open new position or increase size
- Useful for closing positions safely

**Iceberg**:
- Shows small visible quantity on order book
- Full size hidden, replenished as filled
- Prevents market impact for large orders

**Time-In-Force (TIF)** (Dropdown):
- GTC (Good-Till-Cancel): Remains until filled or canceled
- IOC (Immediate-Or-Cancel): Fill immediately, cancel rest
- FOK (Fill-Or-Kill): Fill entire order immediately or cancel all

### Order Button

**Large, Prominent Button**:
- **BUY side**: Bright green (#00FF88), "PLACE BUY ORDER"
- **SELL side**: Bright red (#FF4444), "PLACE SELL ORDER"
- On click: Submits order, shows confirmation toast
- Keyboard shortcut: Ctrl+Enter (configurable to just Enter)

### Pre-Trade Information

**Above Button**:
- Available Balance: Cash available for trading
- Margin Used: Current margin utilization
- Est. Fees: Calculated from order size × fee rate
- Max Position Size: Based on risk limits

**Risk Warnings** (if applicable):
- "This order will use 85% of available margin" (yellow)
- "This order exceeds max position size" (red, disable button)
- "Insufficient balance" (red, disable button)

### Quick Trade Mode (Toggle)

**When Enabled**:
- Removes confirmation dialogs
- One-click order execution
- Hotkeys: `B` for buy, `S` for sell at current best bid/ask
- Useful for scalping, dangerous for beginners
- Prominent indicator when active (e.g., red border)

### Order Confirmation Dialog (When Quick Trade OFF)
```
┌─────────────────────────────────────────┐
│  Confirm Order                          │
├─────────────────────────────────────────┤
│  Side:         BUY                      │
│  Symbol:       BTC/USDT                 │
│  Type:         Limit                    │
│  Price:        $96,910.00               │
│  Quantity:     0.500 BTC                │
│  Notional:     $48,455.00               │
│  Est. Fee:     $24.23 (0.05%)           │
│                                         │
│  ⚠ This order uses 45% of your margin  │
│                                         │
│     [Confirm]         [Cancel]          │
└─────────────────────────────────────────┘
```

---

## 7. ACTIVE ORDERS PANEL

Display and manage all open orders with full control.

### Layout
```
┌────────────────────────────────────────────────────────────────────────────┐
│  ACTIVE ORDERS (3)                              [Cancel All] [Refresh]     │
├────────────────────────────────────────────────────────────────────────────┤
│ ID      Time     Symbol    Side  Type   Price      Filled / Total  Status │
├────────────────────────────────────────────────────────────────────────────┤
│ 12345  14:32:15 BTC/USDT  BUY   Limit  96,900.00  0.000 / 0.500   Open   │ [Edit] [Cancel]
│ 12344  14:28:42 ETH/USDT  SELL  Limit   3,425.00  0.200 / 1.000   Partial │ [Edit] [Cancel]
│ 12343  14:15:10 SOL/USDT  BUY   Stop  145.50 (T) 0.000 / 2.000   Pending │ [Edit] [Cancel]
└────────────────────────────────────────────────────────────────────────────┘
```

### Columns

**Order ID**: Unique identifier (click to copy)
**Time**: When order was placed
**Symbol**: Trading pair
**Side**: BUY (green) / SELL (red)
**Type**: Market, Limit, Stop, etc.
**Price**: Order price (or "Market", or stop trigger price)
**Filled / Total**: Quantity filled so far / Total quantity
**Status**: 
- Open (not filled)
- Partial (partially filled)
- Pending (stop not triggered yet)
- Filled (fully filled, shown briefly then moved to history)
- Canceled (shown briefly)
- Rejected (shown in red, with error tooltip)

### Actions

**Per-Order Buttons**:
- **[Edit]**: Opens edit dialog
  - Change price (for limit orders)
  - Change quantity (reduce only)
  - Change stop trigger
- **[Cancel]**: Cancel this order
  - Shows confirmation for large orders
  - No confirmation if Quick Trade mode

**Bulk Actions**:
- **[Cancel All]**: Cancel all open orders
  - Confirmation dialog: "Cancel all 3 orders?"
  - Option to filter: "Cancel all BUY orders only"

**[Refresh]**: Manually refresh order status (usually auto-updates)

### Filters & Sorting

**Filter Bar**:
- All Orders / Buy Only / Sell Only
- By Symbol: Dropdown to filter by specific pair
- By Type: Limit / Stop / All

**Sort**: Click column headers to sort

### Order Update Notifications

**Visual Indicators**:
- Order filled (partial or complete): Brief green flash on row
- Order canceled: Brief red flash, then fade out
- Order rejected: Red background, stays visible with error icon

**Sound Alerts** (optional, configurable):
- Order filled: Success beep
- Order canceled: Warning beep
- Order rejected: Error sound

---

## 8. POSITIONS & P&L PANEL

Enhanced position tracking with detailed metrics and risk management.

### Layout
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  POSITIONS & P&L                                            [Close All Positions]│
├─────────────────────────────────────────────────────────────────────────────────┤
│ Symbol     Side  Entry      Current     Size   Unrealized P&L  ROE %   Liq.    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ BTC/USDT   LONG  96,500.00  96,910.00   0.500  +$205.00        +4.2%  $92,300 │ [Close]
│ ETH/USDT   SHORT  3,450.00   3,420.00   1.000  +$30.00         +0.9%   $3,580  │ [Close]
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  ACCOUNT SUMMARY                                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Total Equity:        $105,235.00    (+$235.00 / +0.22% today)                 │
│  Available Balance:    $85,000.00                                              │
│  Margin Used:          $12,455.00    (11.8% of equity)                         │
│  Unrealized P&L:          +$235.00                                             │
│                                                                                 │
│  Daily P&L:            +$1,245.00    (+1.18%)                                  │
│  Weekly P&L:           +$3,890.00    (+3.70%)                                  │
│  Monthly P&L:         +$12,450.00   (+11.84%)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Position Table Columns

**Symbol**: Trading pair
**Side**: LONG (green text) / SHORT (red text)
**Entry Price**: Average entry price if multiple fills
**Current Price**: Live market price
**Size**: Position size in base currency
**Unrealized P&L**: 
- $ amount and color (green/red)
- Calculated: (Current - Entry) × Size × (1 if long, -1 if short)
**ROE %**: Return on Equity as percentage
- (Unrealized P&L / Margin Used) × 100
**Liquidation Price**: Price at which position will be force-closed
**Duration**: How long position has been open (hover to see)

### Position Actions

**[Close] Button** per position:
- Immediately closes position at market
- Shows confirmation dialog (unless Quick Trade mode)

**[Close All Positions]** (top right):
- Closes ALL open positions
- Requires confirmation
- Shows total P&L that will be realized

**Right-click on Position** → Context menu:
- Add Stop Loss
- Add Take Profit
- Adjust Position Size
- View Position History
- Close Position

### Account Summary Metrics

**Total Equity**: 
- Cash + Unrealized P&L
- Today's change in $ and %

**Available Balance**: Cash not used as margin

**Margin Used**: 
- Amount locked for open positions
- Percentage of total equity
- Warning color if >80%

**Unrealized P&L**: Sum of all position P&Ls

**Period P&L**:
- Daily: Today's realized + unrealized change
- Weekly: Last 7 days
- Monthly: Last 30 days
- Color-coded: Green (profit), Red (loss)

### Equity Curve Chart (Optional, Below Summary)

Small line chart showing equity over time:
- X-axis: Time (1h, 1d, 1w, 1m)
- Y-axis: Equity value
- Line: Equity curve
- Shaded area: Drawdown from peak (in red)
- Shows visual performance trend

### Risk Metrics (Expandable Section)

**Win Rate**: % of profitable trades
**Profit Factor**: Gross profit / Gross loss
**Average Win**: Avg $ per winning trade
**Average Loss**: Avg $ per losing trade
**Max Drawdown**: Largest peak-to-trough decline
**Sharpe Ratio**: Risk-adjusted return metric

---

## 9. TIME & SALES TAPE

Real-time trade feed showing market microstructure.

### Layout
```
┌──────────────────────────────────────────────────────┐
│  TIME & SALES - BTC/USDT               [⚙] [||]     │
├──────────────────────────────────────────────────────┤
│  Time       Price       Size     Side   Agr   Delta │
├──────────────────────────────────────────────────────┤
│ 14:32:45   96,910.50   0.1250   BUY    ▲    +0.125 │ ← Most recent
│ 14:32:44   96,910.00   0.0850   SELL   ▼    +0.040 │
│ 14:32:44   96,909.80   0.2100   BUY    ▲    +0.250 │
│ 14:32:43   96,909.50   0.0400   SELL   ▼    +0.210 │
│ 14:32:42   96,910.00   1.5000   BUY    ▲    +1.710 │ ← Large trade (bold)
│ 14:32:41   96,909.00   0.0950   SELL   ▼    +0.210 │
│    ...                                               │
│ 14:32:35   96,908.50   0.1100   BUY    ▲    +0.305 │
└──────────────────────────────────────────────────────┘

Cumulative Delta: +53.45 BTC (Last 5min)
```

### Columns

**Time**: HH:MM:SS
**Price**: Trade execution price
- Color: Green if > previous, Red if < previous
**Size**: Trade size
**Side**: 
- BUY (green): Aggressive buy (market buy hit ask)
- SELL (red): Aggressive sell (market sell hit bid)
**Aggressive Indicator**: 
- ▲: Taker was buyer (bullish aggression)
- ▼: Taker was seller (bearish aggression)
**Delta**: 
- Running cumulative: (Buy Volume - Sell Volume)
- Resets every configurable period (1min, 5min, etc.)

### Features

**Auto-Scroll**: 
- On by default, new trades appear at top
- [||] Pause button to freeze scroll for analysis

**Large Trade Highlighting**:
- Trades > threshold (e.g., 1 BTC): Bold text, brighter color
- Optional sound alert
- Configurable threshold in settings

**Cumulative Delta Indicator**:
- Shows below tape
- Positive (green): More buying pressure
- Negative (red): More selling pressure
- Helps identify order flow imbalance

**Trade Clustering**:
- Multiple trades in <1s grouped visually
- Shows total size: "3 trades, 1.45 BTC total"

### Settings (⚙)

**Display**:
- Show Last N Trades: 50 / 100 / 200
- Large Trade Threshold: [1.0] BTC
- Highlight Large Trades: On/Off
- Play Sound on Large Trade: On/Off

**Delta**:
- Delta Period: 1min / 5min / 15min / Session
- Show Delta Column: On/Off
- Show Cumulative Indicator: On/Off

**Filtering**:
- Min Trade Size: Only show trades > [0.01] BTC
- Hide Duplicate Prices: Group same-price trades

---

## 10. SYSTEM LOGS PANEL

Developer-friendly logging for debugging and monitoring.

### Layout (Collapsible Bottom Panel)
```
┌──────────────────────────────────────────────────────────────────────────┐
│  SYSTEM LOGS                    [Clear] [Export] [▲ Collapse]            │
├──────────────────────────────────────────────────────────────────────────┤
│  Level: [All ▼]  Component: [All ▼]  Search: [____]  ☐ Auto-scroll     │
├──────────────────────────────────────────────────────────────────────────┤
│ [14:32:45] [INFO]  [MARKET_DATA] BTC/USDT: Received order book update   │
│ [14:32:44] [WARN]  [ORDER] Order 12345 partially filled: 0.2/1.0 BTC    │
│ [14:32:40] [INFO]  [STRATEGY] Grid_Scalper: RUNNING, P&L: +$1250.21     │
│ [14:32:35] [ERROR] [WEBSOCKET] Connection lost, reconnecting... (3/5)   │
│ [14:32:30] [INFO]  [MARKET_DATA] SOL/USDT: Subscribed to trades stream  │
│    ...                                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Log Entry Format

**[Timestamp] [Level] [Component] Message**

**Levels** (with colors):
- **INFO** (Gray): Normal operations
- **WARN** (Yellow): Warnings, not critical
- **ERROR** (Red): Errors requiring attention
- **DEBUG** (Cyan): Detailed debug info (only if debug mode)

**Components**:
- MARKET_DATA: Data feed events
- ORDER: Order placement, fills, cancellations
- STRATEGY: Strategy execution events
- SYSTEM: Application startup, shutdown
- WEBSOCKET: Connection status
- DATABASE: Data storage operations
- UI: User interface events

### Controls

**Level Filter**: Dropdown to show only specific levels
**Component Filter**: Dropdown to show only specific components
**Search**: Text search in log messages
**Auto-scroll**: Toggle to auto-scroll to bottom as new logs arrive
**[Clear]**: Clear all logs from display
**[Export]**: Save logs to file (TXT or JSON)

### Log Persistence

- Logs saved to: `~/.btquant/logs/btquant_YYYYMMDD.log`
- Daily rotation: New file each day
- Keep last 30 days of logs
- Searchable from UI: "Load Historical Logs"

---

## 11. GLOBAL HOTKEY SYSTEM

Comprehensive keyboard shortcuts for power users.

### Navigation & Windows
- `Ctrl+M` → Open Market Search
- `Ctrl+N` → New Chart Window
- `Ctrl+W` → Close Active Chart
- `Ctrl+Tab` → Next Chart
- `Ctrl+Shift+Tab` → Previous Chart
- `Ctrl+1-9` → Switch to Chart 1-9
- `F11` → Toggle Fullscreen
- `Alt+1-9` → Switch to Workspace 1-9

### Trading
- `B` → Quick Buy (opens order entry, focused on price)
- `S` → Quick Sell (opens order entry, focused on price)
- `Ctrl+Enter` → Confirm & Place Order
- `Esc` → Cancel Order Entry / Close Dialog
- `Ctrl+Shift+X` → Close All Positions (with confirmation)
- `Ctrl+Shift+C` → Cancel All Orders (with confirmation)

### Chart Operations
- `Space` → Pause/Resume Live Data
- `+` / `-` → Zoom In/Out
- `Arrow Keys` → Pan Chart
- `Home` → Jump to Latest Candle
- `End` → Jump to Oldest Candle
- `L` → Toggle Line Drawing Tool
- `H` → Toggle Horizontal Line Tool
- `T` → Toggle Text Annotation Tool
- `Delete` → Remove Selected Drawing
- `Ctrl+Z` → Undo Last Drawing
- `Ctrl+Y` → Redo Drawing

### Timeframes (Quick Switch)
- `1` → 1-minute
- `2` → 5-minute
- `3` → 15-minute
- `4` → 1-hour
- `5` → 4-hour
- `6` → 1-day

### Panels
- `Ctrl+Shift+O` → Toggle Order Book
- `Ctrl+Shift+P` → Toggle Positions Panel
- `Ctrl+Shift+L` → Toggle System Logs
- `Ctrl+Shift+W` → Toggle Watchlist

---

## 12. ALERT SYSTEM

Price and indicator-based alerts with notifications.

### Alert Types

**Price Alerts**:
- Price crosses above/below specific level
- E.g., "Alert me when BTC > $100,000"

**Indicator Alerts**:
- RSI > 70 or < 30
- MACD crosses signal line
- Price crosses moving average

**Order Alerts**:
- Order filled (partial or complete)
- Order rejected
- Position liquidation warning (near liquidation price)

### Alert Creation

**Method 1: Right-click on Chart**
- Right-click on price level → "Create Alert Here"
- Opens alert dialog with price pre-filled

**Method 2: Alert Manager**
- Tools Menu → Alert Manager → [+ New Alert]

**Alert Dialog**:
```
┌─────────────────────────────────────────┐
│  Create Price Alert                     │
├─────────────────────────────────────────┤
│  Symbol:     [BTC/USDT ▼]               │
│  Condition:  [Price Crosses Above ▼]    │
│  Value:      [100,000.00______]         │
│                                         │
│  Notification:                          │
│  ☑ Visual (Pop-up)                      │
│  ☑ Sound                                │
│  ☐ Email                                │
│  ☐ Push Notification (Mobile)           │
│                                         │
│  Once Triggered: [Delete Alert ▼]      │
│                                         │
│     [Create Alert]     [Cancel]         │
└─────────────────────────────────────────┘
```

### Alert Manager

**List View**:
```
┌──────────────────────────────────────────────────────────────┐
│  ALERT MANAGER                              [+ New] [Delete] │
├──────────────────────────────────────────────────────────────┤
│ ☑ BTC/USDT  Price > $100,000     Visual, Sound   [Edit] [×] │
│ ☑ ETH/USDT  RSI < 30             Visual          [Edit] [×] │
│ ☐ SOL/USDT  MACD Cross Signal    Email           [Edit] [×] │
└──────────────────────────────────────────────────────────────┘
```

**Features**:
- Toggle alerts on/off with checkbox (without deleting)
- Edit alert parameters
- Delete alerts
- Bulk delete / disable

### Notification Display

**Visual Alert** (Pop-up Toast):
```
┌─────────────────────────────────────┐
│  🔔 Alert Triggered!                │
│  BTC/USDT crossed above $100,000    │
│  Current Price: $100,125.00         │
│                                     │
│  [View Chart]      [Dismiss]        │
└─────────────────────────────────────┘
Sound Alert: Configurable beep/chime
Menu Bar Indicator: Bell icon (🔔) in menu bar shows active alerts count
13. MARKET SCREENER TOOL
Scan all available markets for specific technical conditions.
Access
Tools Menu → Market Screener
Interface
┌────────────────────────────────────────────────────────────────┐
│  MARKET SCREENER                             [Run Scan] [Save] │
├────────────────────────────────────────────────────────────────┤
│  Add Condition: [Price ▼]                                      │
│                                                                │
│  Active Conditions:                                            │
│  1. RSI (14) < 30                               [Edit] [Remove]│
│  2. Volume > 1M (24h)                           [Edit] [Remove]│
│  3. Change% > +5% (24h)                         [Edit] [Remove]│
│                                                                │
│  Markets to Scan: [All Binance Futures ▼]                     │
│  Timeframe: [15m ▼]                                            │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  RESULTS (5 symbols match)                                     │
├────────────────────────────────────────────────────────────────┤
│  Symbol      RSI    Volume    Change%    Action                │
│  SOL/USDT    28.5   1.2M      +6.8%     [Open Chart]          │
│MATIC/USDT  25.1   1.5M      +5.2%     [Open Chart]          │
│  ...                                                           │
└────────────────────────────────────────────────────────────────┘

### Condition Types

**Price-based**:
- Price > / < / = value
- Price change % > / < value
- Price near 52-week high/low

**Volume-based**:
- Volume > / < value
- Volume spike (> N× average)

**Technical Indicators**:
- RSI > / < value
- MACD above/below signal
- Price above/below MA
- Bollinger Band breakout

### Saved Scans

- Save condition sets as templates
- E.g., "Oversold Breakout", "High Volume Gainers"
- Quick-load from dropdown

---

## 14. VISUAL DESIGN SYSTEM

Consistent professional dark theme across all components.

### Color Palette

**Backgrounds**:
- Primary: #1E1E1E (darkest, main background)
- Secondary: #252525 (panels, cards)
- Tertiary: #2D2D2D (hover states, elevated elements)

**Accents**:
- Primary: #00A8FF (electric blue, highlights, active elements)
- Success: #00FF88 (green, buy side, positive)
- Danger: #FF4444 (red, sell side, negative)
- Warning: #FFA500 (orange/amber, warnings)

**Text**:
- Primary: #FFFFFF (high contrast, main text)
- Secondary: #AAAAAA (gray, labels, less important)
- Disabled: #666666 (dark gray, disabled elements)

**Trading Colors**:
- Buy/Bid: #00FF88 (bright green)
- Sell/Ask: #FF4444 (bright red)
- Neutral: #888888 (gray)

### Typography

**Fonts**:
- **Monospace** (JetBrains Mono or Fira Code): 
  - Numbers, prices, timestamps
  - Code, logs
- **Sans-serif** (Inter or Roboto):
  - Labels, buttons, UI text
  - Headers

**Font Sizes**:
- Headers: 18-24px
- Labels: 14-16px
- Data Tables: 11-12px
- Small Text: 10px

### UI Components

**Buttons**:
- Flat design with subtle shadows
- Hover: 10% lighter
- Active/Click: 10% darker
- Border-radius: 4px
- Padding: 8px 16px

**Input Fields**:
- Dark background (#2D2D2D)
- Light border (#444444)
- Focus: Blue border (#00A8FF)
- Border-radius: 4px
- Padding: 6px 10px

**Panels**:
- Background: #252525
- Border: 1px solid #333333
- Border-radius: 6px
- Drop-shadow: 0 2px 8px rgba(0,0,0,0.3)

**Tables**:
- Header: Bold, #FFFFFF on #2D2D2D
- Rows: Alternate backgrounds (#252525 / #282828)
- Hover: #2D2D2D
- Border: 1px solid #333333

### Spacing & Grid

**8px Grid System**:
- Base unit: 8px
- Padding/margins: 8px, 16px, 24px, 32px
- Component spacing: Multiples of 8

**Consistent Margins**:
- Between panels: 16px
- Within panels: 12px
- Element padding: 8px

### Animations

**Timing**:
- Fast (100-150ms): Hover states, tooltips
- Medium (200-300ms): Panel open/close, transitions
- Slow (400-500ms): Chart animations, large movements

**Easing**: ease-out for most transitions

### Icons

- Use consistent icon set (e.g., Font Awesome, Material Icons)
- Size: 16px for inline, 20-24px for buttons
- Color: Match text color (#FFFFFF or #AAAAAA)

---

## 15. TECHNICAL IMPLEMENTATION NOTES

### Vulkan Rendering Optimizations

**Chart Rendering**:
- Use GPU instancing for multiple candles
- Batch draw calls (one per chart, not per candle)
- Use compute shaders for indicator calculations
- Level-of-detail: Reduce candle detail when zoomed out (e.g., show only close price as line if >10,000 candles visible)

**Text Rendering**:
- Use signed distance field (SDF) fonts for crisp text at any scale
- Cache glyph atlas in GPU texture
- Batch all text rendering per frame

**Order Book Visualization**:
- Use instanced quads for depth bars
- Update only changed levels (delta updates)
- Use vertex buffer streaming for dynamic data

### Dear ImGui Integration

**Custom Widgets**:
- Create custom ImGui widgets for:
  - Candlestick chart
  - Order book depth bars
  - Sparkline mini-charts

**Styling**:
- Override ImGui default style with BTQuant theme
- Use ImGui::PushStyleColor/PopStyleColor for local overrides
- Custom fonts loaded with ImGui::GetIO().Fonts

**Layout**:
- Use ImGui::DockSpace for panel management
- Implement custom tab bar for multi-chart tabs
- Use ImGui::BeginChild/EndChild for scrollable regions

### Shared Memory Architecture

**Data Structures**:
- Ring buffers for time-series (OHLCV, ticks)
- Lock-free queues for order updates
- Atomic variables for current price, volume

**Performance**:
- Producer (data feed) writes to shared memory
- Consumer (UI) reads without blocking producer
- Double-buffering for chart data to prevent tearing

**Synchronization**:
- Use memory-mapped files for inter-process communication
- Spin-locks for very short critical sections
- Avoid mutexes in hot paths

### Threading Model

**Main Thread** (UI):
- ImGui rendering
- User input handling
- 60 FPS target

**Data Thread** (Market Data):
- WebSocket connections
- Shared memory writing
- Order book reconstruction

**Strategy Thread** (Trading Logic):
- Strategy execution
- Order generation
- Risk checks

**Network Thread** (Exchange API):
- REST API calls
- Order submission
- Account queries

### Data Management

**Historical Data**:
- SQLite database for OHLCV bars
- LZ4 compression for older data
- Auto-download missing bars on chart load

**Real-time Data**:
- Keep last N candles in memory (e.g., 5,000)
- Older data: On-demand load from database
- WebSocket reconnection with gap-filling

**Caching**:
- Cache indicator calculations
- Invalidate cache on new candle or parameter change
- LRU cache for frequently accessed data

---

## 16. DEVELOPMENT ROADMAP & PRIORITIES

### Phase 1: Foundation (Weeks 1-2)
**Critical infrastructure for all features**
1. Mac-style menu bar with all menus
2. Symbol selector dialog
3. Multi-chart tab system
4. Shared data pipeline refactor for multi-symbol support

**Goal**: Can open multiple charts for different symbols

### Phase 2: Professional Charts (Weeks 3-4)
**Replace line chart with full trading chart**
1. OHLC candlestick rendering (Vulkan)
2. Volume sub-panel
3. Timeframe selector with data loading
4. Zoom & pan interactions
5. Crosshair with info display

**Goal**: Professional-quality chart comparable to TradingView

### Phase 3: Order Book & Tape (Weeks 5-6)
**Market microstructure visualization**
1. 20-level order book with depth bars
2. Live update animations
3. Time & Sales tape with delta
4. Large trade highlighting

**Goal**: Full market depth visibility for informed trading

### Phase 4: Technical Analysis (Weeks 7-8)
**Indicators and drawing tools**
1. Overlay indicators (MA, Bollinger, VWAP)
2. Sub-panel indicators (RSI, MACD, Stochastic)
3. Drawing tools (trendlines, Fibonacci, annotations)
4. Trade markers on chart

**Goal**: Complete technical analysis toolkit

### Phase 5: Trading Enhancements (Weeks 9-10)
**Advanced order types and management**
1. Enhanced order entry (all order types)
2. Active orders panel with edit/cancel
3. Improved positions display with equity curve
4. Quick trade mode

**Goal**: Professional order execution capabilities

### Phase 6: Power Features (Weeks 11-12)
**Tools for advanced users**
1. Alert system (price, indicator, order)
2. Market screener
3. Hotkey system
4. Watchlist panel

**Goal**: Power user productivity tools

### Phase 7: Polish & Optimization (Weeks 13-14)
**Performance and UX refinement**
1. Performance profiling and optimization
2. Multi-monitor support
3. Workspace save/load
4. Help system and documentation

**Goal**: Production-ready, stable, fast

### Phase 8: Advanced Analytics (Weeks 15+)
**Optional premium features**
1. Correlation matrix
2. Volatility surface
3. Liquidation heatmap
4. Custom strategy IDE

**Goal**: Institutional-grade analytics

---

## SUMMARY OF MISSING FEATURES

Based on current dashboard, here's what needs to be built:

### Critical (Must-Have)
1. ✅ Mac-style menu bar navigation
2. ✅ Professional candlestick charts (replacing line chart)
3. ✅ Multi-chart system (tabs + grids)
4. ✅ Symbol selector/search
5. ✅ Timeframe selection
6. ✅ 20-level order book with visualization
7. ✅ Technical indicators (MA, RSI, MACD, etc.)
8. ✅ Drawing tools
9. ✅ Enhanced order entry (all order types)
10. ✅ Active orders management panel

### Important (Should-Have)
1. ✅ Watchlist panel
2. ✅ Trade markers on chart
3. ✅ Alert system
4. ✅ Hotkey system
5. ✅ Workspace save/load
6. ✅ Equity curve in P&L panel
7. ✅ Market screener
8. ✅ Time & Sales improvements (delta, large trade highlighting)

### Nice-to-Have (Could-Have)
1. ✅ Correlation matrix
2. ✅ Volatility analysis tools
3. ✅ Liquidation heatmap
4. ✅ Multi-monitor support
5. ✅ Mobile companion app
6. ✅ Cloud sync for settings
7. ✅ Social trading features
8. ✅ Advanced backtesting UI

---

## ACCEPTANCE CRITERIA

The redesigned dashboard will be considered complete when:

1. **Navigation**: User can easily switch between 20+ symbols via menu/search
2. **Visualization**: Charts show proper OHLC candles with 10+ indicators
3. **Multi-Chart**: Can view 4+ symbols simultaneously in grid layout
4. **Order Book**: Shows 20 levels with animated depth visualization
5. **Trading**: Can place all order types (Market, Limit, Stop, TWAP, Iceberg)
6. **Performance**: Maintains 60 FPS with 4 charts + indicators
7. **Stability**: Zero crashes during 24-hour continuous operation
8. **Usability**: Professional trader can execute trades without training

---

## FINAL NOTES FOR IMPLEMENTATION

**This specification provides**:
- Complete UI/UX design for all panels
- Detailed feature requirements
- Implementation priorities
- Technical architecture notes

**To begin development**:
1. Start with Phase 1 (menu bar + multi-chart foundation)
2. Use this spec as reference for each component
3. Implement features in priority order
4. Test each phase before moving to next

**For questions about**:
- Specific Vulkan rendering techniques
- Dear ImGui custom widget implementation
- Shared memory synchronization
- Any other technical details

Just ask for deeper technical implementation guidance on that specific topic.