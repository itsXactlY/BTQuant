# BTQ Render Engine - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Dashboard Overview](#dashboard-overview)
6. [Panels and Features](#panels-and-features)
7. [Advanced Features](#advanced-features)
8. [Settings and Customization](#settings-and-customization)
9. [Troubleshooting](#troubleshooting)
10. [Keyboard Shortcuts](#keyboard-shortcuts)

## Introduction

Welcome to the BTQ Render Engine, a sophisticated trading terminal designed to provide professional-grade market analysis tools. This platform offers a comprehensive suite of analytical panels including charts, footprint analysis, order book visualization, volume profiles, and much more.

The BTQ Render Engine is built with performance in mind, utilizing Vulkan graphics and advanced rendering techniques to deliver smooth, real-time market data visualization even with high-frequency data feeds.

## System Requirements

### Minimum Requirements
- Operating System: Linux (Ubuntu 20.04 or later recommended)
- CPU: Quad-core processor (Intel i5 / AMD Ryzen 5 or equivalent)
- RAM: 8 GB
- GPU: Vulkan-compatible graphics card with 2GB VRAM
- Storage: 2 GB available space

### Recommended Requirements
- Operating System: Linux (Ubuntu 22.04 or later)
- CPU: Hexa-core processor (Intel i7 / AMD Ryzen 7 or equivalent)
- RAM: 16 GB or more
- GPU: Modern discrete GPU with 4GB+ VRAM
- Storage: SSD with 2 GB available space

## Installation

### Prerequisites
Before installing the BTQ Render Engine, ensure you have the following packages installed:

```bash
sudo apt update
sudo apt install build-essential cmake vulkan-sdk libglfw3-dev libvulkan-dev libtbb-dev
```

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/your-repo/BTQ_Render_Engine.git
cd BTQ_Render_Engine
```

2. Create build directory:
```bash
mkdir build && cd build
```

3. Configure the build:
```bash
cmake ..
```

4. Compile the application:
```bash
make -j$(nproc)
```

5. Run the application:
```bash
./realtime_dashboard
```

## Getting Started

### First Launch
When you launch the BTQ Render Engine for the first time, you'll see the main dashboard with default panels. The interface follows a dark theme optimized for trading environments.

### Connecting to Data Sources
The application connects to market data through the modern data pipeline. Configure your data source connection in the Settings panel under the Data section.

### Basic Navigation
- Click and drag to pan charts
- Scroll to zoom in/out
- Right-click for context menus
- Double-click on panels to maximize/minimize

## Dashboard Overview

The main dashboard consists of several key areas:

### Main Workspace
The central area where you arrange and customize your trading panels. You can add, remove, and resize panels as needed.

### Dashboard Controls
Located typically at the top or side, this panel provides quick access to:
- Adding new panels
- Changing symbols
- Selecting exchanges
- Adjusting timeframes

### Status Bar
At the bottom of the screen, showing:
- Connection status
- Current symbol
- System performance metrics
- Timestamp

## Panels and Features

### Chart Panel
The Chart panel is the primary visualization tool for price action.

#### Features:
- Multiple chart types (candlestick, line, area)
- Over 20 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Drawing tools (trend lines, Fibonacci, rectangles, text annotations)
- Crosshair with detailed price/time information
- Replay mode for historical analysis

#### Usage:
1. Right-click on the chart to access the drawing tools menu
2. Use the indicator panel to add technical studies
3. Scroll to zoom, click-drag to pan

### Footprint Panel
The Footprint panel provides volume analysis at individual price levels across time periods.

#### Features:
- 16 different volume analysis types
- Heatmap visualization of volume concentration
- Diagonal and stacked imbalance detection
- Time and price aggregation options
- Volume filtering capabilities
- Split volume display mode

#### Volume Analysis Types:
1. Trades - Total number of trades
2. BuyTrades - Number of buy trades
3. SellTrades - Number of sell trades
4. Volume - Total volume
5. BuyVolume - Volume of buy trades
6. SellVolume - Volume of sell trades
7. BuyVolumePercent - Percentage of buy volume
8. SellVolumePercent - Percentage of sell volume
9. BuySellVolume - Buy/sell volume comparison
10. Delta - Difference between buy and sell volume
11. DeltaPercent - Delta as percentage of total volume
12. CumulativeDelta - Running total of delta
13. AverageSize - Average trade size
14. AverageBuySize - Average size of buy trades
15. AverageSellSize - Average size of sell trades
16. MaxOneTradeVolume - Largest single trade volume

### Volume Profile Panel
Displays volume distribution at different price levels.

#### Profile Modes:
- **Step Profile**: Mini histograms on each candlestick
- **Right Profile**: Histogram anchored to right side of chart
- **Left Profile**: Histogram anchored to left side of chart
- **Custom Profile**: User-defined time range selection

#### Features:
- Point of Control (POC) line
- Value Area with VAH/VAL lines
- Value Area percentage adjustment
- Multiple profile overlays
- Session and composite profiles

### Order Book (DOM) Panel
Real-time visualization of market depth and liquidity.

#### Features:
- Configurable number of levels (10, 20, 50, 100, 500, unlimited)
- Heatmap background for volume visualization
- Liquidity bars showing cumulative volume
- Large order highlighting
- Bid/ask imbalance indicators
- Depth chart visualization mode
- Order flow detection
- Historical snapshots

### Time & Sales Panel
Chronological display of executed trades.

#### Features:
- Virtualized scrolling for large datasets
- Trade filtering options
- Color-coded trades (buy/sell/large/block)
- Trade clustering detection
- Size distribution histogram
- Trade pace indicators
- Audio alerts for significant trades
- Export to CSV functionality

### Watchlist Panel
Monitor multiple symbols simultaneously.

#### Features:
- Customizable columns (price, volume, change, etc.)
- Real-time price updates
- Color-coded price changes
- Add/remove symbols
- Drag-and-drop reordering
- Multiple watchlist groups
- Sorting by any column
- Price alerts

### VWAP Panel
Visualize and analyze Volume Weighted Average Price.

#### VWAP Types:
- **Session VWAP**: Resets at session boundaries
- **Anchored VWAP**: User-defined start point
- **Rolling VWAP**: Sliding window calculation
- **Multi VWAP**: Multiple VWAPs simultaneously

#### Features:
- Standard deviation bands (SD1, SD2, SD3)
- Click-to-anchor functionality
- VWAP list management
- Alert triggers

### Time Statistics Panel
Tabular view of time-based market statistics.

#### Features:
- Sortable columns
- Synchronized scrolling with chart
- Color-coded values
- Customizable column visibility
- Hover tooltips

### Time Histogram Panel
Vertical bar visualization of time-based metrics.

#### Features:
- Multiple data type support (all 16 volume types)
- Buy/sell volume stacking
- Delta visualization
- Cumulative delta line
- Auto-scaling

## Advanced Features

### Drawing Tools
Access drawing tools through the chart panel context menu:
- Trend lines and channels
- Horizontal and vertical lines
- Fibonacci retracements and extensions
- Geometric shapes (rectangles, circles, triangles)
- Text annotations
- Gann fans and grids

### Chart Replay Mode
Analyze historical market behavior:
1. Access replay mode from the chart panel
2. Set start and end times
3. Adjust replay speed (0.5x to 10x)
4. Pause/resume functionality

### Multi-Timeframe Analysis
View different timeframes simultaneously:
- Primary chart shows main timeframe
- Indicator panels can show different timeframes
- Synchronized zoom and pan across timeframes

### Alert System
Set up notifications for market events:
- Price alerts (above/below specific levels)
- Indicator-based alerts (RSI overbought/oversold)
- VWAP cross alerts
- Volume-based alerts
- Custom condition alerts

### Layout Management
Save and load different workspace configurations:
- Quick-save layouts (F5-F8)
- Named layout presets
- Import/export layouts
- Template layouts for different trading styles

### Performance Monitoring
Built-in performance diagnostics:
- Frame rate monitoring
- Memory usage tracking
- Draw call counting
- Panel render time analysis
- Pipeline metrics

## Settings and Customization

### Appearance Settings
- Dark/Light/Custom themes
- Font selection and sizing
- Panel opacity adjustments
- Color scheme customization
- Border style options

### Data Settings
- Default timeframe and symbol
- Data retention periods
- WebSocket reconnection settings
- Update frequency configuration

### Performance Settings
- Frame rate limiter
- V-Sync toggle
- Level of Detail (LOD) thresholds
- Caching strategies
- Memory limits

### Alert Settings
- Notification methods (popup, sound, tray)
- Alert history size
- Condition templates
- Sound customization

### Keyboard Shortcuts
Customize all application shortcuts:
- View all current shortcuts
- Assign new shortcuts
- Import/export shortcut profiles
- Reset to defaults

## Troubleshooting

### Common Issues

#### Application Won't Start
- Verify Vulkan is properly installed: `vulkaninfo`
- Check GPU compatibility with Vulkan
- Ensure all dependencies are installed

#### Poor Performance
- Reduce the number of active panels
- Lower the refresh rate in settings
- Decrease the amount of historical data displayed
- Check for background processes consuming resources

#### Data Connection Problems
- Verify internet connectivity
- Check firewall settings
- Confirm data source credentials
- Review WebSocket connection settings

#### Display Issues
- Update graphics drivers
- Verify Vulkan SDK installation
- Check for conflicting display managers

### Performance Optimization Tips

#### For High-Frequency Data
- Enable viewport culling
- Use LOD rendering for distant data
- Limit the number of indicators
- Increase update intervals

#### For Large Datasets
- Use data aggregation options
- Enable memory pooling
- Utilize data compression
- Implement efficient caching

### Diagnostic Tools
- Performance overlay (toggle with F12)
- Frame time graph
- Memory tracker
- Pipeline metrics
- Debug overlay

## Keyboard Shortcuts

### General Shortcuts
- `Ctrl+N`: New layout
- `Ctrl+S`: Save layout
- `Ctrl+O`: Open layout
- `F5-F8`: Quick save/load layouts
- `Shift+F5-F8`: Load quick-saved layouts
- `F12`: Toggle performance overlay
- `Space`: Quick actions toolbar
- `Ctrl+,`: Open settings

### Chart Panel Shortcuts
- `+/-`: Zoom in/out
- Arrow Keys: Pan chart
- `R`: Reset zoom
- `M`: Add marker
- `L`: Add trend line
- `T`: Add text annotation

### Panel Management
- `Ctrl+W`: Close current panel
- `Ctrl+Shift+W`: Close all panels
- `F11`: Toggle fullscreen
- `Ctrl+D`: Duplicate panel

### Trading Functions
- `Ctrl+B`: Buy order
- `Ctrl+S`: Sell order
- `Ctrl+X`: Close position
- `Ctrl+C`: Cancel all orders

---

## About This Guide

This user guide covers all major features of the BTQ Render Engine. For the most up-to-date information, please check the official documentation in the `dependencies/BTQ_Render_Engine/docs/` directory.

For additional support, please refer to the documentation in the `docs/` directory or contact our support team.