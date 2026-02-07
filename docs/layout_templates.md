# Layout Templates

This document describes the predefined layout templates available in the trading application.

## Available Templates

### 1. Scalper Layout (`scalper_layout.json`)
**Target Users:** Active day traders focusing on quick entries and exits
**Primary Components:**
- **DOM Ladder** (Type 2): Displays the order book depth with price levels and volumes
- **Time & Sales (Tape)** (Type 13): Shows real-time trade executions with price, size, and time
- **Price Chart** (Type 0): Visual representation of price movements
- **Order Book** (Type 10): Detailed view of buy/sell orders at each price level
- **Active Orders** (Type 6): List of current working orders
- **Positions** (Type 7): Current position holdings
- **Watchlist** (Type 11): Monitored instruments

### 2. Analyst Layout (`analyst_layout.json`)
**Target Users:** Technical analysts and swing traders
**Primary Components:**
- **Main Price Chart** (Type 0): Large chart with technical indicators
- **Market Depth** (Type 10): Visual representation of order book
- **Volume Profile** (Type 14): Shows volume traded at different price levels
- **TPO Profile** (Type 19): Time Price Opportunity profile
- **Footprint Chart** (Type 18): Advanced visualization showing trade flow
- **Market Statistics** (Type 20): Key market metrics
- **Multi-Timeframe Analysis** (Type 0): Simultaneous view of multiple timeframes

### 3. Options Layout (`options_layout.json`)
**Target Users:** Options traders and risk managers
**Primary Components:**
- **Options Chain** (Type 15): List of available options with prices and greeks
- **Greeks Monitor** (Type 16): Real-time Greek values (delta, gamma, theta, vega, rho)
- **Options Desk** (Type 17): Trading interface for options strategies
- **Risk Matrix** (Type 21): Visual representation of portfolio risk
- **Positions** (Type 7): Current options positions
- **Active Orders** (Type 6): Working options orders
- **Risk Summary** (Type 22): Portfolio risk metrics
- **Options Strategy Builder** (Type 23): Tool for constructing options strategies

## Panel Types Reference

| Type | Name | Description |
|------|------|-------------|
| 0 | Chart | Price chart with various drawing tools |
| 2 | DOM Ladder | Dynamic Order Matching ladder |
| 6 | Active Orders | List of active orders |
| 7 | Positions | Current position holdings |
| 10 | Order Book | Detailed order book view |
| 11 | Watchlist | List of monitored symbols |
| 13 | Time & Sales | Real-time trade executions (the tape) |
| 14 | Volume Profile | Volume distribution by price |
| 15 | Options Chain | Options contracts with pricing |
| 16 | Greeks Monitor | Options Greek values |
| 17 | Options Desk | Options trading interface |
| 18 | Footprint Chart | Advanced trade flow visualization |
| 19 | TPO Profile | Time Price Opportunity profile |
| 20 | Market Statistics | Market metrics and statistics |
| 21 | Risk Matrix | Portfolio risk matrix |
| 22 | Risk Summary | Risk metrics summary |
| 23 | Options Strategy Builder | Options strategy construction tool |
| 27 | Strategy Builder Footer | Strategy building interface |

## Grid System

All layouts use a grid system with:
- 6 columns
- 10 rows
- Panels positioned using grid coordinates (x, y) and dimensions (width, height)