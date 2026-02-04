# Volume Analysis Types Documentation

This document provides detailed explanations of all 18 volume analysis types available in the trading platform. Each type offers unique insights into market activity and can be used for different analytical purposes.

## Overview

Volume analysis is a crucial aspect of market analysis that helps traders understand the strength behind price movements. Different volume metrics reveal various aspects of market sentiment, liquidity, and participant behavior.

## Volume Analysis Types

### 1. Trades
**Description:** Total number of trades executed at a specific price level or time period.
**Calculation:** Count of all trade executions
**Use Case:** Shows the frequency of trading activity regardless of volume size. Useful for identifying areas of high transactional interest.
**Interpretation:** Higher trade counts indicate more market participant activity and interest at that price level.
**Example:** If there were 150 trades at $100.50, the Trades value would be 150.

### 2. BuyTrades
**Description:** Number of buy trades executed at a specific price level or time period.
**Calculation:** Count of all buy-side trade executions
**Use Case:** Identifies the frequency of buying interest at specific price levels.
**Interpretation:** High buy trade counts suggest strong demand at those price points.
**Example:** If there were 85 buy trades at $100.50, the BuyTrades value would be 85.

### 3. SellTrades
**Description:** Number of sell trades executed at a specific price level or time period.
**Calculation:** Count of all sell-side trade executions
**Use Case:** Identifies the frequency of selling pressure at specific price levels.
**Interpretation:** High sell trade counts suggest strong supply or profit-taking activity.
**Example:** If there were 65 sell trades at $100.50, the SellTrades value would be 65.

### 4. Volume
**Description:** Total volume traded (sum of bid and ask volumes) at a specific price level or time period.
**Calculation:** Total quantity of contracts/shares traded
**Use Case:** Measures overall liquidity and activity at price levels. Fundamental metric for volume profile analysis.
**Interpretation:** Higher volumes indicate greater market interest and potential support/resistance levels.
**Example:** If 1,200 contracts were traded at $100.50, the Volume value would be 1,200.

### 5. BuyVolume
**Description:** Total volume of buy trades at a specific price level or time period.
**Calculation:** Sum of all volume from buy-side transactions
**Use Case:** Identifies where buyers are most active and aggressive.
**Interpretation:** High buy volume suggests strong demand and potential support levels.
**Example:** If 750 contracts were bought at $100.50, the BuyVolume value would be 750.

### 6. SellVolume
**Description:** Total volume of sell trades at a specific price level or time period.
**Calculation:** Sum of all volume from sell-side transactions
**Use Case:** Identifies where sellers are most active and aggressive.
**Interpretation:** High sell volume suggests strong supply and potential resistance levels.
**Example:** If 450 contracts were sold at $100.50, the SellVolume value would be 450.

### 7. BuyVolumePercent
**Description:** Percentage of total volume attributed to buy trades at a specific price level or time period.
**Calculation:** (BuyVolume / Total Volume) × 100
**Use Case:** Shows the proportional strength of buying interest relative to total activity.
**Interpretation:** Values closer to 100% indicate overwhelming buying pressure, while lower values suggest selling dominance.
**Example:** If 750 of 1,200 contracts were buys at $100.50, BuyVolumePercent would be 62.5%.

### 8. SellVolumePercent
**Description:** Percentage of total volume attributed to sell trades at a specific price level or time period.
**Calculation:** (SellVolume / Total Volume) × 100
**Use Case:** Shows the proportional strength of selling pressure relative to total activity.
**Interpretation:** Values closer to 100% indicate overwhelming selling pressure, while lower values suggest buying dominance.
**Example:** If 450 of 1,200 contracts were sells at $100.50, SellVolumePercent would be 37.5%.

### 9. BuySellVolume
**Description:** Difference between buy volume and sell volume (BuyVolume - SellVolume) at a specific price level or time period.
**Calculation:** BuyVolume - SellVolume
**Use Case:** Shows net buying or selling pressure at price levels.
**Interpretation:** Positive values indicate net buying pressure, negative values indicate net selling pressure.
**Example:** If 750 contracts were bought and 450 sold at $100.50, BuySellVolume would be +300.

### 10. Delta
**Description:** Net difference between buy and sell volume (BuyVolume - SellVolume) at a specific price level or time period.
**Calculation:** BuyVolume - SellVolume
**Use Case:** Measures the imbalance between buying and selling pressure.
**Interpretation:** Positive delta indicates buying pressure, negative delta indicates selling pressure. Zero delta means balanced activity.
**Example:** If 750 contracts were bought and 450 sold at $100.50, Delta would be +300.

### 11. DeltaPercent
**Description:** Delta expressed as a percentage of total volume at a specific price level or time period.
**Calculation:** ((BuyVolume - SellVolume) / Total Volume) × 100
**Use Case:** Normalizes delta values to account for different volume scales across instruments or timeframes.
**Interpretation:** Values range from -100% (all selling) to +100% (all buying). Helps compare imbalances across different contexts.
**Example:** If delta is +300 out of 1,200 total volume at $100.50, DeltaPercent would be +25%.

### 12. CumulativeDelta
**Description:** Running sum of delta values accumulated over time or price levels.
**Calculation:** Sum of all previous delta values up to current point
**Use Case:** Tracks the overall buying or selling bias over extended periods.
**Interpretation:** Rising cumulative delta suggests sustained buying pressure, falling values suggest sustained selling pressure.
**Example:** If deltas over three price levels were +100, +50, and -25, the cumulative delta would be +125.

### 13. AverageSize
**Description:** Average size of trades at a specific price level or time period.
**Calculation:** Total Volume / Number of Trades
**Use Case:** Identifies typical trade sizes and potential institutional vs retail activity.
**Interpretation:** Larger average sizes may indicate institutional participation, smaller sizes suggest retail activity.
**Example:** If 1,200 contracts were traded in 150 trades at $100.50, AverageSize would be 8 contracts per trade.

### 14. AverageBuySize
**Description:** Average size of buy trades at a specific price level or time period.
**Calculation:** BuyVolume / BuyTrades
**Use Case:** Identifies the typical size of buying transactions.
**Interpretation:** Helps distinguish between large institutional buyers and smaller retail buyers.
**Example:** If 750 contracts were bought in 85 trades at $100.50, AverageBuySize would be approximately 8.8 contracts per buy trade.

### 15. AverageSellSize
**Description:** Average size of sell trades at a specific price level or time period.
**Calculation:** SellVolume / SellTrades
**Use Case:** Identifies the typical size of selling transactions.
**Interpretation:** Helps distinguish between large institutional sellers and smaller retail sellers.
**Example:** If 450 contracts were sold in 65 trades at $100.50, AverageSellSize would be approximately 6.9 contracts per sell trade.

### 16. MaxOneTradeVolume
**Description:** Maximum volume of a single trade at a specific price level or time period.
**Calculation:** Highest volume value among all individual trades
**Use Case:** Identifies the largest single transaction, which may indicate institutional activity.
**Interpretation:** Large values suggest presence of institutional or algorithmic trading at that price level.
**Example:** If the largest single trade at $100.50 was 150 contracts, MaxOneTradeVolume would be 150.

### 17. FilteredVolume
**Description:** Volume filtered by specific criteria (e.g., minimum trade size, trade direction, time windows).
**Calculation:** Volume that meets predefined filtering conditions
**Use Case:** Allows analysis of specific types of market activity based on custom parameters.
**Interpretation:** Depends on the applied filters. Useful for isolating particular market behaviors.
**Example:** Volume from trades of 100+ contracts only, or volume during specific market hours.

### 18. SplitVolume
**Description:** Split volume display showing buy volume on left half and sell volume on right half of the same price level.
**Calculation:** Visual representation splitting buy and sell volumes within the same bar
**Use Case:** Provides immediate visual comparison of buy vs sell volume at each price level.
**Interpretation:** Left side shows buying pressure, right side shows selling pressure. Asymmetry indicates directional bias.
**Example:** At $100.50, a bar split with 750 on the left (buy) and 450 on the right (sell) shows stronger buying interest.

## Practical Applications

### Support and Resistance Identification
- **High Volume** levels often become support or resistance zones
- **Low Volume** areas are typically easy to break through
- **Imbalanced Volume** (high delta) can signal potential reversals

### Market Sentiment Analysis
- **Positive Delta** indicates bullish sentiment
- **Negative Delta** indicates bearish sentiment
- **Cumulative Delta** trends show longer-term sentiment shifts

### Liquidity Assessment
- **Volume** metrics help identify liquid price levels
- **Average Size** can indicate institutional interest
- **Trade Counts** show market participant engagement

### Order Flow Analysis
- **Buy/Sell Volume** ratios reveal directional bias
- **Large Individual Trades** (MaxOneTradeVolume) may indicate institutional activity
- **Filtered Volume** allows focus on specific market participant behavior

## Best Practices

1. **Combine Multiple Metrics:** Use several volume types together for comprehensive analysis
2. **Context Matters:** Consider timeframe and market conditions when interpreting values
3. **Validate with Price Action:** Corroborate volume signals with price movement patterns
4. **Watch for Divergences:** Differences between volume and price trends can signal reversals
5. **Scale Appropriately:** Adjust expectations based on instrument liquidity and typical volume patterns