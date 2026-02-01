# Indicator Alerts System

## Overview
The Indicator Alerts system provides real-time notifications when specific technical indicator conditions are met. It monitors price action relative to various technical indicators and triggers alerts when predefined conditions are detected.

## Supported Alert Types

### 1. Price vs SMA Cross
- Triggers when price crosses above or below the Simple Moving Average
- Includes threshold configuration to avoid noise
- Reports direction (bullish/bearish)

### 2. RSI Overbought/Oversold
- RSI Oversold: Triggers when RSI drops below the oversold threshold (default: 30)
- RSI Overbought: Triggers when RSI rises above the overbought threshold (default: 70)
- Only triggers on entry to oversold/overbought territory

### 3. Bollinger Band Touch/Breakout
- Bollinger Band Touch: Triggers when price touches the upper or lower Bollinger Band
- Bollinger Band Breakout: Triggers when price breaks out above or below the Bollinger Bands
- Includes configurable threshold for touch detection

### 4. MACD Signal Cross
- Triggers when MACD line crosses the signal line
- Reports direction of cross (bullish/bearish)

### 5. Stochastic Oscillator
- Stochastic Oversold: Triggers when Stochastic K or D drops below 20
- Stochastic Overbought: Triggers when Stochastic K or D rises above 80

## Usage Example

```cpp
#include "indicators/indicator_alerts.hpp"

btq::IndicatorAlerts alerts;

// Set up a callback to handle alerts
alerts.setAlertCallback([](const btq::IndicatorAlertEvent& event) {
    switch(event.alert_type) {
        case btq::IndicatorAlertType::PRICE_CROSSES_SMA:
            std::cout << "Price crosses SMA" << std::endl;
            break;
        case btq::IndicatorAlertType::RSI_OVERSOLD:
            std::cout << "RSI oversold" << std::endl;
            break;
        // Handle other alert types...
    }
});

// Configure thresholds
alerts.setSMACrossThreshold(0.001);
alerts.setRSIThresholds(70, 30);  // Overbought at 70, Oversold at 30
alerts.setBollingerBandThreshold(0.01);

// Check for alerts with current market data
alerts.checkAlerts(current_bar, sma_values, rsi_values, bb_upper_values, 
                  bb_lower_values, bb_middle_values, macd_values, 
                  macd_signal_values, stoch_k_values, stoch_d_values, "SYMBOL");
```

## Configuration Options

- `setSMACrossThreshold(double threshold)`: Sets the minimum price difference required to trigger an SMA cross alert
- `setRSIThresholds(int overbought, int oversold)`: Sets the RSI overbought and oversold levels
- `setBollingerBandThreshold(double threshold)`: Sets the threshold for Bollinger Band touch detection
- `setMACDThreshold(double threshold)`: Sets the minimum difference required to trigger a MACD cross alert

## Event Information

Each alert event contains:
- Alert type
- Timestamp
- Current price
- Indicator value(s)
- Symbol
- Direction (bullish/bearish)
- Band values (for Bollinger Bands)