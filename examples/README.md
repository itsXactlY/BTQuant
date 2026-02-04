# Code Examples

This directory contains sample implementations of common tasks and reusable code snippets for the PubBTQuant project. These examples demonstrate how to use the various components of the system, particularly the TaskScheduler for performing financial calculations.

## Available Examples

### 1. Volume Analysis (`volume_analysis_example.cpp`)
Demonstrates how to use the TaskScheduler for common volume-based calculations:
- Volume profile calculation
- Volume Weighted Average Price (VWAP)
- Volume by time calculations
- Rolling volume profiles
- Time-weighted volume

### 2. Technical Indicators (`indicator_calculations_example.cpp`)
Shows how to calculate various technical indicators using the TaskScheduler:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Adaptive SMA
- Hull Moving Average
- Keltner Channels

### 3. Data Processing (`data_processing_example.cpp`)
Illustrates common data processing tasks:
- Candle aggregation
- Trade filtering
- Histogram calculations
- Parallel data transformation
- Dynamic timeframe candles
- Batch processing
- Trade partitioning

### 4. Correlation and Advanced Analytics (`correlation_analytics_example.cpp`)
Demonstrates advanced analytical calculations:
- Cross-series correlation
- Average True Range (ATR)
- Stochastic Oscillator
- On-Balance Volume (OBV)
- Batch indicator calculations
- Multiple timeframe analysis
- Correlation matrices
- Market microstructure filters
- Volume Price Confirmation Indicator (VPCI)

## Building the Examples

To compile these examples, you'll need to link against the BTQ_Render_Engine library:

```bash
g++ -std=c++20 -I. examples/volume_analysis_example.cpp dependencies/BTQ_Render_Engine/src/threading/task_scheduler.cpp -o volume_analysis_example
```

Or use your preferred build system with the appropriate include paths and library linking.

## Usage

Each example is self-contained and demonstrates a specific set of functionality. You can run them individually to see how different calculations work:

```bash
./volume_analysis_example
./indicator_calculations_example
./data_processing_example
./correlation_analytics_example
```

## Best Practices Demonstrated

- Proper initialization of the TaskScheduler
- Asynchronous calculation patterns
- Error handling for futures
- Sample data generation for testing
- Result validation and display