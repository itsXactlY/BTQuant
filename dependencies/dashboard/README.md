# BTQuant QuantStats Performance Dashboard

## Overview

A comprehensive web-based dashboard for analyzing and comparing QuantStats performance reports across multiple trading pairs and strategies. Built with Streamlit for the BTQuant algorithmic trading framework.

## Quick Start Guide

### Installation
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Optional: Install pyker for process management
```bash
git clone https://github.com/mrvi0/pyker.git
cd pyker
python3 install.py
```

### Running the Dashboard
```bash
# Direct Streamlit run
streamlit run quantstats_dashboard.py

# Or using pyker
pyker start dashboard start.py
```

## Features

### 📊 Performance Analytics
- **Multi-Pair Comparison**: Analyze performance across all trading pairs simultaneously
- **Risk-Return Analysis**: Interactive scatter plots with Sharpe ratio visualization
- **Performance Ranking**: Composite scoring system for strategy evaluation
- **Yearly Heatmaps**: Year-over-year performance visualization
- **Detailed Metrics Table**: Comprehensive performance statistics

### 📈 Key Metrics
- Total Return & Compound Annual Growth Rate (CAGR)
- Sharpe & Sortino Ratios (risk-adjusted returns)
- Maximum Drawdown & Recovery Analysis
- Annual Volatility & Risk Metrics
- Win Rate & Profit Factor
- Calmar Ratio & Risk-Adjusted Performance

### 💾 Data Management
- **File Upload**: Support for multiple QuantStats HTML report files
- **Directory Scanning**: Automatic discovery of reports in specified folders
- **CSV Export**: Download analysis results for further processing
- **Real-time Updates**: Live performance monitoring capabilities

## Usage

### Data Input
1. **Upload Files**: Use the sidebar to upload multiple QuantStats HTML files
2. **Specify Folder**: Enter path to folder containing QuantStats reports
3. **Auto-Detection**: The dashboard automatically parses trading pair names from filenames

### Analysis Workflow
1. **Load Data**: Upload or specify QuantStats report files
2. **Review Overview**: Examine key performance metrics across all pairs
3. **Risk Analysis**: Analyze risk-return profiles with interactive charts
4. **Compare Strategies**: Rank and compare different trading approaches
5. **Export Results**: Download comprehensive analysis as CSV

## File Structure

```
quantstats_reports/
├── BTC_USDT_2024-01-01_12-00-00.html
├── ETH_USDT_2024-01-01_12-00-00.html
├── ADA_USDT_2024-01-01_12-00-00.html
└── ... (additional QuantStats HTML reports)
```

## Integration with BTQuant

This dashboard integrates seamlessly with the BTQuant backtesting pipeline:

- **Automatic Report Generation**: QuantStats reports created during backtesting
- **Multi-Strategy Analysis**: Compare different strategies across assets
- **Performance Persistence**: Store and analyze historical performance data
- **Live Trading Integration**: Monitor live strategy performance

## Technical Details

- **Framework**: Streamlit for responsive web interface
- **Data Processing**: Pandas for efficient data manipulation
- **Visualization**: Plotly for interactive charts
- **Export**: CSV format for external analysis tools
- **Performance**: Optimized for large datasets with multiple trading pairs

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- QuantStats (automatically included in BTQuant)

## Development

The dashboard is part of the BTQuant ecosystem and follows the project's architecture patterns for consistency and maintainability.
