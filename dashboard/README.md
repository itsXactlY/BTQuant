# Quantstats Trading Pair Dashboard

## Quick Start Guide

### Installation
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

1.1 Install pyker (PM2 Monitor replacement)
```bash
git clone https://github.com/mrvi0/pyker.git
cd pyker

# Run installer (no sudo required!)
python3 install.py
```

### Running the Dashboard
```bash
streamlit run quantstats_dashboard.py

or

pyker start dashboard start.py
```

### Usage
1. **Upload Files**: Use the sidebar to upload multiple quantstats HTML files
2. **Or Specify Folder**: Enter the path to your folder containing quantstats reports
3. **Analyze**: The dashboard will automatically parse and analyze all trading pairs
4. **Compare**: View performance metrics, risk-return profiles, and rankings
5. **Export**: Download analysis results as CSV

### Features
- 📊 **Performance Overview**: Key metrics comparison across all pairs
- 📈 **Risk-Return Analysis**: Interactive scatter plot with Sharpe ratio sizing
- 🏆 **Performance Ranking**: Composite scoring system
- 🔥 **Yearly Heatmap**: Year-over-year performance visualization
- 📋 **Detailed Metrics**: Comprehensive data table
- 💾 **Export Functionality**: Download results as CSV

### Key Metrics Analyzed
- Total Return & CAGR
- Sharpe & Sortino Ratios
- Maximum Drawdown
- Volatility (Annual)
- Risk-Adjusted Returns
- Yearly Performance
- Drawdown Analysis

### File Structure
```
quantstats_reports/
├── BTC_USDT_2025-09-25_20-41-21.html
├── ETH_USDT_2025-09-25_20-41-21.html
├── ADA_USDT_2025-09-25_20-41-21.html
└── ... (more quantstats HTML files)
```

The dashboard automatically extracts trading pair names from filenames and parses all quantstats metrics for comprehensive analysis.
