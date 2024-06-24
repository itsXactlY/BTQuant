import os
import pandas as pd
import re

def extract_info(content):
    pair = re.search(r'Results for (\w+):', content)
    max_drawdown = re.search(r'Max Drawdown: ([\d.]+)', content)
    sqn = re.search(r'SQN: ([\d.]+)', content)
    final_portfolio_value = re.search(r'Final Portfolio Value: \$([\d.]+)', content)
    pl = re.search(r'P/L: \$([\d.]+)', content)
    
    total_trades = re.search(r"'total': AutoOrderedDict\({'total': (\d+),", content)
    won_trades = re.search(r"'won': AutoOrderedDict\({'total': (\d+),", content)
    
    if total_trades and won_trades:
        total_trades = int(total_trades.group(1))
        won_trades = int(won_trades.group(1))
        win_rate = won_trades / total_trades if total_trades > 0 else 0
    else:
        total_trades = 0
        win_rate = 0

    return {
        'Pair': pair.group(1) if pair else '',
        'Max Drawdown': float(max_drawdown.group(1)) if max_drawdown else 0,
        'SQN': float(sqn.group(1)) if sqn else 0,
        'Final Portfolio Value': float(final_portfolio_value.group(1)) if final_portfolio_value else 0,
        'P/L': float(pl.group(1)) if pl else 0,
        'Total Trades': total_trades,
        'Win Rate': win_rate
    }

def process_files(root_dir):
    results = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    results.append(extract_info(content))
    return results


root_dir = '/usr/share/nginx/html/QuantStats/'
data = process_files(root_dir)
df = pd.DataFrame(data)
df_sorted = df.sort_values('Final Portfolio Value', ascending=False)
json_data = df_sorted.to_json(orient='records')


html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .chart { width: 100%; height: 400px; margin-bottom: 20px; }
        #pair-details { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Quantitative Analysis Dashboard</h1>
    <select id="pair-dropdown"></select>
    <div id="portfolio-value-chart" class="chart"></div>
    <div id="sqn-drawdown-scatter" class="chart"></div>
    <div id="win-rate-hist" class="chart"></div>
    <div id="trades-pl-scatter" class="chart"></div>
    <div id="pair-details"></div>

    <script>
        const data = JSON_DATA_PLACEHOLDER;

        const dropdown = document.getElementById('pair-dropdown');
        data.forEach(item => {
            const option = document.createElement('option');
            option.value = item.Pair;
            option.textContent = item.Pair;
            dropdown.appendChild(option);
        });

        function updateCharts(selectedPair) {
            const pairData = data.find(item => item.Pair === selectedPair);

            // Portfolio Value Chart
            const portfolioTrace = {
                x: data.map(item => item.Pair),
                y: data.map(item => item['Final Portfolio Value']),
                type: 'bar',
                name: 'All Pairs'
            };
            const selectedPairTrace = {
                x: [selectedPair],
                y: [pairData['Final Portfolio Value']],
                type: 'bar',
                name: 'Selected Pair',
                marker: { color: 'red' }
            };
            Plotly.newPlot('portfolio-value-chart', [portfolioTrace, selectedPairTrace], {
                title: 'Final Portfolio Value by Pair',
                xaxis: { title: 'Pair' },
                yaxis: { title: 'Final Portfolio Value' }
            });

            // SQN vs Max Drawdown Scatter
            const sqnDrawdownTrace = {
                x: data.map(item => item['Max Drawdown']),
                y: data.map(item => item.SQN),
                mode: 'markers',
                type: 'scatter',
                text: data.map(item => item.Pair),
                name: 'All Pairs'
            };
            const selectedSqnDrawdownTrace = {
                x: [pairData['Max Drawdown']],
                y: [pairData.SQN],
                mode: 'markers',
                type: 'scatter',
                marker: { color: 'red', size: 10 },
                name: 'Selected Pair'
            };
            Plotly.newPlot('sqn-drawdown-scatter', [sqnDrawdownTrace, selectedSqnDrawdownTrace], {
                title: 'SQN vs Max Drawdown',
                xaxis: { title: 'Max Drawdown' },
                yaxis: { title: 'SQN' }
            });

            // Win Rate Histogram
            const winRateTrace = {
                x: data.map(item => item['Win Rate']),
                type: 'histogram',
                name: 'All Pairs'
            };
            Plotly.newPlot('win-rate-hist', [winRateTrace], {
                title: 'Win Rate Distribution',
                xaxis: { title: 'Win Rate' },
                yaxis: { title: 'Count' },
                shapes: [{
                    type: 'line',
                    x0: pairData['Win Rate'],
                    x1: pairData['Win Rate'],
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: { color: 'red', width: 2 }
                }]
            });

            // Total Trades vs P/L Scatter
            const tradesPLTrace = {
                x: data.map(item => item['Total Trades']),
                y: data.map(item => item['P/L']),
                mode: 'markers',
                type: 'scatter',
                text: data.map(item => item.Pair),
                name: 'All Pairs'
            };
            const selectedTradesPLTrace = {
                x: [pairData['Total Trades']],
                y: [pairData['P/L']],
                mode: 'markers',
                type: 'scatter',
                marker: { color: 'red', size: 10 },
                name: 'Selected Pair'
            };
            Plotly.newPlot('trades-pl-scatter', [tradesPLTrace, selectedTradesPLTrace], {
                title: 'Total Trades vs P/L',
                xaxis: { title: 'Total Trades' },
                yaxis: { title: 'P/L' }
            });

            // Update pair details
            document.getElementById('pair-details').innerHTML = `
                <h3>Details for ${selectedPair}</h3>
                <p>Final Portfolio Value: $${pairData['Final Portfolio Value'].toFixed(2)}</p>
                <p>P/L: $${pairData['P/L'].toFixed(2)}</p>
                <p>Max Drawdown: ${pairData['Max Drawdown'].toFixed(2)}%</p>
                <p>SQN: ${pairData.SQN.toFixed(2)}</p>
                <p>Total Trades: ${pairData['Total Trades']}</p>
                <p>Win Rate: ${pairData['Win Rate'].toFixed(2)}</p>
            `;
        }

        dropdown.addEventListener('change', (event) => {
            updateCharts(event.target.value);
        });

        // Initial chart render
        updateCharts(data[0].Pair);
    </script>
</body>
</html>
'''

html_content = html_template.replace('JSON_DATA_PLACEHOLDER', json_data)

with open('quantstats_dashboard.html', 'w') as f:
    f.write(html_content)

print("Dashboard has been saved as quantstats_dashboard.html")