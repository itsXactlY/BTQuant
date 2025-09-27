// Trading Dashboard Application
class TradingDashboard {
    constructor() {
        this.data = [];
        this.filteredData = [];
        this.charts = {};
        this.clusters = [];
        this.outliers = [];
        this.currentTab = 'overview';
        
        this.init();
        // this.generateSampleData();
        this.updateDashboard();
    }

    init() {
        this.bindEvents();
        this.initializeCharts();
        this.updateSliderValues();
    }

    // Generate comprehensive sample data for 523 trading pairs
    // generateSampleData() {
    //     const pairs = [
    //         'ACA_USDT', 'ADA_USDT', 'ALGO_USDT', 'ATOM_USDT', 'AVAX_USDT',
    //         'BNB_USDT', 'BTC_USDT', 'DOT_USDT', 'ETH_USDT', 'FTM_USDT',
    //         'LINK_USDT', 'MATIC_USDT', 'SOL_USDT', 'UNI_USDT', 'XRP_USDT'
    //     ];
        
    //     // Generate 523 pairs by creating variations
    //     const allPairs = [];
    //     for (let i = 0; i < 523; i++) {
    //         const basePair = pairs[i % pairs.length];
    //         const suffix = i < pairs.length ? '' : `_${Math.floor(i / pairs.length)}`;
    //         allPairs.push(basePair + suffix);
    //     }

    //     this.data = allPairs.map((pair, index) => {
    //         const baseReturn = (Math.random() - 0.3) * 2; // -0.6 to 1.4 range
    //         const volatility = 0.1 + Math.random() * 0.4; // 10% to 50%
    //         const sharpe = (baseReturn / volatility) + (Math.random() - 0.5) * 0.8;
    //         const maxDrawdown = -(0.1 + Math.random() * 0.7); // -10% to -80%
            
    //         return {
    //             pair: pair,
    //             total_return: baseReturn,
    //             cagr: baseReturn * 0.3 + (Math.random() - 0.5) * 0.1,
    //             sharpe: Math.max(-2, Math.min(4, sharpe)),
    //             max_drawdown: maxDrawdown,
    //             volatility: volatility,
    //             yearly_returns: this.generateYearlyReturns(),
    //             category: this.categorizePerformance(baseReturn, sharpe, maxDrawdown),
    //             risk_score: this.calculateRiskScore(volatility, maxDrawdown),
    //             consistency_score: Math.random() * 100
    //         };
    //     });

    //     this.filteredData = [...this.data];
    //     this.performClustering();
    //     this.detectOutliers();
    // }

    generateYearlyReturns() {
        return {
            '2022': (Math.random() - 0.4) * 1.5,
            '2023': (Math.random() - 0.3) * 1.2,
            '2024': (Math.random() - 0.2) * 1.8,
            '2025': (Math.random() - 0.5) * 2.0
        };
    }

    categorizePerformance(return_, sharpe, drawdown) {
        if (sharpe > 1.5 && return_ > 0.3 && drawdown > -0.3) return 'Excellent';
        if (sharpe > 0.8 && return_ > 0.1 && drawdown > -0.5) return 'Good';
        if (sharpe > 0.2 && return_ > -0.1) return 'Average';
        return 'Poor';
    }

    calculateRiskScore(volatility, drawdown) {
        return Math.min(100, Math.max(0, 100 - (volatility * 100 + Math.abs(drawdown) * 50)));
    }

    performClustering() {
        // Simplified K-means clustering
        const features = this.data.map(d => [
            d.total_return, d.sharpe, d.max_drawdown, d.volatility
        ]);
        
        this.clusters = [
            { id: 0, name: 'High Performance', color: '#1FB8CD', pairs: [] },
            { id: 1, name: 'Moderate Risk', color: '#FFC185', pairs: [] },
            { id: 2, name: 'Conservative', color: '#B4413C', pairs: [] },
            { id: 3, name: 'High Risk', color: '#5D878F', pairs: [] }
        ];

        // Assign pairs to clusters based on characteristics
        this.data.forEach((pair, index) => {
            let clusterId = 0;
            if (pair.sharpe > 1.2 && pair.total_return > 0.2) clusterId = 0;
            else if (pair.volatility < 0.25 && pair.max_drawdown > -0.4) clusterId = 2;
            else if (pair.volatility > 0.35 || pair.max_drawdown < -0.5) clusterId = 3;
            else clusterId = 1;
            
            this.clusters[clusterId].pairs.push({...pair, cluster: clusterId});
        });
    }

    detectOutliers() {
        // Simple outlier detection based on extreme values
        this.outliers = this.data.filter(pair => {
            return pair.volatility > 0.45 || 
                   pair.max_drawdown < -0.7 || 
                   pair.sharpe > 3 || 
                   pair.sharpe < -1.5 ||
                   pair.total_return > 1.5 ||
                   pair.total_return < -0.8;
        }).map(pair => ({
            ...pair,
            reason: this.getOutlierReason(pair)
        }));
    }

    getOutlierReason(pair) {
        if (pair.volatility > 0.45) return 'High Volatility';
        if (pair.max_drawdown < -0.7) return 'Extreme Drawdown';
        if (pair.sharpe > 3) return 'Exceptional Sharpe';
        if (pair.sharpe < -1.5) return 'Poor Risk-Adjusted Return';
        if (pair.total_return > 1.5) return 'Exceptional Return';
        if (pair.total_return < -0.8) return 'Severe Losses';
        return 'Statistical Anomaly';
    }

    bindEvents() {
        // Tab navigation
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabId = e.target.getAttribute('data-tab');
                this.switchTab(tabId);
            });
        });

        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-primary)';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = 'var(--color-border)';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--color-border)';
            this.handleFileUpload(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Sliders
        document.getElementById('topNSlider').addEventListener('input', (e) => {
            document.getElementById('topNValue').textContent = e.target.value;
            this.updateCharts();
        });

        document.getElementById('minSharpeSlider').addEventListener('input', (e) => {
            document.getElementById('minSharpeValue').textContent = e.target.value;
            this.applyFilters();
        });

        document.getElementById('maxDrawdownSlider').addEventListener('input', (e) => {
            document.getElementById('maxDrawdownValue').textContent = e.target.value + '%';
            this.applyFilters();
        });

        document.getElementById('minReturnSlider').addEventListener('input', (e) => {
            document.getElementById('minReturnValue').textContent = e.target.value + '%';
            this.applyFilters();
        });

        // Search
        document.getElementById('pairSearch').addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });

        // Export buttons
        document.getElementById('exportAll').addEventListener('click', () => {
            this.exportData('all');
        });

        document.getElementById('exportTop').addEventListener('click', () => {
            this.exportData('top');
        });

        // Settings checkboxes
        document.getElementById('enableClustering').addEventListener('change', (e) => {
            if (e.target.checked) this.performClustering();
            this.updateCharts();
        });

        document.getElementById('enableOutlierDetection').addEventListener('change', (e) => {
            if (e.target.checked) this.detectOutliers();
            this.updateTables();
        });

        // Heatmap controls
        document.getElementById('heatmapPairs').addEventListener('change', () => {
            this.updateHeatmap();
        });
    }

    handleFileUpload(files) {
        if (files.length === 0) return;
        
        const progressSection = document.getElementById('progressSection');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressSection.style.display = 'block';
        
        // Simulate file processing
        let processed = 0;
        const total = files.length;
        
        const processFile = () => {
            processed++;
            const percentage = (processed / total) * 100;
            progressFill.style.width = `${percentage}%`;
            progressText.textContent = `${processed} / ${total} files processed`;
            
            if (processed < total) {
                setTimeout(processFile, 50 + Math.random() * 100);
            } else {
                setTimeout(() => {
                    progressSection.style.display = 'none';
                    // this.generateSampleData();
                    this.updateDashboard();
                }, 500);
            }
        };
        
        processFile();
    }

    switchTab(tabId) {
        // Update active tab
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

        // Update active panel
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(tabId).classList.add('active');

        this.currentTab = tabId;
        
        // Update charts for current tab
        setTimeout(() => {
            this.updateChartsForTab(tabId);
        }, 100);
    }

    updateChartsForTab(tabId) {
        switch(tabId) {
            case 'overview':
                this.updateOverviewCharts();
                break;
            case 'rankings':
                this.updateTables();
                break;
            case 'clustering':
                this.updateClusteringChart();
                break;
            case 'timeseries':
                this.updateTimeSeriesChart();
                break;
            case 'heatmap':
                this.updateHeatmap();
                break;
        }
    }

    initializeCharts() {
        // Initialize all chart containers
        const chartConfigs = {
            returnChart: {
                type: 'bar',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: false },
                        legend: { display: false }
                    }
                }
            },
            sharpeChart: {
                type: 'bar',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: false },
                        legend: { display: false }
                    }
                }
            },
            riskReturnChart: {
                type: 'scatter',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Volatility (%)' } },
                        y: { title: { display: true, text: 'Return (%)' } }
                    }
                }
            },
            categoryChart: {
                type: 'doughnut',
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            }
        };

        // Initialize charts
        Object.keys(chartConfigs).forEach(chartId => {
            const ctx = document.getElementById(chartId);
            if (ctx) {
                this.charts[chartId] = new Chart(ctx, chartConfigs[chartId]);
            }
        });
    }

    updateOverviewCharts() {
        // Return Distribution
        const returns = this.filteredData.map(d => d.total_return);
        const returnBins = this.createHistogramData(returns, 15, 'Return Distribution');
        this.updateChart('returnChart', returnBins);

        // Sharpe Distribution
        const sharpes = this.filteredData.map(d => d.sharpe);
        const sharpeBins = this.createHistogramData(sharpes, 15, 'Sharpe Distribution');
        this.updateChart('sharpeChart', sharpeBins);

        // Risk-Return Scatter
        const topN = parseInt(document.getElementById('topNSlider').value);
        const topPairs = this.filteredData
            .sort((a, b) => b.sharpe - a.sharpe)
            .slice(0, Math.min(topN, 50));
        
        this.updateChart('riskReturnChart', {
            datasets: [{
                label: 'Trading Pairs',
                data: topPairs.map(d => ({
                    x: d.volatility * 100,
                    y: d.total_return * 100
                })),
                backgroundColor: topPairs.map(d => 
                    d.category === 'Excellent' ? '#1FB8CD' :
                    d.category === 'Good' ? '#FFC185' :
                    d.category === 'Average' ? '#5D878F' : '#B4413C'
                )
            }]
        });

        // Category Distribution
        const categories = {};
        this.filteredData.forEach(d => {
            categories[d.category] = (categories[d.category] || 0) + 1;
        });

        this.updateChart('categoryChart', {
            labels: Object.keys(categories),
            datasets: [{
                data: Object.values(categories),
                backgroundColor: ['#1FB8CD', '#FFC185', '#5D878F', '#B4413C']
            }]
        });
    }

    updateTables() {
        // Top by Sharpe
        const topSharpe = this.filteredData
            .sort((a, b) => b.sharpe - a.sharpe)
            .slice(0, 20);
        this.updateTable('sharpeTable', topSharpe, [
            'pair', 'sharpe', 'total_return', 'max_drawdown'
        ]);

        // Top by Return
        const topReturn = this.filteredData
            .sort((a, b) => b.total_return - a.total_return)
            .slice(0, 20);
        this.updateTable('returnTable', topReturn, [
            'pair', 'total_return', 'sharpe', 'volatility'
        ]);

        // Risk-Adjusted
        const riskAdjusted = this.filteredData
            .map(d => ({...d, score: d.sharpe * 0.4 + (d.total_return / Math.abs(d.max_drawdown)) * 0.6}))
            .sort((a, b) => b.score - a.score)
            .slice(0, 20);
        this.updateTable('riskAdjustedTable', riskAdjusted, [
            'pair', 'score', 'sharpe', 'category'
        ]);

        // Outliers
        this.updateTable('outliersTable', this.outliers.slice(0, 20), [
            'pair', 'total_return', 'sharpe', 'volatility', 'reason'
        ]);
    }

    updateClusteringChart() {
        if (!document.getElementById('enableClustering').checked) return;

        const ctx = document.getElementById('clusteringChart');
        if (this.charts.clusteringChart) {
            this.charts.clusteringChart.destroy();
        }

        this.charts.clusteringChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: this.clusters.map(cluster => ({
                    label: cluster.name,
                    data: cluster.pairs.map(p => ({
                        x: p.volatility * 100,
                        y: p.total_return * 100
                    })),
                    backgroundColor: cluster.color + '80',
                    borderColor: cluster.color,
                    borderWidth: 1
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Volatility (%)' } },
                    y: { title: { display: true, text: 'Return (%)' } }
                }
            }
        });

        // Update cluster summary table
        const clusterStats = this.clusters.map(cluster => ({
            name: cluster.name,
            count: cluster.pairs.length,
            avgReturn: (cluster.pairs.reduce((sum, p) => sum + p.total_return, 0) / cluster.pairs.length) || 0,
            avgSharpe: (cluster.pairs.reduce((sum, p) => sum + p.sharpe, 0) / cluster.pairs.length) || 0,
            description: this.getClusterDescription(cluster)
        }));

        this.updateTable('clusterTable', clusterStats, [
            'name', 'count', 'avgReturn', 'avgSharpe', 'description'
        ], false);
    }

    getClusterDescription(cluster) {
        const avgReturn = cluster.pairs.reduce((sum, p) => sum + p.total_return, 0) / cluster.pairs.length;
        const avgVolatility = cluster.pairs.reduce((sum, p) => sum + p.volatility, 0) / cluster.pairs.length;
        
        if (avgReturn > 0.3 && avgVolatility < 0.25) return 'Low risk, high return performers';
        if (avgReturn > 0.1 && avgVolatility < 0.35) return 'Moderate risk, steady performers';
        if (avgVolatility < 0.25) return 'Conservative, low volatility pairs';
        return 'High risk, variable performance';
    }

    updateTimeSeriesChart() {
        const topPairs = this.filteredData
            .sort((a, b) => b.sharpe - a.sharpe)
            .slice(0, 20);

        const ctx = document.getElementById('timeseriesChart');
        if (this.charts.timeseriesChart) {
            this.charts.timeseriesChart.destroy();
        }

        const years = ['2022', '2023', '2024', '2025'];
        const colors = ['#1FB8CD', '#FFC185', '#B4413C', '#5D878F', '#DB4545'];

        this.charts.timeseriesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: topPairs.slice(0, 10).map((pair, index) => ({
                    label: pair.pair,
                    data: years.map(year => (pair.yearly_returns[year] * 100).toFixed(1)),
                    borderColor: colors[index % colors.length],
                    backgroundColor: colors[index % colors.length] + '20',
                    tension: 0.1
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'Return (%)' } }
                },
                plugins: {
                    legend: { position: 'right' }
                }
            }
        });
    }

    updateHeatmap() {
        const pairCount = parseInt(document.getElementById('heatmapPairs').value);
        const topPairs = this.filteredData
            .sort((a, b) => b.sharpe - a.sharpe)
            .slice(0, pairCount);

        const ctx = document.getElementById('heatmapChart');
        if (this.charts.heatmapChart) {
            this.charts.heatmapChart.destroy();
        }

        const metrics = ['total_return', 'sharpe', 'max_drawdown', 'volatility'];
        const data = [];
        
        topPairs.forEach((pair, pairIndex) => {
            metrics.forEach((metric, metricIndex) => {
                let value = pair[metric];
                if (metric === 'max_drawdown') value = Math.abs(value);
                if (metric === 'total_return' || metric === 'volatility') value *= 100;
                
                data.push({
                    x: metricIndex,
                    y: pairIndex,
                    v: value
                });
            });
        });

        // Create a simple heatmap using bar chart
        this.charts.heatmapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: topPairs.map(p => p.pair),
                datasets: [{
                    label: 'Performance Score',
                    data: topPairs.map(p => p.sharpe * 20 + 50),
                    backgroundColor: topPairs.map(p => {
                        const intensity = Math.min(1, Math.max(0, (p.sharpe + 2) / 4));
                        return `rgba(31, 184, 205, ${intensity})`;
                    })
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: { title: { display: true, text: 'Performance Score' } }
                }
            }
        });
    }

    createHistogramData(values, bins, label) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / bins;
        const binCounts = new Array(bins).fill(0);
        const binLabels = [];

        for (let i = 0; i < bins; i++) {
            const binStart = min + i * binWidth;
            const binEnd = min + (i + 1) * binWidth;
            binLabels.push(`${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`);
        }

        values.forEach(value => {
            const binIndex = Math.min(bins - 1, Math.floor((value - min) / binWidth));
            binCounts[binIndex]++;
        });

        return {
            labels: binLabels,
            datasets: [{
                label: label,
                data: binCounts,
                backgroundColor: '#1FB8CD80',
                borderColor: '#1FB8CD',
                borderWidth: 1
            }]
        };
    }

    updateChart(chartId, data) {
        if (this.charts[chartId]) {
            this.charts[chartId].data = data;
            this.charts[chartId].update();
        }
    }

    updateTable(tableId, data, columns, showRank = true) {
        const table = document.getElementById(tableId);
        if (!table) return;

        const tbody = table.querySelector('tbody');
        tbody.innerHTML = '';

        data.forEach((item, index) => {
            const row = tbody.insertRow();
            
            if (showRank) {
                row.insertCell().textContent = index + 1;
            }

            columns.forEach(col => {
                const cell = row.insertCell();
                let value = item[col];
                
                if (typeof value === 'number') {
                    if (col.includes('return') || col === 'cagr') {
                        value = (value * 100).toFixed(2) + '%';
                        cell.className = value.includes('-') ? 'negative' : 'positive';
                    } else if (col === 'max_drawdown') {
                        value = (value * 100).toFixed(2) + '%';
                        cell.className = 'negative';
                    } else if (col === 'volatility') {
                        value = (value * 100).toFixed(2) + '%';
                    } else if (col === 'sharpe' || col === 'score') {
                        value = value.toFixed(2);
                        cell.className = value > 0 ? 'positive' : 'negative';
                    } else {
                        value = value.toFixed(2);
                    }
                }
                
                cell.textContent = value;
            });
        });
    }

    applyFilters() {
        const minSharpe = parseFloat(document.getElementById('minSharpeSlider').value);
        const maxDrawdown = parseFloat(document.getElementById('maxDrawdownSlider').value) / 100;
        const minReturn = parseFloat(document.getElementById('minReturnSlider').value) / 100;

        this.filteredData = this.data.filter(pair => {
            return pair.sharpe >= minSharpe &&
                   pair.max_drawdown >= -Math.abs(maxDrawdown) &&
                   pair.total_return >= minReturn;
        });

        this.updateDashboard();
    }

    handleSearch(query) {
        const searchResults = document.getElementById('searchResults');
        
        if (query.length < 2) {
            searchResults.style.display = 'none';
            return;
        }

        const matches = this.data
            .filter(pair => pair.pair.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 10);

        if (matches.length > 0) {
            searchResults.innerHTML = matches
                .map(pair => `
                    <div class="search-result-item" data-pair="${pair.pair}">
                        ${pair.pair} - Return: ${(pair.total_return * 100).toFixed(2)}%, Sharpe: ${pair.sharpe.toFixed(2)}
                    </div>
                `).join('');
            
            searchResults.style.display = 'block';
            
            // Add click handlers
            searchResults.querySelectorAll('.search-result-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    const pair = e.target.getAttribute('data-pair');
                    document.getElementById('pairSearch').value = pair;
                    searchResults.style.display = 'none';
                });
            });
        } else {
            searchResults.innerHTML = '<div class="search-result-item">No pairs found</div>';
            searchResults.style.display = 'block';
        }
    }

    exportData(type) {
        let exportData = [];
        
        if (type === 'all') {
            exportData = this.filteredData;
        } else if (type === 'top') {
            exportData = this.filteredData
                .sort((a, b) => b.sharpe - a.sharpe)
                .slice(0, 50);
        }

        const csv = this.convertToCSV(exportData);
        this.downloadCSV(csv, `trading_pairs_${type}_${new Date().toISOString().split('T')[0]}.csv`);
    }

    convertToCSV(data) {
        const headers = ['Pair', 'Total Return %', 'CAGR %', 'Sharpe Ratio', 'Max Drawdown %', 'Volatility %', 'Category'];
        const csvRows = [headers.join(',')];
        
        data.forEach(pair => {
            const row = [
                pair.pair,
                (pair.total_return * 100).toFixed(2),
                (pair.cagr * 100).toFixed(2),
                pair.sharpe.toFixed(2),
                (pair.max_drawdown * 100).toFixed(2),
                (pair.volatility * 100).toFixed(2),
                pair.category
            ];
            csvRows.push(row.join(','));
        });
        
        return csvRows.join('\n');
    }

    downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    updateSliderValues() {
        // Initialize slider displays
        document.getElementById('topNValue').textContent = document.getElementById('topNSlider').value;
        document.getElementById('minSharpeValue').textContent = document.getElementById('minSharpeSlider').value;
        document.getElementById('maxDrawdownValue').textContent = document.getElementById('maxDrawdownSlider').value + '%';
        document.getElementById('minReturnValue').textContent = document.getElementById('minReturnSlider').value + '%';
    }

    updateKPIs() {
        const profitable = this.filteredData.filter(p => p.total_return > 0).length;
        const profitablePercentage = ((profitable / this.filteredData.length) * 100).toFixed(1);
        const avgSharpe = (this.filteredData.reduce((sum, p) => sum + p.sharpe, 0) / this.filteredData.length).toFixed(2);
        const excellent = this.filteredData.filter(p => p.category === 'Excellent').length;
        const avgVolatility = (this.filteredData.reduce((sum, p) => sum + p.volatility, 0) / this.filteredData.length * 100).toFixed(1);

        document.getElementById('totalPairs').textContent = this.filteredData.length;
        document.getElementById('profitablePairs').textContent = profitablePercentage + '%';
        document.getElementById('avgSharpe').textContent = avgSharpe;
        document.getElementById('excellentPairs').textContent = excellent;
        document.getElementById('outliers').textContent = this.outliers.length;
        document.getElementById('avgVolatility').textContent = avgVolatility + '%';
    }

    updateDashboard() {
        this.updateKPIs();
        this.updateChartsForTab(this.currentTab);
    }

    updateCharts() {
        if (this.currentTab === 'overview') {
            this.updateOverviewCharts();
        }
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});