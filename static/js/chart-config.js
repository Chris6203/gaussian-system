/**
 * Chart.js Configuration Factory
 * Used by: Live, Training dashboards
 *
 * Requires: Chart.js, chartjs-adapter-date-fns, chartjs-plugin-zoom
 */

/**
 * Default Chart.js colors
 */
const chartColors = {
    primary: '#7c5cff',
    primaryDim: 'rgba(124, 92, 255, 0.1)',
    cyan: '#00c8ff',
    cyanDim: 'rgba(0, 200, 255, 0.2)',
    green: '#00d68f',
    greenDim: 'rgba(0, 214, 143, 0.15)',
    red: '#ff5c7c',
    redDim: 'rgba(255, 92, 124, 0.15)',
    yellow: '#ffc233',
    yellowDim: 'rgba(255, 194, 51, 0.2)',
    vix: 'rgba(32, 150, 90, 0.7)',
    vixDim: 'rgba(32, 150, 90, 0.08)',
    grid: 'rgba(42, 42, 58, 0.5)',
    text: '#555566',
    tooltipBg: 'rgba(18, 18, 26, 0.95)',
    tooltipText: '#f0f0f5',
    tooltipTextDim: '#8888a0',
    border: '#2a2a3a'
};

/**
 * Default tooltip configuration
 */
const defaultTooltip = {
    backgroundColor: chartColors.tooltipBg,
    titleColor: chartColors.tooltipText,
    bodyColor: chartColors.tooltipTextDim,
    borderColor: chartColors.border,
    borderWidth: 1,
    padding: 10,
    displayColors: false,
    titleFont: { family: "'JetBrains Mono', monospace" },
    bodyFont: { family: "'JetBrains Mono', monospace" }
};

/**
 * Default zoom plugin configuration
 */
const defaultZoom = {
    pan: {
        enabled: true,
        mode: 'x',
        scaleMode: 'x'
    },
    zoom: {
        wheel: { enabled: true },
        pinch: { enabled: true },
        mode: 'x'
    }
};

/**
 * Create base chart configuration
 * @param {object} options - Override options
 * @returns {object} Chart.js configuration object
 */
function createBaseChartConfig(options = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            intersect: false,
            mode: 'index'
        },
        plugins: {
            legend: { display: false },
            tooltip: { ...defaultTooltip, ...options.tooltip },
            zoom: options.enableZoom !== false ? defaultZoom : undefined
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    displayFormats: {
                        minute: 'HH:mm',
                        hour: 'HH:mm',
                        day: 'MM/dd'
                    }
                },
                grid: { color: chartColors.grid },
                ticks: {
                    color: chartColors.text,
                    maxTicksLimit: 8
                }
            },
            y: {
                position: 'left',
                grid: { color: chartColors.grid },
                ticks: {
                    color: chartColors.text,
                    callback: (v) => '$' + v.toFixed(0)
                }
            },
            ...options.scales
        },
        ...options
    };
}

/**
 * Create SPY price chart dataset
 * @param {Array} data - Price data array [{x: Date, y: number}]
 * @returns {object} Chart.js dataset
 */
function createSPYDataset(data = []) {
    return {
        label: 'SPY',
        data: data,
        borderColor: chartColors.primary,
        backgroundColor: chartColors.primaryDim,
        borderWidth: 2,
        fill: true,
        tension: 0.2,
        pointRadius: 0
    };
}

/**
 * Create VIX dataset (secondary Y-axis)
 * @param {Array} data - VIX data array
 * @returns {object} Chart.js dataset
 */
function createVIXDataset(data = []) {
    return {
        label: 'VIX',
        data: data,
        borderColor: chartColors.vix,
        backgroundColor: chartColors.vixDim,
        borderWidth: 1.8,
        fill: false,
        tension: 0.2,
        pointRadius: 0,
        yAxisID: 'y2'
    };
}

/**
 * Create TCN/LSTM prediction dataset
 * @param {Array} data - Prediction data
 * @param {boolean} isFuture - True for future predictions (dashed)
 * @returns {object} Chart.js dataset
 */
function createPredictionDataset(data = [], isFuture = true) {
    return {
        label: isFuture ? 'TCN Future' : 'TCN Past',
        data: data,
        borderColor: isFuture ? 'rgba(0, 200, 255, 0.8)' : chartColors.yellow,
        backgroundColor: isFuture ? 'transparent' : chartColors.yellowDim,
        borderWidth: 2.5,
        borderDash: isFuture ? [5, 5] : [],
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4
    };
}

/**
 * Create trade annotation (point marker)
 * @param {object} trade - Trade object with entry_time, entry_price, profit_loss
 * @returns {object} Chart.js annotation
 */
function createTradeAnnotation(trade) {
    const isWin = (trade.profit_loss || 0) > 0;
    const isCall = trade.action?.includes('CALL');

    return {
        type: 'point',
        xValue: parseLocalTime(trade.entry_time),
        yValue: trade.spy_price || trade.entry_price,
        radius: 6,
        backgroundColor: isCall ? chartColors.green : chartColors.red,
        borderColor: isWin ? chartColors.green : chartColors.red,
        borderWidth: 2
    };
}

/**
 * Create equity curve chart configuration
 * @param {Array} pnlValues - Array of cumulative P&L values
 * @param {Array} labels - Array of labels (dates/indices)
 * @returns {object} Chart.js configuration
 */
function createEquityChartConfig(pnlValues, labels) {
    const isPositive = pnlValues.length > 0 && pnlValues[pnlValues.length - 1] >= 0;
    const color = isPositive ? chartColors.green : chartColors.red;
    const colorDim = isPositive ? chartColors.greenDim : chartColors.redDim;

    return {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'P&L',
                data: pnlValues,
                borderColor: color,
                backgroundColor: colorDim,
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: defaultTooltip
            },
            scales: {
                x: {
                    display: true,
                    grid: { color: chartColors.grid },
                    ticks: { color: chartColors.text, maxTicksLimit: 6 }
                },
                y: {
                    grid: { color: chartColors.grid },
                    ticks: {
                        color: chartColors.text,
                        callback: (v) => fmt.currency(v, true)
                    }
                }
            }
        }
    };
}

/**
 * Create bar chart configuration (for distributions)
 * @param {Array} values - Array of values
 * @param {Array} labels - Array of labels
 * @param {string} colorType - 'pnl' (green/red based on value) or 'single' (all same color)
 * @returns {object} Chart.js configuration
 */
function createBarChartConfig(values, labels, colorType = 'pnl') {
    const colors = colorType === 'pnl'
        ? values.map(v => v >= 0 ? chartColors.green : chartColors.red)
        : values.map(() => chartColors.primary);

    return {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: defaultTooltip
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: chartColors.text }
                },
                y: {
                    grid: { color: chartColors.grid },
                    ticks: { color: chartColors.text }
                }
            }
        }
    };
}

/**
 * Reset chart zoom to default
 * @param {Chart} chart - Chart.js instance
 */
function resetChartZoom(chart) {
    if (chart && typeof chart.resetZoom === 'function') {
        chart.resetZoom();
    }
}

// Export for ES modules (if supported)
if (typeof window !== 'undefined') {
    window.chartColors = chartColors;
    window.createBaseChartConfig = createBaseChartConfig;
    window.createSPYDataset = createSPYDataset;
    window.createVIXDataset = createVIXDataset;
    window.createPredictionDataset = createPredictionDataset;
    window.createTradeAnnotation = createTradeAnnotation;
    window.createEquityChartConfig = createEquityChartConfig;
    window.createBarChartConfig = createBarChartConfig;
    window.resetChartZoom = resetChartZoom;
}
