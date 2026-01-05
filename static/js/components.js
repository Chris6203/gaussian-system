/**
 * Reusable Component Renderers
 * Used by: All dashboards
 *
 * Requires: utils.js (fmt, getPnLClass, getTradeTypeClass)
 */

/**
 * Render stats bar items
 * @param {Array} stats - Array of stat objects [{label, value, sub?, class?, icon?}]
 * @returns {string} HTML string
 */
function renderStatsBar(stats) {
    return stats.map(s => `
        <div class="stat-item">
            <div class="stat-label">${s.icon || ''} ${s.label}</div>
            <div class="stat-value ${s.class || ''}">${s.value}</div>
            ${s.sub ? `<div class="stat-sub">${s.sub}</div>` : ''}
        </div>
    `).join('');
}

/**
 * Render a single stat card
 * @param {string} label - Stat label
 * @param {string} value - Stat value
 * @param {string} className - Additional CSS class
 * @returns {string} HTML string
 */
function renderStatCard(label, value, className = '') {
    return `
        <div class="stat-card">
            <div class="value ${className}">${value}</div>
            <div class="label">${label}</div>
        </div>
    `;
}

/**
 * Render trade table row
 * @param {object} trade - Trade object
 * @param {object} options - Render options {showAccount, showRunId, onClick}
 * @returns {string} HTML string
 */
function renderTradeRow(trade, options = {}) {
    const isOpen = ['FILLED', 'OPEN'].includes(trade.status?.toUpperCase());
    const typeClass = getTradeTypeClass(trade.action);
    const pnlClass = getPnLClass(trade.profit_loss || trade.pnl);
    const pnl = trade.profit_loss ?? trade.pnl;

    // Format date/time
    let dateStr = '-';
    let timeStr = '';
    if (trade.entry_time || trade.timestamp) {
        const ts = trade.entry_time || trade.timestamp;
        dateStr = fmt.date(ts);
        timeStr = fmt.time(ts);
    }

    const clickAttr = options.onClick ? `onclick="${options.onClick}('${trade.id || trade.trade_id}')"` : '';
    const clickClass = options.onClick ? 'clickable' : '';

    return `<tr class="${clickClass}" ${clickAttr}>
        <td>${dateStr} ${timeStr}</td>
        ${options.showAccount ? `
            <td><span class="trade-account ${trade.is_real ? 'live' : 'paper'}">
                ${trade.is_real ? 'LIVE' : 'PAPER'}
            </span></td>
        ` : ''}
        ${options.showRunId ? `<td class="mono">${trade.run_id || '-'}</td>` : ''}
        <td><span class="trade-type ${typeClass}">${trade.action || trade.option_type || '-'}</span></td>
        <td>$${(trade.entry_price || 0).toFixed(2)}</td>
        <td>${isOpen ? '-' : '$' + (trade.exit_price || 0).toFixed(2)}</td>
        <td class="trade-pnl ${pnlClass}">${isOpen ? '-' : fmt.currency(pnl, true)}</td>
        <td>${isOpen
            ? '<span class="trade-status open">OPEN</span>'
            : `<span class="trade-status closed">${trade.exit_reason || trade.status || '-'}</span>`
        }</td>
    </tr>`;
}

/**
 * Render trades table
 * @param {Array} trades - Array of trade objects
 * @param {object} options - Table options {showAccount, showRunId, onClick, emptyMessage}
 * @returns {string} HTML string
 */
function renderTradesTable(trades, options = {}) {
    if (!trades || trades.length === 0) {
        return `<div class="empty-state">${options.emptyMessage || 'No trades yet'}</div>`;
    }

    const headers = ['Time'];
    if (options.showAccount) headers.push('Account');
    if (options.showRunId) headers.push('Run ID');
    headers.push('Type', 'Entry', 'Exit', 'P&L', 'Status');

    return `
        <table class="trades-table">
            <thead>
                <tr>
                    ${headers.map(h => `<th>${h}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${trades.map(t => renderTradeRow(t, options)).join('')}
            </tbody>
        </table>
    `;
}

/**
 * Render position card
 * @param {object} pos - Position object
 * @returns {string} HTML string
 */
function renderPositionCard(pos) {
    const isCall = pos.option_type?.toUpperCase().includes('CALL');
    const pnl = pos.unrealized_pnl || pos.pnl || 0;
    const pnlClass = getPnLClass(pnl);
    const pnlSign = pnl >= 0 ? '+' : '';

    return `
        <div class="position-card ${isCall ? 'call' : 'put'}">
            <div class="position-header">
                <span class="position-type ${isCall ? 'call' : 'put'}">
                    ${isCall ? '📈' : '📉'} ${pos.option_type || 'POS'}
                </span>
                <span class="position-entry">${fmt.time(pos.entry_time)}</span>
            </div>
            <div class="position-pnl">
                <span class="muted">$${(pos.entry_price || 0).toFixed(2)} → $${(pos.current_price || pos.current_premium || 0).toFixed(2)}</span>
                <span class="position-pnl-value ${pnlClass}">${pnlSign}$${Math.abs(pnl).toFixed(2)}</span>
            </div>
        </div>
    `;
}

/**
 * Render positions list
 * @param {Array} positions - Array of position objects
 * @returns {string} HTML string
 */
function renderPositionsList(positions) {
    if (!positions || positions.length === 0) {
        return '<div class="empty-state">No open positions</div>';
    }
    return `<div class="position-list">${positions.map(renderPositionCard).join('')}</div>`;
}

/**
 * Render LSTM/TCN predictions grid
 * @param {object} predictions - Predictions object {timeframe: {direction, return, confidence}}
 * @returns {string} HTML string
 */
function renderLSTMGrid(predictions) {
    if (!predictions || Object.keys(predictions).length === 0) {
        return '<div class="empty-state" style="grid-column: span 2;">Waiting for predictions...</div>';
    }

    return Object.entries(predictions).map(([tf, pred]) => {
        const dir = pred.direction || 'NEUTRAL';
        const dirClass = dir === 'UP' ? 'up' : dir === 'DOWN' ? 'down' : '';
        const ret = pred.predicted_return || pred.return || 0;
        const conf = pred.confidence || 0;

        return `
            <div class="lstm-item">
                <div class="lstm-tf">${tf}</div>
                <div class="lstm-dir ${dirClass}">${dir}</div>
                <div class="muted">${fmt.pct(ret * 100, true)} @ ${fmt.pct(conf * 100)}</div>
            </div>
        `;
    }).join('');
}

/**
 * Render market prices grid
 * @param {object} prices - Prices object {symbol: price}
 * @returns {string} HTML string
 */
function renderMarketGrid(prices) {
    if (!prices || Object.keys(prices).length === 0) {
        return '<div class="empty-state">Loading...</div>';
    }

    return Object.entries(prices).map(([symbol, price]) => `
        <div class="market-item">
            <div class="market-symbol">${symbol}</div>
            <div class="market-price">$${price.toFixed(2)}</div>
        </div>
    `).join('');
}

/**
 * Render signal display
 * @param {object} signal - Signal object {action, confidence}
 * @returns {string} HTML string
 */
function renderSignalDisplay(signal) {
    const action = signal?.action || 'HOLD';
    const confidence = signal?.confidence || 0;
    const actionClass = action.includes('CALL') ? 'calls' : action.includes('PUT') ? 'puts' : 'hold';
    const confColor = action === 'HOLD' ? 'var(--text-muted)' :
                     action.includes('CALL') ? 'var(--green)' : 'var(--red)';

    return `
        <div class="signal-display">
            <div class="signal-action ${actionClass}">${action}</div>
            <div class="muted">${fmt.pct(confidence * 100)} confidence</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%; background: ${confColor};"></div>
            </div>
        </div>
    `;
}

/**
 * Render pagination controls
 * @param {number} currentPage - Current page (1-indexed)
 * @param {number} totalPages - Total pages
 * @param {number} totalItems - Total items
 * @param {string} onPageChange - JS function name to call on page change
 * @returns {string} HTML string
 */
function renderPagination(currentPage, totalPages, totalItems, onPageChange = 'goToPage') {
    if (totalPages <= 1) return '';

    return `
        <div class="pagination">
            <button class="btn" onclick="${onPageChange}(1)" ${currentPage === 1 ? 'disabled' : ''}>«</button>
            <button class="btn" onclick="${onPageChange}(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>‹</button>
            <span class="pagination-info">Page ${currentPage} of ${totalPages} (${totalItems} items)</span>
            <button class="btn" onclick="${onPageChange}(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>›</button>
            <button class="btn" onclick="${onPageChange}(${totalPages})" ${currentPage === totalPages ? 'disabled' : ''}>»</button>
        </div>
    `;
}

/**
 * Render loading state
 * @param {string} message - Loading message
 * @returns {string} HTML string
 */
function renderLoading(message = 'Loading...') {
    return `
        <div class="loading">
            <div class="loading-spinner"></div>
            <span>${message}</span>
        </div>
    `;
}

/**
 * Render empty state
 * @param {string} message - Empty state message
 * @param {string} icon - Optional icon
 * @returns {string} HTML string
 */
function renderEmptyState(message, icon = '') {
    return `<div class="empty-state">${icon} ${message}</div>`;
}

/**
 * Render modal
 * @param {string} id - Modal element ID
 * @param {string} title - Modal title
 * @param {string} bodyContent - Modal body HTML
 * @param {Array} footerButtons - Array of button configs [{label, class, onClick}]
 * @returns {string} HTML string
 */
function renderModal(id, title, bodyContent, footerButtons = []) {
    const footer = footerButtons.length > 0 ? `
        <div class="modal-footer">
            ${footerButtons.map(btn => `
                <button class="btn ${btn.class || ''}" onclick="${btn.onClick}">${btn.label}</button>
            `).join('')}
        </div>
    ` : '';

    return `
        <div class="modal-overlay" id="${id}">
            <div class="modal">
                <div class="modal-header">
                    <div class="modal-title">${title}</div>
                    <button class="modal-close" onclick="closeModal('${id}')">&times;</button>
                </div>
                <div class="modal-body">
                    ${bodyContent}
                </div>
                ${footer}
            </div>
        </div>
    `;
}

/**
 * Show modal by ID
 * @param {string} id - Modal element ID
 */
function showModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add('active');
}

/**
 * Close modal by ID
 * @param {string} id - Modal element ID
 */
function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove('active');
}

// Export for ES modules (if supported)
if (typeof window !== 'undefined') {
    window.renderStatsBar = renderStatsBar;
    window.renderStatCard = renderStatCard;
    window.renderTradeRow = renderTradeRow;
    window.renderTradesTable = renderTradesTable;
    window.renderPositionCard = renderPositionCard;
    window.renderPositionsList = renderPositionsList;
    window.renderLSTMGrid = renderLSTMGrid;
    window.renderMarketGrid = renderMarketGrid;
    window.renderSignalDisplay = renderSignalDisplay;
    window.renderPagination = renderPagination;
    window.renderLoading = renderLoading;
    window.renderEmptyState = renderEmptyState;
    window.renderModal = renderModal;
    window.showModal = showModal;
    window.closeModal = closeModal;
}
