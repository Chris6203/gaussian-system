/**
 * Shared Utility Functions - Formatting, Parsing, Helpers
 * Used by: All dashboards
 */

// Formatting utilities
const fmt = {
    /**
     * Format currency value
     * @param {number} v - Value to format
     * @param {boolean} sign - Include +/- sign
     * @returns {string} Formatted currency string
     */
    currency: (v, sign = false) => {
        if (v == null || isNaN(v)) return '--';
        const s = sign && v >= 0 ? '+' : '';
        return `${s}$${Math.abs(v).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
    },

    /**
     * Format percentage value
     * @param {number} v - Value to format (already multiplied by 100 if needed)
     * @param {boolean} sign - Include +/- sign
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted percentage string
     */
    pct: (v, sign = false, decimals = 1) => {
        if (v == null || isNaN(v)) return '--';
        const s = sign && v >= 0 ? '+' : '';
        return `${s}${v.toFixed(decimals)}%`;
    },

    /**
     * Format runtime duration
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted HH:MM:SS string
     */
    runtime: (seconds) => {
        if (seconds == null || isNaN(seconds)) return '--:--:--';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    },

    /**
     * Format date string (YYYY-MM-DD)
     * @param {string} dateStr - ISO date string
     * @returns {string} Formatted date
     */
    date: (dateStr) => {
        if (!dateStr) return '-';
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) return dateStr.split('T')[0] || '-';
            return d.toISOString().split('T')[0];
        } catch (e) {
            return dateStr.split('T')[0] || '-';
        }
    },

    /**
     * Format datetime string (YYYY-MM-DD HH:MM)
     * @param {string} dateStr - ISO date string
     * @returns {string} Formatted datetime
     */
    datetime: (dateStr) => {
        if (!dateStr) return '-';
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) return dateStr;
            return d.toISOString().slice(0, 16).replace('T', ' ');
        } catch (e) {
            return dateStr;
        }
    },

    /**
     * Format time only (HH:MM:SS)
     * @param {string} dateStr - ISO date string
     * @returns {string} Formatted time
     */
    time: (dateStr) => {
        if (!dateStr) return '-';
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) {
                // Try to extract time from string
                const match = dateStr.match(/(\d{2}:\d{2}(:\d{2})?)/);
                return match ? match[1] : '-';
            }
            return d.toTimeString().slice(0, 8);
        } catch (e) {
            return '-';
        }
    },

    /**
     * Format number with commas
     * @param {number} v - Value to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted number
     */
    number: (v, decimals = 0) => {
        if (v == null || isNaN(v)) return '--';
        return v.toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    },

    /**
     * Format compact number (1K, 1M, etc.)
     * @param {number} v - Value to format
     * @returns {string} Compact formatted number
     */
    compact: (v) => {
        if (v == null || isNaN(v)) return '--';
        if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
        if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K';
        return v.toFixed(0);
    }
};

/**
 * Parse timestamp treating it as local time (not UTC)
 * Timestamps from training are in ET - display them as-is without timezone conversion
 * @param {string} ts - Timestamp string
 * @returns {Date} Parsed date object
 */
function parseLocalTime(ts) {
    if (!ts) return new Date();
    // Replace 'T' with space to force local time interpretation
    // "2025-06-13T15:04:00" with T = parsed as UTC in many browsers
    // "2025-06-13 15:04:00" with space = parsed as local time
    const localTs = ts.replace('T', ' ');
    return new Date(localTs);
}

/**
 * Parse Eastern Time timestamp
 * @param {string} timestamp - Timestamp string in ET
 * @returns {Date} Date object
 */
function parseET(timestamp) {
    if (!timestamp) return new Date();
    // Handle various timestamp formats
    const ts = timestamp.replace('T', ' ').replace('Z', '');
    return new Date(ts);
}

/**
 * Get CSS class based on P&L value
 * @param {number} pnl - P&L value
 * @returns {string} CSS class name
 */
function getPnLClass(pnl) {
    if (pnl == null) return '';
    return pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '';
}

/**
 * Get trade type class (call/put)
 * @param {string} action - Trade action (BUY_CALL, BUY_PUT, etc.)
 * @returns {string} CSS class name
 */
function getTradeTypeClass(action) {
    if (!action) return '';
    const upper = action.toUpperCase();
    if (upper.includes('CALL')) return 'call';
    if (upper.includes('PUT')) return 'put';
    return '';
}

/**
 * Debounce function calls
 * @param {Function} fn - Function to debounce
 * @param {number} delay - Delay in ms
 * @returns {Function} Debounced function
 */
function debounce(fn, delay = 250) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn.apply(this, args), delay);
    };
}

/**
 * Throttle function calls
 * @param {Function} fn - Function to throttle
 * @param {number} limit - Time limit in ms
 * @returns {Function} Throttled function
 */
function throttle(fn, limit = 100) {
    let inThrottle;
    return (...args) => {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Sleep/delay helper
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise} Promise that resolves after delay
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Safe JSON parse
 * @param {string} str - JSON string
 * @param {*} fallback - Fallback value if parse fails
 * @returns {*} Parsed object or fallback
 */
function safeJsonParse(str, fallback = null) {
    try {
        return JSON.parse(str);
    } catch (e) {
        return fallback;
    }
}

// Export for ES modules (if supported)
if (typeof window !== 'undefined') {
    window.fmt = fmt;
    window.parseLocalTime = parseLocalTime;
    window.parseET = parseET;
    window.getPnLClass = getPnLClass;
    window.getTradeTypeClass = getTradeTypeClass;
    window.debounce = debounce;
    window.throttle = throttle;
    window.sleep = sleep;
    window.safeJsonParse = safeJsonParse;
}
