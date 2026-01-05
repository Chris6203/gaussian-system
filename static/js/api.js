/**
 * Dashboard API Client - Centralized fetch handling
 * Used by: All dashboards
 */

class DashboardAPI {
    /**
     * Create API client
     * @param {string} baseUrl - Base URL for API requests (default: current origin)
     */
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.defaultTimeout = 30000; // 30 seconds
    }

    /**
     * Make a fetch request with error handling
     * @param {string} endpoint - API endpoint (e.g., '/api/data')
     * @param {object} options - Fetch options
     * @returns {Promise<object>} JSON response
     */
    async fetch(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        try {
            const controller = new AbortController();
            const timeout = options.timeout || this.defaultTimeout;
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error(`[API] Request timeout: ${endpoint}`);
                throw new Error('Request timeout');
            }
            console.error(`[API] Error fetching ${endpoint}:`, error);
            throw error;
        }
    }

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {object} params - Query parameters
     * @returns {Promise<object>} JSON response
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.fetch(url, { method: 'GET' });
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @returns {Promise<object>} JSON response
     */
    async post(endpoint, data = {}) {
        return this.fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
    }

    // ========== Common Dashboard Endpoints ==========

    /** Get main dashboard data */
    async getData() {
        return this.get('/api/data');
    }

    /** Get chart data */
    async getChart() {
        return this.get('/api/chart');
    }

    /** Get trades list */
    async getTrades(params = {}) {
        return this.get('/api/trades', params);
    }

    /** Get trade details */
    async getTradeDetails(tradeId) {
        return this.get(`/api/trades/${tradeId}`);
    }

    /** Get positions */
    async getPositions() {
        return this.get('/api/positions');
    }

    /** Get balance info */
    async getBalance() {
        return this.get('/api/balance');
    }

    // ========== Hub-specific Endpoints ==========

    /** Get scoreboard data */
    async getScoreboard(params = {}) {
        return this.get('/api/scoreboard', params);
    }

    /** Get run details */
    async getRunDetails(runId) {
        return this.get(`/api/runs/${runId}`);
    }

    /** Get run trades */
    async getRunTrades(runId, params = {}) {
        return this.get(`/api/runs/${runId}/trades`, params);
    }

    /** Get P&L curve for a run */
    async getPnLCurve(runId) {
        return this.get('/api/trades/pnl-curve', { run_id: runId });
    }

    // ========== Agent API Endpoints ==========

    /** Get AI-friendly summary */
    async getAgentSummary() {
        return this.get('/api/agent/summary');
    }

    /** Get experiments for agent */
    async getAgentExperiments(params = {}) {
        return this.get('/api/agent/experiments', params);
    }

    /** Get best experiments */
    async getAgentBest(limit = 10) {
        return this.get('/api/agent/experiments/best', { limit });
    }

    /** Submit experiment idea */
    async submitIdea(idea) {
        return this.post('/api/agent/ideas', idea);
    }

    // ========== Training-specific Endpoints ==========

    /** Trigger Tradier sync */
    async syncTradier() {
        return this.post('/api/sync-tradier');
    }

    /** Get log tail */
    async getLogTail(lines = 100) {
        return this.get('/api/log', { lines });
    }

    // ========== History-specific Endpoints ==========

    /** Get model list */
    async getModels() {
        return this.get('/api/models');
    }

    /** Get model details */
    async getModelDetails(modelId) {
        return this.get(`/api/models/${modelId}`);
    }
}

// Create default instance
const api = new DashboardAPI();

// Export for ES modules (if supported)
if (typeof window !== 'undefined') {
    window.DashboardAPI = DashboardAPI;
    window.api = api;
}
