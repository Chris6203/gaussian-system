"""Dashboard HTML template - redesigned for better UX."""

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Manager</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --success: #34d399;
            --success-bg: #064e3b;
            --danger: #f87171;
            --danger-bg: #7f1d1d;
            --warning: #fbbf24;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        /* Layout */
        .app { display: flex; min-height: 100vh; }
        .sidebar {
            width: 220px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 20px 0;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }
        .main { flex: 1; margin-left: 220px; padding: 24px; }

        /* Sidebar */
        .logo { padding: 0 20px 24px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
        .logo h1 { font-size: 18px; color: var(--accent); display: flex; align-items: center; gap: 10px; }
        .logo h1::before { content: ""; font-size: 24px; }
        .nav-section { padding: 8px 12px; }
        .nav-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 1px; padding: 8px; }
        .nav-item {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 12px; margin: 2px 0;
            border-radius: 8px; cursor: pointer;
            color: var(--text-secondary);
            transition: all 0.15s;
            font-size: 14px;
        }
        .nav-item:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        .nav-item.active { background: var(--accent-hover); color: white; }
        .nav-item .icon { width: 18px; text-align: center; }

        .user-section {
            position: absolute; bottom: 0; left: 0; right: 0;
            padding: 16px; border-top: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        .user-info { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .user-avatar { width: 32px; height: 32px; border-radius: 50%; background: var(--accent-hover); display: flex; align-items: center; justify-content: center; font-size: 14px; }
        .user-email { font-size: 12px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; }
        .logout-btn {
            width: 100%; padding: 8px;
            border: 1px solid var(--border); border-radius: 6px;
            background: transparent; color: var(--text-secondary);
            font-size: 12px; cursor: pointer;
            transition: all 0.15s;
        }
        .logout-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }

        /* Page sections */
        .page { display: none; }
        .page.active { display: block; }
        .page-header { margin-bottom: 24px; }
        .page-header h2 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
        .page-header p { color: var(--text-muted); font-size: 14px; }

        /* Cards */
        .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
        .card { background: var(--bg-secondary); border-radius: 12px; padding: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .card-title { font-size: 14px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }

        /* Stats */
        .stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--bg-secondary); border-radius: 12px; padding: 20px; }
        .stat-value { font-size: 28px; font-weight: 700; color: var(--accent); }
        .stat-label { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
        .stat-change { font-size: 12px; margin-top: 8px; }
        .stat-change.positive { color: var(--success); }
        .stat-change.negative { color: var(--danger); }

        /* Forms */
        .form-group { margin-bottom: 16px; }
        .form-label { display: block; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; font-weight: 500; }
        .form-hint { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
        input, select, textarea {
            width: 100%; padding: 10px 12px;
            border: 1px solid var(--border); border-radius: 8px;
            background: var(--bg-primary); color: var(--text-primary);
            font-size: 14px; transition: border-color 0.15s;
        }
        input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
        input:disabled, select:disabled { opacity: 0.6; cursor: not-allowed; }
        input.error, select.error { border-color: var(--danger); }
        .input-group { display: flex; gap: 8px; }
        .input-group input { flex: 1; }

        /* Buttons */
        button {
            padding: 10px 16px; border: none; border-radius: 8px;
            font-size: 14px; font-weight: 500; cursor: pointer;
            transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px;
        }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-primary { background: var(--accent-hover); color: white; }
        .btn-primary:hover:not(:disabled) { background: #0369a1; }
        .btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); }
        .btn-secondary:hover:not(:disabled) { background: #475569; }
        .btn-danger { background: #dc2626; color: white; }
        .btn-danger:hover:not(:disabled) { background: #b91c1c; }
        .btn-sm { padding: 6px 12px; font-size: 12px; }
        .btn-icon { padding: 8px; min-width: 36px; justify-content: center; }

        /* Tags */
        .tag-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
        .tag {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--bg-tertiary); padding: 6px 12px;
            border-radius: 20px; font-size: 13px; font-weight: 500;
        }
        .tag .remove {
            background: none; border: none; color: var(--danger);
            cursor: pointer; padding: 0; font-size: 16px; line-height: 1;
        }
        .tag .remove:hover { color: #ef4444; }

        /* Status badges */
        .status { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; padding: 4px 10px; border-radius: 20px; font-weight: 500; }
        .status::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
        .status.active { background: var(--success-bg); color: var(--success); }
        .status.active::before { background: var(--success); }
        .status.inactive { background: var(--danger-bg); color: var(--danger); }
        .status.inactive::before { background: var(--danger); }
        .status.unknown { background: var(--bg-tertiary); color: var(--text-muted); }
        .status.unknown::before { background: var(--text-muted); }

        /* Checkbox */
        .checkbox-group { display: flex; align-items: center; gap: 8px; }
        .checkbox-group input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--accent); }
        .checkbox-group label { font-size: 14px; cursor: pointer; }

        /* Toggle */
        .toggle { position: relative; width: 44px; height: 24px; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer; inset: 0;
            background: var(--bg-tertiary); border-radius: 24px;
            transition: 0.2s;
        }
        .toggle-slider::before {
            content: ""; position: absolute;
            height: 18px; width: 18px; left: 3px; bottom: 3px;
            background: white; border-radius: 50%; transition: 0.2s;
        }
        .toggle input:checked + .toggle-slider { background: var(--accent); }
        .toggle input:checked + .toggle-slider::before { transform: translateX(20px); }

        /* Sentiment gauge */
        .sentiment-gauge { position: relative; height: 8px; background: linear-gradient(to right, #ef4444, #fbbf24, #22c55e); border-radius: 4px; margin: 16px 0; }
        .sentiment-marker {
            position: absolute; top: -4px; width: 16px; height: 16px;
            background: white; border: 2px solid var(--bg-primary);
            border-radius: 50%; transform: translateX(-50%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .sentiment-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); }

        /* Range slider */
        input[type="range"] {
            -webkit-appearance: none; height: 6px; border-radius: 3px;
            background: var(--bg-tertiary); padding: 0; margin: 8px 0;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; width: 18px; height: 18px;
            background: var(--accent); border-radius: 50%; cursor: pointer;
        }
        .range-value { text-align: center; font-size: 24px; font-weight: 600; color: var(--accent); margin-bottom: 8px; }

        /* History list */
        .history-list { max-height: 300px; overflow-y: auto; }
        .history-item {
            padding: 12px; border-bottom: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: flex-start;
        }
        .history-item:last-child { border-bottom: none; }
        .history-value { font-weight: 600; font-size: 16px; }
        .history-value.positive { color: var(--success); }
        .history-value.negative { color: var(--danger); }
        .history-meta { font-size: 12px; color: var(--text-muted); }
        .history-time { font-size: 11px; color: var(--text-muted); }
        .history-headline { font-size: 12px; color: var(--text-secondary); margin-top: 4px; font-style: italic; }

        /* Service cards */
        .service-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .service-card { background: var(--bg-primary); padding: 16px; border-radius: 8px; }
        .service-name { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
        .service-actions { margin-top: 12px; }

        /* Toast */
        .toast {
            position: fixed; bottom: 24px; right: 24px;
            padding: 14px 20px; border-radius: 10px;
            color: white; font-size: 14px; font-weight: 500;
            transform: translateY(100px); opacity: 0;
            transition: all 0.3s ease;
            display: flex; align-items: center; gap: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast.success { background: #059669; }
        .toast.error { background: #dc2626; }
        .toast.warning { background: #d97706; }

        /* Modal */
        .modal-overlay {
            position: fixed; inset: 0; background: rgba(0,0,0,0.6);
            display: none; align-items: center; justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.show { display: flex; }
        .modal {
            background: var(--bg-secondary); border-radius: 16px;
            padding: 24px; max-width: 400px; width: 90%;
        }
        .modal.modal-large { max-width: 800px; max-height: 90vh; overflow-y: auto; }
        .modal-title { font-size: 18px; font-weight: 600; margin-bottom: 12px; }
        .modal-body { color: var(--text-secondary); margin-bottom: 20px; }
        .modal-actions { display: flex; gap: 12px; justify-content: flex-end; }

        /* Config viewer */
        .config-viewer { background: var(--bg-primary); border-radius: 8px; padding: 16px; max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; }
        .config-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
        .config-tab { padding: 8px 16px; background: var(--bg-tertiary); border-radius: 6px; cursor: pointer; font-size: 13px; }
        .config-tab.active { background: var(--accent-hover); color: white; }
        .config-summary { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
        .config-summary-item { background: var(--bg-primary); padding: 12px; border-radius: 8px; }
        .config-summary-label { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
        .config-summary-value { font-weight: 600; }

        /* Loading */
        .loading { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Symbol dropdown */
        .symbol-select { position: relative; }
        .symbol-select select { padding-right: 32px; appearance: none; }
        .symbol-select::after {
            content: ""; position: absolute; right: 12px; top: 50%;
            transform: translateY(-50%);
            border: 5px solid transparent; border-top-color: var(--text-muted);
            pointer-events: none;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar { width: 60px; padding: 12px 0; }
            .sidebar .logo h1 span, .sidebar .nav-item span, .sidebar .nav-label,
            .sidebar .user-email, .sidebar .logout-btn span { display: none; }
            .sidebar .nav-item { justify-content: center; padding: 12px; }
            .sidebar .user-section { padding: 12px; }
            .sidebar .user-info { justify-content: center; margin-bottom: 0; }
            .sidebar .logout-btn { display: none; }
            .main { margin-left: 60px; padding: 16px; }
            .stat-row { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="app">
        <!-- Sidebar -->
        <nav class="sidebar">
            <div class="logo">
                <h1><span>Data Manager</span></h1>
            </div>
            <div class="nav-section">
                <div class="nav-label">Overview</div>
                <div class="nav-item active" data-page="dashboard">
                    <span class="icon">&#128200;</span><span>Dashboard</span>
                </div>
                <div class="nav-item" data-page="sentiment">
                    <span class="icon">&#128202;</span><span>Sentiment</span>
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-label">Configuration</div>
                <div class="nav-item" data-page="symbols">
                    <span class="icon">&#128176;</span><span>Symbols</span>
                </div>
                <div class="nav-item" data-page="collector">
                    <span class="icon">&#9881;</span><span>Collector</span>
                </div>
                <div class="nav-item" data-page="credentials">
                    <span class="icon">&#128273;</span><span>API Keys</span>
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-label">Tools</div>
                <div class="nav-item" data-page="backfill">
                    <span class="icon">&#128337;</span><span>Backfill</span>
                </div>
                <div class="nav-item" data-page="leaderboard">
                    <span class="icon">&#127942;</span><span>Leaderboard</span>
                </div>
                <div class="nav-item" data-page="bots">
                    <span class="icon">&#129302;</span><span>Bots</span>
                </div>
                <div class="nav-item" data-page="apikeys">
                    <span class="icon">&#128272;</span><span>Bot API Keys</span>
                </div>
            </div>
            <div class="user-section">
                <div class="user-info">
                    <div class="user-avatar">&#128100;</div>
                    <span class="user-email">{{ user }}</span>
                </div>
                <a href="/logout" class="logout-btn"><span>Sign Out</span></a>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="main">
            <!-- Dashboard Page -->
            <div id="page-dashboard" class="page active">
                <div class="page-header">
                    <h2>Dashboard</h2>
                    <p>Overview of your data collection status</p>
                </div>

                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-value" id="stat-symbols">-</div>
                        <div class="stat-label">Active Symbols</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-cycles">-</div>
                        <div class="stat-label">Cycles (24h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-price">-</div>
                        <div class="stat-label">Price Records (24h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat-liquidity">-</div>
                        <div class="stat-label">Liquidity Records (24h)</div>
                    </div>
                </div>

                <div class="card-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Services Status</span>
                            <button class="btn-secondary btn-sm" onclick="loadServiceStatus()">Refresh</button>
                        </div>
                        <div class="service-grid">
                            <div class="service-card">
                                <div class="service-name">Collector Service</div>
                                <span id="collector-status" class="status unknown">checking...</span>
                                <div class="service-actions">
                                    <button class="btn-secondary btn-sm" onclick="confirmRestart('collector')">Restart</button>
                                </div>
                            </div>
                            <div class="service-card">
                                <div class="service-name">Web Server</div>
                                <span id="web-status" class="status unknown">checking...</span>
                                <div class="service-actions">
                                    <button class="btn-secondary btn-sm" onclick="confirmRestart('web')">Restart</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Symbol Performance</span>
                        </div>
                        <div id="symbol-details" style="font-size: 13px; color: var(--text-secondary);"></div>
                    </div>
                </div>
            </div>

            <!-- Sentiment Page -->
            <div id="page-sentiment" class="page">
                <div class="page-header">
                    <h2>Sentiment Tracker</h2>
                    <p>Record and view market sentiment data</p>
                </div>

                <div class="stat-row" style="grid-template-columns: repeat(3, 1fr);">
                    <div class="stat-card">
                        <div class="stat-value" id="sentiment-count">-</div>
                        <div class="stat-label">Records (24h)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="sentiment-avg">-</div>
                        <div class="stat-label">Average Sentiment</div>
                    </div>
                    <div class="stat-card">
                        <div id="sentiment-gauge-container">
                            <div class="sentiment-gauge">
                                <div class="sentiment-marker" id="sentiment-marker" style="left: 50%;"></div>
                            </div>
                            <div class="sentiment-labels">
                                <span>Bearish</span>
                                <span>Neutral</span>
                                <span>Bullish</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Record Sentiment</span>
                        </div>
                        <div class="form-group symbol-select">
                            <label class="form-label">Symbol</label>
                            <select id="sentiment-symbol">
                                <option value="">-- Market Wide --</option>
                            </select>
                            <div class="form-hint">Select a tracked symbol or leave empty for market-wide sentiment</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Sentiment Type</label>
                            <select id="sentiment-type">
                                <option value="bullish">&#128994; Bullish</option>
                                <option value="bearish">&#128308; Bearish</option>
                                <option value="neutral">&#9898; Neutral</option>
                                <option value="fear">&#128560; Fear</option>
                                <option value="greed">&#129297; Greed</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Sentiment Value</label>
                            <div class="range-value" id="sentiment-value-display">0</div>
                            <input type="range" id="sentiment-value" min="-100" max="100" value="0" oninput="updateSentimentDisplay()">
                            <div class="sentiment-labels">
                                <span>-100 (Very Bearish)</span>
                                <span>+100 (Very Bullish)</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Notes (optional)</label>
                            <textarea id="sentiment-notes" rows="2" placeholder="Any observations or context..."></textarea>
                        </div>
                        <button class="btn-primary" onclick="saveSentiment()" style="width: 100%;">Record Sentiment</button>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Recent Sentiment</span>
                            <button class="btn-secondary btn-sm" onclick="loadSentiment()">Refresh</button>
                        </div>
                        <div id="sentiment-history" class="history-list">
                            <div style="text-align: center; padding: 20px; color: var(--text-muted);">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Symbols Page -->
            <div id="page-symbols" class="page">
                <div class="page-header">
                    <h2>Tracked Symbols</h2>
                    <p>Manage the symbols being collected</p>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Active Symbols</span>
                    </div>
                    <div id="symbol-list" class="tag-list"></div>
                    <div class="input-group" style="max-width: 400px;">
                        <input type="text" id="new-symbol" placeholder="Enter symbol (e.g., AAPL)" onkeypress="if(event.key==='Enter')addSymbol()">
                        <button class="btn-primary" onclick="addSymbol()">Add Symbol</button>
                    </div>
                    <div class="form-hint" style="margin-top: 8px;">Enter valid stock ticker symbols. Changes require collector restart.</div>
                </div>

                <div class="card" style="margin-top: 20px;">
                    <div class="card-header">
                        <span class="card-title">Options Chain Collection</span>
                    </div>
                    <div class="form-group">
                        <label class="toggle">
                            <input type="checkbox" id="options-enabled">
                            <span class="toggle-slider"></span>
                        </label>
                        <span style="margin-left: 12px;">Enable Full Chain Collection</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                        <div class="form-group">
                            <label class="form-label">Min DTE</label>
                            <input type="number" id="options-min-dte" value="0" min="0">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Max DTE</label>
                            <input type="number" id="options-max-dte" value="45" min="1">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Strike Range (%)</label>
                            <input type="number" id="options-strike-range" value="10" min="1" max="50">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Max Expirations</label>
                            <input type="number" id="options-max-expirations" value="5" min="1" max="20">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Options Symbols</label>
                        <input type="text" id="options-symbols" placeholder="SPY,QQQ">
                        <div class="form-hint">Comma-separated list of symbols for options collection</div>
                    </div>
                    <button class="btn-primary" onclick="saveOptionsChain()">Save Options Settings</button>
                </div>
            </div>

            <!-- Collector Page -->
            <div id="page-collector" class="page">
                <div class="page-header">
                    <h2>Collector Settings</h2>
                    <p>Configure data collection parameters</p>
                </div>

                <div class="card" style="max-width: 500px;">
                    <div class="card-header">
                        <span class="card-title">Collection Parameters</span>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Collection Interval (seconds)</label>
                        <input type="number" id="collector-interval" value="60" min="10" max="3600">
                        <div class="form-hint">How often to fetch new data (minimum 10 seconds)</div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Integrity Check Interval (cycles)</label>
                        <input type="number" id="collector-integrity" value="60" min="1">
                        <div class="form-hint">Run integrity check every N collection cycles</div>
                    </div>
                    <div class="form-group">
                        <label class="toggle">
                            <input type="checkbox" id="collector-liquidity" checked>
                            <span class="toggle-slider"></span>
                        </label>
                        <span style="margin-left: 12px;">Enable Options Liquidity Collection</span>
                    </div>
                    <button class="btn-primary" onclick="saveCollector()">Save Settings</button>
                </div>
            </div>

            <!-- Credentials Page -->
            <div id="page-credentials" class="page">
                <div class="page-header">
                    <h2>API Credentials</h2>
                    <p>Manage your API keys and tokens</p>
                </div>

                <div class="card-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Tradier API</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Data API Token</label>
                            <input type="password" id="tradier-data-token" placeholder="Enter token">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Access Token</label>
                            <input type="password" id="tradier-access-token" placeholder="Enter token">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Account Number</label>
                            <input type="text" id="tradier-account" placeholder="Account number">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Environment</label>
                            <select id="tradier-sandbox">
                                <option value="true">Sandbox (Testing)</option>
                                <option value="false">Live (Production)</option>
                            </select>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Polygon API</span>
                        </div>
                        <div class="form-group">
                            <label class="form-label">API Key</label>
                            <input type="password" id="polygon-key" placeholder="Enter API key">
                        </div>
                    </div>
                </div>

                <button class="btn-primary" onclick="saveCredentials()" style="margin-top: 20px;">Save All Credentials</button>
            </div>

            <!-- Backfill Page -->
            <div id="page-backfill" class="page">
                <div class="page-header">
                    <h2>Historical Backfill</h2>
                    <p>Fetch historical data for tracked symbols</p>
                </div>

                <div class="card" style="max-width: 500px;">
                    <div class="card-header">
                        <span class="card-title">Backfill Settings</span>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Days to Backfill</label>
                        <input type="number" id="backfill-days" value="7" min="1" max="365">
                        <div class="form-hint">Fetch historical data for the past N days</div>
                    </div>
                    <button class="btn-primary" onclick="confirmBackfill()">Start Backfill</button>
                </div>
            </div>

            <!-- Leaderboard Page -->
            <div id="page-leaderboard" class="page">
                <div class="page-header">
                    <h2>Bot Leaderboard</h2>
                    <p>Rankings of trading bot performance</p>
                </div>

                <div class="card" style="margin-bottom: 20px;">
                    <div class="card-header">
                        <span class="card-title">Filters</span>
                    </div>
                    <div style="display: flex; gap: 16px; align-items: flex-end;">
                        <div class="form-group" style="margin-bottom: 0;">
                            <label class="form-label">Metric</label>
                            <select id="leaderboard-metric" onchange="loadLeaderboard()">
                                <option value="total_pnl">Total P&L</option>
                                <option value="win_rate">Win Rate</option>
                            </select>
                        </div>
                        <div class="form-group" style="margin-bottom: 0;">
                            <label class="form-label">Period</label>
                            <select id="leaderboard-hours" onchange="loadLeaderboard()">
                                <option value="24">Last 24 Hours</option>
                                <option value="168">Last 7 Days</option>
                                <option value="720">Last 30 Days</option>
                            </select>
                        </div>
                        <button class="btn-secondary btn-sm" onclick="loadLeaderboard()">Refresh</button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Rankings</span>
                    </div>
                    <div id="leaderboard-table" style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                            <thead>
                                <tr style="border-bottom: 1px solid var(--border);">
                                    <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Rank</th>
                                    <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Bot Name</th>
                                    <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Owner</th>
                                    <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Type</th>
                                    <th style="padding: 12px 8px; text-align: right; color: var(--text-muted);">Score</th>
                                    <th style="padding: 12px 8px; text-align: right; color: var(--text-muted);">Trades</th>
                                </tr>
                            </thead>
                            <tbody id="leaderboard-body">
                                <tr><td colspan="6" style="padding: 40px; text-align: center; color: var(--text-muted);">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Bots Page -->
            <div id="page-bots" class="page">
                <div class="page-header">
                    <h2>Registered Bots</h2>
                    <p>View and manage all trading bots</p>
                </div>

                <div class="card">
                    <div class="card-header">
                        <span class="card-title">All Bots</span>
                        <button class="btn-secondary btn-sm" onclick="loadBots()">Refresh</button>
                    </div>
                    <div id="bots-list">
                        <div style="text-align: center; padding: 40px; color: var(--text-muted);">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- Bot API Keys Page -->
            <div id="page-apikeys" class="page">
                <div class="page-header">
                    <h2>Bot API Keys</h2>
                    <p>Manage API keys for bot authentication</p>
                </div>

                <!-- Create New Key -->
                <div class="card" style="margin-bottom: 20px;">
                    <div class="card-header">
                        <span class="card-title">Create New API Key</span>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <label class="form-label">Key Name</label>
                            <input type="text" id="new-key-name" class="form-input" placeholder="e.g., GaussianBot-Prod">
                        </div>
                        <div style="flex: 1; min-width: 200px;">
                            <label class="form-label">Permissions</label>
                            <select id="new-key-permissions" class="form-input">
                                <option value="read,write">Read & Write (Full Access)</option>
                                <option value="read">Read Only</option>
                                <option value="write">Write Only</option>
                            </select>
                        </div>
                        <button class="btn-primary" onclick="createApiKey()">Create Key</button>
                    </div>
                    <div id="new-key-result" style="margin-top: 16px; display: none;">
                        <div style="padding: 16px; background: var(--success-bg); border-radius: 8px; border: 1px solid var(--success);">
                            <div style="font-weight: 600; color: var(--success); margin-bottom: 8px;">&#10003; API Key Created!</div>
                            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">Copy this key now - it won't be shown again:</div>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <code id="new-key-value" style="flex: 1; padding: 10px; background: var(--bg-primary); border-radius: 4px; font-family: monospace; font-size: 14px; word-break: break-all;"></code>
                                <button class="btn-secondary btn-sm" onclick="copyApiKey()">Copy</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Existing Keys -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Existing API Keys</span>
                        <button class="btn-secondary btn-sm" onclick="loadApiKeys()">Refresh</button>
                    </div>
                    <div id="apikeys-list">
                        <div style="text-align: center; padding: 40px; color: var(--text-muted);">Loading...</div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Toast -->
    <div id="toast" class="toast"></div>

    <!-- Confirmation Modal -->
    <div id="modal" class="modal-overlay">
        <div class="modal">
            <div class="modal-title" id="modal-title">Confirm Action</div>
            <div class="modal-body" id="modal-body">Are you sure?</div>
            <div class="modal-actions">
                <button class="btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn-primary" id="modal-confirm">Confirm</button>
            </div>
        </div>
    </div>

    <!-- Config Viewer Modal -->
    <div id="config-modal" class="modal-overlay">
        <div class="modal modal-large">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div class="modal-title" id="config-modal-title">Bot Configuration</div>
                <button class="btn-secondary btn-sm" onclick="closeConfigModal()">&#10005;</button>
            </div>
            <div id="config-modal-meta" style="display: flex; gap: 16px; margin-bottom: 16px; font-size: 12px; color: var(--text-muted);">
            </div>
            <div class="config-tabs">
                <div class="config-tab active" onclick="showConfigTab('summary')">Summary</div>
                <div class="config-tab" onclick="showConfigTab('full')">Full Config</div>
                <div class="config-tab" onclick="showConfigTab('history')">History</div>
            </div>
            <div id="config-tab-summary" class="config-tab-content">
                <div id="config-summary-content" class="config-summary"></div>
            </div>
            <div id="config-tab-full" class="config-tab-content" style="display: none;">
                <div id="config-full-content" class="config-viewer"></div>
            </div>
            <div id="config-tab-history" class="config-tab-content" style="display: none;">
                <div id="config-history-content"></div>
            </div>
            <div class="modal-actions" style="margin-top: 20px;">
                <button class="btn-secondary" onclick="copyConfig()">Copy JSON</button>
                <button class="btn-primary" onclick="exportConfig()">Export Config</button>
            </div>
        </div>
    </div>

    <script>
        const API = '';
        let symbols = [];
        let pendingAction = null;

        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const page = item.dataset.page;
                if (!page) return;
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
                item.classList.add('active');
                document.getElementById('page-' + page).classList.add('active');
            });
        });

        // Toast
        function showToast(msg, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.className = `toast show ${type}`;
            setTimeout(() => toast.className = 'toast', 3000);
        }

        // Modal
        function showModal(title, body, onConfirm) {
            document.getElementById('modal-title').textContent = title;
            document.getElementById('modal-body').textContent = body;
            document.getElementById('modal').classList.add('show');
            pendingAction = onConfirm;
        }
        function closeModal() {
            document.getElementById('modal').classList.remove('show');
            pendingAction = null;
        }
        document.getElementById('modal-confirm').addEventListener('click', () => {
            if (pendingAction) pendingAction();
            closeModal();
        });

        // API helper
        async function apiCall(url, options = {}) {
            try {
                const res = await fetch(url, { ...options, credentials: 'same-origin' });
                if (res.status === 401) {
                    window.location.href = '/login';
                    return null;
                }
                return res;
            } catch (e) {
                showToast('Network error', 'error');
                return null;
            }
        }

        // Service Status
        async function loadServiceStatus() {
            const res = await apiCall(`${API}/api/service/status`);
            if (!res) return;
            const data = await res.json();
            updateStatus('collector-status', data.data_manager_collector);
            updateStatus('web-status', data.data_manager_web);
        }
        function updateStatus(id, status) {
            const el = document.getElementById(id);
            const statusClass = status === 'active' ? 'active' : (status === 'inactive' ? 'inactive' : 'unknown');
            el.textContent = status || 'unknown';
            el.className = `status ${statusClass}`;
        }
        function confirmRestart(service) {
            showModal('Restart Service', `Are you sure you want to restart the ${service} service?`, () => restartService(service));
        }
        async function restartService(service) {
            await apiCall(`${API}/api/service/restart`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({service})
            });
            showToast(`${service} restarting...`);
            setTimeout(loadServiceStatus, 2000);
        }

        // Stats
        async function loadStats() {
            const res = await apiCall(`${API}/api/stats`);
            if (!res) return;
            const data = await res.json();
            document.getElementById('stat-symbols').textContent = Object.keys(data.symbols || {}).length;
            document.getElementById('stat-cycles').textContent = (data.collection?.cycles_24h || 0).toLocaleString();
            document.getElementById('stat-price').textContent = (data.collection?.price_records_24h || 0).toLocaleString();
            document.getElementById('stat-liquidity').textContent = (data.collection?.liquidity_records_24h || 0).toLocaleString();

            const details = Object.entries(data.symbols || {}).map(([sym, info]) =>
                `<div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span style="font-weight: 500;">${sym}</span>
                    <span style="color: var(--text-muted);">${info.records.toLocaleString()} records</span>
                </div>`
            ).join('');
            document.getElementById('symbol-details').innerHTML = details || '<div style="color: var(--text-muted);">No data yet</div>';
        }

        // Symbols
        async function loadSymbols() {
            const res = await apiCall(`${API}/api/symbols`);
            if (!res) return;
            const data = await res.json();
            symbols = data.symbols || [];
            renderSymbols();
            updateSymbolDropdowns();
        }
        function renderSymbols() {
            document.getElementById('symbol-list').innerHTML = symbols.length ? symbols.map(s => `
                <span class="tag">${s}<button class="remove" onclick="confirmRemoveSymbol('${s}')">&times;</button></span>
            `).join('') : '<div style="color: var(--text-muted);">No symbols added yet</div>';
        }
        function updateSymbolDropdowns() {
            const select = document.getElementById('sentiment-symbol');
            const current = select.value;
            select.innerHTML = '<option value="">-- Market Wide --</option>' +
                symbols.map(s => `<option value="${s}">${s}</option>`).join('');
            select.value = current;
        }
        async function addSymbol() {
            const input = document.getElementById('new-symbol');
            const sym = input.value.trim().toUpperCase();
            if (!sym) {
                showToast('Please enter a symbol', 'error');
                return;
            }
            if (!/^[A-Z]{1,5}$/.test(sym) && !/^\\^[A-Z]{2,5}$/.test(sym)) {
                showToast('Invalid symbol format', 'error');
                return;
            }
            if (symbols.includes(sym)) {
                showToast('Symbol already exists', 'warning');
                return;
            }
            symbols.push(sym);
            input.value = '';
            await saveSymbols();
        }
        function confirmRemoveSymbol(sym) {
            showModal('Remove Symbol', `Remove ${sym} from tracked symbols?`, () => removeSymbol(sym));
        }
        async function removeSymbol(sym) {
            symbols = symbols.filter(s => s !== sym);
            await saveSymbols();
        }
        async function saveSymbols() {
            await apiCall(`${API}/api/symbols`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbols})
            });
            renderSymbols();
            updateSymbolDropdowns();
            showToast('Symbols saved! Restart collector to apply.');
        }

        // Collector
        async function loadCollector() {
            const res = await apiCall(`${API}/api/collector`);
            if (!res) return;
            const data = await res.json();
            document.getElementById('collector-interval').value = data.interval_seconds || 60;
            document.getElementById('collector-integrity').value = data.integrity_check_interval_cycles || 60;
            document.getElementById('collector-liquidity').checked = data.enable_options_liquidity !== false;
        }
        async function saveCollector() {
            const interval = parseInt(document.getElementById('collector-interval').value);
            if (interval < 10) {
                showToast('Interval must be at least 10 seconds', 'error');
                return;
            }
            await apiCall(`${API}/api/collector`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    interval_seconds: interval,
                    integrity_check_interval_cycles: parseInt(document.getElementById('collector-integrity').value),
                    enable_options_liquidity: document.getElementById('collector-liquidity').checked
                })
            });
            showToast('Settings saved! Restart collector to apply.');
        }

        // Credentials
        async function loadCredentials() {
            const res = await apiCall(`${API}/api/credentials`);
            if (!res) return;
            const data = await res.json();
            document.getElementById('tradier-data-token').placeholder = data.tradier?.data_api_token || 'Not configured';
            document.getElementById('tradier-access-token').placeholder = data.tradier?.access_token || 'Not configured';
            document.getElementById('tradier-account').value = data.tradier?.account_number || '';
            document.getElementById('tradier-sandbox').value = data.tradier?.is_sandbox || 'true';
            document.getElementById('polygon-key').placeholder = data.polygon?.api_key || 'Not configured';
        }
        async function saveCredentials() {
            await apiCall(`${API}/api/credentials`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    tradier: {
                        data_api_token: document.getElementById('tradier-data-token').value,
                        access_token: document.getElementById('tradier-access-token').value,
                        account_number: document.getElementById('tradier-account').value,
                        is_sandbox: document.getElementById('tradier-sandbox').value
                    },
                    polygon: {
                        api_key: document.getElementById('polygon-key').value
                    }
                })
            });
            showToast('Credentials saved! Restart services to apply.');
            loadCredentials();
        }

        // Options Chain
        async function loadOptionsChain() {
            const res = await apiCall(`${API}/api/options_chain`);
            if (!res) return;
            const data = await res.json();
            document.getElementById('options-enabled').checked = data.enabled || false;
            document.getElementById('options-min-dte').value = data.min_dte || 0;
            document.getElementById('options-max-dte').value = data.max_dte || 45;
            document.getElementById('options-strike-range').value = data.strike_range_pct || 10;
            document.getElementById('options-max-expirations').value = data.expirations_to_collect || 5;
            document.getElementById('options-symbols').value = (data.symbols || ['SPY']).join(', ');
        }
        async function saveOptionsChain() {
            const symbolsStr = document.getElementById('options-symbols').value;
            const optSymbols = symbolsStr.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
            await apiCall(`${API}/api/options_chain`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    enabled: document.getElementById('options-enabled').checked,
                    min_dte: parseInt(document.getElementById('options-min-dte').value),
                    max_dte: parseInt(document.getElementById('options-max-dte').value),
                    strike_range_pct: parseFloat(document.getElementById('options-strike-range').value),
                    expirations_to_collect: parseInt(document.getElementById('options-max-expirations').value),
                    collect_greeks: true,
                    symbols: optSymbols.length ? optSymbols : ['SPY']
                })
            });
            showToast('Options settings saved! Restart collector to apply.');
        }

        // Backfill
        function confirmBackfill() {
            const days = document.getElementById('backfill-days').value;
            showModal('Start Backfill', `This will fetch historical data for the past ${days} days. Continue?`, startBackfill);
        }
        async function startBackfill() {
            const days = parseInt(document.getElementById('backfill-days').value);
            await apiCall(`${API}/api/backfill`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({days})
            });
            showToast(`Backfill started for ${days} days`);
        }

        // Sentiment
        function updateSentimentDisplay() {
            const value = document.getElementById('sentiment-value').value;
            const display = document.getElementById('sentiment-value-display');
            display.textContent = (value > 0 ? '+' : '') + value;
            display.style.color = value > 0 ? 'var(--success)' : value < 0 ? 'var(--danger)' : 'var(--text-muted)';
        }
        async function loadSentiment() {
            // Summary
            const summaryRes = await apiCall(`${API}/api/sentiment/summary`);
            if (summaryRes) {
                const summary = await summaryRes.json();
                document.getElementById('sentiment-count').textContent = summary.total_records || 0;
                const avg = summary.avg_value;
                document.getElementById('sentiment-avg').textContent = avg !== null ? (avg > 0 ? '+' : '') + avg : '-';
                document.getElementById('sentiment-avg').style.color = avg > 0 ? 'var(--success)' : avg < 0 ? 'var(--danger)' : '';

                // Update gauge
                const markerPos = avg !== null ? ((avg + 100) / 200) * 100 : 50;
                document.getElementById('sentiment-marker').style.left = markerPos + '%';
            }

            // History
            const historyRes = await apiCall(`${API}/api/sentiment?limit=15`);
            if (historyRes) {
                const data = await historyRes.json();
                const el = document.getElementById('sentiment-history');
                if (data.records && data.records.length > 0) {
                    el.innerHTML = data.records.map(r => {
                        const time = new Date(r.timestamp).toLocaleString();
                        const valueClass = r.value > 0 ? 'positive' : r.value < 0 ? 'negative' : '';
                        return `<div class="history-item">
                            <div>
                                <div class="history-value ${valueClass}">${r.value > 0 ? '+' : ''}${r.value}</div>
                                <div class="history-meta">${r.sentiment_type}${r.symbol ? ' &bull; ' + r.symbol : ''}${r.source ? ' &bull; ' + r.source : ''}</div>
                                ${r.headline ? `<div class="history-headline">"${r.headline.substring(0, 60)}${r.headline.length > 60 ? '...' : ''}"</div>` : ''}
                                ${r.notes ? `<div class="history-headline">${r.notes}</div>` : ''}
                            </div>
                            <div class="history-time">${time}</div>
                        </div>`;
                    }).join('');
                } else {
                    el.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--text-muted);">No sentiment records yet</div>';
                }
            }
        }
        async function saveSentiment() {
            const value = parseInt(document.getElementById('sentiment-value').value);
            const symbol = document.getElementById('sentiment-symbol').value;

            // Validate symbol if provided
            if (symbol && !symbols.includes(symbol)) {
                showToast('Please select a valid symbol', 'error');
                return;
            }

            const res = await apiCall(`${API}/api/sentiment`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbol: symbol || null,
                    sentiment_type: document.getElementById('sentiment-type').value,
                    value: value,
                    notes: document.getElementById('sentiment-notes').value || null,
                    source: 'dashboard'
                })
            });
            if (res && res.ok) {
                showToast('Sentiment recorded');
                document.getElementById('sentiment-notes').value = '';
                document.getElementById('sentiment-value').value = 0;
                updateSentimentDisplay();
                loadSentiment();
            } else {
                showToast('Error recording sentiment', 'error');
            }
        }

        // Leaderboard
        async function loadLeaderboard() {
            const metric = document.getElementById('leaderboard-metric').value;
            const hours = document.getElementById('leaderboard-hours').value;
            const res = await apiCall(`${API}/api/leaderboard?metric=${metric}&hours=${hours}`);
            if (!res) return;
            const data = await res.json();
            const tbody = document.getElementById('leaderboard-body');

            if (!data.leaderboard || data.leaderboard.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="padding: 40px; text-align: center; color: var(--text-muted);">No bots registered yet</td></tr>';
                return;
            }

            tbody.innerHTML = data.leaderboard.map((bot, i) => {
                const rankBadge = i === 0 ? '&#129351;' : i === 1 ? '&#129352;' : i === 2 ? '&#129353;' : bot.rank;
                const score = metric === 'total_pnl'
                    ? `<span style="color: ${bot.total_pnl >= 0 ? 'var(--success)' : 'var(--danger)'}">$${bot.total_pnl.toLocaleString()}</span>`
                    : `${bot.win_rate}%`;
                return `<tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 12px 8px; font-size: 18px;">${rankBadge}</td>
                    <td style="padding: 12px 8px; font-weight: 600;">${bot.name}</td>
                    <td style="padding: 12px 8px; color: var(--text-muted);">${bot.owner}</td>
                    <td style="padding: 12px 8px;"><span style="background: var(--bg-tertiary); padding: 4px 8px; border-radius: 4px; font-size: 12px;">${bot.bot_type}</span></td>
                    <td style="padding: 12px 8px; text-align: right; font-weight: 600;">${score}</td>
                    <td style="padding: 12px 8px; text-align: right; color: var(--text-muted);">${bot.trade_count}</td>
                </tr>`;
            }).join('');
        }

        // Bots
        async function loadBots() {
            const res = await apiCall(`${API}/api/bots`);
            if (!res) return;
            const data = await res.json();
            const container = document.getElementById('bots-list');

            if (!data.bots || data.bots.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-muted);">No bots registered yet. Bots can register via the API.</div>';
                return;
            }

            container.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">ID</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Name</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Owner</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Type</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Config</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Status</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.bots.map(bot => {
                        const statusClass = bot.status === 'active' ? 'active' : 'inactive';
                        const created = new Date(bot.created_at).toLocaleDateString();
                        const hasConfig = bot.config_hash ? true : false;
                        const configBadge = hasConfig
                            ? `<span style="background: var(--success-bg); color: var(--success); padding: 4px 8px; border-radius: 4px; font-size: 11px;">v${bot.config_version || 1}</span>`
                            : '<span style="color: var(--text-muted); font-size: 11px;">None</span>';
                        return `<tr style="border-bottom: 1px solid var(--border);">
                            <td style="padding: 12px 8px; color: var(--text-muted);">#${bot.id}</td>
                            <td style="padding: 12px 8px; font-weight: 600;">${bot.name}</td>
                            <td style="padding: 12px 8px; color: var(--text-muted);">${bot.owner}</td>
                            <td style="padding: 12px 8px;"><span style="background: var(--bg-tertiary); padding: 4px 8px; border-radius: 4px; font-size: 12px;">${bot.bot_type}</span></td>
                            <td style="padding: 12px 8px;">${configBadge}</td>
                            <td style="padding: 12px 8px;"><span class="status ${statusClass}">${bot.status}</span></td>
                            <td style="padding: 12px 8px;">
                                ${hasConfig ? `<button class="btn-secondary btn-sm" style="margin-right: 4px;" onclick="viewBotConfig(${bot.id}, '${bot.name}')">View Config</button>` : ''}
                                ${hasConfig ? `<button class="btn-secondary btn-sm" onclick="quickExportConfig(${bot.id})">Export</button>` : ''}
                            </td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>`;
        }

        // API Keys Management
        async function loadApiKeys() {
            const res = await apiCall(`${API}/api/keys`);
            if (!res) return;
            const data = await res.json();
            const container = document.getElementById('apikeys-list');

            if (!data.keys || data.keys.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-muted);">No API keys yet. Create one above to get started.</div>';
                return;
            }

            container.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">ID</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Name</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Permissions</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Created</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Last Used</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Status</th>
                        <th style="padding: 12px 8px; text-align: left; color: var(--text-muted);">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.keys.map(key => {
                        const statusClass = key.active ? 'active' : 'inactive';
                        const statusText = key.active ? 'Active' : 'Revoked';
                        const created = new Date(key.created_at).toLocaleDateString();
                        const lastUsed = key.last_used ? new Date(key.last_used).toLocaleString() : 'Never';
                        return `<tr style="border-bottom: 1px solid var(--border);">
                            <td style="padding: 12px 8px; color: var(--text-muted);">#${key.id}</td>
                            <td style="padding: 12px 8px; font-weight: 600;">${key.name}</td>
                            <td style="padding: 12px 8px;"><span style="background: var(--bg-tertiary); padding: 4px 8px; border-radius: 4px; font-size: 12px;">${key.permissions}</span></td>
                            <td style="padding: 12px 8px; color: var(--text-muted);">${created}</td>
                            <td style="padding: 12px 8px; color: var(--text-muted); font-size: 12px;">${lastUsed}</td>
                            <td style="padding: 12px 8px;"><span class="status ${statusClass}">${statusText}</span></td>
                            <td style="padding: 12px 8px;">
                                ${key.active ? `<button class="btn-secondary btn-sm" style="margin-right: 4px;" onclick="revokeApiKey(${key.id})">Revoke</button>` : ''}
                                <button class="btn-secondary btn-sm" style="color: var(--danger);" onclick="deleteApiKey(${key.id})">Delete</button>
                            </td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>`;
        }

        async function createApiKey() {
            const name = document.getElementById('new-key-name').value.trim();
            const permissions = document.getElementById('new-key-permissions').value;

            if (!name) {
                showToast('Please enter a key name', 'error');
                return;
            }

            const res = await apiCall(`${API}/api/keys`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, permissions })
            });

            if (!res) return;
            const data = await res.json();

            if (data.ok && data.key) {
                document.getElementById('new-key-value').textContent = data.key;
                document.getElementById('new-key-result').style.display = 'block';
                document.getElementById('new-key-name').value = '';
                showToast('API key created successfully!', 'success');
                loadApiKeys();
            } else {
                showToast(data.error || 'Failed to create key', 'error');
            }
        }

        function copyApiKey() {
            const key = document.getElementById('new-key-value').textContent;
            navigator.clipboard.writeText(key).then(() => {
                showToast('API key copied to clipboard!', 'success');
            }).catch(() => {
                showToast('Failed to copy', 'error');
            });
        }

        async function revokeApiKey(keyId) {
            showModal('Revoke API Key', 'Are you sure you want to revoke this key? It will no longer be able to authenticate.', async () => {
                const res = await apiCall(`${API}/api/keys/${keyId}/revoke`, { method: 'POST' });
                if (res && res.ok) {
                    showToast('API key revoked', 'success');
                    loadApiKeys();
                } else {
                    showToast('Failed to revoke key', 'error');
                }
            });
        }

        async function deleteApiKey(keyId) {
            showModal('Delete API Key', 'Are you sure you want to permanently delete this key?', async () => {
                const res = await apiCall(`${API}/api/keys/${keyId}`, { method: 'DELETE' });
                if (res && res.ok) {
                    showToast('API key deleted', 'success');
                    loadApiKeys();
                } else {
                    showToast('Failed to delete key', 'error');
                }
            });
        }

        // Config Viewer
        let currentBotId = null;
        let currentConfig = null;

        async function viewBotConfig(botId, botName) {
            currentBotId = botId;
            document.getElementById('config-modal-title').textContent = `Configuration: ${botName}`;
            document.getElementById('config-modal').classList.add('show');

            // Reset tabs
            showConfigTab('summary');

            // Fetch config
            const res = await apiCall(`${API}/api/v1/bots/${botId}/config`);
            if (!res) return;
            const data = await res.json();
            currentConfig = data.config;

            // Show meta info
            document.getElementById('config-modal-meta').innerHTML = `
                <span>Version: <strong>v${data.config_version || 1}</strong></span>
                <span>Hash: <code style="background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px;">${(data.config_hash || '').substring(0, 8)}</code></span>
            `;

            // Show summary
            const summary = data.config_summary || {};
            const summaryHtml = Object.entries(summary).map(([key, value]) => `
                <div class="config-summary-item">
                    <div class="config-summary-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                    <div class="config-summary-value">${typeof value === 'object' ? JSON.stringify(value) : value}</div>
                </div>
            `).join('');
            document.getElementById('config-summary-content').innerHTML = summaryHtml || '<div style="color: var(--text-muted);">No summary available</div>';

            // Show full config
            document.getElementById('config-full-content').textContent = JSON.stringify(data.config, null, 2);
        }

        async function loadConfigHistory() {
            if (!currentBotId) return;

            const res = await apiCall(`${API}/api/v1/bots/${currentBotId}/config/history?limit=10`);
            if (!res) return;
            const data = await res.json();

            const historyHtml = data.history && data.history.length > 0
                ? data.history.map(v => {
                    const timestamp = new Date(v.timestamp).toLocaleString();
                    return `<div style="padding: 12px; background: var(--bg-primary); border-radius: 8px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span><strong>Version ${v.config_version}</strong></span>
                            <span style="font-size: 12px; color: var(--text-muted);">${timestamp}</span>
                        </div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                            Hash: <code>${(v.config_hash || '').substring(0, 8)}</code>
                            ${v.previous_hash ? ` (from ${v.previous_hash.substring(0, 8)})` : ''}
                        </div>
                    </div>`;
                }).join('')
                : '<div style="color: var(--text-muted);">No version history available</div>';

            document.getElementById('config-history-content').innerHTML = historyHtml;
        }

        function showConfigTab(tab) {
            document.querySelectorAll('.config-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.config-tab-content').forEach(c => c.style.display = 'none');

            document.querySelector(`.config-tab[onclick="showConfigTab('${tab}')"]`).classList.add('active');
            document.getElementById(`config-tab-${tab}`).style.display = 'block';

            if (tab === 'history') {
                loadConfigHistory();
            }
        }

        function closeConfigModal() {
            document.getElementById('config-modal').classList.remove('show');
            currentBotId = null;
            currentConfig = null;
        }

        function copyConfig() {
            if (!currentConfig) {
                showToast('No config to copy', 'error');
                return;
            }
            navigator.clipboard.writeText(JSON.stringify(currentConfig, null, 2)).then(() => {
                showToast('Config copied to clipboard!', 'success');
            }).catch(() => {
                showToast('Failed to copy', 'error');
            });
        }

        async function exportConfig() {
            if (!currentBotId) return;

            const res = await apiCall(`${API}/api/v1/bots/${currentBotId}/config/export`);
            if (!res) return;
            const data = await res.json();

            // Create download
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bot_${currentBotId}_config.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Config exported!', 'success');
        }

        async function quickExportConfig(botId) {
            const res = await apiCall(`${API}/api/v1/bots/${botId}/config/export`);
            if (!res) return;
            const data = await res.json();

            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bot_${botId}_config.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Config exported!', 'success');
        }

        // Init
        loadServiceStatus();
        loadStats();
        loadSymbols();
        loadCollector();
        loadCredentials();
        loadOptionsChain();
        loadSentiment();
        loadLeaderboard();
        loadBots();
        loadApiKeys();

        setInterval(loadServiceStatus, 10000);
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
'''
