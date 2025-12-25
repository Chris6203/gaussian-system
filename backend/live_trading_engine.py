#!/usr/bin/env python3
"""
Live Trading Engine
===================

Orchestrates live trading cycles after training is complete.
Handles data fetching, signal generation, trade execution, and position management.

This module extracts the Phase 2 live trading loop from train_then_go_live.py
for better separation of concerns and testability.

Enhanced Features Support:
- Fetches all context symbols from config.json for cross-asset features
- Updates bot.market_context for enhanced feature pipeline
- Signal calibration and gating layer for improved win rate
"""

import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    # Additive sidecar for full-state visibility + missed opportunity logging.
    # Must never affect trading behavior; used with best-effort try/except guards.
    from backend.decision_pipeline import DecisionPipeline
    _HAS_DECISION_PIPELINE = True
except Exception:
    DecisionPipeline = None  # type: ignore
    _HAS_DECISION_PIPELINE = False


@dataclass
class CycleStats:
    """Statistics from a single trading cycle."""
    cycle_number: int
    timestamp: datetime
    duration_seconds: float
    current_price: Optional[float] = None
    signal_action: Optional[str] = None
    signal_confidence: Optional[float] = None
    trade_placed: bool = False
    positions_closed: int = 0
    balance: float = 0.0          # Settled cash
    equity: float = 0.0           # Cash + unrealized P&L + pending settlement
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    pending_settlement: float = 0.0
    active_positions: int = 0
    errors: List[str] = field(default_factory=list)


class LiveTradingEngine:
    """
    Orchestrates live trading cycles.
    
    Responsibilities:
    - Fetch and process market data
    - Update and track positions
    - Generate and execute signals
    - Mirror trades to shadow/live broker
    - Log cycle statistics
    """
    
    def __init__(
        self,
        bot: Any,
        config: Any,
        shadow_bridge: Any = None,
        symbol: str = None
    ):
        """
        Initialize the live trading engine.
        
        Args:
            bot: UnifiedOptionsBot instance (trained)
            config: BotConfig instance
            shadow_bridge: ShadowTradingBridge instance (or None)
            symbol: Trading symbol override (uses config if None)
        """
        self.bot = bot
        self.config = config
        self.shadow_bridge = shadow_bridge
        self.symbol = symbol or config.get_symbol()
        
        # Trading parameters from config
        self.cycle_interval = config.get('system', 'cycle_interval_seconds', default=60)
        self.historical_period = config.get('data_sources', 'historical_period', default='7d')
        
        # Load context symbols for market awareness
        self.context_symbols = self._load_context_symbols()
        self._context_fetch_errors = {}  # Track fetch errors to avoid spamming logs
        
        # Use a much lower floor (0.15) to let RL decide, but keep config as default reference
        self.min_confidence = 0.15  # Hard floor at 15% (was 40% in config)
        config_confidence = config.get('trading', 'min_confidence_threshold', default=0.4)
        logger.info(f"[ENGINE] Min confidence: {self.min_confidence:.1%} (Config was {config_confidence:.1%})")
        logger.info(f"[ENGINE] Context symbols: {', '.join(self.context_symbols)}")
        
        # Cycle tracking
        self.cycle_count = 0
        self.total_trades = 0
        self.total_positions_closed = 0
        
        # Trade-per-hour limiting
        self._trades_this_hour = 0
        self._last_trade_hour = -1
        
        # Track current VIX for volatility regime
        self.current_vix = 17.0

        # Optional observability sidecar (no behavior impact)
        self.decision_pipeline = DecisionPipeline() if _HAS_DECISION_PIPELINE else None
        
        # Log gating configuration
        # Calibration (Platt scaling) is now done inside the bot's get_calibrated_confidence()
        # Engine just applies execution gates on the already-calibrated confidence
        min_conf = config.get('calibration', 'min_confidence_threshold', default=0.55)
        max_brier = config.get('calibration', 'max_brier_score', default=0.35)
        max_ece = config.get('calibration', 'max_ece', default=0.15)
        agreement = config.get('calibration', 'agreement_horizons', default=['5min', '15min'])
        max_trades = config.get('calibration', 'max_trades_per_hour', default=5)
        
        logger.info("[ENGINE] ✅ Execution gating configured:")
        logger.info(f"[ENGINE]    - Min calibrated confidence: {min_conf:.1%}")
        logger.info(f"[ENGINE]    - Max Brier score: {max_brier:.3f}")
        logger.info(f"[ENGINE]    - Max ECE: {max_ece:.3f}")
        logger.info(f"[ENGINE]    - Horizon agreement: {agreement}")
        logger.info(f"[ENGINE]    - Max trades/hour: {max_trades}")
        logger.info("[ENGINE]    - Note: Platt calibration runs inside bot.get_calibrated_confidence()")
        
        # =============================================================================
        # HEALTH CHECKS & MODEL MONITORING
        # =============================================================================
        try:
            from backend.health_checks import HealthCheckSystem, AlertManager
            from backend.model_health import ModelHealthMonitor
            
            # Initialize alert manager (checks env vars for Slack/email)
            self.alert_manager = AlertManager(
                alert_cooldown_minutes=config.get('health', 'alert_cooldown_minutes', default=30)
            )
            
            # Initialize health check system
            self.health_check = HealthCheckSystem(
                alert_manager=self.alert_manager,
                max_data_age_seconds=config.get('health', 'max_data_age_seconds', default=300),
                min_health_score=config.get('health', 'min_health_score', default=0.5),
                min_calibration_samples=config.get('health', 'min_calibration_samples', default=50)
            )
            
            # Initialize model health monitor
            self.model_health = ModelHealthMonitor(
                error_threshold=config.get('health', 'error_threshold', default=0.02),
                drift_threshold=config.get('health', 'drift_threshold', default=0.05)
            )
            
            logger.info("[ENGINE] ✅ Health monitoring initialized")
        except ImportError as e:
            logger.warning(f"[ENGINE] Health monitoring not available: {e}")
            self.health_check = None
            self.model_health = None
            self.alert_manager = None
    
    def _load_context_symbols(self) -> List[str]:
        """
        Load context symbols from config for enhanced feature computation.
        
        These symbols are fetched alongside the primary symbol to provide
        cross-asset features, breadth indicators, macro proxies, etc.
        
        Returns:
            List of context symbol names (excludes primary trading symbol)
        """
        default_symbols = ['QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLU', 
                          'TLT', 'UUP', 'BTC-USD', '^VIX']
        
        try:
            # Try to load from config
            config_path = Path('config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                
                # Get symbols from data_fetching section
                all_symbols = cfg.get('data_fetching', {}).get('symbols', default_symbols)
                
                # Filter out primary symbol
                context = [s for s in all_symbols if s != self.symbol]
                
                return context
        except Exception as e:
            logger.warning(f"[ENGINE] Could not load context symbols from config: {e}")
        
        return default_symbols
    
    def _fetch_context_data(self) -> int:
        """
        Fetch data for all context symbols and update bot.market_context.
        
        Uses parallel fetching via AsyncOperations if available for ~4x speedup.
        
        Returns:
            Number of symbols successfully fetched
        """
        if not self.context_symbols:
            return 0
        
        fetched = 0
        
        # Try parallel fetching first (much faster for multiple symbols)
        try:
            from backend.async_operations import AsyncOperations
            
            async_ops = AsyncOperations(max_workers=min(4, len(self.context_symbols)))
            results = async_ops.fetch_multiple_symbols_sync(
                symbols=self.context_symbols,
                data_source=self.bot.data_source,
                period=self.historical_period,
                interval='1m'
            )
            
            for symbol, data in results.items():
                if data is not None and not data.empty:
                    # Normalize column names
                    data.columns = [col.lower() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] 
                                   else col.title() for col in data.columns]
                    
                    # Update bot's market context
                    if hasattr(self.bot, 'update_market_context'):
                        self.bot.update_market_context(symbol, data)
                    
                    fetched += 1
                    
                    # Reset error count
                    if symbol in self._context_fetch_errors:
                        del self._context_fetch_errors[symbol]
                else:
                    self._context_fetch_errors[symbol] = self._context_fetch_errors.get(symbol, 0) + 1
            
            if fetched > 0:
                logger.info(f"[CONTEXT] Parallel fetch: {fetched}/{len(self.context_symbols)} symbols")
            
            return fetched
            
        except ImportError:
            pass  # Fall through to sequential fetching
        except Exception as e:
            logger.debug(f"[CONTEXT] Parallel fetch failed, falling back to sequential: {e}")
        
        # Fallback: Sequential fetching
        for symbol in self.context_symbols:
            try:
                data = self.bot.data_source.get_data(
                    symbol,
                    period=self.historical_period,
                    interval='1m'
                )
                
                if data is not None and not data.empty:
                    data.columns = [col.lower() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] 
                                   else col.title() for col in data.columns]
                    
                    if hasattr(self.bot, 'update_market_context'):
                        self.bot.update_market_context(symbol, data)
                    
                    fetched += 1
                    
                    if symbol in self._context_fetch_errors:
                        del self._context_fetch_errors[symbol]
                    
                    logger.debug(f"[CONTEXT] Updated {symbol}: {len(data)} bars")
                else:
                    self._context_fetch_errors[symbol] = self._context_fetch_errors.get(symbol, 0) + 1
                    if self._context_fetch_errors[symbol] <= 3:
                        logger.warning(f"[CONTEXT] No data for {symbol}")
                        
            except Exception as e:
                self._context_fetch_errors[symbol] = self._context_fetch_errors.get(symbol, 0) + 1
                if self._context_fetch_errors[symbol] <= 3:
                    logger.warning(f"[CONTEXT] Error fetching {symbol}: {e}")
        
        if fetched > 0:
            logger.info(f"[CONTEXT] Updated {fetched}/{len(self.context_symbols)} context symbols")
        
        return fetched
    
    def run_forever(self) -> None:
        """
        Run the live trading loop indefinitely.
        
        Exits on KeyboardInterrupt or fatal error.
        """
        logger.info("[ENGINE] Starting live trading loop...")
        logger.info("[ENGINE] Press Ctrl+C to stop")
        print()
        
        self._running = True
        
        try:
            while self._running:
                cycle_start = time.time()
                self.cycle_count += 1
                
                # Run one cycle
                stats = self.run_cycle()
                
                # Log cycle summary
                self._log_cycle_summary(stats)
                
                # Wait until next cycle
                elapsed = time.time() - cycle_start
                wait_time = max(0, self.cycle_interval - elapsed)
                
                if wait_time > 0:
                    logger.info(f"[ENGINE] Waiting {wait_time:.1f}s until next cycle...")
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            logger.info("[ENGINE] Stopped by user")
            self._running = False
            self._print_final_stats()
    
    def run_cycle(self) -> CycleStats:
        """
        Execute one complete trading cycle.
        
        Returns:
            CycleStats with cycle results
        """
        cycle_start = time.time()
        stats = CycleStats(
            cycle_number=self.cycle_count,
            timestamp=datetime.now(),
            duration_seconds=0.0
        )
        
        # Log cycle header
        self._log_cycle_header()
        
        try:
            # =================================================================
            # HEALTH CHECK: Run pre-cycle health checks
            # =================================================================
            if self.health_check is not None:
                try:
                    # Get current positions count
                    positions_count = len(getattr(self.bot, 'live_positions', {}) or {})
                    
                    # Get data timestamp (approximate - will be more accurate after fetch)
                    data_ts = datetime.now() - timedelta(seconds=self.cycle_interval)
                    
                    # Get calibration metrics from bot
                    cal_metrics = None
                    if hasattr(self.bot, 'calibration_tracker') and self.bot.calibration_tracker:
                        cal_metrics = self.bot.calibration_tracker.get_metrics()
                    elif hasattr(self.bot, 'get_calibration_metrics'):
                        cal_metrics = self.bot.get_calibration_metrics()
                    
                    # Get model health
                    model_health = self.model_health.get_health_status() if self.model_health else None
                    
                    # Run health checks
                    health_result = self.health_check.run_pre_cycle_checks(
                        data_timestamp=data_ts,
                        model_health=model_health,
                        calibration_metrics=cal_metrics,
                        current_positions=positions_count
                    )
                    
                    if not health_result.can_trade:
                        # Log blocking issues but continue cycle for monitoring
                        for issue in health_result.blocking_issues:
                            logger.warning(f"[HEALTH] {issue}")
                            stats.errors.append(f"Health: {issue}")
                        print(f"  [⚠️] Health check blocked: {len(health_result.blocking_issues)} issues")
                    
                    if health_result.warnings:
                        for warn in health_result.warnings[:3]:  # Show first 3
                            logger.info(f"[HEALTH] Warning: {warn}")
                            
                except Exception as e:
                    logger.debug(f"[ENGINE] Health check error: {e}")
            
            # Step 1: Fetch primary symbol data
            data = self._fetch_data()
            
            if data is None or data.empty:
                logger.warning("[ENGINE] No data received")
                print("  [WARN] No data")
                stats.errors.append("No data received")
            else:
                current_price = float(data['close'].iloc[-1])
                stats.current_price = current_price
                
                # Log in format dashboard expects: [PRICE] SPY: $X or [SPY] $X
                logger.info(f"[PRICE] {self.symbol}: ${current_price:.2f}")
                print(f"  [DATA] {self.symbol}: ${current_price:.2f}")
                
                # Save latest price to historical database for dashboard chart
                self._save_price_to_historical_db(data)
                
                # Step 1b: Fetch context symbols for enhanced features
                context_count = self._fetch_context_data()
                if context_count > 0:
                    print(f"  [DATA] Context: {context_count} symbols updated")
                
                # Step 2: Update positions and handle exits
                closed_count = self._update_positions(current_price)
                stats.positions_closed = closed_count
                self.total_positions_closed += closed_count
                
                # Step 3: Generate signal and execute trade
                trade_result = self._generate_and_execute_signal(current_price)
                
                if trade_result:
                    stats.signal_action = trade_result.get('action')
                    stats.signal_confidence = trade_result.get('confidence')
                    stats.trade_placed = trade_result.get('trade_placed', False)
                    
                    if stats.trade_placed:
                        self.total_trades += 1
                
                # Step 4: Log status
                account = self.bot.paper_trader.get_account_summary()
                stats.balance = account.get('account_balance', self.bot.paper_trader.current_balance)
                stats.equity = account.get('total_value', stats.balance)
                stats.realized_pnl = account.get('total_pnl', 0.0)
                stats.unrealized_pnl = account.get('unrealized_pnl', 0.0)
                stats.pending_settlement = account.get('pending_settlement', 0.0)
                stats.active_positions = len([
                    t for t in self.bot.paper_trader.active_trades 
                    if t.status.value == 'FILLED'
                ])
                
                self._log_status(stats)
                
        except Exception as e:
            logger.error(f"[ENGINE] Error in cycle: {e}")
            print(f"  [ERROR] {e}")
            stats.errors.append(str(e))
            import traceback
            traceback.print_exc()
        
        stats.duration_seconds = time.time() - cycle_start
        return stats
    
    def _fetch_data(self) -> Optional[Any]:
        """
        Fetch fresh market data.
        
        Returns:
            DataFrame with market data, or None on error
        """
        logger.info(f"[ENGINE] Fetching data for {self.symbol} (period={self.historical_period})...")
        
        try:
            data = self.bot.data_source.get_data(
                self.symbol, 
                period=self.historical_period, 
                interval='1m'
            )
            
            if data is not None and not data.empty:
                # Normalize column names
                data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.error(f"[ENGINE] Error fetching data: {e}")
            return None
    
    def _save_price_to_historical_db(self, data: Any) -> None:
        """
        Save recent price data to historical database for dashboard chart.
        Only saves the last few rows to keep dashboard updated.
        """
        import sqlite3
        
        historical_db = Path('data/db/historical.db')
        if not historical_db.exists():
            return
        
        try:
            # Only save last 5 rows (most recent data)
            recent_data = data.tail(5)
            
            conn = sqlite3.connect(historical_db)
            cursor = conn.cursor()
            
            saved = 0
            for idx, row in recent_data.iterrows():
                # Get timestamp - could be index or column
                if hasattr(idx, 'isoformat'):
                    timestamp = idx.isoformat()
                else:
                    timestamp = str(idx)
                
                # Normalize timestamp format
                if 'T' not in timestamp:
                    timestamp = timestamp.replace(' ', 'T')
                
                # Get OHLCV values
                open_price = float(row.get('open', row.get('Open', 0)))
                high_price = float(row.get('high', row.get('High', 0)))
                low_price = float(row.get('low', row.get('Low', 0)))
                close_price = float(row.get('close', row.get('Close', 0)))
                volume = int(row.get('volume', row.get('Volume', 0)))
                
                # Insert or replace (upsert)
                cursor.execute('''
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (self.symbol, timestamp, open_price, high_price, low_price, close_price, volume))
                saved += 1
            
            conn.commit()
            conn.close()
            
            if saved > 0:
                logger.debug(f"[DB] Saved {saved} price points to historical database")
                
        except Exception as e:
            logger.warning(f"[DB] Could not save prices to historical: {e}")
    
    def _update_positions(self, current_price: float) -> int:
        """
        Update positions and sync any exits to Tradier.
        
        Args:
            current_price: Current underlying price
            
        Returns:
            Number of positions closed
        """
        # Track positions before update
        positions_before = self._snapshot_positions()
        
        # Update positions
        try:
            self.bot.paper_trader.update_positions(self.symbol, current_price=current_price)
            logger.info("[UPDATE] Position update completed")
        except Exception as e:
            logger.error(f"[ENGINE] Error updating positions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
        
        # Find closed positions
        positions_after = {
            t.id for t in self.bot.paper_trader.active_trades 
            if t.status.value == 'FILLED'
        }
        
        closed_positions = []
        for trade_id, meta in positions_before.items():
            if trade_id not in positions_after:
                # Get updated trade info
                for t in self.bot.paper_trader.active_trades:
                    if t.id == trade_id:
                        meta['trade'] = t
                        break
                closed_positions.append(meta)
        
        # Handle closed positions
        if closed_positions:
            closed_count = len(closed_positions)
            logger.info(f"[EXIT] Closed {closed_count} position(s) in paper trading")
            print(f"  [EXIT] Closed {closed_count} position(s)")
            
            # Issue 1 fix: Record PnL outcomes for calibration
            self._record_pnl_outcomes(closed_positions)
            
            # Sync to Tradier via shadow bridge
            if self.shadow_bridge and self.shadow_bridge.is_active:
                self.shadow_bridge.sync_closed_positions(closed_positions, self.symbol)
            
            return closed_count
        else:
            logger.info(f"[UPDATE] No positions closed (still {len(positions_after)} open)")
            return 0
    
    def _snapshot_positions(self) -> Dict[str, Dict]:
        """
        Create a snapshot of current positions with metadata.
        
        Returns:
            Dict mapping trade_id to metadata dict
        """
        positions = {}
        for trade in self.bot.paper_trader.active_trades:
            if trade.status.value == 'FILLED':
                positions[trade.id] = {
                    'trade': trade,
                    'tradier_order_id': getattr(trade, 'tradier_order_id', None),
                    'tradier_option_symbol': getattr(trade, 'tradier_option_symbol', None)
                }
                logger.info(
                    f"  Position: {trade.option_type} @ ${trade.entry_price:.2f}, "
                    f"SL: ${trade.stop_loss:.2f}, TP: ${trade.take_profit:.2f}"
                )
        return positions
    
    def _record_trade_entry(
        self, 
        trade: Any, 
        confidence: float, 
        action: str, 
        underlying_price: float
    ) -> None:
        """
        Record trade entry for PnL-based calibration.
        
        Issue 1 fix: This links confidence at entry to eventual PnL outcome.
        
        Args:
            trade: Paper trade object
            confidence: Confidence at entry time
            action: Signal action (BUY_CALLS, BUY_PUTS)
            underlying_price: Underlying price at entry
        """
        # Check if bot has calibration tracker with PnL support
        if not hasattr(self.bot, 'calibration_tracker') or self.bot.calibration_tracker is None:
            return
        
        tracker = self.bot.calibration_tracker
        
        # Check if tracker supports PnL recording (Issue 1 fix)
        if not hasattr(tracker, 'record_trade_entry'):
            return
        
        # Determine direction from action
        predicted_direction = 'UP' if 'CALL' in action else 'DOWN'
        
        # Get option price at entry
        option_price = getattr(trade, 'premium_paid', None) or getattr(trade, 'entry_price', 0)
        
        try:
            tracker.record_trade_entry(
                trade_id=trade.id,
                confidence=confidence,
                predicted_direction=predicted_direction,
                entry_price=underlying_price,
                entry_option_price=option_price,
                timestamp=datetime.now()
            )
            logger.debug(f"[PNL_CAL] Recorded entry for trade {trade.id[:8]}... | Conf: {confidence:.1%}")
        except Exception as e:
            logger.debug(f"[PNL_CAL] Could not record trade entry: {e}")
    
    def _record_pnl_outcomes(self, closed_positions: List[Dict]) -> None:
        """
        Record PnL outcomes for closed positions to calibration tracker.
        
        Issue 1 fix: This enables PnL-based calibration which is the PRIMARY
        gate for trade decisions (instead of just directional accuracy).
        
        Args:
            closed_positions: List of dicts with 'trade' objects
        """
        # Check if bot has calibration tracker with PnL support
        if not hasattr(self.bot, 'calibration_tracker') or self.bot.calibration_tracker is None:
            return
        
        tracker = self.bot.calibration_tracker
        
        # Check if tracker supports PnL recording (Issue 1 fix)
        if not hasattr(tracker, 'record_pnl_outcome'):
            return
        
        for meta in closed_positions:
            trade = meta.get('trade')
            if trade is None:
                continue
            
            # Get P&L from trade
            pnl = getattr(trade, 'profit_loss', None)
            if pnl is None:
                pnl = getattr(trade, 'pnl', 0.0)
            
            # Calculate hold time in minutes
            hold_minutes = None
            if hasattr(trade, 'entry_timestamp') and hasattr(trade, 'exit_timestamp'):
                if trade.entry_timestamp and trade.exit_timestamp:
                    hold_seconds = (trade.exit_timestamp - trade.entry_timestamp).total_seconds()
                    hold_minutes = hold_seconds / 60
            
            # Record outcome
            try:
                recorded = tracker.record_pnl_outcome(
                    trade_id=trade.id,
                    option_pnl=float(pnl) if pnl else 0.0,
                    hold_minutes=hold_minutes
                )
                
                if recorded:
                    outcome = "WIN" if pnl and pnl > 0 else "LOSS"
                    logger.debug(f"[PNL_CAL] Recorded {outcome} for trade {trade.id[:8]}... | PnL: ${pnl:.2f}")
            except Exception as e:
                logger.debug(f"[PNL_CAL] Could not record PnL outcome: {e}")
    
    def _generate_and_execute_signal(self, current_price: float) -> Optional[Dict]:
        """
        Generate trading signal and execute if conditions met.
        
        Gating checks (applied AFTER bot calibration):
        1. Minimum calibrated confidence threshold
        2. Calibration quality (Brier score, ECE) 
        3. Multi-horizon agreement (short TFs must agree)
        4. Trade-per-hour limits
        
        Calibration (Platt scaling) is done inside the bot's _combine_signals().
        
        Args:
            current_price: Current underlying price
            
        Returns:
            Dict with signal info and execution status, or None
        """
        logger.info("[ENGINE] Generating signal...")
        
        signal = self.bot.generate_unified_signal(self.symbol)
        
        if not signal:
            logger.info("[SIGNAL] No signal generated")
            return None
        
        action = signal.get('action', 'HOLD')
        # The bot's _combine_signals already applies Platt calibration via get_calibrated_confidence
        calibrated_confidence = signal.get('confidence', 0)
        raw_confidence = signal.get('raw_confidence', calibrated_confidence)
        
        # Update VIX tracking
        if hasattr(self.bot, 'current_vix'):
            self.current_vix = self.bot.current_vix
        
        result = {
            'action': action,
            'raw_confidence': raw_confidence,
            'confidence': calibrated_confidence,
            'trade_placed': False,
            'rejection_reasons': []
        }

        def _record_observation(
            *,
            proposed_action: str,
            executed_action: str,
            trade_placed: bool,
            rejection_reasons: Sequence[str],
            paper_rejection: Optional[Dict[str, Any]] = None,
        ) -> None:
            # Side-effect only; never throw.
            if not self.decision_pipeline:
                return
            try:
                self.decision_pipeline.record_decision(
                    timestamp=datetime.now(),
                    symbol=self.symbol,
                    current_price=current_price,
                    signal=signal,
                    proposed_action=proposed_action,
                    executed_action=executed_action,
                    trade_placed=trade_placed,
                    rejection_reasons=list(rejection_reasons or []),
                    paper_rejection=paper_rejection,
                    bot=self.bot,
                )
            except Exception:
                return
        
        # =========================================================================
        # GATING: Apply execution gates (bot already did calibration)
        # =========================================================================
        rejection_reasons = []
        
        # Gate 1: Basic action check
        if action == 'HOLD':
            logger.info(f"[SIGNAL] HOLD (confidence: {calibrated_confidence:.1%})")
            print(f"  [SIGNAL] HOLD ({calibrated_confidence:.1%})")
            _record_observation(
                proposed_action=action,
                executed_action="HOLD",
                trade_placed=False,
                rejection_reasons=[],
            )
            return result
        
        # Gate 2: Minimum calibrated confidence
        min_conf = self.config.get('calibration', 'min_confidence_threshold', default=0.55)
        if calibrated_confidence < min_conf:
            rejection_reasons.append(f"Confidence too low ({calibrated_confidence:.1%} < {min_conf:.1%})")
        
        # Gate 3: Calibration quality check (only if bot has enough data)
        if hasattr(self.bot, 'get_calibration_metrics'):
            cal_metrics = self.bot.get_calibration_metrics()
            result['calibration_metrics'] = cal_metrics
            
            if cal_metrics['sample_count'] >= 50:
                max_brier = self.config.get('calibration', 'max_brier_score', default=0.35)
                max_ece = self.config.get('calibration', 'max_ece', default=0.15)
                
                if cal_metrics['brier_score'] > max_brier:
                    rejection_reasons.append(f"Poor Brier ({cal_metrics['brier_score']:.3f} > {max_brier})")
                if cal_metrics['ece'] > max_ece:
                    rejection_reasons.append(f"High ECE ({cal_metrics['ece']:.3f} > {max_ece})")
                
                logger.info(f"[CALIBRATION] Brier: {cal_metrics['brier_score']:.3f} | ECE: {cal_metrics['ece']:.3f} | Acc: {cal_metrics['accuracy']:.1%} | n={cal_metrics['sample_count']}")
        
        # Gate 4: Multi-horizon agreement
        multi_tf_preds = getattr(self.bot, 'last_multi_timeframe_predictions', {}) or {}
        if self.config.get('calibration', 'require_horizon_agreement', default=True):
            agreement_horizons = self.config.get('calibration', 'agreement_horizons', default=['5min', '15min'])
            
            directions = {}
            for tf_name in agreement_horizons:
                if tf_name in multi_tf_preds:
                    pred = multi_tf_preds[tf_name]
                    ret = pred.get('predicted_return', 0)
                    directions[tf_name] = 'UP' if ret > 0.001 else 'DOWN' if ret < -0.001 else 'NEUTRAL'
            
            if len(directions) >= 2:
                unique_dirs = set(d for d in directions.values() if d != 'NEUTRAL')
                if len(unique_dirs) > 1:
                    rejection_reasons.append(f"Horizon disagreement: {directions}")
        
        # Gate 5: Trade-per-hour limit
        current_hour = datetime.now().hour
        if current_hour != self._last_trade_hour:
            self._trades_this_hour = 0
            self._last_trade_hour = current_hour
        
        max_trades = self.config.get('calibration', 'max_trades_per_hour', default=5)
        if self._trades_this_hour >= max_trades:
            rejection_reasons.append(f"Max trades/hour reached ({max_trades})")
        
        # Log signal info
        logger.info(f"[SIGNAL] {action} | Raw: {raw_confidence:.1%} → Calibrated: {calibrated_confidence:.1%}")
        print(f"  [SIGNAL] {action} (raw: {raw_confidence:.1%}, calibrated: {calibrated_confidence:.1%})")
        
        # Check if any gates rejected
        result['rejection_reasons'] = rejection_reasons
        if rejection_reasons:
            reasons_str = '; '.join(rejection_reasons[:2])
            logger.info(f"[GATE REJECT] {reasons_str}")
            print(f"  [🚫] Rejected: {reasons_str[:60]}")
            _record_observation(
                proposed_action=action,
                executed_action="HOLD",
                trade_placed=False,
                rejection_reasons=rejection_reasons,
            )
            return result
        
        # =========================================================================
        # EXECUTE TRADE
        # =========================================================================
        logger.info(f"[PAPER] Placing trade: {action}")
        paper_trade = self.bot.paper_trader.place_trade(self.symbol, signal, current_price)
        
        if not paper_trade:
            # Surface *why* the paper system rejected the trade (risk gates, etc.)
            rej = getattr(self.bot.paper_trader, 'last_trade_rejection', None)
            if isinstance(rej, dict) and rej.get('reason'):
                reason = f"{rej.get('type', 'paper_reject')}: {rej.get('code', '')} {rej.get('reason', '')}".strip()
                logger.warning(f"[WARN] Trade rejected by paper system: {reason}")
                result['paper_rejection'] = rej
                result['rejection_reasons'].append(f"PaperTrader: {reason}")
                print(f"  [!] Trade rejected: {reason[:80]}")
            else:
                logger.warning("[WARN] Trade rejected by paper system")
                print("  [!] Trade rejected")
            _record_observation(
                proposed_action=action,
                executed_action="HOLD",
                trade_placed=False,
                rejection_reasons=result.get("rejection_reasons", []) or [],
                paper_rejection=result.get("paper_rejection"),
            )
            return result
        
        # Trade placed successfully
        logger.info(f"[SIMON] SimonSays: {action}")
        logger.info(
            f"[TRADE] TRADE PLACED: {action} at ${current_price:.2f} | "
            f"Strike: ${paper_trade.strike_price:.2f} | Qty: {paper_trade.quantity}"
        )
        logger.info("[PAPER] Trade placed!")
        print(
            f"  [✓] Trade executed: {action} at ${current_price:.2f} "
            f"(Strike: ${paper_trade.strike_price:.2f})"
        )
        
        result['trade_placed'] = True
        result['paper_trade'] = paper_trade
        
        # Issue 1 fix: Record trade entry for PnL calibration
        self._record_trade_entry(paper_trade, raw_confidence, action, current_price)
        
        # Increment trade counter
        self._trades_this_hour += 1
        
        # Mirror to Tradier
        if self.shadow_bridge:
            shadow_result = self.shadow_bridge.mirror_entry(
                paper_trade, signal, self.symbol, current_price
            )
            
            if shadow_result.success:
                if shadow_result.order_id:
                    print(f"  [+] SHADOW: Mirrored to Tradier (Order: {shadow_result.order_id})")
                else:
                    print(f"  [PAPER] Paper-only mode - no live trade placed")
            else:
                print(f"  [!] SHADOW: {shadow_result.reason}")
        _record_observation(
            proposed_action=action,
            executed_action=action,
            trade_placed=True,
            rejection_reasons=[],
        )
        return result
    
    def _log_cycle_header(self) -> None:
        """Log cycle header."""
        logger.info("=" * 70)
        logger.info(f"[CYCLE {self.cycle_count}] | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        print(f"\n[LIVE CYCLE {self.cycle_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _log_status(self, stats: CycleStats) -> None:
        """Log current status."""
        max_positions = self.bot.paper_trader.max_positions
        
        logger.info(f"[BALANCE] Cash: ${stats.balance:,.2f} | Equity: ${stats.equity:,.2f}")
        logger.info(
            f"[P&L] Realized: ${stats.realized_pnl:+,.2f} | "
            f"Unrealized: ${stats.unrealized_pnl:+,.2f} | "
            f"Pending: ${stats.pending_settlement:+,.2f}"
        )
        logger.info(f"[POSITIONS] Active Positions: {stats.active_positions}/{max_positions}")
        total_pnl = stats.realized_pnl + stats.unrealized_pnl + stats.pending_settlement
        print(
            f"  [STATUS] Cash: ${stats.balance:,.2f} | Equity: ${stats.equity:,.2f} | "
            f"P&L: ${total_pnl:+,.2f} | Positions: {stats.active_positions}/{max_positions}"
        )
    
    def _log_cycle_summary(self, stats: CycleStats) -> None:
        """Log end-of-cycle summary."""
        logger.info("[SUMMARY] CYCLE SUMMARY:")
        logger.info(f"   - Completed cycles: {self.cycle_count}")
        logger.info(f"   - Cash balance: ${stats.balance:,.2f}")
        logger.info(f"   - Total equity: ${stats.equity:,.2f}")
        logger.info(
            f"   - P&L -> Realized: ${stats.realized_pnl:+,.2f} | "
            f"Unrealized: ${stats.unrealized_pnl:+,.2f} | "
            f"Pending: ${stats.pending_settlement:+,.2f}"
        )
        logger.info(f"   - Cycle duration: {stats.duration_seconds:.2f}s")
        
        # Log calibration stats periodically (using bot's built-in calibration)
        if self.cycle_count % 10 == 0:
            self._log_calibration_stats()
    
    def _log_calibration_stats(self) -> None:
        """Log calibration statistics from the bot's accuracy tracker."""
        if not hasattr(self.bot, 'get_calibration_metrics'):
            return
        
        try:
            metrics = self.bot.get_calibration_metrics()
            
            logger.info("[CALIBRATION STATS]")
            logger.info(f"   - Samples: {metrics.get('sample_count', 0)}")
            logger.info(f"   - Brier Score: {metrics.get('brier_score', 1.0):.3f} (lower is better, <0.25 = good)")
            logger.info(f"   - ECE: {metrics.get('ece', 1.0):.3f} (lower is better, <0.15 = calibrated)")
            
            # Direction accuracy (new) or legacy accuracy
            if 'direction_accuracy' in metrics:
                logger.info(f"   - Direction Accuracy: {metrics['direction_accuracy']:.1%}")
            elif 'accuracy' in metrics:
                logger.info(f"   - Accuracy: {metrics['accuracy']:.1%}")
            
            logger.info(f"   - Well Calibrated: {'✅ Yes' if metrics.get('is_calibrated', False) else '❌ No (need more data or better model)'}")
            logger.info(f"   - Trades this hour: {self._trades_this_hour}")
            
            # Log calibration method (hybrid/platt/isotonic)
            cal_method = metrics.get('calibration_method', 'unknown')
            use_hybrid = metrics.get('use_hybrid', False)
            logger.info(f"   - Calibration Method: {cal_method} (hybrid={use_hybrid})")
            
            # Log Platt params if fitted
            if metrics.get('platt_fitted', False):
                logger.info(f"   - Platt params: A={metrics.get('platt_A', 0):.4f}, B={metrics.get('platt_B', 0):.4f}")
            
            # Log Isotonic if fitted
            if metrics.get('isotonic_fitted', False):
                logger.info(f"   - Isotonic points: {metrics.get('isotonic_points', 0)}")
            
            # Log pending predictions
            pending = metrics.get('pending_count', 0)
            if pending > 0:
                logger.info(f"   - Pending predictions: {pending}")
            
            # Record calibration metric to model health
            if self.model_health is not None:
                self.model_health.record_calibration_metric(metrics.get('brier_score', 1.0))
                
        except Exception as e:
            logger.debug(f"Could not get calibration stats: {e}")
    
    def _print_final_stats(self) -> None:
        """Print final statistics when stopping."""
        account = self.bot.paper_trader.get_account_summary()
        balance = account.get('account_balance', self.bot.paper_trader.current_balance)
        equity = account.get('total_value', balance)
        realized_pnl = account.get('total_pnl', 0.0)
        unrealized_pnl = account.get('unrealized_pnl', 0.0)
        pending_settlement = account.get('pending_settlement', 0.0)
        positions = len([
            t for t in self.bot.paper_trader.active_trades 
            if t.status.value == 'FILLED'
        ])
        initial = self.bot.paper_trader.initial_balance
        total_pnl = equity - initial
        pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0
        
        print()
        print()
        print("[OK] Stopped")
        print()
        print("Final Status:")
        print(f"  Cash balance: ${balance:,.2f}")
        print(f"  Total equity: ${equity:,.2f}")
        print(f"  Realized P&L: ${realized_pnl:+,.2f}")
        print(f"  Unrealized P&L: ${unrealized_pnl:+,.2f}")
        print(f"  Pending settlement: ${pending_settlement:+,.2f}")
        print(f"  Total P&L: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"  Positions: {positions}")
        print(f"  Cycles: {self.cycle_count}")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Total Exits: {self.total_positions_closed}")
        print()
        
        # Print calibration summary (from bot's accuracy tracker)
        if hasattr(self.bot, 'get_calibration_metrics'):
            try:
                metrics = self.bot.get_calibration_metrics()
                print("Calibration Summary:")
                print(f"  Samples: {metrics['sample_count']}")
                print(f"  Brier Score: {metrics['brier_score']:.3f}")
                print(f"  ECE: {metrics['ece']:.3f}")
                print(f"  Accuracy: {metrics['accuracy']:.1%}")
                print(f"  Calibrated: {'Yes' if metrics['is_calibrated'] else 'No'}")
                if metrics.get('platt_params'):
                    print(f"  Platt: A={metrics['platt_params']['A']:.4f}, B={metrics['platt_params']['B']:.4f}")
                print()
            except Exception as e:
                logger.debug(f"Could not print calibration summary: {e}")
    
    def stop(self) -> None:
        """Stop the trading loop gracefully."""
        logger.info("[ENGINE] Stop requested")
        self._running = False


def log_subsystem_health(bot: Any, config: Any, logger_instance: Any) -> None:
    """
    Log diagnostic information about subsystems.
    
    Args:
        bot: UnifiedOptionsBot instance
        config: BotConfig instance
        logger_instance: Logger to use
    """
    # Check liquidity execution
    if getattr(bot, 'liquidity_exec', None):
        logger_instance.info(
            "[LIQUIDITY] Liquidity execution layer ACTIVE "
            "(Tradier routing via execution/liquidity_exec.py)"
        )
    else:
        logger_instance.warning(
            "[LIQUIDITY] Liquidity execution layer inactive. "
            "Confirm execution/ modules are importable and .trading_mode_config.json "
            "enables Tradier connectivity."
        )
    
    # Check HMM
    hmm_enabled = True
    if hasattr(config, 'is_hmm_enabled'):
        try:
            hmm_enabled = config.is_hmm_enabled()
        except Exception:
            hmm_enabled = True
    
    hmm_detector = getattr(bot, 'hmm_regime_detector', None)
    
    if not hmm_enabled:
        logger_instance.info("[HMM] Multi-Dimensional HMM disabled via config.json (hmm.enabled = false)")
    elif hmm_detector:
        if getattr(hmm_detector, 'is_fitted', False):
            logger_instance.info("[HMM] Multi-Dimensional HMM READY (model loaded/fitted)")
        else:
            logger_instance.info("[HMM] Multi-Dimensional HMM available but still gathering live data for training")
    else:
        logger_instance.warning(
            "[HMM] Multi-Dimensional HMM unavailable - install hmmlearn and ensure "
            "backend/multi_dimensional_hmm.py is present"
        )
    
    # Check enhanced feature pipeline
    try:
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            feature_pipeline = cfg.get('feature_pipeline', {})
            if feature_pipeline.get('enabled', True):
                context_symbols = cfg.get('data_fetching', {}).get('symbols', [])
                logger_instance.info(f"[FEATURES] Enhanced feature pipeline ENABLED ({len(context_symbols)} symbols)")
            else:
                logger_instance.info("[FEATURES] Enhanced feature pipeline disabled via config.json")
    except Exception as e:
        logger_instance.debug(f"[FEATURES] Could not check feature pipeline config: {e}")


