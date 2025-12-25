"""
Consensus Entry Controller - Multi-Signal Agreement for Entry Decisions

Only trades when ALL signals agree:
1. Multi-timeframe agreement (15m, 30m, 1h all same direction)
2. HMM trend alignment (hard veto if misaligned)
3. Momentum confirmation (momentum + jerk + RSI)
4. Volatility filter (VIX range, volume spike)
5. Technical confirmation (MACD, Bollinger Bands, Market Breadth)
6. Straddle detection (high volatility + no clear direction) [NEW]

Target: Improve win rate from ~17% to 40-60% by filtering harder.

Configuration Priority:
1. Environment variables (for quick testing overrides)
2. config.json entry_controller.consensus section (primary config)
3. Default values (fallback)
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _load_config_from_file(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json file."""
    try:
        # Try relative path first, then absolute
        paths_to_try = [
            config_path,
            os.path.join(os.path.dirname(__file__), '..', config_path),
            os.path.join('E:/gaussian/output3', config_path),
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return {}


class ConsensusEntryController:
    """
    Strict multi-signal consensus for entry decisions.
    Only trades when ALL signals agree.

    Configuration is loaded from config.json entry_controller.consensus section.
    Environment variables can override for testing.
    """

    def __init__(self, config: Optional[Dict] = None):
        # Load config from file if not provided
        if config is None:
            config = _load_config_from_file()

        # Get entry controller config section
        entry_cfg = config.get('entry_controller', {})
        consensus_cfg = entry_cfg.get('consensus', {})

        logger.info("[CONSENSUS] Loading configuration from config.json...")

        # Timeframe agreement settings
        tf_cfg = consensus_cfg.get('timeframe_agreement', {})
        self.required_timeframes = tf_cfg.get('required_timeframes', ['15min', '30min', '1h'])
        self.min_timeframe_agreement = tf_cfg.get('min_agreement', 0.67)  # 2/3 must agree
        self.min_weighted_confidence = consensus_cfg.get('min_weighted_confidence', 0.55)

        # HMM alignment settings
        hmm_cfg = consensus_cfg.get('hmm_alignment', {})
        self.hmm_bullish_threshold = hmm_cfg.get('bullish_threshold', 0.6)
        self.hmm_bearish_threshold = hmm_cfg.get('bearish_threshold', 0.4)
        self.min_hmm_confidence = hmm_cfg.get('min_confidence', 0.50)

        # Momentum confirmation settings
        mom_cfg = consensus_cfg.get('momentum', {})
        self.require_momentum_alignment = mom_cfg.get('require_alignment', True)
        self.require_jerk_confirmation = mom_cfg.get('require_jerk', False)
        self.rsi_overbought = mom_cfg.get('rsi_overbought', 70)
        self.rsi_oversold = mom_cfg.get('rsi_oversold', 30)

        # Volatility filter settings
        vol_cfg = consensus_cfg.get('volatility_filter', {})
        self.vix_min = vol_cfg.get('vix_min', 12)
        self.vix_max = vol_cfg.get('vix_max', 35)
        self.max_volume_spike = vol_cfg.get('max_volume_spike', 3.0)

        # Technical confirmation settings (Signal 5)
        self.require_technical_confirmation = True
        self.macd_threshold = 0.0
        self.bb_extreme_low = 0.15
        self.bb_extreme_high = 0.85
        self.require_market_breadth = False
        self.require_hma_confirmation = True

        # Straddle settings
        self.enable_straddles = config.get('trading', {}).get('enable_straddles', True)
        self.straddle_vix_min = 20.0
        self.straddle_vix_max = 40.0
        self.straddle_hmm_vol_min = 0.6
        self.straddle_direction_conflict_required = True
        self.straddle_confidence_max = 0.40

        # Options flow settings (Signal 6)
        flow_cfg = consensus_cfg.get('options_flow', {})
        self.require_options_flow = False
        self.options_flow_boost = True
        self.options_flow_veto_strength = 0.7
        self.put_call_bullish_threshold = flow_cfg.get('put_call_bullish_threshold', 0.7)
        self.put_call_bearish_threshold = flow_cfg.get('put_call_bearish_threshold', 1.3)

        # Mean Reversion settings (Signal 7)
        mr_cfg = consensus_cfg.get('mean_reversion', {})
        self.enable_mean_reversion = consensus_cfg.get('enable_mean_reversion', True)
        self.mr_rsi_oversold = mr_cfg.get('rsi_oversold', 25)
        self.mr_rsi_overbought = mr_cfg.get('rsi_overbought', 75)
        self.mr_bb_oversold = mr_cfg.get('bb_oversold', 0.10)
        self.mr_bb_overbought = mr_cfg.get('bb_overbought', 0.90)
        self.mr_confidence_boost = mr_cfg.get('confidence_boost', 0.15)

        # Regime-Specific Trading settings (Approach 3)
        regime_cfg = consensus_cfg.get('regime_filter', {})
        self.enable_regime_filter = consensus_cfg.get('enable_regime_filter', True)
        self.regime_trend_strength_min = regime_cfg.get('trend_strength_min', 0.15)
        self.regime_avoid_choppy = regime_cfg.get('avoid_choppy', True)
        self.regime_choppy_vol_threshold = regime_cfg.get('choppy_vol_threshold', 0.6)
        self.regime_require_liquidity = regime_cfg.get('require_liquidity', True)
        self.regime_liquidity_min = regime_cfg.get('liquidity_min', 0.3)
        self.regime_confidence_min = regime_cfg.get('confidence_min', 0.4)
        self.regime_trend_alignment_bonus = 0.10

        # Advanced Features settings (Approach 4)
        adv_cfg = consensus_cfg.get('advanced_features', {})
        self.enable_advanced_features = consensus_cfg.get('enable_advanced_features', True)
        self.advanced_veto_strength = adv_cfg.get('veto_strength', 0.6)
        self.advanced_boost_strength = adv_cfg.get('boost_strength', 0.5)
        self.advanced_confidence_boost = adv_cfg.get('confidence_boost', 0.10)

        # CONTRARIAN MODE (NEW) - Trade AGAINST consensus signals
        # When all signals agree, the move has already happened - we're late
        # Instead: trade when signals DISAGREE (early entry on reversals)
        self.contrarian_mode = consensus_cfg.get('contrarian_mode', False)
        self.contrarian_min_disagreement = consensus_cfg.get('contrarian_min_disagreement', 2)
        self.contrarian_hmm_override = consensus_cfg.get('contrarian_hmm_override', True)

        # Log loaded configuration
        logger.info(f"   entry_controller.type: {entry_cfg.get('type', 'consensus')}")
        logger.info(f"   contrarian_mode: {self.contrarian_mode}")
        logger.info(f"   enable_mean_reversion: {self.enable_mean_reversion}")
        logger.info(f"   enable_regime_filter: {self.enable_regime_filter}")
        logger.info(f"   enable_advanced_features: {self.enable_advanced_features}")
        logger.info(f"   min_weighted_confidence: {self.min_weighted_confidence}")
        logger.info(f"   vix_range: {self.vix_min}-{self.vix_max}")

        # Stats tracking
        self.stats = {
            'total_decisions': 0,
            'trades_approved': 0,
            'straddles_approved': 0,
            'mean_reversion_trades': 0,
            'contrarian_trades': 0,
            'regime_filtered': 0,
            'advanced_boosted': 0,
            'advanced_vetoed': 0,
            'flow_boosted': 0,
            'vetoes_by_reason': {
                'timeframe': 0,
                'hmm': 0,
                'momentum': 0,
                'volatility': 0,
                'technical': 0,
                'options_flow': 0,
                'regime': 0,
                'advanced': 0,
            }
        }

    def decide(self, signal: Dict, hmm_regime: Dict, features: Dict) -> Tuple[str, float, Dict]:
        """
        Make entry decision based on multi-signal consensus.

        Args:
            signal: Contains multi_timeframe_predictions, proposed_direction, confidence
            hmm_regime: Contains trend, volatility, liquidity, confidence
            features: Contains momentum_5m, momentum_15m, price_jerk, rsi, vix_level, volume_spike

        Returns:
            (action, confidence, details)
            action: 'HOLD', 'BUY_CALLS', 'BUY_PUTS', or 'BUY_STRADDLE'
            confidence: 0.0-1.0
            details: dict with check results
        """
        self.stats['total_decisions'] += 1

        # Run all 8 checks (including options flow, regime quality, and advanced features)
        checks = {
            'timeframe': self._check_timeframe_agreement(signal),
            'hmm': self._check_hmm_alignment(signal, hmm_regime),
            'momentum': self._check_momentum_confirmation(signal, features),
            'volatility': self._check_volatility_filter(features),
            'technical': self._check_technical_confirmation(signal, features),
            'options_flow': self._check_options_flow(signal, features),
            'regime': self._check_regime_quality(hmm_regime, features),
            'advanced': self._check_advanced_features(signal, features),
        }

        # Options flow and advanced features are optional - only fail if required or strongly opposing
        core_checks = {k: v for k, v in checks.items() if k not in ['options_flow', 'advanced']}
        all_core_pass = all(c['pass'] for c in core_checks.values())

        # Options flow can veto if it strongly opposes
        flow_check = checks['options_flow']
        flow_veto = False
        if flow_check.get('strongly_opposes') and flow_check.get('flow_strength', 0) > self.options_flow_veto_strength:
            flow_veto = True

        # Advanced features can veto if strongly opposing (VIX term structure + skew)
        adv_check = checks['advanced']
        adv_veto = False
        if adv_check.get('strongly_opposes') and adv_check.get('composite_strength', 0) > self.advanced_veto_strength:
            adv_veto = True
            self.stats['advanced_vetoed'] += 1

        all_pass = all_core_pass and not flow_veto and not adv_veto

        if all_pass:
            self.stats['trades_approved'] += 1
            direction = checks['timeframe']['direction']
            confidence = checks['timeframe']['confidence']

            # Boost confidence if options flow aligns
            if self.options_flow_boost and flow_check.get('flow_aligns'):
                boost = 0.1 * flow_check.get('flow_strength', 0)
                confidence = min(confidence + boost, 1.0)
                self.stats['flow_boosted'] += 1

            # Boost confidence if advanced features align (VIX term structure + skew)
            if self.enable_advanced_features and adv_check.get('aligns_with_direction'):
                if adv_check.get('composite_strength', 0) >= self.advanced_boost_strength:
                    confidence = min(confidence + self.advanced_confidence_boost, 1.0)
                    self.stats['advanced_boosted'] += 1

            action = 'BUY_CALLS' if direction == 'UP' else 'BUY_PUTS'
            return action, confidence, {'checks': checks, 'consensus': True}
        else:
            # Track which checks failed
            failed = [k for k, v in checks.items() if not v['pass']]
            for f in failed:
                self.stats['vetoes_by_reason'][f] += 1

            # Check for straddle opportunity when direction is unclear but volatility is high
            if self.enable_straddles:
                straddle_check = self._check_straddle_opportunity(signal, hmm_regime, features, checks)
                if straddle_check['pass']:
                    self.stats['straddles_approved'] += 1
                    return 'BUY_STRADDLE', straddle_check['confidence'], {
                        'checks': checks,
                        'straddle_check': straddle_check,
                        'consensus': False,
                        'straddle': True,
                        'failed': failed
                    }

            # Check for mean reversion opportunity (trade against extremes)
            if self.enable_mean_reversion:
                mr_check = self._check_mean_reversion_opportunity(signal, features)
                if mr_check['pass']:
                    self.stats['mean_reversion_trades'] += 1
                    return mr_check['action'], mr_check['confidence'], {
                        'checks': checks,
                        'mean_reversion': mr_check,
                        'consensus': False,
                        'mean_reversion_trade': True,
                        'failed': failed
                    }

            # CONTRARIAN MODE: Trade when signals disagree
            # The insight: when all signals agree, the move has already happened
            # We want to catch reversals by fading exhausted moves
            if self.contrarian_mode:
                contrarian_check = self._check_contrarian_opportunity(signal, hmm_regime, features, checks, failed)
                if contrarian_check['pass']:
                    self.stats['contrarian_trades'] += 1
                    return contrarian_check['action'], contrarian_check['confidence'], {
                        'checks': checks,
                        'contrarian': contrarian_check,
                        'consensus': False,
                        'contrarian_trade': True,
                        'failed': failed
                    }

            return 'HOLD', 0.0, {'checks': checks, 'consensus': False, 'failed': failed}

    def _check_timeframe_agreement(self, signal: Dict) -> Dict:
        """
        Check if 15m, 30m, 1h predictions all agree on direction.

        Returns dict with:
            pass: bool - whether check passed
            direction: 'UP' or 'DOWN' or None
            confidence: weighted average confidence
            details: human-readable explanation
        """
        predictions = signal.get('multi_timeframe_predictions', {})

        # Also check alternative key names
        if not predictions:
            predictions = signal.get('timeframe_predictions', {})

        directions = []
        weighted_conf = 0.0
        total_weight = 0.0

        # Map timeframe names to what might be in the signal
        tf_map = {
            '15min': ['15min', '15m', '15'],
            '30min': ['30min', '30m', '30'],
            '1h': ['1h', '60min', '60m', '60', '1hour'],
        }

        for tf in self.required_timeframes:
            pred = None
            # Try different key names
            for key in tf_map.get(tf, [tf]):
                if key in predictions:
                    pred = predictions[key]
                    break

            if pred:
                # Handle different prediction formats
                if isinstance(pred, dict):
                    ret = pred.get('predicted_return', pred.get('return', 0))
                    conf = pred.get('neural_confidence', pred.get('confidence', 0.5))
                    weight = pred.get('weight', 1.0)
                else:
                    # Scalar prediction
                    ret = float(pred)
                    conf = 0.5
                    weight = 1.0

                # Determine direction with small threshold to avoid noise
                if ret > 0.0001:
                    direction = 'UP'
                elif ret < -0.0001:
                    direction = 'DOWN'
                else:
                    direction = 'NEUTRAL'

                directions.append(direction)
                weighted_conf += conf * weight
                total_weight += weight

        # Check if we have all required timeframes
        if len(directions) < len(self.required_timeframes):
            return {
                'pass': False,
                'direction': None,
                'confidence': 0,
                'details': f"Missing timeframes: got {len(directions)}/{len(self.required_timeframes)}"
            }

        # Check agreement - exclude NEUTRAL from consensus check
        non_neutral_dirs = [d for d in directions if d != 'NEUTRAL']
        unique_dirs = set(non_neutral_dirs)

        if len(unique_dirs) == 1 and len(non_neutral_dirs) >= 2:
            # At least 2 timeframes agree on a direction
            consensus_dir = unique_dirs.pop()
            avg_conf = weighted_conf / total_weight if total_weight > 0 else 0

            # Check if agreement meets threshold (2/3 = 0.67)
            agreement_ratio = len(non_neutral_dirs) / len(directions) if directions else 0
            passed = avg_conf >= self.min_weighted_confidence and agreement_ratio >= self.min_timeframe_agreement

            return {
                'pass': passed,
                'direction': consensus_dir,
                'confidence': avg_conf,
                'agreement_ratio': agreement_ratio,
                'details': f"{len(non_neutral_dirs)}/{len(directions)} TFs agree: {consensus_dir}, conf={avg_conf:.2f}"
            }
        elif len(unique_dirs) == 0:
            # All neutral
            return {
                'pass': False,
                'direction': None,
                'confidence': 0,
                'details': f"All timeframes neutral: {directions}"
            }
        else:
            # Conflict between UP and DOWN
            up_count = sum(1 for d in non_neutral_dirs if d == 'UP')
            down_count = sum(1 for d in non_neutral_dirs if d == 'DOWN')
            return {
                'pass': False,
                'direction': None,
                'confidence': 0,
                'details': f"Timeframe conflict: UP={up_count}, DOWN={down_count}, NEUTRAL={len(directions)-len(non_neutral_dirs)}"
            }

    def _check_hmm_alignment(self, signal: Dict, hmm_regime: Dict) -> Dict:
        """
        Check if HMM trend aligns with proposed direction.
        HARD VETO if misaligned (no soft penalty).

        Returns dict with:
            pass: bool - whether HMM agrees
            hmm_stance: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            hmm_trend: raw trend value 0-1
            hmm_confidence: regime confidence
            details: human-readable explanation
        """
        # Get proposed direction from timeframe check or signal
        direction = signal.get('proposed_direction')
        if not direction:
            # Try to infer from predictions
            preds = signal.get('multi_timeframe_predictions', {})
            if preds:
                total_ret = sum(
                    p.get('predicted_return', 0) if isinstance(p, dict) else float(p)
                    for p in preds.values()
                )
                direction = 'UP' if total_ret > 0 else 'DOWN'

        if not direction:
            return {
                'pass': False,
                'hmm_stance': 'UNKNOWN',
                'hmm_trend': 0.5,
                'hmm_confidence': 0,
                'details': "No direction to check against HMM"
            }

        # Get HMM values
        hmm_trend = hmm_regime.get('trend', hmm_regime.get('trend_state', 0.5))
        hmm_conf = hmm_regime.get('confidence', hmm_regime.get('hmm_confidence', 0.5))

        # Normalize trend if it's a state index (0, 1, 2)
        if isinstance(hmm_trend, int) and hmm_trend in [0, 1, 2]:
            hmm_trend = hmm_trend / 2.0  # 0->0, 1->0.5, 2->1.0

        # Determine HMM stance
        if hmm_trend > self.hmm_bullish_threshold:
            hmm_stance = 'BULLISH'
        elif hmm_trend < self.hmm_bearish_threshold:
            hmm_stance = 'BEARISH'
        else:
            hmm_stance = 'NEUTRAL'

        # Check alignment - STRICT
        aligned = False
        if direction == 'UP' and hmm_stance in ['BULLISH', 'NEUTRAL']:
            aligned = True
        elif direction == 'DOWN' and hmm_stance in ['BEARISH', 'NEUTRAL']:
            aligned = True

        # Also require confidence threshold
        conf_ok = hmm_conf >= self.min_hmm_confidence

        return {
            'pass': aligned and conf_ok,
            'hmm_stance': hmm_stance,
            'hmm_trend': hmm_trend,
            'hmm_confidence': hmm_conf,
            'aligned': aligned,
            'conf_ok': conf_ok,
            'details': f"HMM {hmm_stance} (trend={hmm_trend:.2f}, conf={hmm_conf:.2f}), dir={direction}"
        }

    def _check_momentum_confirmation(self, signal: Dict, features: Dict) -> Dict:
        """
        Check momentum indicators confirm direction.

        Checks:
        1. momentum_5m and momentum_15m same sign as direction
        2. Jerk (3rd derivative) confirms acceleration
        3. RSI not overbought (>70) for calls, not oversold (<30) for puts

        Returns dict with:
            pass: bool - all momentum checks pass
            momentum_aligned: bool
            jerk_confirms: bool
            rsi_ok: bool
            details: human-readable explanation
        """
        direction = signal.get('proposed_direction')
        if not direction:
            # Infer from predictions
            preds = signal.get('multi_timeframe_predictions', {})
            if preds:
                total_ret = sum(
                    p.get('predicted_return', 0) if isinstance(p, dict) else float(p)
                    for p in preds.values()
                )
                direction = 'UP' if total_ret > 0 else 'DOWN'

        if not direction:
            return {
                'pass': False,
                'momentum_aligned': False,
                'jerk_confirms': False,
                'rsi_ok': False,
                'details': "No direction for momentum check"
            }

        # Get momentum values
        momentum_5m = features.get('momentum_5m', features.get('momentum_5', 0))
        momentum_15m = features.get('momentum_15m', features.get('momentum_15', 0))
        jerk = features.get('price_jerk', features.get('jerk', 0))
        rsi = features.get('rsi', features.get('rsi_14', 50))

        # Check momentum alignment (relaxed: only need ONE momentum to agree, not both)
        if direction == 'UP':
            # Relaxed: at least one momentum positive, or skip if disabled
            mom_ok = (momentum_5m > 0 or momentum_15m > 0) if self.require_momentum_alignment else True
            jerk_ok = jerk > 0 if self.require_jerk_confirmation else True
            rsi_ok = rsi < self.rsi_overbought
        else:  # DOWN
            # Relaxed: at least one momentum negative, or skip if disabled
            mom_ok = (momentum_5m < 0 or momentum_15m < 0) if self.require_momentum_alignment else True
            jerk_ok = jerk < 0 if self.require_jerk_confirmation else True
            rsi_ok = rsi > self.rsi_oversold

        all_ok = mom_ok and jerk_ok and rsi_ok

        return {
            'pass': all_ok,
            'momentum_aligned': mom_ok,
            'jerk_confirms': jerk_ok,
            'rsi_ok': rsi_ok,
            'momentum_5m': momentum_5m,
            'momentum_15m': momentum_15m,
            'jerk': jerk,
            'rsi': rsi,
            'details': f"mom={mom_ok}, jerk={jerk_ok}, rsi={rsi_ok} (RSI={rsi:.1f}, jerk={jerk:.4f})"
        }

    def _check_volatility_filter(self, features: Dict) -> Dict:
        """
        Check volatility conditions are favorable.

        Checks:
        1. VIX in acceptable range (12-30)
        2. HMM volatility not extreme (panic or dead calm)
        3. Volume spike < 3x (avoid news events)

        Returns dict with:
            pass: bool - all volatility checks pass
            vix_ok: bool
            vol_regime_ok: bool
            volume_ok: bool
            details: human-readable explanation
        """
        vix = features.get('vix_level', features.get('vix', 18))
        hmm_vol = features.get('hmm_volatility', features.get('volatility_state', 0.5))
        volume_spike = features.get('volume_spike', features.get('volume_ratio', 1.0))

        # Normalize hmm_vol if it's a state index
        if isinstance(hmm_vol, int) and hmm_vol in [0, 1, 2]:
            hmm_vol = hmm_vol / 2.0

        # Check conditions
        vix_ok = self.vix_min <= vix <= self.vix_max
        vol_regime_ok = 0.2 < hmm_vol < 0.8  # Not extreme
        volume_ok = volume_spike < self.max_volume_spike

        all_ok = vix_ok and vol_regime_ok and volume_ok

        return {
            'pass': all_ok,
            'vix_ok': vix_ok,
            'vol_regime_ok': vol_regime_ok,
            'volume_ok': volume_ok,
            'vix': vix,
            'hmm_volatility': hmm_vol,
            'volume_spike': volume_spike,
            'details': f"VIX={vix:.1f} ({vix_ok}), vol={hmm_vol:.2f} ({vol_regime_ok}), spike={volume_spike:.1f}x ({volume_ok})"
        }

    def _check_options_flow(self, signal: Dict, features: Dict) -> Dict:
        """
        Check options flow signal (Signal 6 - NEW).

        Options flow analysis:
        - Put/call ratio indicates market sentiment
        - Unusual activity may signal informed trading
        - Flow aligning with direction = confirmation
        - Flow strongly opposing direction = warning

        Returns dict with:
            pass: bool - options flow is favorable (or neutral)
            flow_aligns: bool - flow aligns with proposed direction
            strongly_opposes: bool - flow strongly opposes direction
            flow_signal: -1/0/1
            flow_strength: 0-1
            details: explanation
        """
        # Get direction from signal
        direction = signal.get('proposed_direction')
        if not direction:
            preds = signal.get('multi_timeframe_predictions', {})
            if preds:
                total_ret = sum(
                    p.get('predicted_return', 0) if isinstance(p, dict) else float(p)
                    for p in preds.values()
                )
                direction = 'UP' if total_ret > 0 else 'DOWN'

        # Get options flow data
        flow_signal = features.get('options_flow_signal', signal.get('options_flow_signal', 0))
        flow_strength = features.get('options_flow_strength', signal.get('options_flow_strength', 0))
        unusual_activity = features.get('options_unusual_activity', signal.get('options_unusual_activity', False))
        pcr = features.get('put_call_ratio', signal.get('put_call_ratio', 1.0))

        # Default to pass if no flow data
        if flow_signal == 0 and flow_strength == 0:
            return {
                'pass': True,
                'flow_aligns': False,
                'strongly_opposes': False,
                'flow_signal': 0,
                'flow_strength': 0.0,
                'unusual_activity': unusual_activity,
                'put_call_ratio': pcr,
                'details': 'No options flow data (neutral)'
            }

        # Check alignment
        flow_aligns = False
        strongly_opposes = False

        if direction == 'UP':
            # For calls, bullish flow (flow_signal=1) is good
            flow_aligns = flow_signal > 0
            strongly_opposes = flow_signal < 0 and flow_strength > 0.5
        elif direction == 'DOWN':
            # For puts, bearish flow (flow_signal=-1) is good
            flow_aligns = flow_signal < 0
            strongly_opposes = flow_signal > 0 and flow_strength > 0.5

        # Pass if:
        # 1. Flow aligns (confirmation)
        # 2. Flow is neutral (no info)
        # 3. Flow weakly opposes (not enough to veto)
        # Fail only if require_options_flow and doesn't align
        if self.require_options_flow:
            passed = flow_aligns or flow_signal == 0
        else:
            passed = True  # Always pass if not required (but can still boost/veto)

        return {
            'pass': passed,
            'flow_aligns': flow_aligns,
            'strongly_opposes': strongly_opposes,
            'flow_signal': flow_signal,
            'flow_strength': flow_strength,
            'unusual_activity': unusual_activity,
            'put_call_ratio': pcr,
            'direction': direction,
            'details': f"Flow={'bullish' if flow_signal>0 else 'bearish' if flow_signal<0 else 'neutral'} (str={flow_strength:.2f}), PCR={pcr:.2f}, aligns={flow_aligns}"
        }

    def _check_mean_reversion_opportunity(self, signal: Dict, features: Dict) -> Dict:
        """
        Check for mean reversion trading opportunity (Signal 7 - NEW).

        Mean reversion logic:
        - When RSI is oversold (<25) and BB position is low (<0.10) -> BUY CALLS (expect bounce)
        - When RSI is overbought (>75) and BB position is high (>0.90) -> BUY PUTS (expect pullback)

        This is CONTRARIAN to momentum - we're fading extremes.

        Returns dict with:
            pass: bool - mean reversion opportunity detected
            action: 'BUY_CALLS' or 'BUY_PUTS'
            confidence: 0-1
            details: explanation
        """
        if not self.enable_mean_reversion:
            return {'pass': False, 'action': 'HOLD', 'confidence': 0, 'details': 'Mean reversion disabled'}

        # Get indicators
        rsi = features.get('rsi', signal.get('rsi', 50))
        bb_pos = features.get('bb_position', signal.get('bb_position', 0.5))
        vix = features.get('vix_level', features.get('vix', 18))

        # Mean reversion works better in normal volatility (not extreme)
        if vix > 35 or vix < 12:
            return {
                'pass': False,
                'action': 'HOLD',
                'confidence': 0,
                'rsi': rsi,
                'bb_position': bb_pos,
                'details': f'VIX={vix:.1f} too extreme for mean reversion'
            }

        # Check for oversold condition (buy calls expecting bounce)
        oversold = rsi < self.mr_rsi_oversold and bb_pos < self.mr_bb_oversold
        # Check for overbought condition (buy puts expecting pullback)
        overbought = rsi > self.mr_rsi_overbought and bb_pos > self.mr_bb_overbought

        if oversold:
            # How extreme is the oversold condition? More extreme = higher confidence
            rsi_extremity = (self.mr_rsi_oversold - rsi) / self.mr_rsi_oversold
            bb_extremity = (self.mr_bb_oversold - bb_pos) / self.mr_bb_oversold
            confidence = 0.5 + self.mr_confidence_boost + 0.2 * (rsi_extremity + bb_extremity) / 2
            confidence = min(confidence, 0.85)

            return {
                'pass': True,
                'action': 'BUY_CALLS',
                'confidence': confidence,
                'rsi': rsi,
                'bb_position': bb_pos,
                'details': f'OVERSOLD: RSI={rsi:.0f} BB={bb_pos:.2f} -> BUY CALLS (mean reversion)'
            }

        if overbought:
            rsi_extremity = (rsi - self.mr_rsi_overbought) / (100 - self.mr_rsi_overbought)
            bb_extremity = (bb_pos - self.mr_bb_overbought) / (1 - self.mr_bb_overbought)
            confidence = 0.5 + self.mr_confidence_boost + 0.2 * (rsi_extremity + bb_extremity) / 2
            confidence = min(confidence, 0.85)

            return {
                'pass': True,
                'action': 'BUY_PUTS',
                'confidence': confidence,
                'rsi': rsi,
                'bb_position': bb_pos,
                'details': f'OVERBOUGHT: RSI={rsi:.0f} BB={bb_pos:.2f} -> BUY PUTS (mean reversion)'
            }

        return {
            'pass': False,
            'action': 'HOLD',
            'confidence': 0,
            'rsi': rsi,
            'bb_position': bb_pos,
            'details': f'No extreme: RSI={rsi:.0f} BB={bb_pos:.2f}'
        }

    def _check_contrarian_opportunity(self, signal: Dict, hmm_regime: Dict, features: Dict,
                                        checks: Dict, failed: list) -> Dict:
        """
        Check for contrarian trading opportunity (NEW - trade against consensus).

        The insight: when all signals agree strongly, the move has already happened.
        We want to catch early reversals by:
        1. Fading the neural network when HMM disagrees (HMM tracks trend, NN lags)
        2. Trading with HMM trend when NN is uncertain
        3. Looking for exhaustion (strong consensus + extreme RSI/BB)

        Returns dict with:
            pass: bool - contrarian opportunity detected
            action: 'BUY_CALLS' or 'BUY_PUTS'
            confidence: 0-1
            reason: what triggered the contrarian trade
            details: explanation
        """
        # Get key values
        hmm_trend = hmm_regime.get('trend', hmm_regime.get('trend_state', 0.5))
        hmm_conf = hmm_regime.get('confidence', hmm_regime.get('hmm_confidence', 0.5))
        vix = features.get('vix_level', features.get('vix', 18))

        # Normalize trend if state index
        if isinstance(hmm_trend, int) and hmm_trend in [0, 1, 2]:
            hmm_trend = hmm_trend / 2.0

        # Get neural network direction from timeframe check
        tf_check = checks.get('timeframe', {})
        nn_direction = tf_check.get('direction')  # 'UP', 'DOWN', or None
        nn_confidence = tf_check.get('confidence', 0)

        # VIX filter - contrarian works best in moderate volatility
        if vix > 35 or vix < 10:
            return {
                'pass': False,
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'vix_extreme',
                'details': f'VIX={vix:.1f} too extreme for contrarian'
            }

        # CONTRARIAN STRATEGY 1: HMM Override
        # If HMM has strong trend but NN disagrees or is weak, trade with HMM
        if self.contrarian_hmm_override and hmm_conf >= 0.7:
            hmm_is_bullish = hmm_trend > 0.65
            hmm_is_bearish = hmm_trend < 0.35

            if hmm_is_bullish and (nn_direction == 'DOWN' or nn_confidence < 0.20):
                # HMM says up, NN says down or uncertain -> BUY CALLS
                confidence = 0.5 + 0.2 * hmm_conf
                return {
                    'pass': True,
                    'action': 'BUY_CALLS',
                    'confidence': confidence,
                    'reason': 'hmm_override_bullish',
                    'hmm_trend': hmm_trend,
                    'hmm_conf': hmm_conf,
                    'nn_direction': nn_direction,
                    'nn_confidence': nn_confidence,
                    'details': f'HMM bullish ({hmm_trend:.2f}, conf={hmm_conf:.2f}) overrides NN {nn_direction}'
                }

            if hmm_is_bearish and (nn_direction == 'UP' or nn_confidence < 0.20):
                # HMM says down, NN says up or uncertain -> BUY PUTS
                confidence = 0.5 + 0.2 * hmm_conf
                return {
                    'pass': True,
                    'action': 'BUY_PUTS',
                    'confidence': confidence,
                    'reason': 'hmm_override_bearish',
                    'hmm_trend': hmm_trend,
                    'hmm_conf': hmm_conf,
                    'nn_direction': nn_direction,
                    'nn_confidence': nn_confidence,
                    'details': f'HMM bearish ({hmm_trend:.2f}, conf={hmm_conf:.2f}) overrides NN {nn_direction}'
                }

        # CONTRARIAN STRATEGY 2: Fade Exhaustion
        # If NN strongly predicts a direction but momentum is reversing, fade it
        momentum_check = checks.get('momentum', {})
        mom_5m = momentum_check.get('momentum_5m', 0)
        mom_15m = momentum_check.get('momentum_15m', 0)
        rsi = momentum_check.get('rsi', 50)

        if nn_direction == 'UP' and nn_confidence > 0.25:
            # NN says UP but check for exhaustion
            if mom_5m < 0 and mom_15m < 0 and rsi > 65:
                # Short-term momentum reversed, RSI high -> fade the UP call
                confidence = 0.55 + 0.1 * (rsi - 65) / 35  # Higher RSI = higher conf in fade
                return {
                    'pass': True,
                    'action': 'BUY_PUTS',
                    'confidence': min(confidence, 0.75),
                    'reason': 'fade_exhausted_up',
                    'nn_direction': nn_direction,
                    'nn_confidence': nn_confidence,
                    'momentum_5m': mom_5m,
                    'momentum_15m': mom_15m,
                    'rsi': rsi,
                    'details': f'Fading exhausted UP: NN={nn_direction} but mom<0 and RSI={rsi:.0f}'
                }

        if nn_direction == 'DOWN' and nn_confidence > 0.25:
            # NN says DOWN but check for exhaustion
            if mom_5m > 0 and mom_15m > 0 and rsi < 35:
                # Short-term momentum reversed, RSI low -> fade the DOWN call
                confidence = 0.55 + 0.1 * (35 - rsi) / 35  # Lower RSI = higher conf in fade
                return {
                    'pass': True,
                    'action': 'BUY_CALLS',
                    'confidence': min(confidence, 0.75),
                    'reason': 'fade_exhausted_down',
                    'nn_direction': nn_direction,
                    'nn_confidence': nn_confidence,
                    'momentum_5m': mom_5m,
                    'momentum_15m': mom_15m,
                    'rsi': rsi,
                    'details': f'Fading exhausted DOWN: NN={nn_direction} but mom>0 and RSI={rsi:.0f}'
                }

        # CONTRARIAN STRATEGY 3: Multiple Failures = Opportunity
        # If we have exactly the right failures, it might be a contrarian setup
        if len(failed) >= self.contrarian_min_disagreement:
            # Many signals disagree - this is where bandit finds opportunities
            # Use HMM as tiebreaker if it has decent confidence
            if hmm_conf >= 0.5:
                if hmm_trend > 0.55:
                    return {
                        'pass': True,
                        'action': 'BUY_CALLS',
                        'confidence': 0.5 + 0.15 * hmm_conf,
                        'reason': 'disagreement_hmm_bullish',
                        'failed_checks': failed,
                        'hmm_trend': hmm_trend,
                        'hmm_conf': hmm_conf,
                        'details': f'{len(failed)} checks failed, HMM bullish ({hmm_trend:.2f}) tiebreaker'
                    }
                elif hmm_trend < 0.45:
                    return {
                        'pass': True,
                        'action': 'BUY_PUTS',
                        'confidence': 0.5 + 0.15 * hmm_conf,
                        'reason': 'disagreement_hmm_bearish',
                        'failed_checks': failed,
                        'hmm_trend': hmm_trend,
                        'hmm_conf': hmm_conf,
                        'details': f'{len(failed)} checks failed, HMM bearish ({hmm_trend:.2f}) tiebreaker'
                    }

        return {
            'pass': False,
            'action': 'HOLD',
            'confidence': 0,
            'reason': 'no_contrarian_signal',
            'details': 'No contrarian opportunity detected'
        }

    def _check_regime_quality(self, hmm_regime: Dict, features: Dict) -> Dict:
        """
        Check if HMM regime is favorable for trading (Approach 3 - NEW).

        Favorable regimes:
        1. Clear trend direction (not stuck in neutral)
        2. Not choppy (neutral trend + high volatility = bad)
        3. Decent liquidity
        4. High regime confidence

        Returns dict with:
            pass: bool - regime is favorable
            trend_strength: float - how far from neutral
            is_choppy: bool - is the market choppy
            liquidity_ok: bool
            confidence_ok: bool
            regime_score: float - overall regime quality 0-1
            details: explanation
        """
        if not self.enable_regime_filter:
            return {
                'pass': True,
                'trend_strength': 0.5,
                'is_choppy': False,
                'liquidity_ok': True,
                'confidence_ok': True,
                'regime_score': 1.0,
                'details': 'Regime filter disabled'
            }

        # Get HMM values
        hmm_trend = hmm_regime.get('trend', hmm_regime.get('trend_state', 0.5))
        hmm_vol = hmm_regime.get('volatility', hmm_regime.get('volatility_state', 0.5))
        hmm_liq = hmm_regime.get('liquidity', hmm_regime.get('liquidity_state', 0.5))
        hmm_conf = hmm_regime.get('confidence', hmm_regime.get('hmm_confidence', 0.5))

        # Normalize if state index (0, 1, 2)
        if isinstance(hmm_trend, int) and hmm_trend in [0, 1, 2]:
            hmm_trend = hmm_trend / 2.0
        if isinstance(hmm_vol, int) and hmm_vol in [0, 1, 2]:
            hmm_vol = hmm_vol / 2.0
        if isinstance(hmm_liq, int) and hmm_liq in [0, 1, 2]:
            hmm_liq = hmm_liq / 2.0

        # Check trend strength (distance from neutral 0.5)
        trend_strength = abs(hmm_trend - 0.5) * 2  # 0 = neutral, 1 = extreme
        trend_ok = trend_strength >= self.regime_trend_strength_min

        # Check for choppy regime (neutral trend + high volatility)
        is_neutral_trend = trend_strength < self.regime_trend_strength_min
        is_high_vol = hmm_vol >= self.regime_choppy_vol_threshold
        is_choppy = is_neutral_trend and is_high_vol

        if self.regime_avoid_choppy and is_choppy:
            choppy_ok = False
        else:
            choppy_ok = True

        # Check liquidity
        if self.regime_require_liquidity:
            liquidity_ok = hmm_liq >= self.regime_liquidity_min
        else:
            liquidity_ok = True

        # Check confidence
        confidence_ok = hmm_conf >= self.regime_confidence_min

        # Calculate regime score (0-1)
        regime_score = (
            0.3 * trend_strength +
            0.2 * (1.0 - hmm_vol) +  # Lower vol = better (more stable)
            0.2 * hmm_liq +
            0.3 * hmm_conf
        )

        # All checks must pass
        all_ok = trend_ok and choppy_ok and liquidity_ok and confidence_ok

        if not all_ok:
            self.stats['regime_filtered'] += 1

        return {
            'pass': all_ok,
            'trend_strength': trend_strength,
            'is_choppy': is_choppy,
            'trend_ok': trend_ok,
            'choppy_ok': choppy_ok,
            'liquidity_ok': liquidity_ok,
            'confidence_ok': confidence_ok,
            'regime_score': regime_score,
            'hmm_trend': hmm_trend,
            'hmm_volatility': hmm_vol,
            'hmm_liquidity': hmm_liq,
            'hmm_confidence': hmm_conf,
            'details': f"trend_str={trend_strength:.2f} ({trend_ok}), choppy={is_choppy} ({choppy_ok}), liq={hmm_liq:.2f} ({liquidity_ok}), conf={hmm_conf:.2f} ({confidence_ok})"
        }

    def _check_advanced_features(self, signal: Dict, features: Dict) -> Dict:
        """
        Check advanced features: VIX term structure and options skew (Approach 4 - NEW).

        Uses:
        - VIX term structure (contango = bullish, backwardation = bearish)
        - Options skew (high put skew = bearish, high call skew = bullish)
        - Composite fear/greed indicator

        Returns dict with:
            pass: bool - always True (optional check, can only boost/veto)
            aligns_with_direction: bool - composite signal aligns with direction
            strongly_opposes: bool - composite signal strongly opposes direction
            composite_signal: -1/0/1
            composite_strength: 0-1
            details: explanation
        """
        if not self.enable_advanced_features:
            return {
                'pass': True,
                'aligns_with_direction': False,
                'strongly_opposes': False,
                'composite_signal': 0,
                'composite_strength': 0.0,
                'details': 'Advanced features disabled'
            }

        # Get direction from signal
        direction = signal.get('proposed_direction')
        if not direction:
            preds = signal.get('multi_timeframe_predictions', {})
            if preds:
                total_ret = sum(
                    p.get('predicted_return', 0) if isinstance(p, dict) else float(p)
                    for p in preds.values()
                )
                direction = 'UP' if total_ret > 0 else 'DOWN'

        # Get VIX term structure data from features
        vix = features.get('vix_level', features.get('vix', None))
        vix3m = features.get('vix3m', features.get('vix_3m', None))
        vix9d = features.get('vix9d', features.get('vix_9d', None))

        # Get options skew data (may not be available)
        atm_iv = features.get('atm_iv', None)
        put_25d_iv = features.get('put_25d_iv', features.get('put_iv', None))
        call_25d_iv = features.get('call_25d_iv', features.get('call_iv', None))
        pcr = features.get('put_call_ratio', signal.get('put_call_ratio', None))

        # Import the compute functions
        try:
            from bot_modules.technical_indicators import compute_advanced_features
            adv = compute_advanced_features(
                vix=vix, vix9d=vix9d, vix3m=vix3m,
                atm_iv=atm_iv, put_25d_iv=put_25d_iv, call_25d_iv=call_25d_iv,
                put_call_ratio=pcr
            )
        except ImportError:
            return {
                'pass': True,
                'aligns_with_direction': False,
                'strongly_opposes': False,
                'composite_signal': 0,
                'composite_strength': 0.0,
                'details': 'Advanced features module not available'
            }

        composite_signal = adv.get('composite_signal', 0)
        composite_strength = adv.get('composite_strength', 0.0)

        # Check alignment with direction
        aligns = False
        strongly_opposes = False

        if direction == 'UP':
            aligns = composite_signal > 0
            strongly_opposes = composite_signal < 0 and composite_strength > 0.5
        elif direction == 'DOWN':
            aligns = composite_signal < 0
            strongly_opposes = composite_signal > 0 and composite_strength > 0.5

        return {
            'pass': True,  # Always pass (optional check)
            'aligns_with_direction': aligns,
            'strongly_opposes': strongly_opposes,
            'composite_signal': composite_signal,
            'composite_strength': composite_strength,
            'term_structure': adv.get('vix_term_structure', {}).get('term_structure', 'unknown'),
            'skew_ratio': adv.get('options_skew', {}).get('skew_ratio', 1.0),
            'direction': direction,
            'details': adv.get('details', 'No advanced data')
        }

    def _check_straddle_opportunity(self, signal: Dict, hmm_regime: Dict, features: Dict, checks: Dict) -> Dict:
        """
        Check if conditions favor a straddle (buy both call and put).

        Straddles are profitable when:
        1. VIX is elevated (expecting big move)
        2. HMM volatility is high
        3. Direction is unclear (conflicting timeframes OR low confidence)
        4. No extreme news events (volume spike not too high)

        Returns dict with:
            pass: bool - straddle conditions met
            confidence: float - straddle confidence
            details: human-readable explanation
        """
        if not self.enable_straddles:
            return {
                'pass': False,
                'confidence': 0.0,
                'details': 'Straddles disabled'
            }

        # Get volatility metrics
        vix = features.get('vix_level', features.get('vix', 18))
        hmm_vol = hmm_regime.get('volatility', hmm_regime.get('volatility_state', 0.5))
        volume_spike = features.get('volume_spike', features.get('volume_ratio', 1.0))

        # Normalize hmm_vol if it's a state index
        if isinstance(hmm_vol, int) and hmm_vol in [0, 1, 2]:
            hmm_vol = hmm_vol / 2.0

        # Get direction confidence from timeframe check
        tf_check = checks.get('timeframe', {})
        direction_confidence = tf_check.get('confidence', 0.5)
        has_direction_conflict = not tf_check.get('pass', False)

        # Straddle conditions
        vix_ok = self.straddle_vix_min <= vix <= self.straddle_vix_max
        vol_ok = hmm_vol >= self.straddle_hmm_vol_min
        volume_ok = volume_spike < self.max_volume_spike  # Not too extreme

        # Direction unclear: either conflict or low confidence
        if self.straddle_direction_conflict_required:
            direction_unclear = has_direction_conflict or direction_confidence < self.straddle_confidence_max
        else:
            direction_unclear = True  # Always allow if not required

        all_ok = vix_ok and vol_ok and volume_ok and direction_unclear

        # Calculate straddle confidence based on how well conditions are met
        if all_ok:
            # Higher VIX = higher confidence in big move
            vix_score = min((vix - self.straddle_vix_min) / (self.straddle_vix_max - self.straddle_vix_min), 1.0)
            # Higher HMM vol = higher confidence
            vol_score = min((hmm_vol - self.straddle_hmm_vol_min) / (1.0 - self.straddle_hmm_vol_min), 1.0)
            confidence = 0.5 + 0.3 * vix_score + 0.2 * vol_score
        else:
            confidence = 0.0

        return {
            'pass': all_ok,
            'confidence': confidence,
            'vix_ok': vix_ok,
            'vol_ok': vol_ok,
            'volume_ok': volume_ok,
            'direction_unclear': direction_unclear,
            'vix': vix,
            'hmm_volatility': hmm_vol,
            'direction_confidence': direction_confidence,
            'details': f"VIX={vix:.1f} ({vix_ok}), HMM_vol={hmm_vol:.2f} ({vol_ok}), dir_unclear={direction_unclear}"
        }

    def _check_technical_confirmation(self, signal: Dict, features: Dict) -> Dict:
        """
        Check technical indicators confirm direction (Signal 5).

        Checks:
        1. MACD histogram aligns with direction (positive for calls, negative for puts)
        2. Bollinger Band position not at extremes (or extremes favor direction)
        3. Market breadth (QQQ/SPY alignment) if available

        Returns dict with:
            pass: bool - technical indicators confirm
            macd_ok: bool
            bb_ok: bool
            breadth_ok: bool
            details: human-readable explanation
        """
        if not self.require_technical_confirmation:
            return {
                'pass': True,
                'macd_ok': True,
                'bb_ok': True,
                'breadth_ok': True,
                'details': 'Technical confirmation disabled'
            }

        # Get direction from timeframe check (we need to know which direction to confirm)
        direction = signal.get('proposed_direction', signal.get('direction', None))
        if direction is None:
            # Try to infer from multi-timeframe predictions
            preds = signal.get('multi_timeframe_predictions', {})
            up_count = 0
            down_count = 0
            for tf, pred in preds.items():
                ret = pred.get('predicted_return', 0) if isinstance(pred, dict) else 0
                if ret > 0.0001:
                    up_count += 1
                elif ret < -0.0001:
                    down_count += 1
            direction = 'UP' if up_count > down_count else ('DOWN' if down_count > up_count else None)

        # Get technical indicators from features
        macd = features.get('macd', features.get('macd_histogram', features.get('macd_norm', 0)))
        bb_pos = features.get('bb_position', features.get('bollinger_position', features.get('bb_pos', 0.5)))
        market_breadth = features.get('market_breadth', features.get('breadth', None))
        hma_trend = features.get('hma_trend', features.get('hma', 0))

        # MACD check: histogram should align with direction
        if direction == 'UP':
            macd_ok = macd >= self.macd_threshold  # Positive or zero for calls
        elif direction == 'DOWN':
            macd_ok = macd <= -self.macd_threshold  # Negative or zero for puts
        else:
            macd_ok = True  # No direction = pass

        # Bollinger Band check: avoid buying calls at top, puts at bottom
        # BB position: 0 = at lower band, 0.5 = at SMA, 1 = at upper band
        if direction == 'UP':
            # For calls: don't buy if already overbought (BB > 0.85)
            bb_ok = bb_pos < self.bb_extreme_high
        elif direction == 'DOWN':
            # For puts: don't buy if already oversold (BB < 0.15)
            bb_ok = bb_pos > self.bb_extreme_low
        else:
            bb_ok = True

        # HMA trend check: HMA must align with direction
        if self.require_hma_confirmation:
            if direction == 'UP':
                hma_ok = hma_trend >= 0  # HMA trending up or flat
            elif direction == 'DOWN':
                hma_ok = hma_trend <= 0  # HMA trending down or flat
            else:
                hma_ok = True
        else:
            hma_ok = True

        # Market breadth check (optional)
        if self.require_market_breadth and market_breadth is not None:
            if direction == 'UP':
                breadth_ok = market_breadth > 0  # Both moving up
            elif direction == 'DOWN':
                breadth_ok = market_breadth < 0  # Both moving down
            else:
                breadth_ok = True
        else:
            breadth_ok = True  # Skip if not required or no data

        all_ok = macd_ok and bb_ok and hma_ok and breadth_ok

        return {
            'pass': all_ok,
            'macd_ok': macd_ok,
            'bb_ok': bb_ok,
            'hma_ok': hma_ok,
            'breadth_ok': breadth_ok,
            'macd': macd,
            'bb_position': bb_pos,
            'hma_trend': hma_trend,
            'market_breadth': market_breadth,
            'direction': direction,
            'details': f"MACD={macd:.4f} ({macd_ok}), BB={bb_pos:.2f} ({bb_ok}), HMA={hma_trend} ({hma_ok}), breadth={market_breadth} ({breadth_ok})"
        }

    def get_stats(self) -> Dict:
        """Get decision statistics."""
        total = self.stats['total_decisions']
        approved = self.stats['trades_approved']
        rate = approved / total if total > 0 else 0

        return {
            **self.stats,
            'approval_rate': rate,
            'veto_breakdown': {
                k: v / total if total > 0 else 0
                for k, v in self.stats['vetoes_by_reason'].items()
            }
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_decisions': 0,
            'trades_approved': 0,
            'straddles_approved': 0,
            'mean_reversion_trades': 0,
            'contrarian_trades': 0,
            'regime_filtered': 0,
            'advanced_boosted': 0,
            'advanced_vetoed': 0,
            'flow_boosted': 0,
            'vetoes_by_reason': {
                'timeframe': 0,
                'hmm': 0,
                'momentum': 0,
                'volatility': 0,
                'technical': 0,
                'options_flow': 0,
                'regime': 0,
                'advanced': 0,
            }
        }


def decide_from_signal(signal: Dict,
                       hmm_regime: Dict = None,
                       features: Dict = None,
                       account_state: Dict = None,
                       vix_level: float = 18.0,
                       market_open: bool = True,
                       verbose: bool = False) -> Tuple[str, float, Dict]:
    """
    Standalone function for integration with train_time_travel.py.

    Args:
        signal: Bot signal containing predictions
        hmm_regime: HMM regime dict (optional, will extract from signal if not provided)
        features: Feature dict (optional, will extract from signal if not provided)
        account_state: Account state dict (unused, for API compatibility)
        vix_level: Current VIX level
        market_open: Whether market is open (unused, for API compatibility)
        verbose: Print decision details

    Returns:
        (action, confidence, details)
    """
    global _controller_instance

    # Lazy singleton
    if '_controller_instance' not in globals() or _controller_instance is None:
        _controller_instance = ConsensusEntryController()

    controller = _controller_instance

    # Extract HMM regime from signal if not provided
    if hmm_regime is None:
        hmm_regime = {
            'trend': signal.get('hmm_trend', signal.get('trend_state', 0.5)),
            'volatility': signal.get('hmm_volatility', signal.get('volatility_state', 0.5)),
            'liquidity': signal.get('hmm_liquidity', signal.get('liquidity_state', 0.5)),
            'confidence': signal.get('hmm_confidence', 0.5),
        }

    # Extract features from signal if not provided
    if features is None:
        features = {
            'momentum_5m': signal.get('momentum_5m', signal.get('momentum_5', 0)),
            'momentum_15m': signal.get('momentum_15m', signal.get('momentum_15', 0)),
            'price_jerk': signal.get('price_jerk', signal.get('jerk', 0)),
            'rsi': signal.get('rsi', signal.get('rsi_14', 50)),
            'vix_level': vix_level,
            'hmm_volatility': hmm_regime.get('volatility', 0.5),
            'volume_spike': signal.get('volume_spike', signal.get('volume_ratio', 1.0)),
            # NEW: Technical indicators for Signal 5
            'macd': signal.get('macd', signal.get('macd_histogram', signal.get('macd_norm', 0))),
            'bb_position': signal.get('bb_position', signal.get('bollinger_position', signal.get('bb_pos', 0.5))),
            'hma_trend': signal.get('hma_trend', signal.get('hma', 0)),
            'market_breadth': signal.get('market_breadth', signal.get('breadth', None)),
        }
    else:
        # Ensure vix_level is in features
        if 'vix_level' not in features:
            features['vix_level'] = vix_level
        # Ensure technical indicators are in features
        if 'macd' not in features:
            features['macd'] = signal.get('macd', signal.get('macd_histogram', 0))
        if 'bb_position' not in features:
            features['bb_position'] = signal.get('bb_position', signal.get('bb_pos', 0.5))
        if 'hma_trend' not in features:
            features['hma_trend'] = signal.get('hma_trend', signal.get('hma', 0))

    action, confidence, details = controller.decide(signal, hmm_regime, features)

    if verbose:
        if details.get('consensus'):
            flow_boost = " (+flow boost)" if details.get('checks', {}).get('options_flow', {}).get('flow_aligns') else ""
            print(f"   [CONSENSUS] {action} - All 6 signals agree! conf={confidence:.2f}{flow_boost}")
        elif details.get('straddle'):
            straddle_info = details.get('straddle_check', {})
            print(f"   [CONSENSUS] BUY_STRADDLE - High vol + unclear direction! conf={confidence:.2f}")
            print(f"      straddle: {straddle_info.get('details', '')}")
        elif details.get('contrarian_trade'):
            contrarian_info = details.get('contrarian', {})
            reason = contrarian_info.get('reason', 'unknown')
            print(f"   [CONTRARIAN] {action} - {reason}! conf={confidence:.2f}")
            print(f"      details: {contrarian_info.get('details', '')}")
        elif details.get('mean_reversion_trade'):
            mr_info = details.get('mean_reversion', {})
            print(f"   [MEAN_REVERSION] {action} - conf={confidence:.2f}")
            print(f"      details: {mr_info.get('details', '')}")
        else:
            failed = details.get('failed', [])
            print(f"   [CONSENSUS] HOLD - Failed: {failed}")
            for check_name, check_result in details.get('checks', {}).items():
                status = "PASS" if check_result.get('pass') else "FAIL"
                print(f"      {check_name}: {status} - {check_result.get('details', '')}")

    return action, confidence, details


# Module-level controller instance
_controller_instance = None
