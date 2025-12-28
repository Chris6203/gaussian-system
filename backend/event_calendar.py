#!/usr/bin/env python3
"""
Economic Event Calendar for Trading Halts

Jerry's "Event-Driven Halts" feature:
- Automatically halt trading during high-volatility economic events
- CPI, FOMC releases cause massive price swings
- Block entries pre/post event to avoid unpredictable moves

Usage:
    from backend.event_calendar import should_halt_for_event, EventCalendar

    if should_halt_for_event(datetime.now()):
        # Skip trading - event window active
        pass
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# 2025-2026 Economic Event Schedule
# =============================================================================

# CPI Release Dates (8:30 AM ET) - Bureau of Labor Statistics
CPI_DATES_2025 = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-13",
    "2025-09-11", "2025-10-10", "2025-11-13", "2025-12-10"
]

CPI_DATES_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-09",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-10", "2026-10-13", "2026-11-12", "2026-12-10"
]

# FOMC Meeting Dates (2:00 PM ET announcement) - Federal Reserve
# These are the 2-day meeting end dates when announcements are made
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]

FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
]

# Jobs Report (NFP) Dates (8:30 AM ET, first Friday of month)
NFP_DATES_2025 = [
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05"
]

NFP_DATES_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"
]


class EventCalendar:
    """
    Manages economic event schedule and halt windows.

    Jerry's Implementation:
        t ∈ {CPI, FOMC} ⇒ Global Halt

    Default windows:
        - CPI: 30 min before, 60 min after 8:30 AM release
        - FOMC: 30 min before, 120 min after 2:00 PM release (more volatile)
        - NFP: 30 min before, 60 min after 8:30 AM release
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize event calendar.

        Config options (jerry.event_halts section):
            enabled: bool - Enable/disable event halts
            pre_minutes: int - Minutes before event to start halt
            post_minutes: int - Minutes after event to end halt
            events: List[str] - Which events to halt for ["cpi", "fomc", "nfp"]
        """
        self.config = config or {}

        # Load from jerry config section if available
        jerry_cfg = self.config.get('jerry', {}).get('event_halts', {})

        self.enabled = jerry_cfg.get('enabled', True)
        self.pre_minutes = jerry_cfg.get('pre_minutes', 30)
        self.post_minutes_cpi = jerry_cfg.get('post_minutes_cpi', 60)
        self.post_minutes_fomc = jerry_cfg.get('post_minutes_fomc', 120)  # Longer for FOMC
        self.post_minutes_nfp = jerry_cfg.get('post_minutes_nfp', 60)
        self.events_to_halt = jerry_cfg.get('events', ['cpi', 'fomc', 'nfp'])

        # Build event index
        self._build_event_index()

        logger.info(f"📅 Event Calendar initialized (enabled={self.enabled})")
        if self.enabled:
            logger.info(f"   Halt events: {self.events_to_halt}")
            logger.info(f"   Pre-event buffer: {self.pre_minutes} min")
            logger.info(f"   Post-CPI: {self.post_minutes_cpi} min, Post-FOMC: {self.post_minutes_fomc} min")

    def _build_event_index(self):
        """Build fast lookup index for events."""
        self.events = []

        # CPI events (8:30 AM ET)
        if 'cpi' in self.events_to_halt:
            for date_str in CPI_DATES_2025 + CPI_DATES_2026:
                event_time = datetime.strptime(f"{date_str} 08:30", "%Y-%m-%d %H:%M")
                self.events.append({
                    'type': 'CPI',
                    'time': event_time,
                    'pre_minutes': self.pre_minutes,
                    'post_minutes': self.post_minutes_cpi
                })

        # FOMC events (2:00 PM ET)
        if 'fomc' in self.events_to_halt:
            for date_str in FOMC_DATES_2025 + FOMC_DATES_2026:
                event_time = datetime.strptime(f"{date_str} 14:00", "%Y-%m-%d %H:%M")
                self.events.append({
                    'type': 'FOMC',
                    'time': event_time,
                    'pre_minutes': self.pre_minutes,
                    'post_minutes': self.post_minutes_fomc
                })

        # NFP events (8:30 AM ET)
        if 'nfp' in self.events_to_halt:
            for date_str in NFP_DATES_2025 + NFP_DATES_2026:
                event_time = datetime.strptime(f"{date_str} 08:30", "%Y-%m-%d %H:%M")
                self.events.append({
                    'type': 'NFP',
                    'time': event_time,
                    'pre_minutes': self.pre_minutes,
                    'post_minutes': self.post_minutes_nfp
                })

        # Sort by time
        self.events.sort(key=lambda x: x['time'])

        logger.info(f"   Loaded {len(self.events)} economic events")

    def is_in_halt_window(self, current_time: datetime) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current time is in an event halt window.

        Returns:
            (is_halted, event_info) - True if in halt window, with event details
        """
        if not self.enabled:
            return False, None

        for event in self.events:
            event_time = event['time']
            halt_start = event_time - timedelta(minutes=event['pre_minutes'])
            halt_end = event_time + timedelta(minutes=event['post_minutes'])

            if halt_start <= current_time <= halt_end:
                return True, {
                    'type': event['type'],
                    'event_time': event_time,
                    'halt_start': halt_start,
                    'halt_end': halt_end,
                    'minutes_to_event': int((event_time - current_time).total_seconds() / 60),
                    'minutes_since_event': int((current_time - event_time).total_seconds() / 60)
                }

        return False, None

    def get_next_event(self, current_time: datetime) -> Optional[Dict]:
        """Get the next upcoming event."""
        for event in self.events:
            if event['time'] > current_time:
                return {
                    'type': event['type'],
                    'time': event['time'],
                    'minutes_until': int((event['time'] - current_time).total_seconds() / 60)
                }
        return None

    def get_events_in_range(self, start: datetime, end: datetime) -> List[Dict]:
        """Get all events in a date range."""
        return [
            e for e in self.events
            if start <= e['time'] <= end
        ]


# =============================================================================
# Global Calendar Instance
# =============================================================================

_calendar_instance: Optional[EventCalendar] = None
_simulation_time: Optional[datetime] = None  # For backtesting


def get_event_calendar(config: Optional[Dict] = None) -> EventCalendar:
    """Get or create the global event calendar instance."""
    global _calendar_instance

    if _calendar_instance is None:
        # Try to load config from file
        if config is None:
            config_path = Path('config.json')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load config.json: {e}")
                    config = {}

        _calendar_instance = EventCalendar(config)

    return _calendar_instance


def set_simulation_time(sim_time: datetime):
    """Set the simulation time for backtesting."""
    global _simulation_time
    _simulation_time = sim_time


def get_current_time() -> datetime:
    """Get current time (simulation time if set, otherwise real time)."""
    global _simulation_time
    if _simulation_time is not None:
        return _simulation_time
    return datetime.now()


def should_halt_for_event(current_time: Optional[datetime] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Quick check if we should halt trading due to economic event.

    Usage:
        halted, event_info = should_halt_for_event()
        if halted:
            logger.info(f"Halting for {event_info['type']} event")
            return HOLD

    Returns:
        (should_halt, event_info) - True if in event window
    """
    if current_time is None:
        current_time = get_current_time()

    calendar = get_event_calendar()
    return calendar.is_in_halt_window(current_time)


def reset_calendar():
    """Reset the global calendar instance (for testing)."""
    global _calendar_instance, _simulation_time
    _calendar_instance = None
    _simulation_time = None
