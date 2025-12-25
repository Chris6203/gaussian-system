#!/usr/bin/env python3
"""
Data Manager for Gaussian Simulation System
Handles data verification, updates, and integrity checks

Config-aware version: Reads symbols from config.json
"""

import sqlite3
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data verification, validation, and updates"""
    
    def __init__(self, db_path: str = 'data/db/historical.db', config_path: str = 'config.json'):
        """
        Initialize data manager
        
        Args:
            db_path: Path to historical database
            config_path: Path to config.json
        """
        self.db_path = db_path
        
        # Default symbols including MEGA_CAP_TECH (matches config/symbols.py)
        DEFAULT_SYMBOLS = [
            # Core market indices
            'SPY', 'QQQ', 'IWM', 'DIA', '^VIX',
            # Mega-cap tech (SPY influencers)
            'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'PLTR', 'INTC', 'AMD',
            # Sectors
            'XLF', 'XLK', 'XLE', 'XLU', 'XLY', 'XLP', 'HYG', 'LQD',
            # Macro
            'TLT', 'IEF', 'SHY', 'UUP',
            # Crypto
            'BTC-USD', 'ETH-USD'
        ]
        
        # Load symbols from config
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.symbols = config.get('data_fetching', {}).get('symbols', DEFAULT_SYMBOLS)
            self.primary_symbol = config.get('trading', {}).get('symbol', 'SPY')
        except:
            # Fallback defaults
            self.symbols = DEFAULT_SYMBOLS
            self.primary_symbol = 'SPY'
        
        # Normalize VIX symbol (remove ^ if present)
        self.vix_symbol = 'VIX'
        
        # Ensure database exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create historical_data table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Create verification log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                last_verified DATETIME,
                record_count INTEGER,
                date_range_start DATETIME,
                date_range_end DATETIME,
                data_gaps INTEGER,
                status TEXT,
                notes TEXT,
                UNIQUE(symbol)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def verify_data_integrity(self, target_symbols: List[str] = None) -> Dict[str, Any]:
        """
        Verify all historical data integrity
        
        Args:
            target_symbols: Optional list of symbols to verify (defaults to all self.symbols)
        
        Returns:
            Dictionary with integrity check results
        """
        logger.info("Starting data integrity verification...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_checked': [],
            'issues': [],
            'summary': {}
        }
        
        if not os.path.exists(self.db_path):
            logger.error(f"Database not found at {self.db_path}")
            return {'error': 'Database not found', 'status': 'failed'}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        symbols_to_check = target_symbols if target_symbols is not None else self.symbols
        
        # Use 2025-01-01 as the cutoff for "valid" simulation data
        # Any data older than this is likely stale or irrelevant for current regime
        cutoff_date = "2025-01-01"
        
        for i, symbol in enumerate(symbols_to_check):
            if len(symbols_to_check) > 5 and i % 5 == 0:  # Show progress periodically
                logger.info(f"Verifying {symbol} ({i+1}/{len(symbols_to_check)})...")
                
            # Normalize symbol for DB lookup (VIX can be stored as VIX or ^VIX)
            normalized_symbol = symbol.lstrip('^')
            
            # --- CUSTOM VERIFICATION LOGIC ---
            symbol_issues = []
            
            # 1. Check for stale data (older than 2025)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM historical_data 
                WHERE symbol = ? AND timestamp < ?
            """, (normalized_symbol, cutoff_date))
            old_count = cursor.fetchone()[0]
            
            if old_count > 0:
                msg = f"Found {old_count} records older than {cutoff_date} (stale data)"
                symbol_issues.append(msg)
                logger.warning(f"⚠️ {symbol}: {msg}")
            
            # 2. Check basic stats
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM historical_data WHERE symbol = ?", (normalized_symbol,))
            min_ts, max_ts, count = cursor.fetchone()
            
            # 3. Check for duplicates
            cursor.execute("""
                SELECT timestamp, COUNT(*) 
                FROM historical_data 
                WHERE symbol = ? 
                GROUP BY timestamp 
                HAVING COUNT(*) > 1
            """, (normalized_symbol,))
            dupes = cursor.fetchall()
            if dupes:
                msg = f"Found {len(dupes)} duplicate timestamps"
                symbol_issues.append(msg)
                logger.warning(f"⚠️ {symbol}: {msg}")

            symbol_result = {
                'symbol': symbol,
                'record_count': count or 0,
                'date_range': (min_ts, max_ts),
                'issues': symbol_issues
            }
            
            results['symbols_checked'].append(symbol_result)
            
            if symbol_issues:
                results['issues'].extend([f"{symbol}: {issue}" for issue in symbol_issues])
        
        conn.close()
        
        # Generate summary
        total_records = sum(r.get('record_count', 0) for r in results['symbols_checked'])
        results['summary'] = {
            'total_records': total_records,
            'symbols_verified': len(results['symbols_checked']),
            'issues_found': len(results['issues']),
            'status': 'valid' if not results['issues'] else 'needs_attention'
        }
        
        return results
    
    def _verify_symbol_data(self, cursor: sqlite3.Cursor, symbol: str) -> Dict[str, Any]:
        """Verify data for a specific symbol"""
        result = {
            'symbol': symbol,
            'record_count': 0,
            'date_range': (None, None),
            'gaps': 0,
            'issues': [],
            'last_verified': datetime.now().isoformat()
        }
        
        try:
            # Get record count - check both with and without ^ prefix (e.g., VIX and ^VIX)
            symbol_variants = [symbol, f'^{symbol}', symbol.lstrip('^')]
            symbol_variants = list(set(symbol_variants))  # Remove duplicates
            placeholders = ','.join(['?' for _ in symbol_variants])
            
            cursor.execute(f"SELECT COUNT(*) FROM historical_data WHERE symbol IN ({placeholders})", symbol_variants)
            result['record_count'] = cursor.fetchone()[0]
            
            if result['record_count'] == 0:
                result['issues'].append(f"No data found for {symbol}")
                return result
            
            # Get date range
            cursor.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM historical_data WHERE symbol IN ({placeholders})",
                symbol_variants
            )
            min_date, max_date = cursor.fetchone()
            result['date_range'] = (min_date, max_date)
            
            # Check for NULL values
            cursor.execute(f"""
                SELECT COUNT(*) FROM historical_data 
                WHERE symbol IN ({placeholders}) AND (close_price IS NULL OR volume IS NULL)
            """, symbol_variants)
            null_count = cursor.fetchone()[0]
            if null_count > 0:
                result['issues'].append(f"{null_count} records with NULL values")
            
            # Check for duplicates
            cursor.execute(f"""
                SELECT timestamp, COUNT(*) as cnt FROM historical_data 
                WHERE symbol IN ({placeholders}) GROUP BY timestamp HAVING cnt > 1
            """, symbol_variants)
            duplicates = cursor.fetchall()
            if duplicates:
                result['issues'].append(f"{len(duplicates)} duplicate timestamps")
            
            # Check for data gaps (use first variant that has data)
            result['gaps'] = self._check_data_gaps(symbol_variants[0] if '^' in symbol else symbol)
            if result['gaps'] > 0:
                result['issues'].append(f"{result['gaps']} data gaps detected")
            
            # Check price reasonableness
            cursor.execute(f"""
                SELECT COUNT(*) FROM historical_data 
                WHERE symbol IN ({placeholders}) AND (close_price <= 0 OR volume < 0)
            """, symbol_variants)
            invalid_count = cursor.fetchone()[0]
            if invalid_count > 0:
                result['issues'].append(f"{invalid_count} records with invalid prices/volume")
            
        except Exception as e:
            result['issues'].append(f"Error during verification: {str(e)}")
            logger.error(f"Error verifying {symbol}: {e}")
        
        return result
    
    def _check_data_gaps(self, symbol: str, expected_frequency: str = 'daily') -> int:
        """Check for gaps in data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query(
                "SELECT timestamp FROM historical_data WHERE symbol = ? ORDER BY timestamp",
                conn,
                params=(symbol,)
            )
            conn.close()
            
            if len(df) < 2:
                return 0
            
            # Parse timestamps with infer_datetime_format to handle ISO8601
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
            df['diff'] = df['timestamp'].diff()
            
            # For intraday data, expected diff is 1 minute
            expected_diff = pd.Timedelta(minutes=1)
            
            gaps = (df['diff'] > expected_diff).sum()
            return int(gaps)
        except Exception as e:
            logger.warning(f"Could not check gaps for {symbol}: {e}")
            return 0
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all data in the system"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol in self.symbols:
            # Check both with and without ^ prefix (e.g., VIX and ^VIX)
            symbol_variants = [symbol, f'^{symbol}', symbol.lstrip('^')]
            symbol_variants = list(set(symbol_variants))  # Remove duplicates
            placeholders = ','.join(['?' for _ in symbol_variants])
            
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as records,
                    MIN(timestamp) as first_date,
                    MAX(timestamp) as last_date,
                    AVG(close_price) as avg_price,
                    MIN(close_price) as min_price,
                    MAX(close_price) as max_price
                FROM historical_data 
                WHERE symbol IN ({placeholders})
            """, symbol_variants)
            
            row = cursor.fetchone()
            summary['symbols'][symbol] = {
                'records': row[0] if row[0] else 0,
                'first_date': row[1],
                'last_date': row[2],
                'avg_price': round(row[3], 4) if row[3] else 0,
                'min_price': round(row[4], 4) if row[4] else 0,
                'max_price': round(row[5], 4) if row[5] else 0,
            }
        
        conn.close()
        return summary
    
    def update_data_verification_log(self, verification_result: Dict[str, Any]):
        """Log verification results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for symbol_check in verification_result['symbols_checked']:
                cursor.execute("""
                    INSERT OR REPLACE INTO data_verification 
                    (symbol, last_verified, record_count, date_range_start, date_range_end, 
                     data_gaps, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol_check['symbol'],
                    symbol_check['last_verified'],
                    symbol_check['record_count'],
                    symbol_check['date_range'][0],
                    symbol_check['date_range'][1],
                    symbol_check['gaps'],
                    'error' if symbol_check['issues'] else 'valid',
                    ', '.join(symbol_check['issues']) if symbol_check['issues'] else None
                ))
            
            conn.commit()
            conn.close()
            logger.info("Data verification log updated")
        except Exception as e:
            logger.error(f"Error updating verification log: {e}")
    
    def export_data_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate and export comprehensive data report
        
        Args:
            output_file: Optional path to save JSON report
        
        Returns:
            Dictionary with report data
        """
        logger.info("Generating data report...")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'integrity_check': self.verify_data_integrity(),
            'data_summary': self.get_data_summary(),
        }
        
        if output_file:
            try:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Data report exported to {output_file}")
            except Exception as e:
                logger.error(f"Error exporting report: {e}")
        
        return report
    
    def validate_simulation_ready(self) -> Tuple[bool, List[str]]:
        """
        Check if simulation has all required data
        
        Returns:
            Tuple of (is_ready, list_of_issues)
        """
        issues = []
        
        # Check historical data exists
        # OPTIMIZATION: Only verify primary symbol and VIX to save time
        target_symbols = [self.primary_symbol, self.vix_symbol]
        # Also include any symbols that might be used for features if needed, 
        # but primarily we need these two to start.
        
        integrity = self.verify_data_integrity(target_symbols=target_symbols)
        
        primary_symbol_data = next(
            (s for s in integrity['symbols_checked'] if s['symbol'] == self.primary_symbol),
            None
        )
        
        if not primary_symbol_data or primary_symbol_data['record_count'] == 0:
            issues.append(f"No historical data for primary symbol {self.primary_symbol}")
        
        if primary_symbol_data and primary_symbol_data['issues']:
            # Filter out gap warnings - intraday data naturally has gaps during market closure
            filtered_issues = [
                issue for issue in primary_symbol_data['issues'] 
                if 'gap' not in issue.lower()
            ]
            issues.extend(filtered_issues)
        
        # Check VIX data
        vix_data = next(
            (s for s in integrity['symbols_checked'] if self.vix_symbol in s['symbol']),
            None
        )
        
        if not vix_data or vix_data['record_count'] == 0:
            issues.append(f"No historical data for {self.vix_symbol}")
        
        return len(issues) == 0, issues
    
    def print_status(self):
        """Print data manager status"""
        print("\n" + "=" * 70)
        print("DATA MANAGER STATUS")
        print("=" * 70)
        
        print("\nData Summary:")
        summary = self.get_data_summary()
        for symbol, info in summary['symbols'].items():
            print(f"\n  {symbol}:")
            print(f"    Records: {info['records']:,}")
            print(f"    Date Range: {info['first_date']} to {info['last_date']}")
            if info['max_price'] > 0:
                print(f"    Price Range: ${info['min_price']:.2f} - ${info['max_price']:.2f}")
        
        print("\nData Integrity:")
        is_ready, issues = self.validate_simulation_ready()
        print(f"  Simulation Ready: {'✓ YES' if is_ready else '✗ NO'}")
        
        if issues:
            print("  Issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  All checks passed!")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test data manager
    manager = DataManager()
    manager.print_status()
    
    # Generate and save report
    report = manager.export_data_report(Path('data_report.json'))
    print("\nReport generated successfully!")

