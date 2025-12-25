#!/usr/bin/env python3
"""
Secure Configuration Management
===============================

Handles API keys and secrets with priority:
1. Environment variables (most secure)
2. .env file (local development)
3. config.json (fallback, NOT recommended for production)

Environment Variables:
    TRADIER_ACCESS_TOKEN      - Tradier API token
    TRADIER_ACCOUNT_NUMBER    - Tradier account number
    TRADIER_SANDBOX_TOKEN     - Tradier sandbox token
    TRADIER_SANDBOX_ACCOUNT   - Tradier sandbox account
    POLYGON_API_KEY           - Polygon.io API key
    ALPHA_VANTAGE_API_KEY     - Alpha Vantage API key
    FMP_API_KEY               - Financial Modeling Prep API key
    SLACK_WEBHOOK_URL         - Slack alerts webhook
    ALERT_EMAIL_SMTP          - Email SMTP server
    ALERT_EMAIL_USER          - Email username
    ALERT_EMAIL_PASSWORD      - Email password

Usage:
    from backend.secure_config import SecureConfig
    
    config = SecureConfig()
    
    # Get credentials (env var > .env > config.json)
    tradier_token = config.get_credential('tradier', 'access_token')
    polygon_key = config.get_credential('polygon', 'api_key')
    
    # Check if using secure source
    is_secure = config.is_credential_from_env('tradier', 'access_token')
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CredentialSource:
    """Tracks where a credential came from."""
    value: str
    source: str  # 'env', 'dotenv', 'config', 'default'
    is_secure: bool


# Environment variable mappings
ENV_VAR_MAPPINGS = {
    # Tradier
    ('tradier', 'access_token'): 'TRADIER_ACCESS_TOKEN',
    ('tradier', 'account_number'): 'TRADIER_ACCOUNT_NUMBER',
    ('tradier', 'sandbox_token'): 'TRADIER_SANDBOX_TOKEN',
    ('tradier', 'sandbox_account'): 'TRADIER_SANDBOX_ACCOUNT',
    ('tradier', 'live_token'): 'TRADIER_LIVE_TOKEN',
    ('tradier', 'live_account'): 'TRADIER_LIVE_ACCOUNT',
    
    # Data providers
    ('polygon', 'api_key'): 'POLYGON_API_KEY',
    ('alpha_vantage', 'api_key'): 'ALPHA_VANTAGE_API_KEY',
    ('fmp', 'api_key'): 'FMP_API_KEY',
    
    # Alerts
    ('alerts', 'slack_webhook'): 'SLACK_WEBHOOK_URL',
    ('alerts', 'email_smtp'): 'ALERT_EMAIL_SMTP',
    ('alerts', 'email_user'): 'ALERT_EMAIL_USER',
    ('alerts', 'email_password'): 'ALERT_EMAIL_PASSWORD',
}


class SecureConfig:
    """
    Secure configuration manager.
    
    Prioritizes environment variables over config files for sensitive data.
    """
    
    def __init__(
        self,
        config_path: str = 'config.json',
        dotenv_path: str = '.env',
        warn_on_insecure: bool = True
    ):
        """
        Args:
            config_path: Path to config.json
            dotenv_path: Path to .env file
            warn_on_insecure: Warn when using config.json for secrets
        """
        self.config_path = Path(config_path)
        self.dotenv_path = Path(dotenv_path)
        self.warn_on_insecure = warn_on_insecure
        
        self._config: Dict = {}
        self._dotenv: Dict = {}
        self._credential_sources: Dict[tuple, CredentialSource] = {}
        
        # Load sources
        self._load_config()
        self._load_dotenv()
        
        # Log security status
        self._log_security_status()
    
    def _load_config(self) -> None:
        """Load config.json."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    self._config = json.load(f)
                logger.debug(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load config.json: {e}")
    
    def _load_dotenv(self) -> None:
        """Load .env file if it exists."""
        if self.dotenv_path.exists():
            try:
                with open(self.dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            self._dotenv[key.strip()] = value.strip().strip('"\'')
                logger.debug(f"Loaded .env from {self.dotenv_path}")
            except Exception as e:
                logger.warning(f"Could not load .env: {e}")
    
    def _log_security_status(self) -> None:
        """Log which credentials are secure vs insecure."""
        secure_count = 0
        insecure_count = 0
        
        for (provider, key), env_var in ENV_VAR_MAPPINGS.items():
            if os.environ.get(env_var) or self._dotenv.get(env_var):
                secure_count += 1
            elif self._get_from_config(provider, key):
                insecure_count += 1
        
        if insecure_count > 0 and self.warn_on_insecure:
            logger.warning(
                f"⚠️ {insecure_count} credentials loaded from config.json (not secure). "
                f"Use environment variables for production."
            )
        
        if secure_count > 0:
            logger.info(f"✅ {secure_count} credentials loaded from environment variables")
    
    def get_credential(
        self,
        provider: str,
        key: str,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a credential with priority: env var > .env > config.json > default.
        
        Args:
            provider: Provider name (e.g., 'tradier', 'polygon')
            key: Key name (e.g., 'api_key', 'access_token')
            default: Default value if not found
            
        Returns:
            Credential value or default
        """
        cache_key = (provider, key)
        
        # Check cache
        if cache_key in self._credential_sources:
            return self._credential_sources[cache_key].value
        
        value = None
        source = 'default'
        is_secure = False
        
        # 1. Check environment variable
        env_var = ENV_VAR_MAPPINGS.get(cache_key)
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                value = env_value
                source = 'env'
                is_secure = True
            else:
                # 2. Check .env file
                dotenv_value = self._dotenv.get(env_var)
                if dotenv_value:
                    value = dotenv_value
                    source = 'dotenv'
                    is_secure = True
        
        # 3. Check config.json
        if value is None:
            config_value = self._get_from_config(provider, key)
            if config_value:
                value = config_value
                source = 'config'
                is_secure = False
                
                if self.warn_on_insecure:
                    logger.debug(
                        f"⚠️ {provider}.{key} loaded from config.json. "
                        f"Set {env_var or 'env var'} for better security."
                    )
        
        # 4. Use default
        if value is None:
            value = default
            source = 'default'
            is_secure = True  # Default is not sensitive
        
        # Cache result
        if value:
            self._credential_sources[cache_key] = CredentialSource(
                value=value,
                source=source,
                is_secure=is_secure
            )
        
        return value
    
    def _get_from_config(self, provider: str, key: str) -> Optional[str]:
        """Get credential from config.json."""
        credentials = self._config.get('credentials', {})
        
        # Try direct path
        provider_config = credentials.get(provider, {})
        if isinstance(provider_config, dict):
            # Check top level
            if key in provider_config:
                return provider_config[key]
            
            # Check sandbox/live sections for Tradier
            if provider == 'tradier':
                if key == 'access_token':
                    return provider_config.get('sandbox', {}).get('access_token')
                elif key == 'account_number':
                    return provider_config.get('sandbox', {}).get('account_number')
                elif key == 'live_token':
                    return provider_config.get('live', {}).get('access_token')
                elif key == 'live_account':
                    return provider_config.get('live', {}).get('account_number')
        
        return None
    
    def is_credential_from_env(self, provider: str, key: str) -> bool:
        """Check if credential is from environment (secure)."""
        cache_key = (provider, key)
        if cache_key in self._credential_sources:
            return self._credential_sources[cache_key].is_secure
        
        # Trigger loading
        self.get_credential(provider, key)
        if cache_key in self._credential_sources:
            return self._credential_sources[cache_key].is_secure
        
        return False
    
    def get_credential_source(self, provider: str, key: str) -> str:
        """Get the source of a credential."""
        cache_key = (provider, key)
        if cache_key in self._credential_sources:
            return self._credential_sources[cache_key].source
        
        # Trigger loading
        self.get_credential(provider, key)
        if cache_key in self._credential_sources:
            return self._credential_sources[cache_key].source
        
        return 'unknown'
    
    def get_config(self, *keys, default: Any = None) -> Any:
        """
        Get non-sensitive config value.
        
        Args:
            *keys: Path to config value (e.g., 'trading', 'symbol')
            default: Default value
            
        Returns:
            Config value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
    
    def get_tradier_credentials(self, sandbox: bool = True) -> Dict[str, str]:
        """
        Get Tradier credentials for trading.
        
        Args:
            sandbox: Use sandbox (True) or live (False) credentials
            
        Returns:
            Dict with 'access_token' and 'account_number'
        """
        if sandbox:
            return {
                'access_token': self.get_credential('tradier', 'sandbox_token') or 
                               self.get_credential('tradier', 'access_token'),
                'account_number': self.get_credential('tradier', 'sandbox_account') or
                                 self.get_credential('tradier', 'account_number')
            }
        else:
            return {
                'access_token': self.get_credential('tradier', 'live_token') or
                               self.get_credential('tradier', 'access_token'),
                'account_number': self.get_credential('tradier', 'live_account') or
                                 self.get_credential('tradier', 'account_number')
            }
    
    def get_data_provider_key(self, provider: str) -> Optional[str]:
        """Get API key for a data provider."""
        return self.get_credential(provider, 'api_key')
    
    def get_security_report(self) -> Dict:
        """Get a report on credential security status."""
        report = {
            'secure_credentials': [],
            'insecure_credentials': [],
            'missing_credentials': []
        }
        
        for (provider, key), env_var in ENV_VAR_MAPPINGS.items():
            full_key = f"{provider}.{key}"
            
            if os.environ.get(env_var) or self._dotenv.get(env_var):
                report['secure_credentials'].append(full_key)
            elif self._get_from_config(provider, key):
                report['insecure_credentials'].append({
                    'key': full_key,
                    'env_var': env_var,
                    'recommendation': f"Set {env_var} environment variable"
                })
            else:
                report['missing_credentials'].append({
                    'key': full_key,
                    'env_var': env_var
                })
        
        report['summary'] = {
            'total': len(ENV_VAR_MAPPINGS),
            'secure': len(report['secure_credentials']),
            'insecure': len(report['insecure_credentials']),
            'missing': len(report['missing_credentials'])
        }
        
        return report


# Global instance
_secure_config: Optional[SecureConfig] = None


def get_secure_config() -> SecureConfig:
    """Get or create global SecureConfig instance."""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfig()
    return _secure_config


def print_security_report():
    """Print a security report to console."""
    config = get_secure_config()
    report = config.get_security_report()
    
    print("\n" + "=" * 60)
    print("CREDENTIAL SECURITY REPORT")
    print("=" * 60)
    
    print(f"\n✅ Secure ({len(report['secure_credentials'])}):")
    for key in report['secure_credentials']:
        print(f"   - {key}")
    
    print(f"\n⚠️ Insecure - In config.json ({len(report['insecure_credentials'])}):")
    for item in report['insecure_credentials']:
        print(f"   - {item['key']}")
        print(f"     Fix: export {item['env_var']}='your_key'")
    
    print(f"\n❓ Missing ({len(report['missing_credentials'])}):")
    for item in report['missing_credentials']:
        print(f"   - {item['key']} ({item['env_var']})")
    
    print("\n" + "-" * 60)
    summary = report['summary']
    print(f"Total: {summary['total']} | Secure: {summary['secure']} | "
          f"Insecure: {summary['insecure']} | Missing: {summary['missing']}")
    print("=" * 60 + "\n")


# Create .env.example template
def create_env_template(output_path: str = '.env.example'):
    """Create a .env.example template file."""
    template = """# Trading Bot Environment Variables
# Copy this to .env and fill in your values
# NEVER commit .env to version control!

# Tradier API (https://developer.tradier.com/)
TRADIER_ACCESS_TOKEN=your_sandbox_token_here
TRADIER_ACCOUNT_NUMBER=your_sandbox_account_here
TRADIER_LIVE_TOKEN=your_live_token_here
TRADIER_LIVE_ACCOUNT=your_live_account_here

# Polygon.io (https://polygon.io/)
POLYGON_API_KEY=your_polygon_key_here

# Alpha Vantage (https://www.alphavantage.co/)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Financial Modeling Prep (https://financialmodelingprep.com/)
FMP_API_KEY=your_fmp_key_here

# Alerts (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
ALERT_EMAIL_SMTP=smtp.gmail.com
ALERT_EMAIL_USER=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password
"""
    
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Created {output_path}")
    print("Copy to .env and fill in your credentials.")


if __name__ == '__main__':
    # Run security report when executed directly
    print_security_report()
    
    # Create template if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--create-template':
        create_env_template()









