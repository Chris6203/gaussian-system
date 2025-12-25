import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    base_dir: Path
    config_path: Path
    db_path: Path
    log_dir: Path
    host: str
    port: int


def load_settings(base_dir: Path) -> Settings:
    # Load env (if present)
    load_dotenv(base_dir / ".env", override=False)

    config_path = Path(os.getenv("DM_CONFIG_PATH", str(base_dir / "config.json")))
    db_path = Path(os.getenv("DM_DB_PATH", str(base_dir / "data" / "db" / "historical.db")))
    log_dir = Path(os.getenv("DM_LOG_DIR", str(base_dir / "logs")))
    host = os.getenv("DM_HOST", "0.0.0.0")
    port = int(os.getenv("DM_PORT", "5050"))

    return Settings(
        base_dir=base_dir,
        config_path=config_path,
        db_path=db_path,
        log_dir=log_dir,
        host=host,
        port=port,
    )


def load_config(settings: Settings) -> Dict[str, Any]:
    try:
        with open(settings.config_path, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = json.load(f)
    except Exception:
        cfg = {}

    # Env overrides for API keys
    tradier = cfg.setdefault("credentials", {}).setdefault("tradier", {})

    env_data_token = os.getenv("TRADIER_DATA_API_TOKEN")
    if env_data_token:
        tradier["data_api_token"] = env_data_token

    env_access_token = os.getenv("TRADIER_ACCESS_TOKEN")
    env_account = os.getenv("TRADIER_ACCOUNT_NUMBER")
    is_sandbox = _as_bool(os.getenv("TRADIER_IS_SANDBOX"), default=True)

    if env_access_token:
        key = "sandbox" if is_sandbox else "live"
        tradier.setdefault(key, {})
        tradier[key]["access_token"] = env_access_token
        if env_account:
            tradier[key]["account_number"] = env_account

    return cfg


def save_config(settings: Settings, cfg: Dict[str, Any]) -> None:
    settings.config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
