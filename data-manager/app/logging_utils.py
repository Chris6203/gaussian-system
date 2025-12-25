import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def build_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / f"{name.lower()}.log"),
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y%m%d"
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
