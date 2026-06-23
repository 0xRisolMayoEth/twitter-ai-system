"""
core/logger.py — Setup logging terpusat.

Menulis ke console (INFO+) dan rotating file (DEBUG+).
Panggil setup_logger() sekali di main.py, lalu pakai get_logger() di modul lain.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).parent.parent


def setup_logger(name: str = "twitter_ai") -> logging.Logger:
    """
    Inisialisasi root logger dengan console + rotating file handler.
    Harus dipanggil sekali di awal program.
    """
    from core.config import load_config
    cfg = load_config().get("logging", {})

    log_file = ROOT / cfg.get("file", "logs/app.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level_str = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # sudah di-setup, hindari duplikat handler

    logger.setLevel(logging.DEBUG)  # tangkap semua, handler yang filter

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — level dari config (default INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file — selalu DEBUG agar mudah di-debug
    fh = RotatingFileHandler(
        log_file,
        maxBytes=cfg.get("max_bytes", 5 * 1024 * 1024),
        backupCount=cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Kembalikan child logger untuk modul tertentu."""
    return logging.getLogger(f"twitter_ai.{module_name}")
