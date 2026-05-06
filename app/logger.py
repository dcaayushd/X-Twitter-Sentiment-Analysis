"""Centralized logging setup."""

from __future__ import annotations

import logging


def configure_logging(level: str, fmt: str) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(level=level, format=fmt, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the shared application configuration."""
    return logging.getLogger(name)
