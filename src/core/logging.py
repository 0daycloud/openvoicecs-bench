"""Structured logging for the self-training pipeline."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging with rich output."""
    global _configured

    if not _configured:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    show_path=False,
                )
            ],
        )
        _configured = True

    return logging.getLogger("pd_dental")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(f"pd_dental.{name}")
