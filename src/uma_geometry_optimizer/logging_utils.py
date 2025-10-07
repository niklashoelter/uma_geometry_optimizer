"""Logging utilities for uma_geometry_optimizer.

This module provides a single entry point to configure logging for the
application/CLI, while library modules just acquire loggers with
logging.getLogger(__name__).

- By default, library modules do not configure handlers. __init__ adds a
  NullHandler to avoid warnings when users don't configure logging.
- The CLI should call configure_logging() early to set up a StreamHandler
  and formatting, and to set levels for third-party libraries (fairchem,
  torch, torchsim, rdkit, ase) to reduce noise unless verbose.
"""

from __future__ import annotations

import logging
from typing import Optional

# Reasonable defaults
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _root_has_handlers() -> bool:
    root = logging.getLogger()
    return bool(root.handlers)


def configure_logging(level: Optional[int] = None, *, verbose: Optional[bool] = None, quiet: Optional[bool] = None) -> None:
    """Configure root logging for the CLI/application.

    Parameters:
        level: Explicit logging level (e.g., logging.INFO). If provided, overrides verbose/quiet.
        verbose: If True, set INFO level. If False and quiet is None, leaves default WARNING.
        quiet: If True, set ERROR level.

    Behavior:
        - Adds a StreamHandler to the root logger if none exists.
        - Sets a consistent formatter.
        - Tunes third-party library loggers to WARNING to reduce noise.
    """
    # Resolve desired level
    resolved_level: int
    if level is not None:
        resolved_level = level
    else:
        if quiet:
            resolved_level = logging.ERROR
        elif verbose:
            resolved_level = logging.INFO
        else:
            resolved_level = logging.WARNING

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if not _root_has_handlers():
        handler = logging.StreamHandler()
        handler.setLevel(resolved_level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root_logger.addHandler(handler)

    # Always ensure handler levels reflect resolved level
    for h in root_logger.handlers:
        h.setLevel(resolved_level)

    # Third-party libraries can be noisy; keep them at WARNING unless user goes DEBUG explicitly
    third_party_loggers = [
        "fairchem",
        "torch",
        "torchsim",
        "torch_sim",
        "torch-sim",
        "torch-sim-atomistic"
        "rdkit",
        "ase",
    ]
    for name in third_party_loggers:
        lgr = logging.getLogger(name)
        # Set WARNING by default; if user explicitly set DEBUG, allow INFO for third parties as well
        lgr.setLevel(logging.WARNING if resolved_level < logging.DEBUG else logging.INFO)
        lgr.propagate = True


__all__ = ["configure_logging"]
