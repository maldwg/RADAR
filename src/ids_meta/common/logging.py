from __future__ import annotations

import logging
import sys
import time


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging(level: str = "INFO") -> None:
    """Configure UTC timestamped package logging on standard output."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"invalid log level: {level}")
    root = logging.getLogger("ids_meta")
    root.setLevel(numeric_level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(UtcFormatter("%(asctime)sZ %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S"))
    root.addHandler(handler)


def logger(name: str) -> logging.Logger:
    """Return a consistently namespaced logger for a pipeline component."""
    return logging.getLogger(f"ids_meta.{name}")
