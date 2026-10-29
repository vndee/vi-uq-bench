"""Stand-ins for the two lm-polygraph utilities the vendored modules import.

The estimators only use Model / WhiteboxModel as type hints, and
polygraph_module_init only logs the class name on construction.
"""
import logging
from typing import Any

log = logging.getLogger("lm_polygraph")

Model = Any
WhiteboxModel = Any


def polygraph_module_init(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "__init__":
            log.info(f"Initializing {args[0].__class__.__name__}")
        func(*args, **kwargs)

    return wrapper
