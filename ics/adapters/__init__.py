"""Lazy adapter dispatch; importing the evaluation tools never loads a model."""
from .base import Adapter, Prediction


def create_adapter(config: dict) -> Adapter:
    config = dict(config)
    name = config.pop("name", config.pop("adapter", config.pop("method", None)))
    if name == "insid3":
        from .insid3 import create_adapter as create
    elif name == "cgics":
        from .cgics import create_adapter as create
    else:
        raise ValueError("adapter.name must be insid3 or cgics; no simulated adapter in research runs")
    return create(config)
