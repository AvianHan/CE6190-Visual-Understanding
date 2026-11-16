"""Common model boundary: images and reference masks only, never query GT."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prediction:
    mask: Any  # H x W bool/0-1 numpy array at the ORIGINAL query resolution
    telemetry: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)  # JSON-safe concepts/decisions
    # Optional paired controls from the SAME model call/input/candidate pool.
    # Query GT and evaluation are exclusively owned by the runner. These masks
    # are nested controls, never additional independent episodes or GPU tasks.
    alternative_masks: dict = field(default_factory=dict)


class Adapter:
    def predict(self, inputs: dict, *, seed: int = 0) -> Prediction:
        """inputs = {query_image: absolute_path, references: [{image, mask}]}"""
        raise NotImplementedError

    def metadata(self) -> dict:
        return {}

    def close(self) -> None:
        pass
