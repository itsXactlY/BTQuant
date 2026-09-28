from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTradingModel(ABC):
    @abstractmethod
    def predict(self, sequence):
        """Return prediction dict for single sequence."""
