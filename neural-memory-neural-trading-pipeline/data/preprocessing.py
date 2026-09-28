from __future__ import annotations

import numpy as np


class RobustScaler:
    """Median/IQR scaler with optional clipping."""

    def __init__(self, clip_std: float = 6.0) -> None:
        self.clip_std = clip_std
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RobustScaler":
        self.median_ = np.median(x, axis=0)
        q1 = np.percentile(x, 25, axis=0)
        q3 = np.percentile(x, 75, axis=0)
        self.iqr_ = np.maximum(q3 - q1, 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("fit must be called before transform")
        z = (x - self.median_) / self.iqr_
        if self.clip_std is not None:
            z = np.clip(z, -self.clip_std, self.clip_std)
        return z

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
