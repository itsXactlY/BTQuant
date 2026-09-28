from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
except Exception:  # pragma: no cover - optional dependency
    RandomForestRegressor = None
    permutation_importance = None


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    importances: dict[str, float]
    dropped_correlated: list[str]


class FeatureSelector:
    """Permutation-importance + correlation pruning selector."""

    def __init__(
        self,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.85,
        target_features: int = 15,
        random_state: int = 42,
    ) -> None:
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_features = target_features
        self.random_state = random_state

    def _compute_importance(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Iterable[str],
    ) -> dict[str, float]:
        names = list(feature_names)
        if RandomForestRegressor is None or permutation_importance is None:
            corr = np.abs(np.corrcoef(np.c_[x_val, y_val], rowvar=False)[-1, :-1])
            return {n: float(c) for n, c in zip(names, corr)}

        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(x_train, y_train)
        result = permutation_importance(
            model,
            x_val,
            y_val,
            n_repeats=10,
            random_state=self.random_state,
            scoring="r2",
            n_jobs=-1,
        )
        return {n: float(v) for n, v in zip(names, result.importances_mean)}

    def select(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Iterable[str],
    ) -> FeatureSelectionResult:
        names = list(feature_names)
        if x_train.shape[1] != len(names):
            raise ValueError("feature_names length must equal feature dimension")

        importance_map = self._compute_importance(x_train, y_train, x_val, y_val, names)
        candidates = [n for n in names if importance_map[n] >= self.importance_threshold]
        if not candidates:
            candidates = sorted(names, key=lambda n: importance_map[n], reverse=True)

        candidate_idx = [names.index(n) for n in candidates]
        x_candidates = x_val[:, candidate_idx]
        corr = np.corrcoef(x_candidates, rowvar=False)
        keep, dropped = set(candidates), []
        ordered = sorted(candidates, key=lambda n: importance_map[n], reverse=True)

        for i, left in enumerate(ordered):
            if left not in keep:
                continue
            for right in ordered[i + 1 :]:
                if right not in keep:
                    continue
                li, ri = candidates.index(left), candidates.index(right)
                if abs(float(corr[li, ri])) > self.correlation_threshold:
                    keep.remove(right)
                    dropped.append(right)

        final = sorted(keep, key=lambda n: importance_map[n], reverse=True)[: self.target_features]
        return FeatureSelectionResult(final, importance_map, dropped)
