# ==============================================================================
# FIXED FEATURE EXTRACTOR + OPTIMIZED DATA PIPELINE
# ==============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import signal, stats
from sklearn.preprocessing import RobustScaler
from rich.console import Console
import warnings

console = Console()
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# INDICATOR FEATURE EXTRACTOR
# ==============================================================================

class IndicatorFeatureExtractor:
    """
    Extracts deep features from indicator internals.
    Fully robust to NaN/Inf, precision loss, or type mismatches.
    """

    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50]):
        self.lookback_windows = lookback_windows
        self.scaler = RobustScaler()
        self.feature_names = []

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------
    def _safe_float(self, x):
        """Convert any value safely to float."""
        try:
            f = float(x)
            return f if np.isfinite(f) else 0.0
        except Exception:
            return 0.0

    # --------------------------------------------------------------------------
    # Core feature extraction
    # --------------------------------------------------------------------------
    def extract_all_features(self, indicator_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extracts numerical feature vector from indicator internals.
        Args:
            indicator_data: Dict[str, np.ndarray]
        Returns:
            np.ndarray: 1D feature vector (float32)
        """
        features: List[float] = []

        # === 1. RAW VALUES =====================================================
        for key, values in indicator_data.items():
            if key not in ['bar', 'datetime'] and len(values) > 0:
                features.append(self._safe_float(values[-1]))

        # === 2. TEMPORAL DERIVATIVES ==========================================
        for key, values in indicator_data.items():
            if key in ['bar', 'datetime']:
                continue
            for window in [1, 3, 5, 10]:
                if len(values) > window:
                    try:
                        diff = self._safe_float(values[-1]) - self._safe_float(values[-window - 1])
                        accel = 0.0
                        if len(values) > window * 2:
                            vel_now = self._safe_float(values[-1]) - self._safe_float(values[-window - 1])
                            vel_prev = self._safe_float(values[-window - 1]) - self._safe_float(values[-window * 2 - 1])
                            accel = vel_now - vel_prev
                        features.extend([diff, accel])
                    except Exception:
                        features.extend([0.0, 0.0])

        # === 3. STATISTICAL MOMENTS ===========================================
        for key, values in indicator_data.items():
            if key in ['bar', 'datetime']:
                continue
            for window in self.lookback_windows:
                if len(values) >= window:
                    try:
                        window_data = np.asarray(values[-window:], dtype=np.float64)
                        window_data = window_data[np.isfinite(window_data)]
                        if len(window_data) < 3:
                            features.extend([0.0] * 6)
                            continue

                        mean_val = np.mean(window_data)
                        std_val = np.std(window_data)
                        if std_val > 1e-10:
                            with np.errstate(all='ignore'):
                                skew_val = stats.skew(window_data, bias=False)
                                kurt_val = stats.kurtosis(window_data, bias=False)
                        else:
                            skew_val = kurt_val = 0.0
                        p25, p75 = np.percentile(window_data, [25, 75])

                        features.extend([
                            mean_val if np.isfinite(mean_val) else 0.0,
                            std_val if np.isfinite(std_val) else 0.0,
                            skew_val if np.isfinite(skew_val) else 0.0,
                            kurt_val if np.isfinite(kurt_val) else 0.0,
                            p25 if np.isfinite(p25) else 0.0,
                            p75 if np.isfinite(p75) else 0.0
                        ])
                    except Exception:
                        features.extend([0.0] * 6)

        # === 4. CROSS-INDICATOR RELATIONSHIPS =================================
        key_indicators = [
            'RSX_line', 'SchaffTrendCycle_line', 'WaveTrend_line',
            'HurstExponent_line', 'DamianiVolatmeter_line', 'close'
        ]
        available_keys = [k for k in key_indicators if k in indicator_data]
        for i, n1 in enumerate(available_keys):
            for n2 in available_keys[i + 1:]:
                try:
                    v1 = np.asarray(indicator_data[n1][-20:], dtype=np.float64)
                    v2 = np.asarray(indicator_data[n2][-20:], dtype=np.float64)
                    mask = np.isfinite(v1) & np.isfinite(v2)
                    v1, v2 = v1[mask], v2[mask]
                    if len(v1) < 3:
                        features.extend([0.0, 0.0, 0.0])
                        continue
                    corr = np.corrcoef(v1, v2)[0, 1]
                    ratio = (v1[-1] / v2[-1]) if abs(v2[-1]) > 1e-10 else 0.0
                    spread = v1[-1] - v2[-1]
                    for val in [corr, ratio, spread]:
                        features.append(self._safe_float(val))
                except Exception:
                    features.extend([0.0, 0.0, 0.0])

        # === 5. FREQUENCY DOMAIN (CLOSE) ======================================
        if 'close' in indicator_data and len(indicator_data['close']) >= 64:
            try:
                close_data = np.asarray(indicator_data['close'][-64:], dtype=np.float64)
                close_data = close_data[np.isfinite(close_data)]
                if len(close_data) >= 32:
                    fft = np.fft.fft(close_data)
                    power = np.abs(fft[:32])
                    dom_idx = int(np.argmax(power[1:]) + 1)
                    pnorm = power / (np.sum(power) + 1e-10)
                    entropy = -np.sum(pnorm * np.log2(pnorm + 1e-10))
                    features.extend([float(dom_idx), self._safe_float(entropy)])
                else:
                    features.extend([0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0])

        # === 6. HURST REGIME ==================================================
        if 'HurstExponent_line' in indicator_data and len(indicator_data['HurstExponent_line']) > 0:
            hurst = self._safe_float(indicator_data['HurstExponent_line'][-1])
            features.extend([
                hurst,
                1.0 if hurst > 0.55 else 0.0,
                1.0 if hurst < 0.45 else 0.0,
                abs(hurst - 0.5)
            ])

        # === 7. VOLATILITY REGIME =============================================
        if 'StandarizedATR_line' in indicator_data and len(indicator_data['StandarizedATR_line']) >= 20:
            satr = np.asarray(indicator_data['StandarizedATR_line'][-20:], dtype=np.float64)
            satr = satr[np.isfinite(satr)]
            if len(satr) > 0:
                curr, mean, std = satr[-1], np.mean(satr), np.std(satr)
                features.extend([
                    self._safe_float(curr),
                    self._safe_float(curr / (mean + 1e-10)),
                    self._safe_float(std)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

        # === 8. SQUEEZE/EXPANSION =============================================
        if 'SqueezeVolatility_line' in indicator_data and len(indicator_data['SqueezeVolatility_line']) >= 10:
            sqz = np.asarray(indicator_data['SqueezeVolatility_line'][-10:], dtype=np.float64)
            sqz = sqz[np.isfinite(sqz)]
            if len(sqz) > 0:
                count = 0
                for val in reversed(sqz):
                    if abs(val) < 0.5:
                        count += 1
                    else:
                        break
                features.extend([float(count), 1.0 if count >= 3 else 0.0])
            else:
                features.extend([0.0, 0.0])

        # === Final safety =====================================================
        arr = np.asarray(features, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    # --------------------------------------------------------------------------
    def fit_scaler(self, feature_matrix: np.ndarray):
        """Fit the scaler on training data."""
        self.scaler.fit(feature_matrix)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.scaler.transform(features)[0]

