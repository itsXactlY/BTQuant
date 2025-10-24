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
    ✅ FIXED: Extracts deep features from indicator internals.
    Fully robust to NaN/Inf, precision loss, or type mismatches.
    All features are clamped to reasonable ranges to prevent overflow.
    """

    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50]):
        self.lookback_windows = lookback_windows
        self.scaler = RobustScaler()
        self.feature_names = []
        self.fitted = False

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------
    def _safe_float(self, x, clamp_range=(-1e6, 1e6)):
        """
        ✅ FIXED: Convert any value safely to float with clamping.
        Prevents extreme values that cause overflow.
        """
        try:
            f = float(x)
            if not np.isfinite(f):
                return 0.0
            # Clamp to prevent extreme values
            return np.clip(f, clamp_range[0], clamp_range[1])
        except Exception:
            return 0.0

    def _safe_array(self, arr, max_val=1e6):
        """
        ✅ FIXED: Safely convert to array and clamp extreme values.
        """
        try:
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=-max_val)
            arr = np.clip(arr, -max_val, max_val)
            return arr
        except Exception:
            return np.array([0.0])

    # --------------------------------------------------------------------------
    # Core feature extraction
    # --------------------------------------------------------------------------
    def extract_all_features(self, indicator_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        ✅ FIXED: Extracts numerical feature vector from indicator internals.
        All features are clamped to prevent extreme values.

        Args:
            indicator_data: Dict[str, np.ndarray]
        Returns:
            np.ndarray: 1D feature vector (float32), all values in reasonable range
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
                        window_data = self._safe_array(values[-window:])
                        window_data = window_data[np.isfinite(window_data)]
                        if len(window_data) < 3:
                            features.extend([0.0] * 6)
                            continue

                        mean_val = np.mean(window_data)
                        std_val = np.std(window_data)

                        # ✅ FIXED: Safe skew/kurt calculation
                        if std_val > 1e-10 and len(window_data) >= 3:
                            with np.errstate(all='ignore'):
                                try:
                                    skew_val = stats.skew(window_data, bias=False)
                                    kurt_val = stats.kurtosis(window_data, bias=False)
                                    # Clamp skew/kurtosis to reasonable ranges
                                    skew_val = np.clip(skew_val, -10, 10)
                                    kurt_val = np.clip(kurt_val, -10, 10)
                                except Exception:
                                    skew_val = kurt_val = 0.0
                        else:
                            skew_val = kurt_val = 0.0

                        p25, p75 = np.percentile(window_data, [25, 75])

                        features.extend([
                            self._safe_float(mean_val),
                            self._safe_float(std_val),
                            self._safe_float(skew_val),
                            self._safe_float(kurt_val),
                            self._safe_float(p25),
                            self._safe_float(p75)
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
                    v1 = self._safe_array(indicator_data[n1][-20:])
                    v2 = self._safe_array(indicator_data[n2][-20:])
                    mask = np.isfinite(v1) & np.isfinite(v2)
                    v1, v2 = v1[mask], v2[mask]
                    if len(v1) < 3:
                        features.extend([0.0, 0.0, 0.0])
                        continue

                    # ✅ FIXED: Safe correlation calculation
                    with np.errstate(all='ignore'):
                        corr = np.corrcoef(v1, v2)[0, 1]
                        corr = self._safe_float(corr, clamp_range=(-1, 1))

                    # ✅ FIXED: Safe ratio calculation
                    ratio = (v1[-1] / v2[-1]) if abs(v2[-1]) > 1e-10 else 0.0
                    ratio = np.clip(ratio, -100, 100)  # Prevent extreme ratios

                    spread = v1[-1] - v2[-1]

                    features.extend([
                        self._safe_float(corr),
                        self._safe_float(ratio),
                        self._safe_float(spread)
                    ])
                except Exception:
                    features.extend([0.0, 0.0, 0.0])

        # === 5. FREQUENCY DOMAIN (CLOSE) ======================================
        if 'close' in indicator_data and len(indicator_data['close']) >= 64:
            try:
                close_data = self._safe_array(indicator_data['close'][-64:])
                close_data = close_data[np.isfinite(close_data)]
                if len(close_data) >= 32:
                    # ✅ FIXED: Normalize before FFT to prevent overflow
                    close_norm = (close_data - np.mean(close_data)) / (np.std(close_data) + 1e-10)
                    fft = np.fft.fft(close_norm)
                    power = np.abs(fft[:32])
                    dom_idx = int(np.argmax(power[1:]) + 1)
                    pnorm = power / (np.sum(power) + 1e-10)

                    # ✅ FIXED: Safe entropy calculation
                    with np.errstate(all='ignore'):
                        entropy = -np.sum(pnorm * np.log2(pnorm + 1e-10))
                        entropy = np.clip(entropy, 0, 10)  # Clamp entropy

                    features.extend([
                        float(np.clip(dom_idx, 0, 32)),
                        self._safe_float(entropy)
                    ])
                else:
                    features.extend([0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0])

        # === 6. HURST REGIME ==================================================
        if 'HurstExponent_line' in indicator_data and len(indicator_data['HurstExponent_line']) > 0:
            hurst = self._safe_float(indicator_data['HurstExponent_line'][-1], clamp_range=(0, 1))
            features.extend([
                hurst,
                1.0 if hurst > 0.55 else 0.0,
                1.0 if hurst < 0.45 else 0.0,
                abs(hurst - 0.5)
            ])

        # === 7. VOLATILITY REGIME =============================================
        if 'StandarizedATR_line' in indicator_data and len(indicator_data['StandarizedATR_line']) >= 20:
            satr = self._safe_array(indicator_data['StandarizedATR_line'][-20:])
            satr = satr[np.isfinite(satr)]
            if len(satr) > 0:
                curr = satr[-1]
                mean = np.mean(satr)
                std = np.std(satr)

                # ✅ FIXED: Safe ratio calculation
                ratio = curr / (mean + 1e-10)
                ratio = np.clip(ratio, 0, 100)  # Prevent extreme volatility ratios

                features.extend([
                    self._safe_float(curr),
                    self._safe_float(ratio),
                    self._safe_float(std)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

        # === 8. SQUEEZE/EXPANSION =============================================
        if 'SqueezeVolatility_line' in indicator_data and len(indicator_data['SqueezeVolatility_line']) >= 10:
            sqz = self._safe_array(indicator_data['SqueezeVolatility_line'][-10:])
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

        # === FINAL SAFETY =====================================================
        # ✅ CRITICAL: Convert to array and apply final safety checks
        arr = np.asarray(features, dtype=np.float32)

        # Replace any remaining NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)

        # ✅ CRITICAL: Final clamp to prevent extreme values
        arr = np.clip(arr, -100, 100)

        return arr

    # --------------------------------------------------------------------------
    def fit_scaler(self, feature_matrix: np.ndarray):
        """
        ✅ FIXED: Fit the scaler on training data with safety checks.
        """
        if feature_matrix.size == 0:
            console.print("[yellow]⚠️ Empty feature matrix, cannot fit scaler[/yellow]")
            return

        # Safety: Remove any NaN/Inf rows
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=100.0, neginf=-100.0)
        feature_matrix = np.clip(feature_matrix, -100, 100)

        try:
            self.scaler.fit(feature_matrix)
            self.fitted = True
            console.print("[green]✅ Scaler fitted successfully[/green]")
        except Exception as e:
            console.print(f"[red]⚠️ Scaler fit failed: {e}[/red]")
            self.fitted = False

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        ✅ FIXED: Normalize features with safety checks.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Safety checks
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        features = np.clip(features, -100, 100)

        if not self.fitted:
            console.print("[yellow]⚠️ Scaler not fitted, returning unscaled features[/yellow]")
            return features[0] if features.shape[0] == 1 else features

        try:
            transformed = self.scaler.transform(features)
            # ✅ CRITICAL: Clip transformed features to prevent extreme values
            transformed = np.clip(transformed, -10, 10)
            return transformed[0] if transformed.shape[0] == 1 else transformed
        except Exception as e:
            console.print(f"[red]⚠️ Transform failed: {e}, returning unscaled[/red]")
            return features[0] if features.shape[0] == 1 else features