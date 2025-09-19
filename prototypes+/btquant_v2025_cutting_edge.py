# Enhanced Multi-Modal Transformer-GNN Trading Strategy with Deep Reinforcement Learning
# BTQuant v2025 - COMPLETE SYSTEM: Auto-Optimization + Full ML Pipeline + Multithreading

'''

Feedback from cryptoquant ::

Your current environment only processes single-timeframe data, so the RL agent can't learn the multi-timeframe patterns you want it to discover.
The bigger issue remains training efficiency. Even with proper multi-timeframe data, you're still asking the RL agent to:

Learn timeframe relationships
Learn market regime detection
Learn optimal position sizing
Learn entry/exit timing
Learn risk management

All simultaneously from sparse trading rewards. This is why your policy gradients are near zero - the learning signal is too weak and diffuse.
A more practical approach might be:

Use your technical analysis system to identify the multi-timeframe patterns
Train RL only on position sizing given those pre-identified patterns
This gives the RL agent a much clearer, focused learning objective

The multi-timeframe concept is sound, but implementing it effectively in RL requires careful environment design that matches your intended strategy architecture.
'''

import datetime
import hashlib
import json
import math
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import multiprocessing as mp
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Core scientific computing and data handling
import numpy as np
import polars as pl
from numba import jit

# Deep learning and neural networks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# TensorFlow integration
try:
    import tensorflow as tf
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except:
    tf = None

# Reinforcement learning
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Traditional ML and optimization
import optuna
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Backtesting and trading
import backtrader as bt

# Technical analysis and financial metrics
import talib # For Multithreaded vectorized booster

from rich.console import Console
from rich.table import Table
import wandb  # Weights & Biases for experiment tracking

# Custom imports for data loading
from backtrader.feeds.mssql_crypto import get_database_data # , MSSQLData
from backtrader.dontcommit import optuna_connection_string as CONNECTION

warnings.filterwarnings('ignore')
console = Console()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== DATA CONFIGURATION ====================

DEFAULT_COLLATERAL = "USDT"

@dataclass(frozen=True)
class DataSpec:
    """Configuration for data loading"""
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

@dataclass
class TradingConfig:
    """COMPLETE AUTO-OPTIMIZABLE Trading Configuration"""
    init_cash: float = 100_000.0  # USDT capital
    commission: float = 0.00075
    slippage_bps: float = 5.0
    cache_dir: Path = field(default_factory=lambda: Path(".btq_cache_v2025"))
    use_gpu: bool = True
    batch_size: int = 32
    transformer_dim: int = 128
    gnn_hidden_dim: int = 64
    num_attention_heads: int = 4
    num_transformer_layers: int = 3
    rl_algorithm: str = "PPO"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_position_size: float = 0.15
    max_drawdown_threshold: float = 0.10
    var_confidence: float = 0.05
    dropout: float = 0.1
    
    # CRYPTO TRADING SPECIFIC CONFIGS - AUTO-OPTIMIZABLE
    min_trade_amount: float = 50.0
    max_trade_amount: float = 3000.0  # AUTO-OPTIMIZABLE
    risk_per_trade: float = 0.015     # AUTO-OPTIMIZABLE
    reward_risk_ratio: float = 2.0    # AUTO-OPTIMIZABLE
    stop_loss_pct: float = 0.015      # AUTO-OPTIMIZABLE
    take_profit_pct: float = 0.03     # AUTO-OPTIMIZABLE
    
    # SIGNAL GENERATION PARAMETERS - AUTO-OPTIMIZABLE
    short_ma_period: int = 5          # AUTO-OPTIMIZABLE
    long_ma_period: int = 15          # AUTO-OPTIMIZABLE
    rsi_period: int = 14              # AUTO-OPTIMIZABLE
    rsi_oversold: float = 40.0        # AUTO-OPTIMIZABLE
    rsi_overbought: float = 60.0      # AUTO-OPTIMIZABLE
    volume_threshold: float = 1.2     # AUTO-OPTIMIZABLE
    
    # CONFIDENCE THRESHOLDS - AUTO-OPTIMIZABLE
    conf_bull: float = 0.2            # Was 0.4
    conf_bear: float = 0.3            # Was 0.5  
    conf_sideways: float = 0.25       # Was 0.45
    conf_volatile: float = 0.4        # Was 0.6
    # conf_bull: float = 0.4            # AUTO-OPTIMIZABLE
    # conf_bear: float = 0.5            # AUTO-OPTIMIZABLE
    # conf_sideways: float = 0.45       # AUTO-OPTIMIZABLE
    # conf_volatile: float = 0.6        # AUTO-OPTIMIZABLE
    
    # SIGNAL WEIGHTS - AUTO-OPTIMIZABLE
    trend_weight: float = 0.4         # AUTO-OPTIMIZABLE
    rsi_weight: float = 0.3           # AUTO-OPTIMIZABLE
    macd_weight: float = 0.25         # AUTO-OPTIMIZABLE
    volume_weight: float = 0.2        # AUTO-OPTIMIZABLE
    momentum_weight: float = 0.15     # AUTO-OPTIMIZABLE
    
    # MULTITHREADING CONFIGURATION
    max_workers: int = mp.cpu_count()
    use_multiprocessing: bool = True
    
    use_sentiment: bool = True
    use_macro_data: bool = True
    use_options_flow: bool = True
    use_regime_detection: bool = True
    regime_lookback: int = 252
    
    # Data configuration
    symbols: List[str] = field(default_factory=lambda: "BTC")
    timeframe: str = "1m"
    test1: str = "2020-09-28"
    test2: str = "2020-10-10"
    bull_start: str = "2020-09-28"
    bull_end: str = "2021-05-31"
    bear_start: str = "2022-05-28"
    bear_end: str = "2023-06-23"
    test_start: str = "2023-06-12"
    test_end: str = "2025-05-31"

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

# ==================== CUDA COMPATIBILITY CHECKING ====================

def check_cuda_compatibility():
    """Check CUDA compatibility and return best available device"""
    try:
        if not torch.cuda.is_available():
            console.print("[yellow]CUDA not available. Using CPU. :([/yellow]")
            return torch.device('cpu'), False
            
        # Test CUDA functionality
        device = torch.device('cuda')
        test_tensor = torch.randn(10, 10, device=device)
        test_result = test_tensor @ test_tensor.T
        test_result # TODO :: attach an debugger an look at me
        
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]CUDA compatible! Using GPU: {gpu_name}[/green]")
        return device, True
        
    except Exception as e:
        console.print(f"[red]CUDA compatibility issue: {e}[/red]")
        console.print("[yellow]Falling back to CPU for stability.[/yellow]")
        return torch.device('cpu'), False

# Global configuration
CONFIG = TradingConfig()
DEVICE, CUDA_AVAILABLE = check_cuda_compatibility()

# ==================== MULTITHREADED DATA LOADING WITH CACHING ====================

class DataCache:
    """Thread-safe caching of loaded Polars DataFrames with multithreading support"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, pl.DataFrame] = {}
        self._cache_lock = threading.Lock()

    def _get_cache_key(self, spec: DataSpec) -> str:
        """Generate unique cache key for data specification"""
        ranges_str = ""
        if spec.ranges:
            ranges_str = "_".join(f"{s}_{e}" for s, e in spec.ranges)
        else:
            ranges_str = f"{spec.start_date}_{spec.end_date}"
        return f"{spec.symbol}_{spec.interval}_{ranges_str}_{spec.collateral}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached data"""
        return self.cache_dir / f"{cache_key}.parquet"

    def load_data(self, spec: DataSpec, force_reload: bool = False) -> pl.DataFrame:
        """Thread-safe data loading with caching support"""
        cache_key = self._get_cache_key(spec)
        
        # Thread-safe memory cache check
        with self._cache_lock:
            if not force_reload and cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if not force_reload and cache_path.exists():
            try:
                df = pl.read_parquet(cache_path)
                with self._cache_lock:
                    self._memory_cache[cache_key] = df
                return df
            except Exception as e:
                console.print(f"[yellow]Cache read failed: {e}, reloading from database[/yellow]")

        # Load from database
        if spec.ranges:
            dfs = []
            for start_date, end_date in spec.ranges:
                part = get_database_data(
                    ticker=spec.symbol,
                    start_date=start_date,
                    end_date=end_date,
                    time_resolution=spec.interval,
                    pair=spec.collateral,
                )
                
                if part is None or part.is_empty():
                    continue
                dfs.append(part)
            
            if not dfs:
                raise ValueError(f"No data found for {spec.symbol} in any of the specified ranges")
            df = pl.concat(dfs).sort("TimestampStart")
        else:
            df = get_database_data(
                ticker=spec.symbol,
                start_date=spec.start_date,
                end_date=spec.end_date,
                time_resolution=spec.interval,
                pair=spec.collateral,
            )
            
        if df is None or df.is_empty():
            raise ValueError(f"No data for {spec.symbol} {spec.interval}")

        # Validate required columns
        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")

        # Sort by time and ensure clean data
        df = df.sort("TimestampStart")
        df = self._clean_dataframe(df)
        
        # Thread-safe caching
        try:
            df.write_parquet(cache_path)
        except Exception as e:
            console.print(f"[yellow]Failed to cache data: {e}[/yellow]")
        
        with self._cache_lock:
            self._memory_cache[cache_key] = df
        
        return df

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and validate DataFrame"""
        # Remove rows with null values in critical columns
        df = df.drop_nulls(subset=["Open", "High", "Low", "Close", "Volume"])
        
        # Ensure positive prices and volumes
        df = df.with_columns([
            pl.when(pl.col("Open") <= 0).then(pl.col("Close")).otherwise(pl.col("Open")).alias("Open"),
            pl.when(pl.col("High") <= 0).then(pl.col("Close")).otherwise(pl.col("High")).alias("High"),
            pl.when(pl.col("Low") <= 0).then(pl.col("Close")).otherwise(pl.col("Low")).alias("Low"),
            pl.when(pl.col("Close") <= 0).then(None).otherwise(pl.col("Close")).alias("Close"),
            pl.when(pl.col("Volume") <= 0).then(1000.0).otherwise(pl.col("Volume")).alias("Volume"),
        ])
        
        # Remove rows where Close is still null after cleaning
        df = df.drop_nulls(subset=["Close"])
        
        # Ensure OHLC relationships are correct
        df = df.with_columns([
            pl.max_horizontal([pl.col("High"), pl.col("Open"), pl.col("Close")]).alias("High"),
            pl.min_horizontal([pl.col("Low"), pl.col("Open"), pl.col("Close")]).alias("Low"),
        ])
        
        return df

def preload_polars_parallel(specs: List[DataSpec], cache: DataCache) -> Dict[str, pl.DataFrame]:
    """Parallel data loading using multithreading"""
    df_map: Dict[str, pl.DataFrame] = {}
    
    def load_spec(spec):
        try:
            df = cache.load_data(spec)
            return spec.symbol, df
        except Exception as e:
            console.print(f"[red]Failed to load {spec.symbol}: {e}[/red]")
            return spec.symbol, None
    
    # Use ThreadPoolExecutor for I/O bound database operations
    with ThreadPoolExecutor(max_workers=min(len(specs), CONFIG.max_workers)) as executor:
        futures = [executor.submit(load_spec, spec) for spec in specs]
        
        for future in futures:
            symbol, df = future.result()
            if df is not None:
                df_map[symbol] = df
    
    return df_map

# ==================== HELPER FUNCTIONS ====================

@jit(nopython=True)
def rolling_corr_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling correlation using Numba"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        start_idx = i - window + 1
        x_window = x[start_idx:i + 1]
        y_window = y[start_idx:i + 1]
        
        # Calculate correlation
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)
        
        numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
        x_std = np.sqrt(np.sum((x_window - x_mean) ** 2))
        y_std = np.sqrt(np.sum((y_window - y_mean) ** 2))
        
        if x_std > 0 and y_std > 0:
            result[i] = numerator / (x_std * y_std)
        else:
            result[i] = 0.0
    
    return result

@jit(nopython=True)
def rolling_window_numba(data: np.ndarray, window: int, func_type: str) -> np.ndarray:
    """Fast rolling window operations using Numba"""
    n = len(data)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        
        if func_type == "mean":
            result[i] = np.mean(window_data)
        elif func_type == "std":
            result[i] = np.std(window_data)
        elif func_type == "min":
            result[i] = np.min(window_data)
        elif func_type == "max":
            result[i] = np.max(window_data)
        elif func_type == "median":
            result[i] = np.median(window_data)
    
    return result

def to_talib_array(arr: np.ndarray) -> np.ndarray:
    """Convert array to float64 for TA-Lib compatibility"""
    return np.asarray(arr, dtype=np.float64)

# ==================== ADVANCED FEATURE ENGINEERING ====================

class QuantumFeatureEngine:
    """Advanced feature engineering with multithreading support"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create comprehensive feature set using parallel processing"""
        
        # Use Polars lazy evaluation for basic features
        lazy = df.lazy()

        # 1) Basic momentum features
        lazy = lazy.with_columns([
            (pl.col("Close").diff()).alias("price_change"),
            (pl.col("Close").diff().diff()).alias("momentum_2nd_derivative"),
            (pl.col("Close").pct_change()).alias("returns"),
        ])

        # 2) Convert to numpy for complex calculations
        df_collected = lazy.collect()
        
        # Extract arrays and ensure float64 for TA-Lib compatibility
        close_arr = to_talib_array(df_collected["Close"].to_numpy())
        high_arr = to_talib_array(df_collected["High"].to_numpy())
        low_arr = to_talib_array(df_collected["Low"].to_numpy())
        volume_arr = to_talib_array(df_collected["Volume"].to_numpy())
        
        # Ensure no zero or negative prices
        close_arr = np.where(close_arr <= 0, np.median(close_arr[close_arr > 0]), close_arr)
        high_arr = np.where(high_arr <= 0, close_arr, high_arr)
        low_arr = np.where(low_arr <= 0, close_arr, low_arr)
        volume_arr = np.where(volume_arr <= 0, 1000, volume_arr)

        # 3) Parallel technical indicator calculation
        features_dict = self._calculate_indicators_parallel(close_arr, high_arr, low_arr, volume_arr)

        # 4) Create new dataframe with all features
        feature_data = {}
        n_samples = len(close_arr)
        
        # Add original columns
        for col in df_collected.columns:
            feature_data[col] = df_collected[col].to_numpy()

        # Add computed features, ensuring all have same length
        for name, values in features_dict.items():
            if len(values) != n_samples:
                if len(values) < n_samples:
                    values = np.concatenate([np.full(n_samples - len(values), 0.0), values])
                else:
                    values = values[:n_samples]
            # Ensure no NaN or inf values
            values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
            feature_data[name] = values

        # Convert to Polars DataFrame
        try:
            df_feat = pl.DataFrame(feature_data)
        except Exception as e:
            console.print(f"[red]Error creating DataFrame: {e}[/red]")
            # Fallback: create with basic features only
            df_feat = pl.DataFrame({
                col: df_collected[col].to_numpy() for col in df_collected.columns
            })

        # 5) Fill nulls
        df_feat = df_feat.fill_null(strategy="forward").fill_null(0)

        return df_feat

    def _calculate_indicators_parallel(self, close_arr, high_arr, low_arr, volume_arr):
        """Calculate technical indicators using parallel processing"""
        features_dict = {}
        
        # Define indicator calculation functions
        def calc_moving_averages():
            mas = {}
            for period in [10, 20, 50]:
                try:
                    sma = talib.SMA(close_arr, timeperiod=period)
                    ema = talib.EMA(close_arr, timeperiod=period)
                    mas[f"sma_{period}"] = np.nan_to_num(sma, nan=close_arr)
                    mas[f"ema_{period}"] = np.nan_to_num(ema, nan=close_arr)
                except:
                    sma = rolling_window_numba(close_arr, period, "mean")
                    mas[f"sma_{period}"] = np.nan_to_num(sma, nan=close_arr)
                    mas[f"ema_{period}"] = np.nan_to_num(sma, nan=close_arr)
            return mas
        
        def calc_momentum_indicators():
            momentum = {}
            try:
                rsi = talib.RSI(close_arr, timeperiod=14)
                momentum["rsi"] = np.nan_to_num(rsi, nan=50.0)
            except:
                momentum["rsi"] = np.full_like(close_arr, 50.0)
            
            try:
                macd, macd_signal, macd_hist = talib.MACD(close_arr)
                momentum["macd"] = np.nan_to_num(macd, nan=0.0)
                momentum["macd_signal"] = np.nan_to_num(macd_signal, nan=0.0)
                momentum["macd_histogram"] = np.nan_to_num(macd_hist, nan=0.0)
            except:
                momentum["macd"] = np.zeros_like(close_arr)
                momentum["macd_signal"] = np.zeros_like(close_arr)
                momentum["macd_histogram"] = np.zeros_like(close_arr)
            return momentum
        
        def calc_volatility_indicators():
            volatility = {}
            try:
                bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(close_arr)
                volatility["bb_upper"] = np.nan_to_num(bbands_upper, nan=close_arr)
                volatility["bb_lower"] = np.nan_to_num(bbands_lower, nan=close_arr)
                volatility["bb_width"] = np.nan_to_num(bbands_upper - bbands_lower, nan=0.0)
            except:
                volatility["bb_upper"] = close_arr
                volatility["bb_lower"] = close_arr
                volatility["bb_width"] = np.zeros_like(close_arr)
            
            try:
                atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
                volatility["atr"] = np.nan_to_num(atr, nan=0.01)
            except:
                volatility["atr"] = np.full_like(close_arr, 0.01)
            return volatility
        
        def calc_volume_indicators():
            volume = {}
            vol_sma = rolling_window_numba(volume_arr, 20, "mean")
            volume["volume_sma"] = np.nan_to_num(vol_sma, nan=volume_arr)
            volume["volume_ratio"] = np.nan_to_num(volume_arr / vol_sma, nan=1.0)
            return volume
        
        # Execute calculations in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(calc_moving_averages),
                executor.submit(calc_momentum_indicators),
                executor.submit(calc_volatility_indicators),
                executor.submit(calc_volume_indicators)
            ]
            
            for future in futures:
                result = future.result()
                features_dict.update(result)
        
        # Price-based features
        price_change = np.concatenate([[0], np.diff(close_arr)]) / close_arr
        features_dict["price_change_pct"] = np.nan_to_num(price_change, nan=0.0)
        
        volatility = rolling_window_numba(price_change, 20, "std")
        features_dict["volatility_20"] = np.nan_to_num(volatility, nan=0.01)
        
        return features_dict

# ==================== TRANSFORMER-GNN ARCHITECTURE WITH FIXES ====================

class MultiModalTransformerGNN(nn.Module):
    """FIXED Multi-Modal Transformer-GNN hybrid for financial prediction"""

    def __init__(self,
                 input_dim: int,
                 transformer_dim: int = 128,
                 gnn_hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.transformer_dim = transformer_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, transformer_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(transformer_dim, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # FIXED: Replace complex GNN with simple MLP layers
        self.gnn_mlp = nn.Sequential(
            nn.Linear(transformer_dim, gnn_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-attention - now with matching dimensions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # GNN projection to match transformer dimensions
        self.gnn_projection = nn.Linear(gnn_hidden_dim, transformer_dim)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, output_dim)
        )

        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(transformer_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4)  # bull, bear, sideways, volatile
        )

    def forward(self, x, edge_index, batch_idx=None):
        try:
            batch_size, seq_len, _ = x.shape

            # Transformer pathway
            x_proj = self.input_projection(x)
            x_pos = self.positional_encoding(x_proj)
            transformer_output = self.transformer_encoder(x_pos)

            # Take the last time step for processing
            last_step = transformer_output[:, -1, :]

            # FIXED: Simplified GNN pathway using MLP
            gnn_output = self.gnn_mlp(last_step)

            # Cross-attention with proper tensor shapes
            transformer_query = last_step.unsqueeze(1)
            gnn_projected = self.gnn_projection(gnn_output).unsqueeze(1)
            
            attended_output, _ = self.cross_attention(
                transformer_query, gnn_projected, gnn_projected
            )
            attended_output = attended_output.squeeze(1)

            # Combine features
            combined_features = torch.cat([last_step, attended_output], dim=1)

            # Predictions
            action_logits = self.output_layers(combined_features)
            regime_logits = self.regime_classifier(last_step)

            return action_logits, regime_logits

        except RuntimeError as e:
            if "CUDA" in str(e):
                console.print(f"[red]CUDA error in model forward pass: {e}[/red]")
                # Move to CPU and retry
                x_cpu = x.cpu()
                edge_index_cpu = edge_index.cpu()
                self.cpu()
                return self.forward(x_cpu, edge_index_cpu, batch_idx)
            else:
                raise e

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)
            return self.dropout(x)
        except RuntimeError as e:
            if "CUDA" in str(e):
                x_cpu = x.cpu()
                self.cpu()
                seq_len = x_cpu.size(1)
                x_cpu = x_cpu + self.pe[:seq_len].transpose(0, 1).cpu()
                return self.dropout(x_cpu)
            else:
                raise e

# ==================== TENSORFLOW INTEGRATION ====================

class TensorFlowPredictor:
    """TensorFlow-based ensemble model - FIXED predictions"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        
        if tf is not None:
            self._build_model()
    
    def _build_model(self):
        """Build TensorFlow model"""
        try:
            # Simple LSTM model for time series
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 10)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # buy, hold, sell
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            console.print("[green]TensorFlow model built successfully[/green]")
            
        except Exception as e:
            console.print(f"[yellow]TensorFlow model build failed: {e}[/yellow]")
            self.model = None
    
    def train(self, X_train, y_train, epochs=50):
        """Train TensorFlow model"""
        if self.model is None:
            return False
            
        try:
            # Convert labels to categorical
            y_categorical = tf.keras.utils.to_categorical(y_train, num_classes=3)
            
            history = self.model.fit(
                X_train, y_categorical,
                epochs=epochs,
                validation_split=0.2,
                batch_size=32,
                verbose=0
            )
            
            self.is_trained = True
            console.print("[green]TensorFlow model trained successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]TensorFlow training failed: {e}[/red]")
            return False
    
    def predict(self, X):
        """Make predictions with FIXED fallback logic"""
        if self.model is not None and self.is_trained:
            try:
                # Use trained TensorFlow model
                predictions = self.model.predict(X, verbose=0)
                return predictions
            except Exception as e:
                console.print(f"[yellow]TF prediction failed: {e}, using fallback[/yellow]")
        
        # FIXED FALLBACK: Return varied predictions based on input features
        predictions = []
        for sample in X:
            if len(sample) >= 4:
                # Use first 4 features for simple prediction logic
                try:
                    ret_1, ret_5, cv, ma_ratio = sample[:4]
                    
                    # Create intelligent predictions based on features
                    if ret_5 > 0.015 and ret_1 > 0.005:  # Strong upward momentum
                        pred = [0.15, 0.25, 0.6]  # Favor buy
                    elif ret_5 < -0.015 and ret_1 < -0.005:  # Strong downward momentum  
                        pred = [0.6, 0.25, 0.15]  # Favor sell
                    elif abs(ret_1) < 0.003 and cv < 0.02:  # Low volatility, sideways
                        pred = [0.3, 0.4, 0.3]  # Neutral with hold bias
                    elif cv > 0.05:  # High volatility
                        pred = [0.35, 0.3, 0.35]  # Uncertain, slight avoid
                    else:
                        # Medium momentum
                        if ret_1 > 0:
                            pred = [0.25, 0.35, 0.4]  # Slight buy bias
                        else:
                            pred = [0.4, 0.35, 0.25]  # Slight sell bias
                except:
                    pred = [0.3, 0.4, 0.3]  # Default varied
            else:
                pred = [0.3, 0.4, 0.3]  # Default varied
                
            predictions.append(pred)
            
        return np.array(predictions)

# ==================== DEEP REINFORCEMENT LEARNING ENVIRONMENT ====================

# class AdvancedTradingEnv(gym.Env):
#     """FIXED Advanced trading environment with multi-modal inputs"""

#     def __init__(self,
#                  df: Union[pd.DataFrame, np.ndarray],
#                  feature_columns: List[str],
#                  initial_balance: float = 10000,
#                  transaction_cost: float = 0.0001,
#                  max_position: float = 1.0):
#         super().__init__()

#         # Handle different input types and clean data
#         if isinstance(df, pd.DataFrame):
#             df_clean = df.copy()
            
#             # Remove datetime columns that cause issues
#             datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns
#             if len(datetime_cols) > 0:
#                 df_clean = df_clean.drop(columns=datetime_cols)
            
#             # Convert any remaining object/string columns to numeric
#             for col in df_clean.columns:
#                 if df_clean[col].dtype == 'object':
#                     try:
#                         df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
#                     except:
#                         df_clean = df_clean.drop(columns=[col])
            
#             # Convert to numpy and clean NaNs
#             self.data = df_clean.fillna(0).to_numpy()
            
#         elif isinstance(df, np.ndarray):
#             self.data = np.nan_to_num(df, nan=0.0)
#         else:
#             raise ValueError("df must be pandas DataFrame or numpy array")

#         self.feature_columns = feature_columns
#         self.initial_balance = initial_balance
#         self.transaction_cost = transaction_cost
#         self.max_position = max_position

#         # Ensure data is clean
#         self.data = np.nan_to_num(self.data, nan=0.0, posinf=1e6, neginf=-1e6)

#         # Environment state
#         self.current_step = 0
#         self.balance = initial_balance
#         self.position = 0.0
#         self.portfolio_value = initial_balance
#         self.trade_history = []

#         # Action and observation spaces
#         self.action_space = gym.spaces.Box(
#             low=-1.0, high=1.0, shape=(1,), dtype=np.float32
#         )

#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf,
#             shape=(min(len(feature_columns), self.data.shape[1]) + 3,), dtype=np.float32
#         )

#         # Performance tracking
#         self.max_portfolio_value = initial_balance
#         self.drawdown = 0.0
#         self.returns = []

#     def reset(self, seed=None):
#         super().reset(seed=seed)
#         self.current_step = 50
#         self.balance = self.initial_balance
#         self.position = 0.0
#         self.portfolio_value = self.initial_balance
#         self.trade_history = []
#         self.max_portfolio_value = self.initial_balance
#         self.drawdown = 0.0
#         self.returns = []
#         return self._get_observation(), {}

#     def step(self, action):
#         # Use a price from the available data
#         if self.data.shape[1] > 4:
#             current_price = float(self.data[self.current_step, 4])  # Assume close price
#         else:
#             current_price = float(self.data[self.current_step, -1])

#         # Ensure price is valid
#         current_price = max(current_price, 1.0)

#         # Execute action (position change)
#         target_position = np.clip(action[0], -self.max_position, self.max_position)
#         position_change = target_position - self.position

#         # Calculate transaction cost
#         cost = abs(position_change) * current_price * self.transaction_cost

#         # Update position and balance
#         if abs(position_change) > 1e-6:
#             self.balance -= cost
#             self.position = target_position
#             self.trade_history.append({
#                 'step': self.current_step,
#                 'price': current_price,
#                 'position_change': position_change,
#                 'cost': cost
#             })

#         # Calculate portfolio value
#         position_value = self.position * current_price
#         self.portfolio_value = self.balance + position_value

#         # Calculate return
#         if self.current_step > 50:
#             prev_portfolio = self.returns[-1] if self.returns else self.initial_balance
#             ret = (self.portfolio_value - prev_portfolio) / prev_portfolio
#             self.returns.append(self.portfolio_value)
#         else:
#             ret = 0.0
#             self.returns.append(self.portfolio_value)

#         # Update drawdown
#         if self.portfolio_value > self.max_portfolio_value:
#             self.max_portfolio_value = self.portfolio_value
#             self.drawdown = 0.0
#         else:
#             self.drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

#         # Calculate reward
#         reward = self._calculate_reward(ret, cost)

#         # Check if episode is done
#         self.current_step += 1
#         done = (self.current_step >= len(self.data) - 1 or
#                 self.portfolio_value <= self.initial_balance * 0.5)

#         if done:
#             reward += self._terminal_reward()

#         return self._get_observation(), reward, done, False, {}

#     def _get_observation(self):
#         # Get current features (limit to available data)
#         max_features = min(len(self.feature_columns), self.data.shape[1])
#         features = self.data[self.current_step, :max_features]

#         # Add portfolio state
#         portfolio_state = np.array([
#             self.balance / self.initial_balance,
#             self.position,
#             self.portfolio_value / self.initial_balance
#         ])

#         # Ensure all values are float32 and clean
#         obs = np.concatenate([features, portfolio_state]).astype(np.float32)
#         obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
#         return obs

#     def _calculate_reward(self, return_rate, transaction_cost):
#         # Multi-objective reward function
#         return_reward = return_rate * 100

#         # Risk-adjusted reward
#         if len(self.returns) > 20:
#             recent_returns = np.array(self.returns[-20:])
#             volatility = np.std(np.diff(recent_returns) / recent_returns[:-1])
#             if volatility > 0:
#                 risk_adjusted_reward = return_rate / volatility
#             else:
#                 risk_adjusted_reward = return_rate
#         else:
#             risk_adjusted_reward = 0

#         # Drawdown penalty
#         drawdown_penalty = -self.drawdown * 10

#         # Transaction cost penalty
#         cost_penalty = -transaction_cost / self.initial_balance * 1000

#         # Position holding reward
#         if len(self.trade_history) > 1:
#             time_since_last_trade = self.current_step - self.trade_history[-2]['step']
#             holding_reward = min(time_since_last_trade / 100, 0.1)
#         else:
#             holding_reward = 0

#         total_reward = (return_reward +
#                        risk_adjusted_reward * 0.5 +
#                        drawdown_penalty +
#                        cost_penalty +
#                        holding_reward)

#         return float(np.nan_to_num(total_reward, nan=0.0))

#     def _terminal_reward(self):
#         # Final reward based on overall performance
#         total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

#         # Sharpe ratio calculation
#         if len(self.returns) > 1:
#             returns_array = np.array(self.returns)
#             pct_returns = np.diff(returns_array) / returns_array[:-1]
#             if np.std(pct_returns) > 0:
#                 sharpe = np.mean(pct_returns) / np.std(pct_returns) * np.sqrt(252)
#             else:
#                 sharpe = 0
#         else:
#             sharpe = 0

#         # Number of trades (encourage efficiency)
#         num_trades = len(self.trade_history)
#         trade_efficiency = max(0, 1 - num_trades / 1000)

#         terminal_reward = (total_return * 100 +
#                           sharpe * 10 +
#                           trade_efficiency * 5)

#         return float(np.nan_to_num(terminal_reward, nan=0.0))

# ==================== ENSEMBLE ML MODELS ====================

class MLEnsemble:
    """Ensemble of multiple ML models including sklearn classifiers"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.models = {}
        self.is_trained = False
        
        # Initialize sklearn models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        self.models['svm'] = SVC(
            probability=True, 
            random_state=42
        )
        
        # Scaler for preprocessing
        self.scaler = RobustScaler()
        
        console.print(f"[green]Initialized {len(self.models)} ML models[/green]")
    
    def train(self, X_train, y_train):
        """Train all ensemble models"""
        try:
            # Preprocess data
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train models in parallel
            def train_model(name_model_tuple):
                name, model = name_model_tuple
                try:
                    model.fit(X_scaled, y_train)
                    return name, model, True
                except Exception as e:
                    console.print(f"[red]Failed to train {name}: {e}[/red]")
                    return name, model, False
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(train_model, item) for item in self.models.items()]
                
                trained_models = {}
                for future in futures:
                    name, model, success = future.result()
                    if success:
                        trained_models[name] = model
                
                self.models = trained_models
                self.is_trained = True
                
                console.print(f"[green]Successfully trained {len(self.models)} ML models[/green]")
                return True
                
        except Exception as e:
            console.print(f"[red]ML ensemble training failed: {e}[/red]")
            return False
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained or not self.models:
            # Return neutral predictions
            return np.array([[0.1, 0.8, 0.1]] * len(X))
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = []
            
            # Get predictions from all models
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_scaled)
                    else:
                        pred = model.predict(X_scaled)
                        # Convert to probabilities
                        pred_proba = np.zeros((len(pred), 3))
                        for i, p in enumerate(pred):
                            pred_proba[i, p] = 1.0
                        pred = pred_proba
                    predictions.append(pred)
                except Exception as e:
                    console.print(f"[yellow]Prediction failed for {name}: {e}[/yellow]")
                    continue
            
            if predictions:
                # Average ensemble predictions
                ensemble_pred = np.mean(predictions, axis=0)
                return ensemble_pred
            else:
                return np.array([[0.1, 0.8, 0.1]] * len(X))
                
        except Exception as e:
            console.print(f"[red]ML ensemble prediction failed: {e}[/red]")
            return np.array([[0.1, 0.8, 0.1]] * len(X))

# ==================== RISK MANAGEMENT SYSTEM ====================

class CryptoRiskManager:
    """Advanced risk management system for crypto trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.active_positions = {}
        self.trade_history = []
        
    def calculate_position_size_usdt(self, 
                                   entry_price: float, 
                                   stop_loss_price: float,
                                   account_balance: float,
                                   confidence: float = 1.0) -> dict:
        """Calculate position size in USDT based on risk management"""
        
        # Base risk amount 
        base_risk_usdt = account_balance * self.config.risk_per_trade
        
        # Adjust risk based on confidence
        confidence_multiplier = np.clip(confidence, 0.5, 1.5)
        risk_usdt = base_risk_usdt * confidence_multiplier
        
        # Calculate the price difference for risk calculation
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return {'usdt_amount': 0, 'quantity': 0, 'risk_amount': 0, 'reward_target': 0}
        
        # Calculate how many units we can buy with our risk amount
        max_quantity = risk_usdt / risk_per_unit
        
        # Total USDT amount needed
        usdt_amount = max_quantity * entry_price
        
        # Apply position limits
        max_position_usdt = account_balance * self.config.max_position_size
        usdt_amount = min(usdt_amount, max_position_usdt)
        
        # Apply trade amount limits
        usdt_amount = np.clip(usdt_amount, self.config.min_trade_amount, self.config.max_trade_amount)
        
        # Recalculate quantity based on final USDT amount
        final_quantity = usdt_amount / entry_price
        final_risk = final_quantity * risk_per_unit
        
        # Calculate expected reward based on R:R ratio
        reward_per_unit = risk_per_unit * self.config.reward_risk_ratio
        expected_reward = final_quantity * reward_per_unit
        
        return {
            'usdt_amount': round(usdt_amount, 2),
            'quantity': round(final_quantity, 6),
            'risk_amount': round(final_risk, 2),
            'reward_target': round(expected_reward, 2),
            'stop_loss_price': stop_loss_price,
            'take_profit_price': entry_price + (reward_per_unit if entry_price < stop_loss_price else -reward_per_unit)
        }
    
    def should_enter_trade(self, signal_strength: float, market_regime: str) -> bool:
        """Determine if we should enter a trade based on conditions"""
        
        # Check if we have too many active positions
        if len(self.active_positions) >= 3:
            return False
        
        # Use optimizable confidence thresholds
        min_confidence = {
            'bull': self.config.conf_bull,
            'bear': self.config.conf_bear,
            'sideways': self.config.conf_sideways,
            'volatile': self.config.conf_volatile
        }.get(market_regime, self.config.conf_sideways)
        
        return signal_strength >= min_confidence
    
    def calculate_stop_loss_take_profit(self, entry_price: float, direction: str) -> tuple:
        """Calculate stop loss and take profit levels"""
        
        if direction.lower() == 'buy':
            # Long position
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:
            # Short position (if supported)
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 - self.config.take_profit_pct)
            
        return round(stop_loss, 2), round(take_profit, 2)

# ==================== ENHANCED BACKTRADER STRATEGY ====================

class CryptoQuantumStrategy(bt.Strategy):
    """COMPLETE AUTO-OPTIMIZED Advanced crypto strategy with ALL ML models"""

    params = (
        ('config', None),   # TradingConfig object
        ('silent', True),   # Reduce output during optimization
        ('models', None),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_volume = self.datas[0].volume

        # Use passed config or fallback
        self.config = self.p.config if self.p.config else CONFIG

        # ✅ Use passed models if available
        self.models = self.p.models if self.p.models else {}

        # Initialize risk manager
        self.risk_manager = CryptoRiskManager(self.config)

        # Strategy state + tracking
        self.current_regime = "normal"
        self.confidence_scores = deque(maxlen=50)
        self.active_positions = {}
        self.portfolio_history = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl_usdt = 0.0
        self.max_portfolio_value = self.broker.getvalue()

        # Counters
        self.bar_count = 0
        self.signal_count = 0

        # ✅ Use loaded models or fallbacks
        if "ensemble" in self.models:
            self.ml_ensemble = self.models["ensemble"]
        else:
            self.ml_ensemble = MLEnsemble(self.config)

        if "tensorflow" in self.models:
            self.tensorflow_model = self.models["tensorflow"]
        else:
            self.tensorflow_model = TensorFlowPredictor(self.config)

        if not self.p.silent:
            console.print(f"[cyan]🚀 COMPLETE Strategy initialized with ${self.broker.getcash():.2f} USDT[/cyan]")
            console.print(f"[cyan]📊 Risk: {self.config.risk_per_trade:.1%} | R:R: {self.config.reward_risk_ratio}:1[/cyan]")
            console.print(f"[cyan]🤖 ML Models: {list(self.models.keys()) if self.models else 'Fallback only'}[/cyan]")

    def next(self):
        """Main strategy logic executed on each bar"""
        
        self.bar_count += 1
        
        # Get current market data  
        current_data = self._prepare_current_data()
        if current_data is None or current_data.height < 50:
            return

        # 1. Check and manage existing positions first
        self._manage_existing_positions()
        
        # 2. Regime Detection
        if self.config.use_regime_detection:
            self.current_regime = self._detect_regime(current_data)

        # 3. Generate trading signals - FIXED: Handle 3 return values
        signals = self._generate_signals(current_data)
        if signals is None:
            return

        self.signal_count += 1

        # FIXED: Unpack all 3 returned values
        if len(signals) == 3:
            action_probs, regime_probs, confidence = signals
        else:
            action_probs, regime_probs = signals[:2]
            confidence = max(action_probs[0])

        # 4. Risk management check
        if not self._risk_check():
            return

        # 5. Execute trades based on signals
        self._execute_crypto_trades(action_probs, regime_probs, confidence)

        # 6. Update portfolio tracking
        self._update_portfolio_tracking()

    def _prepare_current_data(self):
        """Get real OHLCV data from the backtrader feed"""
        try:
            history_length = 100
            current_idx = len(self.data_close)
            
            if current_idx < history_length:
                return None
                
            # Get REAL data from backtrader lines
            data_dict = {
                'Open': [float(self.datas[0].open[i]) for i in range(-history_length, 0)],
                'High': [float(self.datas[0].high[i]) for i in range(-history_length, 0)], 
                'Low': [float(self.datas[0].low[i]) for i in range(-history_length, 0)],
                'Close': [float(self.datas[0].close[i]) for i in range(-history_length, 0)],
                'Volume': [float(self.datas[0].volume[i]) for i in range(-history_length, 0)]
            }
            
            return pl.DataFrame(data_dict)
        except Exception as e:
            if not self.p.silent:
                console.print(f"[red]Error preparing data: {e}[/red]")
            return None

    def _detect_regime(self, data):
        """Improved regime detection with consistent lookbacks"""
        try:
            close_arr = data["Close"].to_numpy()
            if len(close_arr) < 50:  # Need sufficient data
                return "normal"
                
            # Use consistent lookback periods
            short_window = 20
            long_window = 50
            
            # Calculate metrics
            short_returns = np.diff(close_arr[-short_window:]) / close_arr[-short_window:-1]
            long_returns = np.diff(close_arr[-long_window:]) / close_arr[-long_window:-1]
            
            # Trend: compare short vs long MA
            short_ma = np.mean(close_arr[-short_window:])
            long_ma = np.mean(close_arr[-long_window:])
            trend_strength = (short_ma / long_ma - 1)
            
            # Volatility: use rolling window
            volatility = np.std(short_returns) * np.sqrt(1440)  # Annualized for 1min data
            vol_ma = np.std(long_returns) * np.sqrt(1440)
            
            # More nuanced classification
            if volatility > vol_ma * 2.0:
                return "volatile"
            elif trend_strength > 0.02:  # 2% for shorter timeframes
                return "bull"
            elif trend_strength < -0.02:
                return "bear" 
            else:
                return "sideways"
                
        except Exception:
            return "normal"

    def _rule_based_probabilities(self, ret_1, ret_5, cv, ma_ratio):
        """Convert rule-based logic to probabilities [sell, hold, buy]"""
        
        # Start with neutral
        sell_prob = 0.1
        hold_prob = 0.8 
        buy_prob = 0.1
        
        # Adjust based on momentum
        if ret_1 > 0.008 and ret_5 > 0.025:  # Strong bullish
            buy_prob = 0.7
            hold_prob = 0.25
            sell_prob = 0.05
        elif ret_1 < -0.008 and ret_5 < -0.025:  # Strong bearish
            sell_prob = 0.7
            hold_prob = 0.25
            buy_prob = 0.05
        elif ret_5 > 0.01:  # Moderate bullish
            buy_prob = 0.5
            hold_prob = 0.4
            sell_prob = 0.1
        elif ret_5 < -0.01:  # Moderate bearish
            sell_prob = 0.5
            hold_prob = 0.4
            buy_prob = 0.1
        
        # Volatility adjustment
        if cv > 0.08:  # High volatility - reduce conviction
            hold_prob += 0.2
            buy_prob *= 0.8
            sell_prob *= 0.8
        
        # MA trend adjustment
        if ma_ratio > 0.02:  # Strong uptrend
            buy_prob *= 1.3
            sell_prob *= 0.7
        elif ma_ratio < -0.02:  # Strong downtrend
            sell_prob *= 1.3
            buy_prob *= 0.7
        
        # Normalize to ensure sum = 1
        total = sell_prob + hold_prob + buy_prob
        return np.array([sell_prob/total, hold_prob/total, buy_prob/total])

    def _get_ml_predictions_as_probabilities(self, data):
        """Get ML predictions as [sell, hold, buy] probabilities from all models except RL"""
        try:
            # Prepare feature vector
            close_arr = data["Close"].to_numpy()
            if len(close_arr) < 10:
                return np.array([0.1, 0.8, 0.1])
            
            # Create features
            ret_1 = close_arr[-1] / close_arr[-2] - 1 if len(close_arr) >= 2 else 0
            ret_5 = close_arr[-1] / close_arr[-5] - 1 if len(close_arr) >= 5 else 0
            cv = np.std(close_arr[-10:]) / np.mean(close_arr[-10:]) if len(close_arr) >= 10 else 0.01
            ma_ratio = np.mean(close_arr[-3:]) / np.mean(close_arr[-10:]) - 1 if len(close_arr) >= 10 else 0
            
            features = np.array([ret_1, ret_5, cv, ma_ratio]).reshape(1, -1)
            probabilities = []
            
            # 1. ML Ensemble predictions
            if hasattr(self.ml_ensemble, 'is_trained') and self.ml_ensemble.is_trained:
                try:
                    ml_pred = self.ml_ensemble.predict(features)[0]
                    probabilities.append(ml_pred)
                except Exception:
                    pass
            
            # 2. TensorFlow predictions  
            if hasattr(self.tensorflow_model, 'is_trained') and self.tensorflow_model.is_trained:
                try:
                    tf_pred = self.tensorflow_model.predict(features.reshape(1, 1, -1))[0]
                    probabilities.append(tf_pred)
                except Exception:
                    pass
            
            # 3. Transformer predictions
            if hasattr(self, 'transformer_model') and self.transformer_model is not None:
                try:
                    transformer_pred = self._get_transformer_predictions(features, data)
                    probabilities.append(transformer_pred)
                except Exception:
                    pass
            
            if probabilities:
                # Average all model predictions
                ensemble_pred = np.mean(probabilities, axis=0)
                
                # Ensure proper format [sell, hold, buy]
                if len(ensemble_pred) == 3:
                    ensemble_pred = np.abs(ensemble_pred)
                    ensemble_pred = ensemble_pred / np.sum(ensemble_pred)
                    return ensemble_pred
            
            # Fallback to rule-based
            return self._rule_based_probabilities(ret_1, ret_5, cv, ma_ratio)
            
        except Exception:
            return np.array([0.1, 0.8, 0.1])

    def _get_transformer_predictions(self, features, data):
        """Get predictions from transformer model"""
        try:
            # Prepare data for transformer (needs sequence)
            close_arr = data["Close"].to_numpy()
            if len(close_arr) < 30:  # Transformer needs sequence
                return None
                
            # Create feature sequence for transformer
            feature_engine = QuantumFeatureEngine(self.config)
            enhanced_data = feature_engine.engineer_features(data)
            
            # Get available features that exist in the dataframe
            available_features = [col for col in ['Close', 'Volume', 'rsi', 'macd'] 
                                if col in enhanced_data.columns]
            
            if not available_features:
                return None
                
            # Extract last 30 timesteps as sequence
            sequence = enhanced_data.select(available_features).tail(30).to_numpy()
            
            if sequence.shape[0] != 30:
                return None
                
            # Convert to tensor
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            
            # Create dummy edge index for GNN component
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            # Get predictions
            self.transformer_model.eval()
            with torch.no_grad():
                action_logits, _ = self.transformer_model(sequence_tensor, edge_index)
                probabilities = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
                
            return probabilities
            
        except Exception as e:
            return None

    def _generate_signals(self, data):
        """Simplified signal generation that actually triggers trades"""
        try:
            # Get technical analysis signals
            tech_signal_strength, factors = self._calculate_technical_signals(data)
            
            # Get ML predictions (returns [sell, hold, buy] probabilities)
            ml_probs = self._get_ml_predictions_as_probabilities(data)
            
            # Combine technical signal with ML predictions
            # If technical signal is strong, boost corresponding ML probability
            if tech_signal_strength > 0.2:  # Bullish technical
                ml_probs[2] *= 1.5  # Boost buy probability
            elif tech_signal_strength < -0.2:  # Bearish technical  
                ml_probs[0] *= 1.5  # Boost sell probability
            
            # Normalize probabilities
            ml_probs = ml_probs / np.sum(ml_probs)
            
            # Use the max probability as confidence
            confidence = np.max(ml_probs)
            
            return np.array([ml_probs]), confidence
            
        except Exception as e:
            # Fallback that can actually generate trades
            return np.array([[0.2, 0.3, 0.5]]), 0.5  # Slight buy bias

    def _calculate_technical_signals(self, data):
        """Calculate technical analysis signals"""
        close_arr = to_talib_array(data["Close"].to_numpy())

        # Use optimizable MA periods
        short_ma = np.mean(close_arr[-self.config.short_ma_period:]) if len(close_arr) >= self.config.short_ma_period else close_arr[-1]
        long_ma = np.mean(close_arr[-self.config.long_ma_period:]) if len(close_arr) >= self.config.long_ma_period else close_arr[-1]

        # RSI with optimizable parameters
        try:
            rsi_vals = talib.RSI(close_arr, timeperiod=self.config.rsi_period)
            rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
        except:
            rsi = 50
            
        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_arr)
            macd_cross = macd[-1] - macd_signal[-1] if len(macd) > 1 else 0
        except:
            macd_cross = 0

        # Volume analysis with optimizable threshold
        volume_arr = to_talib_array(data["Volume"].to_numpy())
        avg_volume = np.mean(volume_arr[-10:]) if len(volume_arr) >= 10 else volume_arr[-1]
        current_volume = volume_arr[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # AUTO-OPTIMIZED Multi-factor signal with configurable weights
        signal_strength = 0.0
        factors = []
        
        # Trend factor with optimizable weight
        if short_ma > long_ma:
            factors.append(("Trend_Bull", self.config.trend_weight))
            signal_strength += self.config.trend_weight
        elif short_ma < long_ma:
            factors.append(("Trend_Bear", -self.config.trend_weight))
            signal_strength -= self.config.trend_weight
            
        # RSI factor with optimizable thresholds and weight
        if rsi < self.config.rsi_oversold:
            factors.append(("RSI_Oversold", self.config.rsi_weight))
            signal_strength += self.config.rsi_weight
        elif rsi > self.config.rsi_overbought:
            factors.append(("RSI_Overbought", -self.config.rsi_weight))
            signal_strength -= self.config.rsi_weight
            
        # MACD factor with optimizable weight
        if macd_cross > 0:
            factors.append(("MACD_Bull", self.config.macd_weight))
            signal_strength += self.config.macd_weight
        elif macd_cross < 0:
            factors.append(("MACD_Bear", -self.config.macd_weight))
            signal_strength -= self.config.macd_weight
            
        # Volume confirmation with optimizable threshold and weight
        if volume_ratio > self.config.volume_threshold:
            factors.append(("Volume_Confirm", self.config.volume_weight))
            signal_strength += self.config.volume_weight * (1 if signal_strength > 0 else -1)
        
        # Momentum factor with optimizable weight
        price_change = (close_arr[-1] / close_arr[-5] - 1) if len(close_arr) >= 5 else 0
        if abs(price_change) > 0.001:
            momentum_score = min(abs(price_change) * 100, self.config.momentum_weight)
            if price_change > 0:
                factors.append(("Momentum_Bull", momentum_score))
                signal_strength += momentum_score
            else:
                factors.append(("Momentum_Bear", -momentum_score))
                signal_strength -= momentum_score

        return signal_strength, factors

    def _get_ml_predictions(self, data):
        """Get predictions from ML ensemble with ENHANCED FALLBACK when not trained"""
        try:
            # Prepare feature vector
            close_arr = data["Close"].to_numpy()
            if len(close_arr) < 10:
                return self._smart_ml_fallback_enhanced(data)
            
            # Create features
            ret_1 = close_arr[-1] / close_arr[-2] - 1 if len(close_arr) >= 2 else 0
            ret_5 = close_arr[-1] / close_arr[-5] - 1 if len(close_arr) >= 5 else 0
            cv = np.std(close_arr[-10:]) / np.mean(close_arr[-10:]) if len(close_arr) >= 10 else 0.01
            ma_ratio = np.mean(close_arr[-3:]) / np.mean(close_arr[-10:]) - 1 if len(close_arr) >= 10 else 0
            
            features = np.array([ret_1, ret_5, cv, ma_ratio]).reshape(1, -1)
            
            # Check if ML models are trained
            if hasattr(self.ml_ensemble, 'is_trained') and self.ml_ensemble.is_trained:
                # Use trained ML models
                ml_pred = self.ml_ensemble.predict(features)[0]
                tf_pred = self.tensorflow_model.predict(features.reshape(1, 1, -1))[0] if tf is not None else ml_pred
                
                # Combine predictions (weighted average)
                combined_pred = (ml_pred + tf_pred) / 2
                
                # Convert to signal strength (-1 to 1)
                sell_prob, hold_prob, buy_prob = combined_pred
                ml_signal_strength = buy_prob - sell_prob
                
                return ml_signal_strength
            else:
                # ENHANCED FALLBACK: Use enhanced smart ML when models not trained
                return self._smart_ml_fallback_enhanced(data, features[0])
            
        except Exception as e:
            if not self.p.silent:
                console.print(f"[yellow]ML prediction error: {e}[/yellow]")
            # Return enhanced fallback
            return self._smart_ml_fallback_enhanced(data)

    def _smart_ml_fallback_enhanced(self, data, features=None):
        """ENHANCED smart ML-like predictions with more sophisticated logic"""
        try:
            close_arr = data["Close"].to_numpy()
            
            if features is not None:
                # Use provided features
                ret_1, ret_5, cv, ma_ratio = features
            else:
                # Calculate features from data
                if len(close_arr) < 5:
                    return 0.0
                
                ret_1 = close_arr[-1] / close_arr[-2] - 1 if len(close_arr) >= 2 else 0
                ret_5 = close_arr[-1] / close_arr[-5] - 1 if len(close_arr) >= 5 else 0
                cv = np.std(close_arr[-10:]) / np.mean(close_arr[-10:]) if len(close_arr) >= 10 else 0.01
                ma_ratio = np.mean(close_arr[-3:]) / np.mean(close_arr[-10:]) - 1 if len(close_arr) >= 10 else 0
            
            # Enhanced rule-based ML simulation with multiple factors
            signal = 0.0
            
            # 1. Momentum factors (enhanced)
            if ret_1 > 0.008:  # Strong short-term momentum
                signal += 0.35
            elif ret_1 > 0.003:  # Moderate short-term momentum
                signal += 0.2
            elif ret_1 < -0.008:  # Strong negative momentum
                signal -= 0.35
            elif ret_1 < -0.003:  # Moderate negative momentum
                signal -= 0.2
            
            # 2. Trend factors (enhanced)
            if ret_5 > 0.025:  # Strong trend
                signal += 0.4
            elif ret_5 > 0.01:  # Moderate trend
                signal += 0.25
            elif ret_5 < -0.025:  # Strong downtrend
                signal -= 0.4
            elif ret_5 < -0.01:  # Moderate downtrend
                signal -= 0.25
            
            # 3. Volatility factor (enhanced with regime awareness)
            if cv < 0.015:  # Very low volatility - trending market
                signal *= 1.5  # Amplify signals
            elif cv < 0.03:  # Low volatility
                signal *= 1.2
            elif cv > 0.08:  # Very high volatility - uncertain market
                signal *= 0.6  # Dampen signals
            elif cv > 0.05:  # High volatility
                signal *= 0.8
            
            # 4. Moving average crossover (enhanced)
            if ma_ratio > 0.02:  # Strong upward momentum
                signal += 0.3
            elif ma_ratio > 0.005:  # Moderate upward momentum
                signal += 0.15
            elif ma_ratio < -0.02:  # Strong downward momentum
                signal -= 0.3
            elif ma_ratio < -0.005:  # Moderate downward momentum
                signal -= 0.15
            
            # 5. Momentum consistency check
            if (ret_1 > 0 and ret_5 > 0 and ma_ratio > 0):  # All bullish
                signal += 0.1
            elif (ret_1 < 0 and ret_5 < 0 and ma_ratio < 0):  # All bearish
                signal -= 0.1
            
            # 6. Range trading detection
            if abs(ret_5) < 0.005 and cv < 0.02:  # Range-bound market
                signal *= 0.5  # Reduce signal in sideways markets
            
            # Clip to reasonable range and add some variance
            base_signal = np.clip(signal, -0.6, 0.6)
            
            # Add small random component to avoid repetitive signals
            noise = np.random.uniform(-0.05, 0.05)
            final_signal = base_signal + noise
            
            return np.clip(final_signal, -0.8, 0.8)
            
        except Exception as e:
            # Ultra-simple fallback
            return 0.0

    def _manage_existing_positions(self):
        """Check and manage stop losses and take profits for existing positions"""
        current_price = float(self.data_close[0])
        positions_to_close = []
        
        for pos_id, position_info in self.active_positions.items():
            entry_price = position_info['entry_price']
            stop_loss = position_info['stop_loss']
            take_profit = position_info['take_profit']
            direction = position_info['direction']
            quantity = position_info['quantity']
            
            should_close = False
            close_reason = ""
            
            if direction == 'LONG':
                if current_price <= stop_loss:
                    should_close = True
                    close_reason = f"STOP LOSS"
                elif current_price >= take_profit:
                    should_close = True
                    close_reason = f"TAKE PROFIT"
                    
            if should_close:
                # Calculate P&L
                pnl_usdt = (current_price - entry_price) * quantity
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Close position
                try:
                    current_position = self.getposition().size
                    if current_position > 0:
                        order = self.sell(size=current_position)
                        if order:
                            if not self.p.silent:
                                console.print(f"[yellow]🔄 POSITION CLOSED: {close_reason}[/yellow]")
                                console.print(f"[yellow]💰 P&L: ${pnl_usdt:.2f} USDT ({pnl_pct:+.2f}%)[/yellow]")
                            
                            # Update statistics
                            self.total_trades += 1
                            self.total_pnl_usdt += pnl_usdt
                            if pnl_usdt > 0:
                                self.winning_trades += 1
                                
                            positions_to_close.append(pos_id)
                except Exception as e:
                    print(e)
                    pass
        
        # Remove closed positions
        for pos_id in positions_to_close:
            del self.active_positions[pos_id]

    def _execute_crypto_trades(self, action_probs, regime_probs, confidence=None):
        """Execute crypto trades using USDT amounts with proper Risk:Reward"""
        try:
            sell_prob, hold_prob, buy_prob = action_probs[0]
            
            if not self.p.silent:
                print(f"Trade probabilities: sell={sell_prob:.3f}, hold={hold_prob:.3f}, buy={buy_prob:.3f}")
                print(f"Confidence: {confidence:.3f}, Regime: {self.current_regime}")
                print(f"Should enter trade: {self.risk_manager.should_enter_trade(confidence, self.current_regime)}")
            
            if confidence is None:
                confidence = max(sell_prob, hold_prob, buy_prob)
            
            self.confidence_scores.append(confidence)
            
            # Check if we should enter trade
            if not self.risk_manager.should_enter_trade(confidence, self.current_regime):
                return
            
            current_price = float(self.data_close[0])
            current_position_size = self.getposition().size
            available_cash = self.broker.getcash()
            
            # Only enter new position if we don't have one
            if buy_prob > sell_prob and buy_prob > hold_prob:
                # BUY SIGNAL
                if current_position_size <= 0:
                    
                    # Calculate stop loss and take profit
                    stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                        current_price, 'buy'
                    )
                    
                    # Calculate position size using risk management
                    position_calc = self.risk_manager.calculate_position_size_usdt(
                        entry_price=current_price,
                        stop_loss_price=stop_loss,
                        account_balance=self.broker.getvalue(),
                        confidence=confidence
                    )
                    
                    usdt_amount = position_calc['usdt_amount']
                    crypto_quantity = position_calc['quantity']
                    risk_amount = position_calc['risk_amount']
                    reward_target = position_calc['reward_target']
                    
                    # Check if we have enough cash
                    if usdt_amount > available_cash:
                        usdt_amount = available_cash * 0.95
                        crypto_quantity = usdt_amount / current_price
                        
                    # Check minimum trade amount
                    if usdt_amount >= self.config.min_trade_amount and crypto_quantity > 0:
                        
                        order = self.buy(size=crypto_quantity)
                        
                        if order:
                            # Store position info for management
                            position_id = f"LONG_{len(self.active_positions)}"
                            self.active_positions[position_id] = {
                                'direction': 'LONG',
                                'entry_price': current_price,
                                'quantity': crypto_quantity,
                                'usdt_invested': usdt_amount,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'risk_amount': risk_amount,
                                'reward_target': reward_target,
                                'entry_time': self.datas[0].datetime.datetime(0)
                            }
                            
                            if not self.p.silent:
                                console.print(f"[green]🟢 LONG POSITION OPENED[/green]")
                                console.print(f"[green]💵 Amount: ${usdt_amount:.2f} USDT ({crypto_quantity:.6f} BTC)[/green]")
                                console.print(f"[green]📈 Entry: ${current_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}[/green]")
                                console.print(f"[green]🎯 Confidence: {confidence:.2f} | Regime: {self.current_regime}[/green]")

        except Exception as e:
            print('cant process further', e)
            pass

    def _risk_check(self):
        """Enhanced risk management check"""
        try:
            current_value = self.broker.getvalue()
            
            # Update peak value and check drawdown
            if not hasattr(self, 'peak_value'):
                self.peak_value = current_value
            else:
                self.peak_value = max(self.peak_value, current_value)
                
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.config.max_drawdown_threshold:
                return False

            # Check maximum number of concurrent positions
            if len(self.active_positions) >= 3:
                return False
                
            # Check if we have minimum cash for trading
            if self.broker.getcash() < self.config.min_trade_amount:
                return False

            return True

        except Exception as e:
            return False

    def _update_portfolio_tracking(self):
        """Update portfolio performance tracking"""
        try:
            current_value = self.broker.getvalue()
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            portfolio_info = {
                'datetime': self.datas[0].datetime.datetime(0),
                'value': current_value,
                'cash': self.broker.getcash(),
                'position_size': self.getposition().size,
                'active_positions': len(self.active_positions),
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'total_pnl_usdt': self.total_pnl_usdt,
                'regime': self.current_regime,
                'confidence': self.confidence_scores[-1] if self.confidence_scores else 0
            }

            self.portfolio_history.append(portfolio_info)

        except Exception as e:
            print(e)
            pass

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                if hasattr(self, 'active_orders') and self.active_orders:
                    self.active_orders[-1].order_id = order.ref

        if order.status in [order.Submitted, order.Accepted]:
            # return
            if order.status in [order.Completed]:
                self.update_order_history(order)
                if order.isbuy():
                    self.action = "buy"
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm

                else:  # Sell
                    self.action = "sell"

                self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print("Order Canceled/Margin/Rejected")
                print("Canceled: {}".format(order.status == order.Canceled))
                print("Margin: {}".format(order.status == order.Margin))
                print("Rejected: {}".format(order.status == order.Rejected))
        self.order = None

    def stop(self):
        """Called when strategy finishes - print final statistics"""
        try:
            final_value = self.broker.getvalue()
            initial_value = self.config.init_cash
            total_return = (final_value - initial_value) / initial_value
            
            if not self.p.silent:
                console.print(f"\n[bold cyan]📊 FINAL CRYPTO TRADING STATISTICS:[/bold cyan]")
                console.print(f"[cyan]💰 Initial Capital: ${initial_value:,.2f} USDT[/cyan]")
                console.print(f"[cyan]🏆 Final Value: ${final_value:,.2f} USDT[/cyan]")
                console.print(f"[cyan]📈 Total Return: {total_return:.2%}[/cyan]")
                console.print(f"[cyan]🔢 Total Trades: {self.total_trades}[/cyan]")
                console.print(f"[cyan]✅ Winning Trades: {self.winning_trades}[/cyan]")
                if self.total_trades > 0:
                    console.print(f"[cyan]🎯 Win Rate: {self.winning_trades/self.total_trades:.2%}[/cyan]")
                console.print(f"[cyan]💵 Total P&L: ${self.total_pnl_usdt:.2f} USDT[/cyan]")
                console.print(f"[cyan]📊 Bars Processed: {self.bar_count:,} | Signals: {self.signal_count:,}[/cyan]")
            
            # Store results for optimization
            self.final_return = total_return
            self.final_trades = self.total_trades
            self.final_win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0
            self.final_pnl = self.total_pnl_usdt
            
        except Exception as e:
            # Fallback values for failed runs
            self.final_return = -0.5
            self.final_trades = 0
            self.final_win_rate = 0
            self.final_pnl = -50000

import joblib
from pathlib import Path
import torch

class ModelPersistence:
    """Handles saving and loading of all trained models"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.models_dir = config.cache_dir / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_all_models(self, trained_models: dict, optimized_params: Optional[TradingConfig] = None):
        """Save all trained models and parameters"""
        console.print("[cyan]💾 Saving trained models...[/cyan]")
        saved_models = {}

        # Save PyTorch Transformer model (if present)
        if trained_models.get('transformer') is not None:
            try:
                model = trained_models['transformer']
                model_path = self.models_dir / "transformer_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config.__dict__,
                    'input_dim': getattr(model, 'input_dim', None),
                    'transformer_dim': getattr(model, 'transformer_dim', None),
                    'gnn_hidden_dim': getattr(model, 'gnn_hidden_dim', None),
                }, model_path)
                saved_models['transformer'] = str(model_path)
                console.print(f"[green]✅ Transformer model saved to {model_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save transformer: {e}[/red]")

        # Save RL Agent (stable-baselines3 writes .zip)
        if trained_models.get('rl_agent') is not None:
            try:
                rl_base = self.models_dir / "rl_agent"
                trained_models['rl_agent'].save(str(rl_base))
                rl_zip = rl_base.with_suffix(".zip")
                saved_models['rl_agent'] = str(rl_zip if rl_zip.exists() else rl_base)
                console.print(f"[green]✅ RL agent saved to {saved_models['rl_agent']}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save RL agent: {e}[/red]")

        # Save ML Ensemble
        if trained_models.get('ml_ensemble') is not None:
            try:
                ensemble_path = self.models_dir / "ml_ensemble.pkl"
                joblib.dump(trained_models['ml_ensemble'], ensemble_path)
                saved_models['ml_ensemble'] = str(ensemble_path)
                console.print(f"[green]✅ ML ensemble saved to {ensemble_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save ML ensemble: {e}[/red]")

        # Save TensorFlow model
        if trained_models.get('tensorflow') is not None and getattr(trained_models['tensorflow'], 'model', None) is not None:
            try:
                tf_path = self.models_dir / "tensorflow_model.keras"
                trained_models['tensorflow'].model.save(str(tf_path))
                saved_models['tensorflow'] = str(tf_path)
                console.print(f"[green]✅ TensorFlow model saved to {tf_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save TensorFlow model: {e}[/red]")

        # Save optimized parameters ONCE with JSON-safe conversion
        if optimized_params is not None:
            try:
                params_path = self.models_dir / "optimized_params.json"
                # Convert dataclass -> JSON-safe dict (Path -> str)
                params_dict = {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in getattr(optimized_params, "__dict__", optimized_params).items()
                }
                with open(params_path, 'w') as f:
                    json.dump(params_dict, f, indent=2)
                saved_models['params'] = str(params_path)
                console.print(f"[green]✅ Optimized parameters saved to {params_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to save parameters: {e}[/red]")

        # Save metadata last
        try:
            metadata_path = self.models_dir / "models_metadata.json"
            metadata = {
                'saved_at': datetime.datetime.now().isoformat(),
                'config_hash': hashlib.md5(str(self.config.__dict__).encode()).hexdigest(),
                'models': saved_models
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            console.print(f"[bold green]✅ All models saved successfully![/bold green]")
        except Exception as e:
            console.print(f"[red]Failed to write models metadata: {e}[/red]")

        return saved_models

    def load_all_models(self) -> tuple:
        """Load all previously trained models"""
        console.print("[cyan]📂 Loading trained models...[/cyan]")
        
        loaded_models = {}
        loaded_params = None
        
        if not self.models_dir.exists():
            console.print("[yellow]No saved models found[/yellow]")
            return {}, None
        
        metadata_path = self.models_dir / "models_metadata.json"
        if not metadata_path.exists():
            console.print("[yellow]No models metadata found[/yellow]")
            return {}, None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        console.print(f"[cyan]Found models saved at: {metadata['saved_at']}[/cyan]")
        
        # Load PyTorch Transformer model
        transformer_path = self.models_dir / "transformer_model.pth"
        if transformer_path.exists():
            try:
                # Fix: Add weights_only=False for PyTorch 2.6 compatibility
                checkpoint = torch.load(transformer_path, map_location=DEVICE, weights_only=False)
                # Recreate model with saved dimensions
                model = MultiModalTransformerGNN(
                    input_dim=checkpoint['input_dim'],
                    transformer_dim=checkpoint['transformer_dim'],
                    gnn_hidden_dim=checkpoint['gnn_hidden_dim'],
                    num_heads=self.config.num_attention_heads,
                    num_layers=self.config.num_transformer_layers
                ).to(DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval() # TUrn off training 
                loaded_models['transformer'] = model
                console.print("[green]✅ Transformer model loaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load transformer: {e}[/red]")
        
        # Load RL Agent
        rl_path = self.models_dir / "rl_agent.zip"  # stable-baselines3 adds .zip extension
        if not rl_path.exists():
            rl_path = self.models_dir / "rl_agent"
        if rl_path.exists():
            try:
                if self.config.rl_algorithm == "PPO":
                    model = PPO.load(str(rl_path), device=DEVICE)
                elif self.config.rl_algorithm == "SAC":
                    model = SAC.load(str(rl_path), device=DEVICE)
                elif self.config.rl_algorithm == "TD3":
                    model = TD3.load(str(rl_path), device=DEVICE)
                loaded_models['rl_agent'] = model
                console.print("[green]✅ RL agent loaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load RL agent: {e}[/red]")
        
        # Load ML Ensemble
        ensemble_path = self.models_dir / "ml_ensemble.pkl"
        if ensemble_path.exists():
            try:
                ml_ensemble = joblib.load(ensemble_path)
                loaded_models['ml_ensemble'] = ml_ensemble
                console.print("[green]✅ ML ensemble loaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load ML ensemble: {e}[/red]")
        
        # Load TensorFlow model
        tf_path = self.models_dir / "tensorflow_model.keras"
        if not tf_path.exists():
            tf_path = self.models_dir / "tensorflow_model"  # Fallback for old saves
        if tf_path.exists() and tf is not None:
            try:
                tf_predictor = TensorFlowPredictor(self.config)
                tf_predictor.model = tf.keras.models.load_model(str(tf_path))
                tf_predictor.is_trained = True
                loaded_models['tensorflow'] = tf_predictor
                console.print("[green]✅ TensorFlow model loaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load TensorFlow model: {e}[/red]")
        
        # Load optimized parameters and rebuild TradingConfig
        params_path = self.models_dir / "optimized_params.json"
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    params_dict = json.load(f)
                try:
                    # Fix: Convert string paths back to Path objects
                    if 'cache_dir' in params_dict and isinstance(params_dict['cache_dir'], str):
                        params_dict['cache_dir'] = Path(params_dict['cache_dir'])
                    
                    # Try to reconstruct as TradingConfig
                    loaded_params = TradingConfig(**params_dict)
                    console.print(loaded_params)
                except TypeError as e:
                    console.print(f"[yellow]TradingConfig reconstruction failed: {e}[/yellow]")
                    # Fallback: return dict if schema has changed
                    loaded_params = params_dict
                    console.print(loaded_params)
                    console.print(params_dict)
                console.print("[green]✅ Optimized parameters loaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load parameters: {e}[/red]")
        
        console.print(f"[bold green]✅ Loaded {len(loaded_models)} models successfully![/bold green]")
        return loaded_models, loaded_params

# ==================== POLARS DATA FEED CLASS ====================

class PolarsData(bt.feed.DataBase):
    """Uses a Polars DataFrame as the feed source"""

    params = (
        ('nocase', True),
        ('datetime', 0),
        ('open', 1),      
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
    )

    datafields = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'
    ]

    def __init__(self):
        super(PolarsData, self).__init__()

        if isinstance(self.p.dataname, pl.DataFrame):
            datetime_col = self.p.dataname.columns[0]
            self.p.dataname = self.p.dataname.sort(datetime_col)
    
        self.colnames = self.p.dataname.columns
        self._colmapping = {}
        
        for datafield in self.getlinealiases():
            param_value = getattr(self.params, datafield)
            
            if isinstance(param_value, int):
                if param_value >= 0:
                    if param_value < len(self.colnames):
                        self._colmapping[datafield] = param_value
                    else:
                        self._colmapping[datafield] = None
                elif param_value == -1:
                    found = False
                    for i, colname in enumerate(self.colnames):
                        if self.p.nocase:
                            found = datafield.lower() == colname.lower()
                        else:
                            found = datafield == colname
                            
                        if found:
                            self._colmapping[datafield] = i
                            break
                    
                    if not found:
                        self._colmapping[datafield] = None
                else:
                    self._colmapping[datafield] = None
            
            elif isinstance(param_value, str):
                try:
                    col_idx = self.colnames.index(param_value)
                    self._colmapping[datafield] = col_idx
                except ValueError:
                    if self.p.nocase:
                        found = False
                        for i, colname in enumerate(self.colnames):
                            if param_value.lower() == colname.lower():
                                self._colmapping[datafield] = i
                                found = True
                                break
                        if not found:
                            self._colmapping[datafield] = None
                    else:
                        self._colmapping[datafield] = None
            else:
                self._colmapping[datafield] = None

    def start(self):
        super(PolarsData, self).start()
        self._idx = -1

    def _load(self):
        self._idx += 1
        if self._idx >= len(self.p.dataname):
            return False

        for datafield in self.getlinealiases():
            if datafield == 'datetime':
                continue
            col_idx = self._colmapping[datafield]
            if col_idx is None:
                continue

            line = getattr(self.lines, datafield)
            try:
                val = self.p.dataname[self.colnames[col_idx]][self._idx]
                if hasattr(val, "item"):
                    val = val.item()
                line[0] = float(val)
            except Exception as e:
                line[0] = float('nan')

        dt_idx = self._colmapping['datetime']
        if dt_idx is not None:
            try:
                dt_value = self.p.dataname[self.colnames[dt_idx]][self._idx]
                if hasattr(dt_value, "item"):
                    dt_value = dt_value.item()
                    
                from datetime import datetime
                from backtrader import date2num
                
                if isinstance(dt_value, str):
                    dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                elif isinstance(dt_value, (int, float)):
                    dt = datetime.fromtimestamp(float(dt_value)/1000 if dt_value > 1e10 else float(dt_value))
                else:
                    dt = dt_value
                self.lines.datetime[0] = date2num(dt)
            except Exception as e:
                self.lines.datetime[0] = float('nan')

        return True

# ==================== DATASET CLASS FOR TRANSFORMER TRAINING ====================

class FinancialDataset(Dataset):
    """Dataset class for financial time series - FIXED column references"""

    def __init__(self, data: pl.DataFrame, feature_columns: List[str], config: TradingConfig):
        self.data = data
        self.feature_columns = feature_columns
        self.config = config
        self.sequence_length = 30

        # Engineer features using Polars
        feature_engine = QuantumFeatureEngine(config)
        self.features_df = feature_engine.engineer_features(data)

        # Create labels using CORRECT column name (database format)
        close_series = data.select(pl.col("Close")).to_series()
        returns_series = close_series.pct_change().shift(-1)

        # Convert to numpy for processing
        returns = returns_series.to_numpy()

        # Convert returns to classification labels - FIXED INDEXING
        self.labels = []
        for i, ret in enumerate(returns):
            if np.isnan(ret) or i >= len(returns) - self.sequence_length - 1:
                self.labels.append([1, 0])  # Hold, normal regime
            elif ret > 0.01:  # > 1% gain
                self.labels.append([2, 1])  # Buy, bull regime
            elif ret < -0.01:  # < -1% loss
                self.labels.append([0, 2])  # Sell, bear regime
            else:
                self.labels.append([1, 0])  # Hold, normal regime

        self.labels = np.array(self.labels)

        # Create edge index for graph (simple star topology)
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        console.print(f"[cyan]Dataset created: {len(self.data)} samples, {len(self.labels)} labels, sequence_length={self.sequence_length}[/cyan]")

    def __len__(self):
        max_len = len(self.data) - self.sequence_length - 1
        return max(0, max_len)

    def __getitem__(self, idx):
        # Get feature sequence
        start_idx = idx
        end_idx = idx + self.sequence_length

        # Get available feature columns that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in self.features_df.columns]
        if not available_features:
            # Use basic features as fallback - with CORRECT column names
            available_features = ['Close', 'Volume'][:min(2, len(self.features_df.columns))]

        # Extract feature sequence using Polars
        try:
            feature_sequence = self.features_df.select(available_features).slice(start_idx, self.sequence_length).to_numpy()
        except Exception as e:
            console.print(f"[red]Error extracting features: {e}[/red]")
            feature_sequence = np.ones((self.sequence_length, len(available_features)))

        # Ensure feature sequence has correct shape
        if feature_sequence.shape[0] != self.sequence_length:
            if feature_sequence.shape[0] < self.sequence_length:
                padding = np.zeros((self.sequence_length - feature_sequence.shape[0], feature_sequence.shape[1]))
                feature_sequence = np.vstack([feature_sequence, padding])
            else:
                feature_sequence = feature_sequence[:self.sequence_length]

        # Get labels - FIXED INDEXING
        label_idx = min(end_idx, len(self.labels) - 1)
        label = self.labels[label_idx]

        return (
            torch.tensor(feature_sequence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            self.edge_index
        )

# ==================== COMPREHENSIVE TRAINING PIPELINE ====================

class ModelTrainer:
    """COMPLETE training pipeline for ALL models with multithreading"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.device = DEVICE
        console.print(f"[cyan]Using device: {self.device}[/cyan]")

        # Initialize Weights & Biases for experiment tracking
        try:
            wandb.init(project="btquant-v2025-complete", config=config.__dict__)
            console.print("[green]Weights & Biases initialized successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: W&B initialization failed: {e}[/yellow]")

    def train_all_models(self,
                        train_data: pl.DataFrame,
                        val_data: pl.DataFrame,
                        feature_columns: List[str]):
        """Train all models in parallel using multithreading"""
        console.print("[bold blue]🚀 Training ALL Models in Parallel...[/bold blue]")

        # Results storage
        trained_models = {}

        # FIXED: Clone DataFrames before passing to parallel functions
        train_data_clone = train_data.clone()
        val_data_clone = val_data.clone()
        
        # Define training functions with cloned data
        def train_transformer():
            return self.train_transformer_model(
                train_data_clone.clone(), 
                val_data_clone.clone(), 
                feature_columns, 
                epochs=20
            )

        # def train_rl():
        #     train_pandas = train_data_clone.clone().to_pandas()
        #     return self.train_rl_agent(train_pandas, feature_columns, total_timesteps=500_000)

        def train_ml_ensemble():
            return self.train_ml_ensemble(train_data_clone.clone(), feature_columns)

        def train_tensorflow():
            return self.train_tensorflow_model(train_data_clone.clone(), feature_columns)

        # Execute training in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'transformer': executor.submit(train_transformer),
                # 'rl_agent': executor.submit(train_rl),
                'ml_ensemble': executor.submit(train_ml_ensemble),
                'tensorflow': executor.submit(train_tensorflow)
            }

            # Collect results
            for model_name, future in futures.items():
                try:
                    model = future.result()
                    if model is not None:
                        trained_models[model_name] = model
                        console.print(f"[green]✅ {model_name} training completed[/green]")
                    else:
                        console.print(f"[yellow]⚠️ {model_name} training failed[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ {model_name} training error: {e}[/red]")

        console.print(f"[bold green]🎉 Parallel training completed! {len(trained_models)} models trained[/bold green]")
        return trained_models


    def train_transformer_model(self,
                               train_data: pl.DataFrame,
                               val_data: pl.DataFrame,
                               feature_columns: List[str],
                               epochs: int = 100):
        """Train the Transformer-GNN model with CUDA error handling"""
        
        console.print("[bold blue]Training Transformer-GNN Model...[/bold blue]")
        
        try:
            # Prepare data
            train_data = train_data.clone()
            val_data = val_data.clone()
            train_dataset = FinancialDataset(train_data, feature_columns, self.config)
            val_dataset = FinancialDataset(val_data, feature_columns, self.config)

            # Check if datasets have sufficient samples
            if len(train_dataset) < self.config.batch_size:
                console.print(f"[red]Training dataset too small: {len(train_dataset)} samples[/red]")
                return None

            if len(val_dataset) < self.config.batch_size:
                console.print(f"[red]Validation dataset too small: {len(val_dataset)} samples[/red]")
                return None

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

            # Initialize model
            input_dim = len(feature_columns)
            model = MultiModalTransformerGNN(
                input_dim=input_dim,
                transformer_dim=self.config.transformer_dim,
                gnn_hidden_dim=self.config.gnn_hidden_dim,
                num_heads=self.config.num_attention_heads,
                num_layers=self.config.num_transformer_layers
            ).to(self.device)

            # Optimizer and loss function
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_acc = 0.0
                train_batches = 0

                for batch in train_loader:
                    try:
                        features, targets, edge_index = batch
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                        edge_index = edge_index.to(self.device)

                        optimizer.zero_grad()
                        action_logits, regime_logits = model(features, edge_index)

                        loss = criterion(action_logits, targets[:, 0]) + 0.3 * criterion(regime_logits, targets[:, 1])

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        train_loss += loss.item()
                        train_acc += (action_logits.argmax(1) == targets[:, 0]).float().mean().item()
                        train_batches += 1

                    except RuntimeError as e:
                        console.print("[red]Error in model training at cuda gpu kernel level - Fallback to CPU initiated... :-([/red]")
                        if "CUDA" in str(e):
                            # Move everything to CPU
                            model = model.cpu()
                            self.device = torch.device('cpu')
                            features = features.cpu()
                            targets = targets.cpu()
                            edge_index = edge_index.cpu()

                            # Retry the forward pass
                            optimizer.zero_grad()
                            action_logits, regime_logits = model(features, edge_index)
                            loss = criterion(action_logits, targets[:, 0]) + 0.3 * criterion(regime_logits, targets[:, 1])

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()

                            train_loss += loss.item()
                            train_acc += (action_logits.argmax(1) == targets[:, 0]).float().mean().item()
                            train_batches += 1
                        else:
                            raise e

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            features, targets, edge_index = batch
                            features = features.to(self.device)
                            targets = targets.to(self.device)
                            edge_index = edge_index.to(self.device)

                            action_logits, regime_logits = model(features, edge_index)

                            loss = criterion(action_logits, targets[:, 0]) + 0.3 * criterion(regime_logits, targets[:, 1])

                            val_loss += loss.item()
                            val_acc += (action_logits.argmax(1) == targets[:, 0]).float().mean().item()
                            val_batches += 1

                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                features = features.cpu()
                                targets = targets.cpu()
                                edge_index = edge_index.cpu()
                                model = model.cpu()

                                action_logits, regime_logits = model(features, edge_index)
                                loss = criterion(action_logits, targets[:, 0]) + 0.3 * criterion(regime_logits, targets[:, 1])

                                val_loss += loss.item()
                                val_acc += (action_logits.argmax(1) == targets[:, 0]).float().mean().item()
                                val_batches += 1
                            else:
                                raise e

                # Calculate average losses
                if train_batches > 0:
                    train_loss /= train_batches
                    train_acc /= train_batches
                if val_batches > 0:
                    val_loss /= val_batches
                    val_acc /= val_batches

                scheduler.step(val_loss)

                # Log metrics to wandb
                try:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                except:
                    pass

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, self.config.cache_dir / 'best_transformer_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                        break

                if epoch % 10 == 0:
                    console.print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            console.print("[green]Transformer model training completed successfully![/green]")
            return model

        except Exception as e:
            console.print(f"[red]Error in transformer training: {e}[/red]")
            return None

    # def train_rl_agent(self,
    #                   env_data: pd.DataFrame,
    #                   feature_columns: List[str],
    #                   total_timesteps: int = 100000):
    #     """Train the Deep RL agent with proper environment"""
        
    #     console.print("[bold blue]Training Deep RL Agent...[/bold blue]")
        
    #     try:
    #         # Clean the data for RL environment
    #         env_data_clean = env_data.copy()
            
    #         # Remove datetime columns
    #         datetime_cols = env_data_clean.select_dtypes(include=['datetime64']).columns
    #         if len(datetime_cols) > 0:
    #             env_data_clean = env_data_clean.drop(columns=datetime_cols)
            
    #         # Convert any object columns to numeric
    #         for col in env_data_clean.columns:
    #             if env_data_clean[col].dtype == 'object':
    #                 try:
    #                     env_data_clean[col] = pd.to_numeric(env_data_clean[col], errors='coerce')
    #                 except:
    #                     env_data_clean = env_data_clean.drop(columns=[col])
            
    #         # Fill NaNs
    #         env_data_clean = env_data_clean.fillna(0)
            
    #         # Create training environment
    #         env = AdvancedTradingEnv(
    #             df=env_data_clean,
    #             feature_columns=feature_columns,
    #             initial_balance=self.config.init_cash,
    #             transaction_cost=self.config.commission
    #         )
            
    #         # Wrap in DummyVecEnv for stable-baselines3
    #         env = DummyVecEnv([lambda: env])
            
    #         # Force CPU for RL to avoid CUDA issues
    #         # device_str = 'cpu'
    #         device_str = 'cuda'
            
    #         # Initialize RL agent
    #         if self.config.rl_algorithm == "PPO":
    #             model = PPO(
    #                 'MlpPolicy', 
    #                 env, 
    #                 learning_rate=self.config.learning_rate,
    #                 gamma=self.config.gamma,
    #                 verbose=1,
    #                 device=device_str
    #             )
    #         elif self.config.rl_algorithm == "SAC":
    #             model = SAC(
    #                 'MlpPolicy', 
    #                 env, 
    #                 learning_rate=self.config.learning_rate,
    #                 gamma=self.config.gamma,
    #                 verbose=1,
    #                 device=device_str
    #             )
    #         elif self.config.rl_algorithm == "TD3":
    #             model = TD3(
    #                 'MlpPolicy', 
    #                 env, 
    #                 learning_rate=self.config.learning_rate,
    #                 gamma=self.config.gamma,
    #                 verbose=1,
    #                 device=device_str
    #             )
            
    #         # Training callback for logging
    #         class WandbCallback(BaseCallback):
    #             def __init__(self, verbose=0):
    #                 super().__init__(verbose)
                    
    #             def _on_step(self) -> bool:
    #                 if self.n_calls % 1000 == 0:
    #                     try:
    #                         wandb.log({
    #                             'rl_timestep': self.n_calls,
    #                             'episode_reward': self.locals.get('episode_reward', 0)
    #                         })
    #                     except:
    #                         pass
    #                 return True
            
    #         # Train the agent
    #         callback = WandbCallback()
    #         model.learn(total_timesteps=total_timesteps, callback=callback)
            
    #         # Save the trained agent
    #         model.save(self.config.cache_dir / "rl_agent")
            
    #         console.print("[green]RL agent training completed![/green]")
    #         return model
            
    #     except Exception as e:
    #         console.print(f"[red]Error in RL training: {e}[/red]")
    #         return None

    def train_ml_ensemble(self, data: pl.DataFrame, feature_columns: List[str]):
        """Train ML ensemble with feature engineering - FIXED borrowing issue"""
        
        console.print("[bold blue]Training ML Ensemble...[/bold blue]")
        
        try:
            # CLONE the DataFrame to avoid borrowing conflicts
            data_copy = data.clone()
            
            # Engineer features on the copy
            feature_engine = QuantumFeatureEngine(self.config)
            enhanced_data = feature_engine.engineer_features(data_copy)
            
            # Prepare features and labels
            available_features = [col for col in feature_columns if col in enhanced_data.columns]
            if not available_features:
                available_features = [col for col in enhanced_data.columns if col not in ['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume']][:10]
            
            # CONVERT TO NUMPY IMMEDIATELY to avoid further Polars borrowing
            X = enhanced_data.select(available_features).to_numpy()
            
            # Create labels from returns - use separate operations
            close_values = enhanced_data.select(pl.col("Close")).to_numpy().flatten()
            returns = np.diff(close_values) / close_values[:-1]
            
            y = []
            for ret in returns:
                if np.isnan(ret):
                    y.append(1)  # Hold
                elif ret > 0.005:  # > 0.5% gain
                    y.append(2)  # Buy
                elif ret < -0.005:  # < -0.5% loss
                    y.append(0)  # Sell
                else:
                    y.append(1)  # Hold
            
            y = np.array(y)
            
            # Match array lengths
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Remove NaN rows
            valid_idx = ~np.isnan(X).any(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 100:
                console.print("[yellow]Insufficient data for ML training[/yellow]")
                return None
            
            # Train ensemble
            ml_ensemble = MLEnsemble(self.config)
            success = ml_ensemble.train(X, y)
            
            if success:
                console.print("[green]ML ensemble training completed![/green]")
                return ml_ensemble
            else:
                return None
                
        except Exception as e:
            console.print(f"[red]Error in ML ensemble training: {e}[/red]")
            return None

    def train_tensorflow_model(self, data: pl.DataFrame, feature_columns: List[str]):
        """Train TensorFlow model - FIXED borrowing"""
        
        if tf is None:
            console.print("[yellow]TensorFlow not available, skipping[/yellow]")
            return None
            
        console.print("[bold blue]Training TensorFlow Model...[/bold blue]")
        
        try:
            # CLONE to avoid borrowing
            data_copy = data.clone()
            
            # Prepare data similar to ML ensemble
            feature_engine = QuantumFeatureEngine(self.config)
            enhanced_data = feature_engine.engineer_features(data_copy)
            
            available_features = [col for col in feature_columns if col in enhanced_data.columns][:10]
            if not available_features:
                available_features = [col for col in enhanced_data.columns if col not in ['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume']][:10]
            
            X = enhanced_data.select(available_features).to_numpy()
            
            # Create labels
            close_values = enhanced_data.select(pl.col("Close")).to_numpy().flatten()
            returns = np.diff(close_values) / close_values[:-1]
            
            y = []
            for ret in returns:
                if np.isnan(ret):
                    y.append(1)  # Hold
                elif ret > 0.005:
                    y.append(2)  # Buy
                elif ret < -0.005:
                    y.append(0)  # Sell
                else:
                    y.append(1)  # Hold
            
            y = np.array(y)
            
            # Match lengths
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Create sequences for LSTM
            sequence_length = 30    # NOTE :: oh dead - this needs to be bruteforced with optuna too - might fall back to more simple LTSM
            #                              :: github.com/itsXactlY/2-algos-1-cup
            X_seq = []
            y_seq = []
            
            for i in range(sequence_length, len(X)):
                X_seq.append(X[i-sequence_length:i])
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            if len(X_seq) < 100:
                console.print("[yellow]Insufficient data for TensorFlow training[/yellow]")
                return None
            
            # Train TensorFlow model
            tf_model = TensorFlowPredictor(self.config)
            success = tf_model.train(X_seq, y_seq, epochs=20)
            
            if success:
                console.print("[green]TensorFlow model training completed![/green]")
                return tf_model
            else:
                return None
                
        except Exception as e:
            console.print(f"[red]Error in TensorFlow training: {e}[/red]")
            return None

# ==================== AUTO-OPTIMIZATION SYSTEM ====================

def build_optuna_storage(connection_string: str):
    """Build Optuna storage from connection string"""
    try:
        import optuna
        return optuna.storages.RDBStorage(
            url=connection_string,
            engine_kwargs={'pool_pre_ping': True, 'pool_recycle': 300}
        )
    except Exception as e:
        console.print(f"[red]Failed to build Optuna storage: {e}[/red]")
        return None

class OptimizationEngine:
    """Enhanced with MSSQL persistence and resume capability"""
    
    def __init__(self, cache: DataCache, connection_string: str = None, study_name: str = "BTQuant_v2025_Optimization"):
        self.cache = cache
        self.connection_string = connection_string
        self.study_name = study_name
        self.storage = None
        
        # Setup storage if provided
        if connection_string:
            self.storage = build_optuna_storage(connection_string)
            if self.storage:
                console.print(f"[green]Connected to MSSQL Optuna storage[/green]")
    
    def create_or_load_study(self) -> optuna.Study:
        """Create new study or load existing one from MSSQL"""
        
        if self.storage:
            try:
                # Try to load existing study
                study = optuna.load_study(
                    study_name=self.study_name, 
                    storage=self.storage
                )
                console.print(f"[cyan]Loaded existing study '{self.study_name}' with {len(study.trials)} trials[/cyan]")
                
                if len(study.trials) > 0:
                    best_trial = study.best_trial
                    console.print(f"[cyan]Best trial so far: #{best_trial.number} with score {study.best_value:.4f}[/cyan]")
                    
                return study
                
            except Exception as e:
                console.print(f"[yellow]Study not found, creating new: {e}[/yellow]")
                
                # Create new study
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
                )
                console.print(f"[green]Created new study '{self.study_name}'[/green]")
                return study
        else:
            # In-memory study as fallback
            return optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
            )
    
    def get_best_config_from_study(self) -> Optional[TradingConfig]:
        """Load best configuration from existing study"""
        if not self.storage:
            return None
            
        try:
            study = optuna.load_study(study_name=self.study_name, storage=self.storage)
            if len(study.trials) == 0:
                console.print("[yellow]No trials found in study[/yellow]")
                return None
                
            best_trial = study.best_trial
            best_config = self.create_optimizable_config(best_trial)
            
            console.print(f"[green]Loaded best config from trial #{best_trial.number} (score: {study.best_value:.4f})[/green]")
            return best_config
            
        except Exception as e:
            console.print(f"[red]Failed to load best config: {e}[/red]")
            return None
    
    def optimize_parameters(self, n_trials: int = 100, resume: bool = True) -> TradingConfig:
        """Run optimization with MSSQL persistence and resume capability"""
        
        # Create or load study
        study = self.create_or_load_study()
        
        if resume and len(study.trials) > 0:
            console.print(f"[cyan]Resuming optimization from {len(study.trials)} existing trials[/cyan]")
            remaining_trials = max(0, n_trials - len(study.trials))
            if remaining_trials == 0:
                console.print("[yellow]Already completed requested number of trials[/yellow]")
                return self.create_optimizable_config(study.best_trial)
            console.print(f"[cyan]Running {remaining_trials} additional trials[/cyan]")
            n_trials = remaining_trials
        
        # Early stopping tracker
        best_scores_history = []
        no_improvement_count = 0
        patience = 30
        min_improvement = 0.001
        
        def progress_callback(study, trial):
            nonlocal no_improvement_count, best_scores_history
            
            if trial.number % 5 == 0:
                console.print(f"[cyan]Trial {trial.number} | Best: {study.best_value:.4f} | Current: {trial.value:.4f} | No improve: {no_improvement_count}[/cyan]")
            
            # Early stopping logic
            best_scores_history.append(study.best_value)
            
            if len(best_scores_history) >= patience:
                recent_best = max(best_scores_history[-patience:])
                older_best = max(best_scores_history[:-patience]) if len(best_scores_history) > patience else -float('inf')
                improvement = recent_best - older_best
                
                if improvement < min_improvement:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        console.print(f"[yellow]Early stopping: No improvement in {patience * 3} trials[/yellow]")
                        study.stop()
                else:
                    no_improvement_count = 0
            
            if study.best_value > 0.25:
                console.print(f"[green]Early stopping: Excellent score achieved ({study.best_value:.4f})[/green]")
                study.stop()
        
        # Run optimization
        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                callbacks=[progress_callback],
                show_progress_bar=True,
                n_jobs=1
            )
        except KeyboardInterrupt:
            console.print("[yellow]Optimization interrupted by user[/yellow]")
        
        # Return best config
        best_config = self.create_optimizable_config(study.best_trial)
        console.print(f"[bold green]Best trial: #{study.best_trial.number} with score {study.best_value:.4f}[/bold green]")
        
        return best_config

    def create_optimizable_config(self, trial: optuna.Trial) -> TradingConfig:
        """Create a TradingConfig with Optuna-suggested parameters"""
        
        config = TradingConfig()
        
        # Risk Management Parameters
        config.risk_per_trade = trial.suggest_float('risk_per_trade', 0.005, 0.03, step=0.0025)
        config.reward_risk_ratio = trial.suggest_float('reward_risk_ratio', 1.5, 3.0, step=0.25)
        config.stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.005, 0.025, step=0.0025)
        config.take_profit_pct = config.stop_loss_pct * config.reward_risk_ratio
        config.max_trade_amount = trial.suggest_int('max_trade_amount', 1000, 5000, step=500)
        
        # Technical Indicator Parameters
        config.short_ma_period = trial.suggest_int('short_ma_period', 3, 10)
        config.long_ma_period = trial.suggest_int('long_ma_period', 15, 35, step=5)
        config.rsi_period = trial.suggest_int('rsi_period', 10, 21)
        config.rsi_oversold = trial.suggest_float('rsi_oversold', 25.0, 45.0, step=2.5)
        config.rsi_overbought = trial.suggest_float('rsi_overbought', 55.0, 75.0, step=2.5)
        config.volume_threshold = trial.suggest_float('volume_threshold', 1.1, 2.0, step=0.1)
        
        # Confidence Thresholds
        config.conf_bull = trial.suggest_float('conf_bull', 0.3, 0.6, step=0.05)
        config.conf_bear = trial.suggest_float('conf_bear', 0.4, 0.8, step=0.05)
        config.conf_sideways = trial.suggest_float('conf_sideways', 0.35, 0.65, step=0.05)
        config.conf_volatile = trial.suggest_float('conf_volatile', 0.5, 0.85, step=0.05)
        
        # Signal Weights
        config.trend_weight = trial.suggest_float('trend_weight', 0.2, 0.6, step=0.05)
        config.rsi_weight = trial.suggest_float('rsi_weight', 0.15, 0.45, step=0.05)
        config.macd_weight = trial.suggest_float('macd_weight', 0.1, 0.4, step=0.05)
        config.volume_weight = trial.suggest_float('volume_weight', 0.05, 0.3, step=0.05)
        config.momentum_weight = trial.suggest_float('momentum_weight', 0.05, 0.25, step=0.05)
        config.dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        
        return config
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        try:
            # Create optimized config
            config = self.create_optimizable_config(trial)
            
            # Load data
            data_specs = [
                DataSpec(
                    symbol="BTC",
                    interval=config.timeframe,
                    ranges=[(config.bull_start, config.bear_end)],
                    collateral=DEFAULT_COLLATERAL
                ),
            ]
            
            df_map = preload_polars_parallel(data_specs, self.cache)
            if not df_map:
                return -1.0
                
            btc_data = list(df_map.values())[0]
            
            # Split data for backtesting (80% train, 20% test)
            split_idx = int(len(btc_data) * 0.8)
            test_data = btc_data.slice(split_idx, len(btc_data) - split_idx)
            
            # Create Backtrader cerebro
            cerebro = bt.Cerebro()
            
            backtest_data = test_data.select([
                "TimestampStart", "Open", "High", "Low", "Close", "Volume"
            ]).drop_nulls().sort("TimestampStart")
            
            data_feed = PolarsData(
                dataname=backtest_data,
                datetime="TimestampStart",
                open="Open",
                high="High", 
                low="Low",
                close="Close",
                volume="Volume",
                openinterest=-1
            )
            cerebro.adddata(data_feed)
            
            # Add strategy with optimized config (silent mode)
            cerebro.addstrategy(CryptoQuantumStrategy, config=config, silent=True)
            
            # Set initial cash and commission
            cerebro.broker.setcash(config.init_cash)
            cerebro.broker.setcommission(commission=config.commission)
            
            # Run backtest
            results = cerebro.run()
            strat = results[0]
            
            # Calculate objective score
            total_return = getattr(strat, 'final_return', -0.5)
            total_trades = getattr(strat, 'final_trades', 0)
            win_rate = getattr(strat, 'final_win_rate', 0)
            
            # Objective: Maximize return while ensuring reasonable trade activity
            if total_trades < 5:
                trade_penalty = -0.2
            elif total_trades < 10:
                trade_penalty = -0.1
            else:
                trade_penalty = 0
                
            if win_rate < 0.3:
                win_rate_penalty = -0.3
            else:
                win_rate_penalty = 0
                
            # Combined score
            objective_score = total_return + trade_penalty + win_rate_penalty + (win_rate * 0.1)
            
            return objective_score
            
        except Exception as e:
            return -1.0

# Add method to load best config without running optimization
def load_best_config_from_mssql(connection_string: str, study_name: str) -> Optional[TradingConfig]:
    """Standalone function to load best config from MSSQL"""
    try:
        storage = build_optuna_storage(connection_string)
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        if len(study.trials) == 0:
            return None
            
        # Create a dummy optimization engine to use the config creation method
        dummy_cache = None  # Not needed for config creation
        engine = OptimizationEngine(dummy_cache)
        best_config = engine.create_optimizable_config(study.best_trial)
        
        console.print(f"[green]Loaded best config from MSSQL: trial #{study.best_trial.number}[/green]")
        return best_config
        
    except Exception as e:
        console.print(f"[red]Failed to load from MSSQL: {e}[/red]")
        return None

# ==================== MAIN EXECUTION PIPELINE ====================

def run_backtest(config: TradingConfig, cache: DataCache, use_models=False, train_models=False, 
                 mssql_connection: str = None, study_name: str = None) -> dict:
    """Run backtest with optional ML model loading or training and MSSQL optimization"""

    # --- Load Data ---
    data_specs = [
        DataSpec(
            symbol="BTC",
            interval=config.timeframe,
            ranges=[(config.test_start, config.test_end)],
            collateral=DEFAULT_COLLATERAL
        ),
    ]
    df_map = preload_polars_parallel(data_specs, cache)
    if not df_map:
        console.print("[red]No data loaded![/red]")
        return {}
    btc_data = list(df_map.values())[0]

    # --- Models ---
    trained_models = {}
    if use_models:
        model_persistence = ModelPersistence(config)

        if train_models:
            # Train all models from scratch
            trainer = ModelTrainer(config)
            split_idx = int(len(btc_data) * 0.6)
            val_split = int(len(btc_data) * 0.8)
            train_data = btc_data.slice(0, split_idx)
            val_data = btc_data.slice(split_idx, val_split - split_idx)
            feature_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
            trained_models = trainer.train_all_models(train_data, val_data, feature_columns)
            
            console.print(f"[green]✅ Trained {len(trained_models)} models successfully[/green]")
            
            # SAVE MODELS IMMEDIATELY after training
            model_persistence.save_all_models(trained_models, config)
            console.print("[green]✅ Models saved successfully before optimization[/green]")
            
            # THEN do optimization with MSSQL storage
            try:
                if mssql_connection and study_name:
                    # Use MSSQL-backed optimization
                    engine = OptimizationEngine(cache, mssql_connection, study_name)
                    optimized_config = engine.optimize_parameters(n_trials=100, resume=True)
                else:
                    # Fallback to in-memory optimization
                    engine = OptimizationEngine(cache)
                    optimized_config = engine.optimize_parameters(n_trials=100)
                
                # Save optimized config
                model_persistence.save_all_models(trained_models, optimized_config)
                config = optimized_config
                
            except KeyboardInterrupt:
                console.print("[yellow]Optimization interrupted, but models are already saved[/yellow]")

        else:
            # Load best pre-trained models + params
            trained_models, saved_params = model_persistence.load_all_models()
            console.print(f"[green]✅ Loaded {len(trained_models)} pre-trained models[/green]")

            # If params exist in save, update config
            if saved_params:
                if isinstance(saved_params, dict):
                    config = TradingConfig(**saved_params)
                    print(config)
                else:
                    config = saved_params

    cerebro = bt.Cerebro()
    backtest_data = btc_data.select([
        "TimestampStart", "Open", "High", "Low", "Close", "Volume"
    ]).drop_nulls().sort("TimestampStart")

    data_feed = PolarsData(
        dataname=backtest_data,
        datetime="TimestampStart",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1
    )
    cerebro.adddata(data_feed)

    cerebro.addstrategy(CryptoQuantumStrategy, config=config, models=trained_models if use_models else None)
    cerebro.broker.setcash(config.init_cash)
    cerebro.broker.setcommission(commission=config.commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    console.print("[cyan]🚀 Running backtest...[/cyan]")
    results = cerebro.run()
    strat = results[0]

    # --- Calculate metrics ---
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - config.init_cash) / config.init_cash

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio') if sharpe_analysis else 0.0
    sharpe_ratio = float(sharpe_ratio) if sharpe_ratio else 0.0

    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0.0)
    max_drawdown = float(max_drawdown) if max_drawdown else 0.0

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'strat': strat,
        'backtest_data': backtest_data
    }

def main():
    """Enhanced main with MSSQL Optuna storage"""
    console.print("[bold green]BTQuant prototyping slaughterhouse...[/bold green]")
    
    try:
        cache = DataCache(CONFIG.cache_dir)
        
        # MSSQL connection for Optuna (optional)
        MSSQL_CONNECTION = CONNECTION
        STUDY_NAME = "cuttingedgeprototyping"
        
        console.print("\n[bold yellow]Execution Modes:[/bold yellow]")
        console.print("[yellow]1. Train + optimize (with MSSQL storage)[/yellow]")
        console.print("[yellow]2. Load pre-trained models + best config[/yellow]")
        console.print("[yellow]3. Resume optimization from MSSQL[/yellow]")
        console.print("[yellow]4. Load best config from MSSQL only[/yellow]")
        console.print("[yellow]5. Plain backtest (no ML)[/yellow]")
        
        choice = input("\nEnter choice (1-5, default=2): ").strip() or "2"
        
        if choice == "1":
            console.print("[cyan]Mode 1: Full training + optimization with MSSQL[/cyan]")
            results = run_backtest(CONFIG, cache, use_models=True, train_models=True, 
                                 mssql_connection=MSSQL_CONNECTION, study_name=STUDY_NAME)
                                 
        elif choice == "2":
            console.print("[cyan]Mode 2: Load pre-trained models + config[/cyan]")
            results = run_backtest(CONFIG, cache, use_models=True, train_models=False)
            
        elif choice == "3":
            console.print("[cyan]Mode 3: Resume optimization from MSSQL[/cyan]")
            if not MSSQL_CONNECTION:
                console.print("[red]MSSQL connection required for this mode[/red]")
                return
            engine = OptimizationEngine(cache, MSSQL_CONNECTION, STUDY_NAME)
            optimized_config = engine.optimize_parameters(n_trials=1000, resume=True)
            results = run_backtest(optimized_config, cache, use_models=False)
            
        elif choice == "4":
            console.print("[cyan]Mode 4: Load best config from MSSQL only[/cyan]")
            if not MSSQL_CONNECTION:
                console.print("[red]MSSQL connection required[/red]")
                return
            best_config = load_best_config_from_mssql(MSSQL_CONNECTION, STUDY_NAME)
            if best_config:
                results = run_backtest(best_config, cache, use_models=False)
            else:
                console.print("[red]No best config found in MSSQL[/red]")
                return
        else:
            console.print("[cyan]Mode 5: Plain backtest[/cyan]")
            results = run_backtest(CONFIG, cache, use_models=False)
        
        if not results:
            console.print("[red]❌ Backtest failed![/red]")
            return

        # 4. Show results table
        table = Table(title="🚀 BTQuant v2025 - RESULTS ✅")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("💰 Initial Capital", f"${CONFIG.init_cash:,.2f} USDT")
        table.add_row("🏆 Final Value", f"${results['final_value']:,.2f} USDT")
        table.add_row("📊 Total Return", f"{results['total_return']:.2%}")
        table.add_row("📈 Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
        table.add_row("📉 Max Drawdown", f"{results['max_drawdown']:.2f}%")
        table.add_row("🔄 Total Trades", f"{results['total_trades']}")
        table.add_row("🎯 Win Rate", f"{results['win_rate']:.1f}%")

        if choice in ["1", "2"]:
            table.add_row("🤖 Models Used", "✅ Enabled")
        else:
            table.add_row("🤖 Models Used", "❌ None")

        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    main()
