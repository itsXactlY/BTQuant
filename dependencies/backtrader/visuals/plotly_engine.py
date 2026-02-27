"""
Terrible testing...
"""

from __future__ import annotations

import warnings
import datetime
from typing import Dict, List, Optional, Tuple, Set
import os

import backtrader as bt
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------- Utilities ---------------------------------

def _bt_datetime_to_list(dt_array) -> List:
    """Convert Backtrader datetime numeric array to a list of ISO strings (JSON-safe)."""
    out = []
    try:
        for d in dt_array:
            try:
                dt_obj = bt.num2date(d).replace(tzinfo=None)
                out.append(dt_obj.isoformat())
            except Exception:
                if isinstance(d, (datetime.datetime, datetime.date)):
                    out.append(d.isoformat())
                else:
                    out.append(d)
        return out
    except Exception:
        try:
            return [int(x) for x in dt_array]
        except Exception:
            return list(range(len(dt_array)))

def _safe_array_get(attr) -> np.ndarray:
    """Return a numpy array from a Backtrader line/series attribute if possible"""
    try:
        if hasattr(attr, 'array'):
            return np.asarray(attr.array)
        return np.asarray([float(x) for x in attr])
    except Exception:
        return np.array([])

# ----------------------------- Color Palettes ---------------------------------

class ColorPalette:
    """Bleeding-edge color schemes for modern quant visualization"""

    # Main theme colors
    MATRIX_GREEN = '#00ff41'
    CYBER_RED = '#ff0040'
    NEON_BLUE = '#00d4ff'
    PURPLE_HAZE = '#b24bf3'
    GOLD = '#ffd700'
    ORANGE = '#ff8c00'

    # Indicator colors - carefully selected for visibility
    OVERLAY_COLORS = [
        '#00d4ff',  # Neon blue
        '#ff00ff',  # Magenta
        '#00ffff',  # Cyan
        '#ffff00',  # Yellow
        '#ff8c00',  # Orange
        '#00ff7f',  # Spring green
    ]

    OSCILLATOR_COLORS = [
        '#b24bf3',  # Purple
        '#ff1493',  # Deep pink
        '#00ced1',  # Dark turquoise
        '#ffa500',  # Orange
        '#7fff00',  # Chartreuse
    ]

    VOLATILITY_COLORS = ['#ff6b6b', '#ff8c42', '#ffa07a']
    VOLUME_COLOR = '#4ecdc4'

# ----------------------------- Indicator Categorization ---------------------------------

class IndicatorCategory:
    """Smart categorization of technical indicators for optimal subplot placement"""

    # Indicators that should overlay on price chart
    OVERLAY = {
        'sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'hma', 'alma', 'zlema',  # MAs
        'bb', 'bollinger', 'keltner', 'donchian', 'envelope', 'channel',  # Bands
        'vwap', 'pivots', 'pivot', 'sar', 'parabolic',  # Price-based
        'supertrend', 'chandelier', 'atr_stop', 'ichimoku'
    }

    # Momentum oscillators (0-100 scale or bounded)
    OSCILLATOR = {
        'rsi', 'stoch', 'stochastic', 'cci', 'williams', 'willr',
        'mfi', 'tsi', 'roc', 'ultimateoscillator', 'cmo', 'ppo_hist'
    }

    # MACD-like indicators (separate subplot, typically dual/triple lines)
    MACD_LIKE = {
        'macd', 'ppo', 'trix', 'awesome'
    }

    # Volume-based indicators
    VOLUME_BASED = {
        'obv', 'ad', 'adosc', 'cmf', 'pvt', 'eom', 'force', 'vpt'
    }

    # Volatility indicators
    VOLATILITY = {
        'atr', 'natr', 'true_range', 'tr', 'keltner_width', 
        'bollinger_width', 'bbw', 'stddev', 'variance', 'choppiness'
    }

    @classmethod
    def categorize(cls, indicator_name: str) -> str:
        """Categorize an indicator based on its name"""
        name_lower = indicator_name.lower()

        # Check for multi-line indicators (e.g., bb.top, bb.mid, bb.bot)
        base_name = name_lower.split('.')[0] if '.' in name_lower else name_lower

        # Check each category
        if any(pattern in base_name for pattern in cls.OVERLAY):
            return 'overlay'

        if any(pattern in base_name for pattern in cls.OSCILLATOR):
            return 'oscillator'

        if any(pattern in base_name for pattern in cls.MACD_LIKE):
            return 'macd'

        if any(pattern in base_name for pattern in cls.VOLUME_BASED):
            return 'volume_ind'

        if any(pattern in base_name for pattern in cls.VOLATILITY):
            return 'volatility'

        # Default: separate subplot
        return 'separate'

# --------------------------- Core Plotly Engine ---------------------------

class PlotlyEngine:
    """Professional Plotly engine for Backtrader with advanced quant features."""

    def __init__(
        self,
        cerebro: bt.Cerebro,
        title: str = "BTQuant Backtest",
        theme: str = "plotly_dark",
        show_grid: bool = True,
        show_rangeslider: bool = False,
        width: int = 1920,
        height: int = 1080,
        smart_layout: bool = True  # NEW: Enable smart categorization
    ):
        self.cerebro = cerebro
        self.title = title
        self.theme = theme
        self.show_grid = show_grid
        self.show_rangeslider = show_rangeslider
        self.width = width
        self.height = height
        self.smart_layout = smart_layout  # NEW

        # Data containers
        self.dataframes: Dict[str, pl.DataFrame] = {}
        self.indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.observers: Dict[str, Tuple[List, np.ndarray]] = {}
        self.analyzers: Dict[str, Tuple[List, List]] = {}
        self.trades: Optional[pl.DataFrame] = None
        self.equity: Optional[Tuple[List, List]] = None
        self.drawdown: Optional[Tuple[List, List]] = None

        # NEW: Categorized indicators
        self.overlay_indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.oscillator_indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.macd_indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.volume_indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.volatility_indicators: Dict[str, Tuple[List, np.ndarray]] = {}
        self.separate_indicators: Dict[str, Tuple[List, np.ndarray]] = {}

        self.fig: Optional[go.Figure] = None
        self.row_order: List[str] = []

    # -------------------- Data extraction --------------------

    def _df_from_btdata(self, data: bt.feeds.PolarsData) -> pl.DataFrame:
        """Extract OHLCV and datetime list from a Backtrader data feed into a polars DataFrame."""
        o = _safe_array_get(getattr(data, 'open', []))
        h = _safe_array_get(getattr(data, 'high', []))
        l = _safe_array_get(getattr(data, 'low', []))
        c = _safe_array_get(getattr(data, 'close', []))
        v = _safe_array_get(getattr(data, 'volume', []))
        dt = _safe_array_get(getattr(data, 'datetime', []))
        dt_list = _bt_datetime_to_list(dt)

        maxlen = max(len(o), len(h), len(l), len(c), len(v), len(dt_list))

        def _pad(arr):
            if len(arr) == maxlen:
                return arr
            if len(arr) == 0:
                return np.full(maxlen, np.nan, dtype=float)
            return np.pad(arr, (maxlen - len(arr), 0), 'constant', constant_values=(np.nan,))

        o = _pad(o); h = _pad(h); l = _pad(l); c = _pad(c); v = _pad(v);

        if len(dt_list) != maxlen:
            dt_list = list(range(maxlen))

        df = pl.DataFrame({
            'datetime': dt_list,
            'open': o.tolist(),
            'high': h.tolist(),
            'low': l.tolist(),
            'close': c.tolist(),
            'volume': v.tolist(),
        })

        return df

    def collect_datafeeds(self):
        """Collect all datafeeds into polars DataFrames."""
        for i, data in enumerate(self.cerebro.datas):
            name = getattr(data, '_name', None) or f"Data_{i}"
            try:
                df = self._df_from_btdata(data)
                self.dataframes[name] = df
            except Exception as e:
                warnings.warn(f"Failed to convert datafeed {name}: {e}")

    def collect_indicators_and_observers(self):
        """Collect all indicator lines and observer lines from the first strategy instance."""
        try:
            strat = self.cerebro.runstrats[0][0]
        except Exception:
            strat = None

        if strat is None:
            return

        # Collect indicators
        for attr_name, attr in strat.__dict__.items():
            if attr_name.startswith('_'):
                continue

            try:
                if hasattr(attr, 'lines'):
                    try:
                        for ln_name, ln in attr.lines._getlinealiases().items():
                            arr = _safe_array_get(ln)
                            if arr.size > 0:
                                idx = self._dt_from_stratdata(strat)
                                self.indicators[f"{attr_name}.{ln_name}"] = (idx, arr)
                    except Exception:
                        arr = _safe_array_get(attr)
                        if arr.size > 0:
                            idx = self._dt_from_stratdata(strat)
                            self.indicators[attr_name] = (idx, arr)
            except Exception:
                continue

        # Collect observers
        try:
            for obs in getattr(strat, 'observers', []) or []:
                obs_name = obs.__class__.__name__
                try:
                    for ln_name, ln in obs.lines._getlinealiases().items():
                        arr = _safe_array_get(ln)
                        if arr.size > 0:
                            idx = self._dt_from_stratdata(strat)
                            self.observers[f"{obs_name}.{ln_name}"] = (idx, arr)
                except Exception:
                    continue
        except Exception:
            pass

    def _dt_from_stratdata(self, strat) -> List:
        """Best-effort datetime list from strategy's main data feed (returns ISO strings)."""
        try:
            data = strat.data
            dt_arr = _safe_array_get(getattr(data, 'datetime', []))
            idx = _bt_datetime_to_list(dt_arr)
            return idx
        except Exception:
            if self.dataframes:
                first = next(iter(self.dataframes.values()))
                return first['datetime'].to_list()
            return [0]

    def collect_analyzers(self):
        """Serialize analyzers that expose numeric series or dicts to polars-friendly structures."""
        try:
            strat = self.cerebro.runstrats[0][0]
        except Exception:
            strat = None

        if strat is None:
            return

        def _serialize_key(k):
            """Convert any key type to a JSON-serializable format."""
            if isinstance(k, (datetime.datetime, datetime.date)):
                return k.isoformat()
            elif isinstance(k, (int, float, str, bool, type(None))):
                return k
            else:
                return str(k)

        try:
            for name, analyzer in strat.analyzers.getitems():
                try:
                    analysis = analyzer.get_analysis()
                    if isinstance(analysis, dict):
                        keys = list(analysis.keys())
                        keys_serializable = [_serialize_key(k) for k in keys]
                        vals = [analysis[k] for k in keys]
                        self.analyzers[name] = (keys_serializable, vals)
                    elif isinstance(analysis, list):
                        self.analyzers[name] = (list(range(len(analysis))), analysis)
                    else:
                        self.analyzers[name] = ([0], [analysis])
                except Exception:
                    continue
        except Exception:
            pass

    def collect_trades_and_equity(self):
        """Collect trade events and equity curve from the strategy and analyzers."""
        try:
            strat = self.cerebro.runstrats[0][0]
        except Exception:
            strat = None

        if strat is None:
            return

        # Collect trades
        trades = []
        if hasattr(strat, 'trade_history'):
            try:
                for t in strat.trade_history:
                    trades.append(t)
            except Exception:
                pass

        if hasattr(strat, 'trades'):
            try:
                for k, v in getattr(strat, 'trades').items():
                    trades.append(v)
            except Exception:
                pass

        rows = []
        for t in trades:
            if isinstance(t, dict):
                rows.append(t)
            else:
                try:
                    dt = getattr(t, 'dt', None) or getattr(t, 'datetime', None) or getattr(t, 'executed', None)
                    price = getattr(t, 'price', None) or getattr(t, 'executed', None)
                    side = 'buy' if getattr(t, 'side', None) == 'buy' else ('sell' if getattr(t, 'side', None) == 'sell' else None)
                    rows.append({'dt': dt, 'price': price, 'side': side})
                except Exception:
                    continue

        if rows:
            try:
                normalized = []
                for r in rows:
                    dt = r.get('dt')
                    if isinstance(dt, (int, float)):
                        try:
                            dt = bt.num2date(dt).replace(tzinfo=None).isoformat()
                        except Exception:
                            pass
                    elif isinstance(dt, (datetime.datetime, datetime.date)):
                        dt = dt.isoformat()
                    normalized.append({'dt': dt, 'price': r.get('price'), 'side': r.get('side')})

                self.trades = pl.DataFrame(normalized)
            except Exception:
                self.trades = None

        # Collect equity
        try:
            if 'returns' in getattr(strat, 'analyzers', {}):
                r = strat.analyzers.returns.get_analysis()
                if isinstance(r, dict):
                    try:
                        keys = list(r.keys())
                        keys_serialized = []
                        for k in keys:
                            if isinstance(k, (datetime.datetime, datetime.date)):
                                keys_serialized.append(k.isoformat())
                            else:
                                keys_serialized.append(k)
                        vals = [r[k] for k in keys]
                        self.equity = (keys_serialized, np.cumsum(vals).tolist())
                    except Exception:
                        pass
        except Exception:
            pass

    def collect(self):
        """Run all collection steps."""
        self.collect_datafeeds()
        self.collect_indicators_and_observers()
        self.collect_analyzers()
        self.collect_trades_and_equity()

        # NEW: Categorize indicators if smart_layout is enabled
        if self.smart_layout:
            self._categorize_indicators()

    # NEW: Categorization method
    def _categorize_indicators(self):
        """Smart categorization of indicators for optimal subplot placement"""
        for name, (x_data, arr) in self.indicators.items():
            category = IndicatorCategory.categorize(name)

            if category == 'overlay':
                self.overlay_indicators[name] = (x_data, arr)
            elif category == 'oscillator':
                self.oscillator_indicators[name] = (x_data, arr)
            elif category == 'macd':
                self.macd_indicators[name] = (x_data, arr)
            elif category == 'volume_ind':
                self.volume_indicators[name] = (x_data, arr)
            elif category == 'volatility':
                self.volatility_indicators[name] = (x_data, arr)
            else:
                self.separate_indicators[name] = (x_data, arr)

    # -------------------- Layout & plotting --------------------

    def build_layout(self, show_volume: bool = True, group_indicators: bool = False):
        """Define subplot layout intelligently based on collected content."""
        self.row_order = []
        self.row_order.append('price')

        # Add volume if present
        if show_volume and any(
            'volume' in df.columns and len(df['volume'].drop_nulls()) > 0
            for df in self.dataframes.values()
        ):
            self.row_order.append('volume')

        # NEW: Smart layout based on categorized indicators
        if self.smart_layout:
            # Group oscillators together
            if self.oscillator_indicators:
                self.row_order.append('oscillator')

            # MACD gets its own subplot
            if self.macd_indicators:
                self.row_order.append('macd')

            # Volume-based indicators
            if self.volume_indicators:
                self.row_order.append('volume_ind')

            # Volatility indicators
            if self.volatility_indicators:
                self.row_order.append('volatility')

            # Separate indicators that don't fit categories
            for name in self.separate_indicators.keys():
                self.row_order.append(name)
        else:
            # Old behavior: separate subplot per indicator or grouped
            if group_indicators and self.indicators:
                self.row_order.append('indicators')
            else:
                for name in self.indicators.keys():
                    if name not in self.overlay_indicators:
                        self.row_order.append(name)

        # Add observers
        for name in self.observers.keys():
            self.row_order.append(name)

        # Add analyzers and equity
        if self.analyzers:
            self.row_order.append('analyzers')

        if self.equity is not None or self.drawdown is not None:
            self.row_order.append('equity')

        # Compute optimal heights
        heights = self._compute_row_heights(len(self.row_order))

        # Create subplots
        self.fig = make_subplots(
            rows=len(self.row_order),
            cols=1,
            shared_xaxes=True,
            row_heights=heights,
            vertical_spacing=0.01,
            subplot_titles=[self._format_subplot_title(row) for row in self.row_order]
        )

        # Apply theme and styling
        self.fig.update_layout(
            template=self.theme,
            title={
                'text': self.title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#00ff41', 'family': 'Courier New, monospace'}
            },
            width=self.width,
            height=self.height,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#00ff41",
                borderwidth=1,
                font=dict(size=10, family='Courier New, monospace')
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.8)",
                font_size=12,
                font_family="Courier New, monospace",
                bordercolor="#00ff41"
            ),
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
        )

    def _format_subplot_title(self, row_name: str) -> str:
        """Format subplot titles in quant style."""
        title_map = {
            'price': '📈 PRICE ACTION',
            'volume': '📊 VOLUME PROFILE',
            'oscillator': '🎯 MOMENTUM OSCILLATORS',
            'macd': '📉 MACD',
            'volatility': '💫 VOLATILITY',
            'volume_ind': '📊 VOLUME INDICATORS',
            'indicators': '🎯 TECHNICAL INDICATORS',
            'analyzers': '📉 PERFORMANCE METRICS',
            'equity': '💰 EQUITY CURVE'
        }

        return title_map.get(row_name, f'📌 {row_name.upper()}')

    def _compute_row_heights(self, total: int) -> List[float]:
        """Compute normalized row heights for all subplots with optimal allocation."""
        if total <= 1:
            return [1.0]

        heights = []
        for row_name in self.row_order:
            if row_name == 'price':
                heights.append(5.0)  # 50% for price
            elif row_name == 'volume':
                heights.append(1.0)  # 10% for volume
            elif row_name == 'oscillator':
                heights.append(1.5)  # 15% for oscillators (grouped)
            elif row_name in ['macd', 'volatility', 'volume_ind']:
                heights.append(1.25)  # 12.5% each
            else:
                heights.append(1.0)  # Default

        # Normalize
        total_sum = sum(heights)
        return [h / total_sum for h in heights]

    def _nuclear_datetime_fix(self):
        """Nuclear option: convert ALL datetime objects in the figure to ISO strings."""
        def convert_datetimes(obj):
            """Recursively convert all datetime objects to ISO strings."""
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {convert_datetimes(k): convert_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_datetimes(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_datetimes(obj.tolist())
            else:
                return obj

        try:
            fig_dict = self.fig.to_dict()
            fig_dict_clean = convert_datetimes(fig_dict)
            self.fig = go.Figure(fig_dict_clean)
        except Exception as e:
            warnings.warn(f"Nuclear datetime fix failed: {e}")

    def plot(self, show: bool = True, save_html: Optional[str] = None):
        """Render everything into the Plotly Figure with professional quant styling."""
        if not self.fig:
            self.build_layout()

        row_map = {name: idx + 1 for idx, name in enumerate(self.row_order)}

        # 🔥 PRICE ACTION + OVERLAY INDICATORS
        self._plot_price_and_overlays(row_map)

        # 📊 VOLUME
        if 'volume' in self.row_order:
            row = row_map['volume']
            for df in self.dataframes.values():
                if 'volume' in df.columns:
                    colors = ['#00ff41' if c >= o else '#ff0040'
                             for c, o in zip(df['close'].to_list(), df['open'].to_list())]

                    self.fig.add_trace(go.Bar(
                        x=df['datetime'].to_list(),
                        y=df['volume'].to_list(),
                        name='Volume',
                        marker=dict(
                            color=colors,
                            opacity=0.5,
                            line=dict(width=0)
                        ),
                        showlegend=False
                    ), row=row, col=1)

                    self.fig.update_yaxes(
                        showgrid=False,
                        side='right',
                        row=row, col=1
                    )
                    break

        # 🎯 OSCILLATORS (GROUPED)
        if 'oscillator' in row_map:
            self._plot_oscillators(row_map['oscillator'])

        # 📉 MACD
        if 'macd' in row_map:
            self._plot_macd(row_map['macd'])

        # 💫 VOLATILITY
        if 'volatility' in row_map:
            self._plot_volatility(row_map['volatility'])

        # 📊 VOLUME INDICATORS
        if 'volume_ind' in row_map:
            self._plot_volume_indicators(row_map['volume_ind'])

        # OTHER INDICATORS (old style for backward compatibility)
        if 'indicators' in self.row_order:
            row = row_map['indicators']
            for idx, (name, (x_data, arr)) in enumerate(self.indicators.items()):
                color = ColorPalette.OVERLAY_COLORS[idx % len(ColorPalette.OVERLAY_COLORS)]
                self.fig.add_trace(go.Scatter(
                    x=x_data,
                    y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=1.5),
                    opacity=0.8
                ), row=row, col=1)
        else:
            # Separate subplots for uncategorized indicators
            for name in self.separate_indicators.keys():
                if name in row_map:
                    row = row_map[name]
                    x_data, arr = self.separate_indicators[name]
                    self.fig.add_trace(go.Scatter(
                        x=x_data,
                        y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                        mode='lines',
                        name=name,
                        line=dict(color='#00d4ff', width=2),
                    ), row=row, col=1)

        # 👁️ OBSERVERS
        for idx, (name, (x_data, arr)) in enumerate(self.observers.items()):
            row = row_map.get(name)
            if row:
                color = ColorPalette.OVERLAY_COLORS[idx % len(ColorPalette.OVERLAY_COLORS)]
                self.fig.add_trace(go.Scatter(
                    x=x_data,
                    y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                    mode='lines',
                    name=f'Observer: {name}',
                    line=dict(color=color, width=1.5, dash='dot'),
                ), row=row, col=1)

        # 📉 ANALYZERS
        if 'analyzers' in self.row_order and self.analyzers:
            row = row_map['analyzers']
            for idx, (name, (x, vals)) in enumerate(self.analyzers.items()):
                color = ColorPalette.OVERLAY_COLORS[idx % len(ColorPalette.OVERLAY_COLORS)]
                self.fig.add_trace(go.Scatter(
                    x=x,
                    y=vals,
                    mode='lines+markers',
                    name=f'{name}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ), row=row, col=1)

        # 💰 EQUITY & DRAWDOWN
        if 'equity' in self.row_order:
            row = row_map['equity']
            if self.equity is not None:
                x, vals = self.equity
                self.fig.add_trace(go.Scatter(
                    x=x,
                    y=vals,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00ff41', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 65, 0.1)'
                ), row=row, col=1)

            if self.drawdown is not None:
                x, vals = self.drawdown
                self.fig.add_trace(go.Scatter(
                    x=x,
                    y=vals,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff0040', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 64, 0.2)'
                ), row=row, col=1)

        # 🎯 TRADES - Enhanced markers
        if isinstance(self.trades, pl.DataFrame) and not self.trades.is_empty():
            df = self.trades
            buys = df.filter(pl.col('side') == 'buy') if 'side' in df.columns else df
            sells = df.filter(pl.col('side') == 'sell') if 'side' in df.columns else pl.DataFrame()

            price_col = 'price' if 'price' in df.columns else 'close'
            prow = row_map.get('price', 1)

            if not buys.is_empty():
                self.fig.add_trace(go.Scatter(
                    x=buys['dt'].to_list(),
                    y=buys[price_col].to_list(),
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        color='#00ff41',
                        size=15,  # Larger for visibility
                        line=dict(color='#ffffff', width=2)
                    ),
                    name='🟢 BUY',
                    showlegend=True
                ), row=prow, col=1)

            if not sells.is_empty():
                self.fig.add_trace(go.Scatter(
                    x=sells['dt'].to_list(),
                    y=sells[price_col].to_list(),
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        color='#ff0040',
                        size=15,  # Larger for visibility
                        line=dict(color='#ffffff', width=2)
                    ),
                    name='🔴 SELL',
                    showlegend=True
                ), row=prow, col=1)

        # Apply grid styling to all subplots
        for i in range(1, len(self.row_order) + 1):
            self.fig.update_xaxes(
                showgrid=self.show_grid,
                gridwidth=0.5,
                gridcolor='rgba(0, 255, 65, 0.05)',
                row=i, col=1
            )

            self.fig.update_yaxes(
                showgrid=self.show_grid,
                gridwidth=0.5,
                gridcolor='rgba(0, 255, 65, 0.05)',
                side='right',
                row=i, col=1
            )

        # 🔥 Nuclear datetime fix
        self._nuclear_datetime_fix()

        # 💾 Save and open
        if show or save_html:
            import webbrowser
            from datetime import datetime as dt

            if save_html:
                output_path = save_html
            else:
                timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'backtest_{timestamp}.html'

            output_path = os.path.abspath(output_path)
            self.fig.write_html(output_path, include_plotlyjs='cdn')

            print(f"\n{'='*60}")
            print(f"🚀 BACKTEST PLOT GENERATED (BLEEDING EDGE)")
            print(f"{'='*60}")
            print(f"📊 File: {output_path}")
            print(f"📈 Size: {os.path.getsize(output_path) / 1024:.2f} KB")
            print(f"✨ Smart layout: {self.smart_layout}")
            print(f"{'='*60}\n")

            if show:
                webbrowser.open('file://' + output_path)
                print(f"🌐 Opening in browser...")

    # NEW: Enhanced plotting methods
    def _plot_price_and_overlays(self, row_map: Dict[str, int]):
        """Plot price candlesticks and overlay indicators"""
        row = row_map.get('price', 1)

        # Plot candlesticks
        for name, df in self.dataframes.items():
            x = df['datetime'].to_list()

            # Professional candlestick colors
            self.fig.add_trace(go.Candlestick(
                x=x,
                open=df['open'].to_list(),
                high=df['high'].to_list(),
                low=df['low'].to_list(),
                close=df['close'].to_list(),
                name=name,
                increasing_line_color='#00ff41',  # Matrix green
                decreasing_line_color='#ff0040',  # Cyber red
                increasing_fillcolor='rgba(0, 255, 65, 0.3)',
                decreasing_fillcolor='rgba(255, 0, 64, 0.3)',
                line=dict(width=1),
            ), row=row, col=1)

        # Plot overlay indicators (MAs, BBs, etc.)
        for idx, (name, (x_data, arr)) in enumerate(self.overlay_indicators.items()):
            color = ColorPalette.OVERLAY_COLORS[idx % len(ColorPalette.OVERLAY_COLORS)]

            # Check if it's a Bollinger Band line
            if 'bb' in name.lower() or 'bollinger' in name.lower():
                if 'top' in name.lower() or 'upper' in name.lower():
                    line_style = dict(color=color, width=1, dash='dot')
                elif 'bot' in name.lower() or 'lower' in name.lower():
                    line_style = dict(color=color, width=1, dash='dot')
                else:
                    line_style = dict(color=color, width=1.5)
            else:
                line_style = dict(color=color, width=2)

            self.fig.add_trace(go.Scatter(
                x=x_data,
                y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                mode='lines',
                name=name,
                line=line_style,
            ), row=row, col=1)

        # Style the price subplot
        self.fig.update_xaxes(
            showgrid=self.show_grid,
            gridwidth=0.5,
            gridcolor='rgba(0, 255, 65, 0.1)',
            row=row, col=1
        )

        self.fig.update_yaxes(
            showgrid=self.show_grid,
            gridwidth=0.5,
            gridcolor='rgba(0, 255, 65, 0.1)',
            side='right',
            row=row, col=1
        )

    def _plot_oscillators(self, row: int):
        """Plot all oscillators in one subplot with reference lines"""
        for idx, (name, (x_data, arr)) in enumerate(self.oscillator_indicators.items()):
            color = ColorPalette.OSCILLATOR_COLORS[idx % len(ColorPalette.OSCILLATOR_COLORS)]

            self.fig.add_trace(go.Scatter(
                x=x_data,
                y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
            ), row=row, col=1)

        # Add reference lines for RSI-type indicators (70/30)
        if any('rsi' in name.lower() for name in self.oscillator_indicators.keys()):
            x_range = list(self.oscillator_indicators.values())[0][0]

            # Overbought (70)
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[70, 70],
                mode='lines', line=dict(color='#ff0040', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

            # Oversold (30)
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[30, 30],
                mode='lines', line=dict(color='#00ff41', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

            # Neutral (50)
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[50, 50],
                mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=0.5, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

        # Add reference lines for Stochastic-type indicators (80/20)
        if any('stoch' in name.lower() for name in self.oscillator_indicators.keys()):
            x_range = list(self.oscillator_indicators.values())[0][0]

            # Overbought (80)
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[80, 80],
                mode='lines', line=dict(color='#ff0040', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

            # Oversold (20)
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[20, 20],
                mode='lines', line=dict(color='#00ff41', width=1, dash='dot'),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

        self.fig.update_yaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(0, 255, 65, 0.05)', 
            side='right', 
            row=row, col=1
        )

    def _plot_macd(self, row: int):
        """Plot MACD with color-coded histogram"""
        for name, (x_data, arr) in self.macd_indicators.items():
            if 'hist' in name.lower() or 'histogram' in name.lower():
                # Color-coded histogram
                colors = ['#00ff41' if val >= 0 else '#ff0040' for val in arr]
                self.fig.add_trace(go.Bar(
                    x=x_data,
                    y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                    name=name,
                    marker=dict(color=colors, opacity=0.6),
                ), row=row, col=1)
            else:
                # Line traces
                if 'signal' in name.lower():
                    color = '#ff00ff'  # Magenta for signal
                else:
                    color = '#00d4ff'  # Neon blue for MACD line

                self.fig.add_trace(go.Scatter(
                    x=x_data,
                    y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                ), row=row, col=1)

        # Zero line
        if self.macd_indicators:
            x_range = list(self.macd_indicators.values())[0][0]
            self.fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[-1]], y=[0, 0],
                mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=0.5),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

        self.fig.update_yaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(0, 255, 65, 0.05)', 
            side='right', 
            row=row, col=1
        )

    def _plot_volatility(self, row: int):
        """Plot volatility indicators with area fill"""
        for idx, (name, (x_data, arr)) in enumerate(self.volatility_indicators.items()):
            color = ColorPalette.VOLATILITY_COLORS[idx % len(ColorPalette.VOLATILITY_COLORS)]

            # Convert hex to RGB for alpha
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)

            self.fig.add_trace(go.Scatter(
                x=x_data,
                y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba({r}, {g}, {b}, 0.1)'
            ), row=row, col=1)

        self.fig.update_yaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(0, 255, 65, 0.05)', 
            side='right', 
            row=row, col=1
        )

    def _plot_volume_indicators(self, row: int):
        """Plot volume-based indicators"""
        for idx, (name, (x_data, arr)) in enumerate(self.volume_indicators.items()):
            color = ColorPalette.OVERLAY_COLORS[idx % len(ColorPalette.OVERLAY_COLORS)]

            self.fig.add_trace(go.Scatter(
                x=x_data,
                y=arr.tolist() if hasattr(arr, 'tolist') else list(arr),
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
            ), row=row, col=1)

        self.fig.update_yaxes(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(0, 255, 65, 0.05)', 
            side='right', 
            row=row, col=1
        )

    def save_html(self, path: str):
        if self.fig is None:
            raise RuntimeError("Figure not built yet. Call .plot() or .build_layout() first.")
        self.fig.write_html(path, include_plotlyjs='cdn')

# ----------------------- Patching Cerebro.plot ---------------------------

def plotly_plot(self: bt.Cerebro, *args, **kwargs) -> PlotlyEngine:
    """Replacement for bt.Cerebro.plot that uses PlotlyEngine (polars-only)."""
    engine = PlotlyEngine(
        self,
        title=kwargs.get('title', 'BTQuant Backtest'),
        theme=kwargs.get('theme', 'plotly_dark'),
        width=kwargs.get('width', 1920),
        height=kwargs.get('height', 1080),
        smart_layout=kwargs.get('smart_layout', True)  # NEW: Enable by default
    )

    engine.collect()
    engine.build_layout(
        show_volume=kwargs.get('volume', True),
        group_indicators=kwargs.get('group_indicators', False)
    )

    save_html = kwargs.get('save_html', None)
    show = kwargs.get('show', True)
    engine.plot(show=show, save_html=save_html)

    return engine

def activate_plotly_engine():
    """Activate the Plotly engine by patching bt.Cerebro.plot."""
    bt.Cerebro.plot = plotly_plot

    try:
        bt.plot = lambda *a, **k: warnings.warn('bt.plot disabled; use Cerebro.plot() patched to Plotly')
    except Exception:
        pass
