
# =============================================================================
# REFINED DIRECT INDICATOR CAPTURE - CLEAN OUTPUT
# =============================================================================
"""
Refined version that captures only the important indicators and filters out 
internal Backtrader noise like LineBuffer, MSSQLData, etc.
"""

import polars as pl
import backtrader as bt
from typing import Dict, List, Any
import os

class RefinedDirectCapture:
    """
    Refined direct capture that filters out noise and captures only 
    meaningful indicator values.
    """

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.all_captured_data: List[Dict] = []
        self.bar_counter = 0

        print("✨ Refined Direct Capture initialized!")
        print("   🎯 Filters out noise, captures only meaningful indicators")

    def capture_strategy_indicators(self, strategy_instance):
        """
        Capture only meaningful indicator values, filtering out internal noise.
        """

        try:
            bar_data = {
                'bar': self.bar_counter,
                'timestamp': self.bar_counter
            }

            captures_this_bar = 0

            # Capture clean market data
            try:
                if hasattr(strategy_instance, 'data') and len(strategy_instance.data) > 0:
                    bar_data['close'] = float(strategy_instance.data.close[0])
                    bar_data['high'] = float(strategy_instance.data.high[0])
                    bar_data['low'] = float(strategy_instance.data.low[0])
                    bar_data['open'] = float(strategy_instance.data.open[0])
                    bar_data['volume'] = float(strategy_instance.data.volume[0])
                    captures_this_bar += 5
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Error capturing market data: {e}")

            # REFINED CAPTURE: Only capture meaningful indicators
            for attr_name in dir(strategy_instance):
                if self._should_capture_attribute(attr_name):
                    try:
                        attr = getattr(strategy_instance, attr_name)

                        # Check if this is a meaningful indicator
                        if self._is_meaningful_indicator(attr, attr_name):
                            try:
                                if len(attr) > 0:
                                    value = attr[0]  # Current value
                                    if self._is_numeric_and_valid(value):
                                        column_name = f"{type(attr).__name__}_{attr_name}"
                                        bar_data[column_name] = float(value)
                                        captures_this_bar += 1

                                        if self.debug and self.bar_counter < 3:
                                            print(f"✨ Clean capture: {column_name} = {value}")
                            except Exception as e:
                                continue

                        # Also check for indicators with lines
                        elif hasattr(attr, 'lines') and self._is_indicator_class(attr):
                            try:
                                # Get main output line
                                if hasattr(attr, '__getitem__') and len(attr) > 0:
                                    value = attr[0]
                                    if self._is_numeric_and_valid(value):
                                        column_name = f"{type(attr).__name__}_{attr_name}"
                                        bar_data[column_name] = float(value)
                                        captures_this_bar += 1

                                        if self.debug and self.bar_counter < 3:
                                            print(f"✨ Clean indicator: {column_name} = {value}")

                                # Also try to get line aliases for detailed capture
                                if hasattr(attr.lines, '_getlinealiases'):
                                    aliases = attr.lines._getlinealiases()
                                    for alias in aliases:
                                        if alias in ['hma', 'zero_lag', 'sine_wma', 'sma', 'momentum', 'atr', 'adx', 'crossover']:
                                            try:
                                                line = getattr(attr.lines, alias)
                                                if hasattr(line, '__getitem__') and len(line) > 0:
                                                    value = line[0]
                                                    if self._is_numeric_and_valid(value):
                                                        column_name = f"{type(attr).__name__}_{alias}"
                                                        bar_data[column_name] = float(value)
                                                        captures_this_bar += 1

                                                        if self.debug and self.bar_counter < 3:
                                                            print(f"✨ Clean line: {column_name} = {value}")
                                            except:
                                                continue

                            except Exception as e:
                                continue

                    except Exception as e:
                        continue

            # Store the clean bar data
            if captures_this_bar > 0:
                self.all_captured_data.append(bar_data)

                if self.debug and self.bar_counter < 5:
                    print(f"✨ Bar {self.bar_counter}: Clean captured {captures_this_bar} values")

            self.bar_counter += 1

        except Exception as e:
            if self.debug:
                print(f"Error in refined capture: {e}")
            self.bar_counter += 1

    def _should_capture_attribute(self, attr_name: str) -> bool:
        """Check if attribute name suggests it's worth capturing"""

        # Skip obvious internal stuff
        skip_prefixes = ['_', 'data0', 'data1', 'datas', 'ddatas', 'line', 'lines']
        skip_names = [
            'array', 'broker', 'cerebro', 'env', 'params', 'plotinfo', 'plotlines',
            'stats', 'observers', 'analyzers', 'sizers', 'writers', 'notifs',
            'position', 'positions', 'positionsbyname', 'dnames', 'orderid'
        ]

        # Skip if starts with internal prefixes
        for prefix in skip_prefixes:
            if attr_name.startswith(prefix):
                return False

        # Skip if in skip list
        if attr_name.lower() in skip_names:
            return False

        # Focus on likely indicator names
        likely_indicators = [
            'zero_lag', 'sine_wma', 'sma', 'momentum', 'atr', 'adx', 'crossover',
            'hma', 'ema', 'wma', 'rsi', 'macd', 'bollinger', 'stoch', 'williams',
            'cci', 'roc', 'trix', 'dema', 'tema', 'kama', 'mama', 'sar', 'hilbert'
        ]

        attr_lower = attr_name.lower()
        for indicator in likely_indicators:
            if indicator in attr_lower:
                return True

        # Also check for numbered indicators (sma_200, ema_50, etc.)
        if any(char.isdigit() for char in attr_name):
            for indicator in ['sma', 'ema', 'wma', 'hma', 'rsi', 'atr', 'adx']:
                if indicator in attr_lower:
                    return True

        return False

    def _is_meaningful_indicator(self, attr, attr_name: str) -> bool:
        """Check if attribute is a meaningful indicator"""

        try:
            # Must have __getitem__ and __len__
            if not (hasattr(attr, '__getitem__') and hasattr(attr, '__len__')):
                return False

            # Must have reasonable length (not empty, not too long)
            try:
                length = len(attr)
                if length == 0 or length > 50000:  # Reasonable bounds
                    return False
            except:
                return False

            # Skip if it's obviously internal data
            type_name = type(attr).__name__
            if type_name in ['LineBuffer', 'MSSQLData', 'array', 'list', 'tuple']:
                return False

            return True

        except:
            return False

    def _is_indicator_class(self, attr) -> bool:
        """Check if attribute is a Backtrader indicator class"""

        try:
            # Check if it's a subclass of Backtrader Indicator
            return isinstance(attr, bt.Indicator)
        except:
            return False

    def _is_numeric_and_valid(self, value: Any) -> bool:
        """Check if value is numeric and valid (not NaN)"""
        try:
            float_val = float(value)
            # Check for NaN or infinite values
            import math
            return not (math.isnan(float_val) or math.isinf(float_val))
        except:
            return False

    def get_dataframe(self) -> pl.DataFrame:
        """Get captured data as DataFrame"""

        if not self.all_captured_data:
            return pl.DataFrame()

        try:
            df = pl.DataFrame(self.all_captured_data)
            return df.sort('bar')
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pl.DataFrame()

    def export_data(self, filename: str = 'refined_indicator_data'):
        """Export captured data"""

        try:
            export_dir = 'refined_indicator_exports'
            os.makedirs(export_dir, exist_ok=True)

            if self.all_captured_data:
                df = self.get_dataframe()

                if len(df) > 0:
                    parquet_path = f"{export_dir}/{filename}.parquet"
                    csv_path = f"{export_dir}/{filename}.csv"

                    df.write_parquet(parquet_path)
                    df.write_csv(csv_path)

                    print(f"\n✅ Refined indicator data exported!")
                    print(f"📊 Parquet: {parquet_path}")
                    print(f"📋 CSV: {csv_path}")
                    print(f"📊 Shape: {df.shape}")

                    return df
            else:
                print("\n⚠️ No data captured by refined method")

            return pl.DataFrame()

        except Exception as e:
            print(f"Error exporting refined data: {e}")
            return pl.DataFrame()

    def print_summary(self):
        """Print summary of captured data"""

        try:
            if not self.all_captured_data:
                print("\n⚠️ No data captured by refined method")
                return

            df = self.get_dataframe()

            print(f"\n{'='*100}")
            print("✨ REFINED DIRECT CAPTURE - CLEAN DATA SUMMARY")
            print(f"{'='*100}")
            print(f"Total Bars Captured: {len(df):,}")
            print(f"Total Columns: {len(df.columns)}")
            print(f"Data Points: {len(df) * len(df.columns):,}")

            # Show columns by category
            cols = df.columns
            market_cols = [c for c in cols if c in ['bar', 'timestamp', 'close', 'high', 'low', 'open', 'volume']]
            indicator_cols = [c for c in cols if c not in market_cols]

            print(f"\n📋 Clean Captured Columns:")
            print(f"  📊 Market Data ({len(market_cols)}): {market_cols}")
            print(f"  📈 Indicators ({len(indicator_cols)}): {indicator_cols}")

            print(f"\n📋 Sample Data (last 5 rows):")
            print(df.tail(5))

            print(f"\n✨ REFINED CAPTURE SUCCESS!")
            print(f"   🎯 Clean data - no internal noise")
            print(f"   📊 {len(df):,} bars with {len(df.columns)} meaningful columns")
            print(f"   ✅ Perfect for ML training!")
            print(f"{'='*100}")

        except Exception as e:
            print(f"Error in refined summary: {e}")

# Create refined capture instance
refined_capture = RefinedDirectCapture(debug=True)

def capture_indicators_refined(strategy_instance):
    """
    Capture indicators with refined filtering.
    Call this from your strategy's next() method.
    """
    refined_capture.capture_strategy_indicators(strategy_instance)

def export_refined_data(filename: str = 'refined_strategy_data'):
    """Export refined captured data"""
    return refined_capture.export_data(filename)

def print_refined_summary():
    """Print refined capture summary"""
    refined_capture.print_summary()

print("✨ REFINED DIRECT CAPTURE READY!")
print("   🎯 Captures only meaningful indicators")
print("   🧹 Filters out internal Backtrader noise") 
print("   📊 Clean output perfect for ML training")
print("   ✅ No broker conflicts - safe operation")
