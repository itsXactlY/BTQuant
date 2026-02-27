import backtrader as bt
import polars as pl
from typing import Dict, List, Any
import threading

class TransparencyPatch():
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.debug = False
        self.all_captured_data: List[Dict] = []
        self.bar_counter = 0
        self.indicator_registry: Dict[int, Dict[str, Any]] = {}
        self.batch_size = 100
        self.current_batch = []
        self.patch_applied = False
        self.captured_object_ids: Dict[int, str] = {}
        self.strategy_ref = None
        
        # print("Transparency Patch initialized")

    def apply_indicator_patch(self):
        if self.patch_applied:
            return
        
        if not hasattr(bt.Indicator, '_original_setattr'):
            bt.Indicator._original_setattr = bt.Indicator.__setattr__
        
        def patched_setattr(self, name, value):
            self._original_setattr(name, value)
            
            if (not name.startswith('_') and 
                name not in ['lines', 'params', 'plotinfo', 'plotlines', 'datas', 
                             'data', 'p', 'l', 'o', 'owner', 'close', 'open', 
                             'high', 'low', 'volume', 'openinterest', 'datetime']):
                
                if hasattr(value, '__getitem__') and hasattr(value, '__len__'):
                    instance_id = id(self)
                    patch_instance = TransparencyPatch()

                    if instance_id not in patch_instance.indicator_registry:
                        patch_instance.indicator_registry[instance_id] = {}

                    indicator_name = self.__class__.__name__
                    full_name = f"{indicator_name}_{name}"

                    obj_id = id(value)
                    
                    if obj_id in patch_instance.captured_object_ids:
                        if patch_instance.debug:
                            original_name = patch_instance.captured_object_ids[obj_id]
                            print(f"  Skipping duplicate: {full_name} (same as {original_name})")
                        return
                    
                    patch_instance.captured_object_ids[obj_id] = full_name
                    
                    patch_instance.indicator_registry[instance_id][name] = {
                        'object': value,
                        'full_name': full_name,
                        'indicator_name': indicator_name,
                        'obj_id': obj_id
                    }

                    if patch_instance.debug:
                        print(f"  Registered: {full_name}")
        
        bt.Indicator.__setattr__ = patched_setattr
        self.patch_applied = True
        
        print("Indicator patch applied - will capture intermediate assignments")

    def capture_patch_fast(self, strategy):
        try:
            if self.strategy_ref is None:
                self.strategy_ref = strategy
            
            bar_data = {'bar': self.bar_counter}
            
            try:
                data = strategy.data
                if len(data) > 0:
                    bar_data['open'] = float(data.open[0])
                    bar_data['high'] = float(data.high[0])
                    bar_data['low'] = float(data.low[0])
                    bar_data['close'] = float(data.close[0])
                    bar_data['volume'] = float(data.volume[0])

                    try:
                        dt = data.datetime.datetime(0)
                        bar_data['datetime'] = dt.isoformat()
                    except:
                        pass
            except Exception as e:
                if self.debug:
                    print(f"  Warning: Could not capture OHLCV: {e}")

            value_to_name = {}

            for instance_id, intermediates in self.indicator_registry.items():
                for var_name, info in intermediates.items():
                    obj = info['object']
                    full_name = info['full_name']
                    numeric_val = self._extract_numeric(obj)
                    
                    if numeric_val is not None:
                        value_key = f"{numeric_val:.10f}"
                        
                        if value_key in value_to_name:
                            if self.debug and self.bar_counter == 0:
                                print(f"  Skipping runtime duplicate: {full_name} = {numeric_val} (same as {value_to_name[value_key]})")
                            continue
                        
                        bar_data[full_name] = numeric_val
                        value_to_name[value_key] = full_name
            
            if self.debug and self.bar_counter == 0:
                print(f"\nBar 0: Captured OHLCV + {len(bar_data)-7} unique indicator values")
                print("OHLCV columns:", ', '.join([k for k in ['open', 'high', 'low', 'close', 'volume', 'datetime'] if k in bar_data]))
                print("Indicator variables:", ', '.join(sorted([k for k in bar_data.keys() if k not in ['bar', 'open', 'high', 'low', 'close', 'volume', 'datetime']])))
            
            self.current_batch.append(bar_data)
            
            if len(self.current_batch) >= self.batch_size:
                self.all_captured_data.extend(self.current_batch)
                self.current_batch = []
            
            self.bar_counter += 1
            
        except Exception as e:
            if self.debug:
                print(f"Error in capture: {e}")
            self.bar_counter += 1

    def _extract_numeric(self, var_value) -> float:
        try:
            if isinstance(var_value, (int, float)):
                if not (var_value != var_value or abs(var_value) == float('inf')):
                    return float(var_value)
                return None
            
            if hasattr(var_value, '__getitem__') and hasattr(var_value, '__len__'):
                try:
                    if len(var_value) > 0:
                        val = var_value[0]
                        if isinstance(val, (int, float)):
                            if not (val != val or abs(val) == float('inf')):
                                return float(val)
                except:
                    pass
            
            return None
        except:
            return None

    def flush_remaining_batch(self):
        if self.current_batch:
            self.all_captured_data.extend(self.current_batch)
            self.current_batch = []

    def get_dataframe(self) -> pl.DataFrame:
        self.flush_remaining_batch()
        
        if not self.all_captured_data:
            return pl.DataFrame()

        try:
            df = pl.DataFrame(self.all_captured_data)
            ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            available_ohlcv = [c for c in ohlcv_cols if c in df.columns]
            indicator_cols = sorted([c for c in df.columns if c not in ohlcv_cols])
            ordered_cols = available_ohlcv + indicator_cols
            df = df.select(ordered_cols)
            
            return df.sort('bar')
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pl.DataFrame()

    def export_data(self, filename: str = 'intermediates', export_dir: str = 'exports') -> pl.DataFrame:
        try:
            import os
            os.makedirs(export_dir, exist_ok=True)
            
            self.flush_remaining_batch()
            
            if not self.all_captured_data:
                print("\nNo data to export")
                return pl.DataFrame()

            df = self.get_dataframe()

            if len(df) > 0:
                parquet_path = f"{export_dir}/{filename}.parquet"
                csv_path = f"{export_dir}/{filename}.csv"

                df.write_parquet(parquet_path)
                df.write_csv(csv_path)
                
                print(f"\nExported:")
                print(f"  Parquet: {parquet_path}")
                print(f"  CSV: {csv_path}")
                print(f"\nShape: {df.shape}")

                ohlcv_cols = [c for c in df.columns if c in ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
                indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
                
                print(f"\nOHLCV columns ({len(ohlcv_cols)}):")
                for col in ohlcv_cols:
                    print(f"  - {col}")
                
                print(f"\nIndicator variables ({len(indicator_cols)}):")
                for i, var in enumerate(indicator_cols, 1):
                    print(f"  {i:2d}. {var}")
                
                return df
            else:
                return pl.DataFrame()

        except Exception as e:
            print(f"Error exporting: {e}")
            return pl.DataFrame()

    def print_summary(self):
        try:
            self.flush_remaining_batch()
            
            if not self.all_captured_data:
                print("\nNo data captured")
                return

            df = self.get_dataframe()

            print(f"\n{'='*80}")
            print("INTERMEDIATE CAPTURE RESULTS")
            print(f"{'='*80}")
            print(f"Bars: {len(df):,}")
            print(f"Total Columns: {len(df.columns)}")
            
            ohlcv_cols = [c for c in df.columns if c in ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
            
            print(f"OHLCV Columns: {len(ohlcv_cols)}")
            print(f"Indicator Variables: {len(indicator_cols)}")
            
            print(f"\nSample (last 3 bars):")

            sample_cols = ohlcv_cols + indicator_cols[:5]
            print(df.select(sample_cols).tail(3))
            
            print(f"{'='*80}")

        except Exception as e:
            print(f"Error: {e}")

optimized_patch = TransparencyPatch()

def activate_patch(debug: bool = False):
    print("Activating Transparency Patch")
    print("="*60)
    
    optimized_patch.debug = debug
    optimized_patch.apply_indicator_patch()
    
    print("PATCH ACTIVATED")
    if debug:
        print("  Debug: ON")


def capture_patch(strategy):
    optimized_patch.capture_patch_fast(strategy)

def export_data(filename: str = 'intermediates', export_dir: str = 'exports'):
    return optimized_patch.export_data(filename, export_dir)

def print_patch(auto_export: bool = True, filename: str = 'intermediates'):
    optimized_patch.print_summary()

    if auto_export:
        print(f"\n{'='*80}")
        print("Auto-exporting...")
        export_data(filename=filename)
        print(f"{'='*80}\n")