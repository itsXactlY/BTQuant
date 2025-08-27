import backtrader as bt
from backtrader import date2num
from backtrader.utils.py3 import string_types, integer_types
import polars as pl

class PolarsData(bt.feed.DataBase):
    '''
    Uses a Polars DataFrame as the feed source
    '''

    params = (
        ('nocase', True),
        ('datetime', 0),  # Default: first column is datetime
        ('open', 1),      
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),  # -1 means not present
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
                print(f"Error getting value for {datafield} at index {self._idx}, col_idx {col_idx}: {e}")
                line[0] = float('nan')

        dt_idx = self._colmapping['datetime']
        if dt_idx is not None:
            try:
                dt_value = self.p.dataname[self.colnames[dt_idx]][self._idx]
                if hasattr(dt_value, "item"):
                    dt_value = dt_value.item()
                # convert
                from datetime import datetime
                if isinstance(dt_value, str):
                    dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                elif isinstance(dt_value, (int, float)):
                    dt = datetime.fromtimestamp(float(dt_value)/1000 if dt_value > 1e10 else float(dt_value))
                else:
                    dt = dt_value
                self.lines.datetime[0] = date2num(dt)
            except Exception as e:
                print(f"Error processing datetime at index {self._idx}, col_idx {dt_idx}: {e}")
                self.lines.datetime[0] = float('nan')

        return True