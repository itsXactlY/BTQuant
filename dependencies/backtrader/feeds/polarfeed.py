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
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1), # -1 means not present
    )

    datafields = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'
    ]

    def __init__(self):
        super(PolarsData, self).__init__()
        
        if not isinstance(self.p.dataname, pl.DataFrame):
            if hasattr(self.p.dataname, '__iter__'):
                try:
                    self.p.dataname = pl.DataFrame(self.p.dataname)
                except Exception as e:
                    raise ValueError(f"Could not convert data to Polars DataFrame: {e}")
        
        self.colnames = self.p.dataname.columns
        self._colmapping = {}
        
        for datafield in self.getlinealiases():
            param_value = getattr(self.params, datafield)
            
            if isinstance(param_value, integer_types):
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
            
            elif isinstance(param_value, string_types):
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
            line[0] = self.p.dataname.row(self._idx)[col_idx]
            
        dt_idx = self._colmapping['datetime']
        if dt_idx is not None:
            dt_value = self.p.dataname.row(self._idx)[dt_idx]
            
            if isinstance(dt_value, (str, int, float)):
                if isinstance(dt_value, str):
                    from datetime import datetime
                    dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                else:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(dt_value)
            else:
                dt = dt_value

            dtnum = date2num(dt)
            self.lines.datetime[0] = dtnum
            
        return True