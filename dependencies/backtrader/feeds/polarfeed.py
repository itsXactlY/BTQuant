import backtrader as bt
from backtrader import date2num
import polars as pl
from datetime import datetime

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
        # Spalten EINMAL zu Python-Listen ziehen. Vorher stand in _load()
        # self.p.dataname[self.colnames[col_idx]][self._idx] -- und df["Open"]
        # erzeugt in polars ein NEUES Series-Objekt. Das passierte pro Bar und
        # pro Feld, also 7 Series-Konstruktionen je Bar. Bei 384k 1m-Bars sind
        # das rund 2.7 Mio. Series-Objekte und genau der Grund, warum ein
        # Backtest sich wie x100 langsam anfuehlt.
        df = self.p.dataname
        self._n = len(df)
        self._aliases = list(self.getlinealiases())
        self._cols = {}
        for datafield in self._aliases:
            col_idx = self._colmapping.get(datafield)
            if col_idx is None:
                continue
            try:
                self._cols[datafield] = (col_idx, df[self.colnames[col_idx]].to_list())
            except Exception as e:
                print(f"Error pre-buffering column {datafield}: {e}")
                self._cols[datafield] = None

    def _load(self):
        self._idx += 1
        if self._idx >= self._n:
            return False
        i = self._idx

        for datafield, packed in self._cols.items():
            if datafield == 'datetime' or packed is None:
                continue
            col_idx, values = packed
            line = getattr(self.lines, datafield)
            try:
                val = values[i]
                if hasattr(val, "item"):
                    val = val.item()
                line[0] = float(val)
            except Exception as e:
                print(f"Error getting value for {datafield} at index {i}, col_idx {col_idx}: {e}")
                line[0] = float('nan')

        packed = self._cols.get('datetime')
        if packed is not None:
            dt_idx, values = packed
            try:
                dt_value = values[i]
                if hasattr(dt_value, "item"):
                    dt_value = dt_value.item()
                if isinstance(dt_value, str):
                    dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                elif isinstance(dt_value, (int, float)):
                    dt = datetime.fromtimestamp(float(dt_value)/1000 if dt_value > 1e10 else float(dt_value))
                else:
                    dt = dt_value
                self.lines.datetime[0] = date2num(dt)
            except Exception as e:
                print(f"Error processing datetime at index {i}, col_idx {dt_idx}: {e}")
                self.lines.datetime[0] = float('nan')

        return True