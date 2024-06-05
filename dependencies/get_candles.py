import json, requests, datetime as dt, pandas as pd

def get_binance_bars(symbol, interval, startTime, endTime):
 
    url = "https://api.binance.com/api/v3/klines"
 
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '9999'
 
    req_params = {"symbol" : symbol, 'interval' : interval, 'startTime' : startTime, 'endTime' : endTime, 'limit' : limit}
 
    df = pd.DataFrame(json.loads(requests.get(url, params = req_params).text))
 
    if (len(df.index) == 0):
        return None
     
    df = df.iloc[:, 0:6]
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
 
    df.open      = df.open.astype("float")
    df.high      = df.high.astype("float")
    df.low       = df.low.astype("float")
    df.close     = df.close.astype("float")
    df.volume    = df.volume.astype("float")
    
    df['adj_close'] = df['close']
     
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
    
    return df


_coin = 'PEPEUSDT'
_tf = '1m'


df_list = []
last_datetime = dt.datetime(2017,1,1)
while True:
    new_df = get_binance_bars(_coin, _tf, last_datetime, dt.datetime.now())
    if new_df is None:
        break
    df_list.append(new_df)
    last_datetime = max(new_df.index) + dt.timedelta(0, 1)
 
df = pd.concat(df_list)
df.shape

def save_to_csv(df):
    df.to_csv(f'binance_bars_{_coin}_{_tf}.csv')

save_to_csv(df)

