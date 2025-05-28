from datetime import datetime, timedelta
import time
import ccxt
import polars as pl

DATETIME_FORMAT = {"daily": "%Y-%m-%d", "intraday": "%Y-%m-%d %H:%M:%S"}

def unix_time_millis(date_str: str, time_resolution: str) -> int:
    dt_format = DATETIME_FORMAT["intraday"] if any(x in time_resolution for x in ["m", "h", "s"]) else DATETIME_FORMAT["daily"]
    
    if dt_format == DATETIME_FORMAT["intraday"] and len(date_str.strip()) == 10:
        date_str += " 00:00:00"

    dt = datetime.strptime(date_str, dt_format)
    return int(dt.timestamp() * 1000)

def get_crypto_data(asset, start_date, end_date, time_resolution, exchange):
    dt_format = DATETIME_FORMAT["intraday"] if "m" in time_resolution or "h" in time_resolution else DATETIME_FORMAT["daily"]
    start_date_epoch = unix_time_millis(start_date, time_resolution)
    end_date_epoch = unix_time_millis(end_date, time_resolution)

    if exchange not in ccxt.exchanges:
        raise NotImplementedError(f"The exchange {exchange} is not yet supported. Available exchanges: {', '.join(ccxt.exchanges)}")

    ex = getattr(ccxt, exchange)({"verbose": False})

    previous_request_end_date_epoch = start_date_epoch
    request_start_date_epoch = start_date_epoch

    ohlcv_records = []

    while previous_request_end_date_epoch < end_date_epoch:
        ohlcv_lol = ex.fetch_ohlcv(asset, time_resolution, since=request_start_date_epoch)

        if not ohlcv_lol:
            request_start_date_epoch += int(timedelta(days=1).total_seconds()) * 1000
            request_start_date_epoch = unix_time_millis(
                datetime.utcfromtimestamp(request_start_date_epoch / 1000).strftime(dt_format)
            )
            previous_request_end_date_epoch = request_start_date_epoch - 1
            continue

        ohlcv_records.extend(ohlcv_lol)

        current_request_end_date_epoch = ohlcv_lol[-1][0]

        if current_request_end_date_epoch <= previous_request_end_date_epoch:
            break

        request_start_date_epoch = current_request_end_date_epoch + 1
        previous_request_end_date_epoch = current_request_end_date_epoch

    if not ohlcv_records:
        return None

    df = pl.DataFrame(
        ohlcv_records,
        schema=["dt", "Open", "High", "Low", "Close", "Volume"],
        orient="row"
    )

    df = df.with_columns([
        pl.col("dt").cast(pl.Datetime(time_unit="ms"))
    ])

    parsed_end_date = datetime.strptime(end_date, DATETIME_FORMAT["intraday"] if ":" in end_date else DATETIME_FORMAT["daily"])
    df = df.filter(pl.col("dt") <= parsed_end_date)

    df = df.sort("dt").with_columns([
        pl.lit(start_date).alias("start_date"),
        pl.lit(end_date).alias("end_date"),
        pl.lit(asset).alias("symbol")
    ])

    return df
