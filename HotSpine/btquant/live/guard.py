import os

if os.environ.get("BTQ_LIVE") == "1":
    raise RuntimeError("SQL access forbidden in live mode")
