import sys
import types

# Some optional dependencies are not available in the test environment. Stub them so that
# importing backtrader packages during test collection does not fail.
if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__dict__.setdefault("__path__", [])

    def _noop(*_args, **_kwargs):
        return None

    matplotlib_stub.use = _noop
    sys.modules["matplotlib"] = matplotlib_stub

    pyplot_stub = types.ModuleType("pyplot")
    pyplot_stub.plot = _noop
    pyplot_stub.show = _noop
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "rich" not in sys.modules:
    rich_stub = types.ModuleType("rich")
    progress_stub = types.ModuleType("progress")

    class _DummyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def add_task(self, *args, **kwargs):
            return 0

        def advance(self, *args, **kwargs):
            return None

    progress_stub.Progress = _DummyProgress
    progress_stub.SpinnerColumn = object
    progress_stub.TextColumn = object
    progress_stub.BarColumn = object
    progress_stub.TaskProgressColumn = object
    progress_stub.TimeRemainingColumn = object

    rich_stub.progress = progress_stub
    console_stub = types.ModuleType("console")

    class _DummyConsole:
        def __init__(self, *args, **kwargs):
            pass

        def status(self, *args, **kwargs):
            return _DummyProgress()

        def print(self, *args, **kwargs):
            return None

    console_stub.Console = _DummyConsole
    rich_stub.console = console_stub
    sys.modules["rich"] = rich_stub
    sys.modules["rich.progress"] = progress_stub
    sys.modules["rich.console"] = console_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _DummyResponse:
        status_code = 200

        def json(self):
            return {}

    def _request(*_args, **_kwargs):
        return _DummyResponse()

    requests_stub.get = _request
    requests_stub.post = _request
    requests_stub.put = _request
    requests_stub.delete = _request
    sys.modules["requests"] = requests_stub

if "telethon" not in sys.modules:
    telethon_stub = types.ModuleType("telethon")

    class _DummyTelegramClient:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self):
            return None

        def is_connected(self):
            return True

        async def is_user_authorized(self):
            return True

        async def get_entity(self, *args, **kwargs):
            return None

        async def send_message(self, *args, **kwargs):
            return None

        async def disconnect(self):
            return None

    telethon_stub.TelegramClient = _DummyTelegramClient
    sys.modules["telethon"] = telethon_stub

if "joblib" not in sys.modules:
    joblib_stub = types.ModuleType("joblib")

    def _joblib_noop(*_args, **_kwargs):
        return None

    joblib_stub.load = _joblib_noop
    joblib_stub.dump = _joblib_noop
    sys.modules["joblib"] = joblib_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _DummyDataFrame:
        def __init__(self, *args, **kwargs):
            pass

    pandas_stub.DataFrame = _DummyDataFrame
    pandas_stub.Series = _DummyDataFrame
    sys.modules["pandas"] = pandas_stub

if "fast_mssql" not in sys.modules:
    sys.modules["fast_mssql"] = types.ModuleType("fast_mssql")

if "web3" not in sys.modules:
    web3_stub = types.ModuleType("web3")

    class _DummyWeb3:
        class HTTPProvider:
            def __init__(self, *_args, **_kwargs):
                pass

        def __init__(self, *_args, **_kwargs):
            pass

    web3_stub.Web3 = _DummyWeb3
    sys.modules["web3"] = web3_stub
