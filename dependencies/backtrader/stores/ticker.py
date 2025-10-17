import asyncio, websockets, random, json, threading, time, sys, signal
from datetime import datetime

def createRandomToken(length=12):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(random.choice(chars) for i in range(length))

def getEpoch():
    return int(time.time())

class ticker:
    
    def __init__(self, symbols="BINANCE:BTCUSDT", save=False, database_name="database.db", split_symbols=False, verbose=False):
        if isinstance(symbols, str):
            symbols = [symbols]
        
        self.states = {}
        for symbol in symbols:
            self.states[symbol] = {"volume": 0, "price": 0, "change": 0, "changePercentage": 0, "time": 0}
        
        self.loop = asyncio.get_event_loop()
        self.symbols = symbols
        self.save = save
        self.connected = False
        self.databaseName = database_name
        self.splitSymbols = split_symbols
        self.cb = None
        self.db = False
        self.run = True
        
        # Reconnection parameters
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        self.last_message_time = time.time()
        self.connection_timeout = 30  # 30 seconds without data = reconnect

        self.verbose = verbose
        if verbose:
            self.saves = 0

    async def connect(self):
        while self.run and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                if self.verbose or self.reconnect_attempts > 0:
                    print(f"Connecting to TradingView WebSocket... (attempt {self.reconnect_attempts + 1})")
                
                self.connection = await websockets.connect(
                    "wss://data.tradingview.com/socket.io/websocket", 
                    origin="https://www.tradingview.com",
                    ping_interval=20,
                    ping_timeout=10
                )
                
                if self.verbose or self.reconnect_attempts > 0:
                    print("✅ Connected to TradingView")
                
                self.reconnect_attempts = 0
                self.last_message_time = time.time()

                await self.authenticate()
                await self.waitForMessages()
                
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.WebSocketException,
                    ConnectionRefusedError,
                    OSError) as e:
                self.reconnect_attempts += 1
                print(f"❌ Connection error: {e}")
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    wait_time = self.reconnect_delay * self.reconnect_attempts
                    print(f"🔄 Reconnecting in {wait_time}s... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ Max reconnection attempts reached. Stopping.")
                    self.run = False
                    break
            
            except Exception as e:
                print(f"❌ Unexpected error in connect(): {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.reconnect_delay)

    async def waitForMessages(self):
        try:
            while self.run:
                try:
                    message = await asyncio.wait_for(
                        self.connection.recv(), 
                        timeout=60.0
                    )
                    self.last_message_time = time.time()
                    messages = await self.readMessage(message)
                    for msg in messages:
                        self.parseMessage(msg)
                        
                except asyncio.TimeoutError:
                    if self.verbose:
                        print("⚠️ No message received for 60 seconds")
                    if time.time() - self.last_message_time > self.connection_timeout:
                        print("❌ Connection appears dead (no data for 60+ seconds), forcing reconnect")
                        raise websockets.exceptions.ConnectionClosed(1006, "Timeout")
                    
        except (websockets.exceptions.ConnectionClosed, 
                websockets.exceptions.WebSocketException) as e:
            print(f"⚠️ Connection lost: {e}. Will attempt reconnection...")
            # trigger reconnection in connect()
            raise
        except Exception as e:
            print(f"❌ Unexpected error in waitForMessages(): {e}")
            import traceback
            traceback.print_exc()
            raise

    async def readMessage(self, message):
        messages = message.split("~m~")
        messagesObj = []
        for message in messages:
            if '{' in message or '[' in message:
                messagesObj.append(json.loads(message))
            else:
                if "~h~" in message:
                    await self.connection.send(f"~m~{len(message)}~m~{message}")

        return messagesObj

    def createMessage(self, name, params):
        message = json.dumps({'m': name, 'p': params})
        return f"~m~{len(message)}~m~{message}"

    async def sendMessage(self, name, params):
        message = self.createMessage(name, params)
        await self.connection.send(message)

    async def authenticate(self):
        self.cs = "cs_" + createRandomToken()

        await self.sendMessage("set_auth_token", ["unauthorized_user_token"])
        await self.sendMessage("chart_create_session", [self.cs, ""])

        q = createRandomToken()
        qs = "qs_" + q
        qsl = "qs_snapshoter_basic-symbol-quotes_" + q
        await self.sendMessage("quote_create_session", [qs])
        await self.sendMessage("quote_create_session", [qsl])
        await self.sendMessage("quote_set_fields", [qsl, "base-currency-logoid", "ch", "chp", "currency-logoid", "currency_code", "currency_id", "base_currency_id", "current_session", "description", "exchange", "format", "fractional", "is_tradable", "language", "local_description", "listed_exchange", "logoid", "lp", "lp_time", "minmov", "minmove2", "original_name", "pricescale", "pro_name", "short_name", "type", "typespecs", "update_mode", "volume", "variable_tick_size", "value_unit_id"])
        await self.sendMessage("quote_add_symbols", [qsl] + self.symbols)
        await self.sendMessage("quote_fast_symbols", [qs] + self.symbols)

    def parseMessage(self, message):
        try:
            message['m']
        except KeyError:
            return

        if message['m'] == "qsd":
            self.forTicker(message)

    def forTicker(self, receivedData):
        symbol = receivedData['p'][1]['n']
        data = receivedData['p'][1]['v']
        
        items = {
            "volume": "volume",
            "price": "lp",
            "changePercentage": "chp",
            "change": "ch",
            "time": "lp_time"
        }
        
        for key, data_key in items.items():
            value = data.get(data_key)
            if value is not None:
                self.states[symbol][key] = value
        
        if self.cb is not None: # Send callback to Tradingview Store
            self.cb(symbol, self.states[symbol])
        
        if self.verbose:
            self.saves += 1

    async def monitor_connection(self):
        """Monitor connection health and force reconnect if needed"""
        while self.run:
            await asyncio.sleep(60)  # Check every minute
            
            silence_duration = time.time() - self.last_message_time
            if silence_duration > self.connection_timeout:
                print(f"⚠️ Watchdog: No data for {silence_duration/60:.1f} minutes, forcing reconnect")
                try:
                    if hasattr(self, 'connection') and self.connection:
                        await self.connection.close()
                except Exception as e:
                    if self.verbose:
                        print(f"Error closing connection in watchdog: {e}")

    async def giveAnUpdate(self):
        while self.run:
            await asyncio.sleep(5)
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Watching {len(self.symbols)} tickers → received {self.saves} updates")
            self.saves = 0

    def start(self):
        self.loop = asyncio.new_event_loop()
        def _start(loop):
            asyncio.set_event_loop(loop)
            self.run = True
            self.task = loop.create_task(self.connect())
            self.monitor_task = loop.create_task(self.monitor_connection())
            if self.verbose:
                self.updateTask = loop.create_task(self.giveAnUpdate())
            loop.run_forever()

        t = threading.Thread(target=_start, args=(self.loop,))
        t.daemon = True
        t.start()
        self.thread = t

        signal.signal(signal.SIGINT, self.cleanup_on_exit)  # SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, self.cleanup_on_exit) # SIGTERM (termination signal)

    def stop(self):
        print("Stopping ticker...")
        self.run = False
        
        if hasattr(self, 'task'):
            self.task.cancel()
        if hasattr(self, 'monitor_task'):
            self.monitor_task.cancel()
        if self.verbose and hasattr(self, 'updateTask'):
            self.updateTask.cancel()
        
        if hasattr(self, 'loop') and self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=5)
        
        print("Ticker stopped")

    def cleanup_on_exit(self, a, b):
        print("Closing (can take a few seconds)")
        self.stop()
        sys.exit(0)