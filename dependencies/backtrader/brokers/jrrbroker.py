import requests
import asyncio
import logging
from telethon import TelegramClient
from abc import ABC, abstractmethod
from typing import Dict, Any
from backtrader.dontcommit import identify, jrr_webhook_url, discord_webhook_url, telegram_api_id, telegram_api_hash, telegram_session_file, telegram_channel
from typing import Optional, List, Deque
import backtrader as bt
from backtrader.order import Order, BuyOrder, SellOrder
from backtrader.position import Position
import collections

class MessagingService(ABC):
    @abstractmethod
    async def send_message(self, message: str) -> None:
        pass

class TelegramService(MessagingService):
    def __init__(self, api_id: int, api_hash: str, session_file: str, channel_id: int):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_file = session_file
        self.channel_id = channel_id
        self.client = None
        self.is_connected = False
        self.max_retries = 5
        self.retry_delay = 5
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up basic logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def initialize(self, loop=None):
        """Initialize with automatic reconnection handling"""
        self.client = TelegramClient(
            self.session_file,
            self.api_id,
            self.api_hash,
            loop=loop,
            connection_retries=10,
            retry_delay=self.retry_delay,
            auto_reconnect=True
        )
        
        await self.connect_with_retry()
        self.logger.info("✅ Telegram client is now connected and ready to send messages.")

    async def connect_with_retry(self):
        """Connect to Telegram with retry logic"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                if not self.client.is_connected():
                    await self.client.connect()
                
                if not await self.client.is_user_authorized():
                    self.logger.error("⚠️ Telegram client is not authorized! Use a valid session.")
                    raise Exception("Telegram client is not authorized! Use a valid session.")
                    
                self.is_connected = True
                return
                
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Connection attempt {retry_count} failed: {str(e)}")
                
                if retry_count >= self.max_retries:
                    self.logger.error(f"Failed to connect after {self.max_retries} attempts")
                    raise
                
                await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff

    async def send_message(self, message: str) -> None:
        """Implements abstract method from MessagingService with reconnection logic"""
        if not self.client:
            raise Exception("Telegram client not initialized")
        
        for attempt in range(3):
            try:
                if not self.client.is_connected():
                    self.logger.warning("Connection lost before sending message. Reconnecting...")
                    await self.connect_with_retry()
                
                entity = await self.client.get_entity(self.channel_id)
                await self.client.send_message(entity, message)
                self.logger.info(f"✅ Telegram alert sent: {message}")
                return
                
            except Exception as e:
                self.logger.error(f"Failed to send message (attempt {attempt+1}/3): {str(e)}")
                
                if attempt < 2:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        self.logger.error("Failed to send message after multiple attempts")
        raise Exception("Failed to send Telegram message after multiple attempts")
        
    async def disconnect(self):
        """Safely disconnect the client"""
        if self.client and self.client.is_connected():
            await self.client.disconnect()
            self.is_connected = False
            self.logger.info("Telegram client disconnected")

async def initialize_services():
    telegram_service = TelegramService(
        api_id=telegram_api_id,
        api_hash=telegram_api_hash,
        session_file=telegram_session_file,
        channel_id=telegram_channel
    )
    await telegram_service.initialize()
    
    discord_service = DiscordService(discord_webhook_url)
    alert_manager = AlertManager([telegram_service, discord_service])
    jrr_order_base = JrrOrderBase(alert_manager)
    
    return jrr_order_base

class DiscordService(MessagingService):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_message(self, message: str) -> None:
        payload = {
            "username": "BTQuant Alerts",
            "avatar_url": "https://i.imgur.com/TISmmHs.jpg",
            "embeds": [{
                "title": "Alert arrived!",
                "description": message,
                "color": 3066993,
                "footer": {
                    "text": "Powered by BTQuant!",
                    "icon_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYQY8CsntTC-nQ4nTVvSp_fY6zcWtLfdubhg&s"
                }
            }]
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(self.webhook_url, json=payload, headers={"Content-Type": "application/json"})
            )
            response.raise_for_status()
            print(f"Discord response status code: {response.status_code}")
        except Exception as e:
            print(f"Error in Discord alert: {e}")

class AlertManager:
    def __init__(self, messaging_services: list[MessagingService], loop=None):
        self.messaging_services = messaging_services
        self.loop = loop or asyncio.new_event_loop()
        
    def send_alert(self, message: str) -> None:
        async def _send():
            tasks = [service.send_message(message) for service in self.messaging_services]
            await asyncio.gather(*tasks)
            
        future = asyncio.run_coroutine_threadsafe(_send(), self.loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            print(f"Error sending alert: {e}")

class JrrOrderBase:
    def __init__(self, alert_manager: Optional[AlertManager] = None):
        """
        Initialize with an optional alert_manager.
        If alerts are disabled (alert_manager is None), no alerts will be sent.
        """
        self.alert_manager = alert_manager

    def _send_jrr_request(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.post(jrr_webhook_url, json=payload)
            response.raise_for_status()
            response_msg = response.text
            print(f"Response status code: {response.status_code}")
            print("Response content:\n", response_msg)
            
            # Only attempt to send a Discord alert if an alert_manager is provided.
            if self.alert_manager is not None:
                # JackRabbit responses are sent only to Discord to avoid spamming Telegram.
                discord_service = next(
                    (s for s in self.alert_manager.messaging_services if isinstance(s, DiscordService)),
                    None
                )
                if discord_service is not None:
                    future = asyncio.run_coroutine_threadsafe(
                        discord_service.send_message(response_msg),
                        self.alert_manager.loop
                    )
                    future.result(timeout=10)
                else:
                    print("Discord service not found in alert_manager services.")
            else:
                print("Alert manager is not set (alerts disabled); skipping alert sending.")

            return response_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg

    def send_jrr_buy_request(self, exchange: str, account: str, asset: str, amount: float) -> str:
        payload = {
            "Exchange": exchange, #"mimic",
            "Market": "spot",
            "Account": account,
            "Action": "Buy",
            "Asset": asset,
            "USD": str(amount),
            "Identity": identify
        }
        print(f"payload: {payload}")
        return self._send_jrr_request(payload)

    def send_jrr_close_request(self, exchange: str, account: str, asset: str) -> str:
        payload = {
            "Exchange": exchange, #"mimic",
            "Market": "spot",
            "Account": account,
            "Action": "Close",
            "Asset": asset,
            "Identity": identify
        }
        print(f"payload: {payload}")
        return self._send_jrr_request(payload)

class JrrBroker(bt.BrokerBase):
    """
    Live component broker for JackRabbitRelay (JRR).
    Maps Backtrader order calls to JRR webhook requests.
    """
    params = (
        ('cash', 10000.0),
        ('exchange', 'mimic'),
        ('account', 'default'),
        ('debug', True),
    )

    def __init__(self, alert_manager: Optional[AlertManager] = None):
        super(JrrBroker, self).__init__()
        self.startingcash = self.cash = self.p.cash
        self.positions = collections.defaultdict(Position)
        self.orders: List[Order] = []
        self.notifs: Deque[Order] = collections.deque()
        self.jrr = JrrOrderBase(alert_manager=alert_manager)
        
        if self.p.debug:
            print(f"JrrBroker initialized (Exchange: {self.p.exchange}, Account: {self.p.account})")

    def getcash(self):
        return self.cash

    def get_cash(self):
        return self.getcash()

    def getvalue(self, datas=None):
        # In a real live scenario, this should fetch from exchange.
        # For now, we return tracked cash.
        return self.cash

    def get_value(self, datas=None):
        return self.getvalue(datas)

    def notify(self, order):
        self.notifs.append(order)

    def get_notification(self):
        try:
            return self.notifs.popleft()
        except IndexError:
            pass
        return None

    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):
        
        # Create BT order object
        order = BuyOrder(owner=owner, data=data, size=size, price=price, 
                          pricelimit=plimit, exectype=exectype, valid=valid, 
                          tradeid=tradeid)
        
        # Execute via JRR
        # Note: data.symbol is used as Asset. In BTQuant, data._dataname is often the symbol.
        asset = getattr(data, '_dataname', str(data))
        
        # Calculate USD amount if possible
        exec_price = price if price else data.close[0]
        usd_amount = size * exec_price if exec_price else size
        
        if self.p.debug:
            print(f"JrrBroker: Sending BUY request for {asset} (Size: {size}, Approx USD: {usd_amount:.2f})")
            
        self.jrr.send_jrr_buy_request(
            exchange=self.p.exchange,
            account=self.p.account,
            asset=asset,
            amount=usd_amount
        )
        
        # Automatically mark as completed for JRR (it's a webhook trigger)
        order.submit()
        order.accept()
        # In live trading, we'd wait for execution confirmation, but JRR is fire-and-forget
        order.execute(data.datetime[0], size, exec_price, 0, 0, 0)
        order.completed()
        
        self.notify(order)
        return order

    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):
        
        # Create BT order object
        order = SellOrder(owner=owner, data=data, size=size, price=price, 
                           pricelimit=plimit, exectype=exectype, valid=valid, 
                           tradeid=tradeid)
        
        asset = getattr(data, '_dataname', str(data))
        exec_price = price if price else data.close[0]

        if self.p.debug:
            print(f"JrrBroker: Sending CLOSE request for {asset}")

        self.jrr.send_jrr_close_request(
            exchange=self.p.exchange,
            account=self.p.account,
            asset=asset
        )
        
        order.submit()
        order.accept()
        order.execute(data.datetime[0], size, exec_price, 0, 0, 0)
        order.completed()
        
        self.notify(order)
        return order

    def getposition(self, data):
        return self.positions[data]
