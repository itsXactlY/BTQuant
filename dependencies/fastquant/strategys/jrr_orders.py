import requests
import asyncio
from telethon import TelegramClient
from abc import ABC, abstractmethod
from typing import Dict, Any
from fastquant.dontcommit import identify, jrr_webhook_url, discord_webhook_url, telegram_api_id, telegram_api_hash, telegram_session_file, telegram_channel_debug, telegram_channel_machinelearning
from typing import Optional

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
        
    async def initialize(self, loop=None):
        """Initialize with optional event loop - took me just half an day..."""
        self.client = TelegramClient(
            self.session_file, 
            self.api_id, 
            self.api_hash,
            loop=loop
        )
        
        await self.client.connect()
        if not await self.client.is_user_authorized():
            raise Exception("⚠️ Telegram client is not authorized! Use a valid session.")
        print("✅ Telegram client is now connected and ready to send messages.")

    async def send_message(self, message: str) -> None:
        """Implements abstract method from MessagingService"""
        if not self.client:
            raise Exception("Telegram client not initialized")
        try:
            entity = await self.client.get_entity(self.channel_id)
            await self.client.send_message(entity, message)
            print(f"✅ Telegram alert sent: {message}")
        except Exception as e:
            print(f"❌ Telegram alert failed: {e}")

async def initialize_services():
    telegram_service = TelegramService(
        api_id=telegram_api_id,
        api_hash=telegram_api_hash,
        session_file=telegram_session_file,
        channel_id=telegram_channel_debug
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
            "username": "DEBUG SHOWCASE",
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
            "Exchange": exchange,
            "Market": "spot",
            "Account": account,
            "Action": "Buy",
            "Asset": asset,
            "USD": str(int(amount)),
            "Identity": identify
        }
        return self._send_jrr_request(payload)

    def send_jrr_close_request(self, exchange: str, account: str, asset: str) -> str:
        payload = {
            "Exchange": exchange,
            "Market": "spot",
            "Account": account,
            "Action": "Close",
            "Asset": asset,
            "Identity": identify
        }
        return self._send_jrr_request(payload)
