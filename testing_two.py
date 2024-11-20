import requests
import time
from datetime import datetime

class RaydiumPriceFetcher:
    def __init__(self, token_pair: str, update_interval: int = 5):
        """
        Initialize the Raydium Price Fetcher.
        
        :param token_pair: The token pair (e.g., 'RAY-USDC') to fetch price for.
        :param update_interval: Interval in seconds to fetch price updates.
        """
        self.token_pair = token_pair
        self.update_interval = update_interval
        self.base_url = "https://api.raydium.io/price"
        self.running = True

    def fetch_price(self):
        """Fetch the USD price for the given token pair."""
        try:
            response = requests.get(f"{self.base_url}/{self.token_pair}")
            if response.status_code == 200:
                data = response.json()
                price = data.get("price", None)
                if price:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {self.token_pair} Price: ${price:.2f}")
                    return price
                else:
                    print("Price data not available in response.")
            else:
                print(f"Failed to fetch price data: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching price: {e}")
        return None

    def start(self):
        """Start fetching prices at regular intervals."""
        print(f"Starting price fetcher for {self.token_pair}. Updating every {self.update_interval}s.")
        while self.running:
            self.fetch_price()
            time.sleep(self.update_interval)

    def stop(self):
        """Stop the price fetcher."""
        print(f"Stopping price fetcher for {self.token_pair}.")
        self.running = False


# Example Usage
if __name__ == "__main__":
    raydium_fetcher = RaydiumPriceFetcher(token_pair="RAY-USDC", update_interval=5)
    try:
        raydium_fetcher.start()
    except KeyboardInterrupt:
        raydium_fetcher.stop()
