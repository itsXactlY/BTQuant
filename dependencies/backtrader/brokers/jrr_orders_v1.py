import requests
from dontcommit import identify, jrr_webhook_url, discord_webhook_url


class Alert:
    def __init__(self, discord_webhook_url):
        self.discord_webhook_url = discord_webhook_url

    def send_alert(self, message):
        message_payload = {
            "username": "DEBUG TEST - IGNORE ME",
            "avatar_url": "https://i.imgur.com/TISmmHs.jpg",
            "embeds": [
                {
                    "title": "DEBUG Alert!",
                    "description": message,
                    "color": 3066993,  # Lightning green color code
                    "footer": {
                        "text": "Version3 debug",
                        "icon_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYQY8CsntTC-nQ4nTVvSp_fY6zcWtLfdubhg&s"
                    }
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.discord_webhook_url, json=message_payload, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            print(f"Discord response status code: {response.status_code}")
            print(f"Discord response content: {response.text}")

        except Exception as e:
            print(f"Error in send_alert: {e}")

class JrrOrderBase:
    def __init__(self):
        self.alert_instance = Alert(discord_webhook_url)

    def send_jrr_buy_request(self, exchange, account, asset, amount):
        _amount = int(amount)
        
        payload = {
            "Exchange": exchange,
            "Market": "spot",
            "Account": account,
            "Action": "Buy",
            "Asset": asset,
            "USD": str(_amount),
            "Identity": identify
        }
        try:
            response = requests.post(jrr_webhook_url, json=payload)
            response.raise_for_status()
            jrr_response_buy_msg = response.text
            print(f"Response status code: {response.status_code}")
            print("Response content:\n", response.text)

            self.alert_instance.send_alert(jrr_response_buy_msg)  # Sending response.text to Discord
            return jrr_response_buy_msg

        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return "Error: " + str(e)

    def send_jrr_close_request(self, exchange, account, asset):
        payload = {
            "Exchange": exchange,
            "Market": "spot",
            "Account": account,
            "Action": "Close",
            "Asset": asset,
            "Identity": identify
        }
        try:
            response = requests.post(jrr_webhook_url, json=payload)
            response.raise_for_status()
            jrr_response_close_msg = response.text
            print(f"Response status code: {response.status_code}")
            print("Response content:\n", response.text)

            self.alert_instance.send_alert(jrr_response_close_msg)  # Sending response.text to Discord
            return jrr_response_close_msg

        except requests.exceptions.RequestException as e:
            print("Error:", e)
            return "Error: " + str(e)
