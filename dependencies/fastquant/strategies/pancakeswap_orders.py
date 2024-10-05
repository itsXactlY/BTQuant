import requests
discord_webhook_url = ''

class Alert:
    def __init__(self, discord_webhook_url):
        self.discord_webhook_url = discord_webhook_url

    def send_alert(self, message):
        message_payload = {
            "username": "DEBUG SHOWCASE",
            "avatar_url": "https://i.imgur.com/TISmmHs.jpg",
            "embeds": [
                {
                    "title": "Alert arrived!",
                    "description": message,
                    "color": 3066993,  # Lightning green color code
                    "footer": {
                        "text": "Powered by aLca for Quants!",
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


import time
from web3 import Web3
from fastquant.dontcommit import bsc_privaccount1, bsc_privaccountaddress

class PancakeSwapV2DirectOrderBase:
    def __init__(self, coin, collateral, **kwargs):
        super().__init__(**kwargs)
        self.BSC_RPC_URL = "https://bsc-dataseed.binance.org/"
        self.PRIVATE_KEY = bsc_privaccount1
        self.WALLET_ADDRESS = Web3.to_checksum_address(bsc_privaccountaddress)
        self.PANCAKE_ROUTER_ADDRESS = Web3.to_checksum_address('0x10ED43C718714eb63d5aA57B78B54704E256024E')
        self.TOKEN_ADDRESS = Web3.to_checksum_address(coin)
        self.WBNB_ADDRESS = Web3.to_checksum_address(collateral)
        self.web3 = Web3(Web3.HTTPProvider(self.BSC_RPC_URL))
        if not self.web3.is_connected():
            raise Exception("Failed to connect to Binance Smart Chain")
        self.alert_instance = Alert(discord_webhook_url)

        self.pancake_router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForETH",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

        self.pancake_router = self.web3.eth.contract(address=self.PANCAKE_ROUTER_ADDRESS, abi=self.pancake_router_abi)

    def to_wei(self, amount):
        return self.web3.to_wei(amount, 'ether')

    def get_gas_price(self):
        return self.web3.eth.gas_price

    def send_pcs_buy_request(self, amount):
        try:
            nonce = self.web3.eth.get_transaction_count(self.WALLET_ADDRESS)
            self.swap_bnb_for_token(amount, nonce)
            return f"Swap Transaction successful. Amount: {amount} BNB"
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

    def send_pcs_close_request(self):
        try:
            nonce = self.web3.eth.get_transaction_count(self.WALLET_ADDRESS)
            self.swap_all_tokens_for_bnb(nonce)
            return "Swap Transaction successful. All tokens swapped for BNB."
        except Exception as e:
            print(f"Error: {e}")
            return f"Error: {e}"

    def swap_bnb_for_token(self, bnb_amount, nonce):
        path = [self.WBNB_ADDRESS, self.TOKEN_ADDRESS]
        amount_in_wei = self.to_wei(bnb_amount)
        deadline = int(time.time()) + 60 * 10  # 10 minutes from now

        # Check if the wallet has sufficient BNB balance
        bnb_balance = self.web3.eth.get_balance(self.WALLET_ADDRESS)
        if bnb_balance < amount_in_wei:
            print("Insufficient BNB balance.")
            return

        amounts_out = self.pancake_router.functions.getAmountsOut(amount_in_wei, path).call()
        amount_out_min = amounts_out[1] * 90 // 100  # 10% slippage tolerance

        txn = self.pancake_router.functions.swapExactETHForTokens(
            amount_out_min,
            path,
            self.WALLET_ADDRESS,
            deadline
        ).build_transaction({
            'from': self.WALLET_ADDRESS,
            'value': amount_in_wei,
            'gas': 300000,
            'gasPrice': self.get_gas_price(),
            'nonce': nonce
        })

        # Sign and send the transaction
        signed_txn = self.web3.eth.account.sign_transaction(txn, self.PRIVATE_KEY)
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction).hex()
        print(f"Swap Transaction Hash: https://bscscan.com/tx/0x{tx_hash}")

    def swap_all_tokens_for_bnb(self, nonce):
        path = [self.TOKEN_ADDRESS, self.WBNB_ADDRESS]
        balance_in_wei, balance_human_readable = self.get_token_balance()
        print(f"Token balance: {balance_human_readable} tokens")

        if balance_in_wei == 0:
            print("No tokens available to swap.")
            return

        deadline = int(time.time()) + 60 * 10  # 10 minutes from now
        amounts_out = self.pancake_router.functions.getAmountsOut(balance_in_wei, path).call()
        amount_out_min = amounts_out[1] * 90 // 100  # 10% slippage tolerance

        allowance = self.get_token_allowance()
        if allowance < balance_in_wei:
            print(f"Current allowance: {allowance}. Approving maximum allowance...")

            erc20_abi = [
                {
                    "constant": False,
                    "inputs": [
                        {"name": "_spender", "type": "address"},
                        {"name": "_value", "type": "uint256"}
                    ],
                    "name": "approve",
                    "outputs": [{"name": "success", "type": "bool"}],
                    "type": "function"
                }
            ]
            token_contract = self.web3.eth.contract(address=self.TOKEN_ADDRESS, abi=erc20_abi)
            MAX_ALLOWANCE = int(10**20)
            approve_txn = token_contract.functions.approve(self.PANCAKE_ROUTER_ADDRESS, MAX_ALLOWANCE).build_transaction({
                'from': self.WALLET_ADDRESS,
                'gas': 100000,
                'gasPrice': self.get_gas_price(),
                'nonce': nonce
            })

            # Sign and send approve transaction
            signed_approve_txn = self.web3.eth.account.sign_transaction(approve_txn, self.PRIVATE_KEY)
            approve_tx_hash = self.web3.eth.send_raw_transaction(signed_approve_txn.raw_transaction).hex()
            print(f"Approval Transaction Hash: https://bscscan.com/tx/0x{approve_tx_hash}")

            # Wait for the approval transaction to be mined
            self.web3.eth.wait_for_transaction_receipt(approve_tx_hash)

            # Increment the nonce
            nonce += 1

        # Send the swap transaction
        txn = self.pancake_router.functions.swapExactTokensForETH(
            balance_in_wei,
            amount_out_min,
            path,
            self.WALLET_ADDRESS,
            deadline
        ).build_transaction({
            'from': self.WALLET_ADDRESS,
            'gas': 300000,
            'gasPrice': self.get_gas_price(),
            'nonce': nonce
        })

        # Sign and send the swap transaction
        signed_txn = self.web3.eth.account.sign_transaction(txn, self.PRIVATE_KEY)
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction).hex()
        print(f"Swap Transaction Hash: https://bscscan.com/tx/0x{tx_hash}")

    def get_token_balance(self):
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        token_contract = self.web3.eth.contract(address=self.TOKEN_ADDRESS, abi=erc20_abi)
        balance = token_contract.functions.balanceOf(self.WALLET_ADDRESS).call()
        decimals = token_contract.functions.decimals().call()
        token_balance = balance / (10 ** decimals)
        return balance, token_balance

    def get_token_allowance(self):
        erc20_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "remaining", "type": "uint256"}],
                "type": "function"
            }
        ]
        token_contract = self.web3.eth.contract(address=self.TOKEN_ADDRESS, abi=erc20_abi)
        allowance = token_contract.functions.allowance(self.WALLET_ADDRESS, self.PANCAKE_ROUTER_ADDRESS).call()
        return allowance

