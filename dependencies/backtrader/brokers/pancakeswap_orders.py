import time
from web3 import Web3
from backtrader.dontcommit import bsc_privaccount1, bsc_privaccountaddress

class PancakeSwapV2DirectOrderBase:
    def __init__(self, coin, collateral, **kwargs):
        super().__init__(**kwargs)
        self.BSC_RPC_URL = "https://bscrpc.pancakeswap.finance"# "https://bsc-dataseed.binance.org/"
        self.PRIVATE_KEY = bsc_privaccount1
        self.WALLET_ADDRESS = Web3.to_checksum_address(bsc_privaccountaddress)
        self.PANCAKE_ROUTER_ADDRESS = Web3.to_checksum_address('0x10ED43C718714eb63d5aA57B78B54704E256024E') 
        
        # V2 ->('0x10ED43C718714eb63d5aA57B78B54704E256024E')
        # V3 ->('0x1A0A18AC4BECDDbd6389559687d1A73d8927E416')
        # V4 ->('0xd9C500DfF816a1Da21A48A732d3498Bf09dc9AEB')
        self.TOKEN_ADDRESS = Web3.to_checksum_address(coin)
        self.WBNB_ADDRESS = Web3.to_checksum_address(collateral)
        self.web3 = Web3(Web3.HTTPProvider(self.BSC_RPC_URL))
        if not self.web3.is_connected():
            raise Exception("Failed to connect to Binance Smart Chain")
        print(f"PancakeSwapV2DirectOrderBase connected to BSC: {self.web3.is_connected()}")

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
        print(gp := self.web3.eth.gas_price); return gp

    def get_collateral_balance(self):
        """Get the actual BNB balance from wallet"""
        try:
            balance_wei = self.web3.eth.get_balance(self.WALLET_ADDRESS)
            balance_bnb = self.web3.from_wei(balance_wei, 'ether')
            return float(balance_bnb)
        except Exception as e:
            print(f"Error getting BNB balance: {e}")
            return 0.0

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
            result = self.swap_all_tokens_for_bnb(nonce)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return {"success": False, "error": str(e)}

    def swap_bnb_for_token(self, bnb_amount, nonce):
        path = [self.WBNB_ADDRESS, self.TOKEN_ADDRESS]
        amount_in_wei = self.to_wei(bnb_amount)
        deadline = int(time.time()) + 60 * 10

        bnb_balance = self.web3.eth.get_balance(self.WALLET_ADDRESS)
        if bnb_balance < amount_in_wei:
            print("Insufficient BNB balance.")
            return {"success": False, "error": "Insufficient BNB balance"}

        try:
            amounts_out = self.pancake_router.functions.getAmountsOut(amount_in_wei, path).call()
            print(f"⚠️ getAmountsOut result: {amounts_out}")
            print(f"⚠️ Expected tokens (raw): {amounts_out[1]}")
            print(f"⚠️ BNB in: {bnb_amount}")
            amount_out_min = amounts_out[1] * 90 // 100

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

            signed_txn = self.web3.eth.account.sign_transaction(txn, self.PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            print(f"Transaction sent: https://bscscan.com/tx/{tx_hash_hex}")
            print("Waiting for confirmation...")
            
            # Wait for transaction receipt with timeout
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                print(f"✅ Transaction confirmed in block {receipt['blockNumber']}")
                
                # Calculate actual tokens received
                token_balance_after, _ = self.get_token_balance()
                
                return {
                    "success": True,
                    "tx_hash": tx_hash_hex,
                    "block": receipt['blockNumber'],
                    "gas_used": receipt['gasUsed'],
                    "bnb_spent": bnb_amount,
                    "tokens_received": token_balance_after,  # You'll need to track before/after
                    "receipt": receipt
                }
            else:
                print(f"❌ Transaction failed")
                return {
                    "success": False,
                    "error": "Transaction reverted",
                    "tx_hash": tx_hash_hex
                }
                
        except Exception as e:
            print(f"❌ Error executing swap: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def send_pcs_buy_request(self, amount):
        try:
            nonce = self.web3.eth.get_transaction_count(self.WALLET_ADDRESS)
            result = self.swap_bnb_for_token(amount, nonce)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return {"success": False, "error": str(e)}

    def swap_all_tokens_for_bnb(self, nonce):
        path = [self.TOKEN_ADDRESS, self.WBNB_ADDRESS]
        balance_in_wei, balance_human_readable = self.get_token_balance()
        print(f"Token balance: {balance_human_readable} tokens")

        if balance_in_wei == 0:
            print("No tokens available to swap.")
            return {"success": False, "error": "No tokens available to swap"}

        try:
            deadline = int(time.time()) + 60 * 10  # 10 minutes from now
            
            # Get expected output amounts
            amounts_out = self.pancake_router.functions.getAmountsOut(balance_in_wei, path).call()
            print(f"⚠️ getAmountsOut result: {amounts_out}")
            print(f"⚠️ Expected tokens (raw): {amounts_out[1]}")
            amount_out_min = amounts_out[1] * 90 // 100  # 10% slippage tolerance

            # Check and handle token allowance
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
                
                approve_txn = token_contract.functions.approve(
                    self.PANCAKE_ROUTER_ADDRESS, 
                    MAX_ALLOWANCE
                ).build_transaction({
                    'from': self.WALLET_ADDRESS,
                    'gas': 100000,
                    'gasPrice': self.get_gas_price(),
                    'nonce': nonce
                })

                # Sign and send approve transaction
                signed_approve_txn = self.web3.eth.account.sign_transaction(approve_txn, self.PRIVATE_KEY)
                approve_tx_hash = self.web3.eth.send_raw_transaction(signed_approve_txn.raw_transaction)
                approve_tx_hash_hex = approve_tx_hash.hex()
                print(f"Approval Transaction Hash: https://bscscan.com/tx/{approve_tx_hash_hex}")

                # Wait for the approval transaction to be mined
                print("Waiting for approval confirmation...")
                approval_receipt = self.web3.eth.wait_for_transaction_receipt(approve_tx_hash, timeout=120)
                
                if approval_receipt['status'] != 1:
                    print("❌ Approval transaction failed")
                    return {
                        "success": False, 
                        "error": "Approval transaction failed",
                        "tx_hash": approve_tx_hash_hex
                    }
                
                print(f"✅ Approval confirmed in block {approval_receipt['blockNumber']}")
                # Increment the nonce for the swap transaction
                nonce += 1

            # Get BNB balance before swap
            bnb_before = self.get_collateral_balance()
            
            # Build the swap transaction
            swap_txn = self.pancake_router.functions.swapExactTokensForETH(
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
            signed_txn = self.web3.eth.account.sign_transaction(swap_txn, self.PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            print(f"Swap Transaction sent: https://bscscan.com/tx/{tx_hash_hex}")
            
            # Wait for confirmation
            print("Waiting for swap confirmation...")
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                # Get BNB balance after swap
                bnb_after = self.get_collateral_balance()
                bnb_received = bnb_after - bnb_before
                
                print(f"✅ Swap confirmed in block {receipt['blockNumber']}")
                print(f"✅ Received {bnb_received:.8f} BNB")
                
                return {
                    "success": True,
                    "tx_hash": tx_hash_hex,
                    "block": receipt['blockNumber'],
                    "gas_used": receipt['gasUsed'],
                    "tokens_sold": balance_human_readable,
                    "bnb_received": bnb_received,
                    "bnb_balance": bnb_after,
                    "receipt": receipt
                }
            else:
                print(f"❌ Swap transaction failed")
                return {
                    "success": False,
                    "error": "Swap transaction reverted",
                    "tx_hash": tx_hash_hex,
                    "receipt": receipt
                }
                
        except Exception as e:
            print(f"❌ Error executing swap: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

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

