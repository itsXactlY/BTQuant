"""
Broker Interface Module

Abstract base class and concrete implementations for broker integrations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class Order:
    """Data class representing a trading order"""
    order_id: str
    symbol: str
    order_type: str  # 'market', 'limit', 'stop', etc.
    quantity: float
    price: Optional[float] = None
    side: str = 'buy'  # 'buy' or 'sell'
    status: str = 'pending'  # 'pending', 'filled', 'cancelled', 'rejected'
    timestamp: Optional[str] = None
    strategy_id: Optional[str] = None

@dataclass
class Position:
    """Data class representing an open position"""
    position_id: str
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    strategy_id: str
    timestamp: str

@dataclass
class AccountBalance:
    """Data class representing account balance"""
    total_balance: float
    available_balance: float
    margin_used: float
    margin_available: float
    currency: str = 'USD'

class BrokerInterface(ABC):
    """Abstract base class for broker integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"BrokerInterface.{self.__class__.__name__}")
        self.connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker API"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker API"""
        pass
        
    @abstractmethod
    def get_account_balance(self) -> AccountBalance:
        """Get current account balance"""
        pass
        
    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Get list of open positions"""
        pass
        
    @abstractmethod
    def place_order(self, order: Order) -> Dict[str, Any]:
        """Place a new order"""
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        pass
        
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        pass
        
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str = '1m') -> Dict[str, Any]:
        """Get market data for a symbol"""
        pass
        
    def is_connected(self) -> bool:
        """Check if broker is connected"""
        return self.connected

class SimulatedBroker(BrokerInterface):
    """Simulated broker for testing and development"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.simulated_balance = config.get('initial_balance', 100000.0)
        self.simulated_positions = []
        self.simulated_orders = {}
        self.simulated_market_data = {}
        
    def connect(self) -> bool:
        """Connect to simulated broker"""
        self.logger.info("Connecting to simulated broker")
        self.connected = True
        return True
        
    def disconnect(self) -> bool:
        """Disconnect from simulated broker"""
        self.logger.info("Disconnecting from simulated broker")
        self.connected = False
        return True
        
    def get_account_balance(self) -> AccountBalance:
        """Get simulated account balance"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        return AccountBalance(
            total_balance=self.simulated_balance,
            available_balance=self.simulated_balance,
            margin_used=0.0,
            margin_available=self.simulated_balance
        )
        
    def get_open_positions(self) -> List[Position]:
        """Get simulated open positions"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        return self.simulated_positions
        
    def place_order(self, order: Order) -> Dict[str, Any]:
        """Place a simulated order"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        order.order_id = f"sim_order_{len(self.simulated_orders) + 1}"
        order.status = 'filled'
        
        self.simulated_orders[order.order_id] = order
        
        # Simulate position update
        if order.order_type == 'market':
            # Simulate market execution
            execution_price = self._get_simulated_price(order.symbol)
            
            # Update balance
            order_value = execution_price * order.quantity
            if order.side == 'buy':
                self.simulated_balance -= order_value
            else:
                self.simulated_balance += order_value
                
            # Update or create position
            existing_position = None
            for pos in self.simulated_positions:
                if pos.symbol == order.symbol and pos.strategy_id == order.strategy_id:
                    existing_position = pos
                    break
                    
            if existing_position:
                existing_position.quantity += order.quantity if order.side == 'buy' else -order.quantity
                existing_position.current_price = execution_price
                existing_position.unrealized_pnl = (execution_price - existing_position.entry_price) * existing_position.quantity
            else:
                new_position = Position(
                    position_id=f"sim_pos_{len(self.simulated_positions) + 1}",
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    current_price=execution_price,
                    unrealized_pnl=0.0,
                    strategy_id=order.strategy_id or "unknown",
                    timestamp="2023-01-01T00:00:00Z"  # Simulated timestamp
                )
                self.simulated_positions.append(new_position)
        
        return {
            'order_id': order.order_id,
            'status': order.status,
            'execution_price': self._get_simulated_price(order.symbol),
            'timestamp': '2023-01-01T00:00:00Z'
        }
        
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        if order_id in self.simulated_orders:
            self.simulated_orders[order_id].status = 'cancelled'
            return {'order_id': order_id, 'status': 'cancelled'}
        else:
            return {'order_id': order_id, 'status': 'not_found'}
            
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a simulated order"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        if order_id in self.simulated_orders:
            order = self.simulated_orders[order_id]
            return {
                'order_id': order.order_id,
                'status': order.status,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'side': order.side
            }
        else:
            return {'order_id': order_id, 'status': 'not_found'}
            
    def get_market_data(self, symbol: str, timeframe: str = '1m') -> Dict[str, Any]:
        """Get simulated market data"""
        if not self.connected:
            raise ConnectionError("Broker not connected")
            
        # Generate simulated market data
        if symbol not in self.simulated_market_data:
            self.simulated_market_data[symbol] = {
                'symbol': symbol,
                'price': 100.0 + (hash(symbol) % 50),  # Simulated price
                'bid': 0.0,
                'ask': 0.0,
                'volume': 10000,
                'timestamp': '2023-01-01T00:00:00Z'
            }
            
        return self.simulated_market_data[symbol]
        
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for a symbol"""
        base_price = 100.0 + (hash(symbol) % 50)
        return base_price + (hash(symbol + str(len(self.simulated_orders))) % 10) - 5

class BrokerFactory:
    """Factory class for creating broker instances"""
    
    @staticmethod
    def create_broker(broker_type: str, config: Dict[str, Any]) -> BrokerInterface:
        """Create a broker instance based on type"""
        if broker_type == 'simulated':
            return SimulatedBroker(config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")