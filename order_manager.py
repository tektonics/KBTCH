"""
Order execution and management
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import random
from risk_manager import OrderSignal
from config import TRADING_CONFIG

@dataclass
class OrderResult:
    order_id: str
    market_ticker: str
    side: str
    quantity: int
    filled_quantity: int
    price: float
    filled_price: float
    status: str  # 'filled', 'partial', 'pending', 'cancelled', 'rejected'
    timestamp: datetime
    error_message: str = ''

class OrderExecutor(ABC):
    """Abstract base class for order execution"""
    
    @abstractmethod
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Execute a single order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get status of an existing order"""
        pass

class SimulatedOrderExecutor(OrderExecutor):
    """Simulated order execution for backtesting and paper trading"""
    
    def __init__(self):
        self.order_counter = 0
        self.orders: Dict[str, OrderResult] = {}
        self.fill_probability = 0.95  # 95% fill rate
        self.partial_fill_probability = 0.1  # 10% chance of partial fill
        self.price_slippage = 0.001  # 0.1% price slippage
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Simulate order execution with realistic behavior"""
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        
        # Simulate order processing delay
        time.sleep(0.01)
        
        # Determine if order fills
        fill_random = random.random()
        
        if fill_random > self.fill_probability:
            # Order rejected or failed
            result = OrderResult(
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                price=order.price,
                filled_price=0.0,
                status='rejected',
                timestamp=datetime.now(),
                error_message='Market conditions unfavorable'
            )
        else:
            # Determine fill quantity
            if random.random() < self.partial_fill_probability:
                # Partial fill
                filled_quantity = random.randint(1, order.quantity - 1)
                status = 'partial'
            else:
                # Full fill
                filled_quantity = order.quantity
                status = 'filled'
            
            # Apply price slippage
            slippage_direction = 1 if order.side == 'buy' else -1
            slippage_factor = 1 + (slippage_direction * self.price_slippage * random.random())
            filled_price = order.price * slippage_factor
            
            result = OrderResult(
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=filled_quantity,
                price=order.price,
                filled_price=filled_price,
                status=status,
                timestamp=datetime.now()
            )
        
        self.orders[order_id] = result
        return result
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in ['pending', 'partial']:
                order.status = 'cancelled'
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get simulated order status"""
        return self.orders.get(order_id)

class LiveOrderExecutor(OrderExecutor):
    """Live order execution using Kalshi API"""
    
    def __init__(self, kalshi_client):
        self.client = kalshi_client
        self.orders: Dict[str, OrderResult] = {}
        self.rate_limit_delay = TRADING_CONFIG['api']['rate_limit_delay']
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Execute order via Kalshi API"""
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Prepare order payload
            order_payload = {
                'ticker': order.market_ticker,
                'client_order_id': f"auto_{int(time.time() * 1000)}",
                'side': order.side,
                'action': 'buy' if order.side == 'buy' else 'sell',
                'count': order.quantity,
                'type': 'limit',
                'yes_price': int(order.price * 100) if order.side == 'buy' else None,
                'no_price': int((1 - order.price) * 100) if order.side == 'sell' else None,
            }
            
            # Execute order via API
            response = self.client.create_order(**order_payload)
            
            if response.get('status') == 'success':
                order_data = response.get('order', {})
                result = OrderResult(
                    order_id=order_data.get('order_id'),
                    market_ticker=order.market_ticker,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=order_data.get('remaining_count', 0),
                    price=order.price,
                    filled_price=order_data.get('yes_price', 0) / 100.0,
                    status='pending',
                    timestamp=datetime.now()
                )
            else:
                result = OrderResult(
                    order_id='',
                    market_ticker=order.market_ticker,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=0,
                    price=order.price,
                    filled_price=0.0,
                    status='rejected',
                    timestamp=datetime.now(),
                    error_message=response.get('error', 'Unknown error')
                )
            
            if result.order_id:
                self.orders[result.order_id] = result
            
            return result
            
        except Exception as e:
            return OrderResult(
                order_id='',
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                price=order.price,
                filled_price=0.0,
                status='rejected',
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            response = self.client.cancel_order(order_id)
            return response.get('status') == 'success'
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get order status via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            response = self.client.get_order(order_id)
            
            if response.get('status') == 'success':
                order_data = response.get('order', {})
                
                # Update stored order with latest data
                if order_id in self.orders:
                    stored_order = self.orders[order_id]
                    stored_order.filled_quantity = order_data.get('remaining_count', 0)
                    stored_order.status = order_data.get('status', 'unknown')
                    return stored_order
            
        except Exception as e:
            print(f"Error getting order status {order_id}: {e}")
        
        return self.orders.get(order_id)

class OrderManager:
    """High-level order management with execution abstraction"""
    
    def __init__(self, mode: str = 'simulation', kalshi_client=None):
        self.mode = mode
        
        if mode == 'live':
            if kalshi_client is None:
                raise ValueError("kalshi_client required for live trading")
            self.executor = LiveOrderExecutor(kalshi_client)
        else:
            self.executor = SimulatedOrderExecutor()
        
        self.pending_orders: Dict[str, OrderResult] = {}
        self.completed_orders: List[OrderResult] = []
    
    def execute_orders(self, orders: List[OrderSignal]) -> List[OrderResult]:
        """Execute a batch of orders"""
        results = []
        
        for order in orders:
            print(f"Executing {order.side} {order.quantity} {order.market_ticker} @ {order.price:.3f} - {order.reason}")
            
            result = self.executor.execute_order(order)
            results.append(result)
            
            if result.status in ['pending', 'partial']:
                self.pending_orders[result.order_id] = result
            else:
                self.completed_orders.append(result)
            
            print(f"Order {result.order_id}: {result.status} - filled {result.filled_quantity}/{result.quantity}")
        
        return results
    
    def update_pending_orders(self):
        """Update status of pending orders"""
        completed_order_ids = []
        
        for order_id, order in self.pending_orders.items():
            updated_order = self.executor.get_order_status(order_id)
            
            if updated_order and updated_order.status in ['filled', 'cancelled', 'rejected']:
                self.completed_orders.append(updated_order)
                completed_order_ids.append(order_id)
            elif updated_order:
                self.pending_orders[order_id] = updated_order
        
        # Remove completed orders from pending
        for order_id in completed_order_ids:
            del self.pending_orders[order_id]
    
    def cancel_all_pending_orders(self) -> int:
        """Cancel all pending orders"""
        cancelled_count = 0
        
        for order_id in list(self.pending_orders.keys()):
            if self.executor.cancel_order(order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    def get_order_summary(self) -> Dict:
        """Get summary of all orders"""
        return {
            'pending_orders': len(self.pending_orders),
            'completed_orders': len(self.completed_orders),
            'total_filled': sum(1 for order in self.completed_orders if order.status == 'filled'),
            'total_rejected': sum(1 for order in self.completed_orders if order.status == 'rejected'),
            'pending_order_ids': list(self.pending_orders.keys())
        }
    
    def get_fills(self) -> List[OrderResult]:
        """Get all filled orders"""
        return [order for order in self.completed_orders if order.status == 'filled']
