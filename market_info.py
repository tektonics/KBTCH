# order_manager.py - Enhanced order execution and management for Kalshi integration
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import random
import logging
from risk_manager import OrderSignal
from config import TRADING_CONFIG

logger = logging.getLogger(__name__)

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
    contract_type: str = 'YES'  # 'YES' or 'NO' for Kalshi

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
    """Enhanced simulated order execution with realistic Kalshi behavior"""
    
    def __init__(self):
        self.order_counter = 0
        self.orders: Dict[str, OrderResult] = {}
        self.fill_probability = 0.90  # 90% fill rate
        self.partial_fill_probability = 0.15  # 15% chance of partial fill
        self.price_slippage = 0.002  # 0.2% price slippage
        self.latency_simulation = True
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Simulate order execution with realistic Kalshi behavior"""
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        
        # Simulate network latency
        if self.latency_simulation:
            time.sleep(random.uniform(0.01, 0.05))
        
        # Determine contract type and price from order reason
        contract_type, kalshi_price = self._determine_kalshi_order_details(order)
        
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
                error_message='Market conditions unfavorable',
                contract_type=contract_type
            )
        else:
            # Determine fill quantity
            if random.random() < self.partial_fill_probability:
                # Partial fill
                filled_quantity = random.randint(max(1, order.quantity // 4), order.quantity - 1)
                status = 'partial'
            else:
                # Full fill
                filled_quantity = order.quantity
                status = 'filled'
            
            # Apply realistic price slippage for Kalshi
            slippage_direction = 1 if order.side == 'buy' else -1
            slippage_factor = 1 + (slippage_direction * self.price_slippage * random.random())
            filled_price = kalshi_price * slippage_factor
            
            # Ensure price stays within Kalshi bounds (0-1)
            filled_price = max(0.01, min(0.99, filled_price))
            
            result = OrderResult(
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=filled_quantity,
                price=order.price,
                filled_price=filled_price,
                status=status,
                timestamp=datetime.now(),
                contract_type=contract_type
            )
        
        self.orders[order_id] = result
        return result
    
    def _determine_kalshi_order_details(self, order: OrderSignal) -> tuple:
        """Determine Kalshi contract type and price from order details"""
        if 'YES' in order.reason:
            return 'YES', order.price
        elif 'NO' in order.reason:
            return 'NO', order.price
        else:
            # Default to YES contract
            return 'YES', order.price
    
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

class KalshiOrderExecutor(OrderExecutor):
    """Live order execution using Kalshi API with enhanced error handling"""
    
    def __init__(self, kalshi_client):
        self.client = kalshi_client
        self.orders: Dict[str, OrderResult] = {}
        self.rate_limit_delay = TRADING_CONFIG['api']['rate_limit_delay']
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Execute order via Kalshi API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Determine contract type and prepare order
                contract_type, kalshi_order = self._prepare_kalshi_order(order)
                
                # Execute order via API
                response = self._execute_kalshi_api_call(kalshi_order)
                
                if response.get('status') == 'success':
                    order_data = response.get('order', {})
                    result = self._process_successful_order(order, order_data, contract_type)
                else:
                    result = self._process_failed_order(order, response, contract_type)
                
                if result.order_id:
                    self.orders[result.order_id] = result
                
                return result
                
            except Exception as e:
                logger.warning(f"Order execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return self._create_error_result(order, str(e))
                
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        return self._create_error_result(order, "Max retries exceeded")
    
    def _prepare_kalshi_order(self, order: OrderSignal) -> tuple:
        """Prepare Kalshi-specific order format"""
        # Determine if this is a YES or NO contract order
        contract_type = 'YES'  # Default
        if 'NO' in order.reason.upper():
            contract_type = 'NO'
        
        # Convert price to Kalshi cents format (0-100)
        kalshi_price = int(order.price * 100)
        
        kalshi_order = {
            'ticker': order.market_ticker,
            'client_order_id': f"auto_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            'side': order.side,
            'action': 'buy' if order.side == 'buy' else 'sell',
            'count': order.quantity,
            'type': 'limit',
        }
        
        # Set price based on contract type
        if contract_type == 'YES':
            if order.side == 'buy':
                kalshi_order['yes_price'] = kalshi_price
            else:
                kalshi_order['yes_price'] = kalshi_price
        else:  # NO contract
            if order.side == 'buy':
                kalshi_order['no_price'] = kalshi_price
            else:
                kalshi_order['no_price'] = kalshi_price
        
        return contract_type, kalshi_order
    
    def _execute_kalshi_api_call(self, kalshi_order: Dict) -> Dict:
        """Execute the actual Kalshi API call"""
        # This would be the actual Kalshi API call
        # For now, return a mock response structure
        return {
            'status': 'success',
            'order': {
                'order_id': f"KALSHI_{int(time.time() * 1000)}",
                'status': 'pending',
                'remaining_count': kalshi_order['count'],
                'yes_price': kalshi_order.get('yes_price', 0),
                'no_price': kalshi_order.get('no_price', 0)
            }
        }
    
    def _process_successful_order(self, order: OrderSignal, order_data: Dict, contract_type: str) -> OrderResult:
        """Process successful order response"""
        return OrderResult(
            order_id=order_data.get('order_id'),
            market_ticker=order.market_ticker,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,  # Will be updated when fills occur
            price=order.price,
            filled_price=0.0,  # Will be updated when fills occur
            status='pending',
            timestamp=datetime.now(),
            contract_type=contract_type
        )
    
    def _process_failed_order(self, order: OrderSignal, response: Dict, contract_type: str) -> OrderResult:
        """Process failed order response"""
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
            error_message=response.get('error', 'Unknown error'),
            contract_type=contract_type
        )
    
    def _create_error_result(self, order: OrderSignal, error_message: str) -> OrderResult:
        """Create error result for failed orders"""
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
            error_message=error_message,
            contract_type='YES'
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            # This would be the actual Kalshi cancel API call
            # response = self.client.cancel_order(order_id)
            # return response.get('status') == 'success'
            
            # For now, simulate cancellation
            if order_id in self.orders:
                self.orders[order_id].status = 'cancelled'
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get order status via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            # This would be the actual Kalshi status API call
            # response = self.client.get_order(order_id)
            
            # For now, return stored order
            return self.orders.get(order_id)
            
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return self.orders.get(order_id)

class OrderManager:
    """Enhanced order management with Kalshi-specific features"""
    
    def __init__(self, mode: str = 'simulation', kalshi_client=None):
        self.mode = mode
        
        if mode == 'live':
            if kalshi_client is None:
                raise ValueError("kalshi_client required for live trading")
            self.executor = KalshiOrderExecutor(kalshi_client)
        else:
            self.executor = SimulatedOrderExecutor()
        
        self.pending_orders: Dict[str, OrderResult] = {}
        self.completed_orders: List[OrderResult] = []
        self.fills_cache: List[OrderResult] = []
        
        # Enhanced tracking
        self.order_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'total_fills': 0,
            'total_volume': 0.0
        }
    
    def execute_orders(self, orders: List[OrderSignal]) -> List[OrderResult]:
        """Execute a batch of orders with enhanced logging"""
        results = []
        
        for order in orders:
            logger.info(f"Executing {order.side.upper()} {order.quantity} {order.market_ticker} @ {order.price:.3f} - {order.reason}")
            
            result = self.executor.execute_order(order)
            results.append(result)
            
            # Update statistics
            self.order_stats['total_orders'] += 1
            if result.status == 'rejected':
                self.order_stats['rejected_orders'] += 1
            else:
                self.order_stats['successful_orders'] += 1
            
            # Manage order state
            if result.status in ['pending', 'partial']:
                self.pending_orders[result.order_id] = result
            else:
                self.completed_orders.append(result)
            
            # Log result
            status_emoji = "✅" if result.status in ['filled', 'pending'] else "❌"
            logger.info(f"{status_emoji} Order {result.order_id}: {result.status} - filled {result.filled_quantity}/{result.quantity}")
        
        return results
    
    def update_pending_orders(self):
        """Update status of pending orders with enhanced tracking"""
        completed_order_ids = []
        
        for order_id, order in self.pending_orders.items():
            updated_order = self.executor.get_order_status(order_id)
            
            if updated_order and updated_order.status in ['filled', 'cancelled', 'rejected']:
                self.completed_orders.append(updated_order)
                completed_order_ids.append(order_id)
                
                # Update statistics for completed orders
                if updated_order.status == 'filled':
                    self.order_stats['total_fills'] += 1
                    self.order_stats['total_volume'] += updated_order.filled_quantity * updated_order.filled_price
                elif updated_order.status == 'cancelled':
                    self.order_stats['cancelled_orders'] += 1
                
                # Add to fills cache if filled
                if updated_order.filled_quantity > 0:
                    self.fills_cache.append(updated_order)
                
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
                logger.info(f"Cancelled order {order_id}")
        
        return cancelled_count
    
    def get_order_summary(self) -> Dict:
        """Get comprehensive order summary"""
        return {
            'pending_orders': len(self.pending_orders),
            'completed_orders': len(self.completed_orders),
            'pending_order_ids': list(self.pending_orders.keys()),
            'statistics': self.order_stats.copy(),
            'success_rate': (self.order_stats['successful_orders'] / 
                           max(self.order_stats['total_orders'], 1) * 100),
            'fill_rate': (self.order_stats['total_fills'] / 
                         max(self.order_stats['successful_orders'], 1) * 100)
        }
    
    def get_fills(self) -> List[OrderResult]:
        """Get all filled orders"""
        return [order for order in self.completed_orders if order.status == 'filled']
    
    def get_recent_fills(self, count: int = 10) -> List[OrderResult]:
        """Get recent fills"""
        filled_orders = [order for order in self.completed_orders if order.status == 'filled']
        return sorted(filled_orders, key=lambda x: x.timestamp, reverse=True)[:count]
    
    def get_order_history(self, market_ticker: str = None) -> List[OrderResult]:
        """Get order history, optionally filtered by market"""
        orders = self.completed_orders
        if market_ticker:
            orders = [order for order in orders if order.market_ticker == market_ticker]
        return sorted(orders, key=lambda x: x.timestamp, reverse=True)
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        filled_orders = self.get_fills()
        
        if not filled_orders:
            return {'total_fills': 0, 'avg_fill_time': 0, 'avg_slippage': 0}
        
        # Calculate average slippage
        slippages = []
        for order in filled_orders:
            if order.price > 0:
                slippage = (order.filled_price - order.price) / order.price
                slippages.append(abs(slippage))
        
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        return {
            'total_fills': len(filled_orders),
            'total_volume': self.order_stats['total_volume'],
            'avg_slippage': avg_slippage,
            'yes_contracts': len([o for o in filled_orders if o.contract_type == 'YES']),
            'no_contracts': len([o for o in filled_orders if o.contract_type == 'NO']),
            'buy_orders': len([o for o in filled_orders if o.side == 'buy']),
            'sell_orders': len([o for o in filled_orders if o.side == 'sell'])
        }
