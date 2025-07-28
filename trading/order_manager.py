# order_manager.py - Enhanced order execution and management for Kalshi integration
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import random
import logging
from trading.risk_manager import OrderSignal
from config.config import TRADING_CONFIG

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


class OrderManager:
    
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
    
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
        results = []
        
        for order in orders:
            logger.info(f"Executing {order.side.upper()} {order.quantity} {order.market_ticker} @ {order.price:.3f} - {order.reason}")
            
            result = self.trading_engine.execute_order(order)
            results.append(result)
            
            self.order_stats['total_orders'] += 1
            if result.status == 'rejected':
                self.order_stats['rejected_orders'] += 1
            else:
                self.order_stats['successful_orders'] += 1
            
            if result.status in ['pending', 'partial']:
                self.pending_orders[result.order_id] = result
            else:
                self.completed_orders.append(result)
            
            status_emoji = "✅" if result.status in ['filled', 'pending'] else "❌"
            logger.info(f"{status_emoji} Order {result.order_id}: {result.status} - filled {result.filled_quantity}/{result.quantity}")
        
        return results
    
    def update_pending_orders(self):
        completed_order_ids = []
        
        for order_id, order in self.pending_orders.items():
            updated_order = self.trading_engine.get_order_status(order_id)
            
            if updated_order and updated_order.status in ['filled', 'cancelled', 'rejected']:
                self.completed_orders.append(updated_order)
                completed_order_ids.append(order_id)
                
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
            if self.trading_engine.cancel_order(order_id):
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
