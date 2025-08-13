import time
import uuid
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from event_bus import event_bus, EventTypes
from kalshi_base import KalshiAPIClient

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    WORKING = "working" 
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class Order:
    client_order_id: str
    market_ticker: str
    action: str  # "buy" or "sell"
    side: str    # "yes" or "no"
    count: int
    price: int
    order_type: str = "limit"
    status: OrderStatus = OrderStatus.PENDING
    filled_count: int = 0
    remaining_count: int = 0
    timestamp: float = 0.0
    kalshi_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.remaining_count == 0:
            self.remaining_count = self.count


class OrderManager(KalshiAPIClient):
    """Manages order lifecycle and tracking"""
    
    def __init__(self):
        super().__init__()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: Dict[str, Order] = {}
        
        # Statistics
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.orders_rejected = 0
        
        # Subscribe to risk-approved signals
        event_bus.subscribe(EventTypes.RISK_APPROVED, self._handle_risk_approved_signal)
        
        logger.info("Order Manager initialized and subscribed to risk-approved signals")
    
    def _handle_risk_approved_signal(self, event) -> None:
        """Handle risk-approved trading signals"""
        try:
            signal_data = event.data
            
            # Convert signal to order
            order = self._convert_signal_to_order(signal_data)
            if order:
                # Track the order
                self.active_orders[order.client_order_id] = order
                
                # Publish order placed event
                self._publish_order_event(EventTypes.ORDER_PLACED, order, signal_data)
                
                self.orders_placed += 1
                logger.info(f"📋 Order tracked: {order.action.upper()} {order.count} {order.side.upper()} "
                          f"{order.market_ticker} @ {order.price}¢ (ID: {order.client_order_id[:8]})")
            
        except Exception as e:
            logger.error(f"Error handling risk-approved signal: {e}")
    
    def _convert_signal_to_order(self, signal_data: Dict[str, Any]) -> Optional[Order]:
        """Convert trading signal to order format"""
        try:
            signal_type = signal_data.get("signal_type")
            market_ticker = signal_data.get("market_ticker") 
            quantity = signal_data.get("quantity", 1)
            
            if not all([signal_type, market_ticker]):
                logger.error("Missing required signal data")
                return None
            
            # Parse signal type
            if signal_type == "BUY_YES":
                action = "buy"
                side = "yes"
                price = int(signal_data.get("market_yes_price", 0))
            elif signal_type == "BUY_NO":
                action = "buy" 
                side = "no"
                price = int(signal_data.get("market_no_price", 0))
            elif signal_type == "SELL_YES":
                action = "sell"
                side = "yes"
                price = int(signal_data.get("market_yes_price", 0))
            elif signal_type == "SELL_NO":
                action = "sell"
                side = "no" 
                price = int(signal_data.get("market_no_price", 0))
            else:
                logger.error(f"Unknown signal type: {signal_type}")
                return None
            
            if price <= 0:
                logger.error(f"Invalid price: {price}")
                return None
            
            # Generate unique client order ID
            client_order_id = str(uuid.uuid4())
            
            return Order(
                client_order_id=client_order_id,
                market_ticker=market_ticker,
                action=action,
                side=side,
                count=quantity,
                price=price
            )
            
        except Exception as e:
            logger.error(f"Error converting signal to order: {e}")
            return None
    
    def update_order_status(self, client_order_id: str, status: OrderStatus, 
                           filled_count: int = 0, kalshi_order_id: Optional[str] = None) -> bool:
        """Update order status and tracking"""
        try:
            order = self.active_orders.get(client_order_id)
            if not order:
                logger.warning(f"Order not found for update: {client_order_id}")
                return False
            
            old_status = order.status
            order.status = status
            order.filled_count = filled_count
            order.remaining_count = order.count - filled_count
            
            if kalshi_order_id:
                order.kalshi_order_id = kalshi_order_id
            
            # Update statistics
            if status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
                self.orders_filled += 1
            elif status == OrderStatus.CANCELLED and old_status != OrderStatus.CANCELLED:
                self.orders_cancelled += 1
            elif status == OrderStatus.REJECTED and old_status != OrderStatus.REJECTED:
                self.orders_rejected += 1
            
            # Move to history if terminal status
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.order_history[client_order_id] = order
                del self.active_orders[client_order_id]
            
            # Publish status update event
            event_type = {
                OrderStatus.FILLED: EventTypes.ORDER_FILLED,
                OrderStatus.CANCELLED: EventTypes.ORDER_CANCELLED,
                OrderStatus.REJECTED: EventTypes.ORDER_REJECTED,
                OrderStatus.WORKING: EventTypes.ORDER_UPDATED,
                OrderStatus.PARTIALLY_FILLED: EventTypes.ORDER_UPDATED
            }.get(status, EventTypes.ORDER_UPDATED)
            
            self._publish_order_event(event_type, order)
            
            logger.info(f"📋 Order {client_order_id[:8]} status: {old_status.value} → {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID"""
        return self.active_orders.get(client_order_id) or self.order_history.get(client_order_id)
    
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_orders_for_market(self, market_ticker: str) -> Dict[str, Order]:
        """Get all orders for a specific market"""
        return {
            order_id: order for order_id, order in self.active_orders.items()
            if order.market_ticker == market_ticker
        }
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Mark order for cancellation (execution manager will handle the API call)"""
        try:
            order = self.active_orders.get(client_order_id)
            if not order:
                logger.warning(f"Cannot cancel order - not found: {client_order_id}")
                return False
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Cannot cancel order in status {order.status.value}: {client_order_id}")
                return False
            
            # Update status to cancelled (execution manager should confirm)
            return self.update_order_status(client_order_id, OrderStatus.CANCELLED)
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def _publish_order_event(self, event_type: str, order: Order, signal_data: Dict[str, Any] = None) -> None:
        """Publish order-related events"""
        try:
            event_data = {
                "client_order_id": order.client_order_id,
                "kalshi_order_id": order.kalshi_order_id,
                "market_ticker": order.market_ticker,
                "action": order.action,
                "side": order.side,
                "count": order.count,
                "price": order.price,
                "status": order.status.value,
                "filled_count": order.filled_count,
                "remaining_count": order.remaining_count,
                "timestamp": order.timestamp
            }
            
            # Include original signal data for ORDER_PLACED events
            if signal_data:
                event_data["original_signal"] = signal_data
            
            event_bus.publish(event_type, event_data, source="order_manager")
            
        except Exception as e:
            logger.error(f"Error publishing order event: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status"""
        return {
            "active_orders": len(self.active_orders),
            "total_orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "orders_rejected": self.orders_rejected,
            "order_history_size": len(self.order_history),
            "active_order_breakdown": {
                status.value: len([o for o in self.active_orders.values() if o.status == status])
                for status in OrderStatus
            }
        }
    
    def to_kalshi_api_format(self, order: Order) -> Dict[str, Any]:
        """Convert order to Kalshi API format for execution manager"""
        api_order = {
            "action": order.action,
            "client_order_id": order.client_order_id,
            "count": order.count,
            "side": order.side,
            "ticker": order.market_ticker,
            "type": order.order_type
        }
        
        # Add appropriate price field
        if order.side == "yes":
            api_order["yes_price"] = order.price
        else:
            api_order["no_price"] = order.price
        
        return api_order
