"""
Simple event bus for inter-component communication.
Provides publish/subscribe functionality for the trading system.
"""

import time
import threading
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Basic event structure"""
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    source: Optional[str] = None


class EventBus:
    """Simple publish/subscribe event bus"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_count = 0
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}, {len(self._subscribers[event_type])} total subscribers")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
            
        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type}")
                    return True
                except ValueError:
                    logger.warning(f"Callback not found in {event_type} subscribers")
                    return False
            return False
    
    def publish(self, event_type: str, data: Dict[str, Any], source: Optional[str] = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event being published
            data: Event data
            source: Optional source component name
        """
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            source=source
        )
        
        with self._lock:
            self._event_count += 1
            subscribers = self._subscribers[event_type].copy()
        
        # Call subscribers outside the lock to avoid blocking
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        with self._lock:
            return len(self._subscribers[event_type])
    
    def get_total_events_published(self) -> int:
        """Get total number of events published"""
        with self._lock:
            return self._event_count
    
    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """
        Clear subscribers for a specific event type or all event types.
        
        Args:
            event_type: Specific event type to clear, or None for all
        """
        with self._lock:
            if event_type:
                self._subscribers[event_type].clear()
                logger.debug(f"Cleared all subscribers for {event_type}")
            else:
                self._subscribers.clear()
                logger.debug("Cleared all subscribers")


# Global event bus instance
event_bus = EventBus()


# Common event types (can be extended as needed)
class EventTypes:
    """Standard event types for the trading system"""
    
    # Market data events
    MARKET_DATA_UPDATE = "market_data_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    PRICE_UPDATE = "price_update"
    
    # Trading signals
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXPIRED = "signal_expired"
    
    # Order events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_UPDATED = "order_updated"
    
    # Risk events
    RISK_VIOLATION = "risk_violation"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    LOSS_LIMIT_REACHED = "loss_limit_reached"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    COMPONENT_ERROR = "component_error"
    HEALTH_CHECK = "health_check"
