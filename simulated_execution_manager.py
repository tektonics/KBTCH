# simulated_execution_manager.py
"""
Simulated Execution Manager for papertrading mode.
Simulates realistic order execution with fills, slippage, and market impact.
"""

import time
import uuid
import random
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from order_manager import OrderStatus
from event_bus import event_bus, EventTypes
from config.config_manager import config
from simulated_portfolio_manager import SimulatedOrder

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Current market data for order simulation"""
    market_ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    timestamp: float
    spread: float = 0.0
    
    def __post_init__(self):
        self.spread = self.yes_ask - self.yes_bid


class SimulatedExecutionManager:
    """Simulates realistic order execution for papertrading"""
    
    def __init__(self, order_manager, portfolio_manager):
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.papertrading_settings = config.get_papertrading_settings()
        
        # Market data storage for execution simulation
        self.market_data: Dict[str, MarketData] = {}
        
        # Execution settings
        self.fill_delay_range = (0.1, 1.5)  # Random delay between orders
        self.partial_fill_probability = 0.15  # 15% chance of partial fills
        self.price_improvement_probability = self.papertrading_settings.price_improvement_probability
        
        # Statistics
        self.orders_processed = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.total_slippage = 0.0
        
        # Pending fill tasks
        self.pending_fills: Dict[str, asyncio.Task] = {}
        
        # Subscribe to events
        event_bus.subscribe(EventTypes.ORDER_PLACED, self._handle_order_placed)
        event_bus.subscribe(EventTypes.ORDER_CANCELLED, self._handle_order_cancelled)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, self._handle_market_data_update)
        
        logger.info("🎯 Simulated Execution Manager initialized for papertrading")
    
    def _handle_order_placed(self, event) -> None:
        """Handle order placement in simulation mode"""
        try:
            event_data = event.data
            client_order_id = event_data.get("client_order_id")
            
            if not client_order_id:
                logger.error("No client_order_id in order placed event")
                return
            
            # Get order from order manager
            order = self.order_manager.get_order(client_order_id)
            if not order:
                logger.error(f"Order not found in order manager: {client_order_id}")
                return
            
            # Create simulated order
            simulated_order = self._create_simulated_order(order)
            
            # Add to portfolio manager tracking
            self.portfolio_manager.add_simulated_order(simulated_order)
            
            # Update order status to working immediately
            kalshi_order_id = simulated_order.order_id
            self.order_manager.update_order_status(
                client_order_id, 
                OrderStatus.WORKING,
                kalshi_order_id=kalshi_order_id
            )
            
            # Schedule fill simulation
            self._schedule_fill_simulation(simulated_order)
            
            self.orders_processed += 1
            logger.info(f"📋 Simulated order working: {client_order_id[:8]} → Sim ID: {kalshi_order_id[:8]}")
            
        except Exception as e:
            logger.error(f"Error handling order placement in simulation: {e}")
    
    def _handle_order_cancelled(self, event) -> None:
        """Handle order cancellation in simulation mode"""
        try:
            event_data = event.data
            client_order_id = event_data.get("client_order_id")
            kalshi_order_id = event_data.get("kalshi_order_id")
            
            if not kalshi_order_id:
                logger.warning(f"Cannot cancel simulated order without order ID: {client_order_id}")
                return
            
            # Cancel pending fill task if exists
            if kalshi_order_id in self.pending_fills:
                self.pending_fills[kalshi_order_id].cancel()
                del self.pending_fills[kalshi_order_id]
            
            # Cancel in portfolio manager
            success = self.portfolio_manager.cancel_simulated_order(kalshi_order_id)
            
            if success:
                logger.info(f"❌ Simulated order cancelled: {client_order_id[:8]}")
            else:
                logger.error(f"Failed to cancel simulated order: {client_order_id[:8]}")
                
        except Exception as e:
            logger.error(f"Error handling order cancellation in simulation: {e}")
    
    def _handle_market_data_update(self, event) -> None:
        """Update market data for execution simulation"""
        try:
            data = event.data
            market_ticker = data.get("market_ticker")
            
            if not market_ticker:
                return
            
            # Store market data for order execution
            self.market_data[market_ticker] = MarketData(
                market_ticker=market_ticker,
                yes_bid=data.get("yes_bid", 0),
                yes_ask=data.get("yes_ask", 0),
                no_bid=data.get("no_bid", 0),
                no_ask=data.get("no_ask", 0),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error updating market data for simulation: {e}")
    
    def _create_simulated_order(self, order) -> SimulatedOrder:
        """Create a simulated order from a real order"""
        return SimulatedOrder(
            order_id=f"sim_{uuid.uuid4().hex[:8]}",
            client_order_id=order.client_order_id,
            market_ticker=order.market_ticker,
            side=order.side,
            action=order.action,
            contracts=order.count,
            price=order.price,
            status="working",
            remaining_contracts=order.count
        )
    
    def _schedule_fill_simulation(self, simulated_order: SimulatedOrder) -> None:
        """Schedule a simulated fill with realistic timing"""
        task = asyncio.create_task(self._simulate_order_fill(simulated_order))
        self.pending_fills[simulated_order.order_id] = task
    
    async def _simulate_order_fill(self, simulated_order: SimulatedOrder) -> None:
        """Simulate order fill with realistic market behavior"""
        try:
            # Wait for realistic fill delay
            fill_delay = random.uniform(*self.fill_delay_range)
            await asyncio.sleep(fill_delay)
            
            # Check if order was cancelled during delay
            if simulated_order.order_id not in self.pending_fills:
                return
            
            # Check if we should fill this order
            if not self._should_fill_order(simulated_order):
                await self._reject_order(simulated_order, "Market conditions unfavorable")
                return
            
            # Determine fill details
            fill_price, fill_contracts = self._calculate_fill_details(simulated_order)
            
            if fill_contracts == 0:
                await self._reject_order(simulated_order, "No liquidity at price level")
                return
            
            # Process the fill
            success = self.portfolio_manager.process_simulated_fill(
                simulated_order.order_id, fill_price, fill_contracts
            )
            
            if success:
                # Update order manager
                client_order_id = simulated_order.client_order_id
                
                if fill_contracts == simulated_order.remaining_contracts:
                    # Fully filled
                    self.order_manager.update_order_status(
                        client_order_id, OrderStatus.FILLED, fill_contracts
                    )
                    self.orders_filled += 1
                    logger.info(f"✅ Simulated order filled: {client_order_id[:8]} - "
                               f"{fill_contracts} contracts @ {fill_price}¢")
                else:
                    # Partially filled - schedule another fill attempt
                    self.order_manager.update_order_status(
                        client_order_id, OrderStatus.PARTIALLY_FILLED, fill_contracts
                    )
                    simulated_order.remaining_contracts -= fill_contracts
                    await asyncio.sleep(0.5)  # Brief pause before next fill attempt
                    await self._simulate_order_fill(simulated_order)
            else:
                await self._reject_order(simulated_order, "Fill processing failed")
                
        except asyncio.CancelledError:
            logger.debug(f"Fill simulation cancelled for order {simulated_order.order_id}")
        except Exception as e:
            logger.error(f"Error in fill simulation: {e}")
            await self._reject_order(simulated_order, f"Simulation error: {e}")
        finally:
            # Clean up pending fill task
            if simulated_order.order_id in self.pending_fills:
                del self.pending_fills[simulated_order.order_id]
    
    def _should_fill_order(self, simulated_order: SimulatedOrder) -> bool:
        """Determine if an order should be filled based on market conditions"""
        # Basic probability check
        if random.random() > self.papertrading_settings.fill_probability:
            return False
        
        # Check if we have current market data
        market_data = self.market_data.get(simulated_order.market_ticker)
        if not market_data:
            return False
        
        # Check if order price is within reasonable range of current market
        if simulated_order.side == "yes":
            current_bid = market_data.yes_bid
            current_ask = market_data.yes_ask
        else:
            current_bid = market_data.no_bid
            current_ask = market_data.no_ask
        
        order_price = simulated_order.price
        
        if simulated_order.action == "buy":
            # Buy orders need to be at or above the ask to fill immediately
            # Allow some flexibility for fills within the spread
            return order_price >= current_ask * 0.98  # 2% tolerance
        else:
            # Sell orders need to be at or below the bid to fill immediately
            return order_price <= current_bid * 1.02  # 2% tolerance
    
    def _calculate_fill_details(self, simulated_order: SimulatedOrder) -> Tuple[float, int]:
        """Calculate fill price and quantity with slippage and market impact"""
        market_data = self.market_data.get(simulated_order.market_ticker)
        if not market_data:
            return 0.0, 0
        
        # Determine base fill price
        if simulated_order.side == "yes":
            if simulated_order.action == "buy":
                base_price = market_data.yes_ask
            else:
                base_price = market_data.yes_bid
        else:
            if simulated_order.action == "buy":
                base_price = market_data.no_ask
            else:
                base_price = market_data.no_bid
        
        # Apply slippage
        slippage_bps = self.papertrading_settings.slippage_bps
        slippage_amount = base_price * (slippage_bps / 10000)
        
        if simulated_order.action == "buy":
            fill_price = base_price + slippage_amount  # Pay more when buying
        else:
            fill_price = base_price - slippage_amount  # Receive less when selling
        
        # Apply price improvement occasionally
        if random.random() < self.price_improvement_probability:
            improvement = base_price * 0.001  # 0.1% improvement
            if simulated_order.action == "buy":
                fill_price -= improvement  # Pay less when buying
            else:
                fill_price += improvement  # Receive more when selling
        
        # Ensure fill price doesn't cross the spread inappropriately
        fill_price = max(1, min(99, fill_price))  # Keep within 1-99¢ range
        
        # Determine fill quantity
        fill_contracts = simulated_order.remaining_contracts
        
        # Occasionally do partial fills for large orders
        if (fill_contracts > 3 and 
            random.random() < self.partial_fill_probability):
            # Fill 30-80% of remaining quantity
            fill_pct = random.uniform(0.3, 0.8)
            fill_contracts = max(1, int(fill_contracts * fill_pct))
        
        # Track slippage
        price_diff = abs(fill_price - simulated_order.price)
        self.total_slippage += price_diff * fill_contracts
        
        return fill_price, fill_contracts
    
    async def _reject_order(self, simulated_order: SimulatedOrder, reason: str) -> None:
        """Reject a simulated order"""
        try:
            client_order_id = simulated_order.client_order_id
            self.order_manager.update_order_status(client_order_id, OrderStatus.REJECTED)
            self.orders_rejected += 1
            logger.warning(f"❌ Simulated order rejected: {client_order_id[:8]} - {reason}")
        except Exception as e:
            logger.error(f"Error rejecting simulated order: {e}")
    
    def check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Check simulated order status (compatible with real execution manager)"""
        # In simulation mode, we can return basic status
        return {
            "order": {
                "order_id": order_id,
                "status": "working",  # Simplified status
                "filled_count": 0,
                "remaining_count": 0
            }
        }
    
    def amend_order(self, order_id: str, new_price: int) -> Tuple[bool, Dict[str, Any]]:
        """Amend simulated order (simplified implementation)"""
        logger.info(f"🔄 Simulated order amend request: {order_id} to {new_price}¢")
        return True, {"status": "amended"}
    
    def batch_create_orders(self, orders: list[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """Batch create simulated orders (not implemented in simulation)"""
        logger.warning("Batch order creation not supported in simulation mode")
        return False, {"error": "Not supported in simulation mode"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution manager status for simulation"""
        success_rate = 0.0
        if self.orders_processed > 0:
            success_rate = (self.orders_filled / self.orders_processed) * 100
        
        avg_slippage = 0.0
        if self.orders_filled > 0:
            avg_slippage = self.total_slippage / self.orders_filled
        
        return {
            "orders_processed": self.orders_processed,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "fill_success_rate": f"{success_rate:.1f}%",
            "pending_fills": len(self.pending_fills),
            "avg_slippage_cents": f"{avg_slippage:.2f}",
            "markets_tracked": len(self.market_data),
            "mode": "papertrading",
            "settings": {
                "fill_probability": self.papertrading_settings.fill_probability,
                "slippage_bps": self.papertrading_settings.slippage_bps,
                "price_improvement_prob": self.price_improvement_probability
            }
        }
