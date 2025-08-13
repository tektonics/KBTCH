# simulated_portfolio_manager.py
"""
Simulated Portfolio Manager for papertrading mode.
Tracks virtual positions, balance, and P&L without making API calls.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from config.config_manager import config
from event_bus import event_bus, EventTypes

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a specific market"""
    market_ticker: str
    side: str  # "yes" or "no"
    contracts: int  # Positive for long, negative for short
    avg_entry_price: float  # Average price paid per contract (in cents)
    total_cost: float  # Total amount paid (in cents)
    unrealized_pnl: float = 0.0  # Current unrealized P&L
    realized_pnl: float = 0.0  # Realized P&L from closed positions
    last_mark_price: float = 0.0  # Last market price for position valuation


@dataclass
class Fill:
    """Represents a simulated fill"""
    fill_id: str
    order_id: str
    market_ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    contracts: int
    price: float  # Fill price in cents
    timestamp: float
    commission: float = 0.0


@dataclass
class SimulatedOrder:
    """Represents an order in the simulated system"""
    order_id: str
    client_order_id: str
    market_ticker: str
    side: str
    action: str
    contracts: int
    price: float
    status: str  # "pending", "working", "filled", "cancelled", "rejected"
    filled_contracts: int = 0
    remaining_contracts: int = 0
    timestamp: float = field(default_factory=time.time)
    fills: List[Fill] = field(default_factory=list)


class SimulatedPortfolioManager:
    """Simulated portfolio manager for papertrading"""
    
    def __init__(self):
        self.papertrading_settings = config.get_papertrading_settings()
        
        # Portfolio state
        self.balance = self.papertrading_settings.initial_balance  # Balance in cents
        self.positions: Dict[str, Position] = {}  # market_ticker -> Position
        self.orders: Dict[str, SimulatedOrder] = {}  # order_id -> SimulatedOrder
        self.fills: List[Fill] = []
        
        # Statistics
        self.total_trades = 0
        self.total_commission_paid = 0.0
        self.total_realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.balance
        
        # Daily reset tracking
        self.last_reset_date = time.strftime("%Y-%m-%d")
        
        # Subscribe to market data for position valuation
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, self._handle_market_data_update)
        
        logger.info(f"💰 Simulated Portfolio Manager initialized with ${self.balance/100:,.2f}")
    
    def get_balance(self) -> Dict[str, Any]:
        """Get current balance (compatible with real portfolio manager interface)"""
        self._reset_daily_counters_if_needed()
        
        return {
            "balance": int(self.balance),  # Balance in cents
            "withdrawable_balance": int(self.balance),  # Simplified - all balance is withdrawable
            "account_type": "papertrading"
        }
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions (compatible with real portfolio manager interface)"""
        self._update_unrealized_pnl()
        
        market_positions = []
        for market_ticker, position in self.positions.items():
            if position.contracts != 0:  # Only include active positions
                market_positions.append({
                    "market_ticker": market_ticker,
                    "position": position.contracts,
                    "market_exposure": abs(position.contracts * position.avg_entry_price),
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                    "total_pnl": position.realized_pnl + position.unrealized_pnl,
                    "avg_entry_price": position.avg_entry_price,
                    "side": position.side,
                    "last_mark_price": position.last_mark_price
                })
        
        return {
            "market_positions": market_positions,
            "total_unrealized_pnl": sum(pos.unrealized_pnl for pos in self.positions.values()),
            "total_realized_pnl": self.total_realized_pnl
        }
    
    def get_fills(self) -> Dict[str, Any]:
        """Get fill history (compatible with real portfolio manager interface)"""
        fills_data = []
        for fill in self.fills[-50:]:  # Return last 50 fills
            fills_data.append({
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "market_ticker": fill.market_ticker,
                "side": fill.side,
                "action": fill.action,
                "count": fill.contracts,
                "price": int(fill.price),
                "created_time": fill.timestamp,
                "is_taker": True,  # Simplified
                "commission": int(fill.commission)
            })
        
        return {
            "fills": fills_data,
            "cursor": str(int(time.time()))  # Simple cursor implementation
        }
    
    def get_orders(self) -> Dict[str, Any]:
        """Get order history (compatible with real portfolio manager interface)"""
        orders_data = []
        for order in list(self.orders.values())[-50:]:  # Return last 50 orders
            orders_data.append({
                "order_id": order.order_id,
                "client_order_id": order.client_order_id,
                "market_ticker": order.market_ticker,
                "side": order.side,
                "action": order.action,
                "count": order.contracts,
                "price": int(order.price),
                "status": order.status,
                "filled_count": order.filled_contracts,
                "remaining_count": order.remaining_contracts,
                "created_time": order.timestamp,
                "type": "limit"
            })
        
        return {
            "orders": orders_data,
            "cursor": str(int(time.time()))
        }
    
    def process_simulated_fill(self, order_id: str, fill_price: float, fill_contracts: int) -> bool:
        """Process a simulated fill for an order"""
        try:
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found for fill processing")
                return False
            
            order = self.orders[order_id]
            
            if order.status != "working":
                logger.warning(f"Cannot fill order {order_id} with status {order.status}")
                return False
            
            if fill_contracts > order.remaining_contracts:
                fill_contracts = order.remaining_contracts
            
            # Create fill record
            fill = Fill(
                fill_id=f"fill_{int(time.time() * 1000)}_{len(self.fills)}",
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                action=order.action,
                contracts=fill_contracts,
                price=fill_price,
                timestamp=time.time(),
                commission=self.papertrading_settings.commission_per_contract * fill_contracts
            )
            
            # Update order status
            order.filled_contracts += fill_contracts
            order.remaining_contracts -= fill_contracts
            order.fills.append(fill)
            
            if order.remaining_contracts == 0:
                order.status = "filled"
            else:
                order.status = "partially_filled"
            
            # Process the fill for portfolio accounting
            self._process_fill_accounting(fill)
            
            # Store fill
            self.fills.append(fill)
            
            # Update statistics
            self.total_trades += 1
            self.total_commission_paid += fill.commission
            
            logger.info(f"💹 Simulated fill: {fill.action.upper()} {fill_contracts} {fill.side.upper()} "
                       f"{fill.market_ticker} @ {fill_price}¢")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing simulated fill: {e}")
            return False
    
    def _process_fill_accounting(self, fill: Fill) -> None:
        """Update positions and balance based on fill"""
        market_ticker = fill.market_ticker
        
        # Initialize position if it doesn't exist
        if market_ticker not in self.positions:
            self.positions[market_ticker] = Position(
                market_ticker=market_ticker,
                side=fill.side,
                contracts=0,
                avg_entry_price=0.0,
                total_cost=0.0
            )
        
        position = self.positions[market_ticker]
        fill_value = fill.contracts * fill.price
        
        if fill.action == "buy":
            # Calculate new average entry price
            if position.contracts >= 0:  # Adding to long position
                new_total_cost = position.total_cost + fill_value
                new_contracts = position.contracts + fill.contracts
                position.avg_entry_price = new_total_cost / new_contracts if new_contracts > 0 else 0
                position.total_cost = new_total_cost
                position.contracts = new_contracts
            else:  # Covering short position
                contracts_to_cover = min(fill.contracts, abs(position.contracts))
                remaining_fill_contracts = fill.contracts - contracts_to_cover
                
                # Realize P&L on covered contracts
                realized_pnl = contracts_to_cover * (position.avg_entry_price - fill.price)
                position.realized_pnl += realized_pnl
                self.total_realized_pnl += realized_pnl
                self.daily_pnl += realized_pnl
                
                # Update position
                position.contracts += contracts_to_cover
                position.total_cost -= contracts_to_cover * position.avg_entry_price
                
                # Handle remaining contracts if any
                if remaining_fill_contracts > 0:
                    position.contracts += remaining_fill_contracts
                    position.total_cost += remaining_fill_contracts * fill.price
                    if position.contracts > 0:
                        position.avg_entry_price = position.total_cost / position.contracts
            
            # Reduce balance by fill value and commission
            self.balance -= (fill_value + fill.commission)
            
        else:  # sell
            if position.contracts > 0:  # Closing long position
                contracts_to_sell = min(fill.contracts, position.contracts)
                remaining_fill_contracts = fill.contracts - contracts_to_sell
                
                # Realize P&L on sold contracts
                realized_pnl = contracts_to_sell * (fill.price - position.avg_entry_price)
                position.realized_pnl += realized_pnl
                self.total_realized_pnl += realized_pnl
                self.daily_pnl += realized_pnl
                
                # Update position
                position.contracts -= contracts_to_sell
                position.total_cost -= contracts_to_sell * position.avg_entry_price
                
                # Handle remaining contracts if any (going short)
                if remaining_fill_contracts > 0:
                    position.contracts -= remaining_fill_contracts
                    position.total_cost += remaining_fill_contracts * fill.price
                    if position.contracts < 0:
                        position.avg_entry_price = position.total_cost / abs(position.contracts)
            
            else:  # Adding to short position or creating new short
                new_total_cost = position.total_cost + fill_value
                new_contracts = position.contracts - fill.contracts
                position.avg_entry_price = new_total_cost / abs(new_contracts) if new_contracts != 0 else 0
                position.total_cost = new_total_cost
                position.contracts = new_contracts
            
            # Increase balance by fill value minus commission
            self.balance += (fill_value - fill.commission)
        
        # Update position side
        if position.contracts > 0:
            position.side = fill.side
        elif position.contracts < 0:
            position.side = "no" if fill.side == "yes" else "yes"
        
        # Update drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
    
    def _handle_market_data_update(self, event) -> None:
        """Update position valuations based on market data"""
        try:
            data = event.data
            market_ticker = data.get("market_ticker")
            
            if market_ticker in self.positions:
                position = self.positions[market_ticker]
                
                # Use appropriate market price based on position side
                if position.side == "yes":
                    mark_price = data.get("yes_bid", 0) if position.contracts > 0 else data.get("yes_ask", 0)
                else:
                    mark_price = data.get("no_bid", 0) if position.contracts > 0 else data.get("no_ask", 0)
                
                if mark_price and mark_price > 0:
                    position.last_mark_price = mark_price
                    
        except Exception as e:
            logger.error(f"Error updating position valuations: {e}")
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions"""
        for position in self.positions.values():
            if position.contracts != 0 and position.last_mark_price > 0:
                if position.contracts > 0:  # Long position
                    position.unrealized_pnl = position.contracts * (position.last_mark_price - position.avg_entry_price)
                else:  # Short position
                    position.unrealized_pnl = abs(position.contracts) * (position.avg_entry_price - position.last_mark_price)
    
    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day"""
        current_date = time.strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            logger.info(f"📅 New trading day: resetting daily P&L counter")
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def add_simulated_order(self, order: SimulatedOrder) -> None:
        """Add a new simulated order to tracking"""
        self.orders[order.order_id] = order
        logger.debug(f"📋 Added simulated order: {order.order_id}")
    
    def cancel_simulated_order(self, order_id: str) -> bool:
        """Cancel a simulated order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in ["pending", "working"]:
                order.status = "cancelled"
                logger.info(f"❌ Cancelled simulated order: {order_id}")
                return True
        return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary for papertrading"""
        self._update_unrealized_pnl()
        self._reset_daily_counters_if_needed()
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_portfolio_value = self.balance + total_unrealized_pnl
        
        return {
            "balance": self.balance,
            "total_portfolio_value": total_portfolio_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_commission_paid": self.total_commission_paid,
            "active_positions": len([p for p in self.positions.values() if p.contracts != 0]),
            "return_pct": ((total_portfolio_value - self.papertrading_settings.initial_balance) / 
                          self.papertrading_settings.initial_balance * 100),
            "is_papertrading": True
        }
