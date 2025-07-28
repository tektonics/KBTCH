"""
Risk management and order validation
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from trading.portfolio import Portfolio
from config.config import TRADING_CONFIG

@dataclass
class OrderSignal:
    market_ticker: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    order_type: str = 'limit'  # 'limit' or 'market'
    reason: str = ''  # Strategy reason for the order

@dataclass
class RiskCheckResult:
    approved: bool
    modified_quantity: int
    reasons: List[str]

class RiskManager:
    def __init__(self, config: Dict = None):
        self.config = config or TRADING_CONFIG['risk_limits']
        self.max_position_size = self.config['max_position_size']
        self.max_daily_loss = self.config['max_daily_loss']
        self.max_portfolio_value = self.config['max_portfolio_value']
        self.max_single_market_exposure = self.config['max_single_market_exposure']
        
    def validate_orders(self, orders: List[OrderSignal], portfolio: Portfolio) -> List[OrderSignal]:
        """Validate and potentially modify orders based on risk limits"""
        validated_orders = []
        
        for order in orders:
            risk_check = self._check_order_risk(order, portfolio)
            
            if risk_check.approved:
                if risk_check.modified_quantity != order.quantity:
                    # Create modified order
                    modified_order = OrderSignal(
                        market_ticker=order.market_ticker,
                        side=order.side,
                        quantity=risk_check.modified_quantity,
                        price=order.price,
                        order_type=order.order_type,
                        reason=f"{order.reason} (risk-adjusted from {order.quantity})"
                    )
                    validated_orders.append(modified_order)
                else:
                    validated_orders.append(order)
            else:
                print(f"Order rejected for {order.market_ticker}: {', '.join(risk_check.reasons)}")
        
        return validated_orders
    
    def _check_order_risk(self, order: OrderSignal, portfolio: Portfolio) -> RiskCheckResult:
        """Perform comprehensive risk checks on a single order"""
        reasons = []
        approved = True
        modified_quantity = order.quantity
        
        # 1. Cash availability check
        if order.side == 'buy':
            required_cash = order.quantity * order.price
            if not portfolio.can_afford(order.quantity, order.price):
                available_quantity = int(portfolio.cash / order.price)
                if available_quantity > 0:
                    modified_quantity = available_quantity
                    reasons.append(f"Reduced quantity due to insufficient cash (had ${portfolio.cash:.2f}, need ${required_cash:.2f})")
                else:
                    approved = False
                    reasons.append("Insufficient cash for any quantity")
        
        # 2. Position size limits
        current_position = portfolio.get_position(order.market_ticker)
        current_quantity = current_position.quantity if current_position else 0
        
        if order.side == 'buy':
            new_position_size = current_quantity + modified_quantity
        else:
            new_position_size = current_quantity - modified_quantity
        
        if abs(new_position_size) > self.max_position_size:
            # Adjust quantity to stay within limits
            if order.side == 'buy':
                max_additional = self.max_position_size - current_quantity
                if max_additional > 0:
                    modified_quantity = min(modified_quantity, max_additional)
                    reasons.append(f"Reduced buy quantity to stay within position limit")
                else:
                    approved = False
                    reasons.append("Already at maximum position size")
            else:
                # For sells, we can always reduce positions
                max_reduction = current_quantity + self.max_position_size
                modified_quantity = min(modified_quantity, max_reduction)
        
        # 3. Portfolio exposure check
        order_value = modified_quantity * order.price
        portfolio_value = portfolio.get_portfolio_value()
        max_market_value = portfolio_value * self.max_single_market_exposure
        
        current_exposure = portfolio.get_market_exposure(order.market_ticker)
        if order.side == 'buy':
            new_exposure = current_exposure + order_value
            if new_exposure > max_market_value:
                max_additional_value = max_market_value - current_exposure
                if max_additional_value > 0:
                    max_additional_quantity = int(max_additional_value / order.price)
                    if max_additional_quantity < modified_quantity:
                        modified_quantity = max_additional_quantity
                        reasons.append(f"Reduced quantity due to market exposure limit")
                else:
                    approved = False
                    reasons.append("Market exposure limit reached")
        
        # 4. Daily loss limit check
        daily_pnl = portfolio.get_daily_pnl()
        if daily_pnl <= -self.max_daily_loss:
            # Only allow closing positions if daily loss limit hit
            if order.side == 'buy' or (order.side == 'sell' and current_quantity <= 0):
                approved = False
                reasons.append("Daily loss limit reached, only position-closing trades allowed")
        
        # 5. Minimum quantity check
        if modified_quantity <= 0:
            approved = False
            reasons.append("Quantity reduced to zero or negative")
        
        # 6. Position direction check (prevent excessive position flipping)
        if current_position and current_quantity != 0:
            if order.side == 'buy' and current_quantity < 0:
                # Buying to cover short - limit to position size
                max_cover = abs(current_quantity)
                if modified_quantity > max_cover:
                    modified_quantity = max_cover
                    reasons.append("Limited buy quantity to cover short position")
            elif order.side == 'sell' and current_quantity > 0:
                # Selling long position - can sell up to current position
                max_sell = current_quantity
                if modified_quantity > max_sell:
                    modified_quantity = max_sell
                    reasons.append("Limited sell quantity to current long position")
        
        return RiskCheckResult(approved, modified_quantity, reasons)
    
    def check_portfolio_risk(self, portfolio: Portfolio) -> Dict[str, any]:
        """Perform portfolio-level risk assessment"""
        portfolio_value = portfolio.get_portfolio_value()
        total_exposure = portfolio.get_total_exposure()
        daily_pnl = portfolio.get_daily_pnl()
        unrealized_pnl = portfolio.get_unrealized_pnl()
        
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'daily_pnl': daily_pnl,
            'unrealized_pnl': unrealized_pnl,
            'cash_ratio': portfolio.cash / portfolio_value if portfolio_value > 0 else 1,
            'risk_warnings': []
        }
        
        # Check for risk warnings
        if daily_pnl <= -self.max_daily_loss * 0.8:
            risk_metrics['risk_warnings'].append("Approaching daily loss limit")
        
        if risk_metrics['exposure_ratio'] > 0.8:
            risk_metrics['risk_warnings'].append("High portfolio exposure")
        
        if risk_metrics['cash_ratio'] < 0.1:
            risk_metrics['risk_warnings'].append("Low cash reserves")
        
        if portfolio_value <= self.max_portfolio_value * 0.5:
            risk_metrics['risk_warnings'].append("Portfolio value significantly down")
        
        return risk_metrics
    
    def emergency_close_check(self, portfolio: Portfolio) -> bool:
        """Check if emergency position closing is required"""
        daily_pnl = portfolio.get_daily_pnl()
        portfolio_value = portfolio.get_portfolio_value()
        
        # Emergency close conditions
        if daily_pnl <= -self.max_daily_loss:
            return True
        
        if portfolio_value <= self.max_portfolio_value * 0.3:
            return True
        
        return False
    
    def get_max_order_size(self, market_ticker: str, side: str, price: float, portfolio: Portfolio) -> int:
        """Calculate maximum allowed order size for a market"""
        current_position = portfolio.get_position(market_ticker)
        current_quantity = current_position.quantity if current_position else 0
        
        if side == 'buy':
            # Cash limit
            cash_limit = int(portfolio.cash / price)
            
            # Position size limit
            position_limit = self.max_position_size - current_quantity
            
            # Exposure limit
            portfolio_value = portfolio.get_portfolio_value()
            max_market_value = portfolio_value * self.max_single_market_exposure
            current_exposure = portfolio.get_market_exposure(market_ticker)
            exposure_limit = int((max_market_value - current_exposure) / price)
            
            return max(0, min(cash_limit, position_limit, exposure_limit))
        
        else:  # sell
            # Can't sell more than we have (for long positions)
            if current_quantity > 0:
                return current_quantity
            
            # For short selling, use position limits
            short_limit = self.max_position_size + current_quantity  # current_quantity is negative
            return max(0, short_limit)
