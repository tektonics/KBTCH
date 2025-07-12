"""
Risk management and order validation with Kalshi contract rules
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from portfolio import Portfolio
from config import TRADING_CONFIG

@dataclass
class OrderSignal:
    market_ticker: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    order_type: str = 'limit'  # 'limit' or 'market'
    reason: str = ''  # Strategy reason for the order
    contract_type: str = 'YES'  # 'YES' or 'NO' for Kalshi

@dataclass
class RiskCheckResult:
    approved: bool
    modified_quantity: int
    reasons: List[str]
    contract_type: str = 'YES'

class RiskManager:
    """Risk manager with Kalshi contract rules awareness"""
    
    def __init__(self, config: Dict = None):
        self.config = config or TRADING_CONFIG['risk_limits']
        self.max_position_size = self.config['max_position_size']
        self.max_daily_loss = self.config['max_daily_loss']
        self.max_portfolio_value = self.config['max_portfolio_value']
        self.max_single_market_exposure = self.config['max_single_market_exposure']
        
    def validate_orders(self, orders: List[OrderSignal], portfolio: Portfolio) -> List[OrderSignal]:
        """Validate and potentially modify orders based on risk limits and Kalshi rules"""
        validated_orders = []
        
        for order in orders:
            # First check Kalshi contract rules
            kalshi_check = self._check_kalshi_contract_rules(order, portfolio)
            if not kalshi_check.approved:
                print(f"Order rejected for {order.market_ticker}: {', '.join(kalshi_check.reasons)}")
                continue
            
            # Then check risk limits
            risk_check = self._check_order_risk(order, portfolio)
            
            if risk_check.approved:
                # Use the most restrictive quantity
                final_quantity = min(kalshi_check.modified_quantity, risk_check.modified_quantity)
                
                if final_quantity != order.quantity:
                    # Create modified order
                    modified_order = OrderSignal(
                        market_ticker=order.market_ticker,
                        side=order.side,
                        quantity=final_quantity,
                        price=order.price,
                        order_type=order.order_type,
                        reason=f"{order.reason} (adjusted from {order.quantity})",
                        contract_type=getattr(order, 'contract_type', 'YES')
                    )
                    validated_orders.append(modified_order)
                else:
                    validated_orders.append(order)
            else:
                combined_reasons = kalshi_check.reasons + risk_check.reasons
                print(f"Order rejected for {order.market_ticker}: {', '.join(combined_reasons)}")
        
        return validated_orders
    
    def _check_kalshi_contract_rules(self, order: OrderSignal, portfolio: Portfolio) -> RiskCheckResult:
        """
        Check Kalshi-specific contract rules
        
        Rules:
        - Can always BUY YES or BUY NO
        - Can only SELL YES if you own YES contracts 
        - Can only SELL NO if you own NO contracts
        """
        reasons = []
        approved = True
        modified_quantity = order.quantity
        
        # Get current position
        current_position = portfolio.get_position(order.market_ticker)
        current_quantity = current_position.quantity if current_position else 0
        
        if order.side == 'buy':
            # Buying is always allowed under Kalshi rules
            return RiskCheckResult(True, modified_quantity, [], getattr(order, 'contract_type', 'YES'))
        
        elif order.side == 'sell':
            # Selling requires ownership validation
            contract_type = getattr(order, 'contract_type', 'YES')
            
            if contract_type == 'YES':
                # Selling YES contracts - need to own YES (positive quantity)
                if current_quantity <= 0:
                    approved = False
                    reasons.append(f"Cannot sell YES contracts - no YES position owned (current: {current_quantity})")
                else:
                    # Limit sell to what we actually own
                    max_sellable = current_quantity
                    if order.quantity > max_sellable:
                        modified_quantity = max_sellable
                        reasons.append(f"Reduced sell quantity to owned YES contracts ({max_sellable})")
            
            elif contract_type == 'NO':
                # Selling NO contracts - need to own NO (negative quantity in our simplified model)
                if current_quantity >= 0:
                    approved = False
                    reasons.append(f"Cannot sell NO contracts - no NO position owned (current: {current_quantity})")
                else:
                    # Limit sell to what we actually own (absolute value of negative position)
                    max_sellable = abs(current_quantity)
                    if order.quantity > max_sellable:
                        modified_quantity = max_sellable
                        reasons.append(f"Reduced sell quantity to owned NO contracts ({max_sellable})")
            
            else:
                approved = False
                reasons.append(f"Unknown contract type: {contract_type}")
        
        return RiskCheckResult(approved, modified_quantity, reasons, getattr(order, 'contract_type', 'YES'))
    
    def _check_order_risk(self, order: OrderSignal, portfolio: Portfolio) -> RiskCheckResult:
        """Perform comprehensive risk checks on a single order"""
        reasons = []
        approved = True
        modified_quantity = order.quantity
        
        # 1. Cash availability check (for buys only)
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
            # For Kalshi, we need to consider contract type for position limits
            contract_type = getattr(order, 'contract_type', 'YES')
            
            if contract_type == 'YES':
                # Buying YES increases positive position
                new_position_size = max(0, current_quantity) + modified_quantity
            else:  # NO
                # Buying NO increases negative position (or reduces positive)
                if current_quantity > 0:
                    # Have YES, buying NO reduces YES position first
                    net_change = min(modified_quantity, current_quantity)
                    new_position_size = current_quantity - net_change
                else:
                    # Already have NO or neutral, increase NO position
                    new_position_size = abs(current_quantity) + modified_quantity
        else:
            # Selling always reduces position size
            new_position_size = abs(current_quantity) - modified_quantity
        
        if new_position_size > self.max_position_size:
            # Adjust quantity to stay within limits
            if order.side == 'buy':
                max_additional = self.max_position_size - abs(current_quantity)
                if max_additional > 0:
                    modified_quantity = min(modified_quantity, max_additional)
                    reasons.append(f"Reduced buy quantity to stay within position limit")
                else:
                    approved = False
                    reasons.append("Already at maximum position size")
        
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
            # Only allow position-closing trades if daily loss limit hit
            if order.side == 'buy':
                # Buying when at loss limit - only allow if it closes a short position
                if current_quantity >= 0:  # Not closing a short
                    approved = False
                    reasons.append("Daily loss limit reached, only position-closing trades allowed")
            # Selling is generally allowed as it reduces exposure
        
        # 5. Minimum quantity check
        if modified_quantity <= 0:
            approved = False
            reasons.append("Quantity reduced to zero or negative")
        
        # 6. Contract type validation for position direction
        if hasattr(order, 'contract_type'):
            contract_type = order.contract_type
            if order.side == 'buy':
                # Additional validation could go here
                pass
        
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
    
    def get_max_order_size(self, market_ticker: str, side: str, price: float, 
                          portfolio: Portfolio, contract_type: str = 'YES') -> int:
        """Calculate maximum allowed order size for a market with contract type awareness"""
        current_position = portfolio.get_position(market_ticker)
        current_quantity = current_position.quantity if current_position else 0
        
        if side == 'buy':
            # Cash limit
            cash_limit = int(portfolio.cash / price)
            
            # Position size limit (depends on contract type)
            if contract_type == 'YES':
                if current_quantity >= 0:
                    # Buying more YES or starting YES position
                    position_limit = self.max_position_size - current_quantity
                else:
                    # Have NO position, buying YES reduces NO first
                    position_limit = abs(current_quantity) + self.max_position_size
            else:  # NO
                if current_quantity <= 0:
                    # Buying more NO or starting NO position  
                    position_limit = self.max_position_size - abs(current_quantity)
                else:
                    # Have YES position, buying NO reduces YES first
                    position_limit = current_quantity + self.max_position_size
            
            # Exposure limit
            portfolio_value = portfolio.get_portfolio_value()
            max_market_value = portfolio_value * self.max_single_market_exposure
            current_exposure = portfolio.get_market_exposure(market_ticker)
            exposure_limit = int((max_market_value - current_exposure) / price)
            
            return max(0, min(cash_limit, position_limit, exposure_limit))
        
        else:  # sell
            # Can only sell what we own
            if contract_type == 'YES':
                return max(0, current_quantity)
            else:  # NO
                return max(0, abs(current_quantity)) if current_quantity < 0 else 0
