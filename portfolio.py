"""
Portfolio management and position tracking with Kalshi contract support
"""
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import json

@dataclass
class Position:
    market_ticker: str
    quantity: int  # Positive = YES contracts, Negative = NO contracts (simplified)
    avg_price: float
    current_price: float = 0.0
    timestamp: datetime = None
    contract_type: str = 'YES'  # 'YES' or 'NO'
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def market_value(self) -> float:
        """Current market value of the position"""
        return abs(self.quantity) * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L for the position"""
        return abs(self.quantity) * (self.current_price - self.avg_price)
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position"""
        return abs(self.quantity) * self.avg_price

@dataclass
class Trade:
    market_ticker: str
    quantity: int
    price: float
    side: str  # 'buy' or 'sell'
    contract_type: str = 'YES'  # 'YES' or 'NO'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Portfolio:
    """Portfolio with Kalshi contract awareness"""
    
    def __init__(self, initial_cash: float = 10000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}  # Key: market_ticker
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Kalshi-specific tracking
        self.yes_positions: Dict[str, Position] = {}
        self.no_positions: Dict[str, Position] = {}
        
    def add_trade(self, market_ticker: str, quantity: int, price: float, side: str, contract_type: str = 'YES'):
        """Add a new trade and update positions with contract type awareness"""
        trade = Trade(market_ticker, quantity, price, side, contract_type)
        self.trades.append(trade)
        
        # Update cash
        trade_value = quantity * price
        if side == 'buy':
            self.cash -= trade_value
        else:
            self.cash += trade_value
        
        # Update positions based on contract type
        self._update_position(market_ticker, quantity, price, side, contract_type)
        
        return trade
    
    def _update_position(self, market_ticker: str, quantity: int, price: float, side: str, contract_type: str):
        """Update position with new trade, handling YES/NO contracts"""
        
        # For simplified tracking, we'll use one position per market
        # Positive quantity = net YES contracts, Negative = net NO contracts
        
        if side == 'buy':
            if contract_type == 'YES':
                effective_quantity = quantity  # Positive for YES
            else:  # NO
                effective_quantity = -quantity  # Negative for NO
        else:  # sell
            if contract_type == 'YES':
                effective_quantity = -quantity  # Selling YES reduces YES position
            else:  # NO  
                effective_quantity = quantity   # Selling NO reduces NO position (adds to quantity)
        
        # Get or create position
        if market_ticker in self.positions:
            position = self.positions[market_ticker]
            
            # Calculate new position
            old_total_value = position.quantity * position.avg_price
            new_trade_value = effective_quantity * price
            new_total_quantity = position.quantity + effective_quantity
            
            if new_total_quantity == 0:
                # Position closed
                del self.positions[market_ticker]
                return
            
            # Calculate new average price
            new_total_value = old_total_value + new_trade_value
            new_avg_price = new_total_value / new_total_quantity
            
            # Update position
            position.quantity = new_total_quantity
            position.avg_price = abs(new_avg_price)  # Keep price positive
            position.contract_type = 'YES' if new_total_quantity > 0 else 'NO'
            position.timestamp = datetime.now()
            
        else:
            # New position
            if effective_quantity != 0:
                self.positions[market_ticker] = Position(
                    market_ticker=market_ticker,
                    quantity=effective_quantity,
                    avg_price=price,
                    contract_type='YES' if effective_quantity > 0 else 'NO'
                )
    
    def update_market_prices(self, price_data: Dict[str, float]):
        """Update current market prices for all positions"""
        for ticker, price in price_data.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
    
    def get_position(self, market_ticker: str) -> Position:
        """Get position for a specific market"""
        return self.positions.get(market_ticker)
    
    def get_yes_position_quantity(self, market_ticker: str) -> int:
        """Get quantity of YES contracts owned"""
        position = self.positions.get(market_ticker)
        if position and position.quantity > 0:
            return position.quantity
        return 0
    
    def get_no_position_quantity(self, market_ticker: str) -> int:
        """Get quantity of NO contracts owned"""
        position = self.positions.get(market_ticker)
        if position and position.quantity < 0:
            return abs(position.quantity)
        return 0
    
    def can_sell_yes(self, market_ticker: str, quantity: int) -> bool:
        """Check if we can sell YES contracts"""
        yes_quantity = self.get_yes_position_quantity(market_ticker)
        return yes_quantity >= quantity
    
    def can_sell_no(self, market_ticker: str, quantity: int) -> bool:
        """Check if we can sell NO contracts"""
        no_quantity = self.get_no_position_quantity(market_ticker)
        return no_quantity >= quantity
    
    def get_total_exposure(self) -> float:
        """Get total market exposure across all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_market_exposure(self, market_ticker: str) -> float:
        """Get exposure for a specific market"""
        position = self.positions.get(market_ticker)
        return position.market_value if position else 0.0
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get realized P&L from completed trades"""
        # Calculate from total portfolio value change minus unrealized P&L
        total_change = self.get_portfolio_value() - self.initial_cash
        unrealized_change = self.get_unrealized_pnl()
        return total_change - unrealized_change
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L (placeholder - would need to track daily marks)"""
        return self.daily_pnl
    
    def can_afford(self, quantity: int, price: float) -> bool:
        """Check if we have enough cash for a trade"""
        required_cash = quantity * price
        return self.cash >= required_cash
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions with contract type breakdown"""
        yes_positions = {}
        no_positions = {}
        
        for ticker, pos in self.positions.items():
            pos_info = {
                'quantity': abs(pos.quantity),
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'contract_type': pos.contract_type
            }
            
            if pos.quantity > 0:  # YES contracts
                yes_positions[ticker] = pos_info
            else:  # NO contracts
                no_positions[ticker] = pos_info
        
        return {
            'cash': self.cash,
            'total_value': self.get_portfolio_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'positions': {ticker: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'contract_type': pos.contract_type
            } for ticker, pos in self.positions.items()},
            'yes_positions': yes_positions,
            'no_positions': no_positions,
            'total_yes_contracts': sum(pos.quantity for pos in self.positions.values() if pos.quantity > 0),
            'total_no_contracts': sum(abs(pos.quantity) for pos in self.positions.values() if pos.quantity < 0)
        }
    
    def get_contract_breakdown(self) -> Dict:
        """Get breakdown of YES vs NO contract holdings"""
        yes_value = 0.0
        no_value = 0.0
        yes_count = 0
        no_count = 0
        
        for pos in self.positions.values():
            if pos.quantity > 0:  # YES contracts
                yes_value += pos.market_value
                yes_count += pos.quantity
            else:  # NO contracts
                no_value += pos.market_value
                no_count += abs(pos.quantity)
        
        return {
            'yes_contracts': {
                'count': yes_count,
                'value': yes_value,
                'markets': len([p for p in self.positions.values() if p.quantity > 0])
            },
            'no_contracts': {
                'count': no_count,
                'value': no_value,
                'markets': len([p for p in self.positions.values() if p.quantity < 0])
            },
            'total_contracts': yes_count + no_count,
            'total_value': yes_value + no_value
        }
    
    def save_state(self, filename: str):
        """Save portfolio state to file with contract type info"""
        state = {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'positions': [
                {
                    'market_ticker': pos.market_ticker,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'contract_type': pos.contract_type,
                    'timestamp': pos.timestamp.isoformat()
                } for pos in self.positions.values()
            ],
            'trades': [
                {
                    'market_ticker': trade.market_ticker,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'side': trade.side,
                    'contract_type': trade.contract_type,
                    'timestamp': trade.timestamp.isoformat()
                } for trade in self.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filename: str):
        """Load portfolio state from file with contract type support"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.cash = state['cash']
            self.initial_cash = state['initial_cash']
            
            # Restore positions
            self.positions = {}
            for pos_data in state['positions']:
                pos = Position(
                    market_ticker=pos_data['market_ticker'],
                    quantity=pos_data['quantity'],
                    avg_price=pos_data['avg_price'],
                    current_price=pos_data['current_price'],
                    contract_type=pos_data.get('contract_type', 'YES'),
                    timestamp=datetime.fromisoformat(pos_data['timestamp'])
                )
                self.positions[pos.market_ticker] = pos
            
            # Restore trades
            self.trades = []
            for trade_data in state['trades']:
                trade = Trade(
                    market_ticker=trade_data['market_ticker'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    side=trade_data['side'],
                    contract_type=trade_data.get('contract_type', 'YES'),
                    timestamp=datetime.fromisoformat(trade_data['timestamp'])
                )
                self.trades.append(trade)
                
        except FileNotFoundError:
            print(f"Portfolio state file {filename} not found, starting fresh")
        except Exception as e:
            print(f"Error loading portfolio state: {e}")
    
    def get_trade_history(self, market_ticker: str = None, contract_type: str = None) -> List[Trade]:
        """Get trade history with optional filtering"""
        trades = self.trades
        
        if market_ticker:
            trades = [t for t in trades if t.market_ticker == market_ticker]
        
        if contract_type:
            trades = [t for t in trades if t.contract_type == contract_type]
        
        return sorted(trades, key=lambda x: x.timestamp, reverse=True)
    
    def get_pnl_by_contract_type(self) -> Dict:
        """Get P&L breakdown by contract type"""
        yes_pnl = 0.0
        no_pnl = 0.0
        
        for pos in self.positions.values():
            if pos.quantity > 0:  # YES contracts
                yes_pnl += pos.unrealized_pnl
            else:  # NO contracts
                no_pnl += pos.unrealized_pnl
        
        return {
            'yes_pnl': yes_pnl,
            'no_pnl': no_pnl,
            'total_pnl': yes_pnl + no_pnl
        }
