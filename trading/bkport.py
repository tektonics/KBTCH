"""
Portfolio management and position tracking
"""
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import json

@dataclass
class Position:
    market_ticker: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_price)

@dataclass
class Trade:
    market_ticker: str
    quantity: int
    price: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Portfolio:
    def __init__(self, initial_cash: float = 10000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
    def add_trade(self, market_ticker: str, quantity: int, price: float, side: str):
        """Add a new trade and update positions"""
        trade = Trade(market_ticker, quantity, price, side)
        self.trades.append(trade)
        
        # Update cash
        trade_value = quantity * price
        if side == 'buy':
            self.cash -= trade_value
        else:
            self.cash += trade_value
        
        # Update positions
        self._update_position(market_ticker, quantity if side == 'buy' else -quantity, price)
        
        return trade
    
    def _update_position(self, market_ticker: str, quantity: int, price: float):
        """Update position with new trade"""
        if market_ticker in self.positions:
            position = self.positions[market_ticker]
            
            # Calculate new average price
            total_quantity = position.quantity + quantity
            if total_quantity == 0:
                # Position closed
                del self.positions[market_ticker]
                return
            
            total_cost = (position.quantity * position.avg_price) + (quantity * price)
            new_avg_price = total_cost / total_quantity
            
            position.quantity = total_quantity
            position.avg_price = new_avg_price
        else:
            # New position
            if quantity != 0:
                self.positions[market_ticker] = Position(market_ticker, quantity, price)
    
    def update_market_prices(self, price_data: Dict[str, float]):
        """Update current market prices for all positions"""
        for ticker, price in price_data.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
    
    def get_position(self, market_ticker: str) -> Position:
        """Get position for a specific market"""
        return self.positions.get(market_ticker)
    
    def get_total_exposure(self) -> float:
        """Get total market exposure across all positions"""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_market_exposure(self, market_ticker: str) -> float:
        """Get exposure for a specific market"""
        position = self.positions.get(market_ticker)
        return abs(position.market_value) if position else 0.0
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """Get realized P&L from closed positions"""
        return self.get_portfolio_value() - self.initial_cash - self.get_unrealized_pnl()
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L (placeholder - would need to track daily marks)"""
        return self.daily_pnl
    
    def can_afford(self, quantity: int, price: float) -> bool:
        """Check if we have enough cash for a trade"""
        required_cash = quantity * price
        return self.cash >= required_cash
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions"""
        return {
            'cash': self.cash,
            'total_value': self.get_portfolio_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'positions': {
                ticker: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl
                } for ticker, pos in self.positions.items()
            }
        }
    
    def save_state(self, filename: str):
        """Save portfolio state to file"""
        state = {
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'positions': [
                {
                    'market_ticker': pos.market_ticker,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'timestamp': pos.timestamp.isoformat()
                } for pos in self.positions.values()
            ],
            'trades': [
                {
                    'market_ticker': trade.market_ticker,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'side': trade.side,
                    'timestamp': trade.timestamp.isoformat()
                } for trade in self.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filename: str):
        """Load portfolio state from file"""
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
                    timestamp=datetime.fromisoformat(trade_data['timestamp'])
                )
                self.trades.append(trade)
                
        except FileNotFoundError:
            print(f"Portfolio state file {filename} not found, starting fresh")
        except Exception as e:
            print(f"Error loading portfolio state: {e}")
