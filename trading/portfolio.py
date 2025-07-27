"""
Portfolio management and position tracking with Kalshi integration
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, initial_cash: float = 10000, kalshi_client=None):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.kalshi_client = kalshi_client
        
    def set_kalshi_client(self, kalshi_client):
        """Set the Kalshi client for live data fetching"""
        self.kalshi_client = kalshi_client
    
    def fetch_kalshi_balance(self) -> Optional[Dict]:
        """Fetch account balance from Kalshi"""
        if not self.kalshi_client:
            logger.warning("Kalshi client not configured")
            return None
        
        try:
            balance_data = self.kalshi_client.get_balance()
            if balance_data and 'balance' in balance_data:
                self.cash = balance_data['balance'] / 100  # Convert cents to dollars
                logger.info(f"Updated cash balance from Kalshi: ${self.cash:.2f}")
                return balance_data
        except Exception as e:
            logger.error(f"Error fetching Kalshi balance: {e}")
        
        return None
    
    def fetch_kalshi_positions(self) -> Optional[List[Dict]]:
        """Fetch current positions from Kalshi"""
        if not self.kalshi_client:
            logger.warning("Kalshi client not configured")
            return None
        
        try:
            positions_data = self.kalshi_client.get_positions()
            if positions_data and 'positions' in positions_data:
                self._sync_kalshi_positions(positions_data['positions'])
                logger.info(f"Synced {len(positions_data['positions'])} positions from Kalshi")
                return positions_data['positions']
        except Exception as e:
            logger.error(f"Error fetching Kalshi positions: {e}")
        
        return None
    
    def fetch_kalshi_fills(self) -> Optional[List[Dict]]:
        """Fetch trade fills from Kalshi"""
        if not self.kalshi_client:
            logger.warning("Kalshi client not configured")
            return None
        
        try:
            fills_data = self.kalshi_client.get_fills()
            if fills_data and 'fills' in fills_data:
                self._sync_kalshi_fills(fills_data['fills'])
                logger.info(f"Synced {len(fills_data['fills'])} fills from Kalshi")
                return fills_data['fills']
        except Exception as e:
            logger.error(f"Error fetching Kalshi fills: {e}")
        
        return None
    
    def sync_with_kalshi(self) -> bool:
        if not self.kalshi_client:
            logger.warning("Kalshi client not configured")
            return False
        
        try:
            balance_success = self.fetch_kalshi_balance() is not None
            
            positions_success = self.fetch_kalshi_positions() is not None
            
            fills_success = self.fetch_kalshi_fills() is not None
            
            success = balance_success and positions_success and fills_success
            if success:
                logger.info("Successfully synced portfolio with Kalshi")
            else:
                logger.warning("Partial sync with Kalshi - some data may be incomplete")
            
            return success
            
        except Exception as e:
            logger.error(f"Error syncing with Kalshi: {e}")
            return False
    
    def _sync_kalshi_positions(self, kalshi_positions: List[Dict]):
        self.positions.clear()
        
        for pos_data in kalshi_positions:
            try:
                market_ticker = pos_data.get('market_ticker')
                quantity = pos_data.get('quantity', 0)
                avg_price = pos_data.get('average_price', 0) / 100
                current_price = pos_data.get('market_price', avg_price) / 100
                
                if market_ticker and quantity != 0:
                    position = Position(
                        market_ticker=market_ticker,
                        quantity=quantity,
                        avg_price=avg_price,
                        current_price=current_price
                    )
                    self.positions[market_ticker] = position
                    
            except Exception as e:
                logger.error(f"Error parsing Kalshi position {pos_data}: {e}")
    
    def _sync_kalshi_fills(self, kalshi_fills: List[Dict]):
        recent_fills = [f for f in kalshi_fills if self._is_recent_fill(f)]
        
        for fill_data in recent_fills:
            try:
                market_ticker = fill_data.get('market_ticker')
                quantity = fill_data.get('quantity', 0)
                price = fill_data.get('price', 0) / 100
                side = 'buy' if fill_data.get('side') == 'yes' else 'sell'
                fill_time = fill_data.get('created_time')
                
                if market_ticker and quantity and price:
                    timestamp = datetime.fromisoformat(fill_time.replace('Z', '+00:00')) if fill_time else None
                    
                    trade = Trade(
                        market_ticker=market_ticker,
                        quantity=quantity,
                        price=price,
                        side=side,
                        timestamp=timestamp
                    )
                    
                    if not self._has_duplicate_trade(trade):
                        self.trades.append(trade)
                        
            except Exception as e:
                logger.error(f"Error parsing Kalshi fill {fill_data}: {e}")
    
    def _is_recent_fill(self, fill_data: Dict) -> bool:
        try:
            fill_time = fill_data.get('created_time')
            if not fill_time:
                return False
            
            fill_datetime = datetime.fromisoformat(fill_time.replace('Z', '+00:00'))
            now = datetime.now(fill_datetime.tzinfo)
            return (now - fill_datetime).total_seconds() < 86400
        except:
            return False
    
    def _has_duplicate_trade(self, new_trade: Trade) -> bool:
        for existing_trade in self.trades:
            if (existing_trade.market_ticker == new_trade.market_ticker and
                existing_trade.quantity == new_trade.quantity and
                existing_trade.price == new_trade.price and
                existing_trade.side == new_trade.side and
                existing_trade.timestamp and new_trade.timestamp and
                abs((existing_trade.timestamp - new_trade.timestamp).total_seconds()) < 60):
                return True
        return False
    
    def update_market_prices_from_kalshi(self):
        if not self.kalshi_client:
            logger.warning("Kalshi client not configured")
            return
        
        for ticker in self.positions.keys():
            try:
                ticker_data = self.kalshi_client.get_mid_prices(ticker)
                if ticker_data and ticker_data.price is not None:
                    self.positions[ticker].current_price = ticker_data.price / 100
            except Exception as e:
                logger.error(f"Error updating price for {ticker}: {e}")
        
        logger.info("Updated market prices from Kalshi")

    def add_trade(self, market_ticker: str, quantity: int, price: float, side: str):
        trade = Trade(market_ticker, quantity, price, side)
        self.trades.append(trade)
        
        trade_value = quantity * price
        if side == 'buy':
            self.cash -= trade_value
        else:
            self.cash += trade_value
        
        self._update_position(market_ticker, quantity if side == 'buy' else -quantity, price)
        
        return trade
    
    def _update_position(self, market_ticker: str, quantity: int, price: float):
        if market_ticker in self.positions:
            position = self.positions[market_ticker]
            
            total_quantity = position.quantity + quantity
            if total_quantity == 0:
                del self.positions[market_ticker]
                return
            
            total_cost = (position.quantity * position.avg_price) + (quantity * price)
            new_avg_price = total_cost / total_quantity
            
            position.quantity = total_quantity
            position.avg_price = new_avg_price
        else:
            if quantity != 0:
                self.positions[market_ticker] = Position(market_ticker, quantity, price)
    
    def update_market_prices(self, price_data: Dict[str, float]):
        for ticker, price in price_data.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
    
    def get_position(self, market_ticker: str) -> Position:
        return self.positions.get(market_ticker)
    
    def get_total_exposure(self) -> float:
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_market_exposure(self, market_ticker: str) -> float:
        position = self.positions.get(market_ticker)
        return abs(position.market_value) if position else 0.0
    
    def get_portfolio_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> float:
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
