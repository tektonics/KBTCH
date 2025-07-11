# enhanced_trading_sim.py - High Frequency Micro Trading Version
import asyncio
import json
import time
import sys
import numpy as np
import subprocess
import signal
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from collections import deque
from kalshi_bot.kalshi_client import KalshiClient

@dataclass
class SimulatedPosition:
    """Represents a simulated position in a market"""
    market_ticker: str
    side: str  # "YES" or "NO"
    quantity: int
    entry_price: float
    entry_time: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    peak_pnl: float = 0.0  # Track peak profit for trailing stops
    
    def update_current_price(self, price_cents: float):
        """Update current price and calculate unrealized P&L (prices in cents)"""
        self.current_price = price_cents
        if self.side == "YES":
            # P&L in dollars: (current_cents - entry_cents) / 100 * quantity
            pnl_cents_per_contract = price_cents - self.entry_price
            self.unrealized_pnl = (pnl_cents_per_contract / 100.0) * self.quantity
        else:  # NO position
            pnl_cents_per_contract = self.entry_price - price_cents
            self.unrealized_pnl = (pnl_cents_per_contract / 100.0) * self.quantity
        
        # Track peak P&L for trailing stops
        if self.unrealized_pnl > self.peak_pnl:
            self.peak_pnl = self.unrealized_pnl

@dataclass
class SimulatedTrade:
    """Represents a completed simulated trade"""
    market_ticker: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    realized_pnl: float
    trade_type: str  # "BUY", "SELL"
    exit_reason: str = "MANUAL"

@dataclass
class SimulatedPortfolio:
    """Tracks simulated portfolio state"""
    starting_balance: float = 10000.0
    current_balance: float = field(default_factory=lambda: 10000.0)
    positions: Dict[str, SimulatedPosition] = field(default_factory=dict)
    completed_trades: List[SimulatedTrade] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = field(default_factory=lambda: 10000.0)
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value including unrealized P&L"""
        return self.current_balance + self.total_unrealized_pnl
    
    def update_unrealized_pnl(self):
        """Update total unrealized P&L from all positions"""
        self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update drawdown tracking
        current_value = self.get_total_value()
        if current_value > self.peak_balance:
            self.peak_balance = current_value
        
        drawdown = (self.peak_balance - current_value) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

@dataclass
class MarketInfo:
    ticker: str
    strike: float
    distance: float
    is_primary: bool
    market_data: Optional[Any] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    implied_prob: Optional[float] = None
    edge: Optional[float] = None
    action: str = "HOLD"

@dataclass
class HFTradingParams:
    """High-frequency micro trading parameters"""
    # Micro position sizing for HFT (prices are in CENTS 0-100)
    base_position_contracts: int = 10  # Base number of contracts
    max_position_contracts: int = 50   # Max contracts per position
    max_portfolio_exposure: float = 0.3  # Lower exposure for HFT
    max_positions: int = 3  # Conservative position limit
    
    # Aggressive risk management for HFT
    stop_loss_pct: float = 0.08  # Tighter 8% stop loss
    profit_target_pct: float = 0.12  # Quick 12% profit target
    trailing_stop_pct: float = 0.04  # 4% trailing stop
    max_position_time_seconds: float = 300.0  # 5 minute max hold time
    
    # HFT edge requirements (lower for more opportunities)
    min_edge_threshold: float = 0.03  # Lower threshold for more trades
    max_spread_threshold: float = 12.0  # Accept wider spreads for opportunities
    min_confidence_score: float = 0.4  # Lower confidence threshold
    
    # Momentum and timing
    momentum_window_seconds: float = 30.0  # Very short momentum window
    trade_cooldown_seconds: float = 2.0  # Fast re-entry
    price_change_threshold: float = 0.001  # 0.1% price change trigger

class MomentumTracker:
    """High-frequency momentum tracking"""
    
    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self.price_history = deque(maxlen=100)
        self.last_momentum = 0.0
        
    def update_price(self, price: float, timestamp: float):
        """Update price history with timestamp"""
        self.price_history.append((timestamp, price))
        
    def get_momentum(self) -> float:
        """Calculate short-term momentum score"""
        if len(self.price_history) < 5:
            return self.last_momentum
            
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Filter recent prices
        recent_prices = [(t, p) for t, p in self.price_history if t > cutoff_time]
        
        if len(recent_prices) < 3:
            return self.last_momentum
            
        # Calculate momentum using price velocity
        times = np.array([t for t, _ in recent_prices])
        prices = np.array([p for _, p in recent_prices])
        
        if len(times) < 2:
            return self.last_momentum
            
        # Calculate velocity (price change per second)
        time_diff = times[-1] - times[0]
        if time_diff == 0:
            return self.last_momentum
            
        price_change = (prices[-1] - prices[0]) / prices[0]
        velocity = price_change / time_diff
        
        # Normalize and clamp
        momentum = velocity * 1000  # Scale for readability
        self.last_momentum = max(-1.0, min(1.0, momentum))
        return self.last_momentum
    
    def get_volatility(self) -> float:
        """Calculate short-term volatility"""
        if len(self.price_history) < 10:
            return 0.5
            
        prices = [p for _, p in list(self.price_history)[-10:]]
        if len(prices) < 2:
            return 0.5
            
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if not returns:
            return 0.5
            
        return min(2.0, np.std(returns) * 100)  # Cap volatility

class HFRiskManager:
    """High-frequency risk management"""
    
    def __init__(self, params: HFTradingParams):
        self.params = params
        self.momentum_tracker = MomentumTracker(params.momentum_window_seconds)
        self.last_trade_time = {}
        
    def update_price(self, price: float):
        """Update price for momentum tracking"""
        self.momentum_tracker.update_price(price, time.time())
    
    def can_trade_market(self, ticker: str) -> bool:
        """Check if enough time has passed since last trade"""
        current_time = time.time()
        last_trade = self.last_trade_time.get(ticker, 0)
        return current_time - last_trade >= self.params.trade_cooldown_seconds
    
    def record_trade(self, ticker: str):
        """Record trade time for cooldown tracking"""
        self.last_trade_time[ticker] = time.time()
    
    def calculate_position_size(self, edge: float, confidence: float, 
                              portfolio_value: float) -> int:
        """Calculate position size in contracts (prices are in cents)"""
        # Base contracts scaled by edge and confidence
        edge_multiplier = min(abs(edge) * 8, 2.5)
        confidence_multiplier = max(0.3, confidence)
        
        # Scale with portfolio size but keep reasonable
        portfolio_multiplier = max(0.5, min(2.0, portfolio_value / 10000.0))
        
        contracts = int(self.params.base_position_contracts * 
                       edge_multiplier * 
                       confidence_multiplier * 
                       portfolio_multiplier)
        
        return min(contracts, self.params.max_position_contracts)
    
    def should_enter_trade(self, market_info, btc_price: float, 
                          portfolio_state: Dict) -> Tuple[bool, str]:
        """HFT entry decision logic"""
        
        # Basic data checks
        if not market_info.market_data or not market_info.edge:
            return False, "NO_DATA"
        
        # Cooldown check
        if not self.can_trade_market(market_info.ticker):
            return False, "COOLDOWN"
        
        # Portfolio limits
        if portfolio_state['position_count'] >= self.params.max_positions:
            return False, "MAX_POSITIONS"
        
        exposure_pct = portfolio_state['total_exposure'] / portfolio_state['total_value']
        if exposure_pct > self.params.max_portfolio_exposure:
            return False, "MAX_EXPOSURE"
        
        # Edge requirements (relaxed for HFT)
        if abs(market_info.edge) < self.params.min_edge_threshold:
            return False, "INSUFFICIENT_EDGE"
        
        # Spread check (more lenient for HFT opportunities)
        if market_info.spread_pct and market_info.spread_pct > self.params.max_spread_threshold:
            return False, "SPREAD_TOO_WIDE"
        
        # Momentum alignment (less strict for HFT)
        momentum = self.momentum_tracker.get_momentum()
        is_buying_yes = market_info.edge > 0
        
        # Only block if momentum is strongly against us
        if is_buying_yes and momentum < -0.5:
            return False, "STRONG_BEARISH_MOMENTUM"
        if not is_buying_yes and momentum > 0.5:
            return False, "STRONG_BULLISH_MOMENTUM"
        
        return True, "APPROVED"
    
    def should_exit_position(self, position: SimulatedPosition, 
                           current_price_cents: float) -> Tuple[bool, str]:
        """HFT exit decision logic (prices in cents, P&L in dollars)"""
        current_time = time.time()
        position_age = current_time - position.entry_time
        
        # Time-based exit (much shorter for HFT)
        if position_age > self.params.max_position_time_seconds:
            return True, "TIME_EXIT"
        
        # Calculate position value in dollars for percentage calculations
        entry_value_dollars = (position.entry_price / 100.0) * position.quantity
        
        # Stop loss (P&L is already in dollars)
        if position.unrealized_pnl < 0:
            loss_pct = abs(position.unrealized_pnl) / entry_value_dollars
            if loss_pct > self.params.stop_loss_pct:
                return True, "STOP_LOSS"
        
        # Profit target (P&L is already in dollars)
        if position.unrealized_pnl > 0:
            profit_pct = position.unrealized_pnl / entry_value_dollars
            if profit_pct > self.params.profit_target_pct:
                return True, "PROFIT_TARGET"
        
        # Trailing stop (P&L is already in dollars)
        if position.peak_pnl > 0:
            drawdown_from_peak = position.peak_pnl - position.unrealized_pnl
            drawdown_pct = drawdown_from_peak / entry_value_dollars
            if drawdown_pct > self.params.trailing_stop_pct:
                return True, "TRAILING_STOP"
        
        return False, "HOLD"

class HFTradingSimulator:
    """High-frequency trading simulator"""
    
    def __init__(self, starting_balance: float = 10000.0):
        self.portfolio = SimulatedPortfolio(starting_balance, starting_balance)
        self.params = HFTradingParams()
        self.risk_manager = HFRiskManager(self.params)
        self.trade_log = []
        
        # HFT-specific tracking
        self.last_price_update = 0
        self.price_changed = False
        self.last_btc_price = None
        
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        total_exposure = sum(
            pos.entry_price * pos.quantity 
            for pos in self.portfolio.positions.values()
        )
        
        return {
            'total_value': self.portfolio.get_total_value(),
            'position_count': len(self.portfolio.positions),
            'total_exposure': total_exposure,
            'available_cash': self.portfolio.current_balance,
            'unrealized_pnl': self.portfolio.total_unrealized_pnl
        }
    
    def update_price(self, btc_price: float):
        """Update BTC price for momentum tracking"""
        if self.last_btc_price is None:
            self.last_btc_price = btc_price
            return
        
        # Check if price changed significantly
        price_change_pct = abs(btc_price - self.last_btc_price) / self.last_btc_price
        if price_change_pct > self.params.price_change_threshold:
            self.price_changed = True
        
        self.risk_manager.update_price(btc_price)
        self.last_btc_price = btc_price
        self.last_price_update = time.time()
    
    def execute_hft_trade(self, market_info, btc_price: float) -> Optional[Dict]:
        """Execute high-frequency trade (prices in CENTS 0-100)"""
        portfolio_state = self.get_portfolio_state()
        
        # Check if we should enter
        should_enter, reason = self.risk_manager.should_enter_trade(
            market_info, btc_price, portfolio_state
        )
        
        if not should_enter:
            return {"status": "REJECTED", "reason": reason}
        
        # Calculate position size in contracts
        confidence = self._calculate_confidence(market_info)
        contracts = self.risk_manager.calculate_position_size(
            market_info.edge, confidence, portfolio_state['total_value']
        )
        
        # Determine trade parameters (prices are in cents)
        if market_info.edge > 0:
            side = "YES"
            price_cents = market_info.market_data.yes_ask  # Price in cents (0-100)
            trade_type = "BUY"
        else:
            side = "YES"  # We'll sell YES if edge is negative
            price_cents = market_info.market_data.yes_bid  # Price in cents (0-100)
            trade_type = "SELL"
        
        if not price_cents or price_cents <= 0 or price_cents > 100:
            return {"status": "REJECTED", "reason": "INVALID_PRICE"}
        
        # Convert cents to dollars for cost calculation
        price_dollars = price_cents / 100.0
        
        # Check affordability (cost is in dollars)
        if trade_type == "BUY":
            cost_dollars = price_dollars * contracts
            if cost_dollars > self.portfolio.current_balance:
                # Reduce contracts to what we can afford
                contracts = max(1, int(self.portfolio.current_balance / price_dollars))
                cost_dollars = price_dollars * contracts
                if cost_dollars > self.portfolio.current_balance:
                    return {"status": "REJECTED", "reason": "INSUFFICIENT_FUNDS"}
        
        # Execute the trade
        trade_result = self._execute_trade(market_info.ticker, side, price_cents, contracts, trade_type)
        
        if trade_result:
            self.risk_manager.record_trade(market_info.ticker)
        
        return trade_result
    
    def _execute_trade(self, ticker: str, side: str, price_cents: float, 
                      quantity: int, trade_type: str) -> Dict:
        """Execute the actual trade (price_cents is 0-100, convert to dollars for costs)"""
        current_time = time.time()
        price_dollars = price_cents / 100.0  # Convert cents to dollars
        
        if trade_type == "BUY":
            # Open position
            cost_dollars = price_dollars * quantity
            
            if ticker in self.portfolio.positions:
                # Add to existing position (keep entry price in cents for consistency)
                existing_pos = self.portfolio.positions[ticker]
                existing_cost_dollars = (existing_pos.entry_price / 100.0) * existing_pos.quantity
                total_cost_dollars = existing_cost_dollars + cost_dollars
                total_quantity = existing_pos.quantity + quantity
                avg_price_cents = (total_cost_dollars / total_quantity) * 100  # Back to cents
                
                existing_pos.quantity = total_quantity
                existing_pos.entry_price = avg_price_cents
            else:
                # Create new position (store price in cents)
                self.portfolio.positions[ticker] = SimulatedPosition(
                    market_ticker=ticker,
                    side=side,
                    quantity=quantity,
                    entry_price=price_cents,  # Store in cents
                    entry_time=current_time
                )
            
            self.portfolio.current_balance -= cost_dollars
            
            trade_info = {
                "type": "OPEN",
                "market": ticker,
                "side": side,
                "quantity": quantity,
                "price": price_cents,  # Store in cents for consistency
                "cost": cost_dollars,  # Cost in dollars
                "timestamp": current_time
            }
        
        else:  # SELL
            # Close position
            if ticker not in self.portfolio.positions:
                return {"status": "ERROR", "reason": "NO_POSITION"}
            
            position = self.portfolio.positions[ticker]
            quantity = min(quantity, position.quantity)
            
            # Calculate P&L (both prices in cents, convert result to dollars)
            pnl_cents_per_contract = price_cents - position.entry_price
            pnl_dollars = (pnl_cents_per_contract / 100.0) * quantity
            proceeds_dollars = price_dollars * quantity
            
            self.portfolio.current_balance += proceeds_dollars
            self.portfolio.total_realized_pnl += pnl_dollars
            
            # Update position
            if quantity == position.quantity:
                del self.portfolio.positions[ticker]
            else:
                position.quantity -= quantity
            
            # Record completed trade
            completed_trade = SimulatedTrade(
                market_ticker=ticker,
                side=side,
                quantity=quantity,
                entry_price=position.entry_price,  # In cents
                exit_price=price_cents,  # In cents
                entry_time=position.entry_time,
                exit_time=current_time,
                realized_pnl=pnl_dollars,  # In dollars
                trade_type="SELL",
                exit_reason="MANUAL"
            )
            self.portfolio.completed_trades.append(completed_trade)
            
            trade_info = {
                "type": "CLOSE",
                "market": ticker,
                "side": side,
                "quantity": quantity,
                "price": price_cents,  # In cents
                "proceeds": proceeds_dollars,  # In dollars
                "pnl": pnl_dollars,  # In dollars
                "timestamp": current_time
            }
        
        self.trade_log.append(trade_info)
        return trade_info
    
    def _calculate_confidence(self, market_info) -> float:
        """Calculate trading confidence score"""
        confidence = 1.0
        
        # Spread penalty
        if market_info.spread_pct:
            if market_info.spread_pct > 8:
                confidence *= 0.7
            elif market_info.spread_pct > 5:
                confidence *= 0.85
        
        # Primary market bonus
        if market_info.is_primary:
            confidence *= 1.1
        
        # Volatility adjustment
        volatility = self.risk_manager.momentum_tracker.get_volatility()
        if volatility > 1.5:
            confidence *= 0.8
        
        return min(1.0, max(0.1, confidence))
    
    def check_exits(self) -> List[Tuple[str, str]]:
        """Check all positions for exit conditions"""
        exits_needed = []
        
        for ticker, position in self.portfolio.positions.items():
            current_price_cents = position.current_price or position.entry_price
            should_exit, reason = self.risk_manager.should_exit_position(position, current_price_cents)
            
            if should_exit:
                exits_needed.append((ticker, reason))
        
        return exits_needed
    
    def update_positions(self, market_data_dict: Dict[str, Any]):
        """Update all positions with current market prices (in cents)"""
        for ticker, position in self.portfolio.positions.items():
            if ticker in market_data_dict:
                market_data = market_data_dict[ticker]
                if market_data and market_data.yes_bid and market_data.yes_ask:
                    # Use mid price in cents for marking positions
                    mid_price_cents = (market_data.yes_bid + market_data.yes_ask) / 2
                    position.update_current_price(mid_price_cents)
        
        self.portfolio.update_unrealized_pnl()
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_value = self.portfolio.get_total_value()
        total_return = (total_value - self.portfolio.starting_balance) / self.portfolio.starting_balance
        
        return {
            "starting_balance": self.portfolio.starting_balance,
            "current_balance": self.portfolio.current_balance,
            "total_value": total_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "realized_pnl": self.portfolio.total_realized_pnl,
            "unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "max_drawdown": self.portfolio.max_drawdown,
            "max_drawdown_pct": self.portfolio.max_drawdown * 100,
            "open_positions": len(self.portfolio.positions),
            "completed_trades": len(self.portfolio.completed_trades),
            "total_trades": len(self.trade_log)
        }

# Keep existing classes from original file
class BTCPriceMonitor:
    """Optimized BTC price monitoring with caching"""
    def __init__(self, price_file: str = "aggregate_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.1  # Faster checking for HFT
        self.price_history = []
        self.max_history_minutes = 30
    
    def get_current_price(self) -> Optional[float]:
        now = time.time()
        if now - self.last_check < self.check_interval:
            return self.last_price
        
        self.last_check = now
        
        try:
            if not self.price_file.exists():
                return None
            
            current_modified = self.price_file.stat().st_mtime
            if current_modified == self.last_modified:
                return self.last_price
            
            with open(self.price_file, 'r') as f:
                data = json.load(f)
                price = data.get("price")
                
                if price and isinstance(price, (int, float)) and price > 0:
                    new_price = float(price)
                    
                    if new_price != self.last_price:
                        self.price_history.append((now, new_price))
                        self._cleanup_price_history(now)
                    
                    self.last_price = new_price
                    self.last_modified = current_modified
                    return self.last_price
                    
        except (json.JSONDecodeError, IOError):
            pass
        
        return None
    
    def _cleanup_price_history(self, current_time: float):
        cutoff_time = current_time - (self.max_history_minutes * 60)
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff_time]
    
    def calculate_volatility(self, window_minutes: int = 5) -> float:  # Shorter window for HFT
        cutoff_time = time.time() - (window_minutes * 60)
        recent_history = [(t, p) for t, p in self.price_history if t > cutoff_time]
        
        if len(recent_history) < 3:
            return 0.0
        
        prices = [price for _, price in recent_history]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0.0
        
        std_dev = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        return std_dev * np.sqrt(525600)  # Annualize

class BRTIManager:
    """Lightweight BRTI process manager"""
    def __init__(self, script_path: str = "brti.py"):
        self.script_path = Path(script_path)
        self.process: Optional[subprocess.Popen] = None
        self.price_file = Path("aggregate_price.json")
        
    def is_brti_running(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        
        if not self.price_file.exists():
            return False
        
        try:
            last_modified = self.price_file.stat().st_mtime
            return time.time() - last_modified < 10
        except OSError:
            return False
    
    async def start_brti(self) -> bool:
        if not self.script_path.exists():
            return False
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            # Wait for initialization
            for _ in range(30):
                if self.price_file.exists():
                    try:
                        with open(self.price_file, 'r') as f:
                            data = json.load(f)
                            if data.get("price") and data.get("status") == "active":
                                return True
                    except (json.JSONDecodeError, IOError):
                        pass
                await asyncio.sleep(1)
            
            return self.process.poll() is None
            
        except Exception:
            return False
    
    def stop_brti(self):
        if self.process is None:
            return
        
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        except Exception:
            pass
        finally:
            self.process = None

class MarketAnalyzer:
    """Market analysis logic optimized for HFT"""
    def __init__(self, params: HFTradingParams):
        self.params = params
    
    @staticmethod
    def extract_strike_price(ticker: str) -> float:
        try:
            parts = ticker.split("-")
            if len(parts) >= 3 and parts[-1].startswith("T"):
                return float(parts[-1][1:])
        except (ValueError, IndexError):
            pass
        return 0.0
    
    def calculate_adaptive_market_count(self, volatility: float, time_to_expiry_hours: float) -> int:
        # Conservative market selection for focused trading (max 3 positions)
        base_count = 3
        
        # Only increase in high volatility situations
        if volatility > 1.5:
            base_count = 4
        elif volatility > 1.0:
            base_count = 3
        
        # Slight increase near expiry for more opportunities
        if time_to_expiry_hours < 0.5:  # 30 minutes
            base_count += 1
        
        return min(base_count, 5)  # Never more than 5 markets to choose from
    
    def select_target_markets(self, markets: List[Dict], btc_price: float, volatility: float) -> List[MarketInfo]:
        if not markets or not btc_price:
            return []
        
        target_count = self.calculate_adaptive_market_count(volatility, 1.0)
        
        # Calculate distances and sort
        markets_with_distance = []
        for market in markets:
            strike = self.extract_strike_price(market["ticker"])
            if strike > 0:
                distance = abs(strike - btc_price)
                markets_with_distance.append({
                    'market': market,
                    'ticker': market["ticker"],
                    'strike': strike,
                    'distance': distance
                })
        
        # Get closest markets
        markets_with_distance.sort(key=lambda x: x['distance'])
        selected = markets_with_distance[:target_count]
        
        # Find primary and sort by strike
        primary_ticker = selected[0]['ticker'] if selected else None
        selected.sort(key=lambda x: x['strike'])
        
        # Convert to MarketInfo objects
        return [
            MarketInfo(
                ticker=m['ticker'],
                strike=m['strike'],
                distance=m['distance'],
                is_primary=(m['ticker'] == primary_ticker)
            )
            for m in selected
        ]
    
    def estimate_theoretical_probability(self, strike: float, current_price: float, 
                                       volatility: float, time_hours: float = 1.0) -> float:
        if strike <= 0 or current_price <= 0 or time_hours <= 0:
            return 0.5
        
        price_ratio = current_price / strike
        log_ratio = np.log(price_ratio)
        vol_sqrt_time = volatility * np.sqrt(time_hours / 8760)
        
        if vol_sqrt_time <= 0:
            return 1.0 if current_price > strike else 0.0
        
        z_score = log_ratio / vol_sqrt_time
        prob = 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
        
        return max(0.01, min(0.99, prob))
    
    def analyze_market_opportunity(self, market_info: MarketInfo, btc_price: float, volatility: float) -> MarketInfo:
        if not market_info.market_data:
            return market_info
        
        data = market_info.market_data
        
        # Calculate spread
        if data.yes_bid and data.yes_ask:
            market_info.spread = data.yes_ask - data.yes_bid
            market_info.spread_pct = (market_info.spread / data.yes_ask) * 100
        
        # Calculate implied probability
        if btc_price > market_info.strike:
            if data.yes_ask:
                market_info.implied_prob = data.yes_ask / 100
        else:
            if data.yes_bid:
                market_info.implied_prob = data.yes_bid / 100
        
        # Calculate theoretical probability and edge
        theoretical_prob = self.estimate_theoretical_probability(
            market_info.strike, btc_price, volatility, 1.0
        )
        
        if market_info.implied_prob:
            market_info.edge = theoretical_prob - market_info.implied_prob
        
        # Determine action
        market_info.action = self._determine_trading_action(market_info, btc_price)
        
        return market_info
    
    def _determine_trading_action(self, market_info: MarketInfo, btc_price: float) -> str:
        if not market_info.market_data or not market_info.spread_pct or not market_info.edge:
            return "NO_DATA"
        
        if market_info.spread_pct > self.params.max_spread_threshold:
            return "SPREAD_TOO_WIDE"
        
        if abs(market_info.edge) < self.params.min_edge_threshold:
            return "INSUFFICIENT_EDGE"
        
        # Determine direction based on edge
        if market_info.edge > self.params.min_edge_threshold:
            return "BUY_YES"
        elif market_info.edge < -self.params.min_edge_threshold:
            return "SELL_YES"
        else:
            return "HOLD"

class DisplayManager:
    """Enhanced display manager for HFT"""
    def __init__(self):
        self.display_line_count = 0
        self.last_update_time = 0
        self.update_interval = 0.2  # Faster updates for HFT
    
    def clear_display(self):
        if self.display_line_count > 0:
            for i in range(self.display_line_count):
                sys.stdout.write('\r\033[K')
                if i < self.display_line_count - 1:
                    sys.stdout.write('\033[A')
            sys.stdout.flush()
            self.display_line_count = 0
    
    def update_multiline_display(self, lines: list):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return  # Skip update to reduce flicker
        
        self.clear_display()
        
        for i, line in enumerate(lines):
            if i > 0:
                sys.stdout.write('\n')
            sys.stdout.write(line)
        
        sys.stdout.flush()
        self.display_line_count = len(lines)
        self.last_update_time = current_time
    
    def print_new_line(self, line: str):
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    def format_market_display(self, active_markets: List[MarketInfo], btc_price: float, 
                            volatility: float, brti_running: bool, portfolio_summary: Dict) -> List[str]:
        """Generate display with HFT portfolio information"""
        try:
            edt_time = datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            edt_time = datetime.now()
        
        lines = []
        
        # Header with HFT indicators
        vol_indicator = "🔥" if volatility > 1.0 else "📈" if volatility > 0.5 else "📊"
        brti_status = "🟢" if brti_running else "🔴"
        
        header = (f"{edt_time.strftime('%H:%M:%S.%f')[:-3]} | "  # Include milliseconds
                 f"BTC: ${btc_price:,.2f} | "
                 f"Vol: {volatility:.1%} {vol_indicator} | "
                 f"Markets: {len(active_markets)} | "
                 f"BRTI: {brti_status}")
        lines.append(header)
        
        # HFT Portfolio summary line
        pnl_indicator = "🟢" if portfolio_summary['total_return_pct'] > 0 else "🔴" if portfolio_summary['total_return_pct'] < 0 else "⚪"
        portfolio_line = (f"💰 ${portfolio_summary['total_value']:,.2f} "
                         f"({portfolio_summary['total_return_pct']:+.2f}%) {pnl_indicator} | "
                         f"Pos: {portfolio_summary['open_positions']}/3 | "  # Show max 3 positions
                         f"Trades: {portfolio_summary['completed_trades']} | "
                         f"R-PnL: ${portfolio_summary['realized_pnl']:+,.2f} | "
                         f"U-PnL: ${portfolio_summary['unrealized_pnl']:+,.2f}")
        lines.append(portfolio_line)
        
        # Market ladder with HFT indicators
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            strike_labels = []
            for m in sorted_markets:
                label = f"${m.strike:,.0f}"
                if m.is_primary:
                    label += "🎯"
                if m.action in ["BUY_YES", "SELL_YES"]:
                    label += "⚡"  # HFT opportunity indicator
                strike_labels.append(label)
            lines.append(f"Ladder: {' | '.join(strike_labels)}")
        
        # Individual markets with HFT metrics
        opportunities = 0
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            
            for market in sorted_markets:
                if market.market_data:
                    line = self._format_hft_market_line(market)
                    lines.append(line)
                    
                    if market.action in ["BUY_YES", "SELL_YES"]:
                        opportunities += 1
        
        # HFT Opportunities summary
        if opportunities:
            lines.append(f"⚡ {opportunities} HFT OPPORTUNITIES DETECTED")
        
        return lines
    
    def _format_hft_market_line(self, market: MarketInfo) -> str:
        data = market.market_data
        primary_indicator = "🎯" if market.is_primary else "  "
        
        action_emoji = {
            "BUY_YES": "🟢⚡", "SELL_YES": "🔴⚡", "HOLD": "⚪",
            "SPREAD_TOO_WIDE": "📏", "INSUFFICIENT_EDGE": "⚖️", "NO_DATA": "❓"
        }.get(market.action, "⚪")
        
        if data.yes_bid and data.yes_ask:
            # Prices are in cents (0-100), display as cents
            yes_prices = f"YES: {data.yes_bid:.0f}¢/{data.yes_ask:.0f}¢"
            no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
            no_prices = f"NO: {no_bid:.0f}¢/{no_ask:.0f}¢"
            spread_text = f"Spr: {market.spread:.0f}¢" if market.spread else ""
        else:
            yes_prices = "YES: --¢/--¢"
            no_prices = "NO: --¢/--¢"
            spread_text = "No data"
        
        line = (f"{primary_indicator}${market.strike:,.0f}: "
               f"{yes_prices} | {no_prices} | {spread_text}")
        
        if market.edge:
            line += f" | Edge: {market.edge:+.1%}"
        
        return line + f" {action_emoji}"

class VolatilityAdaptiveHFTrader:
    """High-frequency trading main class"""
    def __init__(self, event_id: Optional[str] = None, starting_balance: float = 10000.0):
        self.event_id = event_id or self._generate_current_event_id()
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.brti_manager = BRTIManager()
        self.params = HFTradingParams()
        self.analyzer = MarketAnalyzer(self.params)
        self.display = DisplayManager()
        self.simulator = HFTradingSimulator(starting_balance)
        
        # State
        self.markets = []
        self.active_markets: List[MarketInfo] = []
        self.market_subscriptions = set()
        self.last_btc_price = None
        self.current_volatility = 0.0
        self.shutdown_requested = False
        
        # HFT-specific state
        self.last_trade_opportunity_check = 0
        self.trade_opportunity_interval = 0.1  # Check every 100ms
        self.last_exit_check = 0
        self.exit_check_interval = 0.5  # Check exits every 500ms
        
        # Statistics
        self.btc_updates = 0
        self.market_updates = 0
        self.volatility_updates = 0
        self.trade_opportunities = 0
        self.trades_executed = 0
        
        self._setup_signal_handlers()
        self._silence_client_output()
    
    def _generate_current_event_id(self) -> str:
        try:
            edt_tz = ZoneInfo("America/New_York")
            now = datetime.now(edt_tz)
        except ImportError:
            now = datetime.now()
        
        next_hour = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour + 1) % 24)
        if next_hour.hour == 0 and now.hour == 23:
            next_hour = next_hour.replace(day=now.day + 1)
        
        return f"KXBTCD-{next_hour.strftime('%y%b%d%H').upper()}"
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'shutdown_requested', True))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'shutdown_requested', True))
    
    def _silence_client_output(self):
        import builtins
        self._original_print = builtins.print
        
        def silent_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            if not any(kw in message.lower() for kw in ['subscribed', 'kalshi']):
                self._original_print(*args, **kwargs)
        
        builtins.print = silent_print
    
    def _restore_client_output(self):
        if hasattr(self, '_original_print'):
            import builtins
            builtins.print = self._original_print
    
    async def initialize(self) -> bool:
        print("🚀 KBTCH HFT Simulator Starting...")
        print(f"💰 Starting Balance: ${self.simulator.portfolio.starting_balance:,.2f}")
        print("⚡ High-Frequency Trading Mode Enabled")
        
        # Start BRTI
        if not self.brti_manager.is_brti_running():
            await self.brti_manager.start_brti()
        
        try:
            # Get markets
            markets_data = self.client.get_markets(self.event_id)
            self.markets = markets_data.get("markets", [])
            
            if not self.markets:
                print(f"❌ No markets found for {self.event_id}")
                return False
            
            # Wait for BTC price
            for _ in range(60):
                btc_price = self.btc_monitor.get_current_price()
                if btc_price:
                    self.last_btc_price = btc_price
                    self.simulator.update_price(btc_price)
                    break
                await asyncio.sleep(1)
            else:
                print("❌ No BTC price data")
                return False
            
            # Setup markets
            await asyncio.sleep(2)
            self.current_volatility = self.btc_monitor.calculate_volatility()
            await self._update_market_subscriptions()
            
            if not self.active_markets:
                print("❌ No target markets")
                return False
            
            print(f"✅ Ready: {len(self.markets)} markets, BTC ${btc_price:,.0f}")
            print(f"🎯 Monitoring {len(self.active_markets)} active markets")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def _update_market_subscriptions(self):
        target_markets = self.analyzer.select_target_markets(
            self.markets, self.last_btc_price, self.current_volatility
        )
        
        if not target_markets:
            return
        
        new_tickers = {market.ticker for market in target_markets}
        
        # Subscribe to new markets
        for ticker in new_tickers - self.market_subscriptions:
            try:
                asyncio.create_task(self.client.subscribe_to_market(ticker))
                self.market_subscriptions.add(ticker)
            except Exception:
                pass
        
        self.active_markets = target_markets
    
    def _execute_hft_trading_logic(self):
        """Execute high-frequency trading logic"""
        current_time = time.time()
        
        # Check for trade opportunities more frequently
        if current_time - self.last_trade_opportunity_check >= self.trade_opportunity_interval:
            self.last_trade_opportunity_check = current_time
            
            for market in self.active_markets:
                if market.action in ["BUY_YES", "SELL_YES"] and market.market_data:
                    self.trade_opportunities += 1
                    
                    trade_result = self.simulator.execute_hft_trade(market, self.last_btc_price)
                    
                    if trade_result and trade_result.get("status") != "REJECTED":
                        self.trades_executed += 1
                        # Brief display of trade
                        action_desc = "🟢⚡ HFT BUY" if trade_result.get("type") == "OPEN" else "🔴⚡ HFT SELL"
                        self.display.print_new_line(
                            f"{action_desc} {trade_result.get('quantity', 0)} {market.ticker.split('-')[-1]} "
                            f"@ {trade_result.get('price', 0):.0f}¢"
                        )
                        time.sleep(0.5)  # Brief pause to show trade
        
        # Check for exits less frequently but still regularly
        if current_time - self.last_exit_check >= self.exit_check_interval:
            self.last_exit_check = current_time
            
            exits_needed = self.simulator.check_exits()
            for ticker, reason in exits_needed:
                # Execute exit
                if ticker in self.simulator.portfolio.positions:
                    position = self.simulator.portfolio.positions[ticker]
                    market_data = self.client.get_mid_prices(ticker)
                    
                    if market_data and market_data.yes_bid:
                        # Force exit at current bid (price in cents)
                        exit_result = self.simulator._execute_trade(
                            ticker, position.side, market_data.yes_bid, 
                            position.quantity, "SELL"
                        )
                        
                        if exit_result:
                            pnl = exit_result.get('pnl', 0)
                            pnl_indicator = "🟢" if pnl > 0 else "🔴"
                            self.display.print_new_line(
                                f"{pnl_indicator}⚡ HFT EXIT {ticker.split('-')[-1]} "
                                f"${pnl:+.2f} ({reason})"
                            )
                            time.sleep(0.5)
    
    async def run_trading_loop(self):
        if not await self.initialize():
            return
        
        # Setup WebSocket message counting
        original_handler = self.client._handle_ws_message
        def counting_handler(msg):
            self.market_updates += 1
            original_handler(msg)
        self.client._handle_ws_message = counting_handler
        
        try:
            while not self.shutdown_requested:
                # Monitor BRTI health
                if not self.brti_manager.is_brti_running():
                    await self.brti_manager.start_brti()
                
                # Update BTC price for HFT
                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.simulator.update_price(current_btc)
                    self.btc_updates += 1
                
                # Update volatility
                new_volatility = self.btc_monitor.calculate_volatility()
                if abs(new_volatility - self.current_volatility) > 0.05:  # More sensitive for HFT
                    self.current_volatility = new_volatility
                    self.volatility_updates += 1
                    await self._update_market_subscriptions()
                
                # Update market data and analyze
                market_data_dict = {}
                for market in self.active_markets:
                    market.market_data = self.client.get_mid_prices(market.ticker)
                    if market.market_data:
                        market_data_dict[market.ticker] = market.market_data
                        market = self.analyzer.analyze_market_opportunity(
                            market, self.last_btc_price, self.current_volatility
                        )
                
                # Update portfolio with current prices
                self.simulator.update_positions(market_data_dict)
                
                # Execute HFT trading logic
                self._execute_hft_trading_logic()
                
                # Get portfolio summary
                portfolio_summary = self.simulator.get_portfolio_summary()
                
                # Display (with reduced frequency to minimize flicker)
                lines = self.display.format_market_display(
                    self.active_markets, self.last_btc_price, 
                    self.current_volatility, self.brti_manager.is_brti_running(),
                    portfolio_summary
                )
                self.display.update_multiline_display(lines)
                
                await asyncio.sleep(0.1)  # Much faster loop for HFT
                
        except KeyboardInterrupt:
            self.display.print_new_line("\n🛑 Shutting down HFT trader...")
        except Exception as e:
            self.display.print_new_line(f"\n❌ HFT Error: {e}")
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        self._restore_client_output()
        
        # Print final HFT summary
        summary = self.simulator.get_portfolio_summary()
        print("\n" + "="*70)
        print("FINAL HFT PORTFOLIO SUMMARY")
        print("="*70)
        print(f"Starting Balance: ${summary['starting_balance']:,.2f}")
        print(f"Final Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: ${summary['total_value'] - summary['starting_balance']:+,.2f} ({summary['total_return_pct']:+.2f}%)")
        print(f"Realized P&L: ${summary['realized_pnl']:+,.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        print(f"Completed Trades: {summary['completed_trades']}")
        print(f"Open Positions: {summary['open_positions']}")
        
        # HFT-specific stats
        print(f"\n⚡ HFT PERFORMANCE")
        print(f"Trade Opportunities: {self.trade_opportunities}")
        print(f"Trades Executed: {self.trades_executed}")
        print(f"Execution Rate: {(self.trades_executed/max(1,self.trade_opportunities)*100):.1f}%")
        print(f"BTC Price Updates: {self.btc_updates}")
        print(f"Market Data Updates: {self.market_updates}")
        
        if summary['completed_trades'] > 0:
            avg_trade_duration = sum([
                t.exit_time - t.entry_time 
                for t in self.simulator.portfolio.completed_trades
            ]) / summary['completed_trades']
            print(f"Avg Trade Duration: {avg_trade_duration:.1f} seconds")
        
        # Save trade log
        if self.simulator.trade_log:
            log_file = f"hft_simulation_log_{int(time.time())}.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'trades': self.simulator.trade_log,
                    'hft_stats': {
                        'trade_opportunities': self.trade_opportunities,
                        'trades_executed': self.trades_executed,
                        'btc_updates': self.btc_updates,
                        'market_updates': self.market_updates
                    },
                    'final_positions': {
                        k: {
                            'ticker': v.market_ticker,
                            'side': v.side,
                            'quantity': v.quantity,
                            'entry_price': v.entry_price,
                            'current_price': v.current_price,
                            'unrealized_pnl': v.unrealized_pnl,
                            'peak_pnl': v.peak_pnl
                        } for k, v in self.simulator.portfolio.positions.items()
                    }
                }, f, indent=2)
            print(f"HFT Trade log saved: {log_file}")
        
        self.brti_manager.stop_brti()
        try:
            await self.client.close()
        except:
            pass

async def main():
    """Entry point for HFT simulation mode"""
    # Check dependencies
    missing_deps = []
    dependencies = ['ccxt', 'numpy', 'kalshi_bot.kalshi_client']
    
    for dep in dependencies:
        try:
            if dep == 'kalshi_bot.kalshi_client':
                from kalshi_bot.kalshi_client import KalshiClient
            else:
                __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        return
    
    if not Path("brti.py").exists():
        print("❌ brti.py not found")
        return
    
    # Parse command line arguments
    event_id = None
    starting_balance = 10000.0
    
    if len(sys.argv) > 1:
        event_id = sys.argv[1]
        
    if len(sys.argv) > 2:
        try:
            starting_balance = float(sys.argv[2])
        except ValueError:
            print("Invalid starting balance, using $10,000")
    
    # Show HFT mode info
    print("⚡ HIGH-FREQUENCY TRADING MODE")
    print(f"🎯 Event: {event_id or 'Auto-generated current hour'}")
    print(f"💰 Starting Balance: ${starting_balance:,.2f}")
    print("📊 HFT Parameters:")
    print("   - Max Position Time: 5 minutes")
    print("   - Stop Loss: 8%")
    print("   - Profit Target: 12%")
    print("   - Trailing Stop: 4%")
    print("   - Min Edge: 3%")
    print("   - Position Size: 10-50 contracts")
    print("   - Max Positions: 3 (focused trading)")
    print("   - Max Exposure: 30%")
    print("   - Prices: Displayed in cents (0-100¢)")
    print("   - Contract Cost: Price in cents = $0.XX per contract")
    
    hft_trader = VolatilityAdaptiveHFTrader(event_id=event_id, starting_balance=starting_balance)
    try:
        await hft_trader.run_trading_loop()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
    finally:
        try:
            if hasattr(hft_trader, 'brti_manager'):
                hft_trader.brti_manager.stop_brti()
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 HFT Session Ended!")
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        sys.exit(1)
