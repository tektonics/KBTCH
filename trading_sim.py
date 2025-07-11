# kalshi_trading_simulator.py
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
    
    def update_current_price(self, price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        if self.side == "YES":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # NO position
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

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
class TradingParams:
    """Consolidated trading parameters"""
    base_market_count: int = 3
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 1.0
    max_markets: int = 7
    min_edge_threshold: float = 0.05
    max_spread_threshold: int = 8
    max_risk_per_trade: float = 0.02
    position_size_dollars: float = 500.0  # Fixed position size for simulation
    max_position_per_market: int = 10  # Max contracts per market

class TradingSimulator:
    """Handles simulated trade execution and portfolio management"""
    
    def __init__(self, starting_balance: float = 10000.0):
        self.portfolio = SimulatedPortfolio(starting_balance, starting_balance)
        self.trade_log = []
        self.params = TradingParams()
        
    def can_open_position(self, market_ticker: str, side: str, price: float, quantity: int) -> bool:
        """Check if we can open a new position"""
        # Check if we already have a position in this market
        if market_ticker in self.portfolio.positions:
            existing_pos = self.portfolio.positions[market_ticker]
            # Don't open opposing positions
            if existing_pos.side != side:
                return False
            # Don't exceed max position size
            if existing_pos.quantity + quantity > self.params.max_position_per_market:
                return False
        
        # Check if we have enough balance
        required_capital = price * quantity
        return self.portfolio.current_balance >= required_capital
    
    def execute_simulated_trade(self, market_ticker: str, action: str, market_data: Any) -> Optional[Dict]:
        """Execute a simulated trade based on action and market data"""
        if not market_data or not market_data.yes_ask or not market_data.yes_bid:
            return None
        
        # Determine trade parameters
        if action == "BUY_YES":
            side = "YES"
            price = market_data.yes_ask  # We pay the ask
            trade_type = "BUY"
        elif action == "SELL_YES":
            side = "YES"
            price = market_data.yes_bid  # We receive the bid
            trade_type = "SELL"
        else:
            return None
        
        # Calculate position size
        target_dollars = self.params.position_size_dollars
        quantity = max(1, int(target_dollars / price))
        
        # Check if trade is possible
        if trade_type == "BUY":
            if not self.can_open_position(market_ticker, side, price, quantity):
                return None
            return self._open_position(market_ticker, side, price, quantity)
        else:  # SELL
            return self._close_position(market_ticker, side, price, quantity)
    
    def _open_position(self, market_ticker: str, side: str, price: float, quantity: int) -> Dict:
        """Open a new position or add to existing one"""
        cost = price * quantity
        
        if market_ticker in self.portfolio.positions:
            # Add to existing position
            existing_pos = self.portfolio.positions[market_ticker]
            total_cost = existing_pos.entry_price * existing_pos.quantity + cost
            total_quantity = existing_pos.quantity + quantity
            avg_price = total_cost / total_quantity
            
            existing_pos.quantity = total_quantity
            existing_pos.entry_price = avg_price
        else:
            # Create new position
            self.portfolio.positions[market_ticker] = SimulatedPosition(
                market_ticker=market_ticker,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=time.time()
            )
        
        self.portfolio.current_balance -= cost
        
        trade_info = {
            "type": "OPEN",
            "market": market_ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "cost": cost,
            "timestamp": time.time()
        }
        
        self.trade_log.append(trade_info)
        return trade_info
    
    def _close_position(self, market_ticker: str, side: str, price: float, quantity: int) -> Optional[Dict]:
        """Close part or all of a position"""
        if market_ticker not in self.portfolio.positions:
            return None
        
        position = self.portfolio.positions[market_ticker]
        if position.side != side:
            return None
        
        # Determine how much we can actually sell
        quantity = min(quantity, position.quantity)
        if quantity <= 0:
            return None
        
        # Calculate P&L
        if side == "YES":
            pnl = (price - position.entry_price) * quantity
        else:  # NO
            pnl = (position.entry_price - price) * quantity
        
        proceeds = price * quantity
        self.portfolio.current_balance += proceeds
        self.portfolio.total_realized_pnl += pnl
        
        # Create completed trade record
        completed_trade = SimulatedTrade(
            market_ticker=market_ticker,
            side=side,
            quantity=quantity,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            realized_pnl=pnl,
            trade_type="SELL"
        )
        self.portfolio.completed_trades.append(completed_trade)
        
        # Update position
        if quantity == position.quantity:
            # Close entire position
            del self.portfolio.positions[market_ticker]
        else:
            # Reduce position size
            position.quantity -= quantity
        
        trade_info = {
            "type": "CLOSE",
            "market": market_ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "proceeds": proceeds,
            "pnl": pnl,
            "timestamp": time.time()
        }
        
        self.trade_log.append(trade_info)
        return trade_info
    
    def update_positions(self, market_data_dict: Dict[str, Any]):
        """Update all positions with current market prices"""
        for ticker, position in self.portfolio.positions.items():
            if ticker in market_data_dict:
                market_data = market_data_dict[ticker]
                if market_data and market_data.yes_bid and market_data.yes_ask:
                    # Use mid price for marking positions
                    mid_price = (market_data.yes_bid + market_data.yes_ask) / 2
                    position.update_current_price(mid_price)
        
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

# [Include all the existing classes from kalshirunner.py here - BTCPriceMonitor, BRTIManager, MarketAnalyzer, DisplayManager]

class BTCPriceMonitor:
    """Optimized BTC price monitoring with caching"""
    def __init__(self, price_file: str = "aggregate_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5
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
    
    def calculate_volatility(self, window_minutes: int = 15) -> float:
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
            for _ in range(30):  # 30 second timeout
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
    """Separated market analysis logic"""
    def __init__(self, params: TradingParams):
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
        base_count = self.params.base_market_count
        
        # Volatility adjustments
        if volatility > self.params.volatility_threshold_high:
            base_count += 3
        elif volatility > self.params.volatility_threshold_low:
            base_count += 1
        
        # Time decay adjustments
        time_adjustments = {1: 3, 3: 2, 6: 1}
        for threshold, adjustment in time_adjustments.items():
            if time_to_expiry_hours < threshold:
                base_count += adjustment
                break
        
        return min(base_count, self.params.max_markets)
    
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
        
        # Determine direction based on position and edge
        if btc_price > market_info.strike:
            return "BUY_YES" if market_info.edge > self.params.min_edge_threshold else "SELL_YES"
        else:
            return "SELL_YES" if market_info.edge > self.params.min_edge_threshold else "BUY_YES"

class DisplayManager:
    """Enhanced display manager with simulation stats"""
    def __init__(self):
        self.display_line_count = 0
    
    def clear_display(self):
        if self.display_line_count > 0:
            for i in range(self.display_line_count):
                sys.stdout.write('\r\033[K')
                if i < self.display_line_count - 1:
                    sys.stdout.write('\033[A')
            sys.stdout.flush()
            self.display_line_count = 0
    
    def update_multiline_display(self, lines: list):
        self.clear_display()
        
        for i, line in enumerate(lines):
            if i > 0:
                sys.stdout.write('\n')
            sys.stdout.write(line)
        
        sys.stdout.flush()
        self.display_line_count = len(lines)
    
    def print_new_line(self, line: str):
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    def format_market_display(self, active_markets: List[MarketInfo], btc_price: float, 
                            volatility: float, brti_running: bool, portfolio_summary: Dict) -> List[str]:
        """Generate display with portfolio information"""
        try:
            edt_time = datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            edt_time = datetime.now()
        
        lines = []
        
        # Header with portfolio info
        vol_indicator = "🔥" if volatility > 1.0 else "📈" if volatility > 0.5 else "📊"
        brti_status = "🟢" if brti_running else "🔴"
        
        header = (f"{edt_time.strftime('%H:%M:%S')} | "
                 f"BTC: ${btc_price:,.2f} | "
                 f"Vol: {volatility:.1%} {vol_indicator} | "
                 f"Markets: {len(active_markets)} | "
                 f"BRTI: {brti_status}")
        lines.append(header)
        
        # Portfolio summary line
        portfolio_line = (f"📊 Portfolio: ${portfolio_summary['total_value']:,.2f} "
                         f"({portfolio_summary['total_return_pct']:+.1f}%) | "
                         f"Positions: {portfolio_summary['open_positions']} | "
                         f"Trades: {portfolio_summary['completed_trades']} | "
                         f"P&L: ${portfolio_summary['realized_pnl']:+,.2f}")
        lines.append(portfolio_line)
        
        # Market ladder
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            strike_labels = [
                f"${m.strike:,.0f}🎯" if m.is_primary else f"${m.strike:,.0f}"
                for m in sorted_markets
            ]
            lines.append(f"Ladder: {' | '.join(strike_labels)}")
        
        # Individual markets with simulation actions
        opportunities = 0
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            
            for market in sorted_markets:
                if market.market_data:
                    line = self._format_market_line(market)
                    lines.append(line)
                    
                    if market.action in ["BUY_YES", "SELL_YES"]:
                        opportunities += 1
        
        # Opportunities summary
        if opportunities:
            lines.append(f"🚨 {opportunities} TRADING OPPORTUNITIES")
        
        return lines
    
    def _format_market_line(self, market: MarketInfo) -> str:
        data = market.market_data
        primary_indicator = "🎯" if market.is_primary else "  "
        
        action_emoji = {
            "BUY_YES": "🟢", "SELL_YES": "🔴", "HOLD": "⚪",
            "SPREAD_TOO_WIDE": "📏", "INSUFFICIENT_EDGE": "⚖️", "NO_DATA": "❓"
        }.get(market.action, "⚪")
        
        if data.yes_bid and data.yes_ask:
            yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
            no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
            no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
            spread_text = f"Spread: {market.spread:.0f}¢" if market.spread else ""
        else:
            yes_prices = "YES: --/--"
            no_prices = "NO: --/--"
            spread_text = "No data"
        
        line = (f"{primary_indicator}${market.strike:,.0f}: "
               f"{yes_prices} | {no_prices} | {spread_text}")
        
        if market.edge:
            line += f" | Edge: {market.edge:+.1%}"
        
        return line + f" {action_emoji}"

class VolatilityAdaptiveSimulator:
    """Main simulator class with trading simulation"""
    def __init__(self, event_id: Optional[str] = None, starting_balance: float = 10000.0):
        self.event_id = event_id or self._generate_current_event_id()
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.brti_manager = BRTIManager()
        self.params = TradingParams()
        self.analyzer = MarketAnalyzer(self.params)
        self.display = DisplayManager()
        self.simulator = TradingSimulator(starting_balance)
        
        # State
        self.markets = []
        self.active_markets: List[MarketInfo] = []
        self.market_subscriptions = set()
        self.last_btc_price = None
        self.current_volatility = 0.0
        self.shutdown_requested = False
        
        # Statistics
        self.btc_updates = 0
        self.market_updates = 0
        self.volatility_updates = 0
        self.last_market_update_time = None
        self.last_trade_check = 0
        self.trade_cooldown = 5.0  # Wait 5 seconds between trades
        
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
        print("🚀 KBTCH Simulator Starting...")
        print(f"💰 Starting Balance: ${self.simulator.portfolio.starting_balance:,.2f}")
        
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
    
    def _execute_trading_logic(self):
        """Execute simulated trades based on market analysis"""
        current_time = time.time()
        if current_time - self.last_trade_check < self.trade_cooldown:
            return
        
        self.last_trade_check = current_time
        
        for market in self.active_markets:
            if market.action in ["BUY_YES", "SELL_YES"] and market.market_data:
                trade_result = self.simulator.execute_simulated_trade(
                    market.ticker, market.action, market.market_data
                )
                
                if trade_result:
                    # Log the trade
                    action_desc = "🟢 BOUGHT" if trade_result["type"] == "OPEN" else "🔴 SOLD"
                    self.display.print_new_line(
                        f"{action_desc} {trade_result['quantity']} {market.ticker} "
                        f"@ ${trade_result['price']:.0f} "
                        f"({trade_result.get('pnl', trade_result.get('cost', 0)):+.2f})"
                    )
                    
                    # Brief pause to show the trade
                    time.sleep(1)
    
    async def run_trading_loop(self):
        if not await self.initialize():
            return
        
        # Setup WebSocket message counting
        original_handler = self.client._handle_ws_message
        def counting_handler(msg):
            self.market_updates += 1
            self.last_market_update_time = time.time()
            original_handler(msg)
        self.client._handle_ws_message = counting_handler
        
        try:
            while not self.shutdown_requested:
                # Monitor BRTI health
                if not self.brti_manager.is_brti_running():
                    await self.brti_manager.start_brti()
                
                # Update BTC price
                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.btc_updates += 1
                
                # Update volatility
                new_volatility = self.btc_monitor.calculate_volatility()
                if abs(new_volatility - self.current_volatility) > 0.1:
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
                
                # Execute trading logic
                self._execute_trading_logic()
                
                # Get portfolio summary
                portfolio_summary = self.simulator.get_portfolio_summary()
                
                # Display
                lines = self.display.format_market_display(
                    self.active_markets, self.last_btc_price, 
                    self.current_volatility, self.brti_manager.is_brti_running(),
                    portfolio_summary
                )
                self.display.update_multiline_display(lines)
                
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            self.display.print_new_line("\n🛑 Shutting down...")
        except Exception as e:
            self.display.print_new_line(f"\n❌ Error: {e}")
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        self._restore_client_output()
        
        # Print final portfolio summary
        summary = self.simulator.get_portfolio_summary()
        print("\n" + "="*60)
        print("FINAL PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Starting Balance: ${summary['starting_balance']:,.2f}")
        print(f"Final Value: ${summary['total_value']:,.2f}")
        print(f"Total Return: ${summary['total_value'] - summary['starting_balance']:+,.2f} ({summary['total_return_pct']:+.1f}%)")
        print(f"Realized P&L: ${summary['realized_pnl']:+,.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        print(f"Max Drawdown: {summary['max_drawdown_pct']:.1f}%")
        print(f"Completed Trades: {summary['completed_trades']}")
        print(f"Open Positions: {summary['open_positions']}")
        
        # Save trade log
        if self.simulator.trade_log:
            log_file = f"simulation_log_{int(time.time())}.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'trades': self.simulator.trade_log,
                    'final_positions': {
                        k: {
                            'ticker': v.market_ticker,
                            'side': v.side,
                            'quantity': v.quantity,
                            'entry_price': v.entry_price,
                            'current_price': v.current_price,
                            'unrealized_pnl': v.unrealized_pnl
                        } for k, v in self.simulator.portfolio.positions.items()
                    }
                }, f, indent=2)
            print(f"Trade log saved: {log_file}")
        
        self.brti_manager.stop_brti()
        try:
            await self.client.close()
        except:
            pass

async def main():
    """Entry point for simulation mode"""
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
        # First argument: event_id (optional)
        event_id = sys.argv[1]
        
    if len(sys.argv) > 2:
        # Second argument: starting_balance (optional)
        try:
            starting_balance = float(sys.argv[2])
        except ValueError:
            print("Invalid starting balance, using $10,000")
    
    # Show usage info
    if event_id:
        print(f"🎯 Using specified event: {event_id}")
    else:
        print("🎯 Using auto-generated current hour event")
    
    simulator = VolatilityAdaptiveSimulator(event_id=event_id, starting_balance=starting_balance)
    try:
        await simulator.run_trading_loop()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
    finally:
        try:
            if hasattr(simulator, 'brti_manager'):
                simulator.brti_manager.stop_brti()
        except:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        sys.exit(1)
