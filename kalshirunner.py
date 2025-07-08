import asyncio
import json
import time
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from dataclasses import dataclass
from kalshi_bot.kalshi_client import KalshiClient

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
class TradingOpportunity:
    ticker: str
    action: str
    edge: float
    confidence: float
    suggested_size: int
    reasoning: str

class BTCPriceMonitor:
    def __init__(self, price_file: str = "aggregate_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5
        # Price history for volatility calculation
        self.price_history = []  # [(timestamp, price), ...]
        self.max_history_minutes = 30
    
    def get_current_price(self) -> Optional[float]:
        """Get BTC price with efficient file monitoring"""
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
                    
                    # Add to price history if it's a new price
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
        """Remove old price history beyond max_history_minutes"""
        cutoff_time = current_time - (self.max_history_minutes * 60)
        self.price_history = [
            (timestamp, price) for timestamp, price in self.price_history
            if timestamp > cutoff_time
        ]
    
    def get_price_history(self, minutes: int = 15) -> List[tuple]:
        """Get price history for the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        return [
            (timestamp, price) for timestamp, price in self.price_history
            if timestamp > cutoff_time
        ]
    
    def calculate_volatility(self, window_minutes: int = 15) -> float:
        """Calculate recent BTC volatility (annualized)"""
        recent_history = self.get_price_history(window_minutes)
        
        if len(recent_history) < 3:
            return 0.0
        
        prices = [price for _, price in recent_history]
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate standard deviation of returns
        std_dev = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        
        # Annualize the volatility (assuming returns are per minute)
        # Minutes in a year = 365 * 24 * 60 = 525,600
        annualized_vol = std_dev * np.sqrt(525600)
        
        return annualized_vol

class VolatilityAdaptiveTrader:
    def __init__(self, event_id: Optional[str] = None):
        if event_id is None:
            event_id = self.generate_current_event_id()
        
        self.event_id = event_id
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.markets = []
        
        # Multi-market tracking
        self.active_markets: List[MarketInfo] = []
        self.market_subscriptions = set()
        
        # Volatility-based parameters
        self.base_market_count = 3
        self.volatility_threshold_low = 0.5   # 50% annualized
        self.volatility_threshold_high = 1.0  # 100% annualized
        self.max_markets = 7
        
        # Trading parameters
        self.min_edge_threshold = 0.05  # 5% edge required
        self.max_spread_threshold = 8   # Max 8% spread
        self.max_risk_per_trade = 0.02  # 2% of capital per trade
        
        # State tracking
        self.last_btc_price = None
        self.last_volatility = 0.0
        self.current_volatility = 0.0
        self.last_market_update_time = None
        
        # Debug counters
        self.btc_updates = 0
        self.market_updates = 0
        self.volatility_updates = 0
        
        # Display management
        self.display_line_count = 0
        self.shutdown_requested = False
    
    def generate_current_event_id(self) -> str:
        """Generate event ID based on NEXT hour in EDT time"""
        try:
            edt_tz = ZoneInfo("America/New_York")
            now = datetime.now(edt_tz)
        except ImportError:
            now = datetime.now()
        
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        next_hour = next_hour.replace(hour=(now.hour + 1) % 24)
        
        if next_hour.hour == 0 and now.hour == 23:
            next_hour = next_hour.replace(day=now.day + 1)
        
        year = next_hour.strftime("%y")
        month = next_hour.strftime("%b").upper()
        day = next_hour.strftime("%d")
        hour = next_hour.strftime("%H")
        
        event_time = f"{year}{month}{day}{hour}"
        event_id = f"KXBTCD-{event_time}"
        
        return event_id
    
    def extract_strike_price(self, ticker: str) -> float:
        """Extract strike price from ticker format"""
        try:
            parts = ticker.split("-")
            if len(parts) >= 3:
                strike_part = parts[-1]
                if strike_part.startswith("T"):
                    return float(strike_part[1:])
        except (ValueError, IndexError):
            pass
        return 0.0
    
    def calculate_adaptive_market_count(self, volatility: float, time_to_expiry_hours: float) -> int:
        """Calculate number of markets to monitor based on volatility and time"""
        base_count = self.base_market_count
        
        # Volatility adjustments
        if volatility > self.volatility_threshold_high:
            base_count += 3  # High vol: monitor more markets
        elif volatility > self.volatility_threshold_low:
            base_count += 1  # Medium vol: monitor one extra
        
        # Time decay adjustments (closer to expiry = more markets)
        if time_to_expiry_hours < 1:
            base_count += 3
        elif time_to_expiry_hours < 3:
            base_count += 2
        elif time_to_expiry_hours < 6:
            base_count += 1
        
        return min(base_count, self.max_markets)
    
    def select_target_markets(self, btc_price: float, volatility: float) -> List[MarketInfo]:
        """Select target markets based on current conditions"""
        if not self.markets or not btc_price:
            return []
        
        # Calculate time to expiry (simplified - you might want to be more precise)
        time_to_expiry_hours = 1.0  # Assuming 1 hour expiry for now
        
        # Determine how many markets to monitor
        target_count = self.calculate_adaptive_market_count(volatility, time_to_expiry_hours)
        
        # Calculate distances and find the primary (closest) market
        markets_with_distance = []
        for market in self.markets:
            strike = self.extract_strike_price(market["ticker"])
            if strike > 0:
                distance = abs(strike - btc_price)
                markets_with_distance.append({
                    'market': market,
                    'ticker': market["ticker"],
                    'strike': strike,
                    'distance': distance
                })
        
        # Sort by distance and take the closest N
        markets_with_distance.sort(key=lambda x: x['distance'])
        selected = markets_with_distance[:target_count]
        
        # Find the primary (closest) market
        primary_market = selected[0] if selected else None
        
        # Sort selected markets by strike price for ladder display
        selected.sort(key=lambda x: x['strike'])
        
        # Convert to MarketInfo objects, maintaining strike price order
        market_infos = []
        for market_data in selected:
            is_primary = (primary_market and 
                         market_data['ticker'] == primary_market['ticker'])
            
            market_info = MarketInfo(
                ticker=market_data['ticker'],
                strike=market_data['strike'],
                distance=market_data['distance'],
                is_primary=is_primary
            )
            market_infos.append(market_info)
        
        return market_infos
    
    def estimate_theoretical_probability(self, strike: float, current_price: float, 
                                       volatility: float, time_hours: float = 1.0) -> float:
        """Estimate theoretical probability using simplified Black-Scholes approach"""
        if strike <= 0 or current_price <= 0 or time_hours <= 0:
            return 0.5
        
        # Simple probability model based on normal distribution
        # This is a simplified version - you might want a more sophisticated model
        
        # Calculate the number of standard deviations
        price_ratio = current_price / strike
        log_ratio = np.log(price_ratio)
        
        # Volatility adjustment for time
        vol_sqrt_time = volatility * np.sqrt(time_hours / 8760)  # Convert to same timeframe
        
        if vol_sqrt_time <= 0:
            return 1.0 if current_price > strike else 0.0
        
        # Z-score calculation
        z_score = log_ratio / vol_sqrt_time
        
        # Probability that price will be above strike (using normal CDF approximation)
        # Using a simple approximation of the normal CDF
        prob = 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
        
        return max(0.01, min(0.99, prob))  # Clamp between 1% and 99%
    
    def analyze_market_opportunity(self, market_info: MarketInfo, btc_price: float) -> MarketInfo:
        """Analyze trading opportunity for a single market"""
        if not market_info.market_data:
            return market_info
        
        data = market_info.market_data
        
        # Calculate spread
        if data.yes_bid and data.yes_ask:
            market_info.spread = data.yes_ask - data.yes_bid
            market_info.spread_pct = (market_info.spread / data.yes_ask) * 100
        
        # Calculate implied probability
        if btc_price > market_info.strike:
            # BTC is above strike, look at YES prices
            if data.yes_ask:
                market_info.implied_prob = data.yes_ask / 100
        else:
            # BTC is below strike, look at YES bid (what market pays for YES)
            if data.yes_bid:
                market_info.implied_prob = data.yes_bid / 100
        
        # Calculate theoretical probability
        theoretical_prob = self.estimate_theoretical_probability(
            strike=market_info.strike,
            current_price=btc_price,
            volatility=self.current_volatility,
            time_hours=1.0
        )
        
        # Calculate edge
        if market_info.implied_prob:
            market_info.edge = theoretical_prob - market_info.implied_prob
        
        # Determine action
        market_info.action = self.determine_trading_action(market_info, btc_price)
        
        return market_info
    
    def determine_trading_action(self, market_info: MarketInfo, btc_price: float) -> str:
        """Determine the trading action for a market"""
        if not market_info.market_data or not market_info.spread_pct or not market_info.edge:
            return "NO_DATA"
        
        # Check spread constraint
        if market_info.spread_pct > self.max_spread_threshold:
            return "SPREAD_TOO_WIDE"
        
        # Check edge constraints
        if abs(market_info.edge) < self.min_edge_threshold:
            return "INSUFFICIENT_EDGE"
        
        # Determine direction
        if btc_price > market_info.strike:
            # BTC above strike - should YES contracts be cheap or expensive?
            if market_info.edge > self.min_edge_threshold:
                return "BUY_YES"  # Market underpricing YES
            elif market_info.edge < -self.min_edge_threshold:
                return "SELL_YES"  # Market overpricing YES
        else:
            # BTC below strike
            if market_info.edge > self.min_edge_threshold:
                return "SELL_YES"  # Market overpricing YES (should be cheap)
            elif market_info.edge < -self.min_edge_threshold:
                return "BUY_YES"   # Market underpricing YES (should be expensive)
        
        return "HOLD"
    
    def clear_display(self):
        """Clear all display lines"""
        if self.display_line_count > 0:
            for i in range(self.display_line_count):
                sys.stdout.write('\r\033[K')
                if i < self.display_line_count - 1:
                    sys.stdout.write('\033[A')
            sys.stdout.flush()
            self.display_line_count = 0
    
    def update_multiline_display(self, lines: list):
        """Update multiple lines of display"""
        self.clear_display()
        
        for i, line in enumerate(lines):
            if i > 0:
                sys.stdout.write('\n')
            sys.stdout.write(line)
        
        sys.stdout.flush()
        self.display_line_count = len(lines)
    
    def print_new_line(self, line: str):
        """Print a new line (clearing current display first)"""
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    def format_time_since(self, last_time: Optional[float]) -> str:
        """Format time since last update"""
        if last_time is None:
            return "never"
        
        seconds_ago = time.time() - last_time
        
        if seconds_ago < 60:
            return f"{int(seconds_ago)}s"
        elif seconds_ago < 3600:
            return f"{int(seconds_ago // 60)}m"
        else:
            return f"{int(seconds_ago // 3600)}h"
    
    async def initialize(self):
        """Initialize markets and setup"""
        self.print_new_line(f"Fetching markets for event: {self.event_id}")
        
        try:
            markets_data = self.client.get_markets(self.event_id)
            self.markets = markets_data.get("markets", [])
        except Exception as e:
            self.print_new_line(f"Failed to get markets: {e}")
            return False
        
        if not self.markets:
            self.print_new_line(f"No markets found for event: {self.event_id}")
            return False
        
        self.print_new_line(f"Found {len(self.markets)} markets")
        
        # Wait for initial BTC price
        self.print_new_line("Waiting for BTC price data...")
        while True:
            btc_price = self.btc_monitor.get_current_price()
            if btc_price:
                self.last_btc_price = btc_price
                self.print_new_line(f"BTC price loaded: ${btc_price:,.2f}")
                break
            await asyncio.sleep(1)
        
        # Calculate initial volatility and select initial markets
        await asyncio.sleep(2)  # Give some time for price history
        self.current_volatility = self.btc_monitor.calculate_volatility()
        
        # Select and subscribe to initial markets
        self.print_new_line("Selecting initial target markets...")
        await self.update_market_subscriptions()
        
        if not self.active_markets:
            self.print_new_line("Failed to select target markets!")
            return False
        
        self.print_new_line(f"Selected {len(self.active_markets)} initial markets")
        
        return True
    
    async def update_market_subscriptions(self):
        """Update WebSocket subscriptions based on current target markets"""
        target_markets = self.select_target_markets(self.last_btc_price, self.current_volatility)
        
        if not target_markets:
            return
        
        new_tickers = {market.ticker for market in target_markets}
        current_tickers = self.market_subscriptions.copy()
        
        # Subscribe to new markets
        for ticker in new_tickers - current_tickers:
            try:
                asyncio.create_task(self.client.subscribe_to_market(ticker))
                self.market_subscriptions.add(ticker)
            except Exception as e:
                self.print_new_line(f"Failed to subscribe to {ticker}: {e}")
        
        self.active_markets = target_markets
    
    def display_current_state(self):
        """Display current trading state"""
        try:
            edt_time = datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            edt_time = datetime.now()
        
        lines = []
        
        # Line 1: Header with BTC price and volatility
        vol_indicator = "🔥" if self.current_volatility > self.volatility_threshold_high else \
                       "📈" if self.current_volatility > self.volatility_threshold_low else "📊"
        
        line1 = (f"{edt_time.strftime('%H:%M:%S')} | "
                f"BTC: ${self.last_btc_price:,.2f} | "
                f"Vol: {self.current_volatility:.1%} {vol_indicator} | "
                f"Markets: {len(self.active_markets)}")
        lines.append(line1)
        
        # Line 2: Market ladder overview
        if self.active_markets:
            sorted_markets = sorted(self.active_markets, key=lambda m: m.strike)
            strike_labels = []
            
            for market in sorted_markets:
                if market.is_primary:
                    strike_labels.append(f"${market.strike:,.0f}🎯")
                else:
                    strike_labels.append(f"${market.strike:,.0f}")
            
            line2 = f"Ladder: {' | '.join(strike_labels)}"
            lines.append(line2)
        
        # Lines 3+: Individual market analysis with YES/NO prices
        opportunities = []
        if self.active_markets:
            sorted_markets = sorted(self.active_markets, key=lambda m: m.strike)
            
            for market in sorted_markets:
                if market.market_data:
                    market = self.analyze_market_opportunity(market, self.last_btc_price)
                    data = market.market_data
                    
                    # Primary indicator
                    primary_indicator = "🎯" if market.is_primary else "  "
                    
                    # Action emoji
                    action_emoji = {
                        "BUY_YES": "🟢",
                        "SELL_YES": "🔴", 
                        "HOLD": "⚪",
                        "SPREAD_TOO_WIDE": "📏",
                        "INSUFFICIENT_EDGE": "⚖️",
                        "NO_DATA": "❓"
                    }.get(market.action, "⚪")
                    
                    # Format YES/NO prices
                    if data.yes_bid and data.yes_ask:
                        yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
                        no_bid = 100 - data.yes_ask
                        no_ask = 100 - data.yes_bid
                        no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
                        spread_text = f"Spread: {market.spread:.0f}¢" if market.spread else ""
                    else:
                        yes_prices = "YES: --/--"
                        no_prices = "NO: --/--"
                        spread_text = "No data"
                    
                    # Edge information
                    edge_text = f"Edge: {market.edge:+.1%}" if market.edge else ""
                    
                    # Build the line
                    market_line = (f"{primary_indicator}${market.strike:,.0f}: "
                                  f"{yes_prices} | {no_prices} | "
                                  f"{spread_text}")
                    
                    if edge_text:
                        market_line += f" | {edge_text}"
                    
                    market_line += f" {action_emoji}"
                    
                    lines.append(market_line)
                    
                    # Collect actionable opportunities
                    if market.action in ["BUY_YES", "SELL_YES"]:
                        opportunities.append(market)
        
        # Final line: Opportunities summary (only if there are any)
        if opportunities:
            opp_summary = f"🚨 {len(opportunities)} TRADING OPPORTUNITIES"
            lines.append(opp_summary)
        
        self.update_multiline_display(lines)
    
    async def run_trading_loop(self):
        """Main trading loop"""
        if not await self.initialize():
            return
        
        self.print_new_line("Starting Volatility-Adaptive 3-Market Ladder...")
        self.print_new_line("=" * 80)
        
        # Setup WebSocket message counting
        original_handler = self.client._handle_ws_message
        
        def counting_handler(msg):
            self.market_updates += 1
            self.last_market_update_time = time.time()
            original_handler(msg)
        
        self.client._handle_ws_message = counting_handler
        
        try:
            while not self.shutdown_requested:
                # Update BTC price and volatility
                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.btc_updates += 1
                
                # Update volatility
                new_volatility = self.btc_monitor.calculate_volatility()
                if abs(new_volatility - self.current_volatility) > 0.1:  # 10% change
                    self.current_volatility = new_volatility
                    self.volatility_updates += 1
                    
                    # Recalculate target markets
                    await self.update_market_subscriptions()
                
                # Update market data for active markets
                for market in self.active_markets:
                    market.market_data = self.client.get_mid_prices(market.ticker)
                
                # Display current state
                self.display_current_state()
                
                await asyncio.sleep(0.5)  # Update twice per second
                
        except KeyboardInterrupt:
            self.print_new_line("\nShutdown requested by user...")
        except asyncio.CancelledError:
            self.print_new_line("\nTask cancelled...")
        except Exception as e:
            self.print_new_line(f"\nUnexpected error: {e}")
        finally:
            self.shutdown_requested = True
            try:
                await self.client.close()
            except:
                pass
            self.print_new_line("Bot stopped.")

async def main():
    trader = VolatilityAdaptiveTrader()
    try:
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
