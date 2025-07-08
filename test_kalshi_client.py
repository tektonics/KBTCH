import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from kalshi_bot.kalshi_client import KalshiClient

class BTCPriceMonitor:
    def __init__(self, price_file: str = "aggregate_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5  # Check file every 500ms
    
    def get_current_price(self) -> Optional[float]:
        """Get BTC price with efficient file monitoring"""
        now = time.time()
        
        # Don't check file too frequently
        if now - self.last_check < self.check_interval:
            return self.last_price
        
        self.last_check = now
        
        try:
            if not self.price_file.exists():
                return None
            
            # Only read if file was modified
            current_modified = self.price_file.stat().st_mtime
            if current_modified == self.last_modified:
                return self.last_price
            
            with open(self.price_file, 'r') as f:
                data = json.load(f)
                price = data.get("price")
                
                if price and isinstance(price, (int, float)) and price > 0:
                    self.last_price = float(price)
                    self.last_modified = current_modified
                    return self.last_price
                    
        except (json.JSONDecodeError, IOError):
            pass
        
        return None

class KalshiMarketAnalyzer:
    def __init__(self, event_id: Optional[str] = None):
        if event_id is None:
            event_id = self.generate_current_event_id()
        
        self.event_id = event_id
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.markets = []
        self.target_market = None
        self.last_btc_price = None
        self.last_market_update = None
        
        # Debug counters
        self.btc_updates = 0
        self.market_updates = 0
        self.ws_messages = 0
        
        # Time tracking for "seconds since last update"
        self.last_btc_update_time = None
        self.last_market_update_time = None
        self.last_ws_message_time = None
        
        # Display state
        self.last_display_lines = []
        self.display_line_count = 0
        
        # Shutdown flag
        self.shutdown_requested = False
    
    def get_edt_time(self) -> datetime:
        """Get current time in EDT timezone"""
        try:
            edt_tz = ZoneInfo("America/New_York")
            return datetime.now(edt_tz)
        except ImportError:
            try:
                import pytz
                edt_tz = pytz.timezone('America/New_York')
                return datetime.now(edt_tz)
            except ImportError:
                return datetime.now()
    
    def generate_current_event_id(self) -> str:
        """Generate event ID based on NEXT hour in EDT time"""
        now = self.get_edt_time()
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
        
        print(f"Generated event ID: {event_id}")
        print(f"Current time (EDT): {now.strftime('%B %d, %Y at %H:%M %Z')}")
        print(f"Target time (EDT): {next_hour.strftime('%B %d, %Y at %H:00 %Z')}")
        
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
    
    def clear_display(self):
        """Clear all display lines"""
        if self.display_line_count > 0:
            # Move cursor up and clear each line
            for i in range(self.display_line_count):
                sys.stdout.write('\r\033[K')  # Clear current line
                if i < self.display_line_count - 1:
                    sys.stdout.write('\033[A')  # Move cursor up
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
        """Format time since last update as '5s', '2m', etc."""
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
        """Initialize markets and select target"""
        print(f"Fetching available markets for event: {self.event_id}")
        
        try:
            markets_data = self.client.get_markets(self.event_id)
            self.markets = markets_data.get("markets", [])
        except Exception as e:
            print(f"Failed to get markets: {e}")
            return False
        
        if not self.markets:
            print(f"No markets found for event: {self.event_id}")
            return False
        
        # Show summary instead of listing all markets
        strikes = [self.extract_strike_price(market["ticker"]) for market in self.markets]
        strikes = [s for s in strikes if s > 0]  # Filter out invalid strikes
        
        if strikes:
            min_strike = min(strikes)
            max_strike = max(strikes)
            print(f"Found {len(self.markets)} markets with strikes from ${min_strike:,.0f} to ${max_strike:,.0f}")
        else:
            print(f"Found {len(self.markets)} markets (unable to parse strike prices)")
        
        # Wait for initial BTC price
        print("Waiting for BTC price data...")
        while True:
            btc_price = self.btc_monitor.get_current_price()
            if btc_price:
                print(f"BTC price loaded: ${btc_price:,.2f}")
                self.last_btc_price = btc_price
                self.last_btc_update_time = time.time()
                break
            await asyncio.sleep(1)
        
        # Select target market
        self.update_target_market()
        return True
    
    def update_target_market(self):
        """Update target market based on current BTC price"""
        if not self.last_btc_price or not self.markets:
            return
        
        closest_market = min(
            self.markets,
            key=lambda m: abs(self.extract_strike_price(m["ticker"]) - self.last_btc_price)
        )
        
        new_target = closest_market["ticker"]
        if new_target != self.target_market:
            self.target_market = new_target
            strike = self.extract_strike_price(new_target)
            diff = abs(strike - self.last_btc_price)
            self.print_new_line(f"Switched to market: {self.target_market}")
            self.print_new_line(f"Strike price: ${strike:,.0f} (${diff:,.0f} from current BTC)")
            self.print_new_line(f"Streaming market: {self.target_market}")
    
    async def debug_websocket_messages(self):
        """Debug WebSocket message flow"""
        print(f"\n{'='*60}")
        print(f"DEBUGGING WEBSOCKET MESSAGES")
        print(f"{'='*60}")
        
        # Override the client's message handler to add debug info
        original_handler = self.client._handle_ws_message
        
        def debug_handler(msg):
            self.ws_messages += 1
            self.last_ws_message_time = time.time()
            msg_type = msg.get("type", "unknown")
            
            if msg_type == "ticker_v2":
                data = msg.get("msg", {})
                market_ticker = data.get("market_ticker", "unknown")
                
                # Show ALL fields in ticker_v2 message
                debug_line = f"[{self.ws_messages}] ticker_v2 FULL: {data}"
                self.print_new_line(debug_line)
            
            elif msg_type == "orderbook_delta":
                data = msg.get("msg", {})
                market_ticker = data.get("market_ticker", "unknown")
                debug_line = f"[{self.ws_messages}] orderbook_delta: {market_ticker}"
                self.print_new_line(debug_line)
            
            else:
                debug_line = f"[{self.ws_messages}] {msg_type}: {msg}"
                self.print_new_line(debug_line)
            
            # Call original handler
            original_handler(msg)
        
        # Temporarily override for debugging
        self.client._handle_ws_message = debug_handler
        
        # Start WebSocket subscription
        try:
            await self.client.subscribe_to_market(self.target_market)
        except Exception as e:
            self.print_new_line(f"WebSocket error: {e}")
    
    async def run_analysis(self):
        """Main analysis loop with proper cleanup"""
        if not await self.initialize():
            return
        
        # Wrap the message handler to count messages in normal mode too
        original_handler = self.client._handle_ws_message
        
        def counting_handler(msg):
            self.ws_messages += 1
            self.last_ws_message_time = time.time()
            original_handler(msg)
        
        self.client._handle_ws_message = counting_handler
        
        print(f"\n{'='*60}")
        print(f"MONITORING: {self.target_market}")
        print(f"{'='*60}")
        
        # Start WebSocket subscription
        ws_task = asyncio.create_task(
            self.client.subscribe_to_market(self.target_market)
        )
        
        # Start monitoring loop
        monitor_task = asyncio.create_task(self.monitor_loop())
        
        try:
            await asyncio.gather(ws_task, monitor_task, return_exceptions=True)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean shutdown
            self.shutdown_requested = True
            ws_task.cancel()
            monitor_task.cancel()
            
            # Wait a bit for tasks to clean up
            try:
                await asyncio.wait_for(asyncio.gather(ws_task, monitor_task, return_exceptions=True), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            self.print_new_line("\nShutting down gracefully...")
            await self.client.close()
    
    async def monitor_loop(self):
        """Monitor prices and display updates"""
        while not self.shutdown_requested:
            try:
                # Check for BTC price updates
                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.btc_updates += 1
                    self.last_btc_update_time = time.time()
                    self.update_target_market()
                
                # Get market data
                market_data = self.client.get_mid_prices(self.target_market)
                
                if market_data and (market_data.yes_bid is not None or market_data.yes_ask is not None):
                    # Always update display with current data
                    self.display_current_state(market_data)
                    
                    # Check if this is new market data
                    if self.last_market_update != market_data.timestamp:
                        self.market_updates += 1
                        self.last_market_update = market_data.timestamp
                        self.last_market_update_time = time.time()
                else:
                    # Show waiting state
                    self.display_waiting_state()
                
                await asyncio.sleep(0.1)  # Update display more frequently
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.shutdown_requested:
                    self.print_new_line(f"Error in monitoring: {e}")
                    await asyncio.sleep(1)
    
    def display_current_state(self, market_data):
        """Display current state on multiple updating lines"""
        strike = self.extract_strike_price(self.target_market)
        edt_time = self.get_edt_time()
        
        # Line 1: Time, BTC, Strike, and position
        diff = self.last_btc_price - strike if self.last_btc_price and strike else 0
        if diff > 0:
            position_str = f"🔥 BTC +${diff:,.0f} ABOVE strike"
        else:
            position_str = f"📉 BTC ${abs(diff):,.0f} below strike"
        
        line1 = f"{edt_time.strftime('%H:%M:%S')} | BTC: ${self.last_btc_price:,.2f} | Strike: ${strike:,.0f} | {position_str}"
        
        # Line 2: YES market data
        yes_parts = ["YES:"]
        if market_data.yes_bid and market_data.yes_ask:
            yes_spread = market_data.yes_ask - market_data.yes_bid
            yes_parts.append(f"Bid: {market_data.yes_bid:.0f}¢")
            yes_parts.append(f"Ask: {market_data.yes_ask:.0f}¢")
            yes_parts.append(f"Spread: {yes_spread:.0f}¢")
        else:
            yes_parts.append("No market data")
        
        if market_data.price:
            yes_parts.append(f"Last Trade: {market_data.price:.0f}¢")
        
        line2 = " | ".join(yes_parts)
        
        # Line 3: NO market data (calculated from YES if not available)
        no_parts = ["NO: "]
        if market_data.no_bid and market_data.no_ask:
            no_spread = market_data.no_ask - market_data.no_bid
            no_parts.append(f"Bid: {market_data.no_bid:.0f}¢")
            no_parts.append(f"Ask: {market_data.no_ask:.0f}¢")
            no_parts.append(f"Spread: {no_spread:.0f}¢")
        elif market_data.yes_bid and market_data.yes_ask:
            # Calculate NO prices from YES (they should add up to 100¢)
            no_bid = 100 - market_data.yes_ask
            no_ask = 100 - market_data.yes_bid
            no_spread = no_ask - no_bid
            no_parts.append(f"Bid: {no_bid:.0f}¢")
            no_parts.append(f"Ask: {no_ask:.0f}¢")
            no_parts.append(f"Spread: {no_spread:.0f}¢")
        else:
            no_parts.append("No market data")
        
        line3 = " | ".join(no_parts)
        
        # Line 4: Update counters with time since last update
        btc_time = self.format_time_since(self.last_btc_update_time)
        market_time = self.format_time_since(self.last_market_update_time)
        ws_time = self.format_time_since(self.last_ws_message_time)
        
        line4 = f"Last Updates → BTC: {btc_time} | Market: {market_time} | WebSocket: {ws_time}"
        
        # Update all lines at once
        self.update_multiline_display([line1, line2, line3, line4])
    
    def display_waiting_state(self):
        """Display waiting state on multiple lines"""
        edt_time = self.get_edt_time()
        
        btc_time = self.format_time_since(self.last_btc_update_time)
        ws_time = self.format_time_since(self.last_ws_message_time)
        
        line1 = f"{edt_time.strftime('%H:%M:%S')} | Waiting for market data..."
        line2 = f"Last Updates → BTC: {btc_time} | WebSocket: {ws_time}"
        line3 = f"Market: {self.target_market or 'Not selected'}"
        
        self.update_multiline_display([line1, line2, line3])

async def main():
    import sys
    
    try:
        # Check if debug mode is requested
        if len(sys.argv) > 1 and sys.argv[1] == "debug":
            print("=== DEBUG MODE ===")
            analyzer = KalshiMarketAnalyzer()
            if await analyzer.initialize():
                await analyzer.debug_websocket_messages()
        else:
            analyzer = KalshiMarketAnalyzer()
            await analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated...")

if __name__ == "__main__":
    asyncio.run(main())
