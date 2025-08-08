"""
Comprehensive monitoring service for the KBTCH trading system.
Displays real-time market data, prices, signals, and system status.
"""

import asyncio
import threading
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from event_bus import event_bus, EventTypes, Event
from strategy_engine import StrategyEngine

@dataclass
class MarketSnapshot:
    ticker: str
    strike_price: float
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    last_update: Optional[float] = None
    update_count: int = 0

@dataclass
class SignalRecord:
    timestamp: float
    market_ticker: str
    signal_type: str
    confidence: float
    current_brti: float
    market_yes_price: float
    market_no_price: float
    reason: str

class MonitoringService:
    def __init__(self):
        # Core data
        self.current_brti: Optional[float] = None
        self.brti_history: deque = deque(maxlen=100)  # Last 100 BRTI values
        self.markets: Dict[str, MarketSnapshot] = {}
        self.signals: List[SignalRecord] = []
        self.strategy: Optional[StrategyEngine] = None
        
        # Statistics
        self.stats = {
            'price_updates': 0,
            'market_updates': 0,
            'signals_generated': 0,
            'start_time': time.time(),
            'last_brti_update': None,
            'last_market_update': None
        }
        
        # Display control
        self.display_lines = 0
        self.update_interval = 1.0  # Update display every second
        self.last_display_update = 0
        
        # Setup
        self.setup_event_subscriptions()
        
    def setup_event_subscriptions(self):
        """Subscribe to all relevant events"""
        
        def price_update_handler(event: Event):
            self.stats['price_updates'] += 1
            self.stats['last_brti_update'] = time.time()
            
            brti_price = event.data.get('brti_price')
            if brti_price:
                self.current_brti = brti_price
                self.brti_history.append((time.time(), brti_price))
        
        def market_data_handler(event: Event):
            self.stats['market_updates'] += 1
            self.stats['last_market_update'] = time.time()
            
            ticker = event.data.get('market_ticker')
            if not ticker:
                return
                
            # Update or create market snapshot
            if ticker not in self.markets:
                strike_price = event.data.get('strike_price', 0.0)
                self.markets[ticker] = MarketSnapshot(ticker=ticker, strike_price=strike_price)
            
            market = self.markets[ticker]
            market.yes_bid = event.data.get('yes_bid')
            market.yes_ask = event.data.get('yes_ask')
            market.no_bid = event.data.get('no_bid')
            market.no_ask = event.data.get('no_ask')
            market.last_update = time.time()
            market.update_count += 1
            
            # Calculate spread
            if market.yes_bid and market.yes_ask:
                market.spread = market.yes_ask - market.yes_bid
                market.spread_pct = (market.spread / market.yes_ask) * 100 if market.yes_ask > 0 else 0
        
        def signal_handler(event: Event):
            self.stats['signals_generated'] += 1
            
            signal = SignalRecord(
                timestamp=time.time(),
                market_ticker=event.data.get('market_ticker', ''),
                signal_type=event.data.get('signal_type', ''),
                confidence=event.data.get('confidence', 0.0),
                current_brti=event.data.get('current_brti', 0.0),
                market_yes_price=event.data.get('market_yes_price', 0.0),
                market_no_price=event.data.get('market_no_price', 0.0),
                reason=event.data.get('reason', '')
            )
            
            self.signals.append(signal)
            
            # Keep only last 50 signals
            if len(self.signals) > 50:
                self.signals = self.signals[-50:]
        
        # Subscribe to events
        event_bus.subscribe(EventTypes.PRICE_UPDATE, price_update_handler)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, market_data_handler)
        event_bus.subscribe(EventTypes.SIGNAL_GENERATED, signal_handler)
    
    def clear_display(self):
        """Clear the terminal display"""
        if self.display_lines > 0:
            # Move cursor up and clear lines
            for _ in range(self.display_lines):
                sys.stdout.write('\033[F\033[K')
            sys.stdout.flush()
        self.display_lines = 0
    
    def print_line(self, line: str = ""):
        """Print a line and track display lines"""
        print(line)
        self.display_lines += 1
    
    def get_brti_trend(self) -> str:
        """Calculate BRTI trend from recent history"""
        if len(self.brti_history) < 2:
            return "─"
        
        recent_prices = [price for _, price in list(self.brti_history)[-10:]]
        if len(recent_prices) < 2:
            return "─"
        
        recent_change = recent_prices[-1] - recent_prices[0]
        if recent_change > 50:
            return "↑↑"
        elif recent_change > 10:
            return "↑"
        elif recent_change < -50:
            return "↓↓"
        elif recent_change < -10:
            return "↓"
        else:
            return "─"
    
    def format_market_table(self) -> List[str]:
        """Format markets into a nice table"""
        if not self.markets:
            return ["No markets data available"]
        
        lines = []
        lines.append("ACTIVE MARKETS")
        lines.append("=" * 80)
        
        # Header
        header = f"{'Market':<25} {'Strike':<12} {'YES Bid/Ask':<15} {'NO Bid/Ask':<15} {'Spread':<10} {'Updates':<8}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Sort markets by strike price
        sorted_markets = sorted(self.markets.values(), key=lambda m: m.strike_price)
        
        for market in sorted_markets:
            # Determine if this market is close to current BRTI
            proximity = ""
            if self.current_brti and market.strike_price:
                diff = abs(self.current_brti - market.strike_price)
                if diff < 1000:
                    proximity = "🎯"
                elif diff < 2000:
                    proximity = "📍"
            
            # Format market name
            market_name = market.ticker[-12:] if len(market.ticker) > 12 else market.ticker
            market_display = f"{proximity}{market_name}"
            
            # Format prices
            yes_prices = f"{market.yes_bid or 0:.0f}/{market.yes_ask or 0:.0f}" if market.yes_bid and market.yes_ask else "--/--"
            no_prices = f"{market.no_bid or 0:.0f}/{market.no_ask or 0:.0f}" if market.no_bid and market.no_ask else "--/--"
            
            # Format spread
            spread_display = f"{market.spread:.0f}¢" if market.spread else "--"
            if market.spread_pct:
                spread_display += f" ({market.spread_pct:.1f}%)"
            
            # Age indicator
            age_indicator = ""
            if market.last_update:
                age = time.time() - market.last_update
                if age > 30:
                    age_indicator = "🔴"  # Stale data
                elif age > 10:
                    age_indicator = "🟡"  # Old data
                else:
                    age_indicator = "🟢"  # Fresh data
            
            line = f"{market_display:<25} ${market.strike_price:<11.0f} {yes_prices:<15} {no_prices:<15} {spread_display:<10} {market.update_count:<3}{age_indicator}"
            lines.append(line)
        
        return lines
    
    def format_signals_table(self) -> List[str]:
        """Format recent signals into a table"""
        if not self.signals:
            return ["No trading signals generated yet"]
        
        lines = []
        lines.append(f"RECENT SIGNALS ({len(self.signals)} total)")
        lines.append("=" * 80)
        
        # Show last 10 signals
        recent_signals = self.signals[-10:]
        
        for signal in reversed(recent_signals):  # Most recent first
            timestamp_str = datetime.fromtimestamp(signal.timestamp).strftime("%H:%M:%S")
            
            # Color code signal type
            signal_icon = {
                'BUY_YES': '🟢',
                'BUY_NO': '🔴', 
                'SELL_YES': '🟡',
                'SELL_NO': '🟠'
            }.get(signal.signal_type, '⚪')
            
            market_short = signal.market_ticker[-12:] if len(signal.market_ticker) > 12 else signal.market_ticker
            
            line = f"{timestamp_str} {signal_icon} {signal.signal_type:<8} {market_short:<15} Conf:{signal.confidence:.2f} BRTI:${signal.current_brti:,.0f}"
            lines.append(line)
            
            # Add reason on next line, truncated
            reason_short = signal.reason[:60] + "..." if len(signal.reason) > 60 else signal.reason
            lines.append(f"         └─ {reason_short}")
        
        return lines
    
    def format_system_status(self) -> List[str]:
        """Format system status information"""
        lines = []
        runtime = time.time() - self.stats['start_time']
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # BRTI Status
        brti_display = f"${self.current_brti:,.2f}" if self.current_brti else "No data"
        trend = self.get_brti_trend()
        brti_age = ""
        if self.stats['last_brti_update']:
            age = time.time() - self.stats['last_brti_update']
            if age > 5:
                brti_age = f" (stale {age:.0f}s)"
        
        lines.append(f"SYSTEM STATUS - {current_time} (Runtime: {runtime:.0f}s)")
        lines.append("=" * 50)
        lines.append(f"BRTI Price:        {brti_display} {trend}{brti_age}")
        lines.append(f"Active Markets:    {len(self.markets)}")
        lines.append(f"Price Updates:     {self.stats['price_updates']}")
        lines.append(f"Market Updates:    {self.stats['market_updates']}")
        lines.append(f"Signals Generated: {self.stats['signals_generated']}")
        
        # Update rates
        if runtime > 0:
            price_rate = self.stats['price_updates'] / runtime
            market_rate = self.stats['market_updates'] / runtime
            lines.append(f"Price Rate:        {price_rate:.1f}/sec")
            lines.append(f"Market Rate:       {market_rate:.1f}/sec")
        
        # Strategy Engine Status - inferred from event activity
        strategy_active = self.stats['signals_generated'] > 0 or self.stats['market_updates'] > 0
        lines.append(f"Strategy Active:   {'✅' if strategy_active else '❓'}")
        lines.append(f"Strategy Signals:  {self.stats['signals_generated']}")
        
        return lines
    
    def display_dashboard(self):
        """Display the complete monitoring dashboard"""
        self.clear_display()
        
        # Header
        self.print_line("🎯 KBTCH TRADING SYSTEM MONITOR")
        self.print_line("=" * 80)
        self.print_line()
        
        # System Status (left column)
        status_lines = self.format_system_status()
        for line in status_lines:
            self.print_line(line)
        
        self.print_line()
        
        # Markets Table
        market_lines = self.format_market_table()
        for line in market_lines:
            self.print_line(line)
        
        self.print_line()
        
        # Recent Signals
        signal_lines = self.format_signals_table()
        for line in signal_lines:
            self.print_line(line)
        
        self.print_line()
        self.print_line("🔄 Live monitoring... Press Ctrl+C to stop")
        
        sys.stdout.flush()
    
    def initialize_strategy_reference(self):
        """Strategy status will be inferred from events - no direct reference needed"""
        # We don't create a strategy engine here - just monitor events from the existing one
        pass
    
    async def run_monitoring(self):
        """Main monitoring loop"""
        self.initialize_strategy_reference()
        
        print("🚀 Starting KBTCH Monitoring Service...")
        print("📊 Waiting for data from UDM and KMS...")
        print("=" * 60)
        
        # Initial display
        self.display_dashboard()
        self.last_display_update = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Update display at regular intervals
                if current_time - self.last_display_update >= self.update_interval:
                    self.display_dashboard()
                    self.last_display_update = current_time
                
                await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting
                
        except KeyboardInterrupt:
            self.clear_display()
            print("\n👋 Monitoring service stopped by user")
            
            # Final summary
            print("\n📊 FINAL MONITORING SUMMARY")
            print("=" * 40)
            runtime = time.time() - self.stats['start_time']
            print(f"Runtime:           {runtime:.1f} seconds")
            print(f"Price Updates:     {self.stats['price_updates']}")
            print(f"Market Updates:    {self.stats['market_updates']}")
            print(f"Markets Tracked:   {len(self.markets)}")
            print(f"Signals Generated: {self.stats['signals_generated']}")
            
            if self.signals:
                print(f"Last Signal:       {self.signals[-1].signal_type} on {self.signals[-1].market_ticker}")

async def main():
    """Entry point for monitoring service"""
    monitor = MonitoringService()
    await monitor.run_monitoring()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Monitoring terminated")
    except Exception as e:
        print(f"\n💥 Error: {e}")
