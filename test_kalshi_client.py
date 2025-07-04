import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from kalshi_bot.kalshi_client import KalshiClient

class BTCPriceMonitor:
    def __init__(self, price_file: str = "btc_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5  # Check file every 500ms instead of every loop
    
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
    def __init__(self, event_id: str = "KXBTCD-25JUN1610"):
        self.event_id = event_id
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.markets = []
        self.target_market = None
        self.last_btc_price = None
        self.last_market_update = None
        
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
    
    async def initialize(self):
        """Initialize markets and select target"""
        print("Fetching available markets...")
        
        markets_data = self.client.get_markets(self.event_id)
        self.markets = markets_data.get("markets", [])
        
        if not self.markets:
            print(f"No markets found for event: {self.event_id}")
            return False
        
        print("Available market tickers:")
        for market in self.markets:
            strike = self.extract_strike_price(market["ticker"])
            print(f"  {market['ticker']} (Strike: ${strike:,.0f})")
        
        # Wait for initial BTC price
        print("Waiting for BTC price data...")
        while True:
            btc_price = self.btc_monitor.get_current_price()
            if btc_price:
                print(f"BTC price loaded: ${btc_price:,.2f}")
                self.last_btc_price = btc_price
                break
            await asyncio.sleep(1)
        
        # Select target market
        self.update_target_market()
        return True
    
    def update_target_market(self):
        """Update target market based on current BTC price"""
        if not self.last_btc_price or not self.markets:
            return
        
        # Find closest market
        closest_market = min(
            self.markets,
            key=lambda m: abs(self.extract_strike_price(m["ticker"]) - self.last_btc_price)
        )
        
        new_target = closest_market["ticker"]
        if new_target != self.target_market:
            self.target_market = new_target
            strike = self.extract_strike_price(new_target)
            diff = abs(strike - self.last_btc_price)
            print(f"\nSwitched to market: {self.target_market}")
            print(f"Strike price: ${strike:,.0f} (${diff:,.0f} from current BTC)")
            print(f"Streaming market: {self.target_market}")
    
    async def run_analysis(self):
        """Main analysis loop"""
        if not await self.initialize():
            return
        
        # Start WebSocket subscription
        ws_task = asyncio.create_task(
            self.client.subscribe_to_market(self.target_market)
        )
        
        # Start monitoring loop
        monitor_task = asyncio.create_task(self.monitor_loop())
        
        try:
            await asyncio.gather(ws_task, monitor_task)
        except KeyboardInterrupt:
            print("\nShutting down...")
            await self.client.close()
    
    async def monitor_loop(self):
        """Monitor prices and display updates"""
        print(f"\n{'='*60}")
        print(f"MONITORING: {self.target_market}")
        print(f"{'='*60}")
        
        while True:
            try:
                # Check for BTC price updates
                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    print(f"\n🔄 BTC Price Update: ${current_btc:,.2f} (was ${self.last_btc_price:,.2f})")
                    self.last_btc_price = current_btc
                    self.update_target_market()
                
                # Get market data
                market_data = self.client.get_mid_prices(self.target_market)
                
                if market_data and market_data.price is not None:
                    # Only display if data changed
                    if self.last_market_update != market_data.timestamp:
                        self.display_market_data(market_data)
                        self.last_market_update = market_data.timestamp
                
                await asyncio.sleep(0.5)  # Check twice per second
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                await asyncio.sleep(1)
    
    def display_market_data(self, market_data):
        """Display current market data with analysis"""
        strike = self.extract_strike_price(self.target_market)
        
        print(f"\n📊 Market Update - {time.strftime('%H:%M:%S')}")
        print(f"BTC: ${self.last_btc_price:,.2f} | Strike: ${strike:,.0f}")
        
        if market_data.price:
            print(f"Market Price: ${market_data.price:.2f}")
        
        if market_data.yes_bid and market_data.yes_ask:
            spread = market_data.yes_ask - market_data.yes_bid
            print(f"YES: ${market_data.yes_bid:.2f} / ${market_data.yes_ask:.2f} (spread: ${spread:.2f})")
        
        if market_data.no_bid and market_data.no_ask:
            spread = market_data.no_ask - market_data.no_bid
            print(f"NO:  ${market_data.no_bid:.2f} / ${market_data.no_ask:.2f} (spread: ${spread:.2f})")
        
        # Show opportunity analysis
        if self.last_btc_price and strike:
            if self.last_btc_price > strike:
                print(f"🔥 BTC is ${self.last_btc_price - strike:,.0f} ABOVE strike!")
            else:
                print(f"📉 BTC is ${strike - self.last_btc_price:,.0f} below strike")
        
        print("-" * 40)

async def main():
    analyzer = KalshiMarketAnalyzer()
    await analyzer.run_analysis()

if __name__ == "__main__":
    asyncio.run(main())
