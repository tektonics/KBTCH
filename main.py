import asyncio
import threading
import logging
import sys
import os
import time
import signal
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UDM import UnifiedCryptoManager
from KMS import KalshiClient
from strategy_engine import StrategyEngine
from event_bus import event_bus, EventTypes

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Quiet the noisy loggers
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class TradingSystemManager:
    def __init__(self):
        self.udm: Optional[UnifiedCryptoManager] = None
        self.kms: Optional[KalshiClient] = None
        self.strategy: Optional[StrategyEngine] = None
        self.udm_thread: Optional[threading.Thread] = None
        self.kms_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'price_updates': 0,
            'market_data_updates': 0,
            'signals_generated': 0,
            'start_time': time.time()
        }
        
        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self.running = False
    
    def setup_event_monitoring(self):
        """Set up event monitoring for statistics and logging"""
        
        def price_update_handler(event):
            self.stats['price_updates'] += 1
            brti_price = event.data.get('brti_price')
            if self.stats['price_updates'] % 50 == 0:  # Log every 50th update
                logger.info(f"📊 Price Update #{self.stats['price_updates']}: BRTI ${brti_price:,.2f}")
        
        def market_data_handler(event):
            self.stats['market_data_updates'] += 1
            
        
        def signal_handler(event):
            self.stats['signals_generated'] += 1
            logger.info(f"🚨 TRADING SIGNAL #{self.stats['signals_generated']}:")
            logger.info(f"    Market: {event.data.get('market_ticker')}")
            logger.info(f"    Signal: {event.data.get('signal_type')}")
            logger.info(f"    Confidence: {event.data.get('confidence', 0):.2f}")
            logger.info(f"    Reason: {event.data.get('reason')}")
        
        # Subscribe to events
        event_bus.subscribe(EventTypes.PRICE_UPDATE, price_update_handler)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, market_data_handler)
        event_bus.subscribe(EventTypes.SIGNAL_GENERATED, signal_handler)
        
        logger.info("✅ Event monitoring setup complete")
    
    def start_udm(self):
        logger.info("🚀 Starting UDM (Unified Data Manager)...")
        
        self.udm = UnifiedCryptoManager()
        
        def udm_runner():
            try:
                asyncio.run(self.udm.run_unified_system())
            except Exception as e:
                logger.error(f"UDM error: {e}")
        
        self.udm_thread = threading.Thread(target=udm_runner, daemon=True)
        self.udm_thread.start()
        logger.info("✅ UDM started in background thread")
    
    async def start_kms(self):
        """Start KMS (Kalshi Market Service)"""
        logger.info("🚀 Starting KMS (Kalshi Market Service)...")
        
        try:
            self.kms = KalshiClient()
            self.kms_task = asyncio.create_task(
                self.kms.start_adaptive_market_tracking()
            )
            logger.info("✅ KMS started successfully")
        except Exception as e:
            logger.error(f"Failed to start KMS: {e}")
            raise
    
    def start_strategy_engine(self):
        """Start the strategy engine"""
        logger.info("🚀 Starting Strategy Engine...")
        
        try:
            self.strategy = StrategyEngine()
            logger.info("✅ Strategy Engine started successfully")
        except Exception as e:
            logger.error(f"Failed to start Strategy Engine: {e}")
            raise
    
    def print_status(self):
        """Print current system status with Kalshi market display"""
        # Clear screen for clean display
        os.system('cls' if os.name == 'nt' else 'clear')
        
        runtime = time.time() - self.stats['start_time']
        
        print(f"📊 KBTCH TRADING SYSTEM - Runtime: {runtime:.0f}s")
        print("=" * 70)
        
        print(f"Trading Signals:         {self.stats['signals_generated']}")
        
# Strategy Engine Status
        if self.strategy:
            status = self.strategy.get_status()
            print(f"Active Markets:          {status.get('active_markets', 0)}")
            print(f"UDM Data Flow:           {status.get('udm_active', '❌')}")  # Add this
            print(f"KMS Data Flow:           {status.get('kms_active', '❌')}")  # Add this

        # Kalshi Market Display
        if self.kms and self.kms.active_market_info:
            print("\n" + "─" * 70)
            
            # Header with BTC price and volatility
            btc_price = self.kms.last_btc_price or 0
            volatility = self.kms.current_volatility or 0
            event_ticker = self.kms.event_ticker or "Unknown"
            
            print(f"📈 KALSHI MARKETS ({event_ticker}) - BTC: ${btc_price:,.2f} | Vol: {volatility:.2f}")
            
            # Strike ladder
            sorted_markets = sorted(self.kms.active_market_info.values(), key=lambda m: m.strike)
            if sorted_markets:
                strike_labels = []
                for market_info in sorted_markets:
                    label = f"${market_info.strike:,.0f}🎯" if market_info.is_primary else f"${market_info.strike:,.0f}"
                    strike_labels.append(label)
                print(f"Strike Ladder: {' | '.join(strike_labels)}")
                
                # Market details (show top 3 to keep display compact)
                for i, market_info in enumerate(sorted_markets[:3]):
                    data = market_info.market_data
                    if data and data.yes_bid is not None and data.yes_ask is not None:
                        primary_indicator = "🎯" if market_info.is_primary else " "
                        yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
                        no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
                        no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
                        spread = data.yes_ask - data.yes_bid
                        spread_text = f"Spread: {spread:.0f}¢"
                        
                        print(f"{primary_indicator}${market_info.strike:,.0f}: {yes_prices} | {no_prices} | {spread_text}")
                
                # Show count if more markets exist
                if len(sorted_markets) > 3:
                    print(f"... and {len(sorted_markets) - 3} more markets")
        else:
            print(f"\n📈 KALSHI MARKETS: Waiting for market data...")
        
        print("=" * 70)
        print("Press Ctrl+C to stop")
    
    async def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("🛑 Shutting down trading system...")
        
        # Stop KMS
        if self.kms:
            try:
                self.kms.stop_adaptive_market_tracking()
                if self.kms_task and not self.kms_task.done():
                    self.kms_task.cancel()
                    try:
                        await self.kms_task
                    except asyncio.CancelledError:
                        pass
                await self.kms.close()
                logger.info("✅ KMS shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down KMS: {e}")
        
        # UDM will stop when the main process exits (daemon thread)
        logger.info("✅ UDM will stop with main process")
        
        logger.info("✅ Trading system shutdown complete")
    
    async def run(self):
        """Main run loop"""
        try:
            logger.info("🎯 KBTCH Trading System Starting...")
            logger.info("=" * 60)
            
            # Set up event monitoring
            self.setup_event_monitoring()
            
            # Start all components
            self.start_udm()
            await asyncio.sleep(10)  # Give UDM time to start
            
            self.start_strategy_engine()
            await asyncio.sleep(1)  # Give strategy time to initialize
            
            await self.start_kms()
            await asyncio.sleep(2)  # Give KMS time to connect
            
            logger.info("🎉 All components started successfully!")
            logger.info("📊 System is now running. Press Ctrl+C to stop.")
            logger.info("=" * 60)
            
            self.running = True
            
            # Main monitoring loop with live display
            status_interval = 2  # Update display every 2 seconds
            last_status_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Print live status
                if current_time - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = current_time
                
                # Check if KMS is still running
                if self.kms_task and self.kms_task.done():
                    logger.error("❌ KMS task has stopped unexpectedly")
                    break
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
        except Exception as e:
            logger.error(f"💥 Unexpected error in main loop: {e}")
        finally:
            await self.shutdown()
            
            # Final status
            print("\n" + "=" * 60)
            print("FINAL STATISTICS")
            print("=" * 60)
            runtime = time.time() - self.stats['start_time']
            print(f"Total Runtime:           {runtime:.1f} seconds")
            print(f"Price Updates:           {self.stats['price_updates']}")
            print(f"Market Data Updates:     {self.stats['market_data_updates']}")
            print(f"Trading Signals:         {self.stats['signals_generated']}")
            
            if self.stats['price_updates'] > 0:
                rate = self.stats['price_updates'] / runtime
                print(f"Price Update Rate:       {rate:.1f} per second")

async def main():
    """Entry point"""
    manager = TradingSystemManager()
    await manager.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading system terminated by user")
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        sys.exit(1)
