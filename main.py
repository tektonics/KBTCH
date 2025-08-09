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
from CEP import CEPEngine  # New CEP engine
from strategy import TradingStrategy  # New pure strategy engine
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
        self.cep_engine: Optional[CEPEngine] = None
        self.strategy: Optional[TradingStrategy] = None
        self.udm_thread: Optional[threading.Thread] = None
        self.kms_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            'price_updates': 0,
            'market_data_updates': 0,
            'enriched_events': 0,
            'signals_generated': 0,
            'start_time': time.time()
        }
        
        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self.running = False

    def setup_event_monitoring(self):

        def price_update_handler(event):
            self.stats['price_updates'] += 1
            brti_price = event.data.get('brti_price')
            if self.stats['price_updates'] % 50 == 0:  # Log every 50th update
                logger.info(f"📊 Price Update #{self.stats['price_updates']}: BRTI ${brti_price:,.2f}")
        
        def market_data_handler(event):
            self.stats['market_data_updates'] += 1
        
        def enriched_event_handler(event):
            """Monitor CEP enriched events"""
            self.stats['enriched_events'] += 1
            # Optional: Log interesting CEP events
            if event.data.get('momentum_pattern') or event.data.get('event_type') == 'CROSS_MARKET_ANALYSIS':
                logger.info(f"🔍 CEP Event: {event.data.get('event_type')} - "
                           f"Pattern: {event.data.get('momentum_pattern', 'None')}, "
                           f"Regime: {event.data.get('market_regime', 'Unknown')}")
        
        def signal_handler(event):
            self.stats['signals_generated'] += 1
            logger.info(f"🚨 TRADING SIGNAL #{self.stats['signals_generated']}:")
            logger.info(f"    Market: {event.data.get('market_ticker')}")
            logger.info(f"    Signal: {event.data.get('signal_type')}")
            logger.info(f"    Confidence: {event.data.get('confidence', 0):.2f}")
            logger.info(f"    Edge: {event.data.get('arbitrage_edge', 0):.1f}¢")
            logger.info(f"    Regime: {event.data.get('market_regime', 'Unknown')}")
            logger.info(f"    Reason: {event.data.get('reason')}")
        
        # Subscribe to events
        event_bus.subscribe(EventTypes.PRICE_UPDATE, price_update_handler)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, market_data_handler)
        event_bus.subscribe(EventTypes.ENRICHED_EVENT, enriched_event_handler)  # New CEP events
        event_bus.subscribe(EventTypes.SIGNAL_GENERATED, signal_handler)
    
    def start_udm(self):
        
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
        try:
            self.kms = KalshiClient()
            self.kms_task = asyncio.create_task(
                self.kms.start_adaptive_market_tracking()
            )
            logger.info("✅ KMS started successfully")
        except Exception as e:
            logger.error(f"Failed to start KMS: {e}")
            raise
    
    def start_cep_engine(self):
        try:
            self.cep_engine = CEPEngine()
            logger.info("✅ CEP Engine started successfully")
        except Exception as e:
            logger.error(f"Failed to start CEP Engine: {e}")
            raise
    
    def start_trading_strategy(self):
        try:
            self.strategy = TradingStrategy()
            logger.info("✅ Trading Strategy started successfully")
        except Exception as e:
            logger.error(f"Failed to start Trading Strategy: {e}")
            raise
    
    def print_status(self):
        """Print current system status"""
        # Clear screen for clean display
        os.system('cls' if os.name == 'nt' else 'clear')
        
        runtime = time.time() - self.stats['start_time']
        
        print(f"KBTCH TRADING SYSTEM - Runtime: {runtime:.0f}s")
        print("=" * 80)
        
        # System Statistics
        print(f"Price Updates:           {self.stats['price_updates']}")
        print(f"Market Data Updates:     {self.stats['market_data_updates']}")
        print(f"CEP Enriched Events:     {self.stats['enriched_events']}")  # New
        print(f"Trading Signals:         {self.stats['signals_generated']}")
        
        # CEP Engine Status
        if self.cep_engine:
            cep_status = self.cep_engine.get_status()
            print(f"\n🔍 CEP ENGINE STATUS")
            print(f"Events Processed:        {cep_status.get('events_processed', 0)}")
            print(f"Patterns Detected:       {cep_status.get('patterns_detected', 0)}")
            print(f"Market Regime:           {cep_status.get('market_regime', 'Unknown')}")
            print(f"Volatility Level:        {cep_status.get('volatility_level', 'Unknown')}")
            print(f"Current Volatility:      {cep_status.get('current_volatility', 0):.4f}")
            print(f"Active Markets:          {cep_status.get('active_markets', 0)}")
        
        # Strategy Status  
        if self.strategy:
            strategy_status = self.strategy.get_status()
            print(f"\n🎯 STRATEGY ENGINE STATUS")
            print(f"Active Opportunities:    {strategy_status.get('active_opportunities', 0)}")
            print(f"Opportunities Analyzed:  {strategy_status.get('opportunities_analyzed', 0)}")
            print(f"Signals Generated:       {strategy_status.get('signals_generated', 0)}")
            signals_by_type = strategy_status.get('signals_by_type', {})
            if signals_by_type:
                print(f"Signal Breakdown:        {', '.join([f'{k}:{v}' for k, v in signals_by_type.items()])}")
        
        # Data Flow Status
        print(f"\n📡 DATA FLOW STATUS")
        # Approximate data flow health based on recent activity
        udm_active = self.stats['price_updates'] > 0 and (time.time() - self.stats['start_time']) > 10
        kms_active = self.stats['market_data_updates'] > 0 and (time.time() - self.stats['start_time']) > 10
        cep_active = self.stats['enriched_events'] > 0 and (time.time() - self.stats['start_time']) > 10
        
        print(f"UDM → CEP:               {'✅' if udm_active else '❌'}")
        print(f"KMS → CEP:               {'✅' if kms_active else '❌'}")
        print(f"CEP → Strategy:          {'✅' if cep_active else '❌'}")
        
        # Kalshi Market Display
        if self.kms and self.kms.active_market_info:
            print(f"\n📈 KALSHI MARKETS")
            
            # Header with BTC price and volatility
            btc_price = self.kms.last_btc_price or 0
            volatility = self.kms.current_volatility or 0
            event_ticker = self.kms.event_ticker or "Unknown"
            
            print(f"Event: {event_ticker} | BTC: ${btc_price:,.2f} | KMS Vol: {volatility:.2f}")
            
            # Show market ladder
            sorted_markets = sorted(self.kms.active_market_info.values(), key=lambda m: m.strike)
            if sorted_markets:
                strike_labels = []
                for market_info in sorted_markets:
                    label = f"${market_info.strike:,.0f}🎯" if market_info.is_primary else f"${market_info.strike:,.0f}"
                    strike_labels.append(label)
                print(f"Strike Ladder: {' | '.join(strike_labels)}")
                
                # Market details
                for i, market_info in enumerate(sorted_markets):
                    data = market_info.market_data
                    if data and data.yes_bid is not None and data.yes_ask is not None:
                        primary_indicator = "🎯" if market_info.is_primary else " "
                        yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
                        no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
                        no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
                        spread = data.yes_ask - data.yes_bid
                        spread_text = f"Spread: {spread:.0f}¢"
                        
                        print(f"{primary_indicator}${market_info.strike:,.0f}: {yes_prices} | {no_prices} | {spread_text}")
        else:
            print(f"\n📈 KALSHI MARKETS: Waiting for market data...")
        
        print("=" * 80)
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
        
        # CEP and Strategy engines will stop when event bus stops receiving events
        logger.info("✅ CEP and Strategy engines will stop with event flow")
        
        # UDM will stop when the main process exits (daemon thread)
        logger.info("✅ UDM will stop with main process")
        
        logger.info("✅ Trading system shutdown complete")
    
    async def run(self):
        try:
            logger.info("=" * 70)
            
            # Set up event monitoring
            self.setup_event_monitoring()
            
            # Start all components in order
            self.start_udm()
            await asyncio.sleep(10)  # Give UDM time to start
            
            self.start_cep_engine()  # Start CEP first
            await asyncio.sleep(2)   # Give CEP time to subscribe
            
            self.start_trading_strategy()  # Then start strategy
            await asyncio.sleep(2)   # Give strategy time to subscribe
            
            await self.start_kms()
            await asyncio.sleep(2)  # Give KMS time to connect
            
            logger.info("🎉 All components started successfully!")
            logger.info("=" * 70)
            
            self.running = True
            
            # Main monitoring loop with live display
            status_interval = .1
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
            print("\n" + "=" * 70)
            print("FINAL STATISTICS")
            print("=" * 70)
            runtime = time.time() - self.stats['start_time']
            print(f"Total Runtime:           {runtime:.1f} seconds")
            print(f"Price Updates:           {self.stats['price_updates']}")
            print(f"Market Data Updates:     {self.stats['market_data_updates']}")
            print(f"CEP Enriched Events:     {self.stats['enriched_events']}")
            print(f"Trading Signals:         {self.stats['signals_generated']}")
            
            if self.stats['price_updates'] > 0:
                rate = self.stats['price_updates'] / runtime
                print(f"Price Update Rate:       {rate:.1f} per second")
            
            if self.stats['enriched_events'] > 0:
                enrichment_rate = self.stats['enriched_events'] / max(self.stats['price_updates'] + self.stats['market_data_updates'], 1)
                print(f"CEP Enrichment Rate:     {enrichment_rate:.2f}")

async def main():
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
