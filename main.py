import asyncio
import threading
import logging
import sys
import os
import time
import signal
from typing import Optional, List
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UDM import UnifiedCryptoManager
from KMS import KalshiClient
from CEP import CEPEngine
from strategy import TradingStrategy
from event_bus import event_bus, EventTypes

os.makedirs('logs', exist_ok=True)

class LogCapture(logging.Handler):
    def __init__(self, maxlen=10):
        super().__init__()
        self.logs = deque(maxlen=maxlen)
    
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append({
            'time': time.time(),
            'level': record.levelname,
            'message': log_entry,
            'name': record.name
        })

log_capture = LogCapture(maxlen=8)
log_capture.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        log_capture  # Capture for dashboard display
        # NO StreamHandler = no console interference
    ],
    force=True
)

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

        self.log_capture = log_capture

        self.stats = {
            'price_updates': 0,
            'market_data_updates': 0,
            'enriched_events': 0,
            'signals_generated': 0,
            'start_time': time.time(),
            'recent_patterns': [],
            'recent_regimes': []
        }
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
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
            # Track recent patterns and regimes
            data = event.data
            momentum_pattern = data.get('momentum_pattern')
            market_regime = data.get('market_regime', 'Unknown')
            
            # Store recent patterns (keep last 5)
            if momentum_pattern and momentum_pattern not in ['None', None]:
                self.stats['recent_patterns'].append({
                    'pattern': momentum_pattern,
                    'time': time.time()
                })
                self.stats['recent_patterns'] = self.stats['recent_patterns'][-5:]
            
            # Store recent regimes (keep last 3 unique)
            if market_regime and market_regime != 'Unknown':
                if not self.stats['recent_regimes'] or self.stats['recent_regimes'][-1]['regime'] != market_regime:
                    self.stats['recent_regimes'].append({
                        'regime': market_regime,
                        'time': time.time()
                    })
                    self.stats['recent_regimes'] = self.stats['recent_regimes'][-3:]
            
            # Log interesting CEP events with enhanced patterns
            event_type = data.get('event_type')
            if momentum_pattern or event_type == 'CROSS_MARKET_ANALYSIS':
                # Show enhanced patterns in logs
                context = data.get('context', {})
                volume_pattern = context.get('volume_pattern')
                rsi = context.get('rsi')
                volume_spikes = context.get('volume_spikes', [])
                
                log_parts = [f"🔍 CEP Event: {event_type}"]
                if momentum_pattern:
                    log_parts.append(f"Momentum: {momentum_pattern}")
                if volume_pattern:
                    log_parts.append(f"Volume: {volume_pattern}")
                if rsi:
                    log_parts.append(f"RSI: {rsi:.0f}")
                if volume_spikes:
                    log_parts.append(f"Spikes: {', '.join(volume_spikes[:2])}")  # Show first 2
                
                logger.info(" | ".join(log_parts))
        
        def signal_handler(event):
            self.stats['signals_generated'] += 1
            data = event.data
            
            # Enhanced signal display with CEP data
            logger.info(f"🚨 ENHANCED TRADING SIGNAL #{self.stats['signals_generated']}:")
            logger.info(f"    Market: {data.get('market_ticker')}")
            logger.info(f"    Signal: {data.get('signal_type')}")
            logger.info(f"    Confidence: {data.get('confidence', 0):.2f}")
            
            # Show CEP enhancements
            edge = data.get('arbitrage_edge', 0)
            if edge:
                logger.info(f"    Arbitrage Edge: {edge:.1f}¢")
            
            momentum_pattern = data.get('momentum_pattern')
            if momentum_pattern:
                logger.info(f"    Momentum Pattern: {momentum_pattern}")
            
            market_regime = data.get('market_regime', 'Unknown')
            volatility_level = data.get('volatility_level', 'Unknown')
            cep_boost = data.get('cep_confidence_boost', 1.0)
            
            logger.info(f"    Market Regime: {market_regime}")
            logger.info(f"    Volatility: {volatility_level}")
            logger.info(f"    CEP Boost: {cep_boost:.2f}x")
            logger.info(f"    Reason: {data.get('reason')}")
        
        event_bus.subscribe(EventTypes.PRICE_UPDATE, price_update_handler)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, market_data_handler)
        event_bus.subscribe(EventTypes.ENRICHED_EVENT, enriched_event_handler)
        event_bus.subscribe(EventTypes.SIGNAL_GENERATED, signal_handler)

    def start_udm(self):
        self.udm = UnifiedCryptoManager()
    
        self.udm.update_single_line_display = lambda x: None
        self.udm.print_new_line = lambda x: None
        self.udm.clear_display = lambda: None
    
        def udm_runner():
            try:
                asyncio.run(self.udm.run_unified_system())
            except Exception as e:
                logger.error(f"UDM error: {e}")

        self.udm_thread = threading.Thread(target=udm_runner, daemon=True)
        self.udm_thread.start()

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
        os.system('cls' if os.name == 'nt' else 'clear')
        
        runtime = time.time() - self.stats['start_time']
        
        print(f"KBTCH TRADING SYSTEM - Runtime: {runtime:.0f}s")
        print("=" * 80)
        
        # System Statistics
        print(f"Price Updates:           {self.stats['price_updates']}")
        print(f"Market Data Updates:     {self.stats['market_data_updates']}")
        print(f"CEP Enriched Events:     {self.stats['enriched_events']}")
        print(f"Trading Signals:         {self.stats['signals_generated']}")
        
        # CEP Engine Status
        if self.cep_engine:
            cep_status = self.cep_engine.get_status()
            udm_data = cep_status.get('udm_data', {})
            
            print(f"\n🔍 CEP ENGINE STATUS")
            print(f"Events Processed:        {cep_status.get('events_processed', 0)}")
            print(f"Patterns Detected:       {cep_status.get('patterns_detected', 0)}")
            
            # Enhanced market regime display
            market_regime = cep_status.get('market_regime', 'Unknown')
            if len(market_regime) > 25:  # Truncate very long regimes
                display_regime = market_regime[:22] + "..."
            else:
                display_regime = market_regime
            print(f"Market Regime:           {display_regime}")
            
            print(f"Volatility Level:        {cep_status.get('volatility_level', 'Unknown')}")
            print(f"Current Volatility:      {cep_status.get('current_volatility', 0):.4f}")
            print(f"Active Markets:          {cep_status.get('active_markets', 0)}")
            
            print(f"\n📊 LIVE UDM DATA")
            depth = udm_data.get('utilized_depth')
            cap = udm_data.get('dynamic_cap')
            
            print(f"BRTI Depth:              {depth:.1f}" if depth is not None and isinstance(depth, (int, float)) else "BRTI Depth:              N/A")
            
            if cap is not None and cap != float('inf') and isinstance(cap, (int, float)):
                print(f"Dynamic Cap:             {cap:.1f}")
            else:
                print(f"Dynamic Cap:             ∞")
            
            valid_exchanges = udm_data.get('valid_exchanges')
            rsi = udm_data.get('rsi')
            udm_momentum = udm_data.get('udm_momentum')
            
            print(f"Valid Exchanges:         {valid_exchanges}" if valid_exchanges is not None else "Valid Exchanges:         N/A")
            print(f"RSI:                     {rsi:.0f}" if rsi is not None and isinstance(rsi, (int, float)) else "RSI:                     N/A")
            print(f"UDM Momentum:            {udm_momentum}" if udm_momentum else "UDM Momentum:            →")
            
            volume_spikes = udm_data.get('volume_spikes', [])
            if volume_spikes and isinstance(volume_spikes, list):
                display_spikes = volume_spikes[:3]
                if len(volume_spikes) > 3:
                    display_spikes.append(f"+{len(volume_spikes)-3} more")
                print(f"Volume Spikes:           {', '.join(display_spikes)}")
            else:
                print(f"Volume Spikes:           None")
            

            if self.stats['recent_patterns']:
                print(f"\n🔍 RECENT CEP PATTERNS")
                for i, pattern_info in enumerate(self.stats['recent_patterns'][-3:]):  # Show last 3
                    pattern = pattern_info['pattern']
                    age = time.time() - pattern_info['time']
                    print(f"Pattern {i+1}:             {pattern} ({age:.0f}s ago)")
            
            # Recent regime changes
            
            if len(self.stats['recent_regimes']) > 1:
                print(f"\n📊 RECENT REGIME CHANGES")
                for i, regime_info in enumerate(self.stats['recent_regimes'][-2:]):  # Show last 2
                    regime = regime_info['regime']
                    age = time.time() - regime_info['time']
                    # Truncate long regime names
                    display_regime = regime[:30] + "..." if len(regime) > 30 else regime
                    print(f"Regime {i+1}:             {display_regime} ({age:.0f}s ago)")
            
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
            
            # Show recent signal timing
            last_signals = strategy_status.get('last_signal_times', {})
            if last_signals:
                recent_signals = [(market, last_time) for market, last_time in last_signals.items() 
                                if time.time() - last_time < 300]  # Last 5 minutes
                if recent_signals:
                    print(f"Recent Signals:          {len(recent_signals)} in last 5min")

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
                
                # Market details (show all markets)
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

        print(f"\n📋 RECENT SYSTEM LOGS")
        print("-" * 50)
        
        if self.log_capture.logs:
            for log_entry in list(self.log_capture.logs)[-5:]:
                age = time.time() - log_entry['time']
                level_icon = {
                    'INFO': 'ℹ️',
                    'WARNING': '⚠️', 
                    'ERROR': '❌',
                    'CRITICAL': '🚨'
                }.get(log_entry['level'], '•')
                
                message = log_entry['message']
                if len(message) > 70:
                    message = message[:67] + "..."
                
                print(f"{level_icon} {message} ({age:.0f}s ago)")
        else:
            print("No recent logs")

        print("=" * 80)
        print("Press Ctrl+C to stop")
    
    async def shutdown(self):
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
            except Exception as e:
                logger.error(f"Error shutting down KMS: {e}")
    async def run(self):
        try:
            self.setup_event_monitoring()
            
            self.start_udm()
            await asyncio.sleep(10)
            
            self.start_cep_engine()
            await asyncio.sleep(2)
            
            self.start_trading_strategy()
            await asyncio.sleep(2)
            
            await self.start_kms()
            await asyncio.sleep(2)
            
            self.running = True
            
            status_interval = .1
            last_status_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                if current_time - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = current_time
                
                
                if self.kms_task and self.kms_task.done():
                    logger.error("❌ KMS task has stopped unexpectedly")
                    break
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"💥 Unexpected error in main loop: {e}")
        finally:
            await self.shutdown()
            
            print("\n" + "=" * 70)
            print("FINAL STATS")
            print("=" * 70)
            runtime = time.time() - self.stats['start_time']
            print(f"Total Runtime:           {runtime:.1f} seconds")
            print(f"Price Updates:           {self.stats['price_updates']}")
            print(f"Market Data Updates:     {self.stats['market_data_updates']}")
            print(f"CEP Enriched Events:     {self.stats['enriched_events']}")
            print(f"Trading Signals:         {self.stats['signals_generated']}")
            print(f"Patterns Detected:       {len(self.stats['recent_patterns'])}")
            print(f"Regime Changes:          {len(self.stats['recent_regimes'])}")
            
            if self.stats['price_updates'] > 0:
                rate = self.stats['price_updates'] / runtime
                print(f"Price Update Rate:       {rate:.1f} per second")
            
            if self.stats['enriched_events'] > 0:
                enrichment_rate = self.stats['enriched_events'] / max(self.stats['price_updates'] + self.stats['market_data_updates'], 1)
                print(f"CEP Enrichment Rate:     {enrichment_rate:.2f}")
                
            if self.stats['recent_patterns']:
                final_patterns = [p['pattern'] for p in self.stats['recent_patterns']]
                print(f"Final Patterns:          {', '.join(final_patterns[-3:])}")  # Last 3

async def main():
    manager = TradingSystemManager()
    await manager.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        sys.exit(1)
