"""
Simple strategy engine for Kalshi BTC binary options trading.
Focuses on BRTI vs market price divergence opportunities.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from event_bus import event_bus, EventTypes
from config.config_manager import config

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    market_ticker: str
    signal_type: str
    confidence: float
    strike_price: float
    current_brti: float
    market_yes_price: float
    market_no_price: float
    reason: str
    timestamp: float


class StrategyEngine:
    
    def __init__(self):
        self.strategy_settings = config.get_strategy_settings()
        
        self.min_divergence_pct = 0.02
        self.max_divergence_pct = 0.20
        self.min_confidence = 0.5
        
        self.current_brti: Optional[float] = None
        self.active_markets: Dict[str, Dict] = {}
        self.last_signal_time: Dict[str, float] = {}
        self.min_signal_interval = 1.0
        
        self.last_udm_update = None
        self.last_kms_update = None

        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, self._handle_market_data)
        logger.info("Subscribed to KMS")
        event_bus.subscribe(EventTypes.PRICE_UPDATE, self._handle_price_update)
        logger.info("Subscribed to UDM")
    
    def _handle_market_data(self, event) -> None:
        try:
            self.last_kms_update = time.time()
            data = event.data
            market_ticker = data.get("market_ticker")
            
            if not market_ticker:
                return
            
            self.active_markets[market_ticker] = {
                "ticker": market_ticker,
                "yes_bid": data.get("yes_bid"),
                "yes_ask": data.get("yes_ask"),
                "no_bid": data.get("no_bid"),
                "no_ask": data.get("no_ask"),
                "strike_price": data.get("strike_price"),
                "timestamp": time.time()
            }
            
            if self.current_brti:
                self._check_for_signals(market_ticker)
                
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    def _handle_price_update(self, event) -> None:
        """Handle BRTI price updates from UDM"""
        try:
            data = event.data
            if "brti_price" in data:
                self.last_udm_update = time.time()
                self.current_brti = data["brti_price"]
                
                # Check all active markets for signals
                for market_ticker in self.active_markets.keys():
                    self._check_for_signals(market_ticker)
                    
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
    
    def _check_for_signals(self, market_ticker: str) -> None:
        """Check if market conditions generate a trading signal"""
        try:
            market_data = self.active_markets.get(market_ticker)
            if not market_data or not self.current_brti:
                return
            
            # Rate limiting - don't generate signals too frequently for same market
            now = time.time()
            last_signal = self.last_signal_time.get(market_ticker, 0)
            if now - last_signal < self.min_signal_interval:
                return
            
            strike_price = market_data["strike_price"]
            yes_bid = market_data["yes_bid"]
            yes_ask = market_data["yes_ask"]
            no_bid = market_data["no_bid"]
            no_ask = market_data["no_ask"]
            
            # Skip if missing critical data
            if None in [strike_price, yes_bid, yes_ask, no_bid, no_ask]:
                return
            
            # Calculate theoretical probability based on BRTI
            # If BRTI > strike, YES should be worth more, NO should be worth less
            distance_from_strike = self.current_brti - strike_price
            distance_pct = abs(distance_from_strike) / strike_price
            
            # Calculate market implied probability
            yes_mid = (yes_bid + yes_ask) / 2
            market_prob = yes_mid / 100  # Convert cents to probability
            
            # Simple signal logic - both BUY and SELL signals
            signal = None
            
            if distance_from_strike > 0:  # BRTI above strike
                # YES should be worth close to 100¢, so buy YES if it's cheap
                if yes_ask < 90 and distance_pct > self.min_divergence_pct:
                    confidence = min(distance_pct * 5, 1.0)  # Scale confidence
                    if confidence >= self.min_confidence:
                        signal = Signal(
                            market_ticker=market_ticker,
                            signal_type="BUY_YES",
                            confidence=confidence,
                            strike_price=strike_price,
                            current_brti=self.current_brti,
                            market_yes_price=yes_ask,
                            market_no_price=no_bid,
                            reason=f"BRTI ${self.current_brti:,.0f} > strike ${strike_price:,.0f}, YES underpriced at {yes_ask}¢",
                            timestamp=now
                        )
                
                # If you own NO positions, consider selling since BRTI suggests YES should win
                elif no_bid > 10 and distance_pct > self.min_divergence_pct:
                    confidence = min(distance_pct * 5, 1.0)
                    if confidence >= self.min_confidence:
                        signal = Signal(
                            market_ticker=market_ticker,
                            signal_type="SELL_NO",
                            confidence=confidence,
                            strike_price=strike_price,
                            current_brti=self.current_brti,
                            market_yes_price=yes_ask,
                            market_no_price=no_bid,
                            reason=f"BRTI ${self.current_brti:,.0f} > strike ${strike_price:,.0f}, exit NO positions at {no_bid}¢",
                            timestamp=now
                        )
            
            else:  # BRTI below strike
                # NO should be worth close to 100¢, so buy NO if it's cheap
                if no_ask < 90 and distance_pct > self.min_divergence_pct:
                    confidence = min(distance_pct * 5, 1.0)
                    if confidence >= self.min_confidence:
                        signal = Signal(
                            market_ticker=market_ticker,
                            signal_type="BUY_NO",
                            confidence=confidence,
                            strike_price=strike_price,
                            current_brti=self.current_brti,
                            market_yes_price=yes_mid,
                            market_no_price=no_ask,
                            reason=f"BRTI ${self.current_brti:,.0f} < strike ${strike_price:,.0f}, NO underpriced at {no_ask}¢",
                            timestamp=now
                        )
                
                # If you own YES positions, consider selling since BRTI suggests NO should win
                elif yes_bid > 10 and distance_pct > self.min_divergence_pct:
                    confidence = min(distance_pct * 5, 1.0)
                    if confidence >= self.min_confidence:
                        signal = Signal(
                            market_ticker=market_ticker,
                            signal_type="SELL_YES",
                            confidence=confidence,
                            strike_price=strike_price,
                            current_brti=self.current_brti,
                            market_yes_price=yes_bid,
                            market_no_price=no_ask,
                            reason=f"BRTI ${self.current_brti:,.0f} < strike ${strike_price:,.0f}, exit YES positions at {yes_bid}¢",
                            timestamp=now
                        )
            
            # Publish signal if generated
            if signal and distance_pct <= self.max_divergence_pct:
                self._publish_signal(signal)
                self.last_signal_time[market_ticker] = now
                
        except Exception as e:
            logger.error(f"Error checking signals for {market_ticker}: {e}")
    
    def _publish_signal(self, signal: Signal) -> None:
        """Publish trading signal to event bus"""
        try:
            event_data = {
                "market_ticker": signal.market_ticker,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "strike_price": signal.strike_price,
                "current_brti": signal.current_brti,
                "market_yes_price": signal.market_yes_price,
                "market_no_price": signal.market_no_price,
                "reason": signal.reason,
                "timestamp": signal.timestamp
            }
            
            event_bus.publish(
                EventTypes.SIGNAL_GENERATED,
                event_data,
                source="strategy_engine"
            )
            
            logger.info(f"Signal: {signal.signal_type} {signal.market_ticker} "
                       f"(confidence: {signal.confidence:.2f}) - {signal.reason}")
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        udm_active = self.last_udm_update and (now - self.last_udm_update < 5)
        kms_active = self.last_kms_update and (now - self.last_kms_update < 5)

        return {
            "current_brti": self.current_brti,
            "active_markets": len(self.active_markets),
            "udm_active": "✅" if udm_active else "❌",
            "kms_active": "✅" if kms_active else "❌",
            "last_signal_times": self.last_signal_time.copy(),
            "parameters": {
                "min_divergence_pct": self.min_divergence_pct,
                "max_divergence_pct": self.max_divergence_pct,
                "min_confidence": self.min_confidence
            }
        }
