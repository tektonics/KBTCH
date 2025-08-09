"""
Complex Event Processing Engine for KBTCH Trading System
Handles event stream analysis, pattern detection, and event enrichment.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass
from collections import deque
from event_bus import event_bus, EventTypes
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PriceEvent:
    timestamp: float
    brti_price: float
    source: str = "udm"


@dataclass
class MarketEvent:
    timestamp: float
    market_ticker: str
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    strike_price: Optional[float]
    source: str = "kms"


@dataclass
class EnrichedEvent:
    """Enriched event with CEP analysis"""
    timestamp: float
    event_type: str
    
    # Original data
    brti_price: Optional[float] = None
    market_ticker: Optional[str] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    strike_price: Optional[float] = None
    
    # CEP enrichments
    momentum_pattern: Optional[str] = None
    market_regime: Optional[str] = None
    volatility_level: Optional[str] = None
    price_trend: Optional[str] = None
    confidence_boost: float = 1.0
    
    # Context
    context: Dict[str, Any] = None


class CEPEngine:
    """Complex Event Processing Engine"""
    
    def __init__(self):
        # Event storage for pattern detection
        self.price_history: Deque[PriceEvent] = deque(maxlen=100)
        self.market_events: Dict[str, Deque[MarketEvent]] = {}
        
        # CEP state tracking
        self.current_brti: Optional[float] = None
        self.volatility_window: Deque[float] = deque(maxlen=20)
        self.current_volatility = 0.0
        self.market_regime = "NORMAL"  # NORMAL, VOLATILE, TRENDING_UP, TRENDING_DOWN
        
        # Pattern detection parameters
        self.momentum_lookback = 4
        self.volatility_threshold_low = 0.005   # 0.5%
        self.volatility_threshold_high = 0.015  # 1.5%
        self.trend_lookback = 10
        self.trend_threshold = 0.005  # 0.5%
        
        # Statistics
        self.events_processed = 0
        self.patterns_detected = 0
        self.enriched_events_published = 0
        
        # Subscribe to raw events
        event_bus.subscribe(EventTypes.PRICE_UPDATE, self._handle_price_update)
        event_bus.subscribe(EventTypes.MARKET_DATA_UPDATE, self._handle_market_data)
        
        logger.info("CEP Engine initialized and subscribed to events")
    
    def _handle_price_update(self, event) -> None:
        """Process incoming price updates from UDM"""
        try:
            self.events_processed += 1
            data = event.data
            
            if "brti_price" not in data:
                return
            
            brti_price = data["brti_price"]
            self.current_brti = brti_price
            
            # Store price event
            price_event = PriceEvent(
                timestamp=time.time(),
                brti_price=brti_price,
                source=event.source or "udm"
            )
            self.price_history.append(price_event)
            
            # Update volatility metrics
            self._update_volatility_metrics()
            
            # Update market regime
            self._update_market_regime()
            
            # Detect patterns
            momentum_pattern = self._detect_momentum_pattern()
            price_trend = self._detect_price_trend()
            
            # Create enriched event
            enriched_event = EnrichedEvent(
                timestamp=time.time(),
                event_type="ENRICHED_PRICE_UPDATE",
                brti_price=brti_price,
                momentum_pattern=momentum_pattern,
                market_regime=self.market_regime,
                volatility_level=self._classify_volatility(),
                price_trend=price_trend,
                confidence_boost=self._calculate_confidence_boost(momentum_pattern, price_trend),
                context={
                    "volatility": self.current_volatility,
                    "price_history_length": len(self.price_history),
                    "source": "cep_engine"
                }
            )
            
            # Publish enriched event
            self._publish_enriched_event(enriched_event)
            
            # Check if we should trigger market analysis
            self._trigger_market_analysis()
            
        except Exception as e:
            logger.error(f"Error processing price update in CEP: {e}")
    
    def _handle_market_data(self, event) -> None:
        """Process incoming market data from KMS"""
        try:
            self.events_processed += 1
            data = event.data
            market_ticker = data.get("market_ticker")
            
            if not market_ticker:
                return
            
            # Store market event
            market_event = MarketEvent(
                timestamp=time.time(),
                market_ticker=market_ticker,
                yes_bid=data.get("yes_bid"),
                yes_ask=data.get("yes_ask"),
                no_bid=data.get("no_bid"),
                no_ask=data.get("no_ask"),
                strike_price=data.get("strike_price"),
                source=event.source or "kms"
            )
            
            # Initialize deque for this market if needed
            if market_ticker not in self.market_events:
                self.market_events[market_ticker] = deque(maxlen=50)
            
            self.market_events[market_ticker].append(market_event)
            
            # Analyze market-specific patterns
            market_patterns = self._analyze_market_patterns(market_ticker)
            
            # Create enriched market event
            enriched_event = EnrichedEvent(
                timestamp=time.time(),
                event_type="ENRICHED_MARKET_UPDATE",
                market_ticker=market_ticker,
                yes_bid=data.get("yes_bid"),
                yes_ask=data.get("yes_ask"),
                no_bid=data.get("no_bid"),
                no_ask=data.get("no_ask"),
                strike_price=data.get("strike_price"),
                brti_price=self.current_brti,
                momentum_pattern=market_patterns.get("momentum"),
                market_regime=self.market_regime,
                volatility_level=self._classify_volatility(),
                confidence_boost=market_patterns.get("confidence_boost", 1.0),
                context={
                    "market_analysis": market_patterns,
                    "source": "cep_engine"
                }
            )
            
            # Publish enriched market event
            self._publish_enriched_event(enriched_event)
            
        except Exception as e:
            logger.error(f"Error processing market data in CEP: {e}")
    
    def _update_volatility_metrics(self) -> None:
        """Calculate rolling volatility from price changes"""
        if len(self.price_history) >= 2:
            recent_price = self.price_history[-1].brti_price
            previous_price = self.price_history[-2].brti_price
            
            if previous_price > 0:
                price_change_pct = abs(recent_price - previous_price) / previous_price
                self.volatility_window.append(price_change_pct)
                
                if len(self.volatility_window) >= 10:
                    self.current_volatility = statistics.mean(self.volatility_window)
    
    def _update_market_regime(self) -> None:
        """Classify current market regime based on volatility and trend"""
        if self.current_volatility > self.volatility_threshold_high:
            self.market_regime = "VOLATILE"
        elif len(self.price_history) >= self.trend_lookback:
            prices = [event.brti_price for event in list(self.price_history)[-self.trend_lookback:]]
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            if price_change > self.trend_threshold:
                self.market_regime = "TRENDING_UP"
            elif price_change < -self.trend_threshold:
                self.market_regime = "TRENDING_DOWN"
            else:
                self.market_regime = "NORMAL"
        else:
            self.market_regime = "NORMAL"
    
    def _detect_momentum_pattern(self) -> Optional[str]:
        """Detect momentum patterns in price history"""
        if len(self.price_history) < self.momentum_lookback:
            return None
        
        recent_prices = [event.brti_price for event in list(self.price_history)[-self.momentum_lookback:]]
        
        # Check for consistent upward movement
        if all(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices))):
            total_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            if total_change > 0.002:  # 0.2% threshold
                self.patterns_detected += 1
                return "UPWARD_MOMENTUM"
        
        # Check for consistent downward movement
        elif all(recent_prices[i] < recent_prices[i-1] for i in range(1, len(recent_prices))):
            total_change = abs((recent_prices[-1] - recent_prices[0]) / recent_prices[0])
            if total_change > 0.002:  # 0.2% threshold
                self.patterns_detected += 1
                return "DOWNWARD_MOMENTUM"
        
        return None
    
    def _detect_price_trend(self) -> Optional[str]:
        """Detect longer-term price trends"""
        if len(self.price_history) < self.trend_lookback:
            return None
        
        prices = [event.brti_price for event in list(self.price_history)[-self.trend_lookback:]]
        
        # Simple linear trend detection
        start_price = statistics.mean(prices[:3])  # Average of first 3
        end_price = statistics.mean(prices[-3:])   # Average of last 3
        
        price_change = (end_price - start_price) / start_price
        
        if price_change > self.trend_threshold:
            return "UPTREND"
        elif price_change < -self.trend_threshold:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _classify_volatility(self) -> str:
        """Classify current volatility level"""
        if self.current_volatility > self.volatility_threshold_high:
            return "HIGH"
        elif self.current_volatility > self.volatility_threshold_low:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence_boost(self, momentum_pattern: Optional[str], price_trend: Optional[str]) -> float:
        """Calculate confidence boost based on patterns"""
        boost = 1.0
        
        # Momentum pattern boosts
        if momentum_pattern == "UPWARD_MOMENTUM":
            boost += 0.1
        elif momentum_pattern == "DOWNWARD_MOMENTUM":
            boost += 0.1
        
        # Trend alignment boosts
        if momentum_pattern and price_trend:
            if (momentum_pattern == "UPWARD_MOMENTUM" and price_trend == "UPTREND") or \
               (momentum_pattern == "DOWNWARD_MOMENTUM" and price_trend == "DOWNTREND"):
                boost += 0.15  # Extra boost for aligned patterns
        
        # Volatility adjustments
        if self.current_volatility > self.volatility_threshold_high:
            boost *= 0.8  # Reduce confidence during high volatility
        
        return min(boost, 1.5)  # Cap at 1.5x boost
    
    def _analyze_market_patterns(self, market_ticker: str) -> Dict[str, Any]:
        """Analyze patterns specific to a market ticker"""
        patterns = {"confidence_boost": 1.0}
        
        if market_ticker not in self.market_events:
            return patterns
        
        market_history = list(self.market_events[market_ticker])
        if len(market_history) < 3:
            return patterns
        
        # Analyze spread trends
        spreads = []
        for event in market_history[-5:]:  # Last 5 events
            if event.yes_bid is not None and event.yes_ask is not None:
                spread = event.yes_ask - event.yes_bid
                spreads.append(spread)
        
        if len(spreads) >= 3:
            if spreads[-1] < spreads[0]:  # Spread tightening
                patterns["spread_trend"] = "TIGHTENING"
                patterns["confidence_boost"] += 0.05
            elif spreads[-1] > spreads[0]:  # Spread widening
                patterns["spread_trend"] = "WIDENING"
                patterns["confidence_boost"] -= 0.05
        
        # Analyze price momentum within this market
        yes_prices = []
        for event in market_history[-4:]:  # Last 4 events
            if event.yes_ask is not None:
                yes_prices.append(event.yes_ask)
        
        if len(yes_prices) >= 3:
            if all(yes_prices[i] > yes_prices[i-1] for i in range(1, len(yes_prices))):
                patterns["momentum"] = "YES_BUYING"
                patterns["confidence_boost"] += 0.03
            elif all(yes_prices[i] < yes_prices[i-1] for i in range(1, len(yes_prices))):
                patterns["momentum"] = "YES_SELLING"
                patterns["confidence_boost"] += 0.03
        
        return patterns
    
    def _trigger_market_analysis(self) -> None:
        """Trigger cross-market analysis when conditions are right"""
        # Only run comprehensive analysis periodically
        if self.events_processed % 10 == 0:  # Every 10th event
            cross_market_patterns = self._analyze_cross_market_patterns()
            
            if cross_market_patterns:
                # Publish cross-market insights
                enriched_event = EnrichedEvent(
                    timestamp=time.time(),
                    event_type="CROSS_MARKET_ANALYSIS",
                    brti_price=self.current_brti,
                    market_regime=self.market_regime,
                    context=cross_market_patterns
                )
                self._publish_enriched_event(enriched_event)
    
    def _analyze_cross_market_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across all markets"""
        if not self.market_events or not self.current_brti:
            return {}
        
        patterns = {}
        arbitrage_opportunities = []
        
        # Look for arbitrage opportunities across all markets
        for market_ticker, events in self.market_events.items():
            if not events:
                continue
            
            latest_event = events[-1]
            if latest_event.strike_price is None:
                continue
            
            # Simple arbitrage detection
            distance_from_strike = self.current_brti - latest_event.strike_price
            
            if distance_from_strike > 100:  # BRTI significantly above strike
                if latest_event.yes_ask and latest_event.yes_ask < 90:
                    arbitrage_opportunities.append({
                        "market": market_ticker,
                        "type": "BUY_YES",
                        "edge": 100 - latest_event.yes_ask,
                        "strike": latest_event.strike_price
                    })
            
            elif distance_from_strike < -100:  # BRTI significantly below strike
                if latest_event.no_ask and latest_event.no_ask < 90:
                    arbitrage_opportunities.append({
                        "market": market_ticker,
                        "type": "BUY_NO", 
                        "edge": 100 - latest_event.no_ask,
                        "strike": latest_event.strike_price
                    })
        
        if arbitrage_opportunities:
            patterns["arbitrage_opportunities"] = arbitrage_opportunities
            patterns["opportunity_count"] = len(arbitrage_opportunities)
        
        return patterns
    
    def _publish_enriched_event(self, enriched_event: EnrichedEvent) -> None:
        """Publish enriched event to event bus"""
        try:
            event_data = {
                "timestamp": enriched_event.timestamp,
                "event_type": enriched_event.event_type,
                "brti_price": enriched_event.brti_price,
                "market_ticker": enriched_event.market_ticker,
                "yes_bid": enriched_event.yes_bid,
                "yes_ask": enriched_event.yes_ask,
                "no_bid": enriched_event.no_bid,
                "no_ask": enriched_event.no_ask,
                "strike_price": enriched_event.strike_price,
                "momentum_pattern": enriched_event.momentum_pattern,
                "market_regime": enriched_event.market_regime,
                "volatility_level": enriched_event.volatility_level,
                "price_trend": enriched_event.price_trend,
                "confidence_boost": enriched_event.confidence_boost,
                "context": enriched_event.context or {}
            }
            
            event_bus.publish(
                EventTypes.ENRICHED_EVENT,  # New event type for enriched events
                event_data,
                source="cep_engine"
            )
            
            self.enriched_events_published += 1
            
            if self.enriched_events_published % 50 == 0:  # Log every 50th enriched event
                logger.info(f"CEP Engine: Published {self.enriched_events_published} enriched events, "
                           f"detected {self.patterns_detected} patterns")
                
        except Exception as e:
            logger.error(f"Error publishing enriched event: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get CEP engine status"""
        return {
            "events_processed": self.events_processed,
            "patterns_detected": self.patterns_detected,
            "enriched_events_published": self.enriched_events_published,
            "current_brti": self.current_brti,
            "market_regime": self.market_regime,
            "volatility_level": self._classify_volatility(),
            "current_volatility": round(self.current_volatility, 4),
            "price_history_length": len(self.price_history),
            "active_markets": len(self.market_events),
            "parameters": {
                "momentum_lookback": self.momentum_lookback,
                "volatility_threshold_low": self.volatility_threshold_low,
                "volatility_threshold_high": self.volatility_threshold_high,
                "trend_lookback": self.trend_lookback
            }
        }
