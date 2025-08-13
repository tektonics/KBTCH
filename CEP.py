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
    
    # Enhanced UDM data
    utilized_depth: Optional[float] = None
    dynamic_cap: Optional[float] = None
    valid_exchanges: Optional[int] = None
    volume_spikes: Optional[List[str]] = None
    rsi: Optional[float] = None
    momentum: Optional[str] = None
    avg_price: Optional[float] = None


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
        """Process incoming price updates from UDM with enhanced data"""
        try:
            self.events_processed += 1
            data = event.data
            
            if "brti_price" not in data:
                return
            
            brti_price = data["brti_price"]
            self.current_brti = brti_price
            
            # Extract enhanced UDM data
            utilized_depth = data.get("utilized_depth")
            dynamic_cap = data.get("dynamic_cap")
            valid_exchanges = data.get("valid_exchanges", 0)
            volume_spikes = data.get("volume_spikes", [])
            rsi = data.get("rsi")
            udm_momentum = data.get("momentum")
            avg_price = data.get("avg_price")
            
            # Store enhanced price event
            price_event = PriceEvent(
                timestamp=time.time(),
                brti_price=brti_price,
                source=event.source or "udm",
                utilized_depth=utilized_depth,
                dynamic_cap=dynamic_cap,
                valid_exchanges=valid_exchanges,
                volume_spikes=volume_spikes,
                rsi=rsi,
                momentum=udm_momentum,
                avg_price=avg_price
            )
            self.price_history.append(price_event)
            
            # Update volatility metrics
            self._update_volatility_metrics()
            
            # Update market regime (enhanced with UDM data)
            self._update_market_regime_enhanced(rsi, udm_momentum, volume_spikes)
            
            # Detect patterns (enhanced with UDM data)
            momentum_pattern = self._detect_momentum_pattern_enhanced(udm_momentum, rsi)
            price_trend = self._detect_price_trend()
            volume_pattern = self._analyze_volume_patterns(volume_spikes)
            
            # Calculate enhanced confidence boost
            confidence_boost = self._calculate_enhanced_confidence_boost(
                momentum_pattern, price_trend, volume_pattern, rsi, valid_exchanges
            )
            
            # Create enriched event with UDM enhancements
            enriched_event = EnrichedEvent(
                timestamp=time.time(),
                event_type="ENRICHED_PRICE_UPDATE",
                brti_price=brti_price,
                momentum_pattern=momentum_pattern,
                market_regime=self.market_regime,
                volatility_level=self._classify_volatility(),
                price_trend=price_trend,
                confidence_boost=confidence_boost,
                context={
                    "volatility": self.current_volatility,
                    "price_history_length": len(self.price_history),
                    "utilized_depth": utilized_depth,
                    "dynamic_cap": dynamic_cap,
                    "valid_exchanges": valid_exchanges,
                    "volume_spikes": volume_spikes,
                    "rsi": rsi,
                    "udm_momentum": udm_momentum,
                    "avg_price": avg_price,
                    "volume_pattern": volume_pattern,
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
        """Legacy market regime classification (kept for compatibility)"""
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
    
    def _update_market_regime_enhanced(self, rsi: Optional[float], udm_momentum: Optional[str], 
                                     volume_spikes: List[str]) -> None:
        """Enhanced market regime classification using UDM data"""
        # Start with volatility-based classification
        if self.current_volatility > self.volatility_threshold_high:
            base_regime = "VOLATILE"
        elif len(self.price_history) >= self.trend_lookback:
            prices = [event.brti_price for event in list(self.price_history)[-self.trend_lookback:]]
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            if price_change > self.trend_threshold:
                base_regime = "TRENDING_UP"
            elif price_change < -self.trend_threshold:
                base_regime = "TRENDING_DOWN"
            else:
                base_regime = "NORMAL"
        else:
            base_regime = "NORMAL"
        
        # Enhance with UDM indicators
        regime_modifiers = []
        
        # RSI enhancements
        if rsi is not None:
            if rsi > 70:
                regime_modifiers.append("OVERBOUGHT")
            elif rsi < 30:
                regime_modifiers.append("OVERSOLD")
        
        # UDM momentum enhancements
        if udm_momentum in ["↑↑", "↓↓"]:
            regime_modifiers.append("STRONG_MOMENTUM")
        
        # Volume spike enhancements
        if len(volume_spikes) >= 2:  # Multiple exchanges showing volume spikes
            regime_modifiers.append("HIGH_VOLUME")
        
        # Combine regime with modifiers
        if regime_modifiers:
            self.market_regime = f"{base_regime}_{'+'.join(regime_modifiers)}"
        else:
            self.market_regime = base_regime
    
    def _detect_momentum_pattern_enhanced(self, udm_momentum: Optional[str], 
                                        rsi: Optional[float]) -> Optional[str]:
        """Enhanced momentum detection combining CEP patterns with UDM indicators"""
        # Get base CEP momentum pattern
        cep_momentum = self._detect_momentum_pattern()
        
        # If no CEP pattern, check UDM momentum
        if not cep_momentum and udm_momentum:
            if udm_momentum == "↑↑":
                return "UDM_STRONG_UP"
            elif udm_momentum == "↓↓":
                return "UDM_STRONG_DOWN"
            elif udm_momentum == "↑":
                return "UDM_MODERATE_UP"
            elif udm_momentum == "↓":
                return "UDM_MODERATE_DOWN"
        
        # If we have CEP momentum, enhance it with UDM data
        if cep_momentum:
            # Check for alignment between CEP and UDM
            if cep_momentum == "UPWARD_MOMENTUM" and udm_momentum in ["↑", "↑↑"]:
                return "CONFIRMED_UPWARD_MOMENTUM"
            elif cep_momentum == "DOWNWARD_MOMENTUM" and udm_momentum in ["↓", "↓↓"]:
                return "CONFIRMED_DOWNWARD_MOMENTUM"
            elif cep_momentum == "UPWARD_MOMENTUM" and udm_momentum in ["↓", "↓↓"]:
                return "CONFLICTING_MOMENTUM_UP"  # CEP says up, UDM says down
            elif cep_momentum == "DOWNWARD_MOMENTUM" and udm_momentum in ["↑", "↑↑"]:
                return "CONFLICTING_MOMENTUM_DOWN"  # CEP says down, UDM says up
            else:
                return cep_momentum  # Return original CEP pattern
        
        return None
    
    def _analyze_volume_patterns(self, volume_spikes: List[str]) -> Optional[str]:
        """Analyze volume spike patterns from UDM"""
        if not volume_spikes:
            return None
        
        # Count number of exchanges with volume spikes
        spike_count = len(volume_spikes)
        
        # Analyze spike magnitudes (extract percentages)
        total_spike_pct = 0
        for spike in volume_spikes:
            try:
                # Extract percentage from format like "CB(+150%)"
                if "(" in spike and "%" in spike:
                    pct_str = spike.split("(+")[1].split("%")[0]
                    total_spike_pct += float(pct_str)
            except (IndexError, ValueError):
                continue
        
        avg_spike_pct = total_spike_pct / spike_count if spike_count > 0 else 0
        
        # Classify volume patterns
        if spike_count >= 3:  # 3+ exchanges
            if avg_spike_pct > 200:
                return "MASSIVE_VOLUME_SURGE"
            elif avg_spike_pct > 100:
                return "MAJOR_VOLUME_SPIKE"
            else:
                return "BROAD_VOLUME_INCREASE"
        elif spike_count == 2:
            if avg_spike_pct > 150:
                return "DUAL_EXCHANGE_SURGE"
            else:
                return "MODERATE_VOLUME_SPIKE"
        elif spike_count == 1:
            if avg_spike_pct > 200:
                return "SINGLE_MASSIVE_SPIKE"
            else:
                return "MINOR_VOLUME_SPIKE"
        
        return None
    
    def _calculate_enhanced_confidence_boost(self, momentum_pattern: Optional[str], 
                                           price_trend: Optional[str], volume_pattern: Optional[str],
                                           rsi: Optional[float], valid_exchanges: int) -> float:
        """Enhanced confidence calculation using all UDM data"""
        boost = 1.0
        
        # Base momentum boosts (enhanced patterns get bigger boosts)
        if momentum_pattern:
            momentum_boosts = {
                "CONFIRMED_UPWARD_MOMENTUM": 0.25,     # CEP + UDM agree
                "CONFIRMED_DOWNWARD_MOMENTUM": 0.25,
                "UPWARD_MOMENTUM": 0.15,               # CEP only
                "DOWNWARD_MOMENTUM": 0.15,
                "UDM_STRONG_UP": 0.12,                 # UDM strong signals
                "UDM_STRONG_DOWN": 0.12,
                "UDM_MODERATE_UP": 0.08,               # UDM moderate signals
                "UDM_MODERATE_DOWN": 0.08,
                "CONFLICTING_MOMENTUM_UP": -0.10,      # Conflicting signals
                "CONFLICTING_MOMENTUM_DOWN": -0.10
            }
            boost += momentum_boosts.get(momentum_pattern, 0)
        
        # Volume pattern boosts
        if volume_pattern:
            volume_boosts = {
                "MASSIVE_VOLUME_SURGE": 0.20,
                "MAJOR_VOLUME_SPIKE": 0.15,
                "BROAD_VOLUME_INCREASE": 0.12,
                "DUAL_EXCHANGE_SURGE": 0.10,
                "SINGLE_MASSIVE_SPIKE": 0.08,
                "MODERATE_VOLUME_SPIKE": 0.05,
                "MINOR_VOLUME_SPIKE": 0.02
            }
            boost += volume_boosts.get(volume_pattern, 0)
        
        # RSI-based adjustments
        if rsi is not None:
            if 30 <= rsi <= 70:  # RSI in normal range
                boost += 0.05
            elif rsi > 80 or rsi < 20:  # Extreme RSI
                boost -= 0.05  # Reduce confidence at extremes
        
        # Data quality adjustments
        if valid_exchanges is not None:
            if valid_exchanges >= 4:  # Strong data quality
                boost += 0.08
            elif valid_exchanges <= 2:  # Weak data quality
                boost -= 0.10
        
        # Trend alignment boosts
        if momentum_pattern and price_trend:
            if (momentum_pattern in ["CONFIRMED_UPWARD_MOMENTUM", "UPWARD_MOMENTUM", "UDM_STRONG_UP"] 
                and price_trend == "UPTREND") or \
               (momentum_pattern in ["CONFIRMED_DOWNWARD_MOMENTUM", "DOWNWARD_MOMENTUM", "UDM_STRONG_DOWN"] 
                and price_trend == "DOWNTREND"):
                boost += 0.10  # Extra boost for aligned patterns
        
        # Volatility adjustments (same as before)
        if self.current_volatility > self.volatility_threshold_high:
            boost *= 0.8  # Reduce confidence during high volatility
        
        return min(boost, 2.0)  # Cap at 2.0x boost
    
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
            
        except Exception as e:
            logger.error(f"Error publishing enriched event: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get CEP engine status with UDM enhancements"""
        # Get latest UDM data from most recent price event
        latest_udm_data = {}
        if self.price_history:
            latest_event = self.price_history[-1]
            latest_udm_data = {
                "utilized_depth": latest_event.utilized_depth,
                "dynamic_cap": latest_event.dynamic_cap,
                "valid_exchanges": latest_event.valid_exchanges,
                "volume_spikes": latest_event.volume_spikes or [],
                "rsi": latest_event.rsi,
                "udm_momentum": latest_event.momentum,
                "avg_price": latest_event.avg_price
            }
        
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
            "udm_data": latest_udm_data,  # New: UDM enhancements
            "parameters": {
                "momentum_lookback": self.momentum_lookback,
                "volatility_threshold_low": self.volatility_threshold_low,
                "volatility_threshold_high": self.volatility_threshold_high,
                "trend_lookback": self.trend_lookback
            }
        }
