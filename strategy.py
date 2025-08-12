"""
Pure Trading Strategy Engine for KBTCH Trading System
Receives enriched events from CEP and makes trading decisions.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict
from event_bus import event_bus, EventTypes
from config.config_manager import config

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    market_ticker: str
    signal_type: str  # BUY_YES, BUY_NO, SELL_YES, SELL_NO
    confidence: float
    strike_price: float
    current_brti: float
    market_yes_price: float
    market_no_price: float
    reason: str
    timestamp: float
    
    # Enhanced fields from CEP
    momentum_pattern: Optional[str] = None
    market_regime: Optional[str] = None
    volatility_level: Optional[str] = None
    cep_confidence_boost: float = 1.0
    arbitrage_edge: Optional[float] = None


class TradingStrategy:
    """Pure trading strategy - receives enriched events and generates trading signals"""
    
    def __init__(self):
        self.strategy_settings = config.get_strategy_settings()
        
        # Strategy parameters
        self.min_divergence_pct = 0.02      # 2% minimum divergence
        self.max_divergence_pct = 0.20      # 20% maximum divergence  
        self.base_confidence_threshold = 0.5
        self.min_arbitrage_edge = 5.0       # 5 cents minimum edge
        
        # Strategy state (NOT event processing state)
        self.current_brti: Optional[float] = None
        self.active_opportunities: Dict[str, Dict] = {}
        self.last_signal_time: Dict[str, float] = {}
        self.signal_frequency_limits: Dict[str, float] = {}
        
        # Signal generation controls
        self.base_signal_interval = 1.0     # Base rate limiting
        self.regime_multipliers = {
            "VOLATILE": 2.0,      # Wait longer during volatility
            "TRENDING_UP": 0.7,   # More frequent during trends
            "TRENDING_DOWN": 0.7,
            "NORMAL": 1.0
        }
        
        # Statistics
        self.signals_generated = 0
        self.signals_by_type = defaultdict(int)
        self.opportunities_analyzed = 0
        
        # Subscribe to enriched events from CEP
        event_bus.subscribe(EventTypes.ENRICHED_EVENT, self._handle_enriched_event)
        
        logger.info("Trading Strategy initialized and subscribed to enriched events")
    
    def _handle_enriched_event(self, event) -> None:
        """Process enriched events from CEP engine"""
        try:
            data = event.data
            event_type = data.get("event_type")
            
            if event_type == "ENRICHED_PRICE_UPDATE":
                self._handle_enriched_price_update(data)
            elif event_type == "ENRICHED_MARKET_UPDATE":
                self._handle_enriched_market_update(data)
            elif event_type == "CROSS_MARKET_ANALYSIS":
                self._handle_cross_market_analysis(data)
                
        except Exception as e:
            logger.error(f"Error handling enriched event in strategy: {e}")
    
    def _handle_enriched_price_update(self, data: Dict[str, Any]) -> None:
        """Handle enriched BRTI price updates"""
        brti_price = data.get("brti_price")
        if brti_price is None:
            return
        
        self.current_brti = brti_price
        
        # Extract CEP insights
        momentum_pattern = data.get("momentum_pattern")
        market_regime = data.get("market_regime", "NORMAL")
        volatility_level = data.get("volatility_level", "MEDIUM")
        confidence_boost = data.get("confidence_boost", 1.0)
        
        # Update regime-based signal frequency
        self._update_signal_frequency_limits(market_regime)
        
        # Check all active opportunities for signals
        for market_ticker in self.active_opportunities.keys():
            self._evaluate_opportunity(
                market_ticker, 
                momentum_pattern, 
                market_regime, 
                volatility_level, 
                confidence_boost
            )
    
    def _handle_enriched_market_update(self, data: Dict[str, Any]) -> None:
        """Handle enriched market data updates"""
        market_ticker = data.get("market_ticker")
        if not market_ticker:
            return
        
        # Store/update market opportunity
        self.active_opportunities[market_ticker] = {
            "market_ticker": market_ticker,
            "yes_bid": data.get("yes_bid"),
            "yes_ask": data.get("yes_ask"),
            "no_bid": data.get("no_bid"),
            "no_ask": data.get("no_ask"),
            "strike_price": data.get("strike_price"),
            "timestamp": time.time(),
            
            # CEP enhancements
            "momentum_pattern": data.get("momentum_pattern"),
            "market_regime": data.get("market_regime"),
            "volatility_level": data.get("volatility_level"),
            "confidence_boost": data.get("confidence_boost", 1.0),
            "context": data.get("context", {})
        }
        
        # Evaluate this specific opportunity
        if self.current_brti:
            self._evaluate_opportunity(
                market_ticker,
                data.get("momentum_pattern"),
                data.get("market_regime", "NORMAL"),
                data.get("volatility_level", "MEDIUM"),
                data.get("confidence_boost", 1.0)
            )
    
    def _handle_cross_market_analysis(self, data: Dict[str, Any]) -> None:
        """Handle cross-market analysis from CEP"""
        context = data.get("context", {})
        arbitrage_opportunities = context.get("arbitrage_opportunities", [])
        
        # Process high-confidence arbitrage opportunities
        for opportunity in arbitrage_opportunities:
            if opportunity.get("edge", 0) > self.min_arbitrage_edge:
                self._process_arbitrage_opportunity(opportunity, data)
    
    def _evaluate_opportunity(self, market_ticker: str, momentum_pattern: Optional[str], 
                            market_regime: str, volatility_level: str, confidence_boost: float) -> None:
        """Evaluate a trading opportunity with CEP insights"""
        try:
            self.opportunities_analyzed += 1
            
            opportunity = self.active_opportunities.get(market_ticker)
            if not opportunity or not self.current_brti:
                return
            
            # Rate limiting based on market regime
            if not self._check_signal_rate_limit(market_ticker, market_regime):
                return
            
            # Extract market data
            strike_price = opportunity["strike_price"]
            yes_bid = opportunity["yes_bid"]
            yes_ask = opportunity["yes_ask"]
            no_bid = opportunity["no_bid"]
            no_ask = opportunity["no_ask"]
            
            if None in [strike_price, yes_bid, yes_ask, no_bid, no_ask]:
                return
            
            # Core arbitrage calculation
            distance_from_strike = self.current_brti - strike_price
            distance_pct = abs(distance_from_strike) / strike_price
            
            # Check if opportunity meets minimum criteria
            if distance_pct < self.min_divergence_pct or distance_pct > self.max_divergence_pct:
                return
            
            # Calculate base confidence
            base_confidence = min(distance_pct * 5, 1.0)
            
            # Apply CEP enhancements
            enhanced_confidence = self._enhance_confidence_with_cep(
                base_confidence, momentum_pattern, market_regime, 
                volatility_level, confidence_boost, distance_from_strike
            )
            
            # Calculate dynamic confidence threshold
            confidence_threshold = self._calculate_dynamic_threshold(
                market_regime, volatility_level
            )
            
            if enhanced_confidence < confidence_threshold:
                return
            
            # Generate appropriate signal
            signal = self._generate_signal(
                market_ticker, distance_from_strike, enhanced_confidence,
                strike_price, yes_bid, yes_ask, no_bid, no_ask,
                momentum_pattern, market_regime, volatility_level, confidence_boost
            )
            
            if signal:
                self._publish_trading_signal(signal)
                self.last_signal_time[market_ticker] = time.time()
                
        except Exception as e:
            logger.error(f"Error evaluating opportunity for {market_ticker}: {e}")
    
    def _enhance_confidence_with_cep(self, base_confidence: float, momentum_pattern: Optional[str],
                                   market_regime: str, volatility_level: str, 
                                   confidence_boost: float, distance_from_strike: float) -> float:
        """Enhance confidence using CEP insights"""
        enhanced_confidence = base_confidence * confidence_boost
        
        # Momentum alignment bonus
        if momentum_pattern and distance_from_strike != 0:
            if (distance_from_strike > 0 and momentum_pattern == "UPWARD_MOMENTUM") or \
               (distance_from_strike < 0 and momentum_pattern == "DOWNWARD_MOMENTUM"):
                enhanced_confidence *= 1.15  # 15% bonus for aligned momentum
            elif (distance_from_strike > 0 and momentum_pattern == "DOWNWARD_MOMENTUM") or \
                 (distance_from_strike < 0 and momentum_pattern == "UPWARD_MOMENTUM"):
                enhanced_confidence *= 0.85  # 15% penalty for opposing momentum
        
        # Volatility adjustments
        volatility_multipliers = {
            "LOW": 1.1,    # Boost confidence in low volatility
            "MEDIUM": 1.0,
            "HIGH": 0.8    # Reduce confidence in high volatility
        }
        enhanced_confidence *= volatility_multipliers.get(volatility_level, 1.0)
        
        # Market regime adjustments
        regime_multipliers = {
            "NORMAL": 1.0,
            "TRENDING_UP": 1.05,    # Slight boost during trends
            "TRENDING_DOWN": 1.05,
            "VOLATILE": 0.9         # Reduce during volatility
        }
        enhanced_confidence *= regime_multipliers.get(market_regime, 1.0)
        
        return min(enhanced_confidence, 1.0)  # Cap at 1.0
    
    def _calculate_dynamic_threshold(self, market_regime: str, volatility_level: str) -> float:
        """Calculate dynamic confidence threshold based on market conditions"""
        base_threshold = self.base_confidence_threshold
        
        # Adjust for volatility
        if volatility_level == "HIGH":
            base_threshold *= 1.3  # Higher threshold during volatility
        elif volatility_level == "LOW":
            base_threshold *= 0.9  # Lower threshold during calm periods
        
        # Adjust for regime
        if market_regime == "VOLATILE":
            base_threshold *= 1.2  # Higher threshold during volatile regimes
        elif market_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            base_threshold *= 0.95  # Slightly lower during trends
        
        return min(base_threshold, 0.8)  # Cap at 0.8

    def _calculate_position_size(self, signal_data: Dict[str, Any]) -> int:
        confidence = signal_data.get('confidence', 0)
        arbitrage_edge = signal_data.get('arbitrage_edge', 0)
        base_size = 1
        confidence_multiplier = confidence * 2
        edge_multiplier = min(arbitrage_edge / 10, 2.0)
        final_size = int(base_size * confidence_multiplier * edge_multiplier)
        return max(1, min(final_size, 5))  # Between 1-5 contracts

    def _generate_signal(self, market_ticker: str, distance_from_strike: float, 
                        confidence: float, strike_price: float,
                        yes_bid: float, yes_ask: float, no_bid: float, no_ask: float,
                        momentum_pattern: Optional[str], market_regime: str,
                        volatility_level: str, cep_confidence_boost: float) -> Optional[TradingSignal]:

        signal = None
        arbitrage_edge = None
        
        if distance_from_strike > 0:
            if yes_ask < 90:
                arbitrage_edge = 100 - yes_ask
                if arbitrage_edge >= self.min_arbitrage_edge:
                    signal = TradingSignal(
                        market_ticker=market_ticker,
                        signal_type="BUY_YES",
                        confidence=confidence,
                        strike_price=strike_price,
                        current_brti=self.current_brti,
                        market_yes_price=yes_ask,
                        market_no_price=no_bid,
                        reason=f"BRTI ${self.current_brti:,.0f} > strike ${strike_price:,.0f}, YES underpriced at {yes_ask}¢ (edge: {arbitrage_edge:.1f}¢) [{market_regime}]",
                        timestamp=time.time(),
                        momentum_pattern=momentum_pattern,
                        market_regime=market_regime,
                        volatility_level=volatility_level,
                        cep_confidence_boost=cep_confidence_boost,
                        arbitrage_edge=arbitrage_edge
                    )
            
            # Alternative: Sell NO if you have positions and NO is overpriced
            elif no_bid > 10:
                arbitrage_edge = no_bid - 0  # Edge from selling overpriced NO
                signal = TradingSignal(
                    market_ticker=market_ticker,
                    signal_type="SELL_NO",
                    confidence=confidence * 0.8,  # Lower confidence for exit signals
                    strike_price=strike_price,
                    current_brti=self.current_brti,
                    market_yes_price=yes_ask,
                    market_no_price=no_bid,
                    reason=f"BRTI ${self.current_brti:,.0f} > strike ${strike_price:,.0f}, exit NO positions at {no_bid}¢ [{market_regime}]",
                    timestamp=time.time(),
                    momentum_pattern=momentum_pattern,
                    market_regime=market_regime,
                    volatility_level=volatility_level,
                    cep_confidence_boost=cep_confidence_boost,
                    arbitrage_edge=arbitrage_edge
                )
        
        else:  # BRTI below strike
            # NO should be worth close to 100¢
            if no_ask < 90:
                arbitrage_edge = 100 - no_ask
                if arbitrage_edge >= self.min_arbitrage_edge:
                    signal = TradingSignal(
                        market_ticker=market_ticker,
                        signal_type="BUY_NO",
                        confidence=confidence,
                        strike_price=strike_price,
                        current_brti=self.current_brti,
                        market_yes_price=yes_ask,
                        market_no_price=no_ask,
                        reason=f"BRTI ${self.current_brti:,.0f} < strike ${strike_price:,.0f}, NO underpriced at {no_ask}¢ (edge: {arbitrage_edge:.1f}¢) [{market_regime}]",
                        timestamp=time.time(),
                        momentum_pattern=momentum_pattern,
                        market_regime=market_regime,
                        volatility_level=volatility_level,
                        cep_confidence_boost=cep_confidence_boost,
                        arbitrage_edge=arbitrage_edge
                    )
            
            # Alternative: Sell YES if you have positions and YES is overpriced
            elif yes_bid > 10:
                arbitrage_edge = yes_bid - 0  # Edge from selling overpriced YES
                signal = TradingSignal(
                    market_ticker=market_ticker,
                    signal_type="SELL_YES",
                    confidence=confidence * 0.8,  # Lower confidence for exit signals
                    strike_price=strike_price,
                    current_brti=self.current_brti,
                    market_yes_price=yes_bid,
                    market_no_price=no_ask,
                    reason=f"BRTI ${self.current_brti:,.0f} < strike ${strike_price:,.0f}, exit YES positions at {yes_bid}¢ [{market_regime}]",
                    timestamp=time.time(),
                    momentum_pattern=momentum_pattern,
                    market_regime=market_regime,
                    volatility_level=volatility_level,
                    cep_confidence_boost=cep_confidence_boost,
                    arbitrage_edge=arbitrage_edge
                )
        
        return signal
    
    def _process_arbitrage_opportunity(self, opportunity: Dict[str, Any], context_data: Dict[str, Any]) -> None:
        market_ticker = opportunity.get("market")
        signal_type = opportunity.get("type")
        edge = opportunity.get("edge")
        strike = opportunity.get("strike")
        
        if not all([market_ticker, signal_type, edge, strike]):
            return

        market_opportunity = self.active_opportunities.get(market_ticker)
        if market_opportunity:
            yes_price = market_opportunity.get("yes_ask", 0)
            no_price = market_opportunity.get("no_ask", 0)
        else:
            yes_price = 0
            no_price = 0
        
        # Create high-confidence arbitrage signal
        arbitrage_signal = TradingSignal(
            market_ticker=market_ticker,
            signal_type=signal_type,
            confidence=min(edge / 10.0, 1.0),  # Scale edge to confidence (10 cent = 1.0)
            strike_price=strike,
            current_brti=self.current_brti or 0,
            market_yes_price=yes_price,
            market_no_price=no_price,
            reason=f"Cross-market arbitrage: {signal_type} with {edge:.1f}¢ edge",
            timestamp=time.time(),
            market_regime=context_data.get("market_regime", "NORMAL"),
            arbitrage_edge=edge
        )
        
        logger.info(f"High-confidence arbitrage detected: {signal_type} {market_ticker} with {edge:.1f}¢ edge")
        self._publish_trading_signal(arbitrage_signal)
    
    def _check_signal_rate_limit(self, market_ticker: str, market_regime: str) -> bool:
        """Check if signal rate limiting allows new signal"""
        now = time.time()
        last_signal = self.last_signal_time.get(market_ticker, 0)
        
        # Get regime-based interval
        interval_multiplier = self.regime_multipliers.get(market_regime, 1.0)
        required_interval = self.base_signal_interval * interval_multiplier
        
        return (now - last_signal) >= required_interval
    
    def _update_signal_frequency_limits(self, market_regime: str) -> None:
        """Update signal frequency limits based on market regime"""
        base_limit = self.base_signal_interval
        
        for market_ticker in self.active_opportunities.keys():
            self.signal_frequency_limits[market_ticker] = base_limit * self.regime_multipliers.get(market_regime, 1.0)
    
    def _publish_trading_signal(self, signal: TradingSignal) -> None:
        try:
            signal_data_for_sizing = {
                "confidence": signal.confidence,
                "arbitrage_edge": signal.arbitrage_edge
            }
            quantity = self._calculate_position_size(signal_data_for_sizing)
        
            event_data = {
                "market_ticker": signal.market_ticker,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "quantity": quantity,
                "strike_price": signal.strike_price,
                "current_brti": signal.current_brti,
                "market_yes_price": signal.market_yes_price,
                "market_no_price": signal.market_no_price,
                "reason": signal.reason,
                "timestamp": signal.timestamp,
                "momentum_pattern": signal.momentum_pattern,
                "market_regime": signal.market_regime,
                "volatility_level": signal.volatility_level,
                "cep_confidence_boost": signal.cep_confidence_boost,
                "arbitrage_edge": signal.arbitrage_edge
            }
            
            event_bus.publish(
                EventTypes.SIGNAL_GENERATED,
                event_data,
                source="strategy_engine"
            )
            
            self.signals_generated += 1
            self.signals_by_type[signal.signal_type] += 1
            
            logger.info(f"🎯 TRADING SIGNAL: {signal.signal_type} {signal.market_ticker} "
                       f"(confidence: {signal.confidence:.2f}, edge: {signal.arbitrage_edge or 0:.1f}¢) - {signal.reason}")
            
        except Exception as e:
            logger.error(f"Error publishing trading signal: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            "current_brti": self.current_brti,
            "active_opportunities": len(self.active_opportunities),
            "signals_generated": self.signals_generated,
            "signals_by_type": dict(self.signals_by_type),
            "opportunities_analyzed": self.opportunities_analyzed,
            "last_signal_times": dict(self.last_signal_time),
            "signal_frequency_limits": dict(self.signal_frequency_limits),
            "parameters": {
                "min_divergence_pct": self.min_divergence_pct,
                "max_divergence_pct": self.max_divergence_pct,
                "base_confidence_threshold": self.base_confidence_threshold,
                "min_arbitrage_edge": self.min_arbitrage_edge,
                "base_signal_interval": self.base_signal_interval
            }
        }
