import time
import logging
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from collections import defaultdict
from event_bus import event_bus, EventTypes

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_value: int = 100
    max_position_count: int = 10
    max_total_exposure: int = 1000
    max_daily_trades: int = 50
    max_daily_loss: int = 5
    min_account_balance: int = 150
    max_market_concentration: float = 0.3
    volatility_scaling: bool = True

class RiskManager:
    
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.limits = RiskLimits()
        
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = time.strftime("%Y-%m-%d")
        self.recent_signals = defaultdict(list)
        self.risk_violations = []
        
        self.min_signal_interval = 5.0
        self.max_signals_per_hour = 20
        
        event_bus.subscribe(EventTypes.SIGNAL_GENERATED, self._check_signal_risk)
        
        logger.info("Risk Manager initialized with limits: "
                   f"max_position=${self.limits.max_position_value}, "
                   f"max_exposure=${self.limits.max_total_exposure}")
    
    def _check_signal_risk(self, event) -> None:
        try:
            signal_data = event.data
            market_ticker = signal_data.get("market_ticker")
            
            self._reset_daily_counters_if_needed()
            
            can_trade, risk_reason = self._validate_all_risks(signal_data)
            
            if can_trade:
                self._track_approved_signal(signal_data)
                
                event_bus.publish(
                    EventTypes.RISK_APPROVED, 
                    signal_data, 
                    source="risk_manager"
                )
                logger.info(f"✅ Risk approved for {market_ticker}: {risk_reason}")
                
            else:
                self._track_risk_violation(signal_data, risk_reason)
                
                rejection_data = {**signal_data, "risk_reason": risk_reason}
                event_bus.publish(
                    EventTypes.RISK_REJECTED, 
                    rejection_data, 
                    source="risk_manager"
                )
                logger.warning(f"❌ Risk rejected for {market_ticker}: {risk_reason}")
                
        except Exception as e:
            logger.error(f"Error in risk checking: {e}")
            event_bus.publish(
                EventTypes.RISK_REJECTED, 
                {**event.data, "risk_reason": f"Risk check error: {e}"}, 
                source="risk_manager"
            )
    
    def _validate_all_risks(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        
        can_trade, reason = self._check_signal_frequency(signal_data)
        if not can_trade:
            return False, reason
        
        can_trade, reason = self._check_daily_limits(signal_data)
        if not can_trade:
            return False, reason
        
        can_trade, reason = self._check_portfolio_risks(signal_data)
        if not can_trade:
            return False, reason
        
        can_trade, reason = self._check_position_sizing(signal_data)
        if not can_trade:
            return False, reason
        
        can_trade, reason = self._check_market_concentration(signal_data)
        if not can_trade:
            return False, reason
        
        return True, "All risk checks passed"
    
    def _check_signal_frequency(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        market_ticker = signal_data.get("market_ticker")
        current_time = time.time()
        
        recent_signals = self.recent_signals[market_ticker]
        
        recent_signals = [t for t in recent_signals if current_time - t < 3600]
        self.recent_signals[market_ticker] = recent_signals
        
        if recent_signals and (current_time - recent_signals[-1]) < self.min_signal_interval:
            return False, f"Signal too frequent: {current_time - recent_signals[-1]:.1f}s < {self.min_signal_interval}s"
        
        if len(recent_signals) >= self.max_signals_per_hour:
            return False, f"Hourly signal limit: {len(recent_signals)} >= {self.max_signals_per_hour}"
        
        return True, "Signal frequency OK"
    
    def _check_daily_limits(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        
        if self.daily_trades >= self.limits.max_daily_trades:
            return False, f"Daily trade limit: {self.daily_trades} >= {self.limits.max_daily_trades}"
        
        if self.daily_pnl < -self.limits.max_daily_loss:
            return False, f"Daily loss limit: ${self.daily_pnl:.2f} < -${self.limits.max_daily_loss}"
        
        return True, "Daily limits OK"
    
    def _check_portfolio_risks(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            balance_data = self.portfolio_manager.get_balance()
            positions_data = self.portfolio_manager.get_positions()
            
            current_balance = balance_data.get('balance', 0) / 100
            
            if current_balance < self.limits.min_account_balance:
                return False, f"Insufficient balance: ${current_balance:.2f} < ${self.limits.min_account_balance}"
            
            positions = positions_data.get('market_positions', [])
            total_exposure = sum(
                abs(pos.get('position', 0)) * pos.get('market_exposure', 0) 
                for pos in positions
            ) / 100
            
            if total_exposure >= self.limits.max_total_exposure:
                return False, f"Total exposure limit: ${total_exposure:.2f} >= ${self.limits.max_total_exposure}"
            
            return True, "Portfolio checks OK"
            
        except Exception as e:
            logger.error(f"Error checking portfolio risks: {e}")
            return False, f"Portfolio data error: {e}"
    
    def _check_position_sizing(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        signal_type = signal_data.get("signal_type")
        market_ticker = signal_data.get("market_ticker")
        quantity = signal_data.get("quantity", 1)
    
        if "YES" in signal_type:
            price_per_contract = signal_data.get("market_yes_price", 0) / 100
        else:
            price_per_contract = signal_data.get("market_no_price", 0) / 100
    
        total_position_value = price_per_contract * quantity
    
        if total_position_value > self.limits.max_position_value:
            return False, f"Position too large: ${total_position_value:.2f} > ${self.limits.max_position_value}"
    
        try:
            positions_data = self.portfolio_manager.get_positions()
            positions = positions_data.get('market_positions', [])
        
            current_position = next(
                (pos for pos in positions if pos.get('market_ticker') == market_ticker), 
                None
            )
        
            if current_position:
                current_count = abs(current_position.get('position', 0))
                if current_count >= self.limits.max_position_count:
                    return False, f"Position count limit: {current_count} >= {self.limits.max_position_count}"
        
        except Exception as e:
            logger.warning(f"Could not check existing positions: {e}")
    
        return True, "Position sizing OK"

    def _check_market_concentration(self, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            market_ticker = signal_data.get("market_ticker")
            
            positions_data = self.portfolio_manager.get_positions()
            positions = positions_data.get('market_positions', [])
            
            total_portfolio_value = sum(
                abs(pos.get('position', 0)) * pos.get('market_exposure', 0) 
                for pos in positions
            ) / 100
            
            if total_portfolio_value == 0:
                return True, "No existing positions"
            
            market_position = next(
                (pos for pos in positions if pos.get('market_ticker') == market_ticker), 
                None
            )
            
            if market_position:
                market_exposure = abs(market_position.get('position', 0)) * market_position.get('market_exposure', 0) / 100
                concentration = market_exposure / total_portfolio_value
                
                if concentration > self.limits.max_market_concentration:
                    return False, f"Market concentration: {concentration:.1%} > {self.limits.max_market_concentration:.1%}"
            
            return True, "Market concentration OK"
            
        except Exception as e:
            logger.warning(f"Could not check market concentration: {e}")
            return True, "Concentration check skipped"
    
    def _reset_daily_counters_if_needed(self) -> None:
        current_date = time.strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            logger.info(f"New trading day: resetting daily counters")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def _track_approved_signal(self, signal_data: Dict[str, Any]) -> None:
        market_ticker = signal_data.get("market_ticker")
        current_time = time.time()
        
        self.recent_signals[market_ticker].append(current_time)
        
        self.daily_trades += 1
    
    def _track_risk_violation(self, signal_data: Dict[str, Any], reason: str) -> None:
        violation = {
            'timestamp': time.time(),
            'market_ticker': signal_data.get('market_ticker'),
            'signal_type': signal_data.get('signal_type'),
            'reason': reason,
            'confidence': signal_data.get('confidence'),
            'arbitrage_edge': signal_data.get('arbitrage_edge')
        }
        
        self.risk_violations.append(violation)
        
        if len(self.risk_violations) > 1000:
            self.risk_violations = self.risk_violations[-1000:]
    
    def update_daily_pnl(self, pnl_change: float) -> None:
        self.daily_pnl += pnl_change
        logger.debug(f"Daily P&L updated: ${self.daily_pnl:.2f}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'risk_violations_today': len([v for v in self.risk_violations 
                                        if time.time() - v['timestamp'] < 86400]),
            'active_markets': len(self.recent_signals),
            'limits': {
                'max_position_value': self.limits.max_position_value,
                'max_total_exposure': self.limits.max_total_exposure,
                'max_daily_trades': self.limits.max_daily_trades,
                'max_daily_loss': self.limits.max_daily_loss
            },
            'recent_violations': self.risk_violations[-5:] if self.risk_violations else []
        }
    
    def emergency_stop(self) -> None:
        logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        self.limits.max_daily_trades = 0
        self.limits.max_position_value = 0
        
        event_bus.publish(
            EventTypes.RISK_VIOLATION,
            {'type': 'EMERGENCY_STOP', 'timestamp': time.time()},
            source="risk_manager"
        )
