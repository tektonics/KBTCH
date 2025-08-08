"""
Configuration manager for the trading system.
Handles all trading parameters and system settings.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Risk management parameters"""
    max_position_size: int = 100  # Maximum position size per market
    max_total_exposure: int = 1000  # Maximum total exposure across all positions
    max_loss_per_trade: int = 50  # Maximum loss per individual trade
    max_daily_loss: int = 500  # Maximum daily loss before stopping


@dataclass
class OrderSettings:
    """Order execution parameters"""
    order_timeout_seconds: int = 30  # How long to wait for order confirmation
    retry_attempts: int = 3  # Number of retry attempts for failed orders
    min_order_size: int = 1  # Minimum order size
    max_order_size: int = 100  # Maximum order size


@dataclass
class StrategySettings:
    """Strategy engine parameters"""
    signal_threshold: float = 0.05  # Minimum signal strength to trade
    max_markets_per_strategy: int = 5  # Maximum markets to trade simultaneously


@dataclass
class SystemSettings:
    """General system parameters"""
    update_frequency_ms: int = 100  # How often to check for trading opportunities
    log_level: str = "INFO"  # Logging level
    enable_live_trading: bool = False  # Safety switch for live trading


class ConfigManager:
    """Centralized configuration management for the trading system"""
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.order_settings = OrderSettings()
        self.strategy_settings = StrategySettings()
        self.system_settings = SystemSettings()
        
        # Override with environment variables if present
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        
        # Risk limits
        if os.getenv("MAX_POSITION_SIZE"):
            self.risk_limits.max_position_size = int(os.getenv("MAX_POSITION_SIZE"))
        if os.getenv("MAX_TOTAL_EXPOSURE"):
            self.risk_limits.max_total_exposure = int(os.getenv("MAX_TOTAL_EXPOSURE"))
        if os.getenv("MAX_LOSS_PER_TRADE"):
            self.risk_limits.max_loss_per_trade = int(os.getenv("MAX_LOSS_PER_TRADE"))
        if os.getenv("MAX_DAILY_LOSS"):
            self.risk_limits.max_daily_loss = int(os.getenv("MAX_DAILY_LOSS"))
        
        # System settings
        if os.getenv("ENABLE_LIVE_TRADING"):
            self.system_settings.enable_live_trading = os.getenv("ENABLE_LIVE_TRADING").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            self.system_settings.log_level = os.getenv("LOG_LEVEL")
    
    def get_risk_limits(self) -> RiskLimits:
        """Get risk management parameters"""
        return self.risk_limits
    
    def get_order_settings(self) -> OrderSettings:
        """Get order execution parameters"""
        return self.order_settings
    
    def get_strategy_settings(self) -> StrategySettings:
        """Get strategy parameters"""
        return self.strategy_settings
    
    def get_system_settings(self) -> SystemSettings:
        """Get system parameters"""
        return self.system_settings
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            "risk_limits": self.risk_limits.__dict__,
            "order_settings": self.order_settings.__dict__,
            "strategy_settings": self.strategy_settings.__dict__,
            "system_settings": self.system_settings.__dict__
        }


config = ConfigManager()
