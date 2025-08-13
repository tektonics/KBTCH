# config/config_manager.py (Updated)
"""
Configuration manager for the trading system.
Handles all trading parameters and system settings.
"""

import os
import json
from pathlib import Path
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


@dataclass
class PapertradingSettings:
    """Papertrading specific configuration"""
    initial_balance: int = 100000  # $1000 in cents
    fill_delay_seconds: float = 0.5  # Simulate realistic fill delays
    fill_probability: float = 0.95  # 95% of orders get filled
    slippage_bps: int = 5  # 5 basis points slippage
    commission_per_contract: int = 0  # Commission in cents per contract
    max_daily_trades: int = 100  # Maximum trades per day in papertrade mode
    enable_market_impact: bool = True  # Simulate market impact on fills
    price_improvement_probability: float = 0.1  # 10% chance of price improvement


class ConfigManager:
    """Centralized configuration management for the trading system"""
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.order_settings = OrderSettings()
        self.strategy_settings = StrategySettings()
        self.system_settings = SystemSettings()
        self.papertrading_settings = PapertradingSettings()
        
        # Load trading mode configuration
        self.trading_config = self._load_trading_config()
        
        # Override with environment variables if present (only for non-trading mode settings)
        self._load_env_overrides()
    
    def _load_trading_config(self) -> Dict[str, Any]:
        """Load trading configuration from separate JSON file"""
        config_file = Path("trading_config.json")
        
        # Default configuration
        default_config = {
            "trading_mode": "papertrading",  # "papertrading" or "live"
            "papertrading": {
                "initial_balance_usd": 1000,
                "fill_probability": 0.95,
                "slippage_bps": 5,
                "fill_delay_seconds": 0.5,
                "commission_per_contract": 0,
                "price_improvement_probability": 0.1,
                "enable_market_impact": True
            },
            "live_trading": {
                "safety_enabled": True,
                "max_position_size": 100,
                "max_total_exposure": 1000
            }
        }
        
        # Create config file if it doesn't exist
        if not config_file.exists():
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📁 Created trading configuration file: {config_file}")
            print("🎮 Default mode: PAPERTRADING (safe for testing)")
        
        # Load existing config
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            for key, value in default_config.items():
                if key not in config_data:
                    config_data[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config_data[key]:
                            config_data[key][subkey] = subvalue
            
            return config_data
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️  Error loading trading config: {e}")
            print("🔄 Using default configuration")
            return default_config
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables (non-trading mode settings only)"""
        
        # Risk limits (these can still come from .env)
        if os.getenv("MAX_POSITION_SIZE"):
            self.risk_limits.max_position_size = int(os.getenv("MAX_POSITION_SIZE"))
        if os.getenv("MAX_TOTAL_EXPOSURE"):
            self.risk_limits.max_total_exposure = int(os.getenv("MAX_TOTAL_EXPOSURE"))
        if os.getenv("MAX_LOSS_PER_TRADE"):
            self.risk_limits.max_loss_per_trade = int(os.getenv("MAX_LOSS_PER_TRADE"))
        if os.getenv("MAX_DAILY_LOSS"):
            self.risk_limits.max_daily_loss = int(os.getenv("MAX_DAILY_LOSS"))
        
        # System settings (non-trading mode)
        if os.getenv("LOG_LEVEL"):
            self.system_settings.log_level = os.getenv("LOG_LEVEL")
        
        # Apply trading config to dataclasses
        self._apply_trading_config()
    
    def _apply_trading_config(self):
        """Apply loaded trading configuration to settings objects"""
        papertrading_config = self.trading_config.get("papertrading", {})
        
        # Update papertrading settings from config file
        self.papertrading_settings.initial_balance = int(
            papertrading_config.get("initial_balance_usd", 1000) * 100
        )
        self.papertrading_settings.fill_probability = papertrading_config.get("fill_probability", 0.95)
        self.papertrading_settings.slippage_bps = papertrading_config.get("slippage_bps", 5)
        self.papertrading_settings.fill_delay_seconds = papertrading_config.get("fill_delay_seconds", 0.5)
        self.papertrading_settings.commission_per_contract = papertrading_config.get("commission_per_contract", 0)
        self.papertrading_settings.price_improvement_probability = papertrading_config.get("price_improvement_probability", 0.1)
        self.papertrading_settings.enable_market_impact = papertrading_config.get("enable_market_impact", True)
        
        # Update live trading safety settings
        live_config = self.trading_config.get("live_trading", {})
        self.system_settings.enable_live_trading = not live_config.get("safety_enabled", True)
    
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
    
    def get_papertrading_settings(self) -> PapertradingSettings:
        """Get papertrading parameters"""
        return self.papertrading_settings
    
    def is_papertrading_enabled(self) -> bool:
        """Check if papertrading mode is enabled"""
        return self.trading_config.get("trading_mode", "papertrading") == "papertrading"
    
    def get_trading_mode(self) -> str:
        """Get current trading mode"""
        return self.trading_config.get("trading_mode", "papertrading")
    
    def set_trading_mode(self, mode: str) -> bool:
        """Set trading mode and save to config file"""
        if mode not in ["papertrading", "live"]:
            print(f"❌ Invalid trading mode: {mode}. Must be 'papertrading' or 'live'")
            return False
        
        self.trading_config["trading_mode"] = mode
        
        try:
            config_file = Path("trading_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.trading_config, f, indent=2)
            
            # Reapply configuration
            self._apply_trading_config()
            
            mode_emoji = "🎮" if mode == "papertrading" else "🔴"
            print(f"{mode_emoji} Trading mode set to: {mode.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving trading mode: {e}")
            return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            "trading_mode": self.get_trading_mode(),
            "is_papertrading": self.is_papertrading_enabled(),
            "risk_limits": self.risk_limits.__dict__,
            "order_settings": self.order_settings.__dict__,
            "strategy_settings": self.strategy_settings.__dict__,
            "system_settings": self.system_settings.__dict__,
            "papertrading_settings": self.papertrading_settings.__dict__,
            "trading_config": self.trading_config
        }


config = ConfigManager()

