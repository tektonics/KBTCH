import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_size: int = 100
    max_total_exposure: int = 1000
    max_loss_per_trade: int = 50
    max_daily_loss: int = 500


@dataclass
class OrderSettings:
    order_timeout_seconds: int = 30
    retry_attempts: int = 3
    min_order_size: int = 1
    max_order_size: int = 100


@dataclass
class StrategySettings:
    signal_threshold: float = 0.05
    max_markets_per_strategy: int = 5


@dataclass
class SystemSettings:
    update_frequency_ms: int = 100
    log_level: str = "INFO"
    enable_live_trading: bool = False


@dataclass
class PapertradingSettings:
    initial_balance: int = 100000
    fill_delay_seconds: float = 0.5
    fill_probability: float = 0.95
    slippage_bps: int = 5
    commission_per_contract: int = 0
    max_daily_trades: int = 100
    enable_market_impact: bool = True
    price_improvement_probability: float = 0.1


class ConfigManager:
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.order_settings = OrderSettings()
        self.strategy_settings = StrategySettings()
        self.system_settings = SystemSettings()
        self.papertrading_settings = PapertradingSettings()
        self._load_env_overrides()

    def _load_env_overrides(self):
        if os.getenv("MAX_POSITION_SIZE"):
            self.risk_limits.max_position_size = int(os.getenv("MAX_POSITION_SIZE"))
        if os.getenv("MAX_TOTAL_EXPOSURE"):
            self.risk_limits.max_total_exposure = int(os.getenv("MAX_TOTAL_EXPOSURE"))
        if os.getenv("MAX_LOSS_PER_TRADE"):
            self.risk_limits.max_loss_per_trade = int(os.getenv("MAX_LOSS_PER_TRADE"))
        if os.getenv("MAX_DAILY_LOSS"):
            self.risk_limits.max_daily_loss = int(os.getenv("MAX_DAILY_LOSS"))
        
        if os.getenv("LOG_LEVEL"):
            self.system_settings.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("ENABLE_LIVE_TRADING"):
            self.system_settings.enable_live_trading = os.getenv("ENABLE_LIVE_TRADING").lower() == "true"
        
        if os.getenv("INITIAL_BALANCE"):
            self.papertrading_settings.initial_balance = int(os.getenv("INITIAL_BALANCE"))
        if os.getenv("FILL_DELAY_SECONDS"):
            self.papertrading_settings.fill_delay_seconds = float(os.getenv("FILL_DELAY_SECONDS"))
        if os.getenv("FILL_PROBABILITY"):
            self.papertrading_settings.fill_probability = float(os.getenv("FILL_PROBABILITY"))
        
    def get_risk_limits(self) -> RiskLimits:
        return self.risk_limits
    
    def get_order_settings(self) -> OrderSettings:
        return self.order_settings
    
    def get_strategy_settings(self) -> StrategySettings:
        return self.strategy_settings
    
    def get_system_settings(self) -> SystemSettings:
        return self.system_settings
    
    def get_papertrading_settings(self) -> PapertradingSettings:
        return self.papertrading_settings
    
    def is_papertrading_enabled(self) -> bool:
        return not self.system_settings.enable_live_trading
    
    def is_live_trading_enabled(self) -> bool:
        return self.system_settings.enable_live_trading
    
    def get_trading_mode(self) -> str:
        return "live" if self.system_settings.enable_live_trading else "papertrading"
    
    def set_trading_mode(self, mode: str) -> bool:
        if mode not in ["papertrading", "live"]:
            print(f"❌ Invalid trading mode: {mode}. Must be 'papertrading' or 'live'")
            return False
        
        self.system_settings.enable_live_trading = (mode == "live")
        
        mode_emoji = "🎮" if mode == "papertrading" else "🔴"
        print(f"{mode_emoji} Trading mode set to: {mode.upper()}")
        return True
    
    def get_all_config(self) -> Dict[str, Any]:
        return {
            "trading_mode": self.get_trading_mode(),
            "is_papertrading": self.is_papertrading_enabled(),
            "is_live_trading": self.is_live_trading_enabled(),
            "risk_limits": self.risk_limits.__dict__,
            "order_settings": self.order_settings.__dict__,
            "strategy_settings": self.strategy_settings.__dict__,
            "system_settings": self.system_settings.__dict__,
            "papertrading_settings": self.papertrading_settings.__dict__
        }


config = ConfigManager()
