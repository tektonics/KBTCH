"""
Trading system configuration
"""

TRADING_CONFIG = {
    'mode': 'simulation',  # 'live' or 'simulation'
    'strategy': 'momentum',  # 'momentum', 'mean_reversion', etc.
    'risk_limits': {
        'max_position_size': 1000,
        'max_daily_loss': 500,
        'max_portfolio_value': 10000,
        'max_single_market_exposure': 0.2  # 20% of portfolio
    },
    'portfolio': {
        'initial_cash': 10000,
        'currency': 'USD'
    },
    'api': {
        'base_url': 'https://trading-api.kalshi.com/trade-api/v2',
        'ws_url': 'wss://trading-api.kalshi.com/trade-api/ws/v2',
        'rate_limit_delay': 0.1  # seconds between API calls
    }
}

# Strategy-specific configurations
STRATEGY_CONFIG = {
    'momentum': {
        'lookback_period': 10,
        'momentum_threshold': 0.02,
        'position_size': 100
    },
    'mean_reversion': {
        'lookback_period': 20,
        'deviation_threshold': 2.0,
        'position_size': 50
    }
}
