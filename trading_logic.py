from typing import List
import numpy as np
from market_info import MarketInfo, TradingParams
from kalshi_bot.kalshi_client import KalshiClient, TickerData
from portfolio import Portfolio
from strategy import evaluate_exit_signal
from risk_manager import RiskManager

def estimate_theoretical_probability(strike: float, current_price: float, volatility: float, time_hours: float = 1.0) -> float:
    if strike <= 0 or current_price <= 0 or time_hours <= 0:
        return 0.5

    price_ratio = current_price / strike
    log_ratio = np.log(price_ratio)
    vol_sqrt_time = volatility * np.sqrt(time_hours / 8760)

    if vol_sqrt_time <= 0:
        return 1.0 if current_price > strike else 0.0

    z_score = log_ratio / vol_sqrt_time
    prob = 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
    return max(0.01, min(0.99, prob))

def determine_trading_action(market_info: MarketInfo, btc_price: float, params: TradingParams, portfolio: Portfolio, risk_manager: RiskManager) -> str:
    if not market_info.market_data or not market_info.spread_pct or not market_info.edge:
        return "NO_DATA"

    if market_info.spread_pct > params.max_spread_threshold:
        return "SPREAD_TOO_WIDE"

    position = portfolio.get_position(market_info.ticker)

    if position:
        # Check for exit
        unrealized_pnl = portfolio.calculate_unrealized_pnl(market_info.ticker, market_info.market_data.price)
        if evaluate_exit_signal(market_info, position, unrealized_pnl) or risk_manager.trigger_exit(unrealized_pnl):
            return "SELL_YES" if position.side == "YES" else "SELL_NO"
        return "HOLD"

    if abs(market_info.edge) < params.min_edge_threshold:
        return "INSUFFICIENT_EDGE"

    if btc_price > market_info.strike:
        if market_info.edge > params.min_edge_threshold:
            return "BUY_YES"
        else:
            return "BUY_NO"
    else:
        if market_info.edge > params.min_edge_threshold:
            return "BUY_NO"
        else:
            return "BUY_YES"

def apply_trading_logic_to_markets(kalshi_client: KalshiClient, markets: List[MarketInfo], btc_price: float, params: TradingParams, volatility: float, portfolio: Portfolio, risk_manager: RiskManager) -> List[MarketInfo]:
    for market in markets:
        market.market_data = kalshi_client.get_mid_prices(market.ticker)

        if market.market_data and market.market_data.yes_bid is not None and market.market_data.yes_ask is not None:
            market.spread = market.market_data.yes_ask - market.market_data.yes_bid
            market.spread_pct = (market.spread / market.market_data.yes_ask) * 100 if market.market_data.yes_ask else None

            if btc_price > market.strike and market.market_data.yes_ask:
                market.implied_prob = market.market_data.yes_ask / 100
            elif btc_price <= market.strike and market.market_data.yes_bid:
                market.implied_prob = market.market_data.yes_bid / 100

            if market.implied_prob:
                theoretical_prob = estimate_theoretical_probability(
                    market.strike, btc_price, volatility, 1.0
                )
                market.edge = theoretical_prob - market.implied_prob

        market.action = determine_trading_action(market, btc_price, params, portfolio, risk_manager)
    return markets

def evaluate_exit_signal(market_info, portfolio, config) -> Optional[str]:
    position = portfolio.get_position(market_info.ticker)

    if not position or position.quantity == 0:
        return None  # No position to exit

    # Calculate P&L percentage
    current_price = market_info.market_data.price
    entry_price = position.average_price
    pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0

    if pnl_pct >= config.profit_take_pct:
        return f"SELL_{'YES' if position.contract_type == 'YES' else 'NO'}"
    elif pnl_pct <= -config.stop_loss_pct:
        return f"SELL_{'YES' if position.contract_type == 'YES' else 'NO'}"

    return None
