from typing import List, Dict, Any, Optional
import numpy as np
import time
from dataclasses import dataclass
from trading.portfolio import Portfolio


@dataclass
class TradingDecision:
    ticker: str
    action: str  
    quantity: int
    price: float
    confidence: float
    edge: float
    spread_pct: float
    reason: str
    market_info: Dict[str, Any]
    risk_approved: bool = False

class TradingLogic:
    
    def __init__(self, risk_manager: RiskManager, config: Dict[str, Any]):
        self.risk_manager = risk_manager
        self.config = config
        
        self.min_edge_threshold = config.get('min_edge_threshold', 0.05)
        self.max_spread_threshold = config.get('max_spread_threshold', 8.0)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        self.base_position_size = config.get('base_position_size', 100)
        
        # Market analysis cache
        self.market_cache = {}
        self.last_analysis_time = {}
    
    def make_trading_decisions(self, trader: VolatilityAdaptiveTrader) -> List[TradingDecision]:
        decisions = []
   
        markets = trader.active_markets
        btc_price = trader.last_btc_price
        volatility = trader.current_volatility
        portfolio_data = trader.portfolio_data
        ohlcv_data = trader.ohlcv_monitor.get_ohlcv_data()
        brti_running = trader.brti_manager.is_brti_running()
        btc_monitor = trader.btc_monitor
   
        if not btc_price or not markets:
            return decisions
   
        portfolio = self._create_portfolio_from_kalshi_data(portfolio_data)
   
        for market in markets:
            try:
                if not market.market_data:
                    continue
               
                market_analysis = self._analyze_market_opportunity(market, btc_price, volatility, ohlcv_data, brti_running)
           
                if not market_analysis:
                    continue
           
                strategy_signal = self._get_strategy_signal(market, market_analysis, portfolio, btc_monitor)
           
                if strategy_signal['action'] == 'NO_TRADE':
                    continue
           
                risk_validated_signal = self._apply_risk_management(strategy_signal, portfolio, market_analysis)
           
                if risk_validated_signal['approved']:
                    decision = TradingDecision(
                        ticker=market.ticker,
                        action=risk_validated_signal['action'],
                        quantity=risk_validated_signal['quantity'],
                        price=risk_validated_signal['price'],
                        confidence=strategy_signal['confidence'],
                        edge=market_analysis['edge'],
                        spread_pct=market_analysis['spread_pct'],
                        reason=risk_validated_signal['reason'],
                        market_info=market_analysis,
                        risk_approved=True
                    )
                    decisions.append(decision)
               
            except Exception as e:
                print(f"Error processing {market.ticker}: {e}")
                continue
   
        return decisions
    
    def _analyze_market_opportunity(self, market_data: MarketDataPoint, 
                                   btc_price: float, volatility: float) -> Optional[Dict[str, Any]]:
        """Analyze a single market for trading opportunities"""
        
        # Check data quality
        if not self._is_market_data_valid(market_data):
            return None
        
        # Calculate basic metrics
        analysis = {
            'ticker': market_data.ticker,
            'strike': market_data.strike,
            'btc_price': btc_price,
            'volatility': volatility,
            'timestamp': time.time()
        }
        
        # Calculate spread
        if market_data.yes_bid and market_data.yes_ask:
            analysis['spread'] = market_data.yes_ask - market_data.yes_bid
            analysis['spread_pct'] = (analysis['spread'] / market_data.yes_ask) * 100
            analysis['mid_price'] = (market_data.yes_bid + market_data.yes_ask) / 2
        else:
            return None
        
        # Check spread threshold
        if analysis['spread_pct'] > self.max_spread_threshold:
            return None
        
        # Calculate implied probability
        analysis['implied_prob'] = self._calculate_implied_probability(
            market_data, btc_price
        )
        
        # Calculate theoretical probability
        analysis['theoretical_prob'] = self._calculate_theoretical_probability(
            market_data.strike, btc_price, volatility
        )
        
        # Calculate edge
        if analysis['implied_prob'] and analysis['theoretical_prob']:
            analysis['edge'] = analysis['theoretical_prob'] - analysis['implied_prob']
        else:
            analysis['edge'] = 0.0
        
        # Check minimum edge threshold
        if abs(analysis['edge']) < self.min_edge_threshold:
            return None
        
        return analysis
    
    def _is_market_data_valid(self, market_data: MarketDataPoint) -> bool:
        """Validate market data quality"""
        if not market_data.yes_bid or not market_data.yes_ask:
            return False
        
        if market_data.yes_bid >= market_data.yes_ask:
            return False
        
        if market_data.yes_bid <= 0 or market_data.yes_ask <= 0:
            return False
        
        if market_data.yes_ask > 100 or market_data.yes_bid > 100:
            return False
        
        return True
    
    def _calculate_implied_probability(self, market_data: MarketDataPoint, btc_price: float) -> float:
        """Calculate market's implied probability"""
        if btc_price > market_data.strike:
            # BTC is above strike, use YES ask price
            return market_data.yes_ask / 100 if market_data.yes_ask else 0.0
        else:
            # BTC is below strike, use YES bid price
            return market_data.yes_bid / 100 if market_data.yes_bid else 0.0
    
    def _calculate_theoretical_probability(self, strike: float, current_price: float, 
                                         volatility: float, time_hours: float = 1.0) -> float:
        """Calculate theoretical probability using simplified model"""
        if strike <= 0 or current_price <= 0 or time_hours <= 0:
            return 0.5
        
        price_ratio = current_price / strike
        log_ratio = np.log(price_ratio)
        vol_sqrt_time = volatility * np.sqrt(time_hours / 8760)  # Convert to fraction of year
        
        if vol_sqrt_time <= 0:
            return 1.0 if current_price > strike else 0.0
        
        # Simplified probability calculation
        z_score = log_ratio / vol_sqrt_time
        prob = 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
        
        return max(0.01, min(0.99, prob))
    
    def _determine_trading_action(self, 
                                market_analysis: Dict[str, Any], 
                                market_data: MarketDataPoint) -> str:
        """Determine specific trading action based on strategy and market analysis"""
        
        edge = market_analysis['edge']
        btc_price = market_analysis['btc_price']
        strike = market_analysis['strike']
        
        # If strategy suggests buying and we have positive edge
        if strategy_signal.side == 'buy' and edge > self.min_edge_threshold:
            if btc_price > strike:
                return 'BUY_YES'  # BTC above strike, buy YES
            else:
                return 'BUY_NO'   # BTC below strike, buy NO
        
        # If strategy suggests selling and we have negative edge (overvalued)
        elif strategy_signal.side == 'sell' and edge < -self.min_edge_threshold:
            if btc_price > strike:
                return 'SELL_NO'  # BTC above strike, sell NO
            else:
                return 'SELL_YES' # BTC below strike, sell YES
        
        return 'NO_TRADE'
    
    def evaluate_exit_signals(self, portfolio: Portfolio, current_market_data: List[MarketDataPoint]) -> List[TradingDecision]:
        """Evaluate existing positions for exit signals"""
        exit_decisions = []
        
        # Create a lookup for current market data
        market_lookup = {data.ticker: data for data in current_market_data}
        
        for ticker, position in portfolio.positions.items():
            if position.quantity == 0:
                continue
            
            current_data = market_lookup.get(ticker)
            if not current_data:
                continue
            
            # Calculate current P&L
            if position.quantity > 0:  # Long position
                current_price = current_data.yes_bid or position.avg_price
            else:  # Short position
                current_price = current_data.yes_ask or position.avg_price
            
            pnl_pct = ((current_price - position.avg_price) / position.avg_price) * 100
            
            # Check exit conditions
            exit_action = None
            exit_reason = None
            
            # Profit taking
            profit_target = self.config.get('profit_take_pct', 20.0)
            if pnl_pct >= profit_target:
                exit_action = 'SELL_YES' if position.quantity > 0 else 'BUY_YES'
                exit_reason = f"Profit taking at {pnl_pct:.1f}%"
            
            # Stop loss
            stop_loss = self.config.get('stop_loss_pct', -10.0)
            if pnl_pct <= stop_loss:
                exit_action = 'SELL_YES' if position.quantity > 0 else 'BUY_YES'
                exit_reason = f"Stop loss at {pnl_pct:.1f}%"
            
            # Time-based exit (if position held too long)
            max_hold_hours = self.config.get('max_hold_hours', 24)
            position_age_hours = (time.time() - position.timestamp.timestamp()) / 3600
            if position_age_hours >= max_hold_hours:
                exit_action = 'SELL_YES' if position.quantity > 0 else 'BUY_YES'
                exit_reason = f"Time-based exit after {position_age_hours:.1f}h"
            
            if exit_action:
                exit_decision = TradingDecision(
                    ticker=ticker,
                    action=exit_action,
                    quantity=abs(position.quantity),
                    price=current_price,
                    confidence=1.0,  # High confidence for exits
                    edge=0.0,  # Not applicable for exits
                    spread_pct=0.0,  # Not applicable for exits
                    reason=exit_reason,
                    market_info={'type': 'exit', 'pnl_pct': pnl_pct},
                    risk_approved=True  # Exits are generally approved
                )
                exit_decisions.append(exit_decision)
        
        return exit_decisions
