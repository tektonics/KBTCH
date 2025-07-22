"""
Trading strategies and signal generation
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict, deque
from portfolio import Portfolio
from risk_manager import OrderSignal
from config import STRATEGY_CONFIG

@dataclass
class MarketData:
    ticker: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: float
    open_interest: int = 0

class Strategy(ABC):
    """Base strategy class"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.market_history = defaultdict(lambda: deque(maxlen=100))  # Store recent data
        self.indicators = defaultdict(dict)  # Store calculated indicators
        
    @abstractmethod
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        """Generate trading signals based on market data and current portfolio"""
        pass
    
    def update_market_data(self, data: MarketData):
        """Update internal market data history"""
        self.market_history[data.ticker].append({
            'price': data.price,
            'bid': data.bid,
            'ask': data.ask,
            'volume': data.volume,
            'timestamp': data.timestamp,
            'open_interest': data.open_interest
        })
    
    def get_price_history(self, ticker: str, periods: int = None) -> List[float]:
        """Get recent price history for a ticker"""
        history = list(self.market_history[ticker])
        if periods:
            history = history[-periods:]
        return [item['price'] for item in history]
    
    def calculate_moving_average(self, ticker: str, periods: int) -> float:
        """Calculate simple moving average"""
        prices = self.get_price_history(ticker, periods)
        if len(prices) < periods:
            return None
        return sum(prices) / len(prices)
    
    def calculate_price_change(self, ticker: str, periods: int = 1) -> float:
        """Calculate price change over specified periods"""
        prices = self.get_price_history(ticker, periods + 1)
        if len(prices) < periods + 1:
            return None
        return (prices[-1] - prices[-periods-1]) / prices[-periods-1]

class MomentumStrategy(Strategy):
    """Simple momentum strategy - buy on upward momentum, sell on downward"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        strategy_config = STRATEGY_CONFIG.get('momentum', {})
        self.lookback_period = strategy_config.get('lookback_period', 10)
        self.momentum_threshold = strategy_config.get('momentum_threshold', 0.02)
        self.position_size = strategy_config.get('position_size', 100)
    
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        signals = []
        
        for data in market_data:
            self.update_market_data(data)
            
            # Calculate momentum
            momentum = self.calculate_price_change(data.ticker, self.lookback_period)
            if momentum is None:
                continue
            
            current_position = portfolio.get_position(data.ticker)
            current_quantity = current_position.quantity if current_position else 0
            
            # Generate signals based on momentum
            if momentum > self.momentum_threshold and current_quantity <= 0:
                # Strong upward momentum - buy signal
                signals.append(OrderSignal(
                    market_ticker=data.ticker,
                    side='buy',
                    quantity=self.position_size,
                    price=data.ask,  # Use ask price for buys
                    reason=f"Momentum buy: {momentum:.3f} > {self.momentum_threshold}"
                ))
            
            elif momentum < -self.momentum_threshold and current_quantity >= 0:
                # Strong downward momentum - sell signal
                quantity = max(current_quantity, self.position_size)  # Sell existing or go short
                signals.append(OrderSignal(
                    market_ticker=data.ticker,
                    side='sell',
                    quantity=quantity,
                    price=data.bid,  # Use bid price for sells
                    reason=f"Momentum sell: {momentum:.3f} < {-self.momentum_threshold}"
                ))
        
        return signals

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy - buy when below mean, sell when above"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        strategy_config = STRATEGY_CONFIG.get('mean_reversion', {})
        self.lookback_period = strategy_config.get('lookback_period', 20)
        self.deviation_threshold = strategy_config.get('deviation_threshold', 2.0)
        self.position_size = strategy_config.get('position_size', 50)
    
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        signals = []
        
        for data in market_data:
            self.update_market_data(data)
            
            # Calculate mean and standard deviation
            prices = self.get_price_history(data.ticker, self.lookback_period)
            if len(prices) < self.lookback_period:
                continue
            
            mean_price = sum(prices) / len(prices)
            variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
            std_dev = variance ** 0.5
            
            current_price = data.price
            deviation = (current_price - mean_price) / std_dev if std_dev > 0 else 0
            
            current_position = portfolio.get_position(data.ticker)
            current_quantity = current_position.quantity if current_position else 0
            
            # Generate signals based on mean reversion
            if deviation < -self.deviation_threshold and current_quantity <= 0:
                # Price significantly below mean - buy signal
                signals.append(OrderSignal(
                    market_ticker=data.ticker,
                    side='buy',
                    quantity=self.position_size,
                    price=data.ask,
                    reason=f"Mean reversion buy: deviation {deviation:.2f}"
                ))
            
            elif deviation > self.deviation_threshold and current_quantity >= 0:
                # Price significantly above mean - sell signal
                quantity = max(current_quantity, self.position_size)
                signals.append(OrderSignal(
                    market_ticker=data.ticker,
                    side='sell',
                    quantity=quantity,
                    price=data.bid,
                    reason=f"Mean reversion sell: deviation {deviation:.2f}"
                ))
        
        return signals

class SpreadStrategy(Strategy):
    """Strategy that trades on spread compression/expansion"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.max_spread_threshold = config.get('max_spread_threshold', 0.05)  # 5%
        self.min_spread_threshold = config.get('min_spread_threshold', 0.01)  # 1%
        self.position_size = config.get('position_size', 50)
    
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        signals = []
        
        for data in market_data:
            self.update_market_data(data)
            
            # Calculate bid-ask spread
            if data.bid <= 0 or data.ask <= 0:
                continue
            
            spread = (data.ask - data.bid) / data.price
            
            current_position = portfolio.get_position(data.ticker)
            current_quantity = current_position.quantity if current_position else 0
            
            # Trade on spread anomalies
            if spread > self.max_spread_threshold and current_quantity == 0:
                # Wide spread - avoid trading or provide liquidity
                # Could implement market making here
                pass
            
            elif spread < self.min_spread_threshold and abs(current_quantity) < self.position_size:
                # Tight spread - good for taking positions
                # Simple momentum signal when spread is tight
                recent_prices = self.get_price_history(data.ticker, 3)
                if len(recent_prices) >= 3:
                    if recent_prices[-1] > recent_prices[-2] > recent_prices[-3]:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='buy',
                            quantity=self.position_size - current_quantity,
                            price=data.ask,
                            reason=f"Tight spread momentum buy: spread {spread:.3f}"
                        ))
                    elif recent_prices[-1] < recent_prices[-2] < recent_prices[-3]:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='sell',
                            quantity=self.position_size + current_quantity,
                            price=data.bid,
                            reason=f"Tight spread momentum sell: spread {spread:.3f}"
                        ))
        
        return signals

class VolumeStrategy(Strategy):
    """Strategy that trades based on volume patterns"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.volume_lookback = config.get('volume_lookback', 10)
        self.volume_threshold = config.get('volume_threshold', 2.0)  # 2x average volume
        self.position_size = config.get('position_size', 75)
    
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        signals = []
        
        for data in market_data:
            self.update_market_data(data)
            
            # Calculate average volume
            volume_history = [item['volume'] for item in list(self.market_history[data.ticker])]
            if len(volume_history) < self.volume_lookback:
                continue
            
            avg_volume = sum(volume_history[-self.volume_lookback:]) / self.volume_lookback
            volume_ratio = data.volume / avg_volume if avg_volume > 0 else 0
            
            # Get price momentum
            price_change = self.calculate_price_change(data.ticker, 1)
            if price_change is None:
                continue
            
            current_position = portfolio.get_position(data.ticker)
            current_quantity = current_position.quantity if current_position else 0
            
            # High volume breakout strategy
            if volume_ratio > self.volume_threshold:
                if price_change > 0.01 and current_quantity <= 0:  # 1% price increase with high volume
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Volume breakout buy: vol_ratio {volume_ratio:.2f}, price_chg {price_change:.3f}"
                    ))
                elif price_change < -0.01 and current_quantity >= 0:  # 1% price decrease with high volume
                    quantity = max(current_quantity, self.position_size)
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='sell',
                        quantity=quantity,
                        price=data.bid,
                        reason=f"Volume breakdown sell: vol_ratio {volume_ratio:.2f}, price_chg {price_change:.3f}"
                    ))
        
        return signals

class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_name: str, config: Dict = None) -> Strategy:
        strategies = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'spread': SpreadStrategy,
            'volume': VolumeStrategy
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
        
        return strategies[strategy_name](config)

class MultiStrategy(Strategy):
    """Combines multiple strategies"""
    
    def __init__(self, strategies: List[Strategy], weights: List[float] = None):
        super().__init__()
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
    
    def generate_signals(self, market_data: List[MarketData], portfolio: Portfolio) -> List[OrderSignal]:
        all_signals = []
        
        # Get signals from all strategies
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_signals = strategy.generate_signals(market_data, portfolio)
            
            # Apply weight to position sizes
            for signal in strategy_signals:
                signal.quantity = int(signal.quantity * weight)
                signal.reason = f"[{strategy.__class__.__name__}] {signal.reason}"
            
            all_signals.extend(strategy_signals)
        
        # Combine signals for the same market
        combined_signals = self._combine_signals(all_signals)
        
        return combined_signals
    
    def _combine_signals(self, signals: List[OrderSignal]) -> List[OrderSignal]:
        """Combine multiple signals for the same market"""
        signal_groups = defaultdict(list)
        
        # Group signals by market
        for signal in signals:
            signal_groups[signal.market_ticker].append(signal)
        
        combined = []
        for ticker, ticker_signals in signal_groups.items():
            if len(ticker_signals) == 1:
                combined.append(ticker_signals[0])
            else:
                # Combine multiple signals
                net_buy_quantity = sum(s.quantity for s in ticker_signals if s.side == 'buy')
                net_sell_quantity = sum(s.quantity for s in ticker_signals if s.side == 'sell')
                
                net_quantity = net_buy_quantity - net_sell_quantity
                
                if net_quantity > 0:
                    # Net buy signal
                    avg_price = sum(s.price * s.quantity for s in ticker_signals if s.side == 'buy') / net_buy_quantity
                    combined.append(OrderSignal(
                        market_ticker=ticker,
                        side='buy',
                        quantity=net_quantity,
                        price=avg_price,
                        reason=f"Combined: {len(ticker_signals)} signals"
                    ))
                elif net_quantity < 0:
                    # Net sell signal
                    avg_price = sum(s.price * s.quantity for s in ticker_signals if s.side == 'sell') / net_sell_quantity
                    combined.append(OrderSignal(
                        market_ticker=ticker,
                        side='sell',
                        quantity=abs(net_quantity),
                        price=avg_price,
                        reason=f"Combined: {len(ticker_signals)} signals"
                    ))
                # If net_quantity == 0, signals cancel out
        
        return combined
