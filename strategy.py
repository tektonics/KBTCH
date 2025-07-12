"""
Trading strategies and signal generation with Kalshi contract awareness
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
    """Base strategy class with Kalshi contract awareness"""
    
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
    
    def _can_sell_position(self, ticker: str, portfolio: Portfolio, contract_type: str = 'YES') -> bool:
        """Check if we can sell a specific contract type for this ticker"""
        current_position = portfolio.get_position(ticker)
        if not current_position:
            return False
        
        current_quantity = current_position.quantity
        
        if contract_type == 'YES':
            return current_quantity > 0  # Can sell YES if we own YES (positive)
        else:  # NO
            return current_quantity < 0  # Can sell NO if we own NO (negative)
    
    def _get_sellable_quantity(self, ticker: str, portfolio: Portfolio, contract_type: str = 'YES') -> int:
        """Get the maximum quantity we can sell for a specific contract type"""
        current_position = portfolio.get_position(ticker)
        if not current_position:
            return 0
        
        current_quantity = current_position.quantity
        
        if contract_type == 'YES':
            return max(0, current_quantity)  # Can sell up to positive quantity
        else:  # NO  
            return max(0, abs(current_quantity)) if current_quantity < 0 else 0

class MomentumStrategy(Strategy):
    """Momentum strategy with Kalshi contract awareness"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        strategy_config = STRATEGY_CONFIG.get('momentum', {})
        self.lookback_period = strategy_config.get('lookback_period', 10)
        self.momentum_threshold = strategy_config.get('momentum_threshold', 0.02)
        self.position_size = strategy_config.get('position_size', 5)  # Small size for Kalshi
    
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
            
            # Generate BUY signals based on momentum (always allowed)
            if momentum > self.momentum_threshold:
                # Strong upward momentum - prefer buying
                if current_quantity <= 0:  # No position or short NO position
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Momentum buy: {momentum:.3f} > {self.momentum_threshold}",
                        contract_type='YES'
                    ))
            
            elif momentum < -self.momentum_threshold:
                # Strong downward momentum - prefer buying NO
                if current_quantity >= 0:  # No position or long YES position
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,  # This will be converted appropriately
                        reason=f"Momentum buy NO: {momentum:.3f} < {-self.momentum_threshold}",
                        contract_type='NO'
                    ))
            
            # Generate SELL signals only if we own the contracts
            if momentum < -self.momentum_threshold:
                # Downward momentum - sell YES if we own them
                if self._can_sell_position(data.ticker, portfolio, 'YES'):
                    sellable_quantity = self._get_sellable_quantity(data.ticker, portfolio, 'YES')
                    if sellable_quantity > 0:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='sell',
                            quantity=min(sellable_quantity, self.position_size),
                            price=data.bid,
                            reason=f"Momentum sell YES: {momentum:.3f} < {-self.momentum_threshold}",
                            contract_type='YES'
                        ))
            
            elif momentum > self.momentum_threshold:
                # Upward momentum - sell NO if we own them
                if self._can_sell_position(data.ticker, portfolio, 'NO'):
                    sellable_quantity = self._get_sellable_quantity(data.ticker, portfolio, 'NO')
                    if sellable_quantity > 0:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='sell',
                            quantity=min(sellable_quantity, self.position_size),
                            price=data.bid,
                            reason=f"Momentum sell NO: {momentum:.3f} > {self.momentum_threshold}",
                            contract_type='NO'
                        ))
        
        return signals

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy with Kalshi contract awareness"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        strategy_config = STRATEGY_CONFIG.get('mean_reversion', {})
        self.lookback_period = strategy_config.get('lookback_period', 20)
        self.deviation_threshold = strategy_config.get('deviation_threshold', 2.0)
        self.position_size = strategy_config.get('position_size', 5)
    
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
            
            # Generate BUY signals based on mean reversion
            if deviation < -self.deviation_threshold:
                # Price significantly below mean - buy YES (expect reversion up)
                if current_quantity <= 0:
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Mean reversion buy YES: deviation {deviation:.2f}",
                        contract_type='YES'
                    ))
            
            elif deviation > self.deviation_threshold:
                # Price significantly above mean - buy NO (expect reversion down)
                if current_quantity >= 0:
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Mean reversion buy NO: deviation {deviation:.2f}",
                        contract_type='NO'
                    ))
            
            # Generate SELL signals only for owned contracts
            if deviation > self.deviation_threshold:
                # Above mean - sell YES if we own them
                if self._can_sell_position(data.ticker, portfolio, 'YES'):
                    sellable_quantity = self._get_sellable_quantity(data.ticker, portfolio, 'YES')
                    if sellable_quantity > 0:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='sell',
                            quantity=min(sellable_quantity, self.position_size),
                            price=data.bid,
                            reason=f"Mean reversion sell YES: deviation {deviation:.2f}",
                            contract_type='YES'
                        ))
            
            elif deviation < -self.deviation_threshold:
                # Below mean - sell NO if we own them
                if self._can_sell_position(data.ticker, portfolio, 'NO'):
                    sellable_quantity = self._get_sellable_quantity(data.ticker, portfolio, 'NO')
                    if sellable_quantity > 0:
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='sell',
                            quantity=min(sellable_quantity, self.position_size),
                            price=data.bid,
                            reason=f"Mean reversion sell NO: deviation {deviation:.2f}",
                            contract_type='NO'
                        ))
        
        return signals

class SpreadStrategy(Strategy):
    """Strategy that trades on spread compression/expansion with Kalshi awareness"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.max_spread_threshold = config.get('max_spread_threshold', 0.05)  # 5%
        self.min_spread_threshold = config.get('min_spread_threshold', 0.01)  # 1%
        self.position_size = config.get('position_size', 5)
    
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
            
            # Trade on tight spreads with momentum
            if spread < self.min_spread_threshold and abs(current_quantity) < self.position_size:
                recent_prices = self.get_price_history(data.ticker, 3)
                if len(recent_prices) >= 3:
                    if recent_prices[-1] > recent_prices[-2] > recent_prices[-3]:
                        # Upward trend with tight spread - buy YES
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='buy',
                            quantity=self.position_size - max(0, current_quantity),
                            price=data.ask,
                            reason=f"Tight spread momentum buy YES: spread {spread:.3f}",
                            contract_type='YES'
                        ))
                    elif recent_prices[-1] < recent_prices[-2] < recent_prices[-3]:
                        # Downward trend with tight spread - buy NO
                        signals.append(OrderSignal(
                            market_ticker=data.ticker,
                            side='buy',
                            quantity=self.position_size - max(0, abs(current_quantity)),
                            price=data.ask,
                            reason=f"Tight spread momentum buy NO: spread {spread:.3f}",
                            contract_type='NO'
                        ))
        
        return signals

class VolumeStrategy(Strategy):
    """Strategy that trades based on volume patterns with Kalshi awareness"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.volume_lookback = config.get('volume_lookback', 10)
        self.volume_threshold = config.get('volume_threshold', 2.0)  # 2x average volume
        self.position_size = config.get('position_size', 5)
    
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
            
            # High volume breakout strategy - only buy signals
            if volume_ratio > self.volume_threshold:
                if price_change > 0.01 and current_quantity <= 0:  # 1% price increase with high volume
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Volume breakout buy YES: vol_ratio {volume_ratio:.2f}, price_chg {price_change:.3f}",
                        contract_type='YES'
                    ))
                elif price_change < -0.01 and current_quantity >= 0:  # 1% price decrease with high volume
                    signals.append(OrderSignal(
                        market_ticker=data.ticker,
                        side='buy',
                        quantity=self.position_size,
                        price=data.ask,
                        reason=f"Volume breakdown buy NO: vol_ratio {volume_ratio:.2f}, price_chg {price_change:.3f}",
                        contract_type='NO'
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
    """Combines multiple strategies with Kalshi awareness"""
    
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
        
        # Combine signals for the same market and contract type
        combined_signals = self._combine_signals(all_signals)
        
        return combined_signals
    
    def _combine_signals(self, signals: List[OrderSignal]) -> List[OrderSignal]:
        """Combine multiple signals for the same market and contract type"""
        signal_groups = defaultdict(list)
        
        # Group signals by market and contract type
        for signal in signals:
            key = (signal.market_ticker, signal.side, getattr(signal, 'contract_type', 'YES'))
            signal_groups[key].append(signal)
        
        combined = []
        for (ticker, side, contract_type), ticker_signals in signal_groups.items():
            if len(ticker_signals) == 1:
                combined.append(ticker_signals[0])
            else:
                # Combine multiple signals
                total_quantity = sum(s.quantity for s in ticker_signals)
                avg_price = sum(s.price * s.quantity for s in ticker_signals) / total_quantity if total_quantity > 0 else 0
                
                combined_signal = OrderSignal(
                    market_ticker=ticker,
                    side=side,
                    quantity=total_quantity,
                    price=avg_price,
                    reason=f"Combined: {len(ticker_signals)} {contract_type} signals",
                    contract_type=contract_type
                )
                combined.append(combined_signal)
        
        return combined
