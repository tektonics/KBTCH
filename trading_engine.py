"""
Main trading engine that orchestrates all components
"""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging

from portfolio import Portfolio
from risk_manager import RiskManager
from strategy import StrategyFactory, Strategy, MarketData
from order_manager import OrderManager
from config import TRADING_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingEngine:
    """Main trading engine that coordinates all trading components"""
    
    def __init__(self, mode: str = 'simulation', kalshi_client=None, config: Dict = None):
        self.mode = mode
        self.config = config or TRADING_CONFIG
        
        # Initialize components
        self.portfolio = Portfolio(self.config['portfolio']['initial_cash'])
        self.risk_manager = RiskManager(self.config['risk_limits'])
        self.order_manager = OrderManager(mode, kalshi_client)
        
        # Initialize strategy
        strategy_name = self.config.get('strategy', 'momentum')
        self.strategy = StrategyFactory.create_strategy(strategy_name)
        
        # State tracking
        self.is_running = False
        self.last_risk_check = datetime.now()
        self.last_portfolio_save = datetime.now()
        self.emergency_shutdown = False
        
        # Performance tracking
        self.trade_count = 0
        self.start_time = datetime.now()
        self.daily_stats = {}
        
        logger.info(f"Trading engine initialized - Mode: {mode}, Strategy: {strategy_name}")
    
    def start(self):
        """Start the trading engine"""
        self.is_running = True
        self.start_time = datetime.now()
        
        # Load existing portfolio state if available
        try:
            self.portfolio.load_state(f'portfolio_{self.mode}.json')
            logger.info("Loaded existing portfolio state")
        except:
            logger.info("Starting with fresh portfolio")
        
        logger.info("Trading engine started")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        
        # Cancel all pending orders
        cancelled_count = self.order_manager.cancel_all_pending_orders()
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} pending orders")
        
        # Save portfolio state
        self.portfolio.save_state(f'portfolio_{self.mode}.json')
        
        # Log final statistics
        self._log_session_summary()
        
        logger.info("Trading engine stopped")
    
    def process_market_data(self, market_data: List[Dict]) -> Dict[str, Any]:
        """
        Main entry point called by kalshirunner.py with new market data
        
        Args:
            market_data: List of market data dictionaries with keys:
                        ticker, price, bid, ask, volume, timestamp, etc.
        
        Returns:
            Dictionary with processing results and statistics
        """
        if not self.is_running:
            return {'status': 'engine_stopped'}
        
        if self.emergency_shutdown:
            return {'status': 'emergency_shutdown'}
        
        try:
            # Convert to MarketData objects
            market_data_objs = []
            price_updates = {}
            
            for data in market_data:
                market_obj = MarketData(
                    ticker=data.get('ticker', ''),
                    price=data.get('price', 0.0),
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    volume=data.get('volume', 0),
                    timestamp=data.get('timestamp', time.time()),
                    open_interest=data.get('open_interest', 0)
                )
                market_data_objs.append(market_obj)
                price_updates[market_obj.ticker] = market_obj.price
            
            # Update portfolio with current prices
            self.portfolio.update_market_prices(price_updates)
            
            # Update pending orders
            self.order_manager.update_pending_orders()
            
            # Process fills and update portfolio
            fills = self.order_manager.get_fills()
            for fill in fills:
                if fill.filled_quantity > 0:
                    self.portfolio.add_trade(
                        fill.market_ticker,
                        fill.filled_quantity,
                        fill.filled_price,
                        fill.side
                    )
                    self.trade_count += 1
            
            # Periodic risk check
            if datetime.now() - self.last_risk_check > timedelta(minutes=5):
                self._perform_risk_check()
                self.last_risk_check = datetime.now()
            
            # Generate trading signals
            signals = self.strategy.generate_signals(market_data_objs, self.portfolio)
            
            results = {
                'status': 'success',
                'signals_generated': len(signals),
                'orders_executed': 0,
                'portfolio_value': self.portfolio.get_portfolio_value(),
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                'unrealized_pnl': self.portfolio.get_unrealized_pnl()
            }
            
            if signals:
                # Risk management validation
                validated_signals = self.risk_manager.validate_orders(signals, self.portfolio)
                
                if validated_signals:
                    # Execute orders
                    order_results = self.order_manager.execute_orders(validated_signals)
                    results['orders_executed'] = len(order_results)
                    results['order_results'] = [
                        {
                            'order_id': r.order_id,
                            'market': r.market_ticker,
                            'side': r.side,
                            'quantity': r.quantity,
                            'status': r.status
                        } for r in order_results
                    ]
            
            # Periodic portfolio save
            if datetime.now() - self.last_portfolio_save > timedelta(minutes=10):
                self.portfolio.save_state(f'portfolio_{self.mode}.json')
                self.last_portfolio_save = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _perform_risk_check(self):
        """Perform comprehensive risk assessment"""
        risk_metrics = self.risk_manager.check_portfolio_risk(self.portfolio)
        
        # Log risk warnings
        for warning in risk_metrics['risk_warnings']:
            logger.warning(f"Risk Warning: {warning}")
        
        # Check for emergency conditions
        if self.risk_manager.emergency_close_check(self.portfolio):
            logger.critical("EMERGENCY: Triggering position closure")
            self._emergency_close_positions()
    
    def _emergency_close_positions(self):
        """Emergency closure of all positions"""
        self.emergency_shutdown = True
        
        # Cancel all pending orders
        self.order_manager.cancel_all_pending_orders()
        
        # Create market orders to close all positions
        close_orders = []
        for ticker, position in self.portfolio.positions.items():
            if position.quantity != 0:
                # Create opposing order to close position
                side = 'sell' if position.quantity > 0 else 'buy'
                from risk_manager import OrderSignal
                close_order = OrderSignal(
                    market_ticker=ticker,
                    side=side,
                    quantity=abs(position.quantity),
                    price=position.current_price,
                    order_type='market',
                    reason='EMERGENCY_CLOSE'
                )
                close_orders.append(close_order)
        
        if close_orders:
            self.order_manager.execute_orders(close_orders)
            logger.critical(f"Emergency closure: {len(close_orders)} positions")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        runtime = datetime.now() - self.start_time
        
        portfolio_summary = self.portfolio.get_position_summary()
        order_summary = self.order_manager.get_order_summary()
        risk_metrics = self.risk_manager.check_portfolio_risk(self.portfolio)
        
        return {
            'engine_status': {
                'running': self.is_running,
                'mode': self.mode,
                'strategy': self.config.get('strategy'),
                'runtime_seconds': runtime.total_seconds(),
                'emergency_shutdown': self.emergency_shutdown
            },
            'portfolio': portfolio_summary,
            'orders': order_summary,
            'risk': risk_metrics,
            'performance': {
                'total_trades': self.trade_count,
                'trades_per_hour': self.trade_count / max(runtime.total_seconds() / 3600, 0.1),
                'start_time': self.start_time.isoformat()
            }
        }
    
    def force_trade(self, market_ticker: str, side: str, quantity: int, price: float) -> Dict:
        """Force a trade (for manual intervention)"""
        if not self.is_running:
            return {'status': 'error', 'message': 'Engine not running'}
        
        from risk_manager import OrderSignal
        order = OrderSignal(
            market_ticker=market_ticker,
            side=side,
            quantity=quantity,
            price=price,
            reason='MANUAL_OVERRIDE'
        )
        
        # Skip risk management for manual trades
        result = self.order_manager.execute_orders([order])
        
        return {
            'status': 'success',
            'order_result': {
                'order_id': result[0].order_id,
                'status': result[0].status,
                'filled_quantity': result[0].filled_quantity
            }
        }
    
    def update_strategy(self, strategy_name: str, config: Dict = None):
        """Update the trading strategy"""
        try:
            self.strategy = StrategyFactory.create_strategy(strategy_name, config)
            self.config['strategy'] = strategy_name
            logger.info(f"Strategy updated to: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy: {e}")
            return False
    
    def _log_session_summary(self):
        """Log summary of trading session"""
        runtime = datetime.now() - self.start_time
        portfolio_value = self.portfolio.get_portfolio_value()
        total_pnl = portfolio_value - self.portfolio.initial_cash
        
        logger.info("=== TRADING SESSION SUMMARY ===")
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Strategy: {self.config.get('strategy')}")
        logger.info(f"Total Trades: {self.trade_count}")
        logger.info(f"Final Portfolio Value: ${portfolio_value:.2f}")
        logger.info(f"Total P&L: ${total_pnl:.2f} ({total_pnl/self.portfolio.initial_cash*100:.2f}%)")
        logger.info(f"Final Cash: ${self.portfolio.cash:.2f}")
        logger.info(f"Open Positions: {len(self.portfolio.positions)}")
        logger.info("==============================")

class TradingEngineManager:
    """Manager for multiple trading engines or advanced features"""
    
    def __init__(self):
        self.engines: Dict[str, TradingEngine] = {}
    
    def create_engine(self, name: str, mode: str = 'simulation', config: Dict = None) -> TradingEngine:
        """Create a new trading engine instance"""
        engine = TradingEngine(mode=mode, config=config)
        self.engines[name] = engine
        return engine
    
    def get_engine(self, name: str) -> Optional[TradingEngine]:
        """Get existing engine by name"""
        return self.engines.get(name)
    
    def start_all(self):
        """Start all engines"""
        for engine in self.engines.values():
            engine.start()
    
    def stop_all(self):
        """Stop all engines"""
        for engine in self.engines.values():
            engine.stop()
    
    def get_combined_status(self) -> Dict:
        """Get status of all engines"""
        return {
            name: engine.get_status() 
            for name, engine in self.engines.items()
        }
