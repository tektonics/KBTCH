# trading_engine.py - Orchestrates trade execution and portfolio management
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging

from portfolio import Portfolio
from order_manager import OrderManager, OrderResult
from risk_manager import RiskManager, OrderSignal
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
    """
    Trading engine that receives decisions from trading_logic.py and executes them
    
    This class orchestrates:
    1. Receiving trading decisions from trading_logic.py
    2. Converting decisions to orders via order_manager.py
    3. Updating portfolio state
    4. Providing status and performance tracking
    """
    
    def __init__(self, mode: str = 'simulation', kalshi_client=None, config: Dict = None):
        self.mode = mode
        self.config = config or TRADING_CONFIG
        
        # Initialize components
        self.portfolio = Portfolio(self.config['portfolio']['initial_cash'])
        self.order_manager = OrderManager(mode, kalshi_client)
        
        # State tracking
        self.is_running = False
        self.last_portfolio_save = datetime.now()
        self.emergency_shutdown = False
        
        # Performance tracking
        self.trade_count = 0
        self.start_time = datetime.now()
        self.session_stats = {
            'decisions_processed': 0,
            'orders_executed': 0,
            'successful_fills': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_pnl': 0.0
        }
        
        logger.info(f"Trading engine initialized - Mode: {mode}")
    
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
    
    def process_trading_decisions(self, trading_decisions: List) -> List[Dict[str, Any]]:
        """
        Main entry point - receives trading decisions from trading_logic.py
        
        Args:
            trading_decisions: List of TradingDecision objects from trading_logic.py
        
        Returns:
            List of execution results
        """
        if not self.is_running:
            return [{'status': 'engine_stopped', 'executed': False}]
        
        if self.emergency_shutdown:
            return [{'status': 'emergency_shutdown', 'executed': False}]
        
        results = []
        
        try:
            self.session_stats['decisions_processed'] += len(trading_decisions)
            
            # Update pending orders and process fills
            self._update_pending_orders_and_fills()
            
            # Convert trading decisions to order signals
            order_signals = self._convert_decisions_to_orders(trading_decisions)
            
            if order_signals:
                # Execute orders through order manager
                order_results = self.order_manager.execute_orders(order_signals)
                self.session_stats['orders_executed'] += len(order_results)
                
                # Process results
                for decision, order_result in zip(trading_decisions, order_results):
                    result = self._process_order_result(decision, order_result)
                    results.append(result)
            
            # Periodic portfolio save
            if datetime.now() - self.last_portfolio_save > timedelta(minutes=10):
                self.portfolio.save_state(f'portfolio_{self.mode}.json')
                self.last_portfolio_save = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing trading decisions: {e}")
            return [{'status': 'error', 'message': str(e), 'executed': False}]
    
    def _convert_decisions_to_orders(self, trading_decisions: List) -> List[OrderSignal]:
        """Convert TradingDecision objects to OrderSignal objects"""
        order_signals = []
        
        for decision in trading_decisions:
            action = getattr(decision, 'action', '')
            
            if action in ['BUY_YES', 'BUY_NO']:
                side = 'buy'
                # For Kalshi, we need to determine the actual price based on YES/NO
                if action == 'BUY_YES':
                    price = getattr(decision, 'price', 0.0)
                else:  # BUY_NO
                    price = 100 - getattr(decision, 'price', 0.0)  # NO price is inverse of YES price
            
            elif action in ['SELL_YES', 'SELL_NO']:
                side = 'sell'
                if action == 'SELL_YES':
                    price = getattr(decision, 'price', 0.0)
                else:  # SELL_NO
                    price = 100 - getattr(decision, 'price', 0.0)
            
            else:
                continue  # Skip HOLD, NO_TRADE, etc.
            
            order_signal = OrderSignal(
                market_ticker=getattr(decision, 'ticker', ''),
                side=side,
                quantity=getattr(decision, 'quantity', 0),
                price=price / 100 if price > 1 else price,  # Convert to 0-1 range for Kalshi if needed
                order_type='limit',
                reason=f"{action}: {getattr(decision, 'reason', '')}"
            )
            order_signals.append(order_signal)
        
        return order_signals
    
    def _update_pending_orders_and_fills(self):
        """Update pending orders and process any new fills"""
        # Update order statuses
        self.order_manager.update_pending_orders()
        
        # Get new fills and update portfolio
        fills = self.order_manager.get_fills()
        new_fills = [fill for fill in fills if not hasattr(fill, '_processed')]
        
        for fill in new_fills:
            if fill.filled_quantity > 0:
                # Update portfolio with the fill
                self.portfolio.add_trade(
                    fill.market_ticker,
                    fill.filled_quantity,
                    fill.filled_price,
                    fill.side
                )
                
                self.trade_count += 1
                self.session_stats['successful_fills'] += 1
                self.session_stats['total_volume'] += fill.filled_quantity * fill.filled_price
                
                # Mark as processed
                fill._processed = True
                
                logger.info(f"Fill processed: {fill.side} {fill.filled_quantity} {fill.market_ticker} @ {fill.filled_price:.3f}")
    
    def _process_order_result(self, decision, order_result: OrderResult) -> Dict[str, Any]:
        """Process the result of an order execution"""
        result = {
            'decision': {
                'ticker': getattr(decision, 'ticker', ''),
                'action': getattr(decision, 'action', ''),
                'quantity': getattr(decision, 'quantity', 0),
                'confidence': getattr(decision, 'confidence', 0.0),
                'edge': getattr(decision, 'edge', 0.0),
                'reason': getattr(decision, 'reason', '')
            },
            'order_result': {
                'order_id': order_result.order_id,
                'status': order_result.status,
                'filled_quantity': order_result.filled_quantity,
                'filled_price': order_result.filled_price,
                'error_message': order_result.error_message
            },
            'executed': order_result.status in ['filled', 'partial', 'pending'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Update session stats
        if order_result.status == 'rejected':
            self.session_stats['rejected_orders'] += 1
        
        return result
    
    def update_market_prices(self, price_data: Dict[str, float]):
        """Update portfolio with current market prices"""
        self.portfolio.update_market_prices(price_data)
        
        # Update total P&L
        self.session_stats['total_pnl'] = self.portfolio.get_portfolio_value() - self.portfolio.initial_cash
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        runtime = datetime.now() - self.start_time
        
        portfolio_summary = self.portfolio.get_position_summary()
        order_summary = self.order_manager.get_order_summary()
        
        return {
            'running': self.is_running,
            'mode': self.mode,
            'runtime_seconds': runtime.total_seconds(),
            'emergency_shutdown': self.emergency_shutdown,
            'portfolio': portfolio_summary,
            'orders': order_summary,
            'session_stats': self.session_stats,
            'performance': {
                'total_trades': self.trade_count,
                'trades_per_hour': self.trade_count / max(runtime.total_seconds() / 3600, 0.1),
                'start_time': self.start_time.isoformat(),
                'success_rate': (self.session_stats['successful_fills'] / 
                               max(self.session_stats['orders_executed'], 1) * 100)
            }
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return self.portfolio.get_position_summary()
    
    def force_trade(self, market_ticker: str, side: str, quantity: int, price: float) -> Dict:
        """Force a trade (for manual intervention)"""
        if not self.is_running:
            return {'status': 'error', 'message': 'Engine not running'}
        
        order = OrderSignal(
            market_ticker=market_ticker,
            side=side,
            quantity=quantity,
            price=price,
            reason='MANUAL_OVERRIDE'
        )
        
        # Execute without risk management for manual trades
        result = self.order_manager.execute_orders([order])
        
        return {
            'status': 'success',
            'order_result': {
                'order_id': result[0].order_id,
                'status': result[0].status,
                'filled_quantity': result[0].filled_quantity
            }
        }
    
    def emergency_close_all_positions(self) -> Dict[str, Any]:
        """Emergency closure of all positions"""
        if not self.is_running:
            return {'status': 'error', 'message': 'Engine not running'}
        
        self.emergency_shutdown = True
        
        # Cancel all pending orders
        cancelled_count = self.order_manager.cancel_all_pending_orders()
        
        # Create orders to close all positions
        close_orders = []
        for ticker, position in self.portfolio.positions.items():
            if position.quantity != 0:
                # Create opposing order to close position
                side = 'sell' if position.quantity > 0 else 'buy'
                close_order = OrderSignal(
                    market_ticker=ticker,
                    side=side,
                    quantity=abs(position.quantity),
                    price=position.current_price,
                    order_type='market',
                    reason='EMERGENCY_CLOSE'
                )
                close_orders.append(close_order)
        
        results = []
        if close_orders:
            results = self.order_manager.execute_orders(close_orders)
        
        logger.critical(f"Emergency closure: cancelled {cancelled_count} orders, closing {len(close_orders)} positions")
        
        return {
            'status': 'emergency_close_initiated',
            'cancelled_orders': cancelled_count,
            'close_orders': len(close_orders),
            'results': [{'order_id': r.order_id, 'status': r.status} for r in results]
        }
    
    def _log_session_summary(self):
        """Log summary of trading session"""
        runtime = datetime.now() - self.start_time
        portfolio_value = self.portfolio.get_portfolio_value()
        total_pnl = portfolio_value - self.portfolio.initial_cash
        
        logger.info("=== TRADING SESSION SUMMARY ===")
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Decisions Processed: {self.session_stats['decisions_processed']}")
        logger.info(f"Orders Executed: {self.session_stats['orders_executed']}")
        logger.info(f"Successful Fills: {self.session_stats['successful_fills']}")
        logger.info(f"Rejected Orders: {self.session_stats['rejected_orders']}")
        logger.info(f"Total Volume: ${self.session_stats['total_volume']:.2f}")
        logger.info(f"Final Portfolio Value: ${portfolio_value:.2f}")
        logger.info(f"Total P&L: ${total_pnl:.2f} ({total_pnl/self.portfolio.initial_cash*100:.2f}%)")
        logger.info(f"Final Cash: ${self.portfolio.cash:.2f}")
        logger.info(f"Open Positions: {len(self.portfolio.positions)}")
        logger.info("==============================")

class TradingEngineManager:
    """Manager for multiple trading engines or advanced features"""
    
    def __init__(self):
        self.engines: Dict[str, TradingEngine] = {}
    
    def create_engine(self, name: str, mode: str = 'simulation', 
                     kalshi_client=None, config: Dict = None) -> TradingEngine:
        """Create a new trading engine instance"""
        engine = TradingEngine(mode=mode, kalshi_client=kalshi_client, config=config)
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
