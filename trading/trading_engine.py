import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import json
import logging
from trading.portfolio import Portfolio
from trading.order_manager import OrderResult
from trading.risk_manager import RiskManager, OrderSignal
from config.config import TRADING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderExecutor(ABC):
    """Abstract base class for order execution"""
    
    @abstractmethod
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Execute a single order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get status of an existing order"""
        pass

class SimulatedOrderExecutor(OrderExecutor):
    """Enhanced simulated order execution with realistic Kalshi behavior"""
    
    def __init__(self):
        self.order_counter = 0
        self.orders: Dict[str, OrderResult] = {}
        self.fill_probability = 0.90  # 90% fill rate
        self.partial_fill_probability = 0.15  # 15% chance of partial fill
        self.price_slippage = 0.002  # 0.2% price slippage
        self.latency_simulation = True
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Simulate order execution with realistic Kalshi behavior"""
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        
        # Simulate network latency
        if self.latency_simulation:
            time.sleep(random.uniform(0.01, 0.05))
        
        # Determine contract type and price from order reason
        contract_type, kalshi_price = self._determine_kalshi_order_details(order)
        
        # Determine if order fills
        fill_random = random.random()
        
        if fill_random > self.fill_probability:
            # Order rejected or failed
            result = OrderResult(
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                price=order.price,
                filled_price=0.0,
                status='rejected',
                timestamp=datetime.now(),
                error_message='Market conditions unfavorable',
                contract_type=contract_type
            )
        else:
            # Determine fill quantity
            if random.random() < self.partial_fill_probability:
                # Partial fill
                filled_quantity = random.randint(max(1, order.quantity // 4), order.quantity - 1)
                status = 'partial'
            else:
                # Full fill
                filled_quantity = order.quantity
                status = 'filled'
            
            # Apply realistic price slippage for Kalshi
            slippage_direction = 1 if order.side == 'buy' else -1
            slippage_factor = 1 + (slippage_direction * self.price_slippage * random.random())
            filled_price = kalshi_price * slippage_factor
            
            # Ensure price stays within Kalshi bounds (0-1)
            filled_price = max(0.01, min(0.99, filled_price))
            
            result = OrderResult(
                order_id=order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=filled_quantity,
                price=order.price,
                filled_price=filled_price,
                status=status,
                timestamp=datetime.now(),
                contract_type=contract_type
            )
        
        self.orders[order_id] = result
        return result
    
    def _determine_kalshi_order_details(self, order: OrderSignal) -> tuple:
        """Determine Kalshi contract type and price from order details"""
        if 'YES' in order.reason:
            return 'YES', order.price
        elif 'NO' in order.reason:
            return 'NO', order.price
        else:
            # Default to YES contract
            return 'YES', order.price
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in ['pending', 'partial']:
                order.status = 'cancelled'
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get simulated order status"""
        return self.orders.get(order_id)

class KalshiOrderExecutor(OrderExecutor):
    """Live order execution using Kalshi API with enhanced error handling"""
    
    def __init__(self, kalshi_client):
        self.client = kalshi_client
        self.orders: Dict[str, OrderResult] = {}
        self.rate_limit_delay = TRADING_CONFIG['api']['rate_limit_delay']
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def execute_order(self, order: OrderSignal) -> OrderResult:
        """Execute order via Kalshi API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Determine contract type and prepare order
                contract_type, kalshi_order = self._prepare_kalshi_order(order)
                
                # Execute order via API
                response = self._execute_kalshi_api_call(kalshi_order)
                
                if response.get('status') == 'success':
                    order_data = response.get('order', {})
                    result = self._process_successful_order(order, order_data, contract_type)
                else:
                    result = self._process_failed_order(order, response, contract_type)
                
                if result.order_id:
                    self.orders[result.order_id] = result
                
                return result
                
            except Exception as e:
                logger.warning(f"Order execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return self._create_error_result(order, str(e))
                
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        return self._create_error_result(order, "Max retries exceeded")
    
    def _prepare_kalshi_order(self, order: OrderSignal) -> tuple:
        """Prepare Kalshi-specific order format"""
        # Determine if this is a YES or NO contract order
        contract_type = 'YES'  # Default
        if 'NO' in order.reason.upper():
            contract_type = 'NO'
        
        # Convert price to Kalshi cents format (0-100)
        kalshi_price = int(order.price * 100)
        
        kalshi_order = {
            'ticker': order.market_ticker,
            'client_order_id': f"auto_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            'side': order.side,
            'action': 'buy' if order.side == 'buy' else 'sell',
            'count': order.quantity,
            'type': 'limit',
        }
        
        # Set price based on contract type
        if contract_type == 'YES':
            if order.side == 'buy':
                kalshi_order['yes_price'] = kalshi_price
            else:
                kalshi_order['yes_price'] = kalshi_price
        else:  # NO contract
            if order.side == 'buy':
                kalshi_order['no_price'] = kalshi_price
            else:
                kalshi_order['no_price'] = kalshi_price
        
        return contract_type, kalshi_order
    
    def _execute_kalshi_api_call(self, kalshi_order: Dict) -> Dict:
        """Execute the actual Kalshi API call"""
        # This would be the actual Kalshi API call
        # For now, return a mock response structure
        return {
            'status': 'success',
            'order': {
                'order_id': f"KALSHI_{int(time.time() * 1000)}",
                'status': 'pending',
                'remaining_count': kalshi_order['count'],
                'yes_price': kalshi_order.get('yes_price', 0),
                'no_price': kalshi_order.get('no_price', 0)
            }
        }
    
    def _process_successful_order(self, order: OrderSignal, order_data: Dict, contract_type: str) -> OrderResult:
        """Process successful order response"""
        return OrderResult(
            order_id=order_data.get('order_id'),
            market_ticker=order.market_ticker,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,  # Will be updated when fills occur
            price=order.price,
            filled_price=0.0,  # Will be updated when fills occur
            status='pending',
            timestamp=datetime.now(),
            contract_type=contract_type
        )
    
    def _process_failed_order(self, order: OrderSignal, response: Dict, contract_type: str) -> OrderResult:
        """Process failed order response"""
        return OrderResult(
            order_id='',
            market_ticker=order.market_ticker,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,
            price=order.price,
            filled_price=0.0,
            status='rejected',
            timestamp=datetime.now(),
            error_message=response.get('error', 'Unknown error'),
            contract_type=contract_type
        )
    
    def _create_error_result(self, order: OrderSignal, error_message: str) -> OrderResult:
        """Create error result for failed orders"""
        return OrderResult(
            order_id='',
            market_ticker=order.market_ticker,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,
            price=order.price,
            filled_price=0.0,
            status='rejected',
            timestamp=datetime.now(),
            error_message=error_message,
            contract_type='YES'
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            # This would be the actual Kalshi cancel API call
            # response = self.client.cancel_order(order_id)
            # return response.get('status') == 'success'
            
            # For now, simulate cancellation
            if order_id in self.orders:
                self.orders[order_id].status = 'cancelled'
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get order status via Kalshi API"""
        try:
            time.sleep(self.rate_limit_delay)
            # This would be the actual Kalshi status API call
            # response = self.client.get_order(order_id)
            
            # For now, return stored order
            return self.orders.get(order_id)
            
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return self.orders.get(order_id)


class TradingEngine:
    
    def __init__(self, mode: str = 'simulation', kalshi_client=None, config: Dict = None):
        self.mode = mode
        self.config = config or TRADING_CONFIG
        
        # Initialize components
        self.portfolio = Portfolio(self.config['portfolio']['initial_cash'])
        # Create executor based on mode
        if mode == 'live':
            if kalshi_client is None:
                raise ValueError("kalshi_client required for live trading")
            self.executor = KalshiOrderExecutor(kalshi_client)
        else:
            self.executor = SimulatedOrderExecutor()
        
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

    def execute_order(self, order: OrderSignal) -> OrderResult:
        return self.executor.execute_order(order)

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        return self.executor.get_order_status(order_id)

    def cancel_order(self, order_id: str) -> bool:
        return self.executor.cancel_order(order_id)

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
            
            pass

            order_signals = self._convert_decisions_to_orders(trading_decisions)
            
            if order_signals:
                # Execute orders through order manager
                order_results = []
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
        
        cancelled_count = self.order_manager.cancel_all_pending_orders()
        
        close_orders = []
        for ticker, position in self.portfolio.positions.items():
            if position.quantity != 0:
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
