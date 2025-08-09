import sys
import asyncio
import ccxt.pro
import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from event_bus import event_bus, EventTypes
import logging
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_crypto.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookData:
    exchange_id: str
    symbol: str
    retrieval_timestamp: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    is_valid: bool = True
    error_reason: Optional[str] = None

class UnifiedCryptoManager:
    def __init__(self, output_file: str = "data/unified_crypto_data.json"):
        self.output_file = Path(output_file)
        self.data_lock = threading.Lock()
        
        # OHLCV components
        self.exchange_history = {}
        self.rsi_period = 14
        self.volume_baseline_periods = 20
        self.momentum_periods = 5
        
        # BRTI components  
        self.max_data_age_seconds = 30
        self.max_price_deviation_pct = 5.0
        self.max_spread_volume_deviation_pct = 0.5
        self.lambda_param = 1.0
        self.max_order_book_depth = 100
        self.order_size_cap_trim_pct = 0.01
        self.order_size_cap_multiplier = 5.0
        self.spacing_parameter = 1.0
        
        # Exchange configuration
        self.exchange_config = {
            'coinbase': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': 'BTC/USD'},
            'kraken': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': 'BTC/USD'},
            'bitstamp': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': 'BTC/USD'},
            'gemini': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': 'BTC/USD'},
            'cryptocom': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': 'BTC/USDT'},
            'okx': {'ohlcv_symbol': 'BTC/USD', 'brti_symbol': None}
        }
        
        # Statistics and display
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'exchange_failures': {ex: 0 for ex in self.exchange_config.keys()},
            'json_writes': 0,
            'json_write_errors': 0
        }
        
        self.display_line_count = 0
        self.last_price = None
        self.calculation_count = 0
        
        self.initialize_json_file()
    
    def initialize_json_file(self):
        try:
            initial_data = {
                "timestamp": time.time(),
                "last_updated": datetime.now().isoformat(),
                "brti": {
                    "price": None,
                    "utilized_depth": None,
                    "dynamic_cap": None,
                    "valid_exchanges": 0,
                    "total_exchanges": 0,
                    "exchange_status": {},
                    "status": "initializing"
                },
                "ohlcv_analysis": {
                    "volume_spikes": [],
                    "rsi": 50,
                    "momentum": "→",
                    "avg_price": 0,
                    "status": "initializing"
                }
            }
            with self.data_lock:
                with open(self.output_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
        except Exception as e:
            print(f"Failed to initialize JSON file: {e}")
    
    def clear_display(self):
        if self.display_line_count > 0:
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
    
    def update_single_line_display(self, line: str):
        self.clear_display()
        sys.stdout.write(line)
        sys.stdout.flush()
        self.display_line_count = 1
    
    def print_new_line(self, line: str):
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    # OHLCV Methods
    def update_exchange_data(self, exchange_id: str, symbol: str, ohlcv_data: list):
        if not ohlcv_data:
            return
            
        latest_candle = ohlcv_data[-1]
        timestamp, open_price, high, low, close, volume = latest_candle
        
        if exchange_id not in self.exchange_history:
            self.exchange_history[exchange_id] = {
                'prices': [],
                'volumes': [],
                'timestamps': []
            }
        
        history = self.exchange_history[exchange_id]
        
        history['prices'].append(close)
        history['volumes'].append(volume)
        history['timestamps'].append(timestamp)
        
        max_history = 50
        if len(history['prices']) > max_history:
            history['prices'] = history['prices'][-max_history:]
            history['volumes'] = history['volumes'][-max_history:]
            history['timestamps'] = history['timestamps'][-max_history:]
    
    def calculate_volume_spikes(self):
        exchange_codes = {
            'coinbase': 'CB', 'kraken': 'KR', 'bitstamp': 'BS', 
            'gemini': 'GM', 'cryptocom': 'CC', 'okx': 'OKX'
        }
        
        volume_spikes = []
        
        for exchange_id, history in self.exchange_history.items():
            volumes = history['volumes']
            if len(volumes) < self.volume_baseline_periods:
                continue
                
            recent_volume = volumes[-1]
            baseline_volume = sum(volumes[-self.volume_baseline_periods:-1]) / (self.volume_baseline_periods - 1)
            
            if baseline_volume > 0:
                spike_pct = ((recent_volume - baseline_volume) / baseline_volume) * 100
                if spike_pct > 50:
                    code = exchange_codes.get(exchange_id, exchange_id[:2].upper())
                    volume_spikes.append(f"{code}(+{spike_pct:.0f}%)")
        
        return volume_spikes
    
    def calculate_rsi(self):
        all_prices = []
        
        for exchange_id, history in self.exchange_history.items():
            for i, price in enumerate(history['prices']):
                timestamp = history['timestamps'][i]
                all_prices.append((timestamp, price))
        
        if len(all_prices) < self.rsi_period + 1:
            return 50
        
        all_prices.sort(key=lambda x: x[0])
        prices = [p[1] for p in all_prices]
        
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        if len(price_changes) < self.rsi_period:
            return 50
        
        recent_changes = price_changes[-self.rsi_period:]
        
        gains = [change for change in recent_changes if change > 0]
        losses = [-change for change in recent_changes if change < 0]
        
        avg_gain = sum(gains) / len(recent_changes)
        avg_loss = sum(losses) / len(recent_changes)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self):
        all_prices = []
        
        for exchange_id, history in self.exchange_history.items():
            for i, price in enumerate(history['prices']):
                timestamp = history['timestamps'][i]
                all_prices.append((timestamp, price))
        
        if len(all_prices) < self.momentum_periods:
            return "→"
        
        all_prices.sort(key=lambda x: x[0])
        prices = [p[1] for p in all_prices]
        
        recent_price = prices[-1]
        past_price = prices[-self.momentum_periods]
        momentum_pct = ((recent_price - past_price) / past_price) * 100
        
        if momentum_pct > 2:
            return "↑↑"
        elif momentum_pct > 0.5:
            return "↑"
        elif momentum_pct < -2:
            return "↓↓"
        elif momentum_pct < -0.5:
            return "↓"
        else:
            return "→"
    
    def calculate_average_price(self):
        current_prices = []
        
        for exchange_id, history in self.exchange_history.items():
            if history['prices']:
                current_prices.append(history['prices'][-1])
        
        if current_prices:
            return sum(current_prices) / len(current_prices)
        return 0
    
    # BRTI Methods
    def validate_order_book(self, ob: OrderBookData, current_time: int) -> bool:
        age_seconds = (current_time - ob.retrieval_timestamp) / 1000
        if age_seconds >= self.max_data_age_seconds:
            ob.is_valid = False
            ob.error_reason = f"Data too old ({age_seconds:.1f}s)"
            return False
        
        if not ob.bids or not ob.asks:
            ob.is_valid = False
            ob.error_reason = "Missing bids or asks"
            return False
        
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        if best_bid >= best_ask:
            ob.is_valid = False
            ob.error_reason = "Crossing order book"
            return False
        
        all_orders = ob.bids + ob.asks
        for order in all_orders:
            if not isinstance(order, (list, tuple)) or len(order) < 2:
                ob.is_valid = False
                ob.error_reason = "Invalid order format"
                return False
            
            price, size = order[0], order[1]
            if not (isinstance(price, (int, float)) and isinstance(size, (int, float))):
                ob.is_valid = False
                ob.error_reason = "Non-numeric price or size"
                return False
            if price <= 0 or size <= 0:
                ob.is_valid = False
                ob.error_reason = "Non-positive price or size"
                return False
        
        return True
    
    def check_price_deviation(self, order_books: List[OrderBookData]) -> List[OrderBookData]:
        if len(order_books) < 2:
            return order_books
        
        mid_prices = []
        for ob in order_books:
            if ob.is_valid and ob.bids and ob.asks:
                mid_price = (ob.bids[0][0] + ob.asks[0][0]) / 2
                mid_prices.append(mid_price)
        
        if len(mid_prices) < 2:
            return order_books
        
        median_mid = statistics.median(mid_prices)
        
        for ob in order_books:
            if ob.is_valid and ob.bids and ob.asks:
                mid_price = (ob.bids[0][0] + ob.asks[0][0]) / 2
                deviation_pct = abs(mid_price - median_mid) / median_mid * 100
                
                if deviation_pct > self.max_price_deviation_pct:
                    ob.is_valid = False
                    ob.error_reason = f"Price deviation {deviation_pct:.1f}% > {self.max_price_deviation_pct}%"
        
        return order_books
    
    def calculate_dynamic_order_size_cap(self, order_books: List[OrderBookData]) -> float:
        all_sizes = []
        
        for ob in order_books:
            if ob.is_valid:
                for _, size in ob.bids + ob.asks:
                    all_sizes.append(size)
        
        if not all_sizes:
            return float('inf')
        
        sizes_array = np.array(all_sizes)
        n_T = len(sizes_array)
        
        k = int(np.floor(self.order_size_cap_trim_pct * n_T))
        
        if k == 0 or n_T <= 2 * k:
            trimmed_sizes = sizes_array
        else:
            sorted_sizes = np.sort(sizes_array)
            trimmed_sizes = sorted_sizes[k:n_T-k]
        
        if len(trimmed_sizes) == 0:
            return float('inf')
        
        trimmed_mean = np.mean(trimmed_sizes)
        
        if len(trimmed_sizes) <= 1:
            winsorized_std = 0.0
        else:
            winsorized_std = np.std(trimmed_sizes, ddof=1)
        
        dynamic_cap = trimmed_mean + (self.order_size_cap_multiplier * winsorized_std)
        
        return dynamic_cap
    
    def apply_order_size_cap(self, orders: List[Tuple[float, float]], cap: float) -> List[Tuple[float, float]]:
        if cap == float('inf'):
            return orders
        
        capped_orders = []
        for price, size in orders:
            capped_size = min(size, cap)
            capped_orders.append((price, capped_size))
        
        return capped_orders
    
    def consolidate_order_books(self, order_books: List[OrderBookData]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        dynamic_cap = self.calculate_dynamic_order_size_cap(order_books)
        
        all_bids = []
        all_asks = []
        
        for ob in order_books:
            if ob.is_valid:
                capped_bids = self.apply_order_size_cap(ob.bids, dynamic_cap)
                capped_asks = self.apply_order_size_cap(ob.asks, dynamic_cap)
                
                all_bids.extend(capped_bids)
                all_asks.extend(capped_asks)
        
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])
        
        consolidated_bids = self._aggregate_price_levels(all_bids)
        consolidated_asks = self._aggregate_price_levels(all_asks)
        
        return consolidated_bids, consolidated_asks, dynamic_cap
    
    def _aggregate_price_levels(self, orders: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        price_volumes = {}
        for price, volume in orders:
            if price in price_volumes:
                price_volumes[price] += volume
            else:
                price_volumes[price] = volume
        
        return [(price, volume) for price, volume in price_volumes.items()]
    
    def calculate_curves(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
        curves = {
            'bid_curve': [],
            'ask_curve': [],
            'mid_curve': [],
            'mid_spread_volume_curve': []
        }
        
        bid_cum = []
        ask_cum = []
        cum_bid_vol = 0.0
        cum_ask_vol = 0.0

        for price, volume in bids:
            cum_bid_vol += volume
            bid_cum.append((cum_bid_vol, price))

        for price, volume in asks:
           cum_ask_vol += volume
           ask_cum.append((cum_ask_vol, price))

        max_volume = min(cum_bid_vol, cum_ask_vol)
        volume_steps = np.arange(self.spacing_parameter, max_volume, self.spacing_parameter)

        for v_step in volume_steps:
            bid_price = next((p for cv, p in bid_cum if cv >= v_step), None)
            ask_price = next((p for cv, p in ask_cum if cv >= v_step), None)

            if bid_price is None or ask_price is None:
                continue

            mid_price = (bid_price + ask_price) / 2
            
            curves['bid_curve'].append((v_step, bid_price))
            curves['ask_curve'].append((v_step, ask_price))
            curves['mid_curve'].append((v_step, mid_price))
            
            if mid_price > 0:
                mid_spread_pct = (ask_price / mid_price) - 1
                curves['mid_spread_volume_curve'].append((v_step, mid_spread_pct))

        return curves
    
    def calculate_utilized_depth(self, curves: Dict[str, List[Tuple[float, float]]]) -> float:
        if not curves['mid_spread_volume_curve']:
            return self.spacing_parameter
        
        utilized_volume = self.spacing_parameter
        
        for v_step, spread_pct in curves['mid_spread_volume_curve']:
            if spread_pct <= self.max_spread_volume_deviation_pct:
                utilized_volume = v_step
            else:
                break
        
        final_utilized_depth = max(utilized_volume, self.spacing_parameter)
        
        return final_utilized_depth
    
    def apply_exponential_weighting(self, curves: Dict[str, List[Tuple[float, float]]], utilized_depth: float) -> float:
        if not curves['mid_curve'] or utilized_depth == 0:
            return 0.0

        weighted_sum = 0.0
        normalization_factor_sum = 0.0
        
        lambda_param = self.lambda_param
        spacing_s = self.spacing_parameter

        relevant_mid_curve_entries = [
            (v_step, mid_price) for v_step, mid_price in curves['mid_curve'] 
            if v_step > 0 and v_step <= utilized_depth
        ]
        
        if not relevant_mid_curve_entries:
            return 0.0

        for v_step, _ in relevant_mid_curve_entries:
            normalization_factor_sum += lambda_param * np.exp(-lambda_param * v_step)
        
        if normalization_factor_sum == 0:
            return 0.0

        for v_step, mid_price_at_v in relevant_mid_curve_entries:
            weight_term = (1 / normalization_factor_sum) * lambda_param * np.exp(-lambda_param * v_step)
            weighted_sum += mid_price_at_v * weight_term
            
        return weighted_sum
    
    def format_single_line_display(self, order_books: List[OrderBookData], brti_value: float, 
                                  utilized_depth: float, dynamic_cap: float, ohlcv_data: dict) -> str:
        self.calculation_count += 1
        
        current_time = datetime.now().strftime("%H:%M:%S")
        
        valid_exchanges = [ob.exchange_id for ob in order_books if ob.is_valid]
        invalid_exchanges = [ob.exchange_id for ob in order_books if not ob.is_valid]
        
        price_indicator = ""
        if self.last_price is not None:
            if brti_value > self.last_price:
                price_indicator = "🔼"
            elif brti_value < self.last_price:
                price_indicator = "🔽"
            else:
                price_indicator = "━"
        self.last_price = brti_value
        
        exchange_codes = {
            'coinbase': 'CB', 'kraken': 'KR', 'bitstamp': 'BS', 
            'gemini': 'GM', 'cryptocom': 'CC'
        }
        
        valid_codes = [exchange_codes.get(ex, ex[:2].upper()) for ex in valid_exchanges]
        invalid_codes = [exchange_codes.get(ex, ex[:2].upper()) for ex in invalid_exchanges]
        
        # Build the display line matching original BRTI format
        parts = [
            f"{current_time}",
            f"#{self.calculation_count:04d}",
            f"BRTI: ${brti_value:,.2f} {price_indicator}",
            f"Depth: {utilized_depth:.1f}",
            f"Cap: {dynamic_cap:.1f}" if dynamic_cap != float('inf') else "Cap: ∞",
            f"Valid: {len(valid_exchanges)}/5 [{','.join(valid_codes)}]"
        ]
        
        if invalid_codes:
            parts.append(f"❌[{','.join(invalid_codes)}]")
        
        # Add OHLCV data
        parts.append(f"RSI: {ohlcv_data['rsi']:.0f}")
        parts.append(f"{ohlcv_data['momentum']}")
        
        if ohlcv_data['volume_spikes']:
            parts.append(f"Vol: {','.join(ohlcv_data['volume_spikes'])}")
        
        return " | ".join(parts)
    
    def write_unified_data_to_json(self, brti_value: Optional[float], brti_data: Dict[str, Any], ohlcv_data: Dict[str, Any]):
        try:
            current_time = time.time()
            timestamp_iso = datetime.fromtimestamp(current_time).isoformat()
            
            data = {
                "timestamp": current_time,
                "last_updated": timestamp_iso,
                "brti": {
                    "price": round(brti_value, 2) if brti_value else None,
                    "utilized_depth": round(brti_data.get('utilized_depth', 0), 2) if brti_value else None,
                    "dynamic_cap": round(brti_data.get('dynamic_cap', 0), 2) if brti_value else None,
                    "valid_exchanges": brti_data.get('valid_exchanges', 0),
                    "total_exchanges": brti_data.get('total_exchanges', 0),
                    "exchange_status": brti_data.get('exchange_status', {}),
                    "status": "active" if brti_value else "error"
                },
                "ohlcv_analysis": {
                    "volume_spikes": ohlcv_data['volume_spikes'],
                    "rsi": round(ohlcv_data['rsi'], 0),
                    "momentum": ohlcv_data['momentum'],
                    "avg_price": round(ohlcv_data['avg_price'], 2),
                    "status": "active"
                }
            }
            
            with self.data_lock:
                with open(self.output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            self.stats['json_writes'] += 1
            
        except Exception as e:
            logger.error(f"Failed to write unified data to JSON: {e}")
            self.stats['json_write_errors'] += 1
    
    async def fetch_ohlcv_continuously(self, exchange, symbol):
        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol)
                if ohlcv:
                    self.update_exchange_data(exchange.id, symbol, ohlcv)
            except Exception as e:
                logger.debug(f"OHLCV fetch error for {exchange.id}: {e}")
                break
    
    async def fetch_order_book_data(self, exchange, exchange_id: str) -> Optional[OrderBookData]:
        try:
            symbol = self.exchange_config[exchange_id]['brti_symbol']
            if not symbol:  # Skip exchanges without BRTI symbol (like OKX)
                return None
                
            order_book = await exchange.fetch_order_book(symbol, limit=self.max_order_book_depth)
            
            bids = []
            asks = []
            
            for bid in order_book['bids']:
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    price, size = float(bid[0]), float(bid[1])
                    if price > 0 and size > 0:
                        bids.append((price, size))
            
            for ask in order_book['asks']:
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    price, size = float(ask[0]), float(ask[1])
                    if price > 0 and size > 0:
                        asks.append((price, size))
            
            return OrderBookData(
                exchange_id=exchange_id,
                symbol=symbol,
                retrieval_timestamp=int(time.time() * 1000),
                bids=bids,
                asks=asks
            )
        except Exception as e:
            logger.warning(f"Failed to fetch order book from {exchange_id}: {e}")
            self.stats['exchange_failures'][exchange_id] += 1
            return None
    
    async def calculate_brti_with_exchanges(self, exchanges):
        try:
            current_time = int(time.time() * 1000)

            # Fetch all order books
            order_book_tasks = []
            for exchange_id, exchange in exchanges.items():
                if self.exchange_config[exchange_id]['brti_symbol']: # Only for BRTI-enabled exchanges
                    task = self.fetch_order_book_data(exchange, exchange_id)
                    order_book_tasks.append(task)
            order_books = []
            if order_book_tasks:
                results = await asyncio.gather(*order_book_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, OrderBookData):
                        order_books.append(result)

            if not order_books:
                return None, {}

            # Validate order books
            for ob in order_books:
                self.validate_order_book(ob, current_time)

            # Check price deviation
            order_books = self.check_price_deviation(order_books)

            # Filter valid order books
            valid_order_books = [ob for ob in order_books if ob.is_valid]

            if len(valid_order_books) < 2:
                brti_data = {
                    'valid_exchanges': len(valid_order_books),
                    'total_exchanges': len(order_books),
                    'exchange_status': {ob.exchange_id: "valid" if ob.is_valid else ob.error_reason for ob in order_books}
                }
                return None, brti_data

            # Consolidate order books
            # CHANGE: Capture dynamic_cap returned by consolidate_order_books
            consolidated_bids, consolidated_asks, dynamic_cap = self.consolidate_order_books(valid_order_books)

            if not consolidated_bids or not consolidated_asks:
                brti_data = {
                    'valid_exchanges': len(valid_order_books),
                    'total_exchanges': len(order_books),
                    'exchange_status': {ob.exchange_id: "valid" if ob.is_valid else ob.error_reason for ob in order_books}
                }
                return None, brti_data

            # Calculate curves
            curves = self.calculate_curves(consolidated_bids, consolidated_asks)

            # Calculate utilized depth
            utilized_depth = self.calculate_utilized_depth(curves)

            # Apply exponential weighting
            brti_value = self.apply_exponential_weighting(curves, utilized_depth)

            brti_data = {
                'utilized_depth': utilized_depth,
                'dynamic_cap': dynamic_cap,
                'valid_exchanges': len(valid_order_books),
                'total_exchanges': len(order_books),
                'exchange_status': {ob.exchange_id: "valid" if ob.is_valid else ob.error_reason for ob in order_books}
            }
            self.stats['successful_calculations'] += 1
            return brti_value, brti_data

        except Exception as e:
            logger.error(f"Error calculating BRTI: {e}")
            return None, {}
        finally:
            self.stats['total_calculations'] += 1

    
    async def start_unified_exchange(self, exchange_name):
        exchange = getattr(ccxt.pro, exchange_name)({})
        
        # Start OHLCV fetching
        ohlcv_symbol = self.exchange_config[exchange_name]['ohlcv_symbol']
        ohlcv_task = asyncio.create_task(self.fetch_ohlcv_continuously(exchange, ohlcv_symbol))
        
        return exchange, ohlcv_task
    
    async def run_unified_system(self):
        self.print_new_line("Starting Unified Crypto Data Manager...")
        self.print_new_line(f"JSON output: {self.output_file.absolute()}")
        self.print_new_line("=" * 100)
        
        # Initialize all exchanges
        exchanges = {}
        ohlcv_tasks = []
        
        for exchange_name in self.exchange_config.keys():
            try:
                exchange, ohlcv_task = await self.start_unified_exchange(exchange_name)
                exchanges[exchange_name] = exchange
                ohlcv_tasks.append(ohlcv_task)
                logger.info(f"Initialized {exchange_name} successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")
        
        try:
            while True:
                calculation_start = time.time()
                
                # Calculate OHLCV metrics
                volume_spikes = self.calculate_volume_spikes()
                rsi = self.calculate_rsi()
                momentum = self.calculate_momentum()
                avg_price = self.calculate_average_price()
                
                ohlcv_data = {
                    'volume_spikes': volume_spikes,
                    'rsi': rsi,
                    'momentum': momentum,
                    'avg_price': avg_price
                }
                
                # Calculate BRTI
                brti_value, brti_data = await self.calculate_brti_with_exchanges(exchanges)
                
               # Publish BRTI update to event bus
                if brti_value is not None:
                    event_bus.publish(
                        EventTypes.PRICE_UPDATE,
                        {
                            "brti_price": brti_value,
                            "utilized_depth": brti_data.get('utilized_depth'),
                            "dynamic_cap": brti_data.get('dynamic_cap'), 
                            "valid_exchanges": brti_data.get('valid_exchanges'),
                            "volume_spikes": ohlcv_data['volume_spikes'],
                            "rsi": ohlcv_data['rsi'],
                            "momentum": ohlcv_data['momentum'],
                            "avg_price": ohlcv_data['avg_price']
                        },
                        source="udm"
                )

                # Display single line update
                if brti_value is not None:
                    # Create mock order books for display formatting
                    mock_order_books = []
                    for ex_id in exchanges.keys():
                        if self.exchange_config[ex_id]['brti_symbol']:  # Only BRTI exchanges
                            is_valid = brti_data['exchange_status'].get(ex_id) == "valid"
                            error_reason = None if is_valid else brti_data['exchange_status'].get(ex_id)
                            mock_order_books.append(OrderBookData(ex_id, "", 0, [], [], is_valid, error_reason))
                    
                    display_line = self.format_single_line_display(
                        mock_order_books, 
                        brti_value, 
                        brti_data.get('utilized_depth', 0), 
                        brti_data.get('dynamic_cap', 0), 
                        ohlcv_data
                    )
                    self.update_single_line_display(display_line)
                else:
                    error_line = f"{datetime.now().strftime('%H:%M:%S')} | #{self.calculation_count + 1:04d} | ERROR: BRTI calculation failed"
                    self.update_single_line_display(error_line)
                
                # Write unified data to JSON
                self.write_unified_data_to_json(brti_value, brti_data, ohlcv_data)
                
                # Sleep to maintain interval
                calculation_time = time.time() - calculation_start
                sleep_time = max(0, .2 - calculation_time)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.print_new_line("\nUnified system stopped by user")
        except Exception as e:
            self.print_new_line(f"\nUnexpected error: {e}")
        finally:
            # Cancel OHLCV tasks
            for task in ohlcv_tasks:
                task.cancel()
            
            # Close all exchanges
            for exchange in exchanges.values():
                await exchange.close()
            
            self.print_final_stats()
    
    def print_final_stats(self):
        self.print_new_line("\n" + "=" * 60)
        self.print_new_line("FINAL STATISTICS")
        self.print_new_line("=" * 60)
        print(f"Total calculations: {self.stats['total_calculations']}")
        print(f"Successful calculations: {self.stats['successful_calculations']}")
        print(f"JSON writes: {self.stats['json_writes']}")
        print(f"JSON write errors: {self.stats['json_write_errors']}")
        
        if self.stats['total_calculations'] > 0:
            success_rate = (self.stats['successful_calculations'] / 
                          self.stats['total_calculations'] * 100)
            print(f"Success rate: {success_rate:.1f}%")
        
        print("\nExchange failure counts:")
        for exchange, failures in self.stats['exchange_failures'].items():
            print(f"  {exchange}: {failures}")

async def main():
    """Main execution function"""
    manager = UnifiedCryptoManager()
    await manager.run_unified_system()

if __name__ == "__main__":
    asyncio.run(main())
