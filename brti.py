import ccxt
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import statistics
import threading
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brti.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookData:
    """Structured order book data with metadata"""
    exchange_id: str
    symbol: str
    retrieval_timestamp: int
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    is_valid: bool = True
    error_reason: Optional[str] = None

@dataclass
class BRTIConfig:
    """Configuration parameters for BRTI calculation"""
    # Data quality parameters
    max_data_age_seconds: int = 30
    max_price_deviation_pct: float = 5.0
    max_spread_volume_deviation_pct: float = 0.5
    
    # Exponential weighting parameter
    lambda_param: float = 1.0
    
    # Order book depth limits
    max_order_book_depth: int = 100
    
    # Dynamic order size cap parameters - CF Benchmarks compliant
    order_size_cap_trim_pct: float = 0.01  # 1% trimming from each side (CF spec)
    order_size_cap_multiplier: float = 5.0  # CF formula: C_T = s + 5 * sigma
    
    # Spacing parameter
    spacing_parameter: float = 1.0  # s = 1 for Bitcoin BRTI
    
    # Timing
    calculation_interval_seconds: float = 1.0
    
    # JSON output configuration
    json_output_file: str = "aggregate_price.json"

class OptimizedBRTI:
    
    def __init__(self, config: BRTIConfig = None):
        self.config = config or BRTIConfig()
        
        self.exchange_config = {
            'coinbase': {'symbol': 'BTC/USD', 'weight': 1.0},
            'kraken': {'symbol': 'BTC/USD', 'weight': 1.0},
            'bitstamp': {'symbol': 'BTC/USD', 'weight': 1.0},
            'gemini': {'symbol': 'BTC/USD', 'weight': 1.0},
            'cryptocom': {'symbol': 'BTC/USDT', 'weight': 1.0}
        }
        
        self.exchanges = {}
        self.initialize_exchanges()
        
        # Statistics tracking
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'exchange_failures': {ex: 0 for ex in self.exchange_config.keys()},
            'json_writes': 0,
            'json_write_errors': 0
        }
        
        # JSON output configuration
        self.json_output_path = Path(self.config.json_output_file)
        self.json_write_lock = threading.Lock()
        
        # Display management
        self.display_line_count = 0
        self.last_price = None
        self.calculation_count = 0
        
        # Initialize JSON file
        self.initialize_json_file()
    
    def clear_display(self):
        """Clear the current display line"""
        if self.display_line_count > 0:
            sys.stdout.write('\r\033[K')  # Clear current line
            sys.stdout.flush()
    
    def update_single_line_display(self, line: str):
        """Update display with a single line, clearing the previous one"""
        self.clear_display()
        sys.stdout.write(line)
        sys.stdout.flush()
        self.display_line_count = 1
    
    def print_new_line(self, line: str):
        """Print a new line (clearing current display first)"""
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    def initialize_json_file(self):
        """Initialize the JSON output file with default structure"""
        try:
            initial_data = {
                "price": None,
                "timestamp": None,
                "last_updated": None,
                "source": "BRTI",
                "status": "initializing"
            }
            
            with self.json_write_lock:
                with open(self.json_output_path, 'w') as f:
                    json.dump(initial_data, f, indent=2)
            
            logger.info(f"Initialized JSON output file: {self.json_output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize JSON file: {e}")
    
    def write_price_to_json(self, price: float, additional_data: Dict[str, Any] = None):
        """Thread-safe method to write price data to JSON file"""
        try:
            current_time = time.time()
            timestamp_iso = datetime.fromtimestamp(current_time).isoformat()
            
            # Prepare data structure
            data = {
                "price": round(price, 2),
                "timestamp": current_time,
                "last_updated": timestamp_iso,
                "source": "BRTI",
                "status": "active"
            }
            
            # Add any additional data
            if additional_data:
                data.update(additional_data)
            
            # Thread-safe write
            with self.json_write_lock:
                with open(self.json_output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            self.stats['json_writes'] += 1
            logger.debug(f"Updated price in JSON: ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to write price to JSON: {e}")
            self.stats['json_write_errors'] += 1
    
    def write_error_to_json(self, error_message: str):
        """Write error status to JSON file"""
        try:
            current_time = time.time()
            timestamp_iso = datetime.fromtimestamp(current_time).isoformat()
            
            data = {
                "price": None,
                "timestamp": current_time,
                "last_updated": timestamp_iso,
                "source": "BRTI",
                "status": "error",
                "error": error_message
            }
            
            with self.json_write_lock:
                with open(self.json_output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            logger.debug(f"Updated JSON with error status: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to write error to JSON: {e}")
    
    def initialize_exchanges(self):
        """Initialize exchange connections with error handling"""
        for exchange_id, config in self.exchange_config.items():
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'timeout': 10000,
                    'enableRateLimit': True,
                    'rateLimit': 100,
                })
                self.exchanges[exchange_id] = exchange
                logger.info(f"Initialized {exchange_id} successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")
    
    def fetch_order_book_data(self, exchange_id: str, exchange: Any) -> Optional[OrderBookData]:
        try:
            symbol = self.exchange_config[exchange_id]['symbol']
            order_book = exchange.fetch_order_book(symbol, limit=self.config.max_order_book_depth)
            
            if exchange_id in ['kraken', 'cryptocom']:
                logger.debug(f"{exchange_id} order book sample: bids={order_book['bids'][:2] if order_book['bids'] else []}")
            
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
            logger.warning(f"Failed to fetch data from {exchange_id}: {e}")
            self.stats['exchange_failures'][exchange_id] += 1
            return None
    
    def fetch_all_order_books(self) -> List[OrderBookData]:
        order_books = []
        
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            future_to_exchange = {
                executor.submit(self.fetch_order_book_data, ex_id, ex): ex_id 
                for ex_id, ex in self.exchanges.items()
            }
            
            for future in as_completed(future_to_exchange, timeout=15):
                result = future.result()
                if result:
                    order_books.append(result)
        
        return order_books
    
    def validate_order_book(self, ob: OrderBookData, current_time: int) -> bool:
        """Comprehensive order book validation according to BRTI methodology"""
        
        # Check data age (30 second rule)
        age_seconds = (current_time - ob.retrieval_timestamp) / 1000
        if age_seconds >= self.config.max_data_age_seconds:
            ob.is_valid = False
            ob.error_reason = f"Data too old ({age_seconds:.1f}s)"
            return False
        
        # Check for empty order book
        if not ob.bids or not ob.asks:
            ob.is_valid = False
            ob.error_reason = "Missing bids or asks"
            return False
        
        # Check for crossing order book
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        if best_bid >= best_ask:
            ob.is_valid = False
            ob.error_reason = "Crossing order book"
            return False
        
        # Validate numeric values and check for zero/negative values
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
        """Check for price deviation from median mid-price"""
        if len(order_books) < 2:
            return order_books
        
        # Calculate mid-prices
        mid_prices = []
        for ob in order_books:
            if ob.is_valid and ob.bids and ob.asks:
                mid_price = (ob.bids[0][0] + ob.asks[0][0]) / 2
                mid_prices.append(mid_price)
        
        if len(mid_prices) < 2:
            return order_books
        
        median_mid = statistics.median(mid_prices)
        
        # Check deviation for each order book
        for ob in order_books:
            if ob.is_valid and ob.bids and ob.asks:
                mid_price = (ob.bids[0][0] + ob.asks[0][0]) / 2
                deviation_pct = abs(mid_price - median_mid) / median_mid * 100
                
                if deviation_pct > self.config.max_price_deviation_pct:
                    ob.is_valid = False
                    ob.error_reason = f"Price deviation {deviation_pct:.1f}% > {self.config.max_price_deviation_pct}%"
        
        return order_books
    
    def calculate_dynamic_order_size_cap(self, order_books: List[OrderBookData]) -> float:
        """Calculate dynamic order size cap according to CF Benchmarks methodology
        
        Formula: C_T = s + 5 * sigma
        where:
        - s = trimmed mean (1% trimming from each side)
        - sigma = winsorized sample standard deviation
        - k = floor(0.01 * n_T) for trimming
        """
        all_sizes = []
        
        # Collect all order sizes from valid order books
        for ob in order_books:
            if ob.is_valid:
                for _, size in ob.bids + ob.asks:
                    all_sizes.append(size)
        
        if not all_sizes:
            return float('inf')  # No cap if no data
        
        # Convert to numpy array for statistical operations
        sizes_array = np.array(all_sizes)
        n_T = len(sizes_array)
        
        # Calculate k for trimming: k = floor(0.01 * n_T)
        k = int(np.floor(self.config.order_size_cap_trim_pct * n_T))
        
        # If k is 0 or we don't have enough data, use all data
        if k == 0 or n_T <= 2 * k:
            trimmed_sizes = sizes_array
        else:
            # Sort the array
            sorted_sizes = np.sort(sizes_array)
            
            # Trim k elements from each end
            trimmed_sizes = sorted_sizes[k:n_T-k]
        
        if len(trimmed_sizes) == 0:
            return float('inf')
        
        # Calculate trimmed mean (s)
        trimmed_mean = np.mean(trimmed_sizes)
        
        # Calculate winsorized sample standard deviation (sigma)
        # Using the standard formula for sample standard deviation
        if len(trimmed_sizes) <= 1:
            winsorized_std = 0.0
        else:
            winsorized_std = np.std(trimmed_sizes, ddof=1)  # ddof=1 for sample std
        
        # Apply CF formula: C_T = s + 5 * sigma
        dynamic_cap = trimmed_mean + (self.config.order_size_cap_multiplier * winsorized_std)
        
        logger.debug(f"Dynamic order size cap (CF method): {dynamic_cap:.4f} "
                    f"(n_T: {n_T}, k: {k}, trimmed_mean: {trimmed_mean:.4f}, "
                    f"winsorized_std: {winsorized_std:.4f})")
        
        return dynamic_cap
    
    def apply_order_size_cap(self, orders: List[Tuple[float, float]], cap: float) -> List[Tuple[float, float]]:
        """Apply dynamic order size cap to orders"""
        if cap == float('inf'):
            return orders
        
        capped_orders = []
        for price, size in orders:
            capped_size = min(size, cap)
            capped_orders.append((price, capped_size))
        
        return capped_orders
    
    def consolidate_order_books(self, order_books: List[OrderBookData]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Consolidate multiple order books into single bid/ask arrays with dynamic size capping"""
        
        # Calculate dynamic order size cap
        dynamic_cap = self.calculate_dynamic_order_size_cap(order_books)
        
        all_bids = []
        all_asks = []
        
        for ob in order_books:
            if ob.is_valid:
                # Apply dynamic order size cap
                capped_bids = self.apply_order_size_cap(ob.bids, dynamic_cap)
                capped_asks = self.apply_order_size_cap(ob.asks, dynamic_cap)
                
                all_bids.extend(capped_bids)
                all_asks.extend(capped_asks)
        
        # Sort bids (descending) and asks (ascending)
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])
        
        # Aggregate volumes at same price levels
        consolidated_bids = self._aggregate_price_levels(all_bids)
        consolidated_asks = self._aggregate_price_levels(all_asks)
        
        return consolidated_bids, consolidated_asks
    
    def _aggregate_price_levels(self, orders: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Aggregate orders at the same price level"""
        price_volumes = {}
        for price, volume in orders:
            if price in price_volumes:
                price_volumes[price] += volume
            else:
                price_volumes[price] = volume
        
        return [(price, volume) for price, volume in price_volumes.items()]
    
    def calculate_curves(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, List[Tuple[float, float]]]:
        """Calculate bid, ask, mid, and mid spread-volume curves according to CF Benchmarks methodology"""
        curves = {
            'bid_curve': [],        # bidPV_T(v) 
            'ask_curve': [],        # askPV_T(v)
            'mid_curve': [],        # midPV_T(v)
            'mid_spread_volume_curve': []  # midSV_T(v)
        }
        
        # Build cumulative bid and ask curves
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

        # Determine max volume depth (v_T) and build volume steps
        max_volume = min(cum_bid_vol, cum_ask_vol)
        volume_steps = np.arange(self.config.spacing_parameter, max_volume, self.config.spacing_parameter)

        # Interpolate prices at each volume step and populate all curves
        for v_step in volume_steps:
            bid_price = next((p for cv, p in bid_cum if cv >= v_step), None)
            ask_price = next((p for cv, p in ask_cum if cv >= v_step), None)

            if bid_price is None or ask_price is None:
                continue

            mid_price = (bid_price + ask_price) / 2
            
            # Populate all curve types as per CF Benchmarks methodology
            curves['bid_curve'].append((v_step, bid_price))      # bidPV_T(v) [Eq. 1b, 1d]
            curves['ask_curve'].append((v_step, ask_price))      # askPV_T(v) [Eq. 1a, 1c]
            curves['mid_curve'].append((v_step, mid_price))      # midPV_T(v) [Eq. 1e]
            
            # Calculate mid spread-volume curve: midSV_T(v) = askPV_T(v) / midPV_T(v) - 1 [Eq. 1f]
            if mid_price > 0:
                mid_spread_pct = (ask_price / mid_price) - 1
                curves['mid_spread_volume_curve'].append((v_step, mid_spread_pct))

        return curves
    
    def calculate_utilized_depth(self, curves: Dict[str, List[Tuple[float, float]]]) -> float:
        """Calculate utilized depth based on spread-volume curve with spacing parameter
        
        Implements: v_T = max(v_i where midSV_T(v_i) <= D AND midSV_T(v_i+1) > D, s)
        where D is the maximum spread deviation percentage
        """
        if not curves['mid_spread_volume_curve']:
            return self.config.spacing_parameter  # Return spacing parameter as minimum
        
        # Find maximum volume where spread doesn't exceed threshold
        utilized_volume = self.config.spacing_parameter  # Initialize with spacing parameter
        
        for v_step, spread_pct in curves['mid_spread_volume_curve']:
            if spread_pct <= self.config.max_spread_volume_deviation_pct:
                utilized_volume = v_step  # Update to current volume step
            else:
                break  # Stop at first violation
        
        # Ensure utilized depth is at least the spacing parameter
        final_utilized_depth = max(utilized_volume, self.config.spacing_parameter)
        
        logger.debug(f"Utilized depth: calculated={utilized_volume}, "
                    f"spacing_param={self.config.spacing_parameter}, "
                    f"final={final_utilized_depth}")
        
        return final_utilized_depth
    
    def apply_exponential_weighting(self, curves: Dict[str, List[Tuple[float, float]]], utilized_depth: float) -> float:
        """Apply exponential weighting to calculate final BRTI value [137, 138, 143, Eq. 3]"""
        if not curves['mid_curve'] or utilized_depth == 0:
            return 0.0

        weighted_sum = 0.0
        normalization_factor_sum = 0.0  # This corresponds to NF from methodology
        
        lambda_param = self.config.lambda_param
        spacing_s = self.config.spacing_parameter  # This is 's'

        # Filter the mid_curve to only include data up to the utilized_depth
        # curves['mid_curve'] is expected to be [(volume_step, mid_price), ...]
        relevant_mid_curve_entries = [
            (v_step, mid_price) for v_step, mid_price in curves['mid_curve'] 
            if v_step > 0 and v_step <= utilized_depth  # Ensure volume step is positive and within utilized depth
        ]
        
        if not relevant_mid_curve_entries:
            return 0.0

        # Calculate the Normalization Factor (NF) first, as per methodology [142, Eq. 3]
        # NF = sum(lambda * exp(-lambda * v)) for v in s, 2s, ... vT
        for v_step, _ in relevant_mid_curve_entries:
            normalization_factor_sum += lambda_param * np.exp(-lambda_param * v_step)
        
        if normalization_factor_sum == 0:
            return 0.0  # Avoid division by zero, though unlikely with positive lambda and v_step

        # Now calculate the weighted sum as per Eq. 3
        # CCRTI_T = sum(midPV_T(v) * (1/NF) * lambda * exp(-lambda * v))
        for v_step, mid_price_at_v in relevant_mid_curve_entries:
            # The weight term is (1/NF) * lambda * exp(-lambda * v)
            weight_term = (1 / normalization_factor_sum) * lambda_param * np.exp(-lambda_param * v_step)
            weighted_sum += mid_price_at_v * weight_term
            
        return weighted_sum
    
    def format_single_line_display(self, order_books: List[OrderBookData], brti_value: float, 
                                  utilized_depth: float, dynamic_cap: float) -> str:
        """Format all information into a single updating line"""
        self.calculation_count += 1
        
        # Get current time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Count valid/invalid exchanges
        valid_exchanges = [ob.exchange_id for ob in order_books if ob.is_valid]
        invalid_exchanges = [ob.exchange_id for ob in order_books if not ob.is_valid]
        
        # Price change indicator
        price_indicator = ""
        if self.last_price is not None:
            if brti_value > self.last_price:
                price_indicator = "🔼"
            elif brti_value < self.last_price:
                price_indicator = "🔽"
            else:
                price_indicator = "━"
        self.last_price = brti_value
        
        # Create exchange status string (short codes)
        exchange_codes = {
            'coinbase': 'CB', 'kraken': 'KR', 'bitstamp': 'BS', 
            'gemini': 'GM', 'cryptocom': 'CC'
        }
        
        valid_codes = [exchange_codes.get(ex, ex[:2].upper()) for ex in valid_exchanges]
        invalid_codes = [exchange_codes.get(ex, ex[:2].upper()) for ex in invalid_exchanges]
        
        # Build the display line
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
        
        return " | ".join(parts)
    
    def calculate_brti(self) -> Optional[float]:
        """Main BRTI calculation method"""
        try:
            current_time = int(time.time() * 1000)  # Convert to milliseconds
            
            # Fetch all order books
            order_books = self.fetch_all_order_books()
            
            if not order_books:
                error_line = f"{datetime.now().strftime('%H:%M:%S')} | #{self.calculation_count + 1:04d} | ERROR: No order book data available"
                self.update_single_line_display(error_line)
                self.write_error_to_json("No order book data available")
                return None
            
            # Validate order books
            for ob in order_books:
                self.validate_order_book(ob, current_time)
            
            # Check price deviation
            order_books = self.check_price_deviation(order_books)
            
            # Filter valid order books
            valid_order_books = [ob for ob in order_books if ob.is_valid]
            
            if len(valid_order_books) < 2:
                error_line = f"{datetime.now().strftime('%H:%M:%S')} | #{self.calculation_count + 1:04d} | ERROR: Insufficient valid exchanges ({len(valid_order_books)}/5)"
                self.update_single_line_display(error_line)
                self.write_error_to_json(f"Insufficient valid order books: {len(valid_order_books)}")
                return None
            
            # Consolidate order books (with dynamic size capping)
            consolidated_bids, consolidated_asks = self.consolidate_order_books(valid_order_books)
            
            if not consolidated_bids or not consolidated_asks:
                error_line = f"{datetime.now().strftime('%H:%M:%S')} | #{self.calculation_count + 1:04d} | ERROR: No consolidated order book data"
                self.update_single_line_display(error_line)
                self.write_error_to_json("No consolidated order book data")
                return None
            
            # Calculate curves (now properly populates all curve types)
            curves = self.calculate_curves(consolidated_bids, consolidated_asks)
            
            # Calculate utilized depth (now works with populated mid_spread_volume_curve)
            utilized_depth = self.calculate_utilized_depth(curves)
            
            # Apply exponential weighting (now uses CF Benchmarks methodology)
            brti_value = self.apply_exponential_weighting(curves, utilized_depth)
            
            # Get dynamic cap for display
            dynamic_cap = self.calculate_dynamic_order_size_cap(valid_order_books)
            
            # Display single line update
            display_line = self.format_single_line_display(order_books, brti_value, utilized_depth, dynamic_cap)
            self.update_single_line_display(display_line)
            
            # Write successful result to JSON
            additional_data = {
                "utilized_depth": round(utilized_depth, 2),
                "dynamic_cap": round(dynamic_cap, 2),
                "valid_exchanges": len(valid_order_books),
                "total_exchanges": len(order_books),
                "exchange_status": {
                    ob.exchange_id: "valid" if ob.is_valid else ob.error_reason 
                    for ob in order_books
                }
            }
            self.write_price_to_json(brti_value, additional_data)
            
            self.stats['successful_calculations'] += 1
            
            return brti_value
            
        except Exception as e:
            error_line = f"{datetime.now().strftime('%H:%M:%S')} | #{self.calculation_count + 1:04d} | ERROR: {str(e)}"
            self.update_single_line_display(error_line)
            self.write_error_to_json(f"Error calculating BRTI: {str(e)}")
            return None
        finally:
            self.stats['total_calculations'] += 1
    
    def run_continuous(self, duration_seconds: Optional[int] = None):
        """Run BRTI calculations continuously"""
        self.print_new_line("Starting BRTI Real-Time Index...")
        self.print_new_line(f"JSON output: {self.json_output_path.absolute()}")
        self.print_new_line("=" * 80)
        
        start_time = time.time()
        
        try:
            while True:
                calculation_start = time.time()
                
                brti_value = self.calculate_brti()
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Sleep to maintain interval
                calculation_time = time.time() - calculation_start
                sleep_time = max(0, self.config.calculation_interval_seconds - calculation_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.print_new_line("\nBRTI calculation stopped by user")
        except Exception as e:
            self.print_new_line(f"\nUnexpected error: {e}")
        finally:
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics"""
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

def main():
    """Main execution function"""
    # Create custom configuration matching CF Benchmarks methodology
    config = BRTIConfig(
        max_data_age_seconds=30,
        max_price_deviation_pct=5.0,
        max_spread_volume_deviation_pct=0.5,
        lambda_param=1.0,
        spacing_parameter=1.0,  # s = 1 for Bitcoin BRTI
        order_size_cap_trim_pct=0.01,  # 1% trimming from each side (CF spec)
        order_size_cap_multiplier=5.0,  # CF formula: C_T = s + 5 * sigma
        calculation_interval_seconds=1.0,
        json_output_file="aggregate_price.json"  # Output file for price data
    )
    
    # Initialize BRTI calculator
    brti = OptimizedBRTI(config)
    
    # Run continuously (remove duration for infinite operation)
    brti.run_continuous()

if __name__ == "__main__":
    main()
