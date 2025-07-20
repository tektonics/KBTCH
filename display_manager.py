import asyncio
import json
import time
import sys
import numpy as np
import subprocess
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from kalshi_bot.kalshi_client import KalshiClient
from portfolio import Portfolio


@dataclass
class MarketInfo:
    ticker: str
    strike: float
    distance: float
    is_primary: bool
    market_data: Optional[Any] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None

@dataclass
class TradingParams:
    base_market_count: int = 3
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 1.0
    max_markets: int = 7

@dataclass
class TradeExecutionInfo:
    ticker: str
    action: str
    quantity: int
    price: float
    status: str
    timestamp: datetime
    reason: str

class BTCPriceMonitor:
    def __init__(self, price_file: str = "aggregate_price.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5
        self.price_history = []
        self.max_history_minutes = 30
    
    def get_current_price(self) -> Optional[float]:
        now = time.time()
        if now - self.last_check < self.check_interval:
            return self.last_price
        
        self.last_check = now
        
        try:
            if not self.price_file.exists():
                return None
            
            current_modified = self.price_file.stat().st_mtime
            if current_modified == self.last_modified:
                return self.last_price
            
            with open(self.price_file, 'r') as f:
                data = json.load(f)
                price = data.get("price")
                
                if price and isinstance(price, (int, float)) and price > 0:
                    new_price = float(price)
                    
                    if new_price != self.last_price:
                        self.price_history.append((now, new_price))
                        self._cleanup_price_history(now)
                    
                    self.last_price = new_price
                    self.last_modified = current_modified
                    return self.last_price
                    
        except (json.JSONDecodeError, IOError):
            pass
        
        return None
    
    def _cleanup_price_history(self, current_time: float):
        cutoff_time = current_time - (self.max_history_minutes * 60)
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff_time]
    
    def calculate_volatility(self, window_minutes: int = 15) -> float:
        cutoff_time = time.time() - (window_minutes * 60)
        recent_history = [(t, p) for t, p in self.price_history if t > cutoff_time]
        
        if len(recent_history) < 3:
            return 0.0
        
        prices = [price for _, price in recent_history]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0.0
        
        std_dev = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        return std_dev * np.sqrt(525600)

class OHLCVMonitor:
    def __init__(self, data_file: str = "ohlcv_data.json"):
        self.data_file = Path(data_file)
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.5
        self.exchange_data = {}
    
    def get_ohlcv_data(self) -> Dict[str, Any]:
        now = time.time()
        if now - self.last_check < self.check_interval:
            return self.exchange_data
        
        self.last_check = now
        
        try:
            if not self.data_file.exists():
                return {}
            
            current_modified = self.data_file.stat().st_mtime
            if current_modified == self.last_modified:
                return self.exchange_data
            
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.exchange_data = data.get("exchanges", {})
                self.last_modified = current_modified
                return self.exchange_data
                
        except (json.JSONDecodeError, IOError):
            return {}

class BRTIManager:
    def __init__(self, script_path: str = "brti.py"):
        self.script_path = Path(script_path)
        self.process: Optional[subprocess.Popen] = None
        self.price_file = Path("aggregate_price.json")
        
    def is_brti_running(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        
        if not self.price_file.exists():
            return False
        
        try:
            last_modified = self.price_file.stat().st_mtime
            return time.time() - last_modified < 10
        except OSError:
            return False
    
    async def start_brti(self) -> bool:
        if not self.script_path.exists():
            return False
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            
            for _ in range(30):
                if self.price_file.exists():
                    try:
                        with open(self.price_file, 'r') as f:
                            data = json.load(f)
                            if data.get("price") and data.get("status") == "active":
                                return True
                    except (json.JSONDecodeError, IOError):
                        pass
                await asyncio.sleep(1)
            
            return self.process.poll() is None
            
        except Exception:
            return False
    
    def stop_brti(self):
        if self.process is None:
            return
        
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        except Exception:
            pass
        finally:
            self.process = None

class OHLCVManager:
    def __init__(self, script_path: str = "OHLCV.py"):
        self.script_path = Path(script_path)
        self.process: Optional[subprocess.Popen] = None
        self.data_file = Path("ohlcv_data.json")
        
    def is_ohlcv_running(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        if not self.data_file.exists():
            return False
        try:
            last_modified = self.data_file.stat().st_mtime
            return time.time() - last_modified < 10
        except OSError:
            return False
    
    async def start_ohlcv(self) -> bool:
        if not self.script_path.exists():
            return False
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )
            for _ in range(30):
                if self.data_file.exists():
                    try:
                        with open(self.data_file, 'r') as f:
                            data = json.load(f)
                            if data.get("exchanges") and data.get("status") == "active":
                                return True
                    except (json.JSONDecodeError, IOError):
                        pass
                await asyncio.sleep(1)
            return self.process.poll() is None
        except Exception:
            return False
    
    def stop_ohlcv(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        except Exception:
            pass
        finally:
            self.process = None

class MarketSelector:
    def __init__(self, params: TradingParams):
        self.params = params
    
    @staticmethod
    def extract_strike_price(ticker: str) -> float:
        try:
            parts = ticker.split("-")
            if len(parts) >= 3 and parts[-1].startswith("T"):
                return float(parts[-1][1:])
        except (ValueError, IndexError):
            pass
        return 0.0
    
    def calculate_adaptive_market_count(self, volatility: float, time_to_expiry_hours: float) -> int:
        base_count = self.params.base_market_count
        
        if volatility > self.params.volatility_threshold_high:
            base_count += 3
        elif volatility > self.params.volatility_threshold_low:
            base_count += 1
        
       
        time_adjustments = {1: 3, 3: 2, 6: 1}
        for threshold, adjustment in time_adjustments.items():
            if time_to_expiry_hours < threshold:
                base_count += adjustment
                break
        
        return min(base_count, self.params.max_markets)
    
    def select_target_markets(self, markets: List[Dict], btc_price: float, volatility: float) -> List[MarketInfo]:
        if not markets or not btc_price:
            return []
        
        target_count = self.calculate_adaptive_market_count(volatility, 1.0)
        
        markets_with_distance = []
        for market in markets:
            strike = self.extract_strike_price(market["ticker"])
            if strike > 0:
                distance = abs(strike - btc_price)
                markets_with_distance.append({
                    'market': market,
                    'ticker': market["ticker"],
                    'strike': strike,
                    'distance': distance
                })
        
        markets_with_distance.sort(key=lambda x: x['distance'])
        selected = markets_with_distance[:target_count]
        
        primary_ticker = selected[0]['ticker'] if selected else None
        selected.sort(key=lambda x: x['strike'])
        
        return [
            MarketInfo(
                ticker=m['ticker'],
                strike=m['strike'],
                distance=m['distance'],
                is_primary=(m['ticker'] == primary_ticker)
            )
            for m in selected
        ]

class DisplayManager:
    def __init__(self):
        self.display_line_count = 0

    def clear_display(self):
        if self.display_line_count > 0:
            for i in range(self.display_line_count):
                sys.stdout.write('\r\033[K')
                if i < self.display_line_count - 1:
                    sys.stdout.write('\033[A')
            sys.stdout.flush()
            self.display_line_count = 0
    
    def update_multiline_display(self, lines: list):
        self.clear_display()
        
        for i, line in enumerate(lines):
            if i > 0:
               sys.stdout.write('\n')
            sys.stdout.write(line)
        
        sys.stdout.flush()
        self.display_line_count = len(lines)
        
    def print_new_line(self, line: str):
        self.clear_display()
        print(line)
        self.display_line_count = 0
    
    def format_market_display(self, active_markets: List[MarketInfo], btc_price: float, 
                        volatility: float, brti_running: bool, ohlcv_running: bool, 
                        ohlcv_data: Dict[str, Any]):
        try:
            edt_time = datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            edt_time = datetime.now()
        
        lines = []
        
        vol_indicator = "🔥" if volatility > 1.0 else "📈" if volatility > 0.5 else "📊"
        ohlcv_status = "🟢" if ohlcv_running else "🔴"
        brti_status = "🟢" if brti_running else "🔴"

        header = (f"{edt_time.strftime('%H:%M:%S')} | "
                 f"BTC: ${btc_price:,.2f} | "
                 f"BRTI: {brti_status} | "
                 f"OHLCV: {ohlcv_status}")
        lines.append(header)
        
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            strike_labels = [
                f"${m.strike:,.0f}🎯" if m.is_primary else f"${m.strike:,.0f}"
                for m in sorted_markets
            ]
            lines.append(f"Ladder: {' | '.join(strike_labels)}")
        
        if active_markets:
            sorted_markets = sorted(active_markets, key=lambda m: m.strike)
            
            for market in sorted_markets:
                if market.market_data:
                    line = self._format_market_line(market)
                    lines.append(line)
        
        return lines

    def _format_market_line(self, market: MarketInfo) -> str:
        data = market.market_data
        primary_indicator = "🎯" if market.is_primary else "  "
        
        if data.yes_bid and data.yes_ask:
            yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
            no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
            no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
            spread = data.yes_ask - data.yes_bid
            spread_text = f"Spread: {spread:.0f}¢"
        else:
            yes_prices = "YES: --/--"
            no_prices = "NO: --/--"
            spread_text = "No data"
        
        line = (f"{primary_indicator}${market.strike:,.0f}: "
               f"{yes_prices} | {no_prices} | {spread_text}")
        
        return line

class VolatilityAdaptiveTrader:
    def __init__(self, event_id: Optional[str] = None):
        self.event_id = event_id or self._generate_current_event_id()
        self.client = KalshiClient()
        self.btc_monitor = BTCPriceMonitor()
        self.brti_manager = BRTIManager()
        self.ohlcv_manager = OHLCVManager()
        self.ohlcv_monitor = OHLCVMonitor()
        self.params = TradingParams()
        self.market_selector = MarketSelector(self.params)
        self.display = DisplayManager()
        
        self.markets = []
        self.active_markets: List[MarketInfo] = []
        self.market_subscriptions = set()
        self.last_btc_price = None
        self.current_volatility = 0.0
        self.shutdown_requested = False
        
        self.btc_updates = 0
        self.market_updates = 0
        self.volatility_updates = 0
        self.last_market_update_time = None

        self._setup_signal_handlers()
        self._silence_client_output()
    
    def _generate_current_event_id(self) -> str:
        try:
            edt_tz = ZoneInfo("America/New_York")
            now = datetime.now(edt_tz)
        except ImportError:
            now = datetime.now()
        
        next_hour = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour + 1) % 24)
        if next_hour.hour == 0 and now.hour == 23:
            next_hour = next_hour.replace(day=now.day + 1)
        
        return f"KXBTCD-{next_hour.strftime('%y%b%d%H').upper()}"
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'shutdown_requested', True))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'shutdown_requested', True))
    
    def _silence_client_output(self):
        import builtins
        self._original_print = builtins.print
        
        def silent_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            if not any(kw in message.lower() for kw in ['subscribed', 'kalshi']):
                self._original_print(*args, **kwargs)
        
        builtins.print = silent_print
    
    def _restore_client_output(self):
        if hasattr(self, '_original_print'):
            import builtins
            builtins.print = self._original_print
    
    async def initialize(self) -> bool:
        print("🚀 KBTCH Starting...")
        
        if not self.brti_manager.is_brti_running():
            await self.brti_manager.start_brti()
        
        if not self.ohlcv_manager.is_ohlcv_running():
            await self.ohlcv_manager.start_ohlcv()

        try:
            markets_data = self.client.get_markets(self.event_id)
            self.markets = markets_data.get("markets", [])
            
            if not self.markets:
                print(f"❌ No markets found for {self.event_id}")
                return False
            
            for _ in range(60):
                btc_price = self.btc_monitor.get_current_price()
                if btc_price:
                    self.last_btc_price = btc_price
                    break
                await asyncio.sleep(1)
            else:
                print("❌ No BTC price data")
                return False
            
            # Setup markets
            await asyncio.sleep(2)
            self.current_volatility = self.btc_monitor.calculate_volatility()
            await self._update_market_subscriptions()
            
            if not self.active_markets:
                print("❌ No target markets")
                return False
            
            print(f"✅ Ready: {len(self.markets)} markets, BTC ${btc_price:,.0f}")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def _update_market_subscriptions(self):
        target_markets = self.market_selector.select_target_markets(
            self.markets, self.last_btc_price, self.current_volatility
        )
        
        if not target_markets:
            return
        
        new_tickers = {market.ticker for market in target_markets}
        
        # Subscribe to new markets
        for ticker in new_tickers - self.market_subscriptions:
            try:
                asyncio.create_task(self.client.subscribe_to_market(ticker))
                self.market_subscriptions.add(ticker)
            except Exception:
                pass
        
        self.active_markets = target_markets
    
    async def run_trading_loop(self):
        if not await self.initialize():
            return
        
        # Setup WebSocket message counting
        original_handler = self.client._handle_ws_message
        def counting_handler(msg):
            self.market_updates += 1
            self.last_market_update_time = time.time()
            original_handler(msg)
        self.client._handle_ws_message = counting_handler
        
        try:
            while not self.shutdown_requested:
                current_time = time.time()
                
                # Monitor BRTI health
                if not self.brti_manager.is_brti_running():
                    await self.brti_manager.start_brti()
                
               # Monitor OHLCV health
                if not self.ohlcv_manager.is_ohlcv_running():
                    await self.ohlcv_manager.start_ohlcv()

                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.btc_updates += 1
                
                new_volatility = self.btc_monitor.calculate_volatility()
                if abs(new_volatility - self.current_volatility) > 0.1:
                    self.current_volatility = new_volatility
                    self.volatility_updates += 1
                    await self._update_market_subscriptions()
                
                for market in self.active_markets:
                    market.market_data = self.client.get_mid_prices(market.ticker)
                    if market.market_data:
                        if market.market_data.yes_bid and market.market_data.yes_ask:
                            market.spread = market.market_data.yes_ask - market.market_data.yes_bid
                            market.spread_pct = (market.spread / market.market_data.yes_ask) * 100
                
                ohlcv_data = self.ohlcv_monitor.get_ohlcv_data()
                
                lines = self.display.format_market_display(
                    self.active_markets, self.last_btc_price, 
                    self.current_volatility, self.brti_manager.is_brti_running(),
                    self.ohlcv_manager.is_ohlcv_running(), ohlcv_data
                )
                self.display.update_multiline_display(lines)
                

                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            self.display.print_new_line("\n🛑 Shutting down...")
        except Exception as e:
            self.display.print_new_line(f"\n❌ Error: {e}")
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        self._restore_client_output()
        self.brti_manager.stop_brti()
        self.ohlcv_manager.stop_ohlcv()
        try:
            await self.client.close()
        except:
            pass

async def main():
    missing_deps = []
    dependencies = ['ccxt', 'numpy', 'kalshi_bot.kalshi_client', 'portfolio', 'trading_logic']
    
    for dep in dependencies:
        try:
            if dep == 'kalshi_bot.kalshi_client':
                from kalshi_bot.kalshi_client import KalshiClient
            else:
                __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        return
    
    if not Path("brti.py").exists():
        print("❌ brti.py not found")
        return
    
    trader = VolatilityAdaptiveTrader()
    try:
        await trader.run_trading_loop()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        sys.exit(1)
