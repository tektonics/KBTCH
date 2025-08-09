import sys
import os
import base64
import time
import json
import asyncio
import logging
import bisect
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from event_bus import event_bus, EventTypes
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.kalshiconfig import (
    REST_API_BASE_URL,
    WEBSOCKET_API_URL,
    KALSHI_API_KEY_ID,
    KALSHI_PRIVATE_KEY_PATH,
    API_ENDPOINTS,
    WS_CHANNELS,
    REQUEST_HEADERS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TickerData:
    price: Optional[float] = None
    previous_price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    volume_delta: Optional[int] = None
    timestamp: Optional[float] = None

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
    max_markets: int = 5

class KalshiClient:
    def __init__(self):
        self.base_url = REST_API_BASE_URL
        self.ws_url = WEBSOCKET_API_URL
        self.key_id = KALSHI_API_KEY_ID
        self.private_key_path = KALSHI_PRIVATE_KEY_PATH

        if not self.key_id or not self.private_key_path:
            raise ValueError("Missing required environment variables: KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH")

        self.private_key = self._load_private_key()
        self.mid_prices: Dict[str, TickerData] = {}
        self.orderbooks: Dict[str, Dict] = {}
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._reconnect_attempts_map: Dict[str, int] = {}
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1
        self.btc_monitor = BTCPriceMonitor()
        self.params = TradingParams()
        self.market_selector = MarketSelector(self.params)
        self.all_markets_for_event: List[Dict] = []
        self.active_market_info: Dict[str, MarketInfo] = {}
        self.market_subscription_tasks: Dict[str, asyncio.Task] = {}
        self.last_btc_price: Optional[float] = None
        self.current_volatility: float = 0.0
        self.event_ticker: Optional[str] = None
        self._shutdown_requested = False
        self.update_interval = .01

    def _load_private_key(self):
        try:
            key_path = Path(self.private_key_path)
            if not key_path.exists():
                raise FileNotFoundError(f"Private key file not found: {self.private_key_path}")
            with open(key_path, "rb") as key_file:
                return serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise

    def _generate_signature(self, timestamp: str, method: str, path: str) -> str:
        try:
            message = f"{timestamp}{method}{path}".encode("utf-8")
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Failed to generate signature: {e}")
            raise

    def _build_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path)
        headers = REQUEST_HEADERS.copy()
        headers.update({
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        })
        return headers

    def get_markets(self, event_ticker: str) -> Dict[str, Any]:
        method = "GET"
        path = API_ENDPOINTS["GET_MARKETS"]
        params = {"event_ticker": event_ticker}
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(.5)

    def get_balance(self) -> Dict[str, Any]:
        method = "GET"
        path = API_ENDPOINTS["GET_BALANCE"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Balance request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def get_positions(self) -> Dict[str, Any]:
        method = "GET"
        path = API_ENDPOINTS["GET_POSITIONS"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Positions request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def get_fills(self) -> Dict[str, Any]:
        method = "GET"
        path = API_ENDPOINTS["GET_FILLS"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Fills request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    def get_orders(self) -> Dict[str, Any]:
        method = "GET"
        path = API_ENDPOINTS["GET_ORDERS"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Orders request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    async def subscribe_to_market(self, market_ticker: str):
        reconnect_attempts = self._reconnect_attempts_map.get(market_ticker, 0)
        while True:
            try:
                await self._connect_and_subscribe(market_ticker)
                self._reconnect_attempts_map[market_ticker] = 0
                reconnect_attempts = 0
            except Exception as e:
                logger.error(f"WebSocket connection for {market_ticker} failed: {e}")
                reconnect_attempts += 1
                self._reconnect_attempts_map[market_ticker] = reconnect_attempts
                if reconnect_attempts < self._max_reconnect_attempts:
                    delay = self._reconnect_delay * (2 ** (reconnect_attempts - 1))
                    logger.info(f"Reconnecting {market_ticker} in {delay:.1f} seconds... (attempt {reconnect_attempts})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max reconnection attempts reached for {market_ticker}. Giving up.")
                    if market_ticker in self._ws_connections:
                        del self._ws_connections[market_ticker]
                    break

    async def _connect_and_subscribe(self, market_ticker: str):
        from urllib.parse import urlparse
        parsed_url = urlparse(self.ws_url)
        path = parsed_url.path
        method = "GET"
        headers = self._build_auth_headers(method, path)

        async with websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10
        ) as ws:
            self._ws_connections[market_ticker] = ws
            subscribe_message = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": [WS_CHANNELS["TICKER_V2"], WS_CHANNELS["ORDERBOOK_DELTA"]],
                    "market_ticker": market_ticker
                }
            }
            await ws.send(json.dumps(subscribe_message))
            logger.info(f"[Kalshi] Subscribed to {market_ticker}")

            async for message in ws:
                try:
                    data = json.loads(message)
                    asyncio.create_task(self._handle_ws_message_async(data))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message for {market_ticker}: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message for {market_ticker}: {e}")

    def _handle_ws_message(self, msg: Dict[str, Any]):
        msg_type = msg.get("type")
        data = msg.get("msg", {})
        market_ticker = data.get("market_ticker")

        if not market_ticker and msg_type != "subscribed":
            logger.warning(f"Message missing market_ticker for type: {msg_type}")
            return

        if msg_type == "ticker_v2":
            current_ticker_data = self.mid_prices.get(market_ticker)
            if current_ticker_data is None:
                current_ticker_data = TickerData()
                self.mid_prices[market_ticker] = current_ticker_data

            if "price" in data:
                current_ticker_data.previous_price = current_ticker_data.price
                current_ticker_data.price = data["price"]
            if "yes_bid" in data:
                current_ticker_data.yes_bid = data["yes_bid"]
            if "yes_ask" in data:
                current_ticker_data.yes_ask = data["yes_ask"]
            if "no_bid" in data:
                current_ticker_data.no_bid = data["no_bid"]
            if "no_ask" in data:
                current_ticker_data.no_ask = data["no_ask"]
            if "volume_delta" in data:
                current_ticker_data.volume_delta = data["volume_delta"]
            current_ticker_data.timestamp = time.time()

            strike_price = MarketSelector.extract_strike_price(market_ticker)

            try:
                event_bus.publish(
                    EventTypes.MARKET_DATA_UPDATE,
                    {
                         "market_ticker": market_ticker,
                         "yes_bid": current_ticker_data.yes_bid,
                         "yes_ask": current_ticker_data.yes_ask, 
                         "no_bid": current_ticker_data.no_bid,
                         "no_ask": current_ticker_data.no_ask,
                         "strike_price": strike_price,
                         "timestamp": current_ticker_data.timestamp
                    },
                    source="kms"
                )
            except Exception as e:
                logger.error(f"Failed to publish market data event: {e}")

        elif msg_type == "orderbook_snapshot":
            self.orderbooks[market_ticker] = {
                "yes": data.get("yes", []),
                "no": data.get("no", [])
            }
        elif msg_type == "orderbook_delta":
            current_orderbook = self.orderbooks.get(market_ticker)
            if current_orderbook is None:
                logger.warning(f"Received orderbook_delta for {market_ticker} before snapshot. Data might be inconsistent.")
                return

            price = data.get("price")
            delta = data.get("delta")
            side = data.get("side")

            if price is None or delta is None or side is None:
                logger.error(f"Incomplete orderbook_delta message for {market_ticker}: {data}")
                return

            target_side_list = current_orderbook.get(side)
            if target_side_list is None:
                logger.error(f"Invalid side '{side}' in orderbook_delta for {market_ticker}. Data: {data}")
                return

            found = False
            for i, (p, count) in enumerate(target_side_list):
                if p == price:
                    new_count = count + delta
                    if new_count <= 0:
                        target_side_list.pop(i)
                    else:
                        target_side_list[i] = [price, new_count]
                    found = True
                    break

            if not found and delta > 0:
                bisect.insort_left(target_side_list, [price, delta])
        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    async def _handle_ws_message_async(self, data):
        try:
            self._handle_ws_message(data)
        except Exception as e:
            logger.error(f"Error in async message handler: {e}")

    def get_mid_prices(self, market_ticker: str) -> Optional[TickerData]:
        return self.mid_prices.get(market_ticker)

    def get_orderbook(self, market_ticker: str) -> Optional[Dict]:
        return self.orderbooks.get(market_ticker)

    def get_all_market_data(self) -> Dict[str, Dict]:
        return {
            "mid_prices": {k: v.__dict__ for k, v in self.mid_prices.items()},
            "orderbooks": self.orderbooks.copy()
        }

    async def close(self):
        for market_ticker, ws_connection in list(self._ws_connections.items()):
            try:
                if ws_connection and not ws_connection.closed:
                    await ws_connection.close()
                    logger.info(f"WebSocket connection for {market_ticker} closed.")
                if market_ticker in self._ws_connections:
                    del self._ws_connections[market_ticker]
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {market_ticker}: {e}")

        for ticker, task in list(self.market_subscription_tasks.items()):
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Subscription task for {ticker} cancelled.")
            except Exception as e:
                logger.warning(f"Error waiting for cancelled task {ticker}: {e}")
            del self.market_subscription_tasks[ticker]
        logger.info("All WebSocket connections and associated tasks closed.")

    def _generate_current_event_id(self) -> str:
        try:
            edt_tz = ZoneInfo("America/New_York")
            now = datetime.now(edt_tz)
        except ImportError:
            logger.warning("zoneinfo not found, using naive datetime. Event ID generation might be time-zone inaccurate.")
            now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour + 1) % 24)
        if next_hour.hour == 0 and now.hour == 23:
            next_hour = next_hour.replace(day=now.day + 1)
        return f"KXBTCD-{next_hour.strftime('%y%b%d%H').upper()}"

    async def _update_market_subscriptions_adaptive(self):
        if not self.all_markets_for_event:
            try:
                markets_data = self.get_markets(self.event_ticker)
                self.all_markets_for_event = markets_data.get("markets", [])
                if not self.all_markets_for_event:
                    logger.warning(f"No markets found for {self.event_ticker}. Cannot perform market selection.")
                    return
                else:
                    logger.info(f"Found {len(self.all_markets_for_event)} markets for event {self.event_ticker}.")
            except Exception as e:
                logger.error(f"Failed to fetch markets for {self.event_ticker}: {e}")
                return

        btc_price = self.btc_monitor.get_current_price()
        if btc_price:
            self.last_btc_price = btc_price
            self.current_volatility = self.btc_monitor.calculate_volatility()
            logger.debug(f"Current BTC: ${self.last_btc_price:,.2f}, Volatility: {self.current_volatility:.2f}")
        else:
            logger.warning("No current BTC price data available for market selection. Skipping market update.")
            return

        target_count = self.market_selector.calculate_adaptive_market_count(self.current_volatility, 1.0)
        target_markets = self.market_selector.select_target_markets(
            self.all_markets_for_event, self.last_btc_price, self.current_volatility
        )

        if not target_markets:
            logger.warning("No target markets selected by the market selector based on current conditions.")
            await self._cancel_all_market_subscriptions()
            self.active_market_info = {}
            return

        new_tickers = {market.ticker for market in target_markets}
        current_subscribed_tickers = set(self.market_subscription_tasks.keys())

        for ticker_to_unsubscribe in current_subscribed_tickers - new_tickers:
            if ticker_to_unsubscribe in self.market_subscription_tasks:
                task = self.market_subscription_tasks[ticker_to_unsubscribe]
                if not task.done():
                    task.cancel()
                    logger.info(f"Cancelled subscription task for {ticker_to_unsubscribe}")
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error waiting for cancellation of {ticker_to_unsubscribe}: {e}")
                del self.market_subscription_tasks[ticker_to_unsubscribe]
                if ticker_to_unsubscribe in self._ws_connections:
                    try:
                        await self._ws_connections[ticker_to_unsubscribe].close()
                        del self._ws_connections[ticker_to_unsubscribe]
                        logger.info(f"Closed WebSocket for {ticker_to_unsubscribe}")
                    except Exception as e:
                        logger.error(f"Error closing WebSocket for {ticker_to_unsubscribe}: {e}")

        for ticker_to_subscribe in new_tickers - current_subscribed_tickers:
            if ticker_to_subscribe not in self.market_subscription_tasks:
                try:
                    task = asyncio.create_task(self.subscribe_to_market(ticker_to_subscribe))
                    self.market_subscription_tasks[ticker_to_subscribe] = task
                    logger.info(f"Started subscription task for {ticker_to_subscribe}")
                except Exception as e:
                    logger.error(f"Failed to start subscription task for {ticker_to_subscribe}: {e}")

        self.active_market_info = {m.ticker: m for m in target_markets}
        for market_info in self.active_market_info.values():
            market_info.market_data = self.get_mid_prices(market_info.ticker)
            if market_info.market_data:
                if market_info.market_data.yes_bid and market_info.market_data.yes_ask:
                    market_info.spread = market_info.market_data.yes_ask - market_info.market_data.yes_bid
                    market_info.spread_pct = (market_info.spread / market_info.market_data.yes_ask) * 100 if market_info.market_data.yes_ask else 0.0

    async def _cancel_all_market_subscriptions(self):
        for ticker_to_cancel in list(self.market_subscription_tasks.keys()):
            task = self.market_subscription_tasks[ticker_to_cancel]
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error waiting for cancelled task {ticker_to_cancel}: {e}")
            del self.market_subscription_tasks[ticker_to_cancel]

            if ticker_to_cancel in self._ws_connections:
                try:
                    await self._ws_connections[ticker_to_cancel].close()
                except Exception as e:
                    logger.error(f"Error closing WS for {ticker_to_cancel} during full cancellation: {e}")
                del self._ws_connections[ticker_to_cancel]

    async def start_adaptive_market_tracking(self, event_ticker: Optional[str] = None):
        self.event_ticker = event_ticker or self._generate_current_event_id()
        logger.info(f"🚀 KalshiClient starting adaptive market tracking for event: {self.event_ticker}")

        await self._update_market_subscriptions_adaptive()
        if not self.active_market_info:
            logger.error("No active markets identified after initial setup. Exiting adaptive tracking.")
           # return

        last_update_time = time.time()
        while not self._shutdown_requested:
            current_time = time.time()
            if current_time - last_update_time >= self.update_interval:
                await self._update_market_subscriptions_adaptive()
                last_update_time = current_time
            await asyncio.sleep(.1)

        logger.info("🛑 Adaptive market tracking stopped.")
        await self.close()

    def stop_adaptive_market_tracking(self):
        logger.info("Shutdown requested for adaptive market tracking.")
        self._shutdown_requested = True

class BTCPriceMonitor:
    def __init__(self, price_file: str = "data/unified_crypto_data.json"):
        self.price_file = Path(price_file)
        self.last_price = None
        self.last_modified = None
        self.last_check = 0
        self.check_interval = 0.05
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
            price = data.get("brti", {}).get("price")
            if price and isinstance(price, (int, float)) and price > 0:
                new_price = float(price)
                if new_price != self.last_price:
                    self.price_history.append((now, new_price))
                    self._cleanup_price_history(now)
                self.last_price = new_price
                self.last_modified = current_modified
                return self.last_price
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Error reading BTC price file: {e}")
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

async def main():
    client = KalshiClient()
    try:
        tracking_task = asyncio.create_task(client.start_adaptive_market_tracking())

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')

            if client.active_market_info:
                display_lines = []
                display_lines.append(f"----- Active Markets (BTC: ${client.last_btc_price:,.2f} | Vol: {client.current_volatility:.2f}) -----")
                sorted_markets_info = sorted(client.active_market_info.values(), key=lambda m: m.strike)
                strike_labels = []
                for market_info in sorted_markets_info:
                    label = f"${market_info.strike:,.0f}🎯" if market_info.is_primary else f"${market_info.strike:,.0f}"
                    strike_labels.append(label)
                if strike_labels:
                    display_lines.append(f"Ladder: {' | '.join(strike_labels)}")
                for market_info in sorted_markets_info:
                    data = market_info.market_data
                    if data:
                        primary_indicator = "🎯" if market_info.is_primary else " "
                        if data.yes_bid is not None and data.yes_ask is not None:
                            yes_prices = f"YES: {data.yes_bid:.0f}/{data.yes_ask:.0f}"
                            no_bid, no_ask = 100 - data.yes_ask, 100 - data.yes_bid
                            no_prices = f"NO: {no_bid:.0f}/{no_ask:.0f}"
                            spread = data.yes_ask - data.yes_bid
                            spread_text = f"Spread: {spread:.0f}¢"
                        else:
                            yes_prices = "YES: --/--"
                            no_prices = "NO: --/--"
                            spread_text = "No data"
                        line = (f"{primary_indicator}${market_info.strike:,.0f}: "
                                f"{yes_prices} | {no_prices} | {spread_text}")
                        display_lines.append(line)
                print("\n".join(display_lines))
            else:
                print("No active markets currently being tracked.")

            await asyncio.sleep(.05)

    except KeyboardInterrupt:
        logger.info("\nCtrl+C detected. Stopping adaptive market tracking and cleaning up...")
    except Exception as e:
        logger.error(f"\n💥 Fatal error in main: {e}")
    finally:
        client.stop_adaptive_market_tracking()
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Program terminated by user. Goodbye!")
    except Exception as e:
        logger.critical(f"\n💥 Startup error: {e}")
        sys.exit(1)
