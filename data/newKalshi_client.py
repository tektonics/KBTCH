import sys
import os
import base64
import time
import json
import asyncio
import logging
import bisect
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
import websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
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

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

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

    def get_price_color(self) -> str:
        if self.price is None or self.previous_price is None:
            return 'gray'
        elif self.price > self.previous_price:
            return 'green'
        elif self.price < self.previous_price:
            return 'red'
        else:
            return 'gray'

class KalshiClient:
    def __init__(self):
        # Use configuration values instead of hardcoded URLs
        self.base_url = REST_API_BASE_URL
        self.ws_url = WEBSOCKET_API_URL
        
        # Use configuration environment variable names
        self.key_id = KALSHI_API_KEY_ID
        self.private_key_path = KALSHI_PRIVATE_KEY_PATH

        if not self.key_id or not self.private_key_path:
            raise ValueError("Missing required environment variables: KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH")

        self.private_key = self._load_private_key()

        self.mid_prices: Dict[str, TickerData] = {}
        self.orderbooks: Dict[str, Dict] = {}
        self._ws_connection = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1

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
                    salt_length=padding.PSS.MAX_LENGTH
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
                time.sleep(1)

    async def subscribe_to_market(self, market_ticker: str):
        while True:
            try:
                await self._connect_and_subscribe(market_ticker)
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
                    logger.info(f"Reconnecting in {delay} seconds... (attempt {self._reconnect_attempts})")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max reconnection attempts reached. Giving up.")
                    break

    async def _connect_and_subscribe(self, market_ticker: str):
        # Use WebSocket path from URL parsing
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
            self._ws_connection = ws
            self._reconnect_attempts = 0

            subscribe_message = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": [WS_CHANNELS["TICKER_V2"], WS_CHANNELS["ORDERBOOK_DELTA"]],
                    "market_ticker": market_ticker
                }
            }
            await ws.send(json.dumps(subscribe_message))
            print(f"[Kalshi] Subscribed to {market_ticker}")

            async for message in ws:
                try:
                    data = json.loads(message)
                    asyncio.create_task(self._handle_ws_message_async(data))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")

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
            self._handle_ws_message(data)  # This calls your existing method
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
        if self._ws_connection:
            await self._ws_connection.close()
            logger.info("WebSocket connection closed")
