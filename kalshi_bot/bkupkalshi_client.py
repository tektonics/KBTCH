import os
import base64
import time
import json
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
import websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ensure output goes to terminal
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class TickerData:
    price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    volume_delta: Optional[int] = None
    timestamp: Optional[float] = None

class KalshiClient:
    def __init__(self):
        self.base_url = "https://api.elections.kalshi.com"
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.key_id = os.getenv("KALSHI_API_KEY")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        if not self.key_id or not self.private_key_path:
            raise ValueError("Missing required environment variables: KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH")
        
        self.private_key = self._load_private_key()
        
        # Thread-safe storage for market data
        self.mid_prices: Dict[str, TickerData] = {}
        self.orderbooks: Dict[str, Dict] = {}
        self._ws_connection = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1  # seconds

    def _load_private_key(self):
        """Load private key with better error handling"""
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
        """Generate signature with proper error handling"""
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
        """Build authentication headers"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def get_markets(self, event_ticker: str) -> Dict[str, Any]:
        """Get markets for an event with retry logic"""
        method = "GET"
        path = "/trade-api/v2/markets"
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
                time.sleep(1)  # Wait before retry

    async def subscribe_to_market(self, market_ticker: str):
        """Subscribe to market with automatic reconnection"""
        while True:
            try:
                await self._connect_and_subscribe(market_ticker)
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))  # Exponential backoff
                    logger.info(f"Reconnecting in {delay} seconds... (attempt {self._reconnect_attempts})")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max reconnection attempts reached. Giving up.")
                    break

    async def _connect_and_subscribe(self, market_ticker: str):
        """Internal method to handle WebSocket connection and subscription"""
        path = "/trade-api/ws/v2"
        method = "GET"
        headers = self._build_auth_headers(method, path)

        async with websockets.connect(
            self.ws_url, 
            additional_headers=headers,
            ping_interval=30,  # Keep connection alive
            ping_timeout=10
        ) as ws:
            self._ws_connection = ws
            self._reconnect_attempts = 0  # Reset counter on successful connection
            
            subscribe_message = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker_v2", "orderbook_delta"],
                    "market_ticker": market_ticker
                }
            }
            
            await ws.send(json.dumps(subscribe_message))
            print(f"[Kalshi] Subscribed to {market_ticker}")  # Use print for immediate visibility

            async for message in ws:
                try:
                    data = json.loads(message)
                    self._handle_ws_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")

    def _handle_ws_message(self, msg: Dict[str, Any]):
        """Handle incoming WebSocket messages with better error handling"""
        msg_type = msg.get("type")
        data = msg.get("msg", {})

        if msg_type == "ticker_v2":
            market_ticker = data.get("market_ticker")
            if market_ticker:
                self.mid_prices[market_ticker] = TickerData(
                    price=data.get("price"),
                    yes_bid=data.get("yes_bid"),
                    yes_ask=data.get("yes_ask"),
                    no_bid=data.get("no_bid"),
                    no_ask=data.get("no_ask"),
                    volume_delta=data.get("volume_delta"),
                    timestamp=time.time()
                )
            else:
                logger.warning("ticker_v2 message missing market_ticker")
                
        elif msg_type == "orderbook_delta":
            market_ticker = data.get("market_ticker")
            if market_ticker:
                self.orderbooks[market_ticker] = data
            else:
                logger.warning(f"orderbook_delta missing market_ticker: {data}")
        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    def get_mid_prices(self, market_ticker: str) -> Optional[TickerData]:
        """Get mid prices for a market"""
        return self.mid_prices.get(market_ticker)

    def get_orderbook(self, market_ticker: str) -> Optional[Dict]:
        """Get orderbook for a market"""
        return self.orderbooks.get(market_ticker)

    def get_all_market_data(self) -> Dict[str, Dict]:
        """Get all market data at once"""
        return {
            "mid_prices": {k: v.__dict__ for k, v in self.mid_prices.items()},
            "orderbooks": self.orderbooks.copy()
        }

    async def close(self):
        """Clean up resources"""
        if self._ws_connection:
            await self._ws_connection.close()
            logger.info("WebSocket connection closed")
