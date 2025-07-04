import os
import base64
import datetime
import requests
import asyncio
import json
import time
import websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

load_dotenv()

class KalshiClient:
    def __init__(self):
        self.base_url = "https://api.elections.kalshi.com"
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.key_id = os.getenv("KALSHI_API_KEY")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.private_key = self.load_private_key()

        # Stores the latest ticker data keyed by market ticker
        self.mid_prices = {}
        self.orderbooks = {}

    def load_private_key(self):
        with open(self.private_key_path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(), password=None, backend=default_backend()
            )

    def generate_signature(self, timestamp, method, path):
        message = f"{timestamp}{method}{path}".encode("utf-8")
        path = "/trade-api/ws/v2"
        signature = self.private_key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def build_auth_headers(self, method, path):
        timestamp = str(int(time.time() * 1000))
        signature = self.generate_signature(timestamp,method, path)

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def get_markets(self, event_ticker):
        method = "GET"
        path = "/trade-api/v2/markets"
        params = {"event_ticker": event_ticker}
        headers = self.build_auth_headers(method, path)
        url = self.base_url + path
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    async def subscribe_to_market(self, market_ticker: str):
        path = "/trade-api/ws/v2"
        method = "GET"
        headers = self.build_auth_headers(method, path)

        async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
            subscribe_message = {
		"id": 1,
		"cmd": "subscribe",
		"params": {
                "channels": ["ticker_v2", "orderbook_delta"],
		"market_ticker": market_ticker
   			 }
            }
            await ws.send(json.dumps(subscribe_message))
            print(f"[Kalshi] Subscribed to {market_ticker}")

            while True:
                raw = await ws.recv()
                message = json.loads(raw)
                self.handle_ws_message(message)

    def handle_ws_message(self, msg):
        msg_type = msg.get("type")
        data = msg.get("msg", {})

        if msg_type == "ticker_v2":
            market_ticker = data.get("market_ticker")
            if market_ticker:
                self.mid_prices[market_ticker] = {
                    "price": data.get("price"),
                    "yes_bid": data.get("yes_bid"),
                    "yes_ask": data.get("yes_ask"),
                    "no_bid": data.get("no_bid"),
                    "no_ask": data.get("no_ask"),
                    "volume_delta": data.get("volume_delta"),
                }
        elif msg_type == "orderbook_delta":
            market_ticker = data.get("market_ticker")
            if market_ticker:
                self.orderbooks[market_ticker] = data
            else:
                print("[Warning] orderbook_delta missing market_ticker:", data)
        else:
            print(f"[Info] Unhandled message type: {msg_type} | Full message: {msg}")


    def get_mid_prices(self, market_ticker):
        return self.mid_prices.get(market_ticker)

    def get_orderbook(self, market_ticker):
        return self.orderbooks.get(market_ticker)

