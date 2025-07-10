import os
import base64
import time
import json
from typing import Optional, Dict, Any
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


class KalshiTradeAPI:
    """Minimal Kalshi Trade API client for executing orders."""

    def __init__(self, use_demo: bool = False):
        self.key_id = os.getenv("KALSHI_API_KEY")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

        if not self.key_id or not self.private_key_path:
            raise ValueError(
                "KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH environment variables are required"
            )

        self.private_key = self._load_private_key()
        if use_demo:
            self.base_url = "https://demo-api.kalshi.co/trade-api/v2"
        else:
            self.base_url = "https://api.elections.kalshi.com/trade-api/v2"

    def _load_private_key(self):
        key_path = Path(self.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key not found: {self.private_key_path}")
        with open(key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        message = f"{timestamp}{method}{path}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(timestamp, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        headers = self._headers(method, path)
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, json=data)
        else:
            raise ValueError("Unsupported HTTP method")

        response.raise_for_status()
        if not response.text:
            return {}
        return response.json()

    def get_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/balance")

    def create_order(self, market_ticker: str, side: str, quantity: int, price: float) -> Dict[str, Any]:
        data = {
            "type": "limit",
            "side": side,
            "market_ticker": market_ticker,
            "size": quantity,
            "price": price,
        }
        return self._request("POST", "/orders", data)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/orders/{order_id}")
