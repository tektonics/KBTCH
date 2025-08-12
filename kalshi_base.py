import os
import base64
import time
from pathlib import Path
from typing import Dict
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from config.kalshiconfig import (
    REST_API_BASE_URL,
    KALSHI_API_KEY_ID, 
    KALSHI_PRIVATE_KEY_PATH,
    REQUEST_HEADERS
)

class KalshiAPIClient:
    """Base class for Kalshi API authentication and common functionality"""
    
    def __init__(self):
        self.base_url = REST_API_BASE_URL
        self.key_id = KALSHI_API_KEY_ID
        self.private_key_path = KALSHI_PRIVATE_KEY_PATH
        
        if not self.key_id or not self.private_key_path:
            raise ValueError("Missing required environment variables")
            
        self.private_key = self._load_private_key()
    
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

