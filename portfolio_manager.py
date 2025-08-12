import requests
import time
import logging
from typing import Dict, Any
from kalshi_base import KalshiAPIClient
from config.kalshiconfig import API_ENDPOINTS

class PortfolioManager(KalshiAPIClient):
    def __init__(self):
        super().__init__() 

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
