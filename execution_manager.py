import time
import json
import logging
import requests
from typing import Dict, Any, Optional
from kalshi_base import KalshiAPIClient
from config.kalshiconfig import API_ENDPOINTS
from event_bus import event_bus, EventTypes
from order_manager import OrderStatus

logger = logging.getLogger(__name__)


class ExecutionManager(KalshiAPIClient):
    """Handles actual order execution via Kalshi API"""
    
    def __init__(self, order_manager):
        super().__init__()
        self.order_manager = order_manager
        
        # Execution settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.request_timeout = 10
        
        # Statistics
        self.orders_sent = 0
        self.orders_failed = 0
        self.api_calls_made = 0
        self.api_failures = 0
        
        # Subscribe to order events
        event_bus.subscribe(EventTypes.ORDER_PLACED, self._handle_order_placed)
        event_bus.subscribe(EventTypes.ORDER_CANCELLED, self._handle_order_cancelled)
        
        logger.info("Execution Manager initialized and subscribed to order events")
    
    def _handle_order_placed(self, event) -> None:
        """Handle order placement requests"""
        try:
            event_data = event.data
            client_order_id = event_data.get("client_order_id")
            
            if not client_order_id:
                logger.error("No client_order_id in order placed event")
                return
            
            # Get order from order manager
            order = self.order_manager.get_order(client_order_id)
            if not order:
                logger.error(f"Order not found in order manager: {client_order_id}")
                return
            
            # Convert to Kalshi API format and execute
            api_order = self.order_manager.to_kalshi_api_format(order)
            success, response = self._execute_order(api_order)
            
            if success:
                # Update order status to working
                kalshi_order_id = response.get("order", {}).get("order_id")
                self.order_manager.update_order_status(
                    client_order_id, 
                    OrderStatus.WORKING,
                    kalshi_order_id=kalshi_order_id
                )
                self.orders_sent += 1
                logger.info(f"✅ Order executed successfully: {client_order_id[:8]} → Kalshi ID: {kalshi_order_id}")
            else:
                # Mark order as rejected
                self.order_manager.update_order_status(client_order_id, OrderStatus.REJECTED)
                self.orders_failed += 1
                logger.error(f"❌ Order execution failed: {client_order_id[:8]}")
                
        except Exception as e:
            logger.error(f"Error handling order placement: {e}")
    
    def _handle_order_cancelled(self, event) -> None:
        """Handle order cancellation requests"""
        try:
            event_data = event.data
            client_order_id = event_data.get("client_order_id")
            kalshi_order_id = event_data.get("kalshi_order_id")
            
            if not kalshi_order_id:
                logger.warning(f"Cannot cancel order without Kalshi order ID: {client_order_id}")
                return
            
            success, response = self._cancel_order(kalshi_order_id)
            
            if success:
                logger.info(f"✅ Order cancelled successfully: {client_order_id[:8]}")
            else:
                logger.error(f"❌ Order cancellation failed: {client_order_id[:8]}")
                
        except Exception as e:
            logger.error(f"Error handling order cancellation: {e}")
    
    def _execute_order(self, api_order: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Execute order via Kalshi API"""
        method = "POST"
        path = API_ENDPOINTS["CREATE_ORDER"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        
        for attempt in range(self.max_retries):
            try:
                self.api_calls_made += 1
                
                logger.debug(f"Sending order (attempt {attempt + 1}): {api_order}")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=api_order,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200 or response.status_code == 201:
                    response_data = response.json()
                    logger.debug(f"Order API response: {response_data}")
                    return True, response_data
                else:
                    logger.warning(f"Order API returned status {response.status_code}: {response.text}")
                    
                    # Don't retry on client errors (400-499)
                    if 400 <= response.status_code < 500:
                        self.api_failures += 1
                        return False, {"error": f"Client error: {response.status_code}"}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Order API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.api_failures += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        return False, {"error": "Max retries exceeded"}
    
    def _cancel_order(self, kalshi_order_id: str) -> tuple[bool, Dict[str, Any]]:
        """Cancel order via Kalshi API"""
        method = "DELETE"
        path = f"{API_ENDPOINTS['CANCEL_ORDER']}/{kalshi_order_id}"
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        
        for attempt in range(self.max_retries):
            try:
                self.api_calls_made += 1
                
                response = requests.delete(
                    url,
                    headers=headers,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return True, response_data
                else:
                    logger.warning(f"Cancel API returned status {response.status_code}: {response.text}")
                    
                    # Don't retry on client errors
                    if 400 <= response.status_code < 500:
                        self.api_failures += 1
                        return False, {"error": f"Client error: {response.status_code}"}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Cancel API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.api_failures += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                
        return False, {"error": "Max retries exceeded"}
    
    def check_order_status(self, kalshi_order_id: str) -> Optional[Dict[str, Any]]:
        """Check order status via Kalshi API"""
        method = "GET"
        path = API_ENDPOINTS["GET_ORDERS"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        
        params = {"order_id": kalshi_order_id}
        
        try:
            self.api_calls_made += 1
            
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Order status API returned status {response.status_code}: {response.text}")
                self.api_failures += 1
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Order status check failed: {e}")
            self.api_failures += 1
            return None
    
    def amend_order(self, kalshi_order_id: str, new_price: int) -> tuple[bool, Dict[str, Any]]:
        """Amend order price via Kalshi API"""
        method = "POST"
        path = API_ENDPOINTS["AMEND_ORDER"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        
        amend_data = {
            "order_id": kalshi_order_id,
            "price": new_price
        }
        
        for attempt in range(self.max_retries):
            try:
                self.api_calls_made += 1
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=amend_data,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return True, response_data
                else:
                    logger.warning(f"Amend API returned status {response.status_code}: {response.text}")
                    
                    if 400 <= response.status_code < 500:
                        self.api_failures += 1
                        return False, {"error": f"Client error: {response.status_code}"}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Amend API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.api_failures += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                
        return False, {"error": "Max retries exceeded"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution manager status"""
        success_rate = 0.0
        if self.api_calls_made > 0:
            success_rate = ((self.api_calls_made - self.api_failures) / self.api_calls_made) * 100
        
        execution_success_rate = 0.0
        total_orders = self.orders_sent + self.orders_failed
        if total_orders > 0:
            execution_success_rate = (self.orders_sent / total_orders) * 100
        
        return {
            "orders_sent": self.orders_sent,
            "orders_failed": self.orders_failed,
            "execution_success_rate": f"{execution_success_rate:.1f}%",
            "api_calls_made": self.api_calls_made,
            "api_failures": self.api_failures,
            "api_success_rate": f"{success_rate:.1f}%",
            "settings": {
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "request_timeout": self.request_timeout
            }
        }
    
    def batch_create_orders(self, orders: list[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """Create multiple orders via Kalshi batch API"""
        method = "POST"
        path = API_ENDPOINTS["BATCH_CREATE_ORDERS"]
        headers = self._build_auth_headers(method, path)
        url = self.base_url + path
        
        batch_data = {"orders": orders}
        
        for attempt in range(self.max_retries):
            try:
                self.api_calls_made += 1
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=batch_data,
                    timeout=self.request_timeout * 2  # Longer timeout for batch
                )
                
                if response.status_code == 200 or response.status_code == 201:
                    response_data = response.json()
                    return True, response_data
                else:
                    logger.warning(f"Batch create API returned status {response.status_code}: {response.text}")
                    
                    if 400 <= response.status_code < 500:
                        self.api_failures += 1
                        return False, {"error": f"Client error: {response.status_code}"}
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Batch create API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.api_failures += 1
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                
        return False, {"error": "Max retries exceeded"}
