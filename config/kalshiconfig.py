import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Environment Configuration (defaults to demo)
KALSHI_ENVIRONMENT = os.getenv("KALSHI_ENV", "production").lower()

# Base URLs for REST API and WebSockets
if KALSHI_ENVIRONMENT == "production":
    REST_API_BASE_URL = "https://api.elections.kalshi.com"
    WEBSOCKET_API_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
else:
    REST_API_BASE_URL = "https://demo-api.kalshi.co"
    WEBSOCKET_API_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

# API Key Management
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# REST API Endpoints (paths relative to REST_API_BASE_URL)
API_ENDPOINTS = {
    "GET_BALANCE": "/trade-api/v2/portfolio/balance",  # Full path
    "GET_FILLS": "/trade-api/v2/portfolio/fills",
    "GET_ORDERS": "/trade-api/v2/portfolio/orders",
    "CREATE_ORDER": "/trade-api/v2/portfolio/orders",
    "BATCH_CREATE_ORDERS": "/trade-api/v2/portfolio/batch-create-orders",
    "CANCEL_ORDER": "/trade-api/v2/portfolio/orders",
    "BATCH_CANCEL_ORDERS": "/trade-api/v2/portfolio/batch-cancel-orders",
    "AMEND_ORDER": "/trade-api/v2/portfolio/amend-order",
    "DECREASE_ORDER": "/trade-api/v2/portfolio/decrease-order",
    "GET_POSITIONS": "/trade-api/v2/portfolio/positions",
    "GET_MARKETS": "/trade-api/v2/markets",
    "GET_MARKET_ORDERBOOK": "/trade-api/v2/markets/{market_ticker}/orderbook",
    "GET_TRADES": "/trade-api/v2/trades",
}

# WebSocket Channels for real-time data
WS_CHANNELS = {
    "ORDERBOOK_DELTA": "orderbook_delta",
    "TICKER_V2": "ticker_v2",
    "TRADE": "trade",
    "FILL": "fill", # Private channel for your account's fills
    "MARKET_LIFECYCLE_V2": "market_lifecycle_v2",
    "MULTIVARIATE": "multivariate",
}

# Standard Request Headers
REQUEST_HEADERS = {
    "Content-Type": "application/json"
}

# Note: Authentication headers (KALSHI-ACCESS-KEY, -TIMESTAMP, -SIGNATURE)
# must be dynamically generated for each request using your private key.
