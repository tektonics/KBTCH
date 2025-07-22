import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Environment Configuration (defaults to demo)
KALSHI_ENVIRONMENT = os.getenv("KALSHI_ENV", "demo").lower()

# Base URLs for REST API and WebSockets
if KALSHI_ENVIRONMENT == "production":
    REST_API_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WEBSOCKET_API_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
else:
    REST_API_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
    WEBSOCKET_API_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

# API Key Management
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")

# REST API Endpoints (paths relative to REST_API_BASE_URL)
API_ENDPOINTS = {
    "GET_BALANCE": "/portfolio/balance",
    "GET_FILLS": "/portfolio/fills",
    "GET_ORDERS": "/portfolio/orders",
    "CREATE_ORDER": "/portfolio/orders", #used to buy and sell
    "BATCH_CREATE_ORDERS": "/portfolio/batch-create-orders",
    "CANCEL_ORDER": "/portfolio/orders",
    "BATCH_CANCEL_ORDERS": "/portfolio/batch-cancel-orders",
    "AMEND_ORDER": "/portfolio/amend-order",
    "DECREASE_ORDER": "/portfolio/decrease-order",
    "GET_POSITIONS": "/portfolio/positions",
    "GET_MARKETS": "/markets",
    "GET_MARKET_ORDERBOOK": "/markets/{market_ticker}/orderbook",
    "GET_TRADES": "/trades",
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
