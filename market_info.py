# market_info.py

from dataclasses import dataclass
from typing import Optional
from kalshi_bot.kalshi_client import TickerData  # Adjust import if path is different

@dataclass
class TradingParams:
    min_edge_threshold: float
    max_spread_threshold: float

@dataclass
class MarketInfo:
    ticker: str
    strike: float
    market_data: Optional[TickerData] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    implied_prob: Optional[float] = None
    edge: Optional[float] = None
    action: Optional[str] = None
