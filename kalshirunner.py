# kalshirunner.py

from kalshi_bot.kalshi_client import KalshiClient
from trading_logic import apply_trading_logic_to_markets
from display import DisplayManager
from portfolio import Portfolio
from risk_manager import RiskManager
from strategy import Strategy
from config import TRADING_CONFIG

import time

class KalshiRunner:
    def __init__(self):
        self.client = KalshiClient()
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager(self.portfolio)
        self.strategy = Strategy()
        self.display = DisplayManager()
        self.event_id = self.client.get_latest_btc_event_id()
        self.tickers = self.client.get_markets(self.event_id)
        self.market_infos = self._init_market_infos()

    def _init_market_infos(self):
        market_infos = []
        for ticker in self.tickers:
            market_data = self.client.get_mid_prices(ticker)
            strike = self._extract_strike_from_ticker(ticker)
            market_infos.append({
                'ticker': ticker,
                'market_data': market_data,
                'strike': strike
            })
        return market_infos

    def _extract_strike_from_ticker(self, ticker: str) -> float:
        # Example format: 'KXBTCD-25JUL2117-67000' => strike = 67000
        try:
            return float(ticker.split("-")[-1])
        except Exception:
            return 0.0

    def run(self):
        while True:
            btc_price = self.client.get_latest_btc_price()
            for market in self.market_infos:
                market['market_data'] = self.client.get_mid_prices(market['ticker'])

            decisions = apply_trading_logic_to_markets(
                self.market_infos, btc_price, self.portfolio, self.risk_manager, self.strategy
            )

            self.display.update(decisions)
            time.sleep(5)


if __name__ == "__main__":
    runner = KalshiRunner()
    runner.run()
