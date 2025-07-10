import asyncio
import time
from typing import Dict

from kalshirunner import VolatilityAdaptiveTrader
from kalshi_bot.kalshi_trade_api import KalshiTradeAPI


class LiveTrader(VolatilityAdaptiveTrader):
    """VolatilityAdaptiveTrader with trade execution using Kalshi REST API."""

    def __init__(self, event_id: str = None, use_demo: bool = False, order_size: int = 1):
        super().__init__(event_id)
        self.api = KalshiTradeAPI(use_demo=use_demo)
        self.order_size = order_size
        self.last_trade: Dict[str, float] = {}

    def _should_trade(self, ticker: str) -> bool:
        last = self.last_trade.get(ticker, 0)
        return time.time() - last > 30

    def _mark_traded(self, ticker: str):
        self.last_trade[ticker] = time.time()

    async def run_trading_loop(self):
        if not await self.initialize():
            return

        try:
            while not self.shutdown_requested:
                if not self.brti_manager.is_brti_running():
                    await self.brti_manager.start_brti()

                current_btc = self.btc_monitor.get_current_price()
                if current_btc and current_btc != self.last_btc_price:
                    self.last_btc_price = current_btc
                    self.btc_updates += 1

                new_volatility = self.btc_monitor.calculate_volatility()
                if abs(new_volatility - self.current_volatility) > 0.1:
                    self.current_volatility = new_volatility
                    self.volatility_updates += 1
                    await self._update_market_subscriptions()

                for market in self.active_markets:
                    market.market_data = self.client.get_mid_prices(market.ticker)
                    if market.market_data:
                        market = self.analyzer.analyze_market_opportunity(
                            market, self.last_btc_price, self.current_volatility
                        )
                        if market.action in ["BUY_YES", "SELL_YES"] and self._should_trade(market.ticker):
                            side = "buy" if market.action == "BUY_YES" else "sell"
                            price = market.market_data.yes_ask if side == "buy" else market.market_data.yes_bid
                            try:
                                resp = self.api.create_order(
                                    market.ticker, side, self.order_size, price
                                )
                                print(f"Placed {side} order on {market.ticker} @ {price}: {resp.get('status','ok')}")
                                self._mark_traded(market.ticker)
                            except Exception as e:
                                print(f"Order failed: {e}")

                lines = self.display.format_market_display(
                    self.active_markets, self.last_btc_price,
                    self.current_volatility, self.brti_manager.is_brti_running()
                )
                self.display.update_multiline_display(lines)

                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            self.display.print_new_line("\n🛑 Shutting down...")
        finally:
            await self._cleanup()


def main():
    trader = LiveTrader()
    asyncio.run(trader.run_trading_loop())


if __name__ == "__main__":
    main()

