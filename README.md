# KBTCH
Kalshi BTC hourly

## Trading Scripts

- `trading_bot.py` runs the live trader using the production Kalshi API. It relies on
  environment variables `KALSHI_API_KEY` and `KALSHI_PRIVATE_KEY_PATH` to sign requests.
- `demo_trading_bot.py` is identical but uses Kalshi's demo environment. Both scripts
  leverage `kalshirunner.py` for market data and signal generation.

The trading REST client is implemented in `kalshi_bot/kalshi_trade_api.py`.

