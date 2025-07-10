from trading_bot import LiveTrader
import asyncio


def main():
    trader = LiveTrader(use_demo=True)
    asyncio.run(trader.run_trading_loop())


if __name__ == "__main__":
    main()

