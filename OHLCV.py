import os
import sys
import asyncio
import ccxt.pro

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')


async def fetch_ohlcv_continuously(exchange, symbol):
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol)
            ohlcv_length = len(ohlcv)
            print('Fetched ', exchange.id, ' - ', symbol, ' candles. last candle: ', ohlcv[ohlcv_length - 1])
        except Exception as e:
            print(e)
            break



async def start_exchange(exchange_name, symbols):
    ex = getattr(ccxt.pro, exchange_name)({})
    promises = []
    for i in range(0, len(symbols)):
        symbol = symbols[i]
        promises.append(fetch_ohlcv_continuously(ex, symbol))
    await asyncio.gather(*promises)
    await ex.close()


async def example():
    exchanges = ['gemini', 'bitstamp', 'coinbase','okx', 'kraken']
    symbols = ['BTC/USD']
    promises = []
    for i in range(0, len(exchanges)):
        exchange_name = exchanges[i]
        promises.append(start_exchange(exchange_name, symbols))
    await asyncio.gather(*promises)


asyncio.run(example())
