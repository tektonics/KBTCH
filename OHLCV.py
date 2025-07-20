import os
import sys
import asyncio
import ccxt.pro
import json
import time
import threading
from pathlib import Path
from datetime import datetime

class OHLCVManager:
    def __init__(self, output_file: str = "ohlcv_data.json"):
        self.output_file = Path(output_file)
        self.data_lock = threading.Lock()
        self.exchange_data = {}
        self.initialize_json_file()
    
    def initialize_json_file(self):
        try:
            initial_data = {
                "timestamp": time.time(),
                "last_updated": datetime.now().isoformat(),
                "exchanges": {},
                "status": "initializing"
            }
            with self.data_lock:
                with open(self.output_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
        except Exception as e:
            print(f"Failed to initialize OHLCV JSON file: {e}")
    
    def update_exchange_data(self, exchange_id: str, symbol: str, ohlcv_data: list):
        if not ohlcv_data:
            return
        latest_candle = ohlcv_data[-1]
        timestamp, open_price, high, low, close, volume = latest_candle
        
        exchange_info = {
            "symbol": symbol,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "last_updated": time.time()
        }
        
        self.exchange_data[exchange_id] = exchange_info
        self.write_to_json()
    
    def write_to_json(self):
        try:
            output_data = {
                "timestamp": time.time(),
                "last_updated": datetime.now().isoformat(),
                "exchanges": self.exchange_data.copy(),
                "status": "active"
            }
            with self.data_lock:
                with open(self.output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
        except Exception as e:
            print(f"Failed to write OHLCV data to JSON: {e}")

ohlcv_manager = OHLCVManager()

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')


async def fetch_ohlcv_continuously(exchange, symbol):
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol)
            ohlcv_length = len(ohlcv)
            if ohlcv_length > 0:
                print('Fetched ', exchange.id, ' - ', symbol, ' candles. last candle: ', ohlcv[ohlcv_length - 1])
                ohlcv_manager.update_exchange_data(exchange.id, symbol, ohlcv)
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
