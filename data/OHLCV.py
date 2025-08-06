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
    def __init__(self, output_file: str = "data/ohlcv_data.json"):
        self.output_file = Path(output_file)
        self.data_lock = threading.Lock()
        self.exchange_history = {}
        self.rsi_period = 14
        self.volume_baseline_periods = 20
        self.momentum_periods = 5
        self.initialize_json_file()
    
    def initialize_json_file(self):
        try:
            initial_data = {
                "timestamp": time.time(),
                "last_updated": datetime.now().isoformat(),
                "analysis": {
                    "volume_spikes": [],
                    "rsi": 50,
                    "momentum": "→",
                    "avg_price": 0
                },
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
        
        # Initialize exchange history if needed
        if exchange_id not in self.exchange_history:
            self.exchange_history[exchange_id] = {
                'prices': [],
                'volumes': [],
                'timestamps': []
            }
        
        history = self.exchange_history[exchange_id]
        
        # Add new data point
        history['prices'].append(close)
        history['volumes'].append(volume)
        history['timestamps'].append(timestamp)
        
        # Keep only needed history (max 50 periods)
        max_history = 50
        if len(history['prices']) > max_history:
            history['prices'] = history['prices'][-max_history:]
            history['volumes'] = history['volumes'][-max_history:]
            history['timestamps'] = history['timestamps'][-max_history:]
        
        # Calculate and write metrics
        self.calculate_and_write_metrics()
    
    def calculate_volume_spikes(self):
        """Calculate volume spikes for each exchange"""
        exchange_codes = {
            'coinbase': 'CB', 'kraken': 'KR', 'bitstamp': 'BS', 
            'gemini': 'GM', 'cryptocom': 'CC', 'okx': 'OKX'
        }
        
        volume_spikes = []
        
        for exchange_id, history in self.exchange_history.items():
            volumes = history['volumes']
            if len(volumes) < self.volume_baseline_periods:
                continue
                
            # Calculate volume spike
            recent_volume = volumes[-1]
            baseline_volume = sum(volumes[-self.volume_baseline_periods:-1]) / (self.volume_baseline_periods - 1)
            
            if baseline_volume > 0:
                spike_pct = ((recent_volume - baseline_volume) / baseline_volume) * 100
                if spike_pct > 50:  # Only show significant spikes
                    code = exchange_codes.get(exchange_id, exchange_id[:2].upper())
                    volume_spikes.append(f"{code}(+{spike_pct:.0f}%)")
        
        return volume_spikes
    
    def calculate_rsi(self):
        """Calculate RSI using all exchange data"""
        all_prices = []
        
        # Combine prices from all exchanges, sorted by timestamp
        for exchange_id, history in self.exchange_history.items():
            for i, price in enumerate(history['prices']):
                timestamp = history['timestamps'][i]
                all_prices.append((timestamp, price))
        
        if len(all_prices) < self.rsi_period + 1:
            return 50  # Neutral RSI
        
        # Sort by timestamp and get price changes
        all_prices.sort(key=lambda x: x[0])
        prices = [p[1] for p in all_prices]
        
        # Calculate price changes
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        if len(price_changes) < self.rsi_period:
            return 50
        
        # Use most recent period for RSI
        recent_changes = price_changes[-self.rsi_period:]
        
        gains = [change for change in recent_changes if change > 0]
        losses = [-change for change in recent_changes if change < 0]
        
        avg_gain = sum(gains) / len(recent_changes)
        avg_loss = sum(losses) / len(recent_changes)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self):
        """Calculate momentum indicator"""
        all_prices = []
        
        # Combine prices from all exchanges
        for exchange_id, history in self.exchange_history.items():
            for i, price in enumerate(history['prices']):
                timestamp = history['timestamps'][i]
                all_prices.append((timestamp, price))
        
        if len(all_prices) < self.momentum_periods:
            return "→"
        
        # Sort by timestamp
        all_prices.sort(key=lambda x: x[0])
        prices = [p[1] for p in all_prices]
        
        # Calculate momentum
        recent_price = prices[-1]
        past_price = prices[-self.momentum_periods]
        momentum_pct = ((recent_price - past_price) / past_price) * 100
        
        if momentum_pct > 2:
            return "↑↑"
        elif momentum_pct > 0.5:
            return "↑"
        elif momentum_pct < -2:
            return "↓↓"
        elif momentum_pct < -0.5:
            return "↓"
        else:
            return "→"
    
    def calculate_average_price(self):
        """Calculate current average price across exchanges"""
        current_prices = []
        
        for exchange_id, history in self.exchange_history.items():
            if history['prices']:
                current_prices.append(history['prices'][-1])
        
        if current_prices:
            return sum(current_prices) / len(current_prices)
        return 0
    
    def calculate_and_write_metrics(self):
        """Calculate all metrics and write to JSON"""
        try:
            volume_spikes = self.calculate_volume_spikes()
            rsi = self.calculate_rsi()
            momentum = self.calculate_momentum()
            avg_price = self.calculate_average_price()
            
            output_data = {
                "timestamp": time.time(),
                "last_updated": datetime.now().isoformat(),
                "analysis": {
                    "volume_spikes": volume_spikes,
                    "rsi": round(rsi, 0),
                    "momentum": momentum,
                    "avg_price": round(avg_price, 2)
                },
                "status": "active"
            }
            
            with self.data_lock:
                with open(self.output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                    
        except Exception as e:
            print(f"Failed to calculate/write OHLCV metrics: {e}")

ohlcv_manager = OHLCVManager()

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
    exchanges = ['gemini', 'bitstamp', 'coinbase','okx', 'kraken', 'cryptocom']
    symbols = ['BTC/USD']
    promises = []
    for i in range(0, len(exchanges)):
        exchange_name = exchanges[i]
        promises.append(start_exchange(exchange_name, symbols))
    await asyncio.gather(*promises)


asyncio.run(example())
