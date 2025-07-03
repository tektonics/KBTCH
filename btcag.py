import websocket
import json
import threading
import time
import ssl
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from datetime import datetime

@dataclass
class PriceData:
    price: float
    timestamp: float
    exchange: str
    last_price: Optional[float] = None
    volume: Optional[float] = None
    side: Optional[str] = None

class PriceAggregator:
    
    def __init__(self):
        self.COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
        self.KRAKEN_WS_URL = "wss://ws.kraken.com"
        self.BITSTAMP_WS_URL = "wss://ws.bitstamp.net"
        self.GEMINI_WS_URL = "wss://api.gemini.com/v2/marketdata"
        self.BULLISH_WS_URL = "wss://api.exchange.bullish.com/trading-api/v1/market-data/trades"
        self.exchange_data: Dict[str, PriceData] = {}
        self.price_lock = threading.Lock()
        
        self.price_callbacks: list[Callable] = []
        
        self.coinbase_ws = None
        self.kraken_ws = None
        self.bitstamp_ws = None
        self.gemini_ws = None
        self.bullish_ws = None

        self.running = False
        self.threads = []
    
    def add_price_callback(self, callback: Callable):
        self.price_callbacks.append(callback)
    
    def get_current_prices(self) -> Dict[str, PriceData]:
        with self.price_lock:
            return self.exchange_data.copy()
    
    def get_aggregate_price(self) -> Optional[float]:
        with self.price_lock:
            prices = [data.price for data in self.exchange_data.values() if data.price is not None]
            if prices:
                return sum(prices) / len(prices)
            return None
    
    def get_price_spread(self) -> Dict[str, float]:
        prices = self.get_current_prices()
        if len(prices) < 2:
            return {}
        
        price_values = [data.price for data in prices.values()]
        min_price = min(price_values)
        max_price = max(price_values)
        avg_price = sum(price_values) / len(price_values)
        
        spread_abs = max_price - min_price
        spread_percent = (spread_abs / avg_price) * 100 if avg_price > 0 else 0
        
        return {
            'absolute': spread_abs,
            'percent': spread_percent,
            'min_price': min_price,
            'max_price': max_price
        }
    
    def _update_price(self, exchange: str, price: float, **kwargs):
        with self.price_lock:
            last_price = None
            if exchange in self.exchange_data:
                last_price = self.exchange_data[exchange].price
            
            self.exchange_data[exchange] = PriceData(
                price=price,
                timestamp=time.time(),
                exchange=exchange,
                last_price=last_price,
                **kwargs
            )
        
        for callback in self.price_callbacks:
            try:
                callback(exchange, self.exchange_data[exchange])
            except Exception as e:
                print(f"Error in price callback: {e}")
    
    def _on_coinbase_open(self, ws):
        subscribe_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": "market_trades"
        }
        ws.send(json.dumps(subscribe_message))

    
    def _on_coinbase_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data.get('channel') == 'market_trades' and 'events' in data:
                events = data['events']
                
                for event in events:
                    if 'trades' in event:
                        trades = event['trades']
                        
                        for trade in trades:
                            price = float(trade['price'])
                            volume = float(trade['size'])
                            side = trade['side']
                            
                            self._update_price('coinbase', price, volume=volume, side=side)
                            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass  
    
    def _on_coinbase_error(self, ws, error):
        print(f"Coinbase WebSocket Error: {error}")
    
    def _on_coinbase_close(self, ws, close_status_code=None, close_msg=None):
        print("Coinbase WebSocket connection closed")
    
    def _on_kraken_open(self, ws):
        subscribe_message = {
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "trade"}
        }
        ws.send(json.dumps(subscribe_message))

    
    def _on_kraken_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if isinstance(data, list) and len(data) >= 4:
                if data[2] == "trade":
                    trades = data[1]
                    
                    for trade in trades:
                        if isinstance(trade, list) and len(trade) >= 3:
                            price = float(trade[0])
                            volume = float(trade[1])
                            side = "BUY" if trade[3] == "b" else "SELL"
                            
                            self._update_price('kraken', price, volume=volume, side=side)
                            
        except (json.JSONDecodeError, KeyError, ValueError, IndexError):
            pass 
    
    def _on_kraken_error(self, ws, error):
        print(f"Kraken WebSocket Error: {error}")
    
    def _on_kraken_close(self, ws, close_status_code=None, close_msg=None):
        print("Kraken WebSocket connection closed")
    
    def _on_bitstamp_open(self, ws):
        subscribe_message = {
            "event": "bts:subscribe",
            "data": {
                "channel": "live_trades_btcusd"
            }
        }
        ws.send(json.dumps(subscribe_message))

    
    def _on_bitstamp_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data.get('event') == 'trade' and 'data' in data:
                trade_data = data['data']
                price = float(trade_data['price'])
                volume = float(trade_data['amount'])
                side = "BUY" if trade_data['type'] == 0 else "SELL"
                
                self._update_price('bitstamp', price, volume=volume, side=side)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass  
    
    def _on_bitstamp_error(self, ws, error):
        print(f"Bitstamp WebSocket Error: {error}")
    
    def _on_bitstamp_close(self, ws, close_status_code=None, close_msg=None):
        print("Bitstamp WebSocket connection closed")
    
    def _on_gemini_open(self, ws):
        subscribe_message = {
            "type": "subscribe",
            "subscriptions": [
                {"name": "l2", "symbols": ["BTCUSD"]} 
            ]
        }
        ws.send(json.dumps(subscribe_message))
        print("Connected to Gemini WebSocket")

    def _on_gemini_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'trade':
                price = float(data['price']) 
                volume = float(data['quantity']) 
                side = data['side'].upper() 
                self._update_price('gemini', price, volume=volume, side=side)
            elif data.get('type') == 'l2_updates' and 'trades' in data:
                for trade in data['trades']:
                    price = float(trade['price']) 
                    volume = float(trade['quantity']) 
                    side = trade['side'].upper()
                    self._update_price('gemini', price, volume=volume, side=side) 

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass

    def _on_gemini_error(self, ws, error):
        print(f"Gemini WebSocket Error: {error}")

    def _on_gemini_close(self, ws, close_status_code=None, close_msg=None):
        print("Gemini WebSocket connection closed")

    def _on_bullish_open(self, ws):
        subscribe_message = {
            "jsonrpc": "2.0",
            "type": "command",
            "method": "subscribe",
            "params": {
                "topic": "anonymousTrades",
                "symbol": "BTCUSDC" 
            },
            "id": str(int(time.time() * 1000)) 
        }
        ws.send(json.dumps(subscribe_message))
        print("Connected to Bullish WebSocket")

    def _on_bullish_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') in ['snapshot', 'update'] and data.get('dataType') == 'V1TAAnonymousTradeUpdate': 
                trades = data['data']['trades']
                for trade in trades:
                    price = float(trade['price'])
                    volume = float(trade['quantity'])
                    side = trade['side'] 
                    self._update_price('bullish', price, volume=volume, side=side)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass

    def _on_bullish_error(self, ws, error):
        print(f"Bullish WebSocket Error: {error}")

    def _on_bullish_close(self, ws, close_status_code=None, close_msg=None):
        print("Bullish WebSocket connection closed")

    def _start_coinbase_ws(self):
        self.coinbase_ws = websocket.WebSocketApp(
            self.COINBASE_WS_URL,
            on_open=self._on_coinbase_open,
            on_message=self._on_coinbase_message,
            on_error=self._on_coinbase_error,
            on_close=self._on_coinbase_close
        )
        self.coinbase_ws.run_forever()
    
    def _start_kraken_ws(self):
        self.kraken_ws = websocket.WebSocketApp(
            self.KRAKEN_WS_URL,
            on_open=self._on_kraken_open,
            on_message=self._on_kraken_message,
            on_error=self._on_kraken_error,
            on_close=self._on_kraken_close
        )
        self.kraken_ws.run_forever()
    
    def _start_bitstamp_ws(self):
        self.bitstamp_ws = websocket.WebSocketApp(
            self.BITSTAMP_WS_URL,
            on_open=self._on_bitstamp_open,
            on_message=self._on_bitstamp_message,
            on_error=self._on_bitstamp_error,
            on_close=self._on_bitstamp_close
        )
        self.bitstamp_ws.run_forever()
    
    def _start_gemini_ws(self):
        import ssl 
        self.gemini_ws = websocket.WebSocketApp(
            self.GEMINI_WS_URL,
            on_open=self._on_gemini_open,
            on_message=self._on_gemini_message,
            on_error=self._on_gemini_error,
            on_close=self._on_gemini_close
        )
        self.gemini_ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def _start_bullish_ws(self):
        def on_open(ws): self._on_bullish_open(ws)
        def on_message(ws, msg): self._on_bullish_message(ws, msg)
        def on_error(ws, err): 
            print(f"[Bullish ERROR] {err}")
            self._on_bullish_error(ws, err)
        def on_close(ws, code, msg): 
            print(f"[Bullish CLOSE] code={code}, msg={msg}")
            self._on_bullish_close(ws, code, msg)

        self.bullish_ws = websocket.WebSocketApp(
            self.BULLISH_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            header=[
                "User-Agent: Mozilla/5.0 (X11; Linux x86_64)",
                "Origin: https://bullish.com"
            ]
        )
        self.bullish_ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def start(self):
        if self.running:
            print("Aggregator is already running")
            return
        
        self.running = True
        print("Starting price aggregator...")
        
        coinbase_thread = threading.Thread(target=self._start_coinbase_ws, daemon=True)
        kraken_thread = threading.Thread(target=self._start_kraken_ws, daemon=True)
        bitstamp_thread = threading.Thread(target=self._start_bitstamp_ws, daemon=True)
        gemini_thread = threading.Thread(target=self._start_gemini_ws, daemon=True)
        bullish_thread = threading.Thread(target=self._start_bullish_ws, daemon=True)
        self.threads = [coinbase_thread, kraken_thread, bitstamp_thread, gemini_thread, bullish_thread]
        
        for thread in self.threads:
            thread.start()
        
        print("All WebSocket connections started")
    
    def stop(self):
        self.running = False
        
        if self.coinbase_ws:
            self.coinbase_ws.close()
        if self.kraken_ws:
            self.kraken_ws.close()
        if self.bitstamp_ws:
            self.bitstamp_ws.close()
        if self.gemini_ws:
            self.gemini_ws.close()
        if self.bullish_ws:
            self.bullish_ws.close()
        print("Price aggregator stopped")
    
    def is_running(self):
        return self.running and any(thread.is_alive() for thread in self.threads)

if __name__ == "__main__":
    def price_update_callback(exchange: str, price_data: PriceData):
        print(f"{exchange}: ${price_data.price:.2f} (Volume: {price_data.volume:.4f})")
    
    aggregator = PriceAggregator()
    aggregator.add_price_callback(price_update_callback)
    aggregator.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping aggregator...")
        aggregator.stop()

