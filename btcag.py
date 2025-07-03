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
        self.PAXOS_WS_URL = "wss://ws.paxos.com/marketdata/BTCUSD"
        self.CRYPTO_COM_WS_URL = "wss://stream.crypto.com/exchange/v1/market"
        self.LMAX_WS_URL = "wss://public-data-api.london-digital.lmax.com/v1/web-socket"
        self.exchange_data: Dict[str, PriceData] = {}
        self.price_lock = threading.Lock()
        
        self.price_callbacks: list[Callable] = []
        
        self.coinbase_ws = None
        self.kraken_ws = None
        self.bitstamp_ws = None
        self.gemini_ws = None
        self.paxos_ws = None
        self.crypto_ws = None
        self.lmax_ws = None
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

    def _on_paxos_open(self, ws):
        print("Connected to Paxos WebSocket")

    def _on_paxos_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'UPDATE':
                price = float(data['price'])
                volume = float(data['amount'])
                side = data['side'].upper()
                self._update_price('paxos', price, volume=volume, side=side)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def _on_paxos_error(self, ws, error):
        print(f"Paxos WebSocket Error: {error}")

    def _on_paxos_close(self, ws, close_status_code=None, close_msg=None):
        print("Paxos WebSocket connection closed")

    def _on_crypto_open(self, ws):
        time.sleep(1)
        subscribe_message = {
            "id": 1,
            "method": "subscribe",
            "params": {
                "channels": ["trade.BTCUSD-PERP"]
            },
            "nonce": int(time.time() * 1000)
        }
        ws.send(json.dumps(subscribe_message))

    def _on_crypto_message(self, ws, message):
        try:
            data = json.loads(message)
            payload = data
            if 'result' in data and isinstance(data['result'], dict):
                payload = data['result']
            if data.get('method') == 'public/heartbeat':
                ws.send(json.dumps({
                    "id": data['id'],
                    "method": "public/respond-heartbeat"
                }))
            elif payload.get('channel') == 'trade' and 'data' in payload:
                trades = payload['data']
                if trades:
                    latest_trade = trades[-1]
                    price = float(latest_trade['p'])
                    volume = float(latest_trade['q'])
                    side = latest_trade['s']
                    self._update_price('crypto.com', price, volume=volume, side=side)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass

    def _on_crypto_error(self, ws, error):
        print(f"Crypto.com WebSocket Error: {error}")

    def _on_crypto_close(self, ws, close_status_code=None, close_msg=None):
        print("Crypto.com WebSocket connection closed")

    def _on_lmax_open(self, ws):
        print("Connected to LMAX Digital WebSocket")
        subscribe_message = {
            "type": "SUBSCRIBE",
            "channels": [
                {
                    "name":"ORDER_BOOK",
                    "instruments": ["btc-usd"]
                }
            ]
        }
        ws.send(json.dumps(subscribe_message))

    def _on_lmax_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'TICKER':
                price = float(data['last_price'])
                volume = float(data['last_quantity'])
                self._update_price('lmax', price, volume=volume, side=None)
            elif data.get('type') == 'SUBSCRIPTIONS':
                pass
            elif data.get('type') == 'ERROR':
                print(f"LMAX Subscription Error: {data.get('error_code')} - {data.get('error_message')}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass

    def _on_lmax_error(self, ws, error):
        print(f"LMAX Digital WebSocket Error: {error}")

    def _on_lmax_close(self, ws, close_status_code=None, close_msg=None):
        print("LMAX Digital WebSocket connection closed")

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

    def _start_paxos_ws(self):
        self.paxos_ws = websocket.WebSocketApp(
            self.PAXOS_WS_URL,
            on_open=self._on_paxos_open,
            on_message=self._on_paxos_message,
            on_error=self._on_paxos_error,
            on_close=self._on_paxos_close
        )
        self.paxos_ws.run_forever()

    def _start_crypto_com_ws(self):
        self.crypto_ws = websocket.WebSocketApp(
            self.CRYPTO_COM_WS_URL,
            on_open=self._on_crypto_open,
            on_message=self._on_crypto_message,
            on_error=self._on_crypto_error,
            on_close=self._on_crypto_close
        )
        self.crypto_ws.run_forever(ping_interval=30, ping_timeout=5, sslopt={"cert_reqs": ssl.CERT_NONE})

    def _start_lmax_ws(self):
        self.lmax_ws = websocket.WebSocketApp(
            self.LMAX_WS_URL,
            on_open=self._on_lmax_open,
            on_message=self._on_lmax_message,
            on_error=self._on_lmax_error,
            on_close=self._on_lmax_close
        )
        self.lmax_ws.run_forever(ping_interval=3, ping_timeout=1, sslopt={"cert_reqs": ssl.CERT_NONE})

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
        paxos_thread = threading.Thread(target=self._start_paxos_ws, daemon=True)
        crypto_com_thread = threading.Thread(target=self._start_crypto_com_ws, daemon=True)
        lmax_thread = threading.Thread(target=self._start_lmax_ws, daemon=True)
        self.threads = [coinbase_thread, kraken_thread, bitstamp_thread, gemini_thread, paxos_thread, crypto_com_thread, lmax_thread]
        
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
        if self.paxos_ws:
            self.paxos_ws.close()
        if self.crypto_ws:
            self.crypto_ws.close()
        if self.lmax_ws:
            self.lmax_ws.close()
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

