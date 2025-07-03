import sys
import time
import os
from btcag import PriceAggregator, PriceData

class PriceDisplay:
    
    def __init__(self):
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.BLUE = '\033[94m'
        self.YELLOW = '\033[93m'
        self.CYAN = '\033[96m'
        self.MAGENTA = '\033[95m'
        self.WHITE = '\033[97m'
        self.RESET = '\033[0m'
        self.BOLD = '\033[1m'
        self.DIM = '\033[2m'
        
        self.CLEAR_SCREEN = '\033[2J'
        self.HOME = '\033[H'
        self.HIDE_CURSOR = '\033[?25l'
        self.SHOW_CURSOR = '\033[?25h'
        
        self.last_display_time = 0
        self.display_throttle = 0.1
        self.header_lines = 9
        self.dynamic_lines = 9
        
        self.exchange_order = ['coinbase', 'kraken', 'bitstamp', 'gemini', 'paxos', 'crypto.com']
        self.exchange_config = {
            'coinbase': {'label': 'Coinbase', 'color': self.BLUE, 'short': 'CB'},
            'kraken': {'label': 'Kraken  ', 'color': self.YELLOW, 'short': 'KR'},
            'bitstamp': {'label': 'Bitstamp', 'color': self.MAGENTA, 'short': 'BS'},
            'gemini' : {'label': 'Gemini  ', 'color': self.CYAN, 'short': 'GM'},
            'paxos': {'label': 'Paxos   ', 'color': self.WHITE, 'short': 'PX'},
            'crypto.com': {'label': 'Crypto  ', 'color': self.GREEN, 'short': 'CD'},
#            'lmax': {'label': 'LMAX', 'color': self.CYAN, 'short': 'LM'},
        }
        
        self.aggregator = PriceAggregator()
        self.aggregator.add_price_callback(self._on_price_update)
        
        self.last_aggregate_price = None
        self.display_initialized = False
    
    def _move_to_line(self, line_number):
        return f'\033[{line_number};1H'
    
    def _clear_line(self):
        return '\033[2K'
    
    def _get_price_color_and_arrow(self, current_price: float, last_price: float = None):
        if last_price is not None:
            if current_price > last_price:
                return self.GREEN, "↗"
            elif current_price < last_price:
                return self.RED, "↘"
            else:
                return self.RESET, "→"
        return self.RESET, "○"
    
    def _format_price_change(self, current_price: float, last_price: float = None):
        if last_price is not None and last_price != 0:
            change = current_price - last_price
            change_percent = (change / last_price) * 100
            return f"({change:+.2f}, {change_percent:+.2f}%)"
        return ""
    
    def _format_exchange_line(self, exchange: str, data: PriceData):
        config = self.exchange_config.get(exchange, {
            'label': exchange.capitalize()[:8].ljust(8),
            'color': self.RESET,
            'short': exchange.upper()[:2]
        })
        
        price_color, arrow = self._get_price_color_and_arrow(data.price, data.last_price)
        change_str = self._format_price_change(data.price, data.last_price)
        
        exchange_part = f"{config['color']}{config['label']}{self.RESET}"
        price_part = f"{price_color}${data.price:>10,.2f} {arrow}{self.RESET}"
        
        line = f"  {exchange_part}: {price_part}"
        
        if change_str:
            line += f" {price_color}{change_str:>18}{self.RESET}"
        else:
            line += " " * 20
        
        if data.volume:
            volume_part = f"{self.DIM}[Vol: {data.volume:>8.4f}]{self.RESET}"
            line += f" {volume_part}"
        
        if data.side:
            side_color = self.GREEN if data.side == "BUY" else self.RED
            side_part = f"{side_color}{data.side:>4}{self.RESET}"
            line += f" {side_part}"
        
        return line
    
    def _format_aggregate_line(self):
        aggregate_price = self.aggregator.get_aggregate_price()
        spread_data = self.aggregator.get_price_spread()
        
        if aggregate_price is None:
            return f"{self.CYAN}{self.BOLD}AGGREGATE: {self.RESET}Calculating..."
        
        agg_color, agg_arrow = self._get_price_color_and_arrow(
            aggregate_price, self.last_aggregate_price
        )
        agg_change_str = self._format_price_change(
            aggregate_price, self.last_aggregate_price
        )
        
        self.last_aggregate_price = aggregate_price
        
        label_part = f"{self.CYAN}{self.BOLD}AGGREGATE:{self.RESET}"
        price_part = f"{agg_color}${aggregate_price:>10,.2f} {agg_arrow}{self.RESET}"
        
        line = f"  {label_part} {price_part}"
        
        if agg_change_str:
            line += f" {agg_color}{agg_change_str:>18}{self.RESET}"
        else:
            line += " " * 20
        
        if spread_data and 'absolute' in spread_data:
            spread_abs = spread_data['absolute']
            spread_percent = spread_data['percent']
            
            if spread_percent < 0.01:  
                spread_color = self.GREEN
            elif spread_percent < 0.05:  
                spread_color = self.YELLOW
            else: 
                spread_color = self.RED
            
            spread_part = f"{spread_color}[Spread: ${spread_abs:.2f} ({spread_percent:.3f}%)]{self.RESET}"
            line += f" {spread_part}"
        
        return line
    
    def _format_status_line(self):
        current_prices = self.aggregator.get_current_prices()
        connected_count = len(current_prices)
        total_count = len(self.exchange_order)
        
        status_color = self.GREEN if connected_count == total_count else self.YELLOW
        status_part = f"{status_color}Connected: {connected_count}/{total_count}{self.RESET}"
        
        current_time = time.strftime("%H:%M:%S")
        time_part = f"{self.DIM}[{current_time}]{self.RESET}"
        
        active_exchanges = []
        for exchange in self.exchange_order:
            if exchange in current_prices:
                config = self.exchange_config[exchange]
                active_exchanges.append(f"{config['color']}{config['short']}{self.RESET}")
        
        active_part = f"Active: {' '.join(active_exchanges)}" if active_exchanges else "Active: None"
        
        return f"  {status_part} | {active_part} | {time_part}"
    
    def _update_display(self):
        current_prices = self.aggregator.get_current_prices()
        
        current_line = self.header_lines + 1
        
        aggregate_line = self._format_aggregate_line()
        print(f"{self._move_to_line(current_line)}{self._clear_line()}{aggregate_line}", end='', flush=True)
        current_line += 1
        
        print(f"{self._move_to_line(current_line)}{self._clear_line()}  {'-' * 80}", end='', flush=True)
        current_line += 1
        
        for exchange in self.exchange_order:
            if exchange in current_prices:
                exchange_line = self._format_exchange_line(exchange, current_prices[exchange])
                print(f"{self._move_to_line(current_line)}{self._clear_line()}{exchange_line}", end='', flush=True)
            else:
                config = self.exchange_config[exchange]
                disconnected_line = f"  {config['color']}{config['label']}{self.RESET}: {self.DIM}Connecting...{self.RESET}"
                print(f"{self._move_to_line(current_line)}{self._clear_line()}{disconnected_line}", end='', flush=True)
            current_line += 1
        
        status_line = self._format_status_line()
        print(f"{self._move_to_line(current_line)}{self._clear_line()}{status_line}", end='', flush=True)
    
    def _on_price_update(self, exchange: str, price_data: PriceData):
        current_time = time.time()
        
        if current_time - self.last_display_time < self.display_throttle:
            return
        
        self.last_display_time = current_time
        
        if self.display_initialized:
            self._update_display()
    
    def _print_static_header(self):
        print(f"{self.CLEAR_SCREEN}{self.HOME}{self.HIDE_CURSOR}", end='')
        print(f"{self.BOLD}{self.CYAN}╔{'═' * 78}╗{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}║{' ' * 20}Multi-Exchange BTC Price Aggregator{' ' * 26}║{self.RESET}")
        print(f"{self.BOLD}{self.CYAN}╚{'═' * 78}╝{self.RESET}")
        print()
        
        exchange_info = []
        for exchange in self.exchange_order:
            config = self.exchange_config[exchange]
            exchange_info.append(f"{config['color']}{config['label']} ({config['short']}){self.RESET}")
        
        print(f"  {self.DIM}Press Ctrl+C to exit{self.RESET}")
        print()
        print("=" * 80)
        
        for _ in range(self.dynamic_lines):
            print()
    
    def start(self):
        self._print_static_header()
        self.display_initialized = True
        
        self.aggregator.start()
        
        self._update_display()
        
        try:
            while True:
                if not self.aggregator.is_running():
                    print(f"\n{self.RED}Error: Aggregator stopped running{self.RESET}")
                    break
                
                time.sleep(0.5)  
                
        except KeyboardInterrupt:
            self._stop()
    
    def _stop(self):
        print(f"{self.SHOW_CURSOR}")  
        print(f"\n\n{self.YELLOW}Shutting down...{self.RESET}")
        self.aggregator.stop()
        print(f"{self.GREEN}Goodbye!{self.RESET}")
        sys.exit(0)

if __name__ == "__main__":
    display = PriceDisplay()
    display.start()
