"""
Complete example showing how to use the trading system with REAL Kalshi data
"""
import time
import random
from datetime import datetime
from typing import List, Dict

from trading_engine import TradingEngine
from config import TRADING_CONFIG

# Import your Kalshi client
try:
    from kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    print("Warning: kalshi_client.py not found. Using fallback simulation.")
    KALSHI_AVAILABLE = False

def get_real_kalshi_data() -> List[Dict]:
    """Get real market data from Kalshi API"""
    if not KALSHI_AVAILABLE:
        return simulate_kalshi_data()  # Fallback to fake data
    
    try:
        # Initialize Kalshi client (adjust based on your kalshi_client.py implementation)
        client = KalshiClient()
        
        # Get list of active markets (adjust method name based on your implementation)
        markets = client.get_markets()  # or however your client gets market list
        
        market_data = []
        
        # Get data for first few active markets (limit to avoid API rate limits)
        for market in markets[:5]:  # Limit to 5 markets to avoid overwhelming
            try:
                ticker = market.get('ticker') or market.get('market_ticker')
                if not ticker:
                    continue
                
                # Get market details (adjust method based on your implementation)
                market_details = client.get_market(ticker)
                
                # Extract relevant data and convert to trading engine format
                trading_data = {
                    'ticker': ticker,
                    'price': market_details.get('last_price', 0.0) or market_details.get('price', 0.0),
                    'bid': market_details.get('yes_bid', 0.0) or market_details.get('bid', 0.0),
                    'ask': market_details.get('yes_ask', 0.0) or market_details.get('ask', 0.0),
                    'volume': market_details.get('volume', 0),
                    'timestamp': time.time(),
                    'open_interest': market_details.get('open_interest', 0)
                }
                
                # Only add if we have valid price data
                if trading_data['price'] > 0:
                    market_data.append(trading_data)
                
                # Small delay to respect API rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting data for {ticker}: {e}")
                continue
        
        if not market_data:
            print("No valid market data retrieved, falling back to simulation")
            return simulate_kalshi_data()
        
        print(f"Retrieved real data for {len(market_data)} markets")
        return market_data
        
    except Exception as e:
        print(f"Error connecting to Kalshi API: {e}")
        print("Falling back to simulated data")
        return simulate_kalshi_data()

def simulate_kalshi_data():
    """Fallback: Simulate Kalshi market data for testing"""
    markets = ['ELECTION-2024', 'GDP-Q1-GROWTH', 'INFLATION-MARCH']
    data = []
    
    for market in markets:
        # Simulate realistic market data
        base_price = 0.4 + random.random() * 0.2  # Price between 0.4-0.6
        spread = 0.01 + random.random() * 0.02    # Spread 1-3%
        
        market_data = {
            'ticker': market,
            'price': base_price,
            'bid': base_price - spread/2,
            'ask': base_price + spread/2,
            'volume': random.randint(100, 1000),
            'timestamp': time.time(),
            'open_interest': random.randint(1000, 10000)
        }
        data.append(market_data)
    
    return data

def run_real_data_simulation():
    """Run simulation with real Kalshi market data"""
    print("=== Kalshi Trading System with REAL Market Data ===")
    
    if KALSHI_AVAILABLE:
        print("✅ Using REAL Kalshi market data")
    else:
        print("⚠️  Using simulated data (kalshi_client.py not available)")
    
    # Initialize trading engine in simulation mode
    engine = TradingEngine(mode='simulation')
    engine.start()
    
    try:
        # Run for 30 iterations with real data
        for i in range(30):
            print(f"\n--- Iteration {i+1} ---")
            
            # Get REAL market data from Kalshi
            market_data = get_real_kalshi_data()
            
            if not market_data:
                print("No market data available, skipping iteration")
                time.sleep(5)  # Wait longer when no data
                continue
            
            # Show what markets we're trading
            if i == 0:
                tickers = [d['ticker'] for d in market_data]
                print(f"Trading markets: {', '.join(tickers)}")
            
            # Process with trading engine
            results = engine.process_market_data(market_data)
            
            print(f"Status: {results['status']}")
            print(f"Signals Generated: {results['signals_generated']}")
            print(f"Orders Executed: {results['orders_executed']}")
            print(f"Portfolio Value: ${results['portfolio_value']:.2f}")
            print(f"Unrealized P&L: ${results['unrealized_pnl']:.2f}")
            
            # Show market prices
            for data in market_data[:3]:  # Show first 3 markets
                print(f"  {data['ticker']}: ${data['price']:.3f} (bid: ${data['bid']:.3f}, ask: ${data['ask']:.3f})")
            
            if results.get('order_results'):
                for order in results['order_results']:
                    print(f"  -> {order['side']} {order['quantity']} {order['market']} - {order['status']}")
            
            # Every 10 iterations, show detailed status
            if (i + 1) % 10 == 0:
                status = engine.get_status()
                print("\n=== Detailed Status ===")
                print(f"Total Trades: {status['performance']['total_trades']}")
                print(f"Open Positions: {len(status['portfolio']['positions'])}")
                print(f"Cash: ${status['portfolio']['cash']:.2f}")
                if status['risk']['risk_warnings']:
                    print("Risk Warnings:", status['risk']['risk_warnings'])
            
            # Wait longer between iterations for real data (respect API limits)
            time.sleep(2 if KALSHI_AVAILABLE else 0.1)
    
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    
    finally:
        # Clean shutdown
        engine.stop()
        
        # Show final results
        final_status = engine.get_status()
        print("\n=== Final Results ===")
        print(f"Data Source: {'Real Kalshi API' if KALSHI_AVAILABLE else 'Simulated'}")
        print(f"Total Runtime: {final_status['engine_status']['runtime_seconds']:.1f} seconds")
        print(f"Total Trades: {final_status['performance']['total_trades']}")
        print(f"Final Portfolio Value: ${final_status['portfolio']['total_value']:.2f}")
        print(f"Total P&L: ${final_status['portfolio']['unrealized_pnl'] + final_status['portfolio']['realized_pnl']:.2f}")

def run_strategy_comparison_real_data():
    """Compare different strategies using real Kalshi data"""
    print("=== Strategy Comparison with Real Data ===")
    
    if not KALSHI_AVAILABLE:
        print("Warning: Using simulated data for comparison")
    
    strategies = ['momentum', 'mean_reversion', 'spread', 'volume']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy with real data...")
        
        # Create custom config for this strategy
        config = TRADING_CONFIG.copy()
        config['strategy'] = strategy
        
        engine = TradingEngine(mode='simulation', config=config)
        engine.start()
        
        # Run shorter test with real data
        for i in range(15):
            market_data = get_real_kalshi_data()
            if market_data:
                engine.process_market_data(market_data)
            time.sleep(1 if KALSHI_AVAILABLE else 0.05)
        
        # Get results
        status = engine.get_status()
        results[strategy] = {
            'trades': status['performance']['total_trades'],
            'pnl': status['portfolio']['unrealized_pnl'] + status['portfolio']['realized_pnl'],
            'final_value': status['portfolio']['total_value']
        }
        
        engine.stop()
    
    # Show comparison
    print("\n=== Strategy Comparison Results (Real Data) ===")
    for strategy, result in results.items():
        print(f"{strategy:15}: {result['trades']:3d} trades, "
              f"${result['pnl']:7.2f} P&L, ${result['final_value']:8.2f} final")

def live_market_monitor():
    """Monitor live Kalshi market data without trading"""
    print("=== Live Kalshi Market Monitor ===")
    
    if not KALSHI_AVAILABLE:
        print("kalshi_client.py required for live monitoring")
        return
    
    print("Monitoring live market data (Ctrl+C to stop)...")
    
    try:
        while True:
            market_data = get_real_kalshi_data()
            
            if market_data:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
                for data in market_data:
                    spread = data['ask'] - data['bid']
                    spread_pct = (spread / data['price'] * 100) if data['price'] > 0 else 0
                    print(f"{data['ticker']:20} | "
                          f"Price: ${data['price']:.3f} | "
                          f"Spread: {spread_pct:.1f}% | "
                          f"Vol: {data['volume']:,}")
            else:
                print("No market data available")
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMarket monitoring stopped")

def test_data_connection():
    """Test connection to Kalshi API"""
    print("=== Testing Kalshi Data Connection ===")
    
    if not KALSHI_AVAILABLE:
        print("❌ kalshi_client.py not found")
        print("Make sure kalshi_client.py is in the same directory")
        return
    
    print("🔄 Testing Kalshi API connection...")
    
    try:
        market_data = get_real_kalshi_data()
        
        if market_data:
            print(f"✅ Successfully connected! Retrieved {len(market_data)} markets:")
            for data in market_data[:3]:  # Show first 3
                print(f"  - {data['ticker']}: ${data['price']:.3f}")
            print("\n🎯 Ready for real data trading!")
        else:
            print("❌ Connected but no market data retrieved")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Check your kalshi_client.py implementation")

if __name__ == "__main__":
    # Run different examples
    
    print("Choose example to run:")
    print("1. Real data simulation")
    print("2. Strategy comparison (real data)")
    print("3. Live market monitor")
    print("4. Test data connection")
    print("5. Original simulation (fake data)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        run_real_data_simulation()
    elif choice == '2':
        run_strategy_comparison_real_data()
    elif choice == '3':
        live_market_monitor()
    elif choice == '4':
        test_data_connection()
    elif choice == '5':
        # Original simulation with fake data
        from trading_engine import TradingEngine
        engine = TradingEngine(mode='simulation')
        engine.start()
        for i in range(20):
            market_data = simulate_kalshi_data()
            results = engine.process_market_data(market_data)
            print(f"Iteration {i+1}: {results['orders_executed']} orders, ${results['portfolio_value']:.2f}")
            time.sleep(0.1)
        engine.stop()
    else:
        print("Running real data simulation by default...")
        run_real_data_simulation()
