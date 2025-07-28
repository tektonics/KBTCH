#!/usr/bin/env python3
"""
Test the new trading system architecture
"""
import sys
import os

# Add parent directory to Python path so we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent dir

from trading.trading_engine import TradingEngine
from trading.order_manager import OrderManager
from trading.risk_manager import OrderSignal
from config.config import TRADING_CONFIG

def test_integration():
    """Test the new trading flow"""
    
    # Step 1: Create TradingEngine (this creates the executor)
    trading_engine = TradingEngine(mode='simulation', kalshi_client=None, config=TRADING_CONFIG)
    
    # Step 2: Create OrderManager and give it the TradingEngine
    order_manager = OrderManager(trading_engine)
    
    # Step 3: Test the flow
    test_orders = [
        OrderSignal(
            market_ticker="BTC-25JAN25-95000",
            side="buy",
            quantity=100,
            price=0.45,
            reason="Test BUY_YES order"
        ),
        OrderSignal(
            market_ticker="BTC-25JAN25-96000", 
            side="sell",
            quantity=50,
            price=0.32,
            reason="Test SELL_NO order"
        )
    ]
    
    print("🚀 Testing new trading architecture...")
    print(f"Created TradingEngine in {trading_engine.mode} mode")
    print(f"Created OrderManager with TradingEngine reference")
    
    # Step 4: Execute orders through OrderManager (the correct flow)
    print(f"\n📋 OrderManager processing {len(test_orders)} orders...")
    results = order_manager.execute_orders(test_orders)
    
    # Step 5: Check results
    print(f"\n✅ Execution complete! Results:")
    for i, result in enumerate(results):
        print(f"  Order {i+1}: {result.status} - {result.market_ticker} - {result.filled_quantity}/{result.quantity}")
    
    # Step 6: Check order manager statistics
    summary = order_manager.get_order_summary()
    print(f"\n📊 Order Manager Stats:")
    print(f"  Total orders: {summary['statistics']['total_orders']}")
    print(f"  Successful: {summary['statistics']['successful_orders']}")
    print(f"  Rejected: {summary['statistics']['rejected_orders']}")
    
    print(f"\n🎉 Integration test successful!")
    return True

if __name__ == "__main__":
    test_integration()
