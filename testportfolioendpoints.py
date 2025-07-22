# Test script to check which endpoints work
from data.kalshi_client import KalshiClient

def test_all_endpoints():
    client = KalshiClient()
    
    endpoints_to_test = [
        ("Balance", client.get_balance),
        ("Positions", client.get_positions), 
        ("Orders", client.get_orders),
        ("Fills", client.get_fills)
    ]
    
    for name, endpoint_func in endpoints_to_test:
        try:
            result = endpoint_func()
            print(f"✅ {name}: SUCCESS - {result}")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")

if __name__ == "__main__":
    test_all_endpoints()
