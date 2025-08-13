#!/usr/bin/env python3
"""
Papertrading Setup Script for KBTCH Trading System
Helps users configure and test the papertrading functionality.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🎮 KBTCH TRADING SYSTEM - PAPERTRADING SETUP")
    print("=" * 70)
    print("This script will help you set up papertrading mode for safe testing.")
    print("In papertrading mode, all trades are simulated locally - no real money is used!")
    print("Your existing .env file will remain unchanged.\n")

def check_dependencies():
    """Check if required files exist"""
    print("📋 Checking system dependencies...")
    
    required_files = [
        "main.py",
        "config/config_manager.py", 
        "simulated_portfolio_manager.py",
        "simulated_execution_manager.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all papertrading files are in place.")
        return False
    
    print("✅ All required files found!")
    return True

def create_trading_config():
    """Create trading configuration file"""
    print("\n🔧 Setting up trading configuration...")
    
    config_file = Path("trading_config.json")
    
    if config_file.exists():
        print(f"📁 Found existing {config_file}")
        overwrite = input("Overwrite existing configuration? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("✅ Keeping existing configuration")
            return True
    
    # Get papertrading preferences
    print("\n💰 Papertrading Configuration:")
    
    initial_balance = input("Starting balance in USD (default: $1000): ").strip()
    if initial_balance:
        try:
            balance_usd = float(initial_balance)
        except ValueError:
            print("Invalid balance, using default $1000")
            balance_usd = 1000
    else:
        balance_usd = 1000
    
    fill_probability = input("Order fill probability 0-1 (default: 0.95): ").strip()
    if fill_probability:
        try:
            fill_prob = float(fill_probability)
            if not 0 <= fill_prob <= 1:
                print("Fill probability must be 0-1, using default 0.95")
                fill_prob = 0.95
        except ValueError:
            print("Invalid fill probability, using default 0.95")
            fill_prob = 0.95
    else:
        fill_prob = 0.95
    
    slippage = input("Slippage in basis points (default: 5): ").strip()
    if slippage:
        try:
            slippage_bps = int(slippage)
        except ValueError:
            print("Invalid slippage, using default 5 basis points")
            slippage_bps = 5
    else:
        slippage_bps = 5
    
    # Create configuration
    config = {
        "trading_mode": "papertrading",
        "papertrading": {
            "initial_balance_usd": balance_usd,
            "fill_probability": fill_prob,
            "slippage_bps": slippage_bps,
            "fill_delay_seconds": 0.5,
            "commission_per_contract": 0,
            "price_improvement_probability": 0.1,
            "enable_market_impact": True
        },
        "live_trading": {
            "safety_enabled": True,
            "max_position_size": 100,
            "max_total_exposure": 1000
        }
    }
    
    # Write configuration file
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        print(f"🎮 Mode: PAPERTRADING")
        print(f"💰 Starting Balance: ${balance_usd:,.2f}")
        print(f"📊 Fill Probability: {fill_prob:.1%}")
        print(f"📈 Slippage: {slippage_bps} basis points")
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def check_env_file():
    """Check if .env file exists with required settings"""
    print("\n📡 Checking API configuration...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  No .env file found")
        print("💡 You'll need to create .env with your Kalshi API credentials")
        print("   See .env.example for template")
        return False
    
    # Check for required variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    required_vars = ['KALSHI_API_KEY', 'KALSHI_PRIVATE_KEY_PATH']
    missing_vars = []
    
    for var in required_vars:
        if var not in env_content or f"{var}=your_" in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing or incomplete API configuration:")
        for var in missing_vars:
            print(f"   - {var}")
        print("💡 Update your .env file with real Kalshi API credentials")
        return False
    
    print("✅ API configuration found in .env")
    return True

def create_test_portfolio():
    """Create a simple test to verify papertrading works"""
    print("\n🧪 Creating papertrading test...")
    
    test_file = Path("test_papertrading.py")
    
    test_code = '''#!/usr/bin/env python3
"""
Simple test to verify papertrading functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config_manager import config
from simulated_portfolio_manager import SimulatedPortfolioManager
from simulated_execution_manager import SimulatedExecutionManager
from order_manager import OrderManager

async def test_papertrading():
    """Test basic papertrading functionality"""
    print("🧪 Testing papertrading components...")
    
    # Check configuration
    if not config.is_papertrading_enabled():
        print("❌ Papertrading not enabled in configuration")
        print(f"Current mode: {config.get_trading_mode()}")
        return False
    
    print("✅ Papertrading mode enabled")
    
    # Test portfolio manager
    try:
        portfolio = SimulatedPortfolioManager()
        balance = portfolio.get_balance()
        print(f"✅ Portfolio Manager: Starting balance ${balance['balance']/100:,.2f}")
        
        # Test portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"✅ Portfolio Summary: Return {summary['return_pct']:.2f}%")
        
    except Exception as e:
        print(f"❌ Portfolio Manager failed: {e}")
        return False
    
    # Test order manager
    try:
        order_manager = OrderManager()
        print("✅ Order Manager initialized")
    except Exception as e:
        print(f"❌ Order Manager failed: {e}")
        return False
    
    # Test execution manager
    try:
        execution = SimulatedExecutionManager(order_manager, portfolio)
        status = execution.get_status()
        print(f"✅ Execution Manager: Mode = {status['mode']}")
        print(f"   Fill Probability: {status['settings']['fill_probability']}")
        print(f"   Slippage: {status['settings']['slippage_bps']} bps")
    except Exception as e:
        print(f"❌ Execution Manager failed: {e}")
        return False
    
    print("\\n🎉 All papertrading components working correctly!")
    print("\\n📋 Configuration Summary:")
    print(f"   Trading Mode: {config.get_trading_mode().upper()}")
    print(f"   Papertrading: {config.is_papertrading_enabled()}")
    
    papertrading_settings = config.get_papertrading_settings()
    print(f"   Starting Balance: ${papertrading_settings.initial_balance / 100:,.2f}")
    print(f"   Fill Probability: {papertrading_settings.fill_probability:.1%}")
    print(f"   Slippage: {papertrading_settings.slippage_bps} basis points")
    
    print("\\n🚀 Ready to run: python main.py")
    return True

if __name__ == "__main__":
    asyncio.run(test_papertrading())
'''
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    # Make executable
    os.chmod(test_file, 0o755)
    
    print(f"✅ Test file created: {test_file}")
    return test_file

def run_test(test_file: Path):
    """Run the papertrading test"""
    print(f"\n🚀 Running papertrading test...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, timeout=30)
        
        print("Test output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Test passed!")
            return True
        else:
            print("❌ Test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run test: {e}")
        return False

def show_next_steps():
    """Show user what to do next"""
    print("\n🎯 NEXT STEPS:")
    print("-" * 40)
    print("1. Run the test: python test_papertrading.py")
    print("2. Start the trading system: python main.py")
    print("3. Look for '💰 PAPERTRADING' in the interface")
    print("4. Monitor simulated trades in the logs")
    print("5. Check portfolio performance in the display")
    
    print("\n🔄 MODE SWITCHING:")
    print("- Use: python trading_mode_switcher.py")
    print("- Or edit trading_config.json directly")
    
    print("\n📚 TIPS:")
    print("- All trades are simulated - no real money is used")
    print("- Your .env file remains unchanged")
    print("- Adjust settings in trading_config.json")
    print("- Test different strategies safely")
    print("- Monitor performance metrics")
    
    print("\n⚠️  TO ENABLE LIVE TRADING LATER:")
    print("- Run: python trading_mode_switcher.py")
    print("- Select option 2 (Live Trading)")
    print("- Follow safety confirmations")
    print("- Test with small position sizes first!")

def main():
    """Main setup function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check .env file
    env_ok = check_env_file()
    if not env_ok:
        print("\n⚠️  API configuration incomplete, but continuing with setup...")
    
    # Create trading configuration
    if not create_trading_config():
        sys.exit(1)
    
    # Create test
    test_file = create_test_portfolio()
    
    # Ask if user wants to run test
    run_test_now = input("\n🧪 Run papertrading test now? (y/n): ").strip().lower()
    
    if run_test_now in ['y', 'yes']:
        success = run_test(test_file)
        if not success:
            print("\n⚠️  Test failed, but you can still try running the main system.")
            if not env_ok:
                print("💡 The test failure might be due to missing API credentials.")
    
    show_next_steps()
    
    print("\n🎮 Papertrading setup complete!")
    print("📁 Configuration saved to: trading_config.json")
    print("🔧 Mode switcher available: trading_mode_switcher.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
