#!/usr/bin/env python3
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
    
    print("\n🎉 All papertrading components working correctly!")
    print("\n📋 Configuration Summary:")
    print(f"   Trading Mode: {config.get_trading_mode().upper()}")
    print(f"   Papertrading: {config.is_papertrading_enabled()}")
    
    papertrading_settings = config.get_papertrading_settings()
    print(f"   Starting Balance: ${papertrading_settings.initial_balance / 100:,.2f}")
    print(f"   Fill Probability: {papertrading_settings.fill_probability:.1%}")
    print(f"   Slippage: {papertrading_settings.slippage_bps} basis points")
    
    print("\n🚀 Ready to run: python main.py")
    return True

if __name__ == "__main__":
    asyncio.run(test_papertrading())
