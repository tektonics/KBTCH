#!/usr/bin/env python3
"""
Trading Mode Switcher for KBTCH Trading System
Easy utility to switch between papertrading and live trading modes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def print_banner():
    """Print banner"""
    print("=" * 60)
    print("🔄 KBTCH TRADING SYSTEM - MODE SWITCHER")
    print("=" * 60)

def load_config() -> Dict[str, Any]:
    """Load current trading configuration"""
    config_file = Path("trading_config.json")
    
    if not config_file.exists():
        print("❌ No trading_config.json found!")
        print("💡 Run the main system first to create default configuration")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error reading config file: {e}")
        sys.exit(1)

def save_config(config: Dict[str, Any]) -> bool:
    """Save trading configuration"""
    config_file = Path("trading_config.json")
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def show_current_config(config: Dict[str, Any]):
    """Display current configuration"""
    current_mode = config.get("trading_mode", "papertrading")
    
    print(f"\n📊 CURRENT CONFIGURATION")
    print("-" * 40)
    
    if current_mode == "papertrading":
        print("🎮 Mode: PAPERTRADING (Safe - No real money)")
        print("   Simulated trading with virtual money")
        
        papertrading = config.get("papertrading", {})
        balance = papertrading.get("initial_balance_usd", 1000)
        fill_prob = papertrading.get("fill_probability", 0.95)
        slippage = papertrading.get("slippage_bps", 5)
        
        print(f"   Starting Balance: ${balance:,.2f}")
        print(f"   Fill Probability: {fill_prob:.1%}")
        print(f"   Slippage: {slippage} basis points")
        
    else:
        print("🔴 Mode: LIVE TRADING (Real money at risk!)")
        print("   Actual trades executed on Kalshi")
        
        live = config.get("live_trading", {})
        safety = live.get("safety_enabled", True)
        
        if safety:
            print("   ⚠️  Safety mode enabled")
        else:
            print("   🚨 Safety mode DISABLED")

def switch_to_papertrading(config: Dict[str, Any]) -> Dict[str, Any]:
    """Switch to papertrading mode"""
    print("\n🎮 Switching to PAPERTRADING mode...")
    
    config["trading_mode"] = "papertrading"
    
    # Ensure papertrading config exists
    if "papertrading" not in config:
        config["papertrading"] = {}
    
    # Get user preferences
    print("\n💰 Papertrading Configuration:")
    
    current_balance = config["papertrading"].get("initial_balance_usd", 1000)
    balance_input = input(f"Starting balance USD (current: ${current_balance:,.2f}): ").strip()
    
    if balance_input:
        try:
            config["papertrading"]["initial_balance_usd"] = float(balance_input)
        except ValueError:
            print("Invalid balance, keeping current value")
    
    current_fill_prob = config["papertrading"].get("fill_probability", 0.95)
    fill_input = input(f"Fill probability 0-1 (current: {current_fill_prob}): ").strip()
    
    if fill_input:
        try:
            fill_prob = float(fill_input)
            if 0 <= fill_prob <= 1:
                config["papertrading"]["fill_probability"] = fill_prob
            else:
                print("Fill probability must be between 0 and 1, keeping current value")
        except ValueError:
            print("Invalid fill probability, keeping current value")
    
    print("✅ Configured for papertrading mode")
    return config

def switch_to_live(config: Dict[str, Any]) -> Dict[str, Any]:
    """Switch to live trading mode"""
    print("\n🔴 Switching to LIVE TRADING mode...")
    print("⚠️  WARNING: This will use REAL MONEY!")
    
    # Safety confirmation
    confirm1 = input("\nType 'I UNDERSTAND' to confirm you want live trading: ").strip()
    if confirm1 != "I UNDERSTAND":
        print("❌ Confirmation failed. Staying in current mode.")
        return config
    
    confirm2 = input("Type 'REAL MONEY' to confirm you understand real money will be used: ").strip()
    if confirm2 != "REAL MONEY":
        print("❌ Confirmation failed. Staying in current mode.")
        return config
    
    config["trading_mode"] = "live"
    
    # Ensure live trading config exists
    if "live_trading" not in config:
        config["live_trading"] = {}
    
    # Safety settings
    print("\n🛡️  Safety Configuration:")
    safety_input = input("Enable safety mode? (Y/n): ").strip().lower()
    
    if safety_input in ['n', 'no']:
        config["live_trading"]["safety_enabled"] = False
        print("🚨 Safety mode DISABLED")
    else:
        config["live_trading"]["safety_enabled"] = True
        print("✅ Safety mode enabled")
    
    print("🔴 Configured for live trading mode")
    return config

def main():
    """Main function"""
    print_banner()
    
    # Load current config
    config = load_config()
    
    # Show current status
    show_current_config(config)
    
    print("\n🔄 MODE SWITCHING OPTIONS")
    print("-" * 40)
    print("1. Switch to PAPERTRADING mode (safe)")
    print("2. Switch to LIVE TRADING mode (real money)")
    print("3. Show current configuration")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            config = switch_to_papertrading(config)
            if save_config(config):
                print("\n🎮 Successfully switched to PAPERTRADING mode!")
                print("💡 Restart the trading system to apply changes")
            break
            
        elif choice == "2":
            config = switch_to_live(config)
            if save_config(config):
                print("\n🔴 Successfully switched to LIVE TRADING mode!")
                print("⚠️  RESTART THE SYSTEM - REAL MONEY WILL BE USED!")
            break
            
        elif choice == "3":
            show_current_config(config)
            continue
            
        elif choice == "4":
            print("👋 Exiting without changes")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
