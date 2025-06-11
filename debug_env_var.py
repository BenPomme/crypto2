#!/usr/bin/env python3
"""
Debug environment variable parsing
"""
import os

print("=== ENVIRONMENT VARIABLE DEBUG ===")
print(f"ENABLE_STOCK_TRADING raw: '{os.environ.get('ENABLE_STOCK_TRADING', 'NOT SET')}'")
print(f"STOCK_ENABLE_STOCK_TRADING raw: '{os.environ.get('STOCK_ENABLE_STOCK_TRADING', 'NOT SET')}'")

# Test boolean parsing
enable_str = os.environ.get('ENABLE_STOCK_TRADING', 'false')
print(f"\nString value: '{enable_str}'")
print(f"Lower: '{enable_str.lower()}'")
print(f"Is 'true': {enable_str.lower() == 'true'}")
print(f"Bool(): {bool(enable_str)}")

# Test pydantic parsing
try:
    from config.stock_settings import is_stock_trading_enabled, StockTradingSettings
    print(f"\nis_stock_trading_enabled(): {is_stock_trading_enabled()}")
    
    settings = StockTradingSettings()
    print(f"StockTradingSettings.enable_stock_trading: {settings.enable_stock_trading}")
    print(f"StockTradingSettings.stock_symbols: '{settings.stock_symbols}'")
except Exception as e:
    print(f"\nError loading settings: {e}")

print("\n=== EXPECTED BEHAVIOR ===")
print("If ENABLE_STOCK_TRADING='true', stock trading should be enabled")
print("The system should load SPY,QQQ,AAPL,TSLA symbols")
print("Stock trading happens during extended hours (4 AM - 8 PM ET)")
print("Current time should be within trading hours")