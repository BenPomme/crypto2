#!/usr/bin/env python3
"""
Test environment variable loading for stock settings
"""
import os
import sys
sys.path.append('.')

# Set test environment variables
os.environ['ENABLE_STOCK_TRADING'] = 'true'
os.environ['STOCK_SYMBOLS'] = 'SPY,QQQ,AAPL,TSLA'

print("=== ENVIRONMENT VARIABLE TEST ===")
print(f"ENABLE_STOCK_TRADING = '{os.environ.get('ENABLE_STOCK_TRADING')}'")
print(f"STOCK_SYMBOLS = '{os.environ.get('STOCK_SYMBOLS')}'")

# Test pydantic loading
try:
    from config.stock_settings import StockTradingSettings, is_stock_trading_enabled, get_stock_settings
    
    print("\n=== TESTING is_stock_trading_enabled() ===")
    enabled = is_stock_trading_enabled()
    print(f"Result: {enabled}")
    
    print("\n=== TESTING StockTradingSettings() ===")
    settings = StockTradingSettings()
    print(f"enable_stock_trading: {settings.enable_stock_trading}")
    print(f"stock_symbols: '{settings.stock_symbols}'")
    print(f"stock_symbols type: {type(settings.stock_symbols)}")
    print(f"stock_symbols bool: {bool(settings.stock_symbols)}")
    print(f"stock_symbols == '': {settings.stock_symbols == ''}")
    
    print("\n=== TESTING get_stock_settings() ===")
    stock_settings = get_stock_settings()
    if stock_settings:
        print(f"Settings loaded successfully")
        print(f"stock_symbols: '{stock_settings.stock_symbols}'")
    else:
        print("Settings returned None")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DIAGNOSIS ===")
print("If stock_symbols is empty string, pydantic might be using the default='' instead of env var")
print("The env_prefix='STOCK_' might be conflicting with env='STOCK_SYMBOLS'")