#!/usr/bin/env python3
"""
Test script for market hours checking functionality
"""
import sys
import os
sys.path.append('src')

from datetime import datetime
import pytz
from src.data.market_data import AlpacaDataProvider

def test_market_hours():
    """Test market hours checking functionality"""
    print("🧪 Testing Market Hours Functionality")
    print("=" * 50)
    
    try:
        # Initialize data provider
        data_provider = AlpacaDataProvider()
        
        # Test crypto symbol detection
        print("\n1. Testing Crypto Symbol Detection:")
        crypto_symbols = ['BTC/USD', 'BTCUSD', 'ETH/USD', 'ETHUSD', 'DOGE/USD']
        stock_symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
        
        for symbol in crypto_symbols:
            is_crypto = data_provider.is_crypto_symbol(symbol)
            print(f"   {symbol}: {'✅ Crypto' if is_crypto else '❌ Not Crypto'}")
            
        for symbol in stock_symbols:
            is_crypto = data_provider.is_crypto_symbol(symbol)
            print(f"   {symbol}: {'❌ Incorrectly identified as Crypto' if is_crypto else '✅ Correctly identified as Stock'}")
        
        # Test market hours checking
        print("\n2. Testing Market Hours:")
        
        # Get current time in different timezones
        utc_now = datetime.now(pytz.UTC)
        et_now = utc_now.astimezone(pytz.timezone('US/Eastern'))
        
        print(f"   Current time UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   Current time ET:  {et_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   Day of week: {et_now.strftime('%A')} (0=Monday, 6=Sunday: {et_now.weekday()})")
        
        # Check market status
        is_us_market_open = data_provider.is_us_market_open(include_extended_hours=True)
        is_us_market_regular = data_provider.is_us_market_open(include_extended_hours=False)
        
        print(f"\n   US Market Status:")
        print(f"   Extended Hours (4 AM - 8 PM ET): {'✅ Open' if is_us_market_open else '❌ Closed'}")
        print(f"   Regular Hours (9:30 AM - 4 PM ET): {'✅ Open' if is_us_market_regular else '❌ Closed'}")
        
        # Test combined symbol checking
        print("\n3. Testing Combined Symbol Market Checking:")
        test_symbols = ['BTC/USD', 'AAPL', 'ETH/USD', 'GOOGL', 'DOGE/USD', 'TSLA']
        
        for symbol in test_symbols:
            is_open = data_provider.is_market_open(symbol)
            is_crypto = data_provider.is_crypto_symbol(symbol)
            symbol_type = "crypto" if is_crypto else "stock"
            status = "open" if is_open else "closed"
            print(f"   {symbol} ({symbol_type}): {'✅ Open' if is_open else '❌ Closed'}")
        
        print("\n✅ Market hours testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_market_hours()
    sys.exit(0 if success else 1)