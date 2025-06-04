#!/usr/bin/env python3
"""
Test script to check which symbols are available for trading
"""
import sys
import os

# Add src to path
sys.path.append('src')

from config.settings import get_settings
from src.data.market_data import AlpacaDataProvider

def test_symbols():
    """Test which symbols are available"""
    symbols_to_test = ["BTC/USD", "ETH/USD", "SPY", "QQQ"]
    
    provider = AlpacaDataProvider()
    
    print("Testing symbol availability:")
    print("=" * 50)
    
    for symbol in symbols_to_test:
        try:
            print(f"\nTesting {symbol}...")
            
            # Test getting latest price
            price = provider.get_latest_price(symbol)
            print(f"  ✅ Latest price: ${price:.2f}")
            
            # Test getting historical data
            historical = provider.get_historical_data(symbol, "1Min", 5)
            print(f"  ✅ Historical data: {len(historical)} bars")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_symbols()