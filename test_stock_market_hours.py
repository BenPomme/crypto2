#!/usr/bin/env python3
"""
Test script to verify stock market hours logic
"""

import sys
import os
from datetime import datetime, timedelta
import pytz

# Add src to path
sys.path.append('src')

from src.data.market_data import AlpacaDataProvider
from config.stock_settings import get_stock_settings, is_stock_trading_enabled

def test_market_hours():
    """Test market hours logic"""
    print("🔍 TESTING STOCK MARKET HOURS LOGIC")
    print("=" * 50)
    
    # Check stock trading configuration
    print("\n📋 STOCK TRADING CONFIGURATION:")
    enabled = is_stock_trading_enabled()
    print(f"Stock Trading Enabled: {enabled}")
    
    if enabled:
        settings = get_stock_settings()
        print(f"Stock Symbols: {settings.stock_symbols}")
        print(f"Short Selling: {settings.enable_short_selling}")
    else:
        print("❌ Stock trading is NOT enabled in environment")
        print("To enable, set: ENABLE_STOCK_TRADING=true")
        return
    
    # Initialize data provider
    try:
        data_provider = AlpacaDataProvider()
        print("\n✅ AlpacaDataProvider initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize AlpacaDataProvider: {e}")
        return
    
    # Test market hours detection
    print("\n🕐 MARKET HOURS TESTING:")
    
    # Get current time in ET
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    print(f"Current ET Time: {now_et}")
    print(f"Day of Week: {now_et.strftime('%A')}")
    
    # Test different symbols
    test_symbols = [
        "BTC/USD",    # Crypto - should always be open
        "ETH/USD",    # Crypto - should always be open
        "SPY",        # Stock - should follow market hours
        "QQQ",        # Stock - should follow market hours
        "AAPL",       # Stock - should follow market hours
    ]
    
    print("\n📊 SYMBOL MARKET STATUS:")
    for symbol in test_symbols:
        is_crypto = data_provider.is_crypto_symbol(symbol)
        is_open = data_provider.is_market_open(symbol)
        symbol_type = "Crypto" if is_crypto else "Stock"
        status = "OPEN" if is_open else "CLOSED"
        
        print(f"{symbol:10} ({symbol_type:6}): {status}")
    
    # Test US market hours specifically
    print("\n🇺🇸 US MARKET HOURS:")
    regular_hours = data_provider.is_us_market_open(include_extended_hours=False)
    extended_hours = data_provider.is_us_market_open(include_extended_hours=True)
    
    print(f"Regular Hours (9:30 AM - 4:00 PM ET): {'OPEN' if regular_hours else 'CLOSED'}")
    print(f"Extended Hours (4:00 AM - 8:00 PM ET): {'OPEN' if extended_hours else 'CLOSED'}")
    
    # Simulate different times to test logic
    print("\n⏰ SIMULATED TIME TESTING:")
    
    # Test times (in hours from midnight ET)
    test_times = [
        (3.5, "3:30 AM ET - Pre-extended hours"),
        (4.0, "4:00 AM ET - Extended hours open"),
        (9.0, "9:00 AM ET - Pre-market"),
        (9.5, "9:30 AM ET - Market open"),
        (12.0, "12:00 PM ET - Mid-day"),
        (16.0, "4:00 PM ET - Market close"),
        (17.0, "5:00 PM ET - After-hours"),
        (20.0, "8:00 PM ET - Extended close"),
        (21.0, "9:00 PM ET - After extended"),
    ]
    
    # Get current date at midnight ET
    today_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    
    for hours, description in test_times:
        # Create test time
        test_time = today_et + timedelta(hours=hours)
        
        # Skip if weekend
        if test_time.weekday() >= 5:
            print(f"{description}: WEEKEND - Markets closed")
            continue
        
        # Check market status at this time
        # We need to mock the time for this test
        # For now, just show what the expected result would be
        hour = int(hours)
        minute = int((hours - hour) * 60)
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        is_regular = (hour == 9 and minute >= 30) or (10 <= hour <= 15) or (hour == 16 and minute == 0)
        
        # Extended hours: 4:00 AM - 8:00 PM ET
        is_extended = 4 <= hour <= 20
        
        print(f"{description}: Regular={'OPEN' if is_regular else 'CLOSED'}, Extended={'OPEN' if is_extended else 'CLOSED'}")
    
    print("\n" + "=" * 50)
    print("💡 SUMMARY:")
    if regular_hours:
        print("✅ Stock market is currently OPEN for regular trading")
    elif extended_hours:
        print("⚡ Stock market is in EXTENDED hours trading")
    else:
        print("❌ Stock market is currently CLOSED")
    
    print("\n📝 RECOMMENDATIONS:")
    if not enabled:
        print("1. Enable stock trading by setting ENABLE_STOCK_TRADING=true")
    elif not regular_hours and not extended_hours:
        print("1. Wait for market hours to test stock trading")
        print("2. Check Railway logs during market hours (9:30 AM - 4:00 PM ET)")
    else:
        print("1. Check Railway logs for stock trading signals")
        print("2. Look for 'SPY', 'QQQ', 'AAPL', 'TSLA' in logs")

if __name__ == "__main__":
    test_market_hours()