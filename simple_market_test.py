#!/usr/bin/env python3
"""
Simple test for market hours logic
"""
from datetime import datetime, timezone, timedelta

def test_market_hours_logic():
    """Test market hours checking logic standalone"""
    print("🧪 Testing Market Hours Logic")
    print("=" * 50)
    
    # Timezone setup (EST/EDT is UTC-5/-4)
    # Use a simple offset for testing (EST = UTC-5)
    eastern_offset = timezone(timedelta(hours=-5))  # EST offset
    
    # Get current time in US Eastern timezone (approximate)
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(eastern_offset)
    
    print(f"Current time UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current time ET:  {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Day of week: {now_et.strftime('%A')} (0=Monday, 6=Sunday: {now_et.weekday()})")
    
    # Check if it's a weekday
    is_weekday = now_et.weekday() < 5  # Monday=0, Sunday=6
    print(f"Is weekday: {'✅ Yes' if is_weekday else '❌ No (Weekend)'}")
    
    if is_weekday:
        # Extended hours: 4:00 AM - 8:00 PM ET
        market_open_ext = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
        market_close_ext = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
        is_extended_open = market_open_ext <= now_et <= market_close_ext
        
        # Regular hours: 9:30 AM - 4:00 PM ET
        market_open_reg = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_reg = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_regular_open = market_open_reg <= now_et <= market_close_reg
        
        print(f"\nExtended Hours (4:00 AM - 8:00 PM ET):")
        print(f"  Open time:  {market_open_ext.strftime('%H:%M:%S')}")
        print(f"  Close time: {market_close_ext.strftime('%H:%M:%S')}")
        print(f"  Status: {'✅ Open' if is_extended_open else '❌ Closed'}")
        
        print(f"\nRegular Hours (9:30 AM - 4:00 PM ET):")
        print(f"  Open time:  {market_open_reg.strftime('%H:%M:%S')}")
        print(f"  Close time: {market_close_reg.strftime('%H:%M:%S')}")
        print(f"  Status: {'✅ Open' if is_regular_open else '❌ Closed'}")
    else:
        print(f"\n❌ Weekend - Market is closed")
    
    # Test crypto symbol detection logic
    print(f"\nTesting Crypto Symbol Detection:")
    crypto_symbols = {
        'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'DOGE/USD',
        'ADA/USD', 'DOT/USD', 'UNI/USD', 'LINK/USD', 'AAVE/USD',
        'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'DOGEUSD',
        'ADAUSD', 'DOTUSD', 'UNIUSD', 'LINKUSD', 'AAVEUSD'
    }
    
    test_symbols = ['BTC/USD', 'BTCUSD', 'ETH/USD', 'AAPL', 'GOOGL', 'TSLA', 'SPY']
    
    for symbol in test_symbols:
        # Normalize symbol format
        normalized = symbol.upper().replace('/', '')
        with_slash = symbol.upper()
        
        is_crypto = (normalized in crypto_symbols or 
                    with_slash in crypto_symbols or
                    any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'LTC', 'BCH', 'DOGE', 'ADA', 'DOT', 'UNI', 'LINK', 'AAVE']))
        
        symbol_type = "crypto" if is_crypto else "stock"
        print(f"  {symbol}: {symbol_type}")
    
    print(f"\n✅ Market hours logic test completed!")

if __name__ == "__main__":
    test_market_hours_logic()