#!/usr/bin/env python3
"""
Monitor stock trading activity in Railway logs
"""

import subprocess
import time
from datetime import datetime
import pytz

def check_stock_activity():
    """Check Railway logs for stock trading activity"""
    print("🔍 MONITORING STOCK TRADING ACTIVITY")
    print("=" * 50)
    
    # Get current time in ET
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    print(f"Current ET Time: {now_et}")
    print(f"Market Status: ", end="")
    
    # Check if market is open
    if now_et.weekday() >= 5:
        print("CLOSED (Weekend)")
    else:
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_open <= now_et <= market_close:
            print("OPEN ✅")
        else:
            if now_et < market_open:
                print(f"CLOSED (Opens in {market_open - now_et})")
            else:
                print(f"CLOSED (Closed {now_et - market_close} ago)")
    
    print("\n📋 MONITORING COMMANDS:")
    print("-" * 50)
    
    # Commands to check stock trading
    commands = [
        ("Check stock initialization", 
         'railway logs | grep -i "stock" | grep -E "(Added|initialized|symbols)" | tail -5'),
        
        ("Check stock signals", 
         'railway logs | grep -E "(SPY|QQQ|AAPL|TSLA)" | grep -i "signal" | tail -5'),
        
        ("Check market status for stocks",
         'railway logs | grep -E "(Market open|Market closed)" | grep -E "(SPY|QQQ|stock)" | tail -5'),
        
        ("Check mean reversion strategy",
         'railway logs | grep -i "mean reversion" | tail -5'),
        
        ("Check for stock trading errors",
         'railway logs | grep -i "stock" | grep -iE "(error|failed|exception)" | tail -5')
    ]
    
    for description, command in commands:
        print(f"\n{description}:")
        print(f"Command: {command}")
        print("-" * 30)
        
        try:
            # Note: This won't work locally but shows what commands to run
            print("Run this command in your terminal to check")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n💡 SUMMARY:")
    print("-" * 50)
    print("1. Stock trading should now be enabled after Railway redeploy")
    print("2. Monitor logs during market hours (9:30 AM - 4:00 PM ET)")
    print("3. Look for 'SPY', 'QQQ', 'AAPL', 'TSLA' in the logs")
    print("4. Check for 'Stock mean reversion strategy initialized' message")
    
    print("\n🚀 QUICK CHECK COMMAND:")
    print('railway logs | grep -E "(stock|SPY|QQQ|AAPL|TSLA)" | tail -20')

if __name__ == "__main__":
    check_stock_activity()