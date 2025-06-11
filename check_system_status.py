#!/usr/bin/env python3
"""
System Status Check
Analyzes what should be happening based on current time and configuration
"""

from datetime import datetime
import pytz

def check_market_status():
    """Check current market status"""
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    print("🕐 CURRENT TIME ANALYSIS")
    print("=" * 50)
    print(f"UTC Time: {datetime.utcnow()}")
    print(f"ET Time: {now_et}")
    print(f"Day: {now_et.strftime('%A')}")
    
    # Check if it's a weekday
    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        print("\n❌ WEEKEND - Stock markets closed")
        print("✅ Crypto markets open 24/7")
        return False, True
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    stock_market_open = market_open <= now_et <= market_close
    
    if stock_market_open:
        print(f"\n✅ US STOCK MARKET OPEN")
        print(f"   Market hours: 9:30 AM - 4:00 PM ET")
        print(f"   Time until close: {market_close - now_et}")
    else:
        if now_et < market_open:
            print(f"\n⏰ US STOCK MARKET CLOSED (Pre-market)")
            print(f"   Opens in: {market_open - now_et}")
        else:
            print(f"\n🌙 US STOCK MARKET CLOSED (After-hours)")
            print(f"   Closed {now_et - market_close} ago")
    
    print("\n✅ CRYPTO MARKET OPEN (24/7)")
    
    return stock_market_open, True

def expected_behavior(stock_open, crypto_open):
    """Describe expected system behavior"""
    print("\n📋 EXPECTED SYSTEM BEHAVIOR")
    print("=" * 50)
    
    print("\n🔧 CONFIGURATION:")
    print("- Crypto: BTC/USD, ETH/USD, SOL/USD, AVAX/USD")
    print("- Stocks: SPY, QQQ, AAPL, TSLA (if enabled)")
    print("- Stock Trading: ENABLED")
    print("- Short Selling: ENABLED")
    
    print("\n🎯 EXPECTED OPERATIONS:")
    
    if crypto_open:
        print("\n✅ CRYPTO TRADING (Active):")
        print("   - MA Crossover Strategy")
        print("   - 3x leverage available")
        print("   - Binance volume data")
        print("   - Multiple entry signals:")
        print("     • Golden/Death crosses")
        print("     • Trend continuation")
        print("     • Momentum entries")
        print("     • Breakout signals")
    
    if stock_open:
        print("\n✅ STOCK TRADING (Active):")
        print("   - Mean Reversion Strategy")
        print("   - Long and short positions")
        print("   - 4x leverage available")
        print("   - Entry signals:")
        print("     • Oversold bounce (long)")
        print("     • Overbought reversal (short)")
        print("   - 2% profit target, 1% stop loss")
    else:
        print("\n⏸️  STOCK TRADING (Paused - Market Closed)")
    
    print("\n💰 POSITION SIZING:")
    print("   - 2% risk per trade")
    print("   - 50% of buying power max per position")
    print("   - Min 1% price risk (prevents huge positions)")
    print("   - $100 minimum position size")

def check_recent_fixes():
    """List recent fixes that should be working"""
    print("\n🔧 RECENT FIXES DEPLOYED")
    print("=" * 50)
    
    fixes = [
        ("Position Sizing", "Fixed oversized positions exceeding buying power"),
        ("Firebase", "Fixed JSON parsing for multi-line private keys"),
        ("Stock Integration", "Connected stock strategy to main trading loop"),
        ("Volume Data", "Binance volume for crypto, Alpaca volume for stocks"),
        ("Order Execution", "Added null checks to prevent NoneType errors")
    ]
    
    for fix, description in fixes:
        print(f"✅ {fix}: {description}")

def common_log_patterns():
    """Show what to look for in logs"""
    print("\n📊 LOG PATTERNS TO LOOK FOR")
    print("=" * 50)
    
    print("\n🟢 GOOD SIGNS:")
    print('- "Added 4 stock symbols"')
    print('- "Stock mean reversion strategy initialized"')
    print('- "Using Firebase service account key from environment"')
    print('- "🎯 [SYMBOL] Signal: BUY/SELL"')
    print('- "Binance WebSocket: ✅ Connected"')
    
    print("\n🔴 ISSUES TO WATCH:")
    print('- "Firebase not initialized" (should be fixed)')
    print('- "insufficient balance" (should be fixed)')
    print('- "NoneType object has no attribute" (should be fixed)')
    print('- "No stock strategy available"')
    print('- "Market closed for stock"')

if __name__ == "__main__":
    print("🔍 CRYPTO TRADING SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # Check market status
    stock_open, crypto_open = check_market_status()
    
    # Expected behavior
    expected_behavior(stock_open, crypto_open)
    
    # Recent fixes
    check_recent_fixes()
    
    # Log patterns
    common_log_patterns()
    
    print("\n" + "=" * 50)
    print("💡 SUMMARY:")
    if stock_open:
        print("   Both crypto and stock trading should be active")
    else:
        print("   Only crypto trading should be active (stocks closed)")
    print("\n📝 Check Railway logs for actual behavior")
    print("   Use: railway logs | grep -i 'signal\\|error\\|warning'")