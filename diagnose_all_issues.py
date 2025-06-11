#!/usr/bin/env python3
"""
Diagnose all current issues:
1. Volume calculation not using Binance
2. Stock trading not happening
3. Balance errors
"""

print("""
=== CRITICAL ISSUES IDENTIFIED ===

1. VOLUME BUG REINTRODUCED:
   - Seeing "No volume rate available" warnings
   - Volume manager enhance_dataframe_volume() is failing
   - Need to check Binance WebSocket connection

2. NO STOCK TRADING:
   - No stock symbols in logs
   - Stock strategy might not be initialized
   - Need to verify ENABLE_STOCK_TRADING env var

3. BALANCE ERRORS:
   - Trying to use $10,466 with only $10,244 available
   - Position sizing is trying to use >100% of balance
   - The 50% buying power limit isn't working

4. POSSIBLE ROOT CAUSE:
   - The _check_closed_positions() method I added might be causing issues
   - Need to verify initialization sequence

IMMEDIATE FIXES NEEDED:
1. Revert or fix the _check_closed_positions() call
2. Verify stock trading environment variable
3. Fix position sizing to respect buying power
4. Check Binance WebSocket initialization
""")