#!/usr/bin/env python3
"""
Emergency fix for critical issues
"""

print("""
=== EMERGENCY FIXES NEEDED ===

1. REMOVE _check_closed_positions():
   - This method might be causing initialization issues
   - Comment it out in main.py line ~311

2. FIX POSITION SIZING:
   - The system is trying to use >100% of balance
   - Need to enforce buying_power_usage in position_sizer.py

3. CHECK STOCK TRADING ENV:
   - ENABLE_STOCK_TRADING might not be set
   - Stock symbols might not be loaded

4. BINANCE VOLUME:
   - Volume manager might not be initialized properly
   - Check if Binance WebSocket is connecting

QUICK FIX COMMANDS:
1. Comment out the problematic method call
2. Add logging to see what's happening
3. Redeploy immediately
""")