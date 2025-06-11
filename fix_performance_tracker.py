#!/usr/bin/env python3
"""
Fix for performance tracker to properly calculate P&L from closed positions
"""
import sys
sys.path.append('src')

print("""
=== PERFORMANCE TRACKER FIX ===

The issue: Performance metrics show zero because:
1. Trades are recorded with pnl=0 initially
2. P&L should be calculated from closed positions
3. The system needs to update P&L when positions are closed

Solution:
1. Add a method to update trade P&L when positions close
2. Track position entry/exit properly
3. Calculate realized P&L from actual closed positions

This requires modifying:
- performance_tracker.py: Add update_trade_pnl() method
- trade_executor.py: Call update when positions close
- main.py: Track closed positions and update P&L

The fix will be implemented in the next commit.
""")