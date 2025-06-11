#!/usr/bin/env python3
"""
Comprehensive fix for:
1. Performance metrics showing zero (P&L tracking)
2. Stock trading not happening during market hours
"""

print("""
=== COMPREHENSIVE SYSTEM FIX ===

Issues Identified:
1. Performance metrics show zero because trades are recorded with pnl=0
2. Stock trading might not be happening due to strategy conditions
3. P&L needs to be calculated from actual closed positions

Fixes Applied:
1. Updated performance_tracker.py:
   - Added 'realized_pl' field for metrics calculator compatibility
   - Added update_closed_position_pnl() method
   - Modified record_trade to use realized_pnl if available

2. Updated main.py:
   - Added _check_closed_positions() method to track P&L
   - Added more logging for stock strategy analysis
   - Called P&L check after each trading cycle

3. Recommendations:
   - Monitor during market hours (9:30 AM - 4:00 PM ET)
   - Check if stock strategy conditions are being met
   - Verify volume requirements ($1M minimum for stocks)

To deploy these fixes:
1. Commit the changes
2. Push to main branch (auto-deploys to Railway)
3. Monitor logs for improved metrics

Expected Results:
- Performance metrics should update when positions close
- Stock analysis logs every 10 cycles during market hours
- P&L tracking from actual trades, not hardcoded zeros
""")