#!/usr/bin/env python3
"""
Position Sizing Fix
Addresses the issue of oversized positions exceeding buying power
"""

# The main issue is in position_sizer.py line 417-420:
# When stop loss is very close to entry, it creates huge positions
# Example: 2% risk on $100k with 0.5% stop = $200 / 0.005 = $40,000 position!

# We need to add additional constraints:
# 1. Max position should not exceed available buying power
# 2. Apply leverage limits properly for crypto
# 3. Ensure proper error handling when orders fail

def calculate_safe_position_size(account_value, buying_power, entry_price, stop_loss_price=None, is_crypto=True):
    """
    Calculate position size with proper constraints
    """
    # Base risk per trade (2%)
    risk_per_trade = 0.02
    risk_amount = account_value * risk_per_trade
    
    # Maximum position limits
    if is_crypto:
        # For crypto with 3x leverage, limit to 30% of account value per position
        max_position_by_account = account_value * 0.30
        # But also respect available buying power (with buffer)
        max_position_by_buying_power = buying_power * 0.80  # Use 80% of available
    else:
        # For stocks with 4x leverage
        max_position_by_account = account_value * 0.25
        max_position_by_buying_power = buying_power * 0.90
    
    # Calculate risk-based position size
    if stop_loss_price and entry_price != stop_loss_price:
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        
        # Add minimum price risk to prevent huge positions
        min_price_risk = 0.01  # At least 1% price movement
        price_risk = max(price_risk, min_price_risk)
        
        risk_based_position = risk_amount / price_risk
    else:
        # Default 4% price risk
        risk_based_position = risk_amount / 0.04
    
    # Take the minimum of all constraints
    final_position = min(
        risk_based_position,
        max_position_by_account,
        max_position_by_buying_power
    )
    
    # Ensure minimum position size
    final_position = max(final_position, 100)  # Min $100
    
    return final_position

# Example fix for the current issue:
# Account: $100k
# Buying Power: $10,244
# Instead of trying to use $16,737, it would limit to:
# min($16,737, $30,000, $8,195) = $8,195

print("Position Sizing Fix Analysis:")
print("="*50)
print("Problem: Position sizer calculated $16,737 position")
print("         But only $10,244 buying power available")
print()
print("Root Cause: Small stop loss distance creates huge positions")
print("           Example: 2% risk / 0.5% stop = 4x multiplier!")
print()
print("Solution: Add buying power constraint")
print("         Position = min(risk_based, account_limit, buying_power*0.8)")
print()
print("With fix: $16,737 → $8,195 (80% of $10,244)")
print("="*50)