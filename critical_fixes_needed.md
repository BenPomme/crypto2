# Critical Fixes Needed - June 11, 2025

## 1. Position Sizing Using 100% of Cash
**Issue**: System tries to use entire cash balance, causing insufficient balance errors
**Location**: `src/risk/position_sizer.py` - calculate_crypto_position()
**Fix**: Apply 50% limit to max_capital_available, not just max_position

## 2. Stock Trading Not Enabled
**Issue**: ENABLE_STOCK_TRADING=true but stocks not loading
**Possible Causes**:
- Environment variable not propagating to container
- Deployment not completing
- Railway caching old container

## 3. Market Order Slippage
**Issue**: Market orders fill at higher prices than expected
**Example**: Requested $10,244, filled at $10,425 (1.8% higher)
**Fix**: Add 2% buffer to position sizing for market orders

## Current State:
- Portfolio: $105,460
- Cash: $10,244
- Buying Power: $20,488
- Trades Failing: Due to using 100% of cash + slippage

## Emergency Fix Commands:
```bash
# Force new deployment
railway up --detach

# Check if env var is set
railway run env | grep STOCK

# Restart service
railway restart
```