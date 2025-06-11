#!/usr/bin/env python3
"""
Check historical volatility to determine if 1% stop loss is realistic
"""

import numpy as np

def analyze_volatility_impact():
    """Analyze if 1% stop loss is viable given typical market volatility"""
    
    print("📊 VOLATILITY ANALYSIS FOR STOP LOSS VIABILITY")
    print("=" * 70)
    
    # Typical daily volatility (standard deviation of daily returns)
    # Based on historical data
    volatility_data = {
        'BTC/USD': {'daily_vol': 0.04, 'hourly_vol': 0.01},    # 4% daily, 1% hourly
        'ETH/USD': {'daily_vol': 0.05, 'hourly_vol': 0.013},   # 5% daily, 1.3% hourly
        'SOL/USD': {'daily_vol': 0.07, 'hourly_vol': 0.018},   # 7% daily, 1.8% hourly
        'AVAX/USD': {'daily_vol': 0.065, 'hourly_vol': 0.017}, # 6.5% daily, 1.7% hourly
        'SPY': {'daily_vol': 0.012, 'hourly_vol': 0.003},      # 1.2% daily, 0.3% hourly
        'QQQ': {'daily_vol': 0.015, 'hourly_vol': 0.004},      # 1.5% daily, 0.4% hourly
        'AAPL': {'daily_vol': 0.018, 'hourly_vol': 0.005},     # 1.8% daily, 0.5% hourly
        'TSLA': {'daily_vol': 0.04, 'hourly_vol': 0.01},       # 4% daily, 1% hourly
    }
    
    print("\n🎯 PROBABILITY OF HITTING STOP LOSS")
    print("-" * 70)
    
    # Calculate probability of hitting stop loss due to normal volatility
    # Using normal distribution approximation
    
    for stop_loss in [0.01, 0.02]:  # 1% and 2% stop losses
        print(f"\n{stop_loss*100:.0f}% Stop Loss Analysis:")
        print("-" * 50)
        
        for symbol, vol_data in volatility_data.items():
            # Calculate how many standard deviations the stop loss represents
            hourly_vol = vol_data['hourly_vol']
            daily_vol = vol_data['daily_vol']
            
            # For 1-hour holding period
            z_score_hourly = stop_loss / hourly_vol
            # Approximate probability (assuming normal distribution)
            # For z-score > 1, probability decreases significantly
            if z_score_hourly >= 2:
                prob_hit_hourly = 0.05  # ~5% for 2 std devs
            elif z_score_hourly >= 1:
                prob_hit_hourly = 0.32  # ~32% for 1 std dev
            else:
                prob_hit_hourly = 0.5 + (0.5 - z_score_hourly) * 0.18
            
            # For 1-day holding period
            z_score_daily = stop_loss / daily_vol
            if z_score_daily >= 2:
                prob_hit_daily = 0.05
            elif z_score_daily >= 1:
                prob_hit_daily = 0.32
            else:
                prob_hit_daily = 0.5 + (0.5 - z_score_daily) * 0.18
            
            # With leverage
            leverages = {'crypto': 3, 'stock': 4}
            leverage = leverages['crypto'] if '/' in symbol else leverages['stock']
            effective_stop = stop_loss / leverage
            
            z_score_lev = effective_stop / hourly_vol
            if z_score_lev >= 2:
                prob_hit_lev = 0.05
            elif z_score_lev >= 1:
                prob_hit_lev = 0.32
            else:
                prob_hit_lev = 0.5 + (0.5 - z_score_lev) * 0.18
            
            print(f"\n{symbol}:")
            print(f"  Hourly volatility: {hourly_vol*100:.1f}%")
            print(f"  Stop loss is {z_score_hourly:.1f} standard deviations away")
            print(f"  Probability of random hit (1hr): {prob_hit_hourly*100:.1f}%")
            print(f"  With {leverage}x leverage: {prob_hit_lev*100:.1f}% chance of hit")
    
    print("\n💡 IMPLICATIONS")
    print("-" * 70)
    print("\n1% STOP LOSS ISSUES:")
    print("- For volatile cryptos (SOL, AVAX), there's a 40-60% chance of hitting")
    print("  the stop due to NORMAL price movement within 1 hour")
    print("- With 3x leverage, effective stop is only 0.33%, making it even worse")
    print("- This leads to excessive stop-outs and poor performance")
    
    print("\n2% STOP LOSS ADVANTAGES:")
    print("- Provides 2x the buffer against normal volatility")
    print("- Reduces false stop-outs significantly")
    print("- Better suited for leveraged trading")
    
    print("\n🎯 EXPECTED OUTCOMES")
    print("-" * 70)
    
    # Simulate expected number of trades and outcomes
    trades_per_day = 10  # Approximate
    days = 30
    
    for strategy in [
        {'name': '6%/2%', 'tp': 0.06, 'sl': 0.02, 'false_stop_rate': 0.15},
        {'name': '3%/1%', 'tp': 0.03, 'sl': 0.01, 'false_stop_rate': 0.40}
    ]:
        total_trades = trades_per_day * days
        false_stops = int(total_trades * strategy['false_stop_rate'])
        real_trades = total_trades - false_stops
        
        # Assume 50% win rate on "real" trades
        wins = int(real_trades * 0.5)
        real_losses = real_trades - wins
        total_losses = real_losses + false_stops
        
        # Calculate P&L (with 0.5% round-trip fees for crypto)
        fee_per_trade = 0.005
        gross_profit = wins * strategy['tp'] - total_losses * strategy['sl']
        total_fees = total_trades * fee_per_trade
        net_profit = gross_profit - total_fees
        
        print(f"\n{strategy['name']} Strategy (30 days):")
        print(f"  Total trades: {total_trades}")
        print(f"  False stops: {false_stops} ({strategy['false_stop_rate']*100:.0f}%)")
        print(f"  Actual wins: {wins}")
        print(f"  Total losses: {total_losses}")
        print(f"  Gross profit: {gross_profit*100:.1f}%")
        print(f"  Total fees: {total_fees*100:.1f}%")
        print(f"  Net profit: {net_profit*100:.1f}%")
        print(f"  Profit per trade: {(net_profit/total_trades)*100:.3f}%")

if __name__ == "__main__":
    analyze_volatility_impact()