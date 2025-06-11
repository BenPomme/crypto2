#!/usr/bin/env python3
"""
Analyze profitability of different take profit/stop loss strategies
Comparing 6%/2% vs 3%/1% with leverage and fees
"""

import numpy as np
import pandas as pd

def analyze_strategy_profitability():
    """Compare different TP/SL strategies with fees and leverage"""
    
    print("📊 STRATEGY PROFITABILITY ANALYSIS")
    print("=" * 70)
    
    # Alpaca fee structure
    CRYPTO_FEE = 0.0025  # 0.25% per trade (0.5% round trip)
    STOCK_FEE = 0.0  # No commission on Alpaca
    
    # Leverage available
    CRYPTO_LEVERAGE = 3
    STOCK_LEVERAGE = 4
    
    # Strategy configurations
    strategies = {
        'Current (6%/2%)': {
            'take_profit': 0.06,
            'stop_loss': 0.02,
            'risk_reward_ratio': 3.0
        },
        'Proposed (3%/1%)': {
            'take_profit': 0.03,
            'stop_loss': 0.01,
            'risk_reward_ratio': 3.0
        }
    }
    
    # Win rate requirements based on risk/reward ratio
    # Break-even win rate = 1 / (1 + RR)
    # For 3:1 RR, break-even = 25%
    
    print("\n🎯 STRATEGY COMPARISON")
    print("-" * 70)
    
    for strategy_name, params in strategies.items():
        print(f"\n{strategy_name}:")
        print(f"  Take Profit: {params['take_profit']*100:.1f}%")
        print(f"  Stop Loss: {params['stop_loss']*100:.1f}%")
        print(f"  Risk/Reward Ratio: {params['risk_reward_ratio']:.1f}")
        
        # Calculate break-even win rate
        breakeven_wr = 1 / (1 + params['risk_reward_ratio'])
        print(f"  Break-even Win Rate: {breakeven_wr*100:.1f}%")
    
    print("\n💰 PROFITABILITY ANALYSIS WITH FEES")
    print("-" * 70)
    
    # Simulate different win rates
    win_rates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    num_trades = 100
    
    for asset_type in ['CRYPTO', 'STOCK']:
        fee = CRYPTO_FEE if asset_type == 'CRYPTO' else STOCK_FEE
        leverage = CRYPTO_LEVERAGE if asset_type == 'CRYPTO' else STOCK_LEVERAGE
        
        print(f"\n{asset_type} TRADING (Leverage: {leverage}x, Fee: {fee*100:.2f}% per trade)")
        print("-" * 50)
        
        results = []
        
        for strategy_name, params in strategies.items():
            strategy_results = []
            
            for win_rate in win_rates:
                # Calculate profit per winning trade (with leverage and fees)
                gross_profit = params['take_profit'] * leverage
                net_profit = gross_profit - (2 * fee)  # Entry + exit fees
                
                # Calculate loss per losing trade (with leverage and fees)
                gross_loss = params['stop_loss'] * leverage
                net_loss = gross_loss + (2 * fee)  # Entry + exit fees
                
                # Expected value per trade
                ev_per_trade = (win_rate * net_profit) - ((1 - win_rate) * net_loss)
                
                # Total return over num_trades
                total_return = ev_per_trade * num_trades
                
                strategy_results.append({
                    'Win Rate': f"{win_rate*100:.0f}%",
                    'Net Profit/Win': f"{net_profit*100:.2f}%",
                    'Net Loss/Loss': f"{net_loss*100:.2f}%",
                    'EV/Trade': f"{ev_per_trade*100:.3f}%",
                    'Total Return': f"{total_return*100:.1f}%"
                })
            
            results.append((strategy_name, pd.DataFrame(strategy_results)))
        
        # Display comparison
        for strategy_name, df in results:
            print(f"\n{strategy_name}:")
            print(df.to_string(index=False))
    
    print("\n🔍 KEY INSIGHTS")
    print("-" * 70)
    
    # Calculate specific scenarios
    print("\n1. IMPACT OF FEES:")
    for strategy_name, params in strategies.items():
        crypto_roundtrip_fee = 2 * CRYPTO_FEE * 100
        fee_impact_on_tp = (crypto_roundtrip_fee / (params['take_profit'] * 100)) * 100
        fee_impact_on_sl = (crypto_roundtrip_fee / (params['stop_loss'] * 100)) * 100
        
        print(f"\n   {strategy_name}:")
        print(f"   - Fees consume {fee_impact_on_tp:.1f}% of take profit")
        print(f"   - Fees add {fee_impact_on_sl:.1f}% to stop loss impact")
    
    print("\n2. LEVERAGE CONSIDERATIONS:")
    print("   - Higher leverage amplifies both profits AND losses")
    print("   - With 3x leverage, a 1% stop loss = 3% actual loss + fees")
    print("   - Risk of liquidation increases with tighter stops")
    
    print("\n3. EXECUTION CHALLENGES WITH TIGHTER STOPS:")
    print("   - 1% stop loss is very tight, especially for volatile crypto")
    print("   - Higher chance of premature stop-outs due to normal volatility")
    print("   - Slippage impact is proportionally larger on smaller moves")
    
    print("\n4. PRACTICAL CONSIDERATIONS:")
    print("   - More trades = more fees (especially impactful for crypto)")
    print("   - Tighter stops = more frequent trading = higher fee burden")
    print("   - Need higher win rate to overcome increased trading frequency")
    
    print("\n📈 RECOMMENDATION")
    print("-" * 70)
    print("The 3%/1% strategy with max leverage is RISKIER because:")
    print("1. Fees consume a larger percentage of profits (16.7% vs 8.3%)")
    print("2. Tighter stops increase premature exit risk")
    print("3. Higher leverage increases liquidation risk")
    print("4. More frequent trading increases total fee burden")
    print("\nThe 6%/2% strategy is likely more profitable in practice due to:")
    print("- Lower fee impact relative to profits")
    print("- More room for price fluctuations")
    print("- Less frequent stop-outs")
    print("- Better risk-adjusted returns")

if __name__ == "__main__":
    analyze_strategy_profitability()