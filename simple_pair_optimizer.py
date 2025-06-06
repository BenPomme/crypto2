"""
Simple Pair Optimizer - No Dependencies Required

This is a simplified version that will work without the full environment setup.
Use this to understand the optimization process and generate sample results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Your 4 trading pairs
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

def simulate_market_data(symbol, days=90):
    """Generate realistic market data for testing"""
    
    # Base prices for different coins
    base_prices = {
        "BTC/USD": 45000,
        "ETH/USD": 2500, 
        "SOL/USD": 100,
        "DOGE/USD": 0.08
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generate price series with trend and volatility
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days*24*60, freq='1min')
    
    # Random walk with trend
    returns = np.random.normal(0.0001, 0.002, len(dates))  # Small positive trend with volatility
    
    # Add some regime changes
    for i in range(len(returns)):
        if random.random() < 0.001:  # 0.1% chance of regime change
            returns[i:i+1000] *= random.choice([0.5, 2.0])  # Volatility change
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, prices * 0.001),
        'high': prices + np.abs(np.random.normal(0, prices * 0.002)),
        'low': prices - np.abs(np.random.normal(0, prices * 0.002)),
        'close': prices,
        'volume': np.random.randint(1000, 50000, len(dates))
    }, index=dates)
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def calculate_ma_crossover_performance(data, fast_ma, slow_ma, risk_per_trade=0.02):
    """
    Simulate MA crossover strategy performance
    """
    # Calculate moving averages
    data['ma_fast'] = data['close'].rolling(window=fast_ma).mean()
    data['ma_slow'] = data['close'].rolling(window=slow_ma).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1  # Buy signal
    data.loc[data['ma_fast'] < data['ma_slow'], 'signal'] = -1  # Sell signal
    
    # Find signal changes
    data['position'] = data['signal'].diff()
    
    # Calculate returns
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    
    initial_capital = 100000
    current_capital = initial_capital
    equity_curve = [initial_capital]
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        if row['position'] == 1 and position == 0:  # Enter long
            position = 1
            entry_price = row['close']
            entry_time = timestamp
            
        elif row['position'] == -1 and position == 1:  # Exit long
            exit_price = row['close']
            exit_time = timestamp
            
            # Calculate trade performance
            price_change = (exit_price - entry_price) / entry_price
            trade_return = price_change * risk_per_trade  # Position sizing
            trade_pnl = current_capital * trade_return
            
            current_capital += trade_pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': price_change,
                'pnl': trade_pnl,
                'hold_time': (exit_time - entry_time).total_seconds() / 3600  # hours
            })
            
            position = 0
        
        equity_curve.append(current_capital)
    
    if not trades:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_trade_return': 0
        }
    
    # Calculate performance metrics
    trade_returns = [t['return'] for t in trades]
    winning_trades = [t for t in trades if t['pnl'] > 0]
    
    total_return = (current_capital - initial_capital) / initial_capital
    
    if len(trade_returns) > 1:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_trade_return': np.mean(trade_returns) if trade_returns else 0,
        'trades': trades,
        'final_capital': current_capital
    }

def optimize_parameters_for_pair(symbol, iterations=50):
    """
    Optimize MA parameters for a single trading pair
    """
    print(f"🔬 Optimizing {symbol}...")
    
    # Generate market data
    data = simulate_market_data(symbol, days=90)
    
    best_score = -999
    best_params = None
    best_results = None
    
    optimization_history = []
    
    # Parameter ranges
    fast_ma_range = range(5, 21)
    slow_ma_range = range(20, 51)
    risk_range = [0.01, 0.015, 0.02, 0.025, 0.03]
    
    for i in range(iterations):
        # Random parameter selection
        fast_ma = random.choice(fast_ma_range)
        slow_ma = random.choice(slow_ma_range)
        risk_per_trade = random.choice(risk_range)
        
        # Ensure fast < slow
        if fast_ma >= slow_ma:
            continue
        
        # Test these parameters
        results = calculate_ma_crossover_performance(data, fast_ma, slow_ma, risk_per_trade)
        
        # Score based on Sharpe ratio with penalties for few trades
        score = results['sharpe_ratio']
        if results['total_trades'] < 5:
            score *= 0.5  # Penalty for too few trades
        if results['max_drawdown'] > 0.2:
            score *= 0.7  # Penalty for high drawdown
        
        optimization_history.append({
            'iteration': i,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'risk_per_trade': risk_per_trade,
            'score': score,
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate']
        })
        
        if score > best_score:
            best_score = score
            best_params = {
                'fast_ma_period': fast_ma,
                'slow_ma_period': slow_ma,
                'risk_per_trade': risk_per_trade,
                'confidence_threshold': 0.6  # Default
            }
            best_results = results
    
    return {
        'symbol': symbol,
        'best_score': best_score,
        'best_parameters': best_params,
        'best_results': best_results,
        'optimization_history': optimization_history
    }

def run_simple_optimization():
    """
    Run optimization for all trading pairs
    """
    print("🚀 Simple Trading Pair Optimizer")
    print("=" * 40)
    print(f"Optimizing: {', '.join(TRADING_PAIRS)}")
    print("Using simulated market data for demonstration")
    print()
    
    all_results = {}
    best_overall_score = -999
    best_overall_symbol = None
    
    # Optimize each pair
    for symbol in TRADING_PAIRS:
        result = optimize_parameters_for_pair(symbol, iterations=50)
        all_results[symbol] = result
        
        # Track best overall
        if result['best_score'] > best_overall_score:
            best_overall_score = result['best_score']
            best_overall_symbol = symbol
        
        # Print results
        print(f"💰 {symbol} Results:")
        print(f"   Best Score (Sharpe): {result['best_score']:.3f}")
        print(f"   Total Return: {result['best_results']['total_return']:.1%}")
        print(f"   Win Rate: {result['best_results']['win_rate']:.1%}")
        print(f"   Total Trades: {result['best_results']['total_trades']}")
        print(f"   Max Drawdown: {result['best_results']['max_drawdown']:.1%}")
        print(f"   Parameters:")
        print(f"     Fast MA: {result['best_parameters']['fast_ma_period']}")
        print(f"     Slow MA: {result['best_parameters']['slow_ma_period']}")
        print(f"     Risk/Trade: {result['best_parameters']['risk_per_trade']:.1%}")
        print()
    
    # Create summary for live trading
    live_config = {
        'optimization_date': datetime.now().isoformat(),
        'method': 'simple_random_search',
        'best_overall': {
            'symbol': best_overall_symbol,
            'score': best_overall_score,
            'parameters': all_results[best_overall_symbol]['best_parameters'] if best_overall_symbol else {}
        },
        'pairs': {}
    }
    
    for symbol, result in all_results.items():
        live_config['pairs'][symbol] = {
            'parameters': result['best_parameters'],
            'confidence': min(0.8, max(0.4, result['best_score'] / 2)),  # Convert score to confidence
            'adaptation_mode': 'optimization_result',
            'performance': {
                'total_return': result['best_results']['total_return'],
                'sharpe_ratio': result['best_results']['sharpe_ratio'],
                'win_rate': result['best_results']['win_rate'],
                'max_drawdown': result['best_results']['max_drawdown']
            }
        }
    
    # Save results
    with open('simple_optimized_parameters.json', 'w') as f:
        json.dump(live_config, f, indent=2, default=str)
    
    print("🎉 Optimization Complete!")
    print("=" * 30)
    print(f"🏆 Best performing pair: {best_overall_symbol}")
    print(f"📊 Best Sharpe ratio: {best_overall_score:.3f}")
    print(f"💾 Results saved to: simple_optimized_parameters.json")
    
    # Integration instructions
    print(f"\n🔗 How to Use These Results:")
    print("1. Open simple_optimized_parameters.json")
    print("2. Use the parameters in your trading strategy:")
    print()
    
    for symbol in TRADING_PAIRS:
        if symbol in all_results:
            params = all_results[symbol]['best_parameters']
            print(f"   {symbol}:")
            print(f"     Fast MA: {params['fast_ma_period']}")
            print(f"     Slow MA: {params['slow_ma_period']}")  
            print(f"     Risk: {params['risk_per_trade']:.1%}")
    
    print(f"\n📝 Add to your main.py:")
    print("""
import json

# Load optimized parameters
with open('simple_optimized_parameters.json', 'r') as f:
    config = json.load(f)

# Apply to your strategy
for symbol in ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]:
    if symbol in config['pairs']:
        params = config['pairs'][symbol]['parameters']
        
        # Update your strategy parameters
        strategy_config[symbol] = {
            'fast_ma_period': params['fast_ma_period'],
            'slow_ma_period': params['slow_ma_period'],
            'risk_per_trade': params['risk_per_trade']
        }
""")
    
    return live_config

if __name__ == "__main__":
    print("🎯 This is a simplified demonstration of the optimization process.")
    print("For production use with real Alpaca data, use optimize_trading_pairs.py")
    print()
    
    # Auto-run for demonstration
    print("🚀 Running optimization demo automatically...")
    results = run_simple_optimization()
    print(f"\n✅ Demo completed! Check simple_optimized_parameters.json for results.")