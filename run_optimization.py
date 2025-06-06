#!/usr/bin/env python3
"""
Crypto Trading Parameter Optimization
Run this script to optimize parameters for your crypto trading instruments
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def run_crypto_optimization():
    """Run parameter optimization for crypto instruments"""
    print("🚀 Starting Crypto Trading Parameter Optimization")
    print("=" * 60)
    
    from src.backtesting import BacktestOrchestrator
    from src.strategy.ma_crossover_strategy import MACrossoverStrategy
    
    # 1. Initialize the orchestrator with your crypto symbols
    crypto_symbols = [
        "BTC/USD",  # Bitcoin
        "ETH/USD",  # Ethereum  
        "SOL/USD",  # Solana
        "AVAX/USD", # Avalanche
        "MATIC/USD" # Polygon
    ]
    
    orchestrator = BacktestOrchestrator(
        cache_dir="data/crypto_optimization",
        optimization_symbols=crypto_symbols
    )
    
    print(f"✅ Orchestrator initialized for {len(crypto_symbols)} symbols")
    print(f"   Symbols: {', '.join(crypto_symbols)}")
    
    # 2. Run comprehensive optimization
    print("\n📊 Running Parameter Optimization...")
    print("   This will take 10-15 minutes...")
    
    optimization_results = orchestrator.run_comprehensive_optimization(
        strategy_class=MACrossoverStrategy,
        optimization_period_days=90,      # 3 months of data
        validation_period_days=30,        # 1 month validation  
        optimization_methods=['bayesian', 'genetic', 'random'],
        n_iterations_per_method=50        # 50 iterations per method
    )
    
    # 3. Display results
    print("\n🎯 Optimization Results:")
    print("=" * 40)
    
    best_overall = optimization_results.get('best_overall', {})
    print(f"🏆 Best Overall Performance:")
    print(f"   Symbol: {best_overall.get('symbol', 'N/A')}")
    print(f"   Score: {best_overall.get('score', 0):.4f}")
    print(f"   Parameters: {best_overall.get('parameters', {})}")
    
    # Results by symbol
    print(f"\n📈 Results by Symbol:")
    for symbol, results in optimization_results.get('symbols', {}).items():
        print(f"\n   {symbol}:")
        print(f"      Best Score: {results.get('best_score', 0):.4f}")
        print(f"      Best Method: {results.get('best_method', 'N/A')}")
        print(f"      Parameters: {results.get('best_parameters', {})}")
    
    # Method comparison
    print(f"\n🔬 Method Comparison:")
    for method, stats in optimization_results.get('method_comparison', {}).items():
        print(f"   {method.title()}:")
        print(f"      Mean Score: {stats.get('mean_score', 0):.4f}")
        print(f"      Best Score: {stats.get('best_score', 0):.4f}")
        print(f"      Success Rate: {stats.get('success_rate', 0):.1%}")
    
    # 4. Train ML models
    print(f"\n🧠 Training ML Models...")
    orchestrator.train_ml_from_optimization_results(optimization_results)
    print("✅ ML models trained on optimization results")
    
    # 5. Get system status
    status = orchestrator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   ML Models Trained: {status['ml_learning']['models_trained']}")
    print(f"   Cache Size: {status['cache_info']['total_size_mb']:.1f} MB")
    
    return optimization_results, orchestrator

def get_live_trading_parameters(orchestrator, symbol="BTC/USD"):
    """Get optimized parameters for live trading"""
    print(f"\n🎯 Getting Parameters for Live Trading ({symbol})")
    print("=" * 50)
    
    # Simulate getting recent market data (replace with your actual data source)
    import pandas as pd
    import numpy as np
    
    # Create sample recent data (replace this with your actual data feed)
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=24), 
        periods=1440, 
        freq='1min'
    )
    
    recent_data = pd.DataFrame({
        'open': 50000 + np.random.randn(1440) * 100,
        'high': 50000 + np.random.randn(1440) * 100 + 50,
        'low': 50000 + np.random.randn(1440) * 100 - 50,
        'close': 50000 + np.random.randn(1440) * 100,
        'volume': np.random.randint(1000, 10000, 1440)
    }, index=dates)
    
    # Get ML-adaptive parameters
    ml_params = orchestrator.get_optimized_parameters_for_live_trading(
        current_market_data=recent_data,
        symbol=symbol,
        adaptation_mode="ml_adaptive"
    )
    
    print(f"🧠 ML-Adaptive Parameters:")
    print(f"   Confidence: {ml_params['confidence']:.2f}")
    print(f"   Adaptation Mode: {ml_params['adaptation_mode']}")
    
    for param, value in ml_params['parameters'].items():
        print(f"   {param}: {value}")
    
    # Get latest optimization parameters for comparison
    opt_params = orchestrator.get_optimized_parameters_for_live_trading(
        current_market_data=recent_data,
        symbol=symbol,
        adaptation_mode="latest_optimization"
    )
    
    print(f"\n📊 Latest Optimization Parameters:")
    print(f"   Confidence: {opt_params['confidence']:.2f}")
    
    for param, value in opt_params['parameters'].items():
        print(f"   {param}: {value}")
    
    # Decision logic
    print(f"\n💡 Parameter Selection Strategy:")
    if ml_params['confidence'] > 0.7:
        print("   ✅ HIGH CONFIDENCE - Use ML-adaptive parameters")
        recommended = ml_params
    elif opt_params['confidence'] > 0.6:
        print("   📊 MEDIUM CONFIDENCE - Use latest optimization parameters")
        recommended = opt_params
    else:
        print("   ⚠️  LOW CONFIDENCE - Use conservative default parameters")
        recommended = {
            'parameters': {
                'fast_ma_period': 10,
                'slow_ma_period': 30,
                'risk_per_trade': 0.015,
                'confidence_threshold': 0.6
            },
            'confidence': 0.5,
            'adaptation_mode': 'default'
        }
    
    print(f"\n🎯 RECOMMENDED PARAMETERS FOR LIVE TRADING:")
    print(f"   Confidence: {recommended['confidence']:.2f}")
    print(f"   Source: {recommended.get('adaptation_mode', 'default')}")
    
    for param, value in recommended['parameters'].items():
        print(f"   {param}: {value}")
    
    return recommended

async def main():
    """Main optimization workflow"""
    try:
        # Run optimization
        results, orchestrator = await run_crypto_optimization()
        
        # Get parameters for each symbol
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        
        print(f"\n🔧 PARAMETERS FOR LIVE TRADING")
        print("=" * 60)
        
        all_params = {}
        for symbol in symbols:
            params = get_live_trading_parameters(orchestrator, symbol)
            all_params[symbol] = params
        
        # Save parameters for use in main.py
        import json
        with open('optimized_parameters.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'optimization_results': {
                    'best_overall_score': results.get('best_overall', {}).get('score', 0),
                    'symbols_optimized': list(results.get('symbols', {}).keys())
                },
                'live_trading_parameters': {
                    symbol: {
                        'parameters': params['parameters'],
                        'confidence': params['confidence'],
                        'source': params.get('adaptation_mode', 'unknown')
                    }
                    for symbol, params in all_params.items()
                }
            }, f, indent=2)
        
        print(f"\n💾 Parameters saved to 'optimized_parameters.json'")
        print(f"✅ Ready to integrate with live trading system!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Crypto Trading Parameter Optimization")
    print("This script will optimize parameters for your crypto trading strategy")
    print("and provide ML-adaptive parameters for live trading.\n")
    
    success = asyncio.run(main())
    
    if success:
        print(f"\n🎉 Optimization completed successfully!")
        print(f"📝 Integration instructions:")
        print(f"   1. Review 'optimized_parameters.json'")
        print(f"   2. Update your main.py to use these parameters")
        print(f"   3. Run optimization weekly/monthly for best results")
    else:
        print(f"\n❌ Optimization failed - check logs above")