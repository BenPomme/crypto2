#!/usr/bin/env python3
"""
🎯 Robust Parameter Optimization - Extended Version
Use this for thorough optimization with 180+ days data and sufficient iterations
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def run_robust_optimization():
    """Run robust parameter optimization with extended data and iterations"""
    print("🎯 Starting ROBUST Crypto Trading Parameter Optimization")
    print("=" * 70)
    print("⚡ This will use:")
    print("   • 180 days of historical data (6 months)")
    print("   • 150 iterations per method (450 total per symbol)")
    print("   • 1-minute granularity data")
    print("   • Estimated time: 60-90 minutes")
    print("=" * 70)
    
    confirm = input("🚀 Proceed with robust optimization? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Optimization cancelled")
        return
    
    from src.backtesting import BacktestOrchestrator
    from src.strategy.ma_crossover_strategy import MACrossoverStrategy
    
    # Crypto symbols
    crypto_symbols = [
        "BTC/USD",  # Bitcoin
        "ETH/USD",  # Ethereum  
        "SOL/USD",  # Solana
        "DOGE/USD", # Dogecoin
    ]
    
    orchestrator = BacktestOrchestrator(
        cache_dir="data/robust_optimization",
        optimization_symbols=crypto_symbols
    )
    
    print(f"✅ Orchestrator initialized for {len(crypto_symbols)} symbols")
    print(f"   Symbols: {', '.join(crypto_symbols)}")
    
    # ROBUST optimization settings
    print("\n📊 Running ROBUST Parameter Optimization...")
    
    optimization_results = orchestrator.run_comprehensive_optimization(
        strategy_class=MACrossoverStrategy,
        optimization_period_days=180,     # 6 months of data  
        validation_period_days=60,        # 2 months validation
        optimization_methods=['bayesian', 'genetic', 'random'],
        n_iterations_per_method=150,      # 150 iterations per method = 450 total
        use_walk_forward=True,            # Enable walk-forward analysis
        walk_forward_periods=3            # 3 periods for validation
    )
    
    # Save robust results
    import json
    robust_results = {
        "timestamp": datetime.now().isoformat(),
        "optimization_metadata": {
            "action_taken": "robust_optimization",
            "symbols_optimized": len(crypto_symbols),
            "data_period_days": 180,
            "iterations_per_method": 150,
            "total_iterations": 450 * len(crypto_symbols),
            "optimization_type": "robust_extended",
            "note": "6 months data, 450 iterations per symbol"
        },
        "live_trading_parameters": {}
    }
    
    # Process results
    print("\n🎯 Robust Optimization Results:")
    print("=" * 50)
    
    best_overall = optimization_results.get('best_overall', {})
    print(f"🏆 Best Overall Performance:")
    print(f"   Symbol: {best_overall.get('symbol', 'N/A')}")
    print(f"   Score: {best_overall.get('score', 0):.4f}")
    
    for symbol, results in optimization_results.get('symbols', {}).items():
        print(f"\n📈 {symbol}:")
        print(f"   Best Score: {results.get('best_score', 0):.4f}")
        print(f"   Method: {results.get('best_method', 'N/A')}")
        print(f"   Parameters: {results.get('best_parameters', {})}")
        
        # Add to robust results
        robust_results["live_trading_parameters"][symbol] = {
            "parameters": results.get('best_parameters', {}),
            "confidence": results.get('confidence', 0.0),
            "sharpe_ratio": results.get('best_score', 0.0),
            "optimization_method": results.get('best_method', 'unknown'),
            "source": "robust_optimization"
        }
    
    # Save to file
    with open('robust_optimized_parameters.json', 'w') as f:
        json.dump(robust_results, f, indent=2)
    
    print(f"\n💾 Results saved to: robust_optimized_parameters.json")
    print(f"🎉 Robust optimization complete!")
    
    # Display data quality info
    print(f"\n📊 Data Quality Summary:")
    print(f"   • Period: 180 days (6 months)")
    print(f"   • Granularity: 1-minute bars")
    print(f"   • Total data points per symbol: ~259,200")
    print(f"   • Optimization iterations: 450 per symbol")
    print(f"   • Walk-forward validation: 3 periods")

if __name__ == "__main__":
    asyncio.run(run_robust_optimization())