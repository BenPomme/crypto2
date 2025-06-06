"""
Example Usage of ML-Driven Backtesting System

This script demonstrates how to use the comprehensive backtesting and optimization system
to pre-inform the ML system and get optimized parameters for live trading.
"""
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.backtesting import BacktestOrchestrator
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.data.market_data import AlpacaDataProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demonstrate_backtesting_system():
    """
    Comprehensive demonstration of the ML-driven backtesting system
    """
    print("🚀 ML-Driven Backtesting System Demo")
    print("=" * 50)
    
    # 1. Initialize the backtesting orchestrator
    print("\n1. Initializing Backtesting System...")
    orchestrator = BacktestOrchestrator(
        cache_dir="data/backtest_demo",
        optimization_symbols=["BTC/USD", "ETH/USD"]
    )
    
    # 2. Run comprehensive parameter optimization
    print("\n2. Running Comprehensive Parameter Optimization...")
    optimization_results = orchestrator.run_comprehensive_optimization(
        strategy_class=MACrossoverStrategy,
        optimization_period_days=90,  # Use 90 days for demo (faster)
        validation_period_days=30,
        optimization_methods=['bayesian', 'genetic'],
        n_iterations_per_method=50  # Reduced for demo
    )
    
    print(f"✅ Optimization completed!")
    print(f"   Best overall score: {optimization_results['best_overall']['score']:.4f}")
    print(f"   Best symbol: {optimization_results['best_overall']['symbol']}")
    print(f"   Best parameters: {optimization_results['best_overall']['parameters']}")
    
    # 3. Train ML models from optimization results
    print("\n3. Training ML Models from Optimization Results...")
    orchestrator.train_ml_from_optimization_results(optimization_results)
    print("✅ ML models trained!")
    
    # 4. Demonstrate parameter prediction for live trading
    print("\n4. Getting Optimized Parameters for Live Trading...")
    
    # Get some recent market data for regime detection
    data_provider = AlpacaDataProvider()
    recent_data = await data_provider.get_historical_data(
        symbol="BTC/USD",
        timeframe="1Min",
        limit=1000  # Recent 1000 minutes
    )
    
    if not recent_data.empty:
        # Get ML-adaptive parameters
        ml_params = orchestrator.get_optimized_parameters_for_live_trading(
            current_market_data=recent_data,
            symbol="BTC/USD",
            adaptation_mode="ml_adaptive"
        )
        
        print(f"🧠 ML-Adaptive Parameters:")
        print(f"   Confidence: {ml_params['confidence']:.2f}")
        print(f"   Adaptation mode: {ml_params['adaptation_mode']}")
        print(f"   Parameters: {ml_params['parameters']}")
        
        # Compare with latest optimization results
        opt_params = orchestrator.get_optimized_parameters_for_live_trading(
            current_market_data=recent_data,
            symbol="BTC/USD", 
            adaptation_mode="latest_optimization"
        )
        
        print(f"\n📊 Latest Optimization Parameters:")
        print(f"   Confidence: {opt_params['confidence']:.2f}")
        print(f"   Parameters: {opt_params['parameters']}")
    
    # 5. Run walk-forward analysis for validation
    print("\n5. Running Walk-Forward Analysis...")
    wfa_results = orchestrator.run_walk_forward_analysis(
        strategy_class=MACrossoverStrategy,
        symbol="BTC/USD",
        total_days=180,  # Reduced for demo
        train_days=60,
        test_days=20,
        step_days=5
    )
    
    print(f"✅ Walk-forward analysis completed!")
    if wfa_results.get('analysis'):
        analysis = wfa_results['analysis']
        print(f"   Mean in-sample score: {analysis.get('mean_in_sample_score', 0):.4f}")
        print(f"   Mean out-of-sample score: {analysis.get('mean_out_of_sample_score', 0):.4f}")
        print(f"   Overfitting ratio: {analysis.get('overfitting_ratio', 0):.4f}")
        print(f"   Score stability: {analysis.get('score_stability', 0):.4f}")
    
    # 6. Get system status
    print("\n6. System Status Report...")
    status = orchestrator.get_system_status()
    
    print(f"📈 System Status:")
    print(f"   ML Models Trained: {status['ml_learning']['models_trained']}")
    print(f"   Learning Sessions: {status['ml_learning']['learning_sessions']}")
    print(f"   Optimization Sessions: {status['optimization_history']['total_sessions']}")
    print(f"   Cache Size: {status['cache_info']['total_size_mb']:.1f} MB")
    
    if 'ml_performance' in status:
        print(f"\n🧠 ML Model Performance:")
        for param, perf in status['ml_performance']['parameter_performance'].items():
            print(f"   {param}: MAE={perf['mae']:.4f}, Samples={perf['samples']}")
    
    print("\n🎉 Backtesting System Demo Complete!")
    print("\nThe system is now ready to provide optimized parameters for live trading.")

def demonstrate_live_trading_integration():
    """
    Show how the live trading system would integrate with backtesting
    """
    print("\n🔗 Live Trading Integration Example")
    print("=" * 40)
    
    # This simulates how main.py would use the backtesting system
    orchestrator = BacktestOrchestrator()
    
    # Simulate getting recent market data (this would come from live data feed)
    print("📊 Getting current market data...")
    
    # In practice, this would be live data from your data feed
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=1440, freq='1min'),
        'open': 50000 + np.random.randn(1440) * 100,
        'high': 50000 + np.random.randn(1440) * 100 + 50,
        'low': 50000 + np.random.randn(1440) * 100 - 50,
        'close': 50000 + np.random.randn(1440) * 100,
        'volume': np.random.randint(1000, 10000, 1440)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Get adaptive parameters for current market conditions
    adaptive_params = orchestrator.get_optimized_parameters_for_live_trading(
        current_market_data=sample_data,
        symbol="BTC/USD",
        adaptation_mode="ml_adaptive"
    )
    
    print(f"🎯 Adaptive Parameters for Live Trading:")
    print(f"   Fast MA: {adaptive_params['parameters']['fast_ma_period']}")
    print(f"   Slow MA: {adaptive_params['parameters']['slow_ma_period']}")
    print(f"   Risk per trade: {adaptive_params['parameters']['risk_per_trade']:.3f}")
    print(f"   Confidence threshold: {adaptive_params['parameters']['confidence_threshold']:.3f}")
    print(f"   ML Confidence: {adaptive_params['confidence']:.2f}")
    
    print("\n💡 Integration Notes:")
    print("   - Run optimization weekly/monthly to update ML models")
    print("   - Get adaptive parameters before each trading session")
    print("   - Use confidence scores to decide when to apply ML suggestions")
    print("   - Fall back to proven parameters when ML confidence is low")

if __name__ == "__main__":
    import asyncio
    import numpy as np
    
    print("Starting ML-Driven Backtesting Demo...")
    print("This will take a few minutes to run comprehensive optimization...")
    
    # Run the comprehensive demo
    asyncio.run(demonstrate_backtesting_system())
    
    # Show integration example
    demonstrate_live_trading_integration()