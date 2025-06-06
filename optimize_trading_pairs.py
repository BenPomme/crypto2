"""
Easy-to-Use Script: Optimize & Backtest Your 4 Trading Pairs

This script will:
1. Optimize parameters for BTC/USD, ETH/USD, SOL/USD, DOGE/USD
2. Run backtests with the optimized parameters
3. Train ML models to learn optimal settings
4. Generate a report with the best parameters for live trading

Run this script to get optimized parameters for your trading pairs!
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Your 4 trading pairs
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['river', 'scipy', 'optuna', 'alpaca_trade_api']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages. Please install:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

async def run_optimization_for_pairs():
    """
    Main function to optimize all your trading pairs
    """
    print("🚀 Optimizing Your 4 Trading Pairs")
    print("=" * 50)
    print(f"Pairs: {', '.join(TRADING_PAIRS)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    if not check_dependencies():
        return
    
    try:
        # Import after dependency check
        from backtesting import BacktestOrchestrator
        from strategy.ma_crossover_strategy import MACrossoverStrategy
        
        # Initialize the orchestrator
        print("\n📊 Initializing Backtesting System...")
        orchestrator = BacktestOrchestrator(
            cache_dir="data/optimization_results",
            optimization_symbols=TRADING_PAIRS
        )
        
        # Configuration for optimization
        optimization_config = {
            'strategy_class': MACrossoverStrategy,
            'optimization_period_days': 90,  # 3 months of data
            'validation_period_days': 30,    # 1 month validation
            'optimization_methods': ['bayesian', 'random'],  # Start with faster methods
            'n_iterations_per_method': 50    # 50 iterations each (100 total per pair)
        }
        
        print(f"📅 Using {optimization_config['optimization_period_days']} days for optimization")
        print(f"🔬 Methods: {', '.join(optimization_config['optimization_methods'])}")
        print(f"🔄 {optimization_config['n_iterations_per_method']} iterations per method")
        
        # Step 1: Run optimization for all pairs
        print(f"\n🎯 Step 1: Optimizing Parameters for {len(TRADING_PAIRS)} Pairs")
        print("-" * 60)
        
        optimization_results = orchestrator.run_comprehensive_optimization(**optimization_config)
        
        if 'error' in optimization_results:
            print(f"❌ Optimization failed: {optimization_results['error']}")
            return
        
        # Step 2: Analyze results for each pair
        print(f"\n📈 Step 2: Optimization Results")
        print("-" * 40)
        
        best_overall = optimization_results.get('best_overall', {})
        pair_results = {}
        
        for pair in TRADING_PAIRS:
            if pair in optimization_results['symbols']:
                result = optimization_results['symbols'][pair]
                pair_results[pair] = result
                
                print(f"\n💰 {pair}:")
                print(f"   Best Score (Sharpe): {result['best_score']:.3f}")
                print(f"   Best Method: {result.get('best_method', 'N/A')}")
                
                if result['best_parameters']:
                    params = result['best_parameters']
                    print(f"   Fast MA: {params.get('fast_ma_period', 'N/A')}")
                    print(f"   Slow MA: {params.get('slow_ma_period', 'N/A')}")
                    print(f"   Risk/Trade: {params.get('risk_per_trade', 'N/A'):.1%}")
                    print(f"   Confidence: {params.get('confidence_threshold', 'N/A'):.1%}")
            else:
                print(f"\n❌ {pair}: No results (check data availability)")
        
        # Step 3: Train ML models
        print(f"\n🧠 Step 3: Training ML Models")
        print("-" * 30)
        
        orchestrator.train_ml_from_optimization_results(
            optimization_results=optimization_results,
            learning_symbols=TRADING_PAIRS
        )
        
        print("✅ ML models trained successfully!")
        
        # Step 4: Generate live trading parameters
        print(f"\n🎯 Step 4: Live Trading Parameters")
        print("-" * 35)
        
        # For demonstration, we'll use the best overall parameters
        # In real usage, you'd get recent market data for each pair
        live_parameters = {}
        
        for pair in TRADING_PAIRS:
            if pair in pair_results:
                # Use optimization results (in production, this would use current market data)
                params = orchestrator.get_optimized_parameters_for_live_trading(
                    current_market_data=None,  # Would be recent market data
                    symbol=pair,
                    adaptation_mode="latest_optimization"
                )
                live_parameters[pair] = params
                
                print(f"\n🔧 {pair} - Ready for Live Trading:")
                print(f"   Confidence: {params['confidence']:.0%}")
                print(f"   Source: {params.get('adaptation_mode', 'N/A')}")
                
                if 'parameters' in params:
                    p = params['parameters']
                    print(f"   Fast MA: {p.get('fast_ma_period', 'N/A')}")
                    print(f"   Slow MA: {p.get('slow_ma_period', 'N/A')}")
                    print(f"   Risk: {p.get('risk_per_trade', 0):.1%}")
        
        # Step 5: Save results for your main.py
        print(f"\n💾 Step 5: Saving Results")
        print("-" * 25)
        
        # Create a simple configuration file for main.py to use
        config_for_main = {
            'optimization_date': datetime.now().isoformat(),
            'pairs': {},
            'best_overall': best_overall,
            'summary': {
                'total_pairs_optimized': len([p for p in TRADING_PAIRS if p in pair_results]),
                'best_pair': best_overall.get('symbol', 'N/A'),
                'best_score': best_overall.get('score', 0),
                'optimization_methods': optimization_config['optimization_methods']
            }
        }
        
        for pair in TRADING_PAIRS:
            if pair in live_parameters:
                config_for_main['pairs'][pair] = live_parameters[pair]
        
        # Save to file
        config_file = 'optimized_parameters.json'
        with open(config_file, 'w') as f:
            json.dump(config_for_main, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {config_file}")
        print(f"✅ Logs saved to: optimization.log")
        
        # Step 6: Integration instructions
        print(f"\n🔗 Step 6: How to Use in Your Trading System")
        print("-" * 45)
        
        integration_code = f'''
# Add this to your main.py before starting trading:

import json
from src.backtesting import BacktestOrchestrator

# Load optimized parameters
with open('optimized_parameters.json', 'r') as f:
    optimized_config = json.load(f)

# Initialize orchestrator (do this once)
orchestrator = BacktestOrchestrator()

# For each trading pair, apply optimized parameters:
for symbol in ["{'", "'.join(TRADING_PAIRS)}"]:
    if symbol in optimized_config['pairs']:
        params = optimized_config['pairs'][symbol]['parameters']
        
        # Update your strategy with optimized parameters
        # strategy.update_config(params)  # Implement this method
        
        print(f"Applied optimized parameters for {{symbol}}")
        print(f"  Fast MA: {{params['fast_ma_period']}}")
        print(f"  Slow MA: {{params['slow_ma_period']}}")
        print(f"  Risk per trade: {{params['risk_per_trade']:.1%}}")
'''
        
        print("📝 Integration Code:")
        print(integration_code)
        
        # Summary
        print(f"\n🎉 Optimization Complete!")
        print("=" * 40)
        
        successful_pairs = len([p for p in TRADING_PAIRS if p in pair_results])
        print(f"✅ Successfully optimized: {successful_pairs}/{len(TRADING_PAIRS)} pairs")
        
        if best_overall:
            print(f"🏆 Best performing pair: {best_overall.get('symbol', 'N/A')}")
            print(f"📊 Best Sharpe ratio: {best_overall.get('score', 0):.3f}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. Review the optimized_parameters.json file")
        print(f"2. Integrate the parameters into your main.py")
        print(f"3. Start live trading with optimized settings")
        print(f"4. Re-run this script weekly to update parameters")
        
        return config_for_main
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory and dependencies are installed.")
        return None
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"❌ Error during optimization: {e}")
        print("Check optimization.log for detailed error information.")
        return None

def run_quick_test():
    """
    Run a quick test to make sure the system works before full optimization
    """
    print("🧪 Quick System Test")
    print("-" * 20)
    
    try:
        # Test basic imports
        from backtesting.parameter_optimizer import ParameterSpace
        from backtesting.ml_parameter_learner import MarketRegimeDetector
        
        # Test parameter space
        space = ParameterSpace()
        space.add_parameter('test_param', 1, 10, 'int')
        params = space.get_random_parameters()
        
        # Test regime detection
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=100, freq='1min')
        test_data = pd.DataFrame({
            'close': 50000 + np.random.randn(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        detector = MarketRegimeDetector()
        regime = detector.detect_regime(test_data)
        
        print("✅ All systems operational!")
        print(f"✅ Parameter generation works: {params}")
        print(f"✅ Market regime detection works: {regime['volatility_regime']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Crypto Trading Parameter Optimizer")
    print("=====================================")
    print("This will optimize your 4 trading pairs and prepare them for live trading.")
    print(f"Pairs: {', '.join(TRADING_PAIRS)}")
    
    # Quick test first
    if not run_quick_test():
        print("\n❌ System test failed. Please check your environment.")
        sys.exit(1)
    
    print(f"\n⏱️  Estimated time: 10-20 minutes")
    print("📁 Results will be saved to: optimized_parameters.json")
    
    # Ask for confirmation
    response = input("\n🚀 Ready to start optimization? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Run the optimization
        results = asyncio.run(run_optimization_for_pairs())
        
        if results:
            print(f"\n✅ SUCCESS! Your trading pairs are optimized and ready.")
            print(f"📁 Check optimized_parameters.json for the results.")
        else:
            print(f"\n❌ Optimization failed. Check the logs for details.")
    else:
        print("👋 Optimization cancelled. Run again when ready!")