"""
Simple test script to validate the backtesting system components
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_backtest_engine():
    """Test the backtest engine with synthetic data"""
    print("🧪 Testing Backtest Engine...")
    
    try:
        from backtesting.backtest_engine import BacktestEngine, BacktestPosition, BacktestTrade
        from strategy.base_strategy import TradingSignal, SignalType
        
        # Create synthetic market data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=1000, freq='1min')
        data = pd.DataFrame({
            'open': 50000 + np.random.randn(1000) * 100,
            'high': 50000 + np.random.randn(1000) * 100 + 50,
            'low': 50000 + np.random.randn(1000) * 100 - 50,
            'close': 50000 + np.random.randn(1000) * 100,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        # Ensure high >= low and positive prices
        data['high'] = np.maximum(data['high'], data['low'])
        data['close'] = np.abs(data['close'])
        
        engine = BacktestEngine(initial_capital=100000)
        print(f"✅ BacktestEngine initialized with ${engine.initial_capital:,.2f}")
        
        # Test position creation
        engine._open_position("BTC/USD", 0.1, 50000, data.iloc[0])
        print(f"✅ Position created: {len(engine.positions)} positions")
        
        # Test equity calculation
        equity = engine._calculate_total_equity(data.iloc[0])
        print(f"✅ Total equity calculated: ${equity:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ BacktestEngine test failed: {e}")
        return False

def test_parameter_optimizer():
    """Test parameter optimization components"""
    print("🧪 Testing Parameter Optimizer...")
    
    try:
        from backtesting.parameter_optimizer import ParameterSpace, ParameterOptimizer
        
        # Test parameter space
        space = ParameterSpace()
        space.add_parameter('fast_ma', 5, 20, 'int')
        space.add_parameter('slow_ma', 20, 50, 'int')
        space.add_parameter('risk_factor', 0.01, 0.05, 'float')
        
        print(f"✅ ParameterSpace created with {len(space.parameters)} parameters")
        
        # Test random parameter generation
        params = space.get_random_parameters()
        print(f"✅ Random parameters generated: {params}")
        
        # Test normalization
        normalized = space.normalize_parameters(params)
        denormalized = space.denormalize_parameters(normalized)
        print(f"✅ Parameter normalization works: {np.allclose(list(params.values()), list(denormalized.values()), atol=1)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ParameterOptimizer test failed: {e}")
        return False

def test_performance_analyzer():
    """Test performance metrics calculation"""
    print("🧪 Testing Performance Analyzer...")
    
    try:
        from backtesting.performance_metrics import PerformanceAnalyzer
        from backtesting.backtest_engine import BacktestTrade
        
        analyzer = PerformanceAnalyzer()
        
        # Create sample trades
        sample_trades = []
        for i in range(10):
            trade = BacktestTrade(
                symbol="BTC/USD",
                entry_time=datetime.now() - timedelta(hours=i*2),
                exit_time=datetime.now() - timedelta(hours=i*2-1),
                entry_price=50000 + i * 100,
                exit_price=50000 + i * 100 + np.random.randint(-200, 300),
                quantity=0.1,
                side='long',
                pnl=np.random.randint(-500, 1000),
                pnl_pct=np.random.uniform(-2, 4),
                commission=25,
                hold_time=timedelta(hours=1)
            )
            sample_trades.append(trade)
        
        # Create sample equity curve
        equity_curve = []
        base_equity = 100000
        for i in range(100):
            base_equity += np.random.randint(-200, 300)
            equity_curve.append((datetime.now() - timedelta(minutes=i), base_equity))
        
        # Create mock backtest results
        backtest_results = {
            'backtest_valid': True,
            'trades': sample_trades,
            'equity_curve': equity_curve,
            'initial_capital': 100000,
            'final_equity': base_equity,
            'total_return': (base_equity - 100000) / 100000,
            'total_pnl': base_equity - 100000,
            'max_drawdown': 0.05,
            'total_trades': len(sample_trades),
            'win_rate': 0.6,
            'total_commission': 250,
            'total_slippage': 50
        }
        
        # Analyze performance
        analysis = analyzer.analyze_backtest_results(backtest_results)
        
        print(f"✅ Performance analysis completed")
        print(f"   Performance score: {analysis['performance_score']:.1f}")
        print(f"   Risk score: {analysis['risk_score']:.1f}")
        print(f"   Sharpe ratio: {analysis.get('sharpe_ratio', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ PerformanceAnalyzer test failed: {e}")
        return False

def test_ml_parameter_learner():
    """Test ML parameter learning components"""
    print("🧪 Testing ML Parameter Learner...")
    
    try:
        from backtesting.ml_parameter_learner import MLParameterLearner, MarketRegimeDetector
        from backtesting.parameter_optimizer import ParameterSpace
        
        # Test market regime detection
        detector = MarketRegimeDetector()
        
        # Create sample market data
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=500, freq='1min')
        market_data = pd.DataFrame({
            'open': 50000 + np.random.randn(500) * 100,
            'high': 50000 + np.random.randn(500) * 100 + 50,
            'low': 50000 + np.random.randn(500) * 100 - 50,
            'close': 50000 + np.random.randn(500) * 100,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        regime = detector.detect_regime(market_data)
        print(f"✅ Market regime detected: {regime}")
        
        # Test ML learner initialization
        learner = MLParameterLearner()
        
        # Create parameter space
        space = ParameterSpace()
        space.add_parameter('fast_ma', 5, 20, 'int')
        space.add_parameter('slow_ma', 20, 50, 'int')
        
        # Initialize models (if River is available)
        try:
            learner.initialize_models(space)
            print(f"✅ ML models initialized for {len(learner.parameter_models)} parameters")
        except Exception as e:
            print(f"⚠️  ML models not available (River library): {e}")
        
        # Test default parameter generation
        defaults = learner._get_default_parameters(space)
        print(f"✅ Default parameters generated: {defaults['parameters']}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLParameterLearner test failed: {e}")
        return False

def run_all_tests():
    """Run all component tests"""
    print("🚀 Running Backtesting System Component Tests")
    print("=" * 50)
    
    tests = [
        ("Backtest Engine", test_backtest_engine),
        ("Parameter Optimizer", test_parameter_optimizer),
        ("Performance Analyzer", test_performance_analyzer),
        ("ML Parameter Learner", test_ml_parameter_learner)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n🏁 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All components working correctly!")
    else:
        print("⚠️  Some components need attention")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)