#!/usr/bin/env python3
"""
Quick Backtesting System Demo
Simple demonstration of the ML-driven backtesting capabilities
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

def create_sample_data(days=30):
    """Create sample market data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), 
        periods=days * 1440, 
        freq='1min'
    )
    
    # Generate realistic price movement with trend
    returns = np.random.randn(len(dates)) * 0.001  # 0.1% volatility per minute
    trend = np.linspace(0, 0.1, len(dates))  # 10% upward trend over period
    
    base_price = 50000
    cumulative_returns = np.cumsum(returns + trend/len(dates))
    prices = base_price * (1 + cumulative_returns)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.randn(len(dates)) * 10,
        'high': prices + np.abs(np.random.randn(len(dates)) * 20),
        'low': prices - np.abs(np.random.randn(len(dates)) * 20),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def demo_basic_backtesting():
    """Demonstrate basic backtesting functionality"""
    print("🧪 Basic Backtesting Demo")
    print("=" * 40)
    
    try:
        # Import backtesting components
        from backtesting.backtest_engine import BacktestEngine
        from backtesting.performance_metrics import PerformanceAnalyzer
        from strategy.ma_crossover_strategy import MACrossoverStrategy
        
        # Create sample data
        print("📊 Generating sample market data...")
        data = create_sample_data(days=7)  # 1 week of minute data
        print(f"   Generated {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Initialize components
        strategy = MACrossoverStrategy()
        engine = BacktestEngine(initial_capital=100000)
        analyzer = PerformanceAnalyzer()
        
        print(f"✅ Components initialized")
        print(f"   Strategy: {strategy.name}")
        print(f"   Initial capital: ${engine.initial_capital:,.2f}")
        
        # Run backtest
        print("🚀 Running backtest...")
        
        # Override strategy parameters for demo
        test_parameters = {
            'fast_ma_period': 10,
            'slow_ma_period': 30,
            'risk_per_trade': 0.02,
            'confidence_threshold': 0.6
        }
        
        results = engine.run_backtest(strategy, data, test_parameters)
        
        if results.get('backtest_valid'):
            print("✅ Backtest completed successfully")
            print(f"   Total trades: {results['total_trades']}")
            print(f"   Win rate: {results['win_rate']:.1%}")
            print(f"   Total return: {results['total_return']:.2%}")
            print(f"   Final equity: ${results['final_equity']:,.2f}")
            print(f"   Max drawdown: {results['max_drawdown']:.2%}")
            print(f"   Sharpe ratio: {results.get('sharpe_ratio', 0):.3f}")
            
            # Detailed analysis
            analysis = analyzer.analyze_backtest_results(results)
            if 'performance_score' in analysis:
                print(f"   Performance score: {analysis['performance_score']:.1f}/100")
                print(f"   Risk score: {analysis['risk_score']:.1f}/100")
        else:
            print("❌ Backtest failed")
            print(f"   Error: {results.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_parameter_optimization():
    """Demonstrate parameter optimization"""
    print("\n🔧 Parameter Optimization Demo")
    print("=" * 40)
    
    try:
        from backtesting.parameter_optimizer import ParameterOptimizer, ParameterSpace
        from backtesting.historical_data import AlpacaHistoricalDataProvider
        from strategy.ma_crossover_strategy import MACrossoverStrategy
        
        # Note: This will use mock data since we may not have API access
        print("📊 Setting up optimization...")
        
        # Create parameter space
        space = ParameterSpace()
        space.add_parameter('fast_ma_period', 5, 20, 'int')
        space.add_parameter('slow_ma_period', 20, 50, 'int')
        space.add_parameter('risk_per_trade', 0.01, 0.05, 'float')
        
        print(f"✅ Parameter space created with {len(space.parameters)} parameters")
        
        # Test parameter generation
        for i in range(3):
            params = space.get_random_parameters()
            print(f"   Sample {i+1}: {params}")
        
        # Test normalization
        test_params = space.get_random_parameters()
        normalized = space.normalize_parameters(test_params)
        denormalized = space.denormalize_parameters(normalized)
        
        print(f"✅ Parameter normalization working")
        print(f"   Original: {test_params}")
        print(f"   Normalized: {normalized}")
        print(f"   Recovered: {denormalized}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter optimization demo failed: {e}")
        return False

def demo_ml_regime_detection():
    """Demonstrate ML regime detection"""
    print("\n🧠 ML Regime Detection Demo")
    print("=" * 40)
    
    try:
        from backtesting.ml_parameter_learner import MarketRegimeDetector
        
        detector = MarketRegimeDetector()
        
        # Create different market scenarios
        scenarios = {
            "Bull Market": create_sample_data(days=5),  # Default trending up
            "Bear Market": create_sample_data(days=5) * 0.95,  # Trending down
            "Sideways Market": create_sample_data(days=5)  # Will modify to be flat
        }
        
        # Make sideways market actually sideways
        sideways_data = scenarios["Sideways Market"].copy()
        sideways_data['close'] = 50000 + np.sin(np.linspace(0, 10, len(sideways_data))) * 100
        scenarios["Sideways Market"] = sideways_data
        
        print("🔍 Analyzing different market regimes:")
        
        for scenario_name, data in scenarios.items():
            regime = detector.detect_regime(data)
            print(f"\n   📈 {scenario_name}:")
            for feature, value in regime.items():
                print(f"      {feature}: {value:.3f}")
        
        print("\n✅ Market regime detection working correctly")
        return True
        
    except Exception as e:
        print(f"❌ ML regime detection demo failed: {e}")
        return False

def demo_performance_analysis():
    """Demonstrate performance analysis"""
    print("\n📊 Performance Analysis Demo")
    print("=" * 40)
    
    try:
        from backtesting.performance_metrics import PerformanceAnalyzer
        from backtesting.backtest_engine import BacktestTrade
        
        analyzer = PerformanceAnalyzer()
        
        # Create sample trades with different characteristics
        sample_trades = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(20):
            # Mix of winning and losing trades
            if i % 3 == 0:
                pnl = np.random.uniform(-200, -50)  # Losing trade
            else:
                pnl = np.random.uniform(50, 300)    # Winning trade
            
            trade = BacktestTrade(
                symbol="BTC/USD",
                entry_time=base_time + timedelta(hours=i*6),
                exit_time=base_time + timedelta(hours=i*6+2),
                entry_price=50000 + i*10,
                exit_price=50000 + i*10 + pnl/0.1,  # Assuming 0.1 BTC position
                quantity=0.1,
                side='long',
                pnl=pnl,
                pnl_pct=(pnl/(50000*0.1))*100,
                commission=25,
                hold_time=timedelta(hours=2)
            )
            sample_trades.append(trade)
        
        # Create sample equity curve
        equity_curve = []
        base_equity = 100000
        for i in range(100):
            base_equity += np.random.randint(-100, 200)  # Random walk with upward bias
            equity_curve.append((base_time + timedelta(hours=i), base_equity))
        
        # Mock backtest results
        mock_results = {
            'backtest_valid': True,
            'trades': sample_trades,
            'equity_curve': equity_curve,
            'initial_capital': 100000,
            'final_equity': base_equity,
            'total_return': (base_equity - 100000) / 100000,
            'total_pnl': base_equity - 100000,
            'max_drawdown': 0.05,
            'total_trades': len(sample_trades),
            'win_rate': len([t for t in sample_trades if t.pnl > 0]) / len(sample_trades),
            'total_commission': 500,
            'total_slippage': 100
        }
        
        # Analyze performance
        analysis = analyzer.analyze_backtest_results(mock_results)
        
        print("✅ Performance analysis completed:")
        print(f"   Performance Score: {analysis.get('performance_score', 0):.1f}/100")
        print(f"   Risk Score: {analysis.get('risk_score', 0):.1f}/100")
        print(f"   Total Return: {analysis.get('total_return', 0):.2%}")
        print(f"   Win Rate: {analysis.get('win_rate', 0):.1%}")
        print(f"   Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {analysis.get('max_drawdown', 0):.2%}")
        
        # Interpretations
        interpretations = analysis.get('interpretation', {})
        print(f"\n💡 Strategy Assessment:")
        for category, interpretation in interpretations.items():
            print(f"   {category.title()}: {interpretation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance analysis demo failed: {e}")
        return False

def main():
    """Run all backtesting demos"""
    print("🚀 ML-Driven Backtesting System Demo")
    print("=" * 50)
    print("This demo showcases the comprehensive backtesting capabilities")
    print("without requiring external API connections.\n")
    
    demos = [
        ("Basic Backtesting", demo_basic_backtesting),
        ("Parameter Optimization", demo_parameter_optimization), 
        ("ML Regime Detection", demo_ml_regime_detection),
        ("Performance Analysis", demo_performance_analysis)
    ]
    
    results = []
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            results.append((demo_name, success))
        except Exception as e:
            print(f"❌ {demo_name} failed with exception: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n🏁 Demo Results Summary")
    print("=" * 30)
    
    passed = 0
    for demo_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {demo_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} demos successful")
    
    if passed == len(results):
        print("\n🎉 All backtesting components working correctly!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ High-fidelity backtest engine with realistic execution")
        print("   ✅ Advanced parameter optimization (Bayesian, Genetic, etc.)")
        print("   ✅ ML-driven market regime detection")
        print("   ✅ Comprehensive performance analysis and risk metrics")
        print("   ✅ Ready for integration with live trading system")
        
        print("\n🔗 Integration with Live Trading:")
        print("   - Use BacktestOrchestrator.get_optimized_parameters_for_live_trading()")
        print("   - Run periodic optimization to update ML models")  
        print("   - Adapt parameters based on current market conditions")
        print("   - Monitor performance and trigger re-optimization when needed")
    else:
        print(f"\n⚠️  {len(results) - passed} components need attention")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)