"""
Backtesting Orchestrator
Main interface for ML-driven parameter optimization and learning
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import asyncio
from pathlib import Path

from .historical_data import AlpacaHistoricalDataProvider
from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceAnalyzer
from .parameter_optimizer import ParameterOptimizer, ParameterSpace
from .ml_parameter_learner import MLParameterLearner
from ..strategy.ma_crossover_strategy import MACrossoverStrategy

logger = logging.getLogger(__name__)

class BacktestOrchestrator:
    """
    Main orchestrator for ML-driven backtesting and parameter optimization
    Provides high-level interface for strategy optimization and parameter learning
    """
    
    def __init__(
        self,
        cache_dir: str = "data/backtest_cache",
        optimization_symbols: List[str] = ["BTC/USD", "ETH/USD"]
    ):
        """
        Initialize backtesting orchestrator
        
        Args:
            cache_dir: Directory for caching results
            optimization_symbols: Symbols to use for parameter optimization
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_symbols = optimization_symbols
        
        # Initialize components
        self.data_provider = AlpacaHistoricalDataProvider(
            cache_dir=str(self.cache_dir / "historical_data")
        )
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        self.parameter_optimizer = ParameterOptimizer(
            self.data_provider,
            cache_dir=str(self.cache_dir / "optimization")
        )
        self.ml_learner = MLParameterLearner(
            model_cache_dir=str(self.cache_dir / "ml_models")
        )
        
        # Optimization state
        self.optimization_results = {}
        self.learning_sessions = []
        
        logger.info(f"Backtest orchestrator initialized for symbols: {optimization_symbols}")
    
    def run_comprehensive_optimization(
        self,
        strategy_class=MACrossoverStrategy,
        optimization_period_days: int = 180,
        validation_period_days: int = 60,
        optimization_methods: List[str] = ['bayesian', 'genetic'],
        n_iterations_per_method: int = 100
    ) -> Dict[str, Any]:
        """
        Run comprehensive parameter optimization across multiple methods and symbols
        
        Args:
            strategy_class: Strategy class to optimize
            optimization_period_days: Days of data for optimization
            validation_period_days: Days for out-of-sample validation
            optimization_methods: List of optimization methods to try
            n_iterations_per_method: Iterations per optimization method
            
        Returns:
            Comprehensive optimization results
        """
        logger.info("Starting comprehensive parameter optimization")
        
        # Initialize ML models if not already done
        parameter_space = self.parameter_optimizer.create_default_parameter_space()
        self.ml_learner.initialize_models(parameter_space)
        
        optimization_results = {
            'timestamp': datetime.now(),
            'strategy': strategy_class.__name__,
            'symbols': {},
            'best_overall': None,
            'method_comparison': {}
        }
        
        # Run optimization for each symbol
        for symbol in self.optimization_symbols:
            logger.info(f"Optimizing {symbol}")
            
            symbol_results = {
                'symbol': symbol,
                'methods': {},
                'best_parameters': None,
                'best_score': -np.inf
            }
            
            # Try each optimization method
            for method in optimization_methods:
                logger.info(f"Running {method} optimization for {symbol}")
                
                try:
                    strategy = strategy_class()
                    
                    method_results = self.parameter_optimizer.optimize_parameters(
                        strategy=strategy,
                        symbol=symbol,
                        parameter_space=parameter_space,
                        optimization_method=method,
                        n_iterations=n_iterations_per_method,
                        objective='sharpe_ratio',
                        data_period_days=optimization_period_days,
                        validation_split=validation_period_days / optimization_period_days
                    )
                    
                    symbol_results['methods'][method] = method_results
                    
                    # Track best parameters
                    score = method_results.get('best_score', -np.inf)
                    if score > symbol_results['best_score']:
                        symbol_results['best_score'] = score
                        symbol_results['best_parameters'] = method_results.get('best_parameters')
                        symbol_results['best_method'] = method
                    
                    logger.info(f"{method} optimization for {symbol} completed: score={score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in {method} optimization for {symbol}: {e}")
                    symbol_results['methods'][method] = {'error': str(e)}
            
            optimization_results['symbols'][symbol] = symbol_results
        
        # Find best overall parameters across all symbols
        best_overall_score = -np.inf
        best_overall_params = None
        best_symbol = None
        
        for symbol, results in optimization_results['symbols'].items():
            if results['best_score'] > best_overall_score:
                best_overall_score = results['best_score']
                best_overall_params = results['best_parameters']
                best_symbol = symbol
        
        optimization_results['best_overall'] = {
            'parameters': best_overall_params,
            'score': best_overall_score,
            'symbol': best_symbol
        }
        
        # Analyze method performance
        method_scores = {method: [] for method in optimization_methods}
        for symbol_results in optimization_results['symbols'].values():
            for method, results in symbol_results['methods'].items():
                if 'best_score' in results:
                    method_scores[method].append(results['best_score'])
        
        for method, scores in method_scores.items():
            if scores:
                optimization_results['method_comparison'][method] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'best_score': max(scores),
                    'success_rate': len(scores) / len(self.optimization_symbols)
                }
        
        # Store results
        self.optimization_results = optimization_results
        self._save_optimization_results(optimization_results)
        
        logger.info(f"Comprehensive optimization completed. Best score: {best_overall_score:.4f}")
        return optimization_results
    
    def train_ml_from_optimization_results(
        self,
        optimization_results: Dict[str, Any] = None,
        learning_symbols: List[str] = None
    ):
        """
        Train ML models from optimization results
        
        Args:
            optimization_results: Results from optimization (uses latest if None)
            learning_symbols: Symbols to learn from (uses all if None)
        """
        if optimization_results is None:
            optimization_results = self.optimization_results
        
        if not optimization_results:
            logger.error("No optimization results available for ML training")
            return
        
        if learning_symbols is None:
            learning_symbols = self.optimization_symbols
        
        logger.info(f"Training ML models from optimization results for {len(learning_symbols)} symbols")
        
        parameter_space = self.parameter_optimizer.create_default_parameter_space()
        
        # Train on results from each symbol
        for symbol in learning_symbols:
            symbol_results = optimization_results['symbols'].get(symbol)
            if not symbol_results:
                continue
            
            # Get historical data for this symbol
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # Use more data for learning
                
                market_data = self.data_provider.fetch_historical_data(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe='1Min'
                )
                
                if market_data.empty:
                    logger.warning(f"No market data for {symbol}")
                    continue
                
                # Learn from each optimization method's results
                for method, method_results in symbol_results['methods'].items():
                    if 'optimization_history' in method_results:
                        logger.info(f"Learning from {method} results for {symbol}")
                        
                        self.ml_learner.learn_from_backtest_results(
                            optimization_results=method_results,
                            market_data=market_data,
                            target_metric='sharpe_ratio'
                        )
                
            except Exception as e:
                logger.error(f"Error training ML for {symbol}: {e}")
                continue
        
        # Save trained models
        model_path = self.ml_learner.save_models()
        
        # Record learning session
        learning_session = {
            'timestamp': datetime.now(),
            'symbols': learning_symbols,
            'optimization_source': optimization_results.get('timestamp'),
            'model_path': model_path
        }
        
        self.learning_sessions.append(learning_session)
        
        logger.info("ML training completed")
    
    def get_optimized_parameters_for_live_trading(
        self,
        current_market_data: pd.DataFrame,
        symbol: str = "BTC/USD",
        adaptation_mode: str = "ml_adaptive"
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for live trading based on current market conditions
        
        Args:
            current_market_data: Recent market data for regime detection
            symbol: Trading symbol
            adaptation_mode: 'ml_adaptive', 'latest_optimization', 'default'
            
        Returns:
            Optimized parameters with confidence scores
        """
        parameter_space = self.parameter_optimizer.create_default_parameter_space()
        
        if adaptation_mode == "ml_adaptive" and self.ml_learner.parameter_models:
            # Use ML predictions based on current market conditions
            logger.info("Using ML-adaptive parameter selection")
            
            result = self.ml_learner.predict_optimal_parameters(
                current_market_data=current_market_data,
                parameter_space=parameter_space,
                confidence_threshold=0.6
            )
            
            result['adaptation_mode'] = 'ml_adaptive'
            return result
            
        elif adaptation_mode == "latest_optimization" and self.optimization_results:
            # Use best parameters from latest optimization
            logger.info("Using latest optimization results")
            
            best_params = self.optimization_results.get('best_overall', {}).get('parameters')
            if best_params:
                return {
                    'parameters': best_params,
                    'confidence': 0.8,
                    'adaptation_mode': 'latest_optimization',
                    'source': 'optimization_results'
                }
        
        # Fall back to default parameters
        logger.info("Using default parameters")
        defaults = self._get_default_parameters(parameter_space)
        defaults['adaptation_mode'] = 'default'
        return defaults
    
    def run_walk_forward_analysis(
        self,
        strategy_class=MACrossoverStrategy,
        symbol: str = "BTC/USD",
        total_days: int = 365,
        train_days: int = 90,
        test_days: int = 30,
        step_days: int = 7
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis to validate parameter stability
        
        Args:
            strategy_class: Strategy to analyze
            symbol: Trading symbol
            total_days: Total analysis period
            train_days: Training period length
            test_days: Testing period length
            step_days: Step size between periods
            
        Returns:
            Walk-forward analysis results
        """
        logger.info(f"Running walk-forward analysis for {symbol}")
        
        parameter_space = self.parameter_optimizer.create_default_parameter_space()
        strategy = strategy_class()
        
        wfa_results = self.parameter_optimizer.walk_forward_analysis(
            strategy=strategy,
            symbol=symbol,
            parameter_space=parameter_space,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            total_days=total_days
        )
        
        # Analyze results for ML learning
        if wfa_results.get('walk_forward_results'):
            logger.info("Training ML from walk-forward results")
            
            # Convert walk-forward results to optimization format for ML learning
            for period_result in wfa_results['walk_forward_results']:
                try:
                    # Get market data for this period
                    start_date = period_result['train_start']
                    end_date = period_result['train_end']
                    
                    market_data = self.data_provider.fetch_historical_data(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe='1Min'
                    )
                    
                    if not market_data.empty:
                        # Create pseudo-optimization results for ML learning
                        pseudo_results = {
                            'optimization_history': [{
                                'parameters': period_result['optimal_parameters'],
                                'score': period_result['out_of_sample_score'],
                                'iteration': 0
                            }]
                        }
                        
                        self.ml_learner.learn_from_backtest_results(
                            optimization_results=pseudo_results,
                            market_data=market_data,
                            target_metric='sharpe_ratio'
                        )
                        
                except Exception as e:
                    logger.warning(f"Error learning from WFA period: {e}")
                    continue
        
        return wfa_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now(),
            'components': {
                'data_provider': 'initialized',
                'backtest_engine': 'initialized',
                'parameter_optimizer': 'initialized',
                'ml_learner': 'initialized' if self.ml_learner.parameter_models else 'not_trained'
            },
            'optimization_history': {
                'total_sessions': len(self.optimization_results),
                'last_optimization': self.optimization_results.get('timestamp'),
                'symbols_optimized': list(self.optimization_results.get('symbols', {}).keys())
            },
            'ml_learning': {
                'learning_sessions': len(self.learning_sessions),
                'models_trained': len(self.ml_learner.parameter_models),
                'last_learning': self.learning_sessions[-1]['timestamp'] if self.learning_sessions else None
            },
            'cache_info': self.data_provider.get_cache_info()
        }
        
        # Add ML model performance if available
        if self.ml_learner.parameter_models:
            status['ml_performance'] = self.ml_learner.get_model_performance_report()
        
        return status
    
    def _get_default_parameters(self, parameter_space: ParameterSpace) -> Dict[str, Any]:
        """Get sensible default parameters"""
        return {
            'parameters': {
                'fast_ma_period': 10,
                'slow_ma_period': 30,
                'risk_per_trade': 0.02,
                'confidence_threshold': 0.6,
                'volume_threshold': 1.5,
                'mfi_oversold': 20,
                'mfi_overbought': 80,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2.0
            },
            'confidence': 0.5,
            'source': 'default_parameters'
        }
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results to cache"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_results_{timestamp}.json"
        filepath = self.cache_dir / filename
        
        try:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Error saving optimization results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj