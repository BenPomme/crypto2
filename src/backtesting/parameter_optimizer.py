"""
Advanced Parameter Optimization System
Intelligent parameter search using Bayesian optimization, genetic algorithms, and ML
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
from pathlib import Path

# ML libraries for optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
import optuna

from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceAnalyzer
from .historical_data import AlpacaHistoricalDataProvider

logger = logging.getLogger(__name__)

class ParameterSpace:
    """Define parameter search space with bounds and types"""
    
    def __init__(self):
        self.parameters = {}
        self.bounds = {}
        self.types = {}
    
    def add_parameter(self, name: str, min_val: float, max_val: float, param_type: str = 'float'):
        """
        Add parameter to search space
        
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
            param_type: 'float', 'int', or 'categorical'
        """
        self.parameters[name] = (min_val, max_val)
        self.bounds[name] = (min_val, max_val)
        self.types[name] = param_type
    
    def get_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds"""
        params = {}
        for name, (min_val, max_val) in self.parameters.items():
            if self.types[name] == 'int':
                params[name] = np.random.randint(min_val, max_val + 1)
            elif self.types[name] == 'float':
                params[name] = np.random.uniform(min_val, max_val)
        return params
    
    def normalize_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Normalize parameters to [0, 1] range for optimization"""
        normalized = []
        for name in sorted(self.parameters.keys()):
            min_val, max_val = self.parameters[name]
            normalized.append((params[name] - min_val) / (max_val - min_val))
        return np.array(normalized)
    
    def denormalize_parameters(self, normalized: np.ndarray) -> Dict[str, Any]:
        """Convert normalized parameters back to original range"""
        params = {}
        for i, name in enumerate(sorted(self.parameters.keys())):
            min_val, max_val = self.parameters[name]
            value = normalized[i] * (max_val - min_val) + min_val
            
            if self.types[name] == 'int':
                value = int(round(value))
            
            params[name] = value
        return params

class ParameterOptimizer:
    """
    Advanced parameter optimization using multiple algorithms
    """
    
    def __init__(
        self,
        data_provider: AlpacaHistoricalDataProvider,
        cache_dir: str = "data/optimization_cache"
    ):
        """
        Initialize parameter optimizer
        
        Args:
            data_provider: Historical data provider
            cache_dir: Directory for caching optimization results
        """
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Optimization history
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        logger.info("Parameter optimizer initialized")
    
    def create_default_parameter_space(self) -> ParameterSpace:
        """Create default parameter space for MA crossover strategy"""
        space = ParameterSpace()
        
        # Moving average parameters
        space.add_parameter('fast_ma_period', 5, 20, 'int')
        space.add_parameter('slow_ma_period', 15, 50, 'int')
        
        # Risk management parameters
        space.add_parameter('risk_per_trade', 0.005, 0.05, 'float')
        space.add_parameter('confidence_threshold', 0.4, 0.8, 'float')
        
        # Volume confirmation parameters
        space.add_parameter('volume_threshold', 1.0, 3.0, 'float')
        space.add_parameter('mfi_oversold', 10, 30, 'int')
        space.add_parameter('mfi_overbought', 70, 90, 'int')
        
        # MACD parameters
        space.add_parameter('macd_fast', 8, 16, 'int')
        space.add_parameter('macd_slow', 20, 35, 'int')
        space.add_parameter('macd_signal', 5, 15, 'int')
        
        # Bollinger Bands parameters
        space.add_parameter('bb_period', 15, 25, 'int')
        space.add_parameter('bb_std', 1.5, 2.5, 'float')
        
        return space
    
    def optimize_parameters(
        self,
        strategy,
        symbol: str,
        parameter_space: ParameterSpace = None,
        optimization_method: str = 'bayesian',
        n_iterations: int = 100,
        objective: str = 'sharpe_ratio',
        data_period_days: int = 365,
        validation_split: float = 0.3
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using specified method
        
        Args:
            strategy: Trading strategy to optimize
            symbol: Trading symbol
            parameter_space: Parameter search space
            optimization_method: 'bayesian', 'genetic', 'grid', 'random'
            n_iterations: Number of optimization iterations
            objective: Optimization objective ('sharpe_ratio', 'total_return', 'calmar_ratio', etc.)
            data_period_days: Historical data period for optimization
            validation_split: Fraction of data for out-of-sample validation
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting parameter optimization for {symbol} using {optimization_method}")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=data_period_days)
        
        data = self.data_provider.fetch_historical_data(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe='1Min'
        )
        
        if data.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Split data for validation
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        validation_data = data.iloc[split_idx:]
        
        # Use default parameter space if not provided
        if parameter_space is None:
            parameter_space = self.create_default_parameter_space()
        
        # Create objective function
        objective_func = self._create_objective_function(
            strategy, train_data, objective
        )
        
        # Run optimization
        if optimization_method == 'bayesian':
            results = self._bayesian_optimization(
                objective_func, parameter_space, n_iterations
            )
        elif optimization_method == 'genetic':
            results = self._genetic_optimization(
                objective_func, parameter_space, n_iterations
            )
        elif optimization_method == 'grid':
            results = self._grid_search_optimization(
                objective_func, parameter_space, n_iterations
            )
        elif optimization_method == 'random':
            results = self._random_search_optimization(
                objective_func, parameter_space, n_iterations
            )
        else:
            return {'error': f'Unknown optimization method: {optimization_method}'}
        
        # Validate best parameters on out-of-sample data
        if results.get('best_parameters'):
            validation_results = self._validate_parameters(
                strategy, results['best_parameters'], validation_data, objective
            )
            results['validation'] = validation_results
        
        # Save optimization results
        self._save_optimization_results(results, symbol, optimization_method)
        
        logger.info(f"Optimization completed. Best {objective}: {results.get('best_score', 'N/A')}")
        
        return results
    
    def walk_forward_analysis(
        self,
        strategy,
        symbol: str,
        parameter_space: ParameterSpace = None,
        train_days: int = 180,
        test_days: int = 30,
        step_days: int = 7,
        total_days: int = 365
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis for parameter stability
        
        Args:
            strategy: Trading strategy
            symbol: Trading symbol
            parameter_space: Parameter search space
            train_days: Days for training optimization
            test_days: Days for out-of-sample testing
            step_days: Days to step forward between tests
            total_days: Total analysis period
            
        Returns:
            Walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis for {symbol}")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days + train_days)
        
        data = self.data_provider.fetch_historical_data(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe='1Min'
        )
        
        if data.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Use default parameter space if not provided
        if parameter_space is None:
            parameter_space = self.create_default_parameter_space()
        
        walk_forward_results = []
        current_start = 0
        
        while current_start + train_days + test_days <= len(data):
            # Define training and testing periods
            train_end = current_start + train_days
            test_end = train_end + test_days
            
            train_data = data.iloc[current_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Optimize parameters on training data
            objective_func = self._create_objective_function(
                strategy, train_data, 'sharpe_ratio'
            )
            
            optimization_results = self._bayesian_optimization(
                objective_func, parameter_space, n_iterations=50
            )
            
            # Test on out-of-sample data
            if optimization_results.get('best_parameters'):
                test_results = self._validate_parameters(
                    strategy, optimization_results['best_parameters'], 
                    test_data, 'sharpe_ratio'
                )
                
                walk_forward_results.append({
                    'train_start': data.index[current_start],
                    'train_end': data.index[train_end - 1],
                    'test_start': data.index[train_end],
                    'test_end': data.index[test_end - 1],
                    'optimal_parameters': optimization_results['best_parameters'],
                    'in_sample_score': optimization_results['best_score'],
                    'out_of_sample_score': test_results['score'],
                    'out_of_sample_metrics': test_results
                })
            
            current_start += step_days
        
        # Analyze walk-forward results
        analysis = self._analyze_walk_forward_results(walk_forward_results)
        
        return {
            'walk_forward_results': walk_forward_results,
            'analysis': analysis,
            'parameter_stability': self._calculate_parameter_stability(walk_forward_results),
            'performance_consistency': self._calculate_performance_consistency(walk_forward_results)
        }
    
    def _create_objective_function(
        self, 
        strategy, 
        data: pd.DataFrame, 
        objective: str
    ) -> Callable:
        """Create objective function for optimization"""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Run backtest with given parameters
                backtest_results = self.backtest_engine.run_backtest(
                    strategy, data, parameters
                )
                
                if not backtest_results.get('backtest_valid', False):
                    return -1000  # Penalty for invalid backtests
                
                # Extract objective value
                if objective == 'sharpe_ratio':
                    score = backtest_results.get('sharpe_ratio', -10)
                elif objective == 'total_return':
                    score = backtest_results.get('total_return', -1)
                elif objective == 'calmar_ratio':
                    score = backtest_results.get('calmar_ratio', -10)
                elif objective == 'win_rate':
                    score = backtest_results.get('win_rate', 0)
                elif objective == 'profit_factor':
                    score = backtest_results.get('profit_factor', 0)
                else:
                    score = backtest_results.get('sharpe_ratio', -10)
                
                # Penalty for low number of trades
                if backtest_results.get('total_trades', 0) < 5:
                    score *= 0.5
                
                return score if not np.isnan(score) else -1000
                
            except Exception as e:
                logger.warning(f"Error in objective function: {e}")
                return -1000
        
        return objective_function
    
    def _bayesian_optimization(
        self, 
        objective_func: Callable, 
        parameter_space: ParameterSpace, 
        n_iterations: int
    ) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian Process"""
        
        # Initialize with random samples
        n_initial = min(10, n_iterations // 4)
        X_sample = []
        y_sample = []
        
        for _ in range(n_initial):
            params = parameter_space.get_random_parameters()
            score = objective_func(params)
            
            X_sample.append(parameter_space.normalize_parameters(params))
            y_sample.append(score)
            
            self.optimization_history.append({
                'parameters': params,
                'score': score,
                'iteration': len(self.optimization_history)
            })
        
        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)
        
        # Gaussian Process model
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        # Bayesian optimization loop
        for iteration in range(n_initial, n_iterations):
            # Fit GP model
            gp.fit(X_sample, y_sample)
            
            # Acquisition function (Expected Improvement)
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                
                # Expected Improvement
                best_y = np.max(y_sample)
                z = (mu - best_y) / (sigma + 1e-9)
                ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
                
                return -ei  # Minimize negative EI
            
            # Optimize acquisition function
            bounds = [(0, 1)] * len(parameter_space.parameters)
            result = minimize(
                acquisition, 
                x0=np.random.random(len(bounds)),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Convert back to parameter space
            next_params = parameter_space.denormalize_parameters(result.x)
            next_score = objective_func(next_params)
            
            # Update samples
            X_sample = np.vstack([X_sample, result.x])
            y_sample = np.append(y_sample, next_score)
            
            self.optimization_history.append({
                'parameters': next_params,
                'score': next_score,
                'iteration': len(self.optimization_history)
            })
            
            logger.debug(f"Iteration {iteration}: Score = {next_score:.4f}")
        
        # Find best parameters
        best_idx = np.argmax(y_sample)
        best_params = parameter_space.denormalize_parameters(X_sample[best_idx])
        best_score = y_sample[best_idx]
        
        return {
            'method': 'bayesian',
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history.copy(),
            'convergence_data': y_sample.tolist()
        }
    
    def _genetic_optimization(
        self, 
        objective_func: Callable, 
        parameter_space: ParameterSpace, 
        n_iterations: int
    ) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        
        population_size = min(50, n_iterations // 2)
        n_generations = n_iterations // population_size
        
        # Initialize population
        population = []
        fitness_scores = []
        
        for _ in range(population_size):
            params = parameter_space.get_random_parameters()
            score = objective_func(params)
            
            population.append(params)
            fitness_scores.append(score)
            
            self.optimization_history.append({
                'parameters': params,
                'score': score,
                'iteration': len(self.optimization_history)
            })
        
        # Evolution loop
        for generation in range(n_generations):
            # Selection (tournament selection)
            new_population = []
            new_fitness = []
            
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(
                    population_size, tournament_size, replace=False
                )
                winner_idx = tournament_indices[
                    np.argmax([fitness_scores[i] for i in tournament_indices])
                ]
                
                # Crossover and mutation
                parent1 = population[winner_idx].copy()
                
                # Random second parent for crossover
                parent2_idx = np.random.choice(population_size)
                parent2 = population[parent2_idx].copy()
                
                # Crossover
                child = {}
                for param_name in parent1.keys():
                    if np.random.random() < 0.5:
                        child[param_name] = parent1[param_name]
                    else:
                        child[param_name] = parent2[param_name]
                
                # Mutation
                for param_name in child.keys():
                    if np.random.random() < 0.1:  # 10% mutation rate
                        min_val, max_val = parameter_space.parameters[param_name]
                        if parameter_space.types[param_name] == 'int':
                            child[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            child[param_name] = np.random.uniform(min_val, max_val)
                
                # Evaluate child
                child_score = objective_func(child)
                
                new_population.append(child)
                new_fitness.append(child_score)
                
                self.optimization_history.append({
                    'parameters': child,
                    'score': child_score,
                    'iteration': len(self.optimization_history)
                })
            
            population = new_population
            fitness_scores = new_fitness
            
            best_score = max(fitness_scores)
            logger.debug(f"Generation {generation}: Best score = {best_score:.4f}")
        
        # Find best parameters
        best_idx = np.argmax(fitness_scores)
        best_params = population[best_idx]
        best_score = fitness_scores[best_idx]
        
        return {
            'method': 'genetic',
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history.copy(),
            'final_population': population
        }
    
    def _random_search_optimization(
        self, 
        objective_func: Callable, 
        parameter_space: ParameterSpace, 
        n_iterations: int
    ) -> Dict[str, Any]:
        """Random search optimization"""
        
        best_params = None
        best_score = -np.inf
        
        for iteration in range(n_iterations):
            params = parameter_space.get_random_parameters()
            score = objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            self.optimization_history.append({
                'parameters': params,
                'score': score,
                'iteration': len(self.optimization_history)
            })
            
            if iteration % 20 == 0:
                logger.debug(f"Iteration {iteration}: Best score = {best_score:.4f}")
        
        return {
            'method': 'random_search',
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history.copy()
        }
    
    def _grid_search_optimization(
        self, 
        objective_func: Callable, 
        parameter_space: ParameterSpace, 
        n_iterations: int
    ) -> Dict[str, Any]:
        """Grid search optimization"""
        
        # Calculate grid points per parameter
        n_params = len(parameter_space.parameters)
        points_per_param = max(2, int(n_iterations ** (1/n_params)))
        
        # Generate grid
        param_names = sorted(parameter_space.parameters.keys())
        param_grids = []
        
        for param_name in param_names:
            min_val, max_val = parameter_space.parameters[param_name]
            if parameter_space.types[param_name] == 'int':
                grid = np.linspace(min_val, max_val, points_per_param, dtype=int)
            else:
                grid = np.linspace(min_val, max_val, points_per_param)
            param_grids.append(grid)
        
        # Grid search
        best_params = None
        best_score = -np.inf
        
        from itertools import product
        for param_values in product(*param_grids):
            params = dict(zip(param_names, param_values))
            score = objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            self.optimization_history.append({
                'parameters': params,
                'score': score,
                'iteration': len(self.optimization_history)
            })
        
        return {
            'method': 'grid_search',
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history.copy(),
            'grid_size': points_per_param ** n_params
        }
    
    def _validate_parameters(
        self, 
        strategy, 
        parameters: Dict[str, Any], 
        validation_data: pd.DataFrame,
        objective: str
    ) -> Dict[str, Any]:
        """Validate parameters on out-of-sample data"""
        
        backtest_results = self.backtest_engine.run_backtest(
            strategy, validation_data, parameters
        )
        
        if not backtest_results.get('backtest_valid', False):
            return {'score': -1000, 'error': 'Invalid backtest'}
        
        # Calculate comprehensive analysis
        analysis = self.performance_analyzer.analyze_backtest_results(backtest_results)
        
        # Extract objective score
        if objective == 'sharpe_ratio':
            score = backtest_results.get('sharpe_ratio', -10)
        elif objective == 'total_return':
            score = backtest_results.get('total_return', -1)
        elif objective == 'calmar_ratio':
            score = backtest_results.get('calmar_ratio', -10)
        else:
            score = backtest_results.get('sharpe_ratio', -10)
        
        return {
            'score': score,
            'backtest_results': backtest_results,
            'performance_analysis': analysis
        }
    
    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze walk-forward results for stability"""
        if not results:
            return {}
        
        in_sample_scores = [r['in_sample_score'] for r in results]
        out_of_sample_scores = [r['out_of_sample_score'] for r in results]
        
        return {
            'mean_in_sample_score': np.mean(in_sample_scores),
            'mean_out_of_sample_score': np.mean(out_of_sample_scores),
            'score_correlation': np.corrcoef(in_sample_scores, out_of_sample_scores)[0, 1],
            'overfitting_ratio': np.mean(out_of_sample_scores) / np.mean(in_sample_scores),
            'score_stability': 1 - (np.std(out_of_sample_scores) / (abs(np.mean(out_of_sample_scores)) + 1e-10))
        }
    
    def _calculate_parameter_stability(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate parameter stability across walk-forward periods"""
        if not results:
            return {}
        
        # Extract all parameter sets
        param_sets = [r['optimal_parameters'] for r in results]
        
        stability_scores = {}
        for param_name in param_sets[0].keys():
            values = [params[param_name] for params in param_sets]
            stability_scores[param_name] = 1 - (np.std(values) / (abs(np.mean(values)) + 1e-10))
        
        return stability_scores
    
    def _calculate_performance_consistency(self, results: List[Dict]) -> float:
        """Calculate performance consistency across periods"""
        scores = [r['out_of_sample_score'] for r in results]
        positive_periods = sum(1 for score in scores if score > 0)
        return positive_periods / len(scores) if scores else 0
    
    def _save_optimization_results(self, results: Dict, symbol: str, method: str):
        """Save optimization results to cache"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_{symbol}_{method}_{timestamp}.json"
        filepath = self.cache_dir / filename
        
        # Convert non-serializable objects
        serializable_results = results.copy()
        if 'optimization_history' in serializable_results:
            # Convert datetime objects to strings
            for item in serializable_results['optimization_history']:
                if 'timestamp' in item:
                    item['timestamp'] = item['timestamp'].isoformat()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            logger.warning(f"Error saving optimization results: {e}")
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history"""
        return self.optimization_history.copy()
    
    def clear_optimization_history(self):
        """Clear optimization history"""
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf