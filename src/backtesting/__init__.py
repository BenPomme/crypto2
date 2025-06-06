"""
Backtesting and Parameter Optimization Module

This module provides comprehensive backtesting capabilities for strategy optimization,
including historical data management, performance analysis, and ML-driven parameter learning.
"""

from .historical_data import AlpacaHistoricalDataProvider
from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceAnalyzer
from .parameter_optimizer import ParameterOptimizer
from .ml_parameter_learner import MLParameterLearner
from .backtest_orchestrator import BacktestOrchestrator

__all__ = [
    'AlpacaHistoricalDataProvider',
    'BacktestEngine', 
    'PerformanceAnalyzer',
    'ParameterOptimizer',
    'MLParameterLearner',
    'BacktestOrchestrator'
]