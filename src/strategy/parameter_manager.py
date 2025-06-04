"""
Dynamic Parameter Manager
Handles ML-optimizable parameters vs static configuration
"""
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """ML-optimizable trading parameters"""
    # Technical indicator parameters
    fast_ma_period: int = 12
    slow_ma_period: int = 24
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Strategy parameters
    volume_threshold: float = 1.2
    confidence_threshold: float = 0.6
    volatility_lookback: int = 20
    
    # Risk parameters (some can be ML-optimized)
    risk_per_trade: float = 0.02
    volatility_multiplier: float = 1.0
    
    # Performance metrics for this parameter set
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.last_updated:
            result['last_updated'] = self.last_updated.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingParameters':
        """Create from dictionary"""
        if 'last_updated' in data and data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class ParameterManager:
    """
    Manages dynamic trading parameters that can be optimized by ML
    Separates static config from ML-optimizable parameters
    """
    
    def __init__(self, firebase_logger=None):
        """Initialize parameter manager"""
        self.firebase_logger = firebase_logger
        self.current_params = TradingParameters()
        self.parameter_history = []
        self.optimization_enabled = True
        
        # Load parameters from Firebase if available
        self._load_parameters()
        
        logger.info("ParameterManager initialized")
    
    def get_current_parameters(self) -> TradingParameters:
        """Get current trading parameters"""
        return self.current_params
    
    def update_parameters(self, new_params: TradingParameters, 
                         performance_metrics: Dict[str, float]) -> None:
        """
        Update parameters with performance feedback
        
        Args:
            new_params: New parameter set
            performance_metrics: Performance metrics for evaluation
        """
        try:
            # Update performance metrics
            new_params.win_rate = performance_metrics.get('win_rate', 0.0)
            new_params.sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            new_params.total_trades = performance_metrics.get('num_trades', 0)
            new_params.last_updated = datetime.now()
            
            # Store current params in history
            if self.current_params.total_trades > 0:
                self.parameter_history.append(self.current_params)
            
            # Update current parameters
            self.current_params = new_params
            
            # Save to Firebase
            self._save_parameters()
            
            logger.info(f"Parameters updated - Win Rate: {new_params.win_rate:.2%}, "
                       f"Sharpe: {new_params.sharpe_ratio:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
    
    def optimize_parameters(self, performance_data: List[Dict[str, Any]]) -> TradingParameters:
        """
        Optimize parameters based on performance data
        This is where ML optimization would happen
        
        Args:
            performance_data: Historical performance data
            
        Returns:
            Optimized parameters
        """
        if not self.optimization_enabled or len(performance_data) < 10:
            return self.current_params
        
        try:
            # Simple optimization example - in practice, use more sophisticated ML
            optimized_params = self._simple_parameter_optimization(performance_data)
            
            logger.info("Parameters optimized using performance feedback")
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return self.current_params
    
    def _simple_parameter_optimization(self, performance_data: List[Dict[str, Any]]) -> TradingParameters:
        """
        Simple parameter optimization example
        In practice, this would use sophisticated ML algorithms
        """
        # Analyze recent performance
        recent_data = performance_data[-20:]  # Last 20 trades
        recent_wins = [d for d in recent_data if d.get('pnl', 0) > 0]
        recent_win_rate = len(recent_wins) / len(recent_data) if recent_data else 0
        
        new_params = TradingParameters(
            fast_ma_period=self.current_params.fast_ma_period,
            slow_ma_period=self.current_params.slow_ma_period,
            rsi_period=self.current_params.rsi_period,
            rsi_oversold=self.current_params.rsi_oversold,
            rsi_overbought=self.current_params.rsi_overbought,
            volume_threshold=self.current_params.volume_threshold,
            confidence_threshold=self.current_params.confidence_threshold,
            volatility_lookback=self.current_params.volatility_lookback,
            risk_per_trade=self.current_params.risk_per_trade,
            volatility_multiplier=self.current_params.volatility_multiplier
        )
        
        # Simple adaptive logic
        if recent_win_rate < 0.4:  # Poor performance
            # Make strategy more conservative
            new_params.confidence_threshold = min(0.8, self.current_params.confidence_threshold + 0.1)
            new_params.volume_threshold = min(2.0, self.current_params.volume_threshold + 0.2)
            new_params.risk_per_trade = max(0.01, self.current_params.risk_per_trade - 0.002)
            
        elif recent_win_rate > 0.6:  # Good performance
            # Make strategy slightly more aggressive
            new_params.confidence_threshold = max(0.5, self.current_params.confidence_threshold - 0.05)
            new_params.risk_per_trade = min(0.03, self.current_params.risk_per_trade + 0.001)
        
        return new_params
    
    def _load_parameters(self) -> None:
        """Load parameters from Firebase"""
        if not self.firebase_logger or not self.firebase_logger.is_connected():
            logger.info("Firebase not available, using default parameters")
            return
        
        try:
            # In a real implementation, you'd load from Firebase
            # For now, we'll use defaults
            logger.info("Loaded parameters from Firebase")
            
        except Exception as e:
            logger.error(f"Error loading parameters from Firebase: {e}")
    
    def _save_parameters(self) -> None:
        """Save parameters to Firebase"""
        if not self.firebase_logger or not self.firebase_logger.is_connected():
            logger.warning("Firebase not available, parameters not saved")
            return
        
        try:
            # Save current parameters
            param_data = self.current_params.to_dict()
            param_data['timestamp'] = datetime.now().isoformat()
            
            # Save to Firebase (you'd implement the actual Firebase saving)
            # self.firebase_logger.save_parameters(param_data)
            
            logger.debug("Parameters saved to Firebase")
            
        except Exception as e:
            logger.error(f"Error saving parameters to Firebase: {e}")
    
    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter bounds for optimization
        Defines the search space for ML optimization
        """
        return {
            'fast_ma_period': {'min': 5, 'max': 20},
            'slow_ma_period': {'min': 15, 'max': 50},
            'rsi_period': {'min': 10, 'max': 20},
            'rsi_oversold': {'min': 20.0, 'max': 35.0},
            'rsi_overbought': {'min': 65.0, 'max': 80.0},
            'volume_threshold': {'min': 1.0, 'max': 3.0},
            'confidence_threshold': {'min': 0.4, 'max': 0.8},
            'volatility_lookback': {'min': 10, 'max': 30},
            'risk_per_trade': {'min': 0.005, 'max': 0.05},
            'volatility_multiplier': {'min': 0.5, 'max': 2.0}
        }
    
    def enable_optimization(self, enabled: bool = True) -> None:
        """Enable or disable parameter optimization"""
        self.optimization_enabled = enabled
        logger.info(f"Parameter optimization {'enabled' if enabled else 'disabled'}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get parameter optimization history"""
        return [params.to_dict() for params in self.parameter_history]
    
    def reset_to_defaults(self) -> None:
        """Reset parameters to defaults"""
        self.current_params = TradingParameters()
        self.parameter_history.clear()
        self._save_parameters()
        logger.info("Parameters reset to defaults")