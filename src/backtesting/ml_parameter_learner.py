"""
ML Parameter Learner
Intelligent parameter adaptation using online learning from backtesting results
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import pickle
from pathlib import Path

# Online learning with River
try:
    from river import linear_model, preprocessing, compose, metrics, ensemble
    from river.tree import HoeffdingTreeRegressor
except ImportError:
    logging.warning("River library not available. Install with: pip install river")
    linear_model = preprocessing = compose = metrics = ensemble = None

from .parameter_optimizer import ParameterOptimizer, ParameterSpace
from .performance_metrics import PerformanceAnalyzer
from .historical_data import AlpacaHistoricalDataProvider

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detect market regimes for parameter adaptation"""
    
    def __init__(self):
        self.features = [
            'volatility_regime',
            'trend_strength', 
            'volume_regime',
            'momentum_regime'
        ]
        
    def detect_regime(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect current market regime based on recent data
        
        Args:
            market_data: Recent OHLCV data
            
        Returns:
            Dictionary with regime features (0-1 normalized)
        """
        if len(market_data) < 50:
            return {feature: 0.5 for feature in self.features}
        
        # Calculate recent metrics
        returns = market_data['close'].pct_change().dropna()
        volume = market_data['volume']
        
        # Volatility regime (0=low vol, 1=high vol)
        recent_vol = returns.tail(20).std()
        historical_vol = returns.std()
        volatility_regime = min(1.0, recent_vol / (historical_vol + 1e-10))
        
        # Trend strength (0=sideways, 1=strong trend)
        sma_short = market_data['close'].tail(10).mean()
        sma_long = market_data['close'].tail(50).mean()
        trend_strength = abs(sma_short - sma_long) / sma_long
        trend_strength = min(1.0, trend_strength * 10)
        
        # Volume regime (0=low volume, 1=high volume) 
        recent_volume = volume.tail(20).mean()
        historical_volume = volume.mean()
        volume_regime = min(1.0, recent_volume / (historical_volume + 1e-10))
        
        # Momentum regime (0=low momentum, 1=high momentum)
        price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20]
        momentum_regime = min(1.0, abs(price_change) * 5)
        
        return {
            'volatility_regime': volatility_regime,
            'trend_strength': trend_strength,
            'volume_regime': volume_regime, 
            'momentum_regime': momentum_regime
        }

class MLParameterLearner:
    """
    ML-driven parameter learning system
    Uses online learning to adapt strategy parameters based on market conditions
    """
    
    def __init__(self, model_cache_dir: str = "data/ml_models"):
        """
        Initialize ML parameter learner
        
        Args:
            model_cache_dir: Directory to save/load trained models
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Online learning models for each parameter
        self.parameter_models = {}
        self.feature_preprocessor = None
        self.learning_history = []
        
        # Performance tracking
        self.model_performance = {}
        
        logger.info("ML parameter learner initialized")
    
    def initialize_models(self, parameter_space: ParameterSpace):
        """
        Initialize online learning models for each parameter
        
        Args:
            parameter_space: Parameter search space definition
        """
        if linear_model is None:
            logger.error("River library not available. Cannot initialize ML models.")
            return
        
        # Feature preprocessing pipeline
        self.feature_preprocessor = compose.Pipeline(
            preprocessing.StandardScaler(),
            preprocessing.MinMaxScaler()
        )
        
        # Create model for each parameter
        for param_name in parameter_space.parameters.keys():
            # Ensemble model for better performance
            model = ensemble.VotingRegressor([
                ('linear', linear_model.LinearRegression()),
                ('tree', HoeffdingTreeRegressor()),
                ('passive_aggressive', linear_model.PARegressor())
            ])
            
            self.parameter_models[param_name] = model
            self.model_performance[param_name] = {
                'predictions': [],
                'actuals': [],
                'mae': metrics.MAE(),
                'rmse': metrics.RMSE()
            }
        
        logger.info(f"Initialized ML models for {len(self.parameter_models)} parameters")
    
    def learn_from_backtest_results(
        self,
        optimization_results: Dict[str, Any],
        market_data: pd.DataFrame,
        target_metric: str = 'sharpe_ratio'
    ):
        """
        Learn from backtesting optimization results
        
        Args:
            optimization_results: Results from parameter optimization
            market_data: Market data used for backtesting
            target_metric: Performance metric to optimize for
        """
        if not self.parameter_models:
            logger.warning("Models not initialized. Call initialize_models first.")
            return
        
        # Extract optimization history
        history = optimization_results.get('optimization_history', [])
        if not history:
            logger.warning("No optimization history found")
            return
        
        # Detect market regime for this period
        market_features = self.regime_detector.detect_regime(market_data)
        
        logger.info(f"Learning from {len(history)} optimization samples")
        
        # Train models with optimization results
        for result in history:
            parameters = result.get('parameters', {})
            score = result.get('score', 0)
            
            # Skip invalid results
            if score <= -1000 or not parameters:
                continue
            
            # Prepare features (market regime + parameter context)
            features = self._prepare_features(market_features, parameters)
            
            # Update each parameter model
            for param_name, param_value in parameters.items():
                if param_name in self.parameter_models:
                    model = self.parameter_models[param_name]
                    
                    # Learn: features -> optimal parameter value
                    model.learn_one(features, param_value)
                    
                    # Track performance
                    perf = self.model_performance[param_name]
                    perf['actuals'].append(param_value)
                    
                    # Make prediction to track model performance
                    pred = model.predict_one(features)
                    perf['predictions'].append(pred)
                    perf['mae'].update(param_value, pred)
                    perf['rmse'].update(param_value, pred)
        
        # Store learning instance
        self.learning_history.append({
            'timestamp': datetime.now(),
            'market_features': market_features,
            'samples_learned': len([h for h in history if h.get('score', -1000) > -1000]),
            'best_score': max([h.get('score', -1000) for h in history]),
            'target_metric': target_metric
        })
        
        logger.info(f"Updated ML models with backtesting results")
    
    def predict_optimal_parameters(
        self,
        current_market_data: pd.DataFrame,
        parameter_space: ParameterSpace,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Predict optimal parameters for current market conditions
        
        Args:
            current_market_data: Current market data
            parameter_space: Parameter search space
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary with predicted parameters and confidence scores
        """
        if not self.parameter_models:
            logger.warning("Models not trained. Returning default parameters.")
            return self._get_default_parameters(parameter_space)
        
        # Detect current market regime
        current_features = self.regime_detector.detect_regime(current_market_data)
        
        # Prepare features for prediction
        features = self._prepare_features(current_features, {})
        
        # Predict each parameter
        predictions = {}
        confidences = {}
        
        for param_name, model in self.parameter_models.items():
            try:
                # Get prediction
                prediction = model.predict_one(features)
                
                # Constrain to parameter bounds
                min_val, max_val = parameter_space.parameters[param_name]
                prediction = max(min_val, min(max_val, prediction))
                
                # Apply parameter type constraints
                if parameter_space.types[param_name] == 'int':
                    prediction = int(round(prediction))
                
                # Calculate confidence based on model performance
                perf = self.model_performance[param_name]
                if len(perf['predictions']) > 10:
                    mae = perf['mae'].get()
                    param_range = max_val - min_val
                    confidence = max(0, 1 - (mae / param_range))
                else:
                    confidence = 0.5  # Low confidence for new models
                
                predictions[param_name] = prediction
                confidences[param_name] = confidence
                
            except Exception as e:
                logger.warning(f"Error predicting {param_name}: {e}")
                # Fall back to middle of range
                min_val, max_val = parameter_space.parameters[param_name]
                predictions[param_name] = (min_val + max_val) / 2
                confidences[param_name] = 0.0
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(confidences.values()))
        
        # Use predictions only if confidence is high enough
        if overall_confidence >= confidence_threshold:
            logger.info(f"High confidence parameter predictions: {overall_confidence:.2f}")
            return {
                'parameters': predictions,
                'confidence': overall_confidence,
                'market_regime': current_features,
                'prediction_source': 'ml_model'
            }
        else:
            logger.info(f"Low confidence predictions ({overall_confidence:.2f}), using defaults")
            return self._get_default_parameters(parameter_space)
    
    def adaptive_parameter_update(
        self,
        current_parameters: Dict[str, Any],
        recent_performance: Dict[str, float],
        market_data: pd.DataFrame,
        parameter_space: ParameterSpace,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Adaptively update parameters based on recent performance
        
        Args:
            current_parameters: Current strategy parameters
            recent_performance: Recent performance metrics
            market_data: Recent market data
            parameter_space: Parameter search space
            learning_rate: How aggressively to adjust parameters
            
        Returns:
            Updated parameters
        """
        # Get ML predictions
        ml_predictions = self.predict_optimal_parameters(market_data, parameter_space)
        
        if ml_predictions['prediction_source'] != 'ml_model':
            # No high-confidence ML predictions, use performance-based adaptation
            return self._performance_based_adaptation(
                current_parameters, recent_performance, parameter_space, learning_rate
            )
        
        # Blend current parameters with ML predictions
        updated_parameters = {}
        for param_name in current_parameters.keys():
            current_val = current_parameters[param_name]
            predicted_val = ml_predictions['parameters'].get(param_name, current_val)
            
            # Weighted blend based on confidence and learning rate
            confidence = ml_predictions['confidence']
            blend_weight = learning_rate * confidence
            
            blended_val = current_val * (1 - blend_weight) + predicted_val * blend_weight
            
            # Apply constraints
            min_val, max_val = parameter_space.parameters[param_name]
            blended_val = max(min_val, min(max_val, blended_val))
            
            if parameter_space.types[param_name] == 'int':
                blended_val = int(round(blended_val))
            
            updated_parameters[param_name] = blended_val
        
        logger.info("Applied ML-based parameter adaptation")
        return {
            'parameters': updated_parameters,
            'adaptation_method': 'ml_blend',
            'confidence': ml_predictions['confidence'],
            'learning_rate': learning_rate
        }
    
    def _prepare_features(self, market_features: Dict[str, float], parameter_context: Dict[str, Any]) -> Dict[str, float]:
        """Prepare feature vector for ML models"""
        features = market_features.copy()
        
        # Add time-based features
        now = datetime.now()
        features['hour_of_day'] = now.hour / 24.0
        features['day_of_week'] = now.weekday() / 6.0
        features['day_of_month'] = now.day / 31.0
        
        # Add parameter interaction features if available
        if parameter_context:
            features['param_count'] = len(parameter_context)
            features['param_avg'] = np.mean(list(parameter_context.values())) if parameter_context else 0
        
        return features
    
    def _get_default_parameters(self, parameter_space: ParameterSpace) -> Dict[str, Any]:
        """Get default parameters when ML predictions are not available"""
        defaults = {}
        for param_name, (min_val, max_val) in parameter_space.parameters.items():
            # Use middle of range as default
            default_val = (min_val + max_val) / 2
            
            if parameter_space.types[param_name] == 'int':
                default_val = int(round(default_val))
            
            defaults[param_name] = default_val
        
        return {
            'parameters': defaults,
            'confidence': 0.5,
            'market_regime': {},
            'prediction_source': 'default'
        }
    
    def _performance_based_adaptation(
        self,
        current_parameters: Dict[str, Any],
        recent_performance: Dict[str, float],
        parameter_space: ParameterSpace,
        learning_rate: float
    ) -> Dict[str, Any]:
        """Simple performance-based parameter adaptation"""
        updated_parameters = current_parameters.copy()
        
        # Get performance score (higher is better)
        performance_score = recent_performance.get('sharpe_ratio', 0)
        
        # If performance is poor, make larger adjustments
        if performance_score < 0:
            adjustment_factor = learning_rate * 2
        elif performance_score < 0.5:
            adjustment_factor = learning_rate
        else:
            adjustment_factor = learning_rate * 0.5  # Smaller adjustments when doing well
        
        # Make random adjustments for exploration
        for param_name in current_parameters.keys():
            current_val = current_parameters[param_name]
            min_val, max_val = parameter_space.parameters[param_name]
            param_range = max_val - min_val
            
            # Random adjustment within bounds
            max_change = param_range * adjustment_factor
            change = np.random.uniform(-max_change, max_change)
            new_val = current_val + change
            
            # Apply constraints
            new_val = max(min_val, min(max_val, new_val))
            
            if parameter_space.types[param_name] == 'int':
                new_val = int(round(new_val))
            
            updated_parameters[param_name] = new_val
        
        return {
            'parameters': updated_parameters,
            'adaptation_method': 'performance_based',
            'confidence': 0.3,
            'adjustment_factor': adjustment_factor
        }
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        if not self.parameter_models:
            return {'error': 'No models trained'}
        
        report = {
            'models_trained': len(self.parameter_models),
            'learning_sessions': len(self.learning_history),
            'parameter_performance': {}
        }
        
        for param_name, perf in self.model_performance.items():
            if len(perf['predictions']) > 0:
                report['parameter_performance'][param_name] = {
                    'samples': len(perf['predictions']),
                    'mae': perf['mae'].get(),
                    'rmse': perf['rmse'].get(),
                    'last_prediction': perf['predictions'][-1] if perf['predictions'] else None
                }
        
        return report
    
    def save_models(self, filename: str = None):
        """Save trained models to disk"""
        if filename is None:
            filename = f"ml_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = self.model_cache_dir / filename
        
        try:
            model_data = {
                'parameter_models': self.parameter_models,
                'model_performance': self.model_performance,
                'learning_history': self.learning_history,
                'saved_at': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return None
    
    def load_models(self, filename: str):
        """Load trained models from disk"""
        filepath = self.model_cache_dir / filename
        
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.parameter_models = model_data.get('parameter_models', {})
            self.model_performance = model_data.get('model_performance', {})
            self.learning_history = model_data.get('learning_history', [])
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False