"""
Feature Engineering Module
Processes market data and creates features for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for trading strategies
    Combines technical indicators with price action features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration for indicators and features
        """
        self.config = config or {}
        self.indicators = TechnicalIndicators()
        
        # Default feature configuration
        self.default_config = {
            # Technical indicator periods
            'ma_fast': 12,
            'ma_slow': 24,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'mfi_period': 14,
            
            # Feature engineering parameters
            'price_change_periods': [1, 3, 5, 10],
            'volatility_lookback': 20,
            'volume_ma_period': 20,
            'momentum_periods': [5, 10, 20],
        }
        
        self.default_config.update(self.config)
        logger.info(f"FeatureEngineer initialized with MA periods: fast={self.default_config['ma_fast']}, slow={self.default_config['ma_slow']}")
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features added
        """
        result_df = df.copy()
        
        try:
            # Price changes over different periods
            for period in self.default_config['price_change_periods']:
                result_df[f'price_change_{period}'] = result_df['close'].pct_change(period)
                result_df[f'price_change_{period}_abs'] = result_df[f'price_change_{period}'].abs()
            
            # High-Low spread
            result_df['hl_spread'] = (result_df['high'] - result_df['low']) / result_df['close']
            
            # Open-Close spread
            result_df['oc_spread'] = (result_df['close'] - result_df['open']) / result_df['open']
            
            # Daily range position (where close is within daily range)
            result_df['range_position'] = (
                (result_df['close'] - result_df['low']) / 
                (result_df['high'] - result_df['low'])
            ).fillna(0.5)
            
            # Gap detection (difference between current open and previous close)
            result_df['gap'] = (
                (result_df['open'] - result_df['close'].shift(1)) / 
                result_df['close'].shift(1)
            )
            
            # Price momentum over different periods
            for period in self.default_config['momentum_periods']:
                result_df[f'momentum_{period}'] = (
                    result_df['close'] / result_df['close'].shift(period) - 1
                )
            
            logger.debug("Price features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
        
        return result_df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features added
        """
        result_df = df.copy()
        
        try:
            volume_ma_period = self.default_config['volume_ma_period']
            
            # Volume moving average
            result_df['volume_ma'] = result_df['volume'].rolling(volume_ma_period).mean()
            
            # Volume ratio (current volume vs average)
            result_df['volume_ratio'] = result_df['volume'] / result_df['volume_ma']
            
            # Volume changes
            result_df['volume_change'] = result_df['volume'].pct_change()
            result_df['volume_change_ma'] = result_df['volume_change'].rolling(5).mean()
            
            # Price-Volume relationship
            result_df['price_volume_trend'] = (
                result_df['close'].pct_change() * result_df['volume_ratio']
            )
            
            # Volume-weighted features
            result_df['vwap_ratio'] = result_df['close'] / result_df.get('vwap', result_df['close'])
            
            # Volume spikes (unusually high volume)
            volume_std = result_df['volume'].rolling(volume_ma_period).std()
            result_df['volume_spike'] = (
                (result_df['volume'] - result_df['volume_ma']) / volume_std
            ).fillna(0)
            
            logger.debug("Volume features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
        
        return result_df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features added
        """
        result_df = df.copy()
        
        try:
            lookback = self.default_config['volatility_lookback']
            
            # Rolling volatility (standard deviation of returns)
            returns = result_df['close'].pct_change()
            result_df['volatility'] = returns.rolling(lookback).std()
            
            # Volatility percentile (current volatility vs historical)
            result_df['volatility_percentile'] = (
                result_df['volatility'].rolling(lookback * 2).rank(pct=True)
            )
            
            # True Range and ATR ratio
            if 'atr' in result_df.columns:
                current_tr = np.maximum(
                    result_df['high'] - result_df['low'],
                    np.maximum(
                        abs(result_df['high'] - result_df['close'].shift(1)),
                        abs(result_df['low'] - result_df['close'].shift(1))
                    )
                )
                result_df['tr_atr_ratio'] = current_tr / result_df['atr']
            
            # Volatility regime detection
            vol_ma_short = result_df['volatility'].rolling(5).mean()
            vol_ma_long = result_df['volatility'].rolling(20).mean()
            result_df['vol_regime'] = np.where(vol_ma_short > vol_ma_long, 1, 0)
            
            logger.debug("Volatility features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating volatility features: {e}")
        
        return result_df
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend features added
        """
        result_df = df.copy()
        
        try:
            # Moving average crossover signals
            if 'sma_fast' in result_df.columns and 'sma_slow' in result_df.columns:
                result_df['ma_crossover'] = np.where(
                    result_df['sma_fast'] > result_df['sma_slow'], 1, 0
                )
                result_df['ma_crossover_change'] = result_df['ma_crossover'].diff()
            
            # Price position relative to moving averages
            if 'sma_slow' in result_df.columns:
                result_df['price_ma_ratio'] = result_df['close'] / result_df['sma_slow']
            
            # Trend strength based on consecutive price movements
            result_df['trend_direction'] = np.sign(result_df['close'].diff())
            result_df['trend_consecutive'] = (
                result_df['trend_direction'].groupby(
                    (result_df['trend_direction'] != result_df['trend_direction'].shift()).cumsum()
                ).cumcount() + 1
            )
            
            # Higher highs and lower lows
            lookback = 10
            result_df['higher_high'] = (
                result_df['high'] > result_df['high'].rolling(lookback).max().shift(1)
            ).astype(int)
            
            result_df['lower_low'] = (
                result_df['low'] < result_df['low'].rolling(lookback).min().shift(1)
            ).astype(int)
            
            # Trend score (combination of various trend indicators)
            trend_components = []
            
            if 'ma_crossover' in result_df.columns:
                trend_components.append(result_df['ma_crossover'])
            
            if 'rsi' in result_df.columns:
                trend_components.append((result_df['rsi'] > 50).astype(int))
            
            if trend_components:
                result_df['trend_score'] = np.mean(trend_components, axis=0)
            
            logger.debug("Trend features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating trend features: {e}")
        
        return result_df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features added
        """
        result_df = df.copy()
        
        try:
            # RSI-based features
            if 'rsi' in result_df.columns:
                result_df['rsi_oversold'] = (result_df['rsi'] < 30).astype(int)
                result_df['rsi_overbought'] = (result_df['rsi'] > 70).astype(int)
                result_df['rsi_momentum'] = result_df['rsi'].diff()
                
                # RSI divergence (simplified)
                price_peaks = result_df['close'].rolling(5).max() == result_df['close']
                rsi_peaks = result_df['rsi'].rolling(5).max() == result_df['rsi']
                result_df['rsi_divergence'] = (price_peaks & ~rsi_peaks).astype(int)
            
            # MACD-based features
            if 'MACD_12_26_9' in result_df.columns:
                macd_col = 'MACD_12_26_9'
                signal_col = 'MACDs_12_26_9'
                
                if signal_col in result_df.columns:
                    result_df['macd_signal'] = np.where(
                        result_df[macd_col] > result_df[signal_col], 1, 0
                    )
                    result_df['macd_crossover'] = result_df['macd_signal'].diff()
            
            # Stochastic features
            if 'STOCHk_14_3_3' in result_df.columns:
                stoch_k = 'STOCHk_14_3_3'
                result_df['stoch_oversold'] = (result_df[stoch_k] < 20).astype(int)
                result_df['stoch_overbought'] = (result_df[stoch_k] > 80).astype(int)
            
            # Williams %R features
            if 'williams_r' in result_df.columns:
                result_df['willr_oversold'] = (result_df['williams_r'] < -80).astype(int)
                result_df['willr_overbought'] = (result_df['williams_r'] > -20).astype(int)
            
            logger.debug("Momentum features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating momentum features: {e}")
        
        return result_df
    
    def engineer_features(self, df: pd.DataFrame, custom_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to engineer_features")
            return df.copy()
        
        logger.info(f"Starting feature engineering for {len(df)} bars")
        
        # Merge custom config with defaults
        indicator_config = self.default_config.copy()
        if custom_config:
            indicator_config.update(custom_config)
        
        # Start with technical indicators using merged config
        result_df = self.indicators.calculate_all_indicators(df, indicator_config)
        
        # Add various feature categories
        result_df = self.create_price_features(result_df)
        result_df = self.create_volume_features(result_df)
        result_df = self.create_volatility_features(result_df)
        result_df = self.create_trend_features(result_df)
        result_df = self.create_momentum_features(result_df)
        
        # Clean up any infinite or extremely large values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate defaults
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_columns] = result_df[numeric_columns].ffill().fillna(0)
        
        logger.info(f"Feature engineering completed. Added {len(result_df.columns) - len(df.columns)} features")
        
        return result_df
    
    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic feature importance based on correlation with future returns
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary of feature importance scores
        """
        if df.empty or 'close' not in df.columns:
            return {}
        
        try:
            # Calculate forward returns
            forward_returns = df['close'].pct_change().shift(-1)
            
            # Calculate correlations
            feature_columns = [col for col in df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            correlations = {}
            for col in feature_columns:
                if df[col].dtype in [np.float64, np.int64]:
                    corr = abs(df[col].corr(forward_returns))
                    if not np.isnan(corr):
                        correlations[col] = corr
            
            # Sort by importance
            sorted_features = dict(sorted(correlations.items(), 
                                        key=lambda x: x[1], reverse=True))
            
            logger.info(f"Calculated importance for {len(sorted_features)} features")
            
            return sorted_features
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}