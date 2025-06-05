"""
Technical Indicators for crypto trading
Implements various indicators: MA, RSI, OBV, ATR, Bollinger Bands, etc.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical indicators calculator using pandas-ta
    Provides common indicators for trading strategies
    """
    
    @staticmethod
    def simple_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            result = ta.sma(data, length=period)
            return result if result is not None else pd.Series(dtype='float64', index=data.index)
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series(dtype='float64', index=data.index)
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            result = ta.ema(data, length=period)
            return result if result is not None else pd.Series(dtype='float64', index=data.index)
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series(dtype='float64', index=data.index)
    
    @staticmethod
    def relative_strength_index(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            result = ta.rsi(data, length=period)
            return result if result is not None else pd.Series(50.0, index=data.index, dtype='float64')
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50.0, index=data.index, dtype='float64')
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        Returns DataFrame with columns: BBL, BBM, BBU, BBB, BBP
        """
        return ta.bbands(data, length=period, std=std)
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, 
                          period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return ta.atr(high=high, low=low, close=close, length=period)
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        # Ensure inputs are float64 to prevent dtype warnings
        close = close.astype('float64')
        volume = volume.astype('float64')
        
        result = ta.obv(close=close, volume=volume)
        return result.astype('float64') if result is not None else pd.Series(dtype='float64')
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index - Custom implementation to avoid pandas-ta dtype issues"""
        # Ensure all inputs are float64
        high = high.astype('float64')
        low = low.astype('float64') 
        close = close.astype('float64')
        volume = volume.astype('float64')
        
        try:
            # Custom MFI calculation to avoid pandas-ta dtype warnings
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            # Calculate positive and negative money flow
            positive_flow = pd.Series(0.0, index=high.index)
            negative_flow = pd.Series(0.0, index=high.index)
            
            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = money_flow.iloc[i]
            
            # Calculate money flow ratio and MFI
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            # Avoid division by zero
            money_flow_ratio = positive_mf / (negative_mf + 1e-10)
            mfi = 100 - (100 / (1 + money_flow_ratio))
            
            return mfi.fillna(50.0)  # Fill NaN with neutral MFI value
            
        except Exception as e:
            logger.error(f"Error calculating custom MFI: {e}")
            return pd.Series(50.0, index=high.index, dtype='float64')
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        Returns DataFrame with %K and %D
        """
        return ta.stoch(high=high, low=low, close=close, k=k_period, d=d_period)
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD
        Returns DataFrame with MACD, MACD Histogram, MACD Signal
        """
        return ta.macd(data, fast=fast, slow=slow, signal=signal)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        return ta.willr(high=high, low=low, close=close, length=period)
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series,
                               period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        return ta.cci(high=high, low=low, close=close, length=period)
    
    @staticmethod
    def volume_weighted_average_price(high: pd.Series, low: pd.Series, 
                                     close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        return ta.vwap(high=high, low=low, close=close, volume=volume)
    
    @staticmethod
    def calculate_support_resistance(data: pd.Series, window: int = 20) -> Tuple[float, float]:
        """
        Calculate basic support and resistance levels
        
        Args:
            data: Price series
            window: Lookback window
            
        Returns:
            Tuple of (support, resistance)
        """
        if len(data) < window:
            return data.min(), data.max()
        
        recent_data = data.tail(window)
        support = recent_data.min()
        resistance = recent_data.max()
        
        return support, resistance
    
    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate pivot points for the next trading session
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def volatility_bands(data: pd.Series, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate volatility bands using ATR
        
        Args:
            data: Price series
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            DataFrame with upper and lower bands
        """
        # For volatility bands, we need high/low data
        # This is a simplified version using close prices
        sma = ta.sma(data, length=period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * multiplier)
        lower_band = sma - (std * multiplier)
        
        return pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band
        })
    
    @staticmethod
    def trend_strength(data: pd.Series, period: int = 14) -> float:
        """
        Calculate trend strength (0-1 scale)
        Based on price direction consistency
        """
        if len(data) < period + 1:
            return 0.5
        
        changes = data.diff().dropna()
        if len(changes) < period:
            return 0.5
        
        recent_changes = changes.tail(period)
        positive_changes = (recent_changes > 0).sum()
        
        return positive_changes / period
    
    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame, 
                               config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculate all indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration for indicator parameters
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to calculate_all_indicators")
            return df.copy()
        
        # Default configuration
        default_config = {
            'ma_fast': 12,
            'ma_slow': 24,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'mfi_period': 14,
            'stoch_k': 14,
            'stoch_d': 3
        }
        
        if config:
            default_config.update(config)
        
        result_df = df.copy()
        
        try:
            # Moving Averages - SKIP pandas-ta, use pure pandas
            logger.info(f"Calculating indicators with {len(df)} bars using pure pandas")
            
            # Use pure pandas calculations to avoid pandas-ta issues
            logger.info(f"Using MA periods: fast={default_config['ma_fast']}, slow={default_config['ma_slow']}")
            result_df['sma_fast'] = df['close'].rolling(window=default_config['ma_fast']).mean()
            result_df['sma_slow'] = df['close'].rolling(window=default_config['ma_slow']).mean()
            result_df['ema_fast'] = df['close'].ewm(span=default_config['ma_fast']).mean()
            result_df['ema_slow'] = df['close'].ewm(span=default_config['ma_slow']).mean()
            
            # Simple RSI calculation using pandas
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=default_config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=default_config['rsi_period']).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            result_df['rsi'] = 100 - (100 / (1 + rs))
            result_df['rsi'] = result_df['rsi'].fillna(50.0)  # Fill NaN with neutral RSI
            
            logger.info(f"Pure pandas calculations complete: SMA_fast={result_df['sma_fast'].iloc[-1]:.2f}, SMA_slow={result_df['sma_slow'].iloc[-1]:.2f}, RSI={result_df['rsi'].iloc[-1]:.1f}")
            
            # SKIP PROBLEMATIC INDICATORS temporarily to isolate the error
            # Only calculate essential indicators needed for MA strategy
            
            # Add minimal required indicators with safe defaults
            try:
                # Bollinger Bands (optional)
                bb_sma = df['close'].rolling(window=default_config['bb_period'], min_periods=1).mean()
                bb_std = df['close'].rolling(window=default_config['bb_period'], min_periods=1).std()
                result_df['bb_upper'] = bb_sma + (bb_std * default_config['bb_std'])
                result_df['bb_lower'] = bb_sma - (bb_std * default_config['bb_std'])
                result_df['bb_middle'] = bb_sma
            except Exception as e:
                logger.warning(f"Skipping Bollinger Bands: {e}")
            
            # Add basic volume analysis
            try:
                result_df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
                result_df['volume_ratio'] = df['volume'] / (result_df['volume_ma'] + 1e-10)
            except Exception as e:
                logger.warning(f"Skipping volume analysis: {e}")
                result_df['volume_ma'] = df['volume']
                result_df['volume_ratio'] = pd.Series(1.0, index=df.index)
            
            # Simple trend strength calculation
            try:
                price_changes = df['close'].diff().rolling(window=default_config['rsi_period'], min_periods=1)
                result_df['trend_strength'] = (price_changes.apply(lambda x: (x > 0).sum()) / default_config['rsi_period']).fillna(0.5)
            except Exception as e:
                logger.warning(f"Skipping trend strength: {e}")
                result_df['trend_strength'] = pd.Series(0.5, index=df.index)
            
            logger.info(f"Calculated indicators for {len(result_df)} bars")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df.copy()
        
        return result_df