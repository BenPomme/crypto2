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
            # Moving Averages - essential for MA strategy
            logger.info(f"Calculating moving averages with {len(df)} bars")
            try:
                sma_fast = cls.simple_moving_average(df['close'], default_config['ma_fast'])
                sma_slow = cls.simple_moving_average(df['close'], default_config['ma_slow'])
                ema_fast = cls.exponential_moving_average(df['close'], default_config['ma_fast'])
                ema_slow = cls.exponential_moving_average(df['close'], default_config['ma_slow'])
                
                # Ensure we got valid results
                result_df['sma_fast'] = sma_fast if sma_fast is not None else pd.Series(df['close'].rolling(default_config['ma_fast']).mean(), dtype='float64')
                result_df['sma_slow'] = sma_slow if sma_slow is not None else pd.Series(df['close'].rolling(default_config['ma_slow']).mean(), dtype='float64')
                result_df['ema_fast'] = ema_fast if ema_fast is not None else pd.Series(df['close'].ewm(span=default_config['ma_fast']).mean(), dtype='float64')
                result_df['ema_slow'] = ema_slow if ema_slow is not None else pd.Series(df['close'].ewm(span=default_config['ma_slow']).mean(), dtype='float64')
                
                logger.info(f"Successfully calculated MAs: SMA_fast={result_df['sma_fast'].iloc[-1]:.2f}, SMA_slow={result_df['sma_slow'].iloc[-1]:.2f}")
            except Exception as e:
                logger.error(f"Error calculating moving averages: {e}")
                # Create fallback manual calculations
                result_df['sma_fast'] = df['close'].rolling(default_config['ma_fast']).mean()
                result_df['sma_slow'] = df['close'].rolling(default_config['ma_slow']).mean()
                result_df['ema_fast'] = df['close'].ewm(span=default_config['ma_fast']).mean()
                result_df['ema_slow'] = df['close'].ewm(span=default_config['ma_slow']).mean()
                logger.info(f"Used fallback MA calculations")
            
            # RSI - essential for strategy filters
            try:
                rsi = cls.relative_strength_index(df['close'], default_config['rsi_period'])
                result_df['rsi'] = rsi if rsi is not None else pd.Series(50.0, index=df.index, dtype='float64')
                logger.info(f"Successfully calculated RSI: {result_df['rsi'].iloc[-1]:.1f}")
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
                result_df['rsi'] = pd.Series(50.0, index=df.index, dtype='float64')
                logger.info(f"Used fallback RSI value: 50.0")
            
            # Bollinger Bands
            bb_data = cls.bollinger_bands(
                df['close'], default_config['bb_period'], default_config['bb_std']
            )
            if bb_data is not None and not bb_data.empty:
                result_df = result_df.join(bb_data, rsuffix='_bb')
            
            # ATR
            result_df['atr'] = cls.average_true_range(
                df['high'], df['low'], df['close'], default_config['atr_period']
            )
            
            # OBV
            result_df['obv'] = cls.on_balance_volume(df['close'], df['volume'])
            
            # MFI
            result_df['mfi'] = cls.money_flow_index(
                df['high'], df['low'], df['close'], df['volume'], 
                default_config['mfi_period']
            )
            
            # MACD
            macd_data = cls.macd(df['close'])
            if macd_data is not None and not macd_data.empty:
                result_df = result_df.join(macd_data, rsuffix='_macd')
            
            # Stochastic
            stoch_data = cls.stochastic_oscillator(
                df['high'], df['low'], df['close'],
                default_config['stoch_k'], default_config['stoch_d']
            )
            if stoch_data is not None and not stoch_data.empty:
                result_df = result_df.join(stoch_data, rsuffix='_stoch')
            
            # Williams %R
            result_df['williams_r'] = cls.williams_r(
                df['high'], df['low'], df['close']
            )
            
            # CCI
            result_df['cci'] = cls.commodity_channel_index(
                df['high'], df['low'], df['close']
            )
            
            # Calculate trend strength
            result_df['trend_strength'] = result_df['close'].rolling(
                window=default_config['rsi_period']
            ).apply(lambda x: cls.trend_strength(pd.Series(x)))
            
            logger.info(f"Calculated indicators for {len(result_df)} bars")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df.copy()
        
        return result_df