"""
Volume Data Manager
Combines Alpaca price data with real Binance volume data for accurate analysis
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from .binance_volume_provider import BinanceVolumeProvider

logger = logging.getLogger(__name__)

class VolumeDataManager:
    """
    Manages volume data by combining Alpaca price data with Binance volume data
    Provides realistic volume estimates for crypto trading analysis
    """
    
    def __init__(self):
        """Initialize volume data manager"""
        self.binance_provider = BinanceVolumeProvider()
        self.volume_cache = {}  # Symbol -> cached volume estimates
        self.last_cache_update = {}  # Symbol -> last cache timestamp
        self.cache_duration = 60  # Cache for 60 seconds
        
        logger.info("Volume Data Manager initialized")
    
    def start(self) -> None:
        """Start the volume data providers"""
        logger.info("🚀 Starting real-time volume data collection...")
        self.binance_provider.start()
    
    def stop(self) -> None:
        """Stop the volume data providers"""
        self.binance_provider.stop()
        logger.info("Volume data collection stopped")
    
    def enhance_dataframe_volume(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Enhance a DataFrame with realistic volume estimates
        
        Args:
            df: DataFrame with OHLCV data from Alpaca
            symbol: Trading symbol (e.g., 'BTC/USD')
            
        Returns:
            DataFrame with enhanced volume data
        """
        if df.empty or 'volume' not in df.columns:
            return df
            
        try:
            # Get current market volume rate from Binance
            volume_rate = self._get_volume_rate(symbol)
            if not volume_rate:
                logger.warning(f"No volume rate available for {symbol}, using price-based proxy")
                return self._use_price_based_volume(df)
            
            # Calculate realistic volume estimates for each bar
            enhanced_df = df.copy()
            
            # Estimate volume per bar based on market data
            bar_volume_estimates = self._estimate_bar_volumes(df, volume_rate, symbol)
            
            # Replace Alpaca's tiny volumes with realistic estimates
            enhanced_df['volume_original'] = df['volume']  # Keep original for reference
            enhanced_df['volume'] = bar_volume_estimates
            enhanced_df['volume_source'] = 'binance_estimated'
            
            # Calculate dollar volume for indicators
            enhanced_df['volume_usd'] = enhanced_df['volume'] * enhanced_df['close']
            
            logger.debug(f"Enhanced {symbol} volume: Original sum={df['volume'].sum():.3f}, "
                        f"Enhanced sum=${enhanced_df['volume_usd'].sum():,.0f}")
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error enhancing volume for {symbol}: {e}")
            return self._use_price_based_volume(df)
    
    def _get_volume_rate(self, symbol: str) -> Optional[float]:
        """Get volume rate (USD per minute) for a symbol"""
        # Check cache first
        if self._is_cache_valid(symbol):
            return self.volume_cache[symbol]
        
        # Get fresh data from Binance
        volume_rate = self.binance_provider.get_current_volume_rate(symbol)
        
        if volume_rate:
            # Cache the result
            self.volume_cache[symbol] = volume_rate
            self.last_cache_update[symbol] = datetime.now()
            
            # Log major exchanges volume for context
            volume_24h = self.binance_provider.get_24h_volume_usd(symbol)
            if volume_24h:
                logger.info(f"📊 {symbol} Binance Volume: ${volume_24h:,.0f}/24h (${volume_rate:,.0f}/min)")
        
        return volume_rate
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached volume rate is still valid"""
        if symbol not in self.volume_cache:
            return False
            
        last_update = self.last_cache_update.get(symbol)
        if not last_update:
            return False
            
        return (datetime.now() - last_update).seconds < self.cache_duration
    
    def _estimate_bar_volumes(self, df: pd.DataFrame, volume_rate: float, symbol: str) -> pd.Series:
        """
        Estimate realistic volume for each bar based on price action and market volume
        
        Args:
            df: OHLCV DataFrame
            volume_rate: USD volume per minute from Binance
            symbol: Trading symbol
            
        Returns:
            Series with estimated volumes in base currency
        """
        if len(df) == 0:
            return pd.Series([], dtype=float)
        
        # Base volume per bar (assuming 1-minute bars)
        base_volume_usd = volume_rate
        
        # Adjust volume based on price volatility (more volatility = more volume)
        price_volatility = (df['high'] - df['low']) / df['close']
        volatility_multiplier = 1 + (price_volatility * 2)  # 0-200% boost for high volatility
        
        # Adjust volume based on price movement (big moves = more volume)
        price_change = abs(df['close'].pct_change().fillna(0))
        movement_multiplier = 1 + (price_change * 5)  # 0-500% boost for big moves
        
        # Combine factors
        volume_multiplier = (volatility_multiplier * movement_multiplier).clip(0.1, 10)  # Reasonable bounds
        
        # Calculate USD volume per bar
        estimated_volume_usd = base_volume_usd * volume_multiplier
        
        # Convert to base currency volume (BTC, ETH, etc.)
        estimated_volume_base = estimated_volume_usd / df['close']
        
        # Add some randomness to make it more realistic
        noise = pd.Series(index=df.index, data=1).apply(lambda x: 0.8 + (hash(str(x)) % 100) / 250)  # 0.8-1.2x
        estimated_volume_base *= noise
        
        return estimated_volume_base.fillna(0)
    
    def _use_price_based_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback to price-based volume proxy when real data unavailable"""
        enhanced_df = df.copy()
        
        # Use price volatility as volume proxy
        price_movement = abs(df['close'].pct_change()).fillna(0)
        avg_price = df['close'].mean()
        
        # Estimate volume based on price action (more movement = more volume)
        proxy_volume = price_movement * avg_price * 100  # Scale factor
        
        enhanced_df['volume_original'] = df['volume']
        enhanced_df['volume'] = proxy_volume
        enhanced_df['volume_source'] = 'price_proxy'
        enhanced_df['volume_usd'] = proxy_volume * df['close']
        
        logger.warning("Using price-based volume proxy - may be less accurate")
        return enhanced_df
    
    def get_status(self) -> Dict[str, Any]:
        """Get volume manager status"""
        binance_status = self.binance_provider.get_status()
        
        return {
            'binance_provider': binance_status,
            'cache_entries': len(self.volume_cache),
            'cached_symbols': list(self.volume_cache.keys()),
            'last_cache_updates': {
                symbol: update.isoformat() 
                for symbol, update in self.last_cache_update.items()
            }
        }
    
    def get_volume_summary(self) -> Dict[str, Any]:
        """Get summary of current volume data for all symbols"""
        summary = {}
        
        for our_symbol in ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD']:
            volume_24h = self.binance_provider.get_24h_volume_usd(our_symbol)
            volume_rate = self.binance_provider.get_current_volume_rate(our_symbol)
            
            summary[our_symbol] = {
                'volume_24h_usd': volume_24h,
                'volume_per_minute_usd': volume_rate,
                'data_available': volume_24h is not None
            }
        
        return summary