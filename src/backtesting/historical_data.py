"""
Historical Data Provider for Backtesting
Fetches and manages historical market data from Alpaca API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import pickle
import hashlib

from alpaca_trade_api import REST
from config.settings import get_settings

logger = logging.getLogger(__name__)

class AlpacaHistoricalDataProvider:
    """
    High-performance historical data provider for backtesting
    Features: Caching, batch fetching, data quality validation
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize historical data provider
        
        Args:
            cache_dir: Directory for caching historical data
        """
        self.settings = get_settings()
        self.api = REST(
            key_id=self.settings.alpaca.api_key,
            secret_key=self.settings.alpaca.secret_key,
            base_url=self.settings.alpaca.base_url
        )
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data quality settings
        self.min_volume_threshold = 100  # Minimum volume for valid bar
        self.max_gap_hours = 4  # Maximum gap between bars before flagging
        
        logger.info(f"Historical data provider initialized with cache: {cache_dir}")
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: datetime = None,
        end: datetime = None,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with intelligent caching
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            timeframe: Data timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            start: Start datetime (if None, calculated from days_back)
            end: End datetime (if None, uses current time)
            days_back: Days of historical data if start not specified
            
        Returns:
            DataFrame with OHLCV data and timestamp index
        """
        try:
            # Calculate date range
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(days=days_back)
            
            # Check cache first
            cache_key = self._generate_cache_key(symbol, timeframe, start, end)
            cached_data = self._load_from_cache(cache_key)
            
            if cached_data is not None:
                logger.info(f"Loaded {symbol} {timeframe} data from cache: {len(cached_data)} bars")
                return cached_data
            
            # Fetch from Alpaca API
            logger.info(f"Fetching {symbol} {timeframe} data: {start.date()} to {end.date()}")
            
            # Convert timeframe for Alpaca API
            alpaca_timeframe = self._convert_timeframe(timeframe)
            
            # Fetch data with retry logic
            raw_data = self._fetch_with_retry(symbol, alpaca_timeframe, start, end)
            
            if raw_data is None or raw_data.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Process and validate data
            processed_data = self._process_raw_data(raw_data, symbol, timeframe)
            
            # Cache the processed data
            self._save_to_cache(cache_key, processed_data)
            
            logger.info(f"Fetched and cached {symbol} {timeframe}: {len(processed_data)} bars")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1Min",
        start: datetime = None,
        end: datetime = None,
        days_back: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Efficiently fetch historical data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime  
            days_back: Days of historical data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                data = self.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    days_back=days_back
                )
                
                if not data.empty:
                    results[symbol] = data
                    logger.debug(f"Fetched {symbol}: {len(data)} bars")
                else:
                    logger.warning(f"No data for {symbol}")
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_data_quality_report(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Generate data quality report for historical data
        
        Args:
            data: Historical data DataFrame
            symbol: Trading symbol
            
        Returns:
            Dictionary with data quality metrics
        """
        if data.empty:
            return {'status': 'empty', 'issues': ['No data available']}
        
        issues = []
        
        # Check for missing data
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"Missing values: {missing_values}")
        
        # Check for zero volume
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Zero volume bars: {zero_volume}")
        
        # Check for data gaps
        time_diffs = data.index.to_series().diff()[1:]
        max_gap = time_diffs.max()
        expected_interval = time_diffs.mode()[0] if not time_diffs.empty else pd.Timedelta(minutes=1)
        
        if max_gap > expected_interval * 10:  # Gap > 10x expected interval
            issues.append(f"Large data gap detected: {max_gap}")
        
        # Check for suspicious price movements
        returns = data['close'].pct_change().abs()
        extreme_moves = (returns > 0.1).sum()  # >10% moves
        if extreme_moves > len(data) * 0.01:  # More than 1% of bars
            issues.append(f"Excessive extreme price movements: {extreme_moves}")
        
        # Check for low volume periods
        low_volume = (data['volume'] < self.min_volume_threshold).sum()
        if low_volume > len(data) * 0.1:  # More than 10% of bars
            issues.append(f"High frequency of low volume bars: {low_volume}")
        
        return {
            'status': 'good' if not issues else 'issues_detected',
            'total_bars': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'completeness': f"{(1 - missing_values/data.size)*100:.1f}%",
            'issues': issues
        }
    
    def _generate_cache_key(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
        """Generate unique cache key for data request"""
        key_string = f"{symbol}_{timeframe}_{start.isoformat()}_{end.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if exists and not expired"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired (24 hours for intraday data)
        if time.time() - cache_file.stat().st_mtime > 86400:
            logger.debug(f"Cache expired for key {cache_key}")
            cache_file.unlink()  # Remove expired cache
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Error loading cache {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_key}: {e}")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca API format"""
        mapping = {
            '1Min': '1Min',
            '5Min': '5Min', 
            '15Min': '15Min',
            '1Hour': '1Hour',
            '1Day': '1Day'
        }
        return mapping.get(timeframe, '1Min')
    
    def _fetch_with_retry(self, symbol: str, timeframe: str, start: datetime, end: datetime, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data with retry logic for robustness"""
        for attempt in range(max_retries):
            try:
                # Convert symbol format for Alpaca
                alpaca_symbol = symbol.replace('/', '')
                
                bars = self.api.get_crypto_bars(
                    symbol=alpaca_symbol,
                    timeframe=timeframe,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    asof=None,
                    page_token=None,
                    limit=10000
                ).df
                
                return bars
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed for {symbol}")
                    return None
    
    def _process_raw_data(self, raw_data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Process and validate raw data from Alpaca"""
        if raw_data.empty:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns for {symbol}: {missing_columns}")
            return pd.DataFrame()
        
        # Clean and validate data
        processed = raw_data.copy()
        
        # Remove invalid data points
        processed = processed[processed['volume'] >= 0]  # Non-negative volume
        processed = processed[processed['high'] >= processed['low']]  # High >= Low
        processed = processed[processed['close'] > 0]  # Positive prices
        
        # Fill missing values with forward fill (limited to 3 periods)
        processed = processed.fillna(method='ffill', limit=3)
        
        # Remove remaining NaN rows
        processed = processed.dropna()
        
        # Sort by timestamp
        processed = processed.sort_index()
        
        # Add derived columns for analysis
        processed['returns'] = processed['close'].pct_change()
        processed['volume_ma'] = processed['volume'].rolling(window=20, min_periods=1).mean()
        processed['volatility'] = processed['returns'].rolling(window=20, min_periods=1).std()
        
        return processed
    
    def clear_cache(self, symbol: str = None) -> None:
        """Clear cached data"""
        if symbol:
            # Clear cache for specific symbol
            for cache_file in self.cache_dir.glob(f"*{symbol}*.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared all cached data")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cached_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }