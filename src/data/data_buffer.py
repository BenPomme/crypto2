"""
Data Buffer for maintaining in-memory sliding window of market data
Optimized for real-time trading operations
"""
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DataBuffer:
    """
    Circular buffer for storing recent market data
    Maintains a sliding window of OHLCV data for indicators
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize data buffer
        
        Args:
            max_size: Maximum number of bars to keep in memory
        """
        self.max_size = max_size
        self._data = deque(maxlen=max_size)
        self._symbol = None
        self._timeframe = None
        
        logger.info(f"DataBuffer initialized with max_size={max_size}")
    
    def add_bar(self, bar_data: Dict) -> None:
        """
        Add a new bar to the buffer
        
        Args:
            bar_data: Dictionary containing OHLCV data
        """
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Validate bar data
        for field in required_fields:
            if field not in bar_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure timestamp is datetime
        if not isinstance(bar_data['timestamp'], datetime):
            if isinstance(bar_data['timestamp'], str):
                bar_data['timestamp'] = pd.to_datetime(bar_data['timestamp'])
            else:
                raise ValueError("Timestamp must be datetime or string")
        
        self._data.append(bar_data.copy())
        
        if len(self._data) % 100 == 0:  # Log every 100 bars
            logger.debug(f"Buffer size: {len(self._data)}/{self.max_size}")
    
    def bulk_add(self, df: pd.DataFrame) -> None:
        """
        Add multiple bars from DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        """
        if df.empty:
            logger.warning("Attempted to add empty DataFrame to buffer")
            return
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            # If index is datetime, use it as timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                raise ValueError("DataFrame must have timestamp column or datetime index")
        
        # Add each row to buffer
        for _, row in df.iterrows():
            bar_data = {
                'timestamp': row['timestamp'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            self.add_bar(bar_data)
        
        logger.info(f"Added {len(df)} bars to buffer. Total size: {len(self._data)}")
    
    def get_dataframe(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Get buffer data as DataFrame
        
        Args:
            periods: Number of recent periods to return (None for all)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self._data:
            return pd.DataFrame()
        
        # Convert deque to list for slicing
        data_list = list(self._data)
        
        if periods is not None:
            data_list = data_list[-periods:]
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_latest_bar(self) -> Optional[Dict]:
        """Get the most recent bar"""
        if not self._data:
            return None
        return dict(self._data[-1])
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest close price"""
        latest = self.get_latest_bar()
        return latest['close'] if latest else None
    
    def get_price_series(self, periods: Optional[int] = None) -> pd.Series:
        """
        Get close price series
        
        Args:
            periods: Number of recent periods
            
        Returns:
            Series of close prices
        """
        df = self.get_dataframe(periods)
        return df['close'] if not df.empty else pd.Series()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self._data)
    
    def is_ready(self, min_periods: int) -> bool:
        """Check if buffer has enough data for analysis"""
        return len(self._data) >= min_periods
    
    def clear(self) -> None:
        """Clear all data from buffer"""
        self._data.clear()
        logger.info("Buffer cleared")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        if not self._data:
            return {
                'size': 0,
                'max_size': self.max_size,
                'is_full': False,
                'latest_timestamp': None,
                'oldest_timestamp': None
            }
        
        return {
            'size': len(self._data),
            'max_size': self.max_size,
            'is_full': len(self._data) == self.max_size,
            'latest_timestamp': self._data[-1]['timestamp'],
            'oldest_timestamp': self._data[0]['timestamp'],
            'latest_price': self._data[-1]['close']
        }