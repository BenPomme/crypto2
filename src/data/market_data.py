"""
Market Data Provider for fetching crypto data from Alpaca
Handles both historical and real-time data feeds
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging
import pytz
from alpaca_trade_api import REST, TimeFrame
from alpaca_trade_api.common import URL

from config.settings import get_settings

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        pass
    
    @abstractmethod
    def is_market_open(self, symbol: Optional[str] = None) -> bool:
        """Check if market is open for given symbol"""
        pass
    
    @abstractmethod
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto pair"""
        pass

class AlpacaDataProvider(MarketDataProvider):
    """Alpaca market data provider for crypto data"""
    
    def __init__(self):
        settings = get_settings()
        
        # Ensure base URL doesn't end with /v2 to prevent duplication
        base_url = settings.alpaca.endpoint.rstrip('/v2').rstrip('/')
        
        self.api = REST(
            key_id=settings.alpaca.key,
            secret_key=settings.alpaca.secret,
            base_url=base_url,
            api_version='v2'
        )
        
        # Timeframe mapping
        self.timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Minute),
            "15Min": TimeFrame(15, TimeFrame.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
        
        # Timezone setup for market hours
        self.us_eastern = pytz.timezone('US/Eastern')
        self.utc = pytz.UTC
        
        # Known crypto symbols (common crypto pairs on Alpaca)
        self.crypto_symbols = {
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'DOGE/USD',
            'ADA/USD', 'DOT/USD', 'UNI/USD', 'LINK/USD', 'AAVE/USD',
            'SOL/USD', 'AVAX/USD',  # Added missing 4 Crypto Pairs Strategy symbols
            'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'DOGEUSD',
            'ADAUSD', 'DOTUSD', 'UNIUSD', 'LINKUSD', 'AAVEUSD',
            'SOLUSD', 'AVAXUSD'  # Added missing 4 Crypto Pairs Strategy symbols (no slash format)
        }
        
        logger.info("Alpaca data provider initialized")
    
    def get_historical_data(self, symbol: str, timeframe: str = "1Min", 
                          periods: int = 100) -> pd.DataFrame:
        """
        Get historical OHLCV data for specified periods
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            timeframe: Timeframe string (e.g., '1Min', '5Min')
            periods: Number of periods to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get end time as now, start time based on periods (in UTC)
            from datetime import timezone
            end_time = datetime.now(timezone.utc)
            
            # Calculate start time based on timeframe and periods
            if timeframe == "1Min":
                start_time = end_time - timedelta(minutes=periods)
            elif timeframe == "5Min":
                start_time = end_time - timedelta(minutes=periods * 5)
            elif timeframe == "15Min":
                start_time = end_time - timedelta(minutes=periods * 15)
            elif timeframe == "1Hour":
                start_time = end_time - timedelta(hours=periods)
            elif timeframe == "1Day":
                start_time = end_time - timedelta(days=periods)
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Get timeframe object
            tf = self.timeframe_map.get(timeframe)
            if not tf:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Format timestamps as RFC3339 strings for Alpaca API
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Fetch data from Alpaca
            logger.info(f"Fetching {periods} periods of {timeframe} data for {symbol}")
            bars = self.api.get_crypto_bars(
                symbol,
                timeframe=tf,
                start=start_str,
                end=end_str,
                limit=periods
            )
            
            # Convert to DataFrame
            df = bars.df
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            initial_count = len(df)
            logger.debug(f"Received {initial_count} bars before filtering for {symbol}")
            
            # Filter for specific exchange if needed (Coinbase Pro)
            # Only filter if we have exchange column and CBSE data is available
            if 'exchange' in df.columns:
                available_exchanges = df['exchange'].unique()
                logger.debug(f"Available exchanges for {symbol}: {list(available_exchanges)}")
                
                # Prefer CBSE but fall back to all data if CBSE is insufficient
                cbse_data = df[df.exchange == 'CBSE'].copy()
                if len(cbse_data) >= max(20, periods * 0.5):  # At least 20 bars or 50% of requested
                    df = cbse_data
                    logger.debug(f"Using CBSE exchange data: {len(df)} bars")
                else:
                    logger.info(f"CBSE has insufficient data ({len(cbse_data)} bars), using all exchanges")
                    # Keep all exchange data for better coverage
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            # Log data quality metrics
            final_count = len(df)
            coverage_pct = (final_count / periods) * 100 if periods > 0 else 0
            logger.info(f"Successfully fetched {final_count}/{periods} bars for {symbol} ({coverage_pct:.1f}% coverage)")
            
            if final_count < periods * 0.8:  # Less than 80% coverage
                logger.warning(f"Low data coverage for {symbol}: {final_count}/{periods} bars ({coverage_pct:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    def get_latest_bar(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest OHLCV bar for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with latest OHLCV bar data
        """
        try:
            # Get the latest 2 bars to ensure we have current data
            historical_data = self.get_historical_data(symbol, timeframe="1Min", periods=2)
            
            if historical_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Get the most recent bar
            latest_row = historical_data.iloc[-1]
            
            # Ensure timestamp is properly formatted
            timestamp = latest_row.name
            if pd.isna(timestamp):
                timestamp = datetime.now(self.utc)
            elif not isinstance(timestamp, datetime):
                timestamp = pd.to_datetime(timestamp)
            
            return {
                'timestamp': timestamp,
                'open': float(latest_row['open']),
                'high': float(latest_row['high']),
                'low': float(latest_row['low']),
                'close': float(latest_row['close']),
                'volume': float(latest_row['volume'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            raise
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        try:
            # Get latest bar (most recent 1-minute candle)
            bars = self.api.get_crypto_bars(
                symbol,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            df = bars.df
            if not df.empty:
                return float(df.iloc[-1]['close'])
            
            raise ValueError("No price data available")
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            raise
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is a crypto pair
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is a crypto pair
        """
        # Normalize symbol format
        normalized = symbol.upper().replace('/', '')
        with_slash = symbol.upper()
        
        return (normalized in self.crypto_symbols or 
                with_slash in self.crypto_symbols or
                any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'LTC', 'BCH', 'DOGE', 'ADA', 'DOT', 'UNI', 'LINK', 'AAVE', 'SOL', 'AVAX']))
    
    def is_us_market_open(self, include_extended_hours: bool = True) -> bool:
        """
        Check if US stock market is currently open
        
        Args:
            include_extended_hours: Include pre-market and after-hours trading
            
        Returns:
            True if US market is open
        """
        try:
            # Get current time in US Eastern timezone
            now_utc = datetime.now(self.utc)
            now_et = now_utc.astimezone(self.us_eastern)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if now_et.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Market hours in ET
            if include_extended_hours:
                # Extended hours: 4:00 AM - 8:00 PM ET
                market_open = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
                market_close = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                # Regular hours: 9:30 AM - 4:00 PM ET
                market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now_et <= market_close
            
        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            # Default to open to avoid blocking trades due to timezone issues
            return True
    
    def is_market_open(self, symbol: Optional[str] = None) -> bool:
        """
        Check if market is open for given symbol
        
        Args:
            symbol: Trading symbol to check. If None, checks general market status
            
        Returns:
            True if market is open for the symbol
        """
        if symbol is None:
            # Default to US market check
            return self.is_us_market_open()
        
        if self.is_crypto_symbol(symbol):
            # Crypto markets are always open
            return True
        else:
            # Stock market - check US market hours
            return self.is_us_market_open()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate API connection"""
        try:
            account = self.api.get_account()
            logger.info(f"Connection validated. Account: {account.id}")
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False