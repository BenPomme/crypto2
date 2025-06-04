"""
Market Data Provider for fetching crypto data from Alpaca
Handles both historical and real-time data feeds
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging
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
    def is_market_open(self) -> bool:
        """Check if market is open (crypto markets are always open)"""
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
            
            # Filter for specific exchange if needed (Coinbase Pro)
            if 'exchange' in df.columns:
                df = df[df.exchange == 'CBSE'].copy()
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            logger.info(f"Successfully fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
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
    
    def is_market_open(self) -> bool:
        """Crypto markets are always open"""
        return True
    
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