"""
Configuration settings for the crypto trading system.
Loads environment variables and provides typed configuration.
"""
import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv('.env.local')

class AlpacaSettings(BaseSettings):
    """Alpaca API configuration"""
    endpoint: str = Field(default="https://paper-api.alpaca.markets/v2", env="ALPACA_ENDPOINT")
    key: str = Field(env="ALPACA_KEY")
    secret: str = Field(env="ALPACA_SECRET")
    
    class Config:
        env_prefix = "ALPACA_"

class FirebaseSettings(BaseSettings):
    """Firebase configuration"""
    api_key: str = Field(env="FIREBASE_API_KEY")
    auth_domain: str = Field(env="FIREBASE_AUTH_DOMAIN") 
    project_id: str = Field(env="FIREBASE_PROJECT_ID")
    storage_bucket: str = Field(env="FIREBASE_STORAGE_BUCKET")
    messaging_sender_id: str = Field(env="FIREBASE_MESSAGING_SENDER_ID")
    app_id: str = Field(env="FIREBASE_APP_ID")
    measurement_id: str = Field(env="FIREBASE_MEASUREMENT_ID")
    
    class Config:
        env_prefix = "FIREBASE_"

class TradingSettings(BaseSettings):
    """Trading configuration"""
    # Asset to trade
    symbol: str = Field(default="BTCUSD", env="TRADING_SYMBOL")
    
    # Risk management
    max_position_size: float = Field(default=1000.0, env="MAX_POSITION_SIZE")  # USD
    risk_per_trade: float = Field(default=0.02, env="RISK_PER_TRADE")  # 2%
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")  # 5%
    
    # Strategy parameters
    fast_ma_period: int = Field(default=12, env="FAST_MA_PERIOD")
    slow_ma_period: int = Field(default=24, env="SLOW_MA_PERIOD")
    rsi_period: int = Field(default=14, env="RSI_PERIOD")
    rsi_oversold: float = Field(default=30.0, env="RSI_OVERSOLD")
    rsi_overbought: float = Field(default=70.0, env="RSI_OVERBOUGHT")
    
    # Data settings
    data_timeframe: str = Field(default="1Min", env="DATA_TIMEFRAME")
    lookback_periods: int = Field(default=100, env="LOOKBACK_PERIODS")
    
    # Execution settings
    order_type: str = Field(default="market", env="ORDER_TYPE")
    
    class Config:
        env_prefix = "TRADING_"

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_prefix = "LOG_"

class AppSettings(BaseSettings):
    """Main application settings"""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Component settings
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    firebase: FirebaseSettings = Field(default_factory=FirebaseSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

# Global settings instance
settings = AppSettings()

def get_settings() -> AppSettings:
    """Get application settings"""
    return settings