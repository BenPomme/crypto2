"""
Stock Trading Settings
Optional configuration for stock trading features
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class StockTradingSettings(BaseSettings):
    """Stock trading configuration"""
    enable_stock_trading: bool = Field(default=False, env="ENABLE_STOCK_TRADING")
    stock_symbols: str = Field(default="", env="STOCK_SYMBOLS")
    stock_strategies: str = Field(default="mean_reversion", env="STOCK_STRATEGY")
    enable_short_selling: bool = Field(default=False, env="ENABLE_SHORT_SELLING")
    max_short_exposure: float = Field(default=0.5, env="MAX_SHORT_EXPOSURE")
    stock_risk_per_trade: float = Field(default=0.01, env="STOCK_RISK_PER_TRADE")
    
    class Config:
        env_prefix = "STOCK_"

# Create a function to check if stock trading is enabled
def is_stock_trading_enabled() -> bool:
    """Check if stock trading is enabled without breaking existing system"""
    try:
        settings = StockTradingSettings()
        return settings.enable_stock_trading
    except:
        return False

def get_stock_settings() -> Optional[StockTradingSettings]:
    """Get stock settings if enabled"""
    try:
        if is_stock_trading_enabled():
            return StockTradingSettings()
        return None
    except:
        return None