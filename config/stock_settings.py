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
    stock_order_type: str = Field(default="market", env="STOCK_ORDER_TYPE")
    stock_stop_loss_pct: float = Field(default=0.01, env="STOCK_STOP_LOSS_PCT")
    stock_take_profit_pct: float = Field(default=0.02, env="STOCK_TAKE_PROFIT_PCT")
    
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
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug environment variables
        logger.info(f"DEBUG get_stock_settings: STOCK_SYMBOLS='{os.environ.get('STOCK_SYMBOLS', 'NOT SET')}'")
        logger.info(f"DEBUG get_stock_settings: ENABLE_STOCK_TRADING='{os.environ.get('ENABLE_STOCK_TRADING', 'NOT SET')}'")
        
        if is_stock_trading_enabled():
            settings = StockTradingSettings()
            logger.info(f"DEBUG StockTradingSettings created: stock_symbols='{settings.stock_symbols}', type={type(settings.stock_symbols)}")
            return settings
        logger.warning("Stock trading not enabled")
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading stock settings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None