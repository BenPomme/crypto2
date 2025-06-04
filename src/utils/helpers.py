"""
Helper utilities for the trading system
Common functions and utilities used across modules
"""
import pandas as pd
import numpy as np
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def format_currency(amount: float, symbol: str = "$") -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Amount to format
        symbol: Currency symbol
        
    Returns:
        Formatted currency string
    """
    if pd.isna(amount) or amount is None:
        return f"{symbol}0.00"
    
    if abs(amount) >= 1_000_000:
        return f"{symbol}{amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"{symbol}{amount/1_000:.2f}K"
    else:
        return f"{symbol}{amount:.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage for display
    
    Args:
        value: Percentage value (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "0.00%"
    
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safe division with default for zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    
    return numerator / denominator

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        
    Returns:
        Returns series
    """
    return prices.pct_change(periods=periods)

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility from returns
    
    Args:
        returns: Returns series
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility value
    """
    if len(returns) < 2:
        return 0.0
    
    vol = returns.std()
    
    if annualize:
        vol *= np.sqrt(252)  # Assuming 252 trading days per year
    
    return vol

def normalize_symbol(symbol: str) -> str:
    """
    Normalize trading symbol format
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Normalized symbol
    """
    return symbol.upper().strip()

def validate_price(price: Any) -> bool:
    """
    Validate if price is valid
    
    Args:
        price: Price value to validate
        
    Returns:
        True if valid price
    """
    try:
        price_float = float(price)
        return price_float > 0 and not pd.isna(price_float) and np.isfinite(price_float)
    except (ValueError, TypeError):
        return False

def validate_quantity(quantity: Any) -> bool:
    """
    Validate if quantity is valid
    
    Args:
        quantity: Quantity value to validate
        
    Returns:
        True if valid quantity
    """
    try:
        qty_float = float(quantity)
        return qty_float > 0 and not pd.isna(qty_float) and np.isfinite(qty_float)
    except (ValueError, TypeError):
        return False

def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to valid tick size
    
    Args:
        price: Price to round
        tick_size: Minimum tick size
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    return round(price / tick_size) * tick_size

def is_market_hours(current_time: Optional[datetime] = None) -> bool:
    """
    Check if current time is within market hours
    Note: Crypto markets are 24/7, so this always returns True
    Can be modified for traditional market hours if needed
    
    Args:
        current_time: Time to check (current time if None)
        
    Returns:
        True if market is open
    """
    # Crypto markets are always open
    return True

def calculate_position_size_units(usd_amount: float, price: float) -> float:
    """
    Calculate position size in units from USD amount
    
    Args:
        usd_amount: USD amount to invest
        price: Current price per unit
        
    Returns:
        Number of units to buy
    """
    if not validate_price(price) or usd_amount <= 0:
        return 0.0
    
    return usd_amount / price

def calculate_position_value(quantity: float, price: float) -> float:
    """
    Calculate position value in USD
    
    Args:
        quantity: Number of units
        price: Price per unit
        
    Returns:
        Total position value in USD
    """
    if not validate_quantity(quantity) or not validate_price(price):
        return 0.0
    
    return quantity * price

def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Get number of business days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of business days
    """
    return pd.bdate_range(start_date, end_date).size

def ensure_timezone_aware(dt: datetime, timezone_str: str = 'UTC') -> datetime:
    """
    Ensure datetime is timezone aware
    
    Args:
        dt: Datetime object
        timezone_str: Timezone string
        
    Returns:
        Timezone-aware datetime
    """
    import pytz
    
    if dt.tzinfo is None:
        tz = pytz.timezone(timezone_str)
        return tz.localize(dt)
    
    return dt

def chunked_list(lst: list, chunk_size: int):
    """
    Break list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying function calls on exception
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by removing invalid values
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill then backward fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    return df

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    
    if excess_returns.std() == 0:
        return 0.0
    
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Maximum drawdown as negative percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    
    return drawdown.min()

def format_timespan(seconds: float) -> str:
    """
    Format timespan in human readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timespan string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"