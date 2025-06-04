"""
Position Sizing Module
Calculates optimal position sizes based on risk management rules
"""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_RISK = "percent_risk"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    size_usd: float
    size_units: float
    method: SizingMethod
    risk_amount: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'size_usd': self.size_usd,
            'size_units': self.size_units,
            'method': self.method.value,
            'risk_amount': self.risk_amount,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

class PositionSizer:
    """
    Position sizing calculator implementing various sizing methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize position sizer
        
        Args:
            config: Configuration for position sizing
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'method': SizingMethod.PERCENT_RISK,
            'risk_per_trade': 0.02,  # 2% of capital per trade
            'max_position_size': 0.25,  # 25% max of capital per position
            'min_position_size': 100,  # Minimum $100 position
            'volatility_lookback': 20,
            'volatility_multiplier': 1.0,
            'kelly_lookback': 100,
            'max_leverage': 1.0,  # No leverage for crypto
        }
        
        self.config = {**self.default_config, **self.config}
        logger.info("PositionSizer initialized")
    
    def calculate_position_size(self, 
                              account_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float] = None,
                              volatility: Optional[float] = None,
                              win_rate: Optional[float] = None,
                              avg_win_loss_ratio: Optional[float] = None) -> PositionSizeResult:
        """
        Calculate position size based on configured method
        
        Args:
            account_value: Total account value
            entry_price: Entry price for position
            stop_loss_price: Stop loss price (if using stop-based sizing)
            volatility: Asset volatility (for volatility-adjusted sizing)
            win_rate: Historical win rate (for Kelly criterion)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly criterion)
            
        Returns:
            PositionSizeResult with calculated size
        """
        method = self.config['method']
        
        if method == SizingMethod.FIXED_DOLLAR:
            return self._fixed_dollar_sizing(account_value, entry_price)
        elif method == SizingMethod.PERCENT_RISK:
            return self._percent_risk_sizing(account_value, entry_price, stop_loss_price)
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_sizing(account_value, entry_price, volatility)
        elif method == SizingMethod.KELLY_CRITERION:
            return self._kelly_criterion_sizing(account_value, entry_price, 
                                              win_rate, avg_win_loss_ratio)
        else:
            logger.warning(f"Unknown sizing method: {method}, using percent risk")
            return self._percent_risk_sizing(account_value, entry_price, stop_loss_price)
    
    def _fixed_dollar_sizing(self, account_value: float, entry_price: float) -> PositionSizeResult:
        """
        Fixed dollar amount sizing
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            
        Returns:
            PositionSizeResult
        """
        fixed_amount = self.config.get('fixed_amount', 1000.0)
        max_position = account_value * self.config['max_position_size']
        
        size_usd = min(fixed_amount, max_position)
        size_usd = max(size_usd, self.config['min_position_size'])
        size_units = size_usd / entry_price
        
        return PositionSizeResult(
            size_usd=size_usd,
            size_units=size_units,
            method=SizingMethod.FIXED_DOLLAR,
            risk_amount=size_usd,  # Full position at risk without stop loss
            confidence=0.8,
            metadata={'fixed_amount': fixed_amount}
        )
    
    def _percent_risk_sizing(self, account_value: float, entry_price: float, 
                           stop_loss_price: Optional[float]) -> PositionSizeResult:
        """
        Percentage risk sizing (risk a fixed % of capital per trade)
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            PositionSizeResult
        """
        risk_amount = account_value * self.config['risk_per_trade']
        
        if stop_loss_price is None:
            # Without stop loss, use default risk assumption
            default_risk_pct = 0.05  # Assume 5% risk per trade
            size_usd = risk_amount / default_risk_pct
            confidence = 0.5  # Lower confidence without stop loss
        else:
            # Calculate position size based on stop loss distance
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                logger.warning("Invalid stop loss price, using default sizing")
                return self._fixed_dollar_sizing(account_value, entry_price)
            
            size_units = risk_amount / risk_per_unit
            size_usd = size_units * entry_price
            confidence = 0.9
        
        # Apply position size limits
        max_position = account_value * self.config['max_position_size']
        size_usd = min(size_usd, max_position)
        size_usd = max(size_usd, self.config['min_position_size'])
        size_units = size_usd / entry_price
        
        return PositionSizeResult(
            size_usd=size_usd,
            size_units=size_units,
            method=SizingMethod.PERCENT_RISK,
            risk_amount=risk_amount,
            confidence=confidence,
            metadata={
                'stop_loss_price': stop_loss_price,
                'risk_per_unit': abs(entry_price - stop_loss_price) if stop_loss_price else None
            }
        )
    
    def _volatility_adjusted_sizing(self, account_value: float, entry_price: float, 
                                  volatility: Optional[float]) -> PositionSizeResult:
        """
        Volatility-adjusted position sizing
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            volatility: Asset volatility (standard deviation of returns)
            
        Returns:
            PositionSizeResult
        """
        if volatility is None or volatility <= 0:
            logger.warning("Invalid volatility, using percent risk sizing")
            return self._percent_risk_sizing(account_value, entry_price, None)
        
        # Base risk amount
        base_risk = account_value * self.config['risk_per_trade']
        
        # Adjust for volatility (higher volatility = smaller position)
        # Target volatility for sizing (e.g., 2% daily volatility)
        target_volatility = 0.02
        volatility_multiplier = self.config['volatility_multiplier']
        
        # Scale position inversely with volatility
        volatility_adjustment = (target_volatility / volatility) * volatility_multiplier
        volatility_adjustment = np.clip(volatility_adjustment, 0.1, 2.0)  # Limit adjustment
        
        adjusted_risk = base_risk * volatility_adjustment
        
        # Estimate position size (assuming volatility represents potential loss)
        size_usd = adjusted_risk / volatility
        
        # Apply limits
        max_position = account_value * self.config['max_position_size']
        size_usd = min(size_usd, max_position)
        size_usd = max(size_usd, self.config['min_position_size'])
        size_units = size_usd / entry_price
        
        return PositionSizeResult(
            size_usd=size_usd,
            size_units=size_units,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            risk_amount=base_risk,
            confidence=0.7,
            metadata={
                'volatility': volatility,
                'target_volatility': target_volatility,
                'volatility_adjustment': volatility_adjustment
            }
        )
    
    def _kelly_criterion_sizing(self, account_value: float, entry_price: float,
                               win_rate: Optional[float], 
                               avg_win_loss_ratio: Optional[float]) -> PositionSizeResult:
        """
        Kelly Criterion position sizing
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            
        Returns:
            PositionSizeResult
        """
        if win_rate is None or avg_win_loss_ratio is None:
            logger.warning("Missing data for Kelly criterion, using percent risk sizing")
            return self._percent_risk_sizing(account_value, entry_price, None)
        
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            logger.warning("Invalid Kelly parameters, using percent risk sizing")
            return self._percent_risk_sizing(account_value, entry_price, None)
        
        # Kelly formula: f = (bp - q) / b
        # where:
        # f = fraction of capital to wager
        # b = odds received on the wager (avg_win_loss_ratio)
        # p = probability of winning (win_rate)
        # q = probability of losing (1 - win_rate)
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win_loss_ratio
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly fraction limits (never bet more than 25% even if Kelly suggests it)
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)
        
        # Conservative Kelly (use fraction of the Kelly recommendation)
        conservative_factor = self.config.get('kelly_conservative_factor', 0.5)
        kelly_fraction *= conservative_factor
        
        size_usd = account_value * kelly_fraction
        
        # Apply position size limits
        max_position = account_value * self.config['max_position_size']
        size_usd = min(size_usd, max_position)
        size_usd = max(size_usd, self.config['min_position_size'])
        size_units = size_usd / entry_price
        
        # Risk amount approximation
        risk_amount = size_usd * (1 - win_rate)  # Expected loss amount
        
        return PositionSizeResult(
            size_usd=size_usd,
            size_units=size_units,
            method=SizingMethod.KELLY_CRITERION,
            risk_amount=risk_amount,
            confidence=0.8,
            metadata={
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'avg_win_loss_ratio': avg_win_loss_ratio,
                'conservative_factor': conservative_factor
            }
        )
    
    def calculate_stop_loss_price(self, entry_price: float, 
                                 direction: str, 
                                 atr: Optional[float] = None,
                                 volatility: Optional[float] = None) -> float:
        """
        Calculate stop loss price based on volatility
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range
            volatility: Price volatility
            
        Returns:
            Stop loss price
        """
        # Default stop loss percentage
        default_stop_pct = 0.05  # 5%
        
        if atr is not None and atr > 0:
            # Use ATR-based stop loss
            atr_multiplier = self.config.get('atr_stop_multiplier', 2.0)
            stop_distance = atr * atr_multiplier
        elif volatility is not None and volatility > 0:
            # Use volatility-based stop loss
            vol_multiplier = self.config.get('volatility_stop_multiplier', 2.0)
            stop_distance = entry_price * volatility * vol_multiplier
        else:
            # Use percentage-based stop loss
            stop_distance = entry_price * default_stop_pct
        
        if direction.lower() == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        return max(stop_price, 0.01)  # Ensure positive price
    
    def calculate_take_profit_price(self, entry_price: float, 
                                  stop_loss_price: float,
                                  direction: str,
                                  risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: 'long' or 'short'
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            Take profit price
        """
        risk_distance = abs(entry_price - stop_loss_price)
        reward_distance = risk_distance * risk_reward_ratio
        
        if direction.lower() == 'long':
            take_profit_price = entry_price + reward_distance
        else:  # short
            take_profit_price = entry_price - reward_distance
        
        return max(take_profit_price, 0.01)  # Ensure positive price