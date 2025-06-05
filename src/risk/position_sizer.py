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
    LEVERAGE_AWARE = "leverage_aware"

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
            'method': SizingMethod.LEVERAGE_AWARE,  # Use leverage-aware sizing
            'risk_per_trade': 0.02,  # 2% of capital per trade (leverage amplifies)
            'max_position_size': 0.75,  # 75% max of capital per position with leverage
            'min_position_size': 100,  # Minimum $100 position
            'volatility_lookback': 20,
            'volatility_multiplier': 1.0,
            'kelly_lookback': 100,
            'max_leverage': 2.0,  # Smart leverage usage
            'crypto_max_position': 0.30,  # 30% max per crypto pair
            'target_utilization': 0.8,   # Use 80% of available buying power
        }
        
        self.config = {**self.default_config, **self.config}
        logger.info("PositionSizer initialized")
    
    def calculate_position_size(self, 
                              account_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float] = None,
                              volatility: Optional[float] = None,
                              win_rate: Optional[float] = None,
                              avg_win_loss_ratio: Optional[float] = None,
                              buying_power: Optional[float] = None,
                              is_crypto: bool = True,
                              is_pattern_day_trader: bool = False) -> PositionSizeResult:
        """
        Calculate position size based on configured method
        
        Args:
            account_value: Total account value
            entry_price: Entry price for position
            stop_loss_price: Stop loss price (if using stop-based sizing)
            volatility: Asset volatility (for volatility-adjusted sizing)
            win_rate: Historical win rate (for Kelly criterion)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly criterion)
            buying_power: Available buying power (includes margin)
            is_crypto: Whether this is a crypto trade (no margin available)
            is_pattern_day_trader: Whether account has PDT status (4x leverage)
            
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
        elif method == SizingMethod.LEVERAGE_AWARE:
            return self._leverage_aware_sizing(account_value, entry_price, stop_loss_price,
                                             buying_power, is_crypto, is_pattern_day_trader)
        else:
            logger.warning(f"Unknown sizing method: {method}, using leverage aware")
            return self._leverage_aware_sizing(account_value, entry_price, stop_loss_price,
                                             buying_power, is_crypto, is_pattern_day_trader)
    
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
    
    def _leverage_aware_sizing(self, account_value: float, entry_price: float,
                              stop_loss_price: Optional[float] = None,
                              buying_power: Optional[float] = None,
                              is_crypto: bool = True,
                              is_pattern_day_trader: bool = False) -> PositionSizeResult:
        """
        Leverage-aware position sizing that optimizes use of buying power
        
        Strategy:
        - Crypto: Use cash only (no margin), up to 50% of portfolio
        - Stocks: Use margin intelligently, up to 100% of portfolio value
        - PDT accounts: Access to 4x intraday buying power
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            buying_power: Available buying power
            is_crypto: Whether this is crypto (no margin)
            is_pattern_day_trader: PDT status for enhanced leverage
            
        Returns:
            PositionSizeResult with optimized size
        """
        # Use buying power for position sizing (includes leverage for crypto)
        available_capital = buying_power if buying_power else account_value
        
        logger.info(f"Position sizing inputs: account_value=${account_value:.2f}, buying_power=${buying_power:.2f}, available_capital=${available_capital:.2f}")
        
        # Calculate risk-based position size
        risk_per_trade = self.config['risk_per_trade']
        risk_amount = account_value * risk_per_trade
        
        # Calculate position size based on stop loss
        if stop_loss_price and entry_price != stop_loss_price:
            price_risk = abs(entry_price - stop_loss_price) / entry_price
            if price_risk > 0:
                # Size position so maximum loss equals risk_amount
                max_position_value = risk_amount / price_risk
            else:
                max_position_value = available_capital * 0.1  # Fallback to 10%
        else:
            # No stop loss defined, use default 4% price risk assumption
            default_price_risk = 0.04
            max_position_value = risk_amount / default_price_risk
        
        # Crypto position limits (can use leverage via buying power)
        max_crypto_position = account_value * self.config['crypto_max_position']
        
        # Use available buying power as the upper limit
        max_capital_position = available_capital * 0.80  # Use 80% of buying power, keep 20% buffer
        
        # Use the most restrictive limit
        position_value = min(max_position_value, max_crypto_position, max_capital_position)
        
        # Calculate effective leverage being used
        leverage_used = position_value / account_value if account_value > 0 else 1.0
        
        logger.info(f"Crypto position sizing: max_risk_based=${max_position_value:.0f}, "
                   f"max_crypto_limit=${max_crypto_position:.0f}, "
                   f"max_capital_available=${max_capital_position:.0f}")
        
        # Apply minimum position size
        min_position = self.config['min_position_size']
        position_value = max(position_value, min_position)
        
        # Calculate position size in units
        position_units = position_value / entry_price
        
        # Calculate actual risk amount
        if stop_loss_price:
            actual_risk = position_value * abs(entry_price - stop_loss_price) / entry_price
        else:
            actual_risk = position_value * 0.04  # Assume 4% risk
        
        # Calculate confidence based on position size utilization
        confidence = min(0.9, position_value / (account_value * 0.3))  # Higher confidence for reasonable crypto positions
        
        metadata = {
            'leverage_used': leverage_used,
            'is_crypto': True,  # All symbols are crypto now
            'available_capital': available_capital,
            'risk_percentage': (actual_risk / account_value) * 100,
            'position_percentage': (position_value / account_value) * 100
        }
        
        logger.info(f"Leverage-aware sizing: ${position_value:.0f} ({position_units:.6f} units), "
                   f"risk=${actual_risk:.0f} ({actual_risk/account_value*100:.1f}%), "
                   f"leverage={leverage_used:.2f}x, confidence={confidence:.2f}")
        
        return PositionSizeResult(
            size_usd=position_value,
            size_units=position_units,
            method=SizingMethod.LEVERAGE_AWARE,
            risk_amount=actual_risk,
            confidence=confidence,
            metadata=metadata
        )