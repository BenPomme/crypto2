"""
Base Strategy Class
Abstract base class for all trading strategies
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    confidence: float = 0.5  # 0-1 scale
    reason: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata or {}
        }

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    Provides common interface and functionality
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.last_signal = None
        
        # Strategy state
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = None
        self.entry_timestamp = None
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        
        logger.info(f"Strategy {self.name} initialized")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal based on market data
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            TradingSignal or None if no signal
        """
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for this strategy
        
        Returns:
            List of required indicator names
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data contains required indicators
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning(f"Empty data provided to {self.name}")
            return False
        
        required_indicators = self.get_required_indicators()
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
        
        if missing_indicators:
            logger.error(f"Missing required indicators for {self.name}: {missing_indicators}")
            return False
        
        return True
    
    def can_generate_signal(self, data: pd.DataFrame) -> bool:
        """
        Check if strategy can generate a signal with current data
        
        Args:
            data: DataFrame with market data
            
        Returns:
            True if signal can be generated
        """
        return self.validate_data(data) and len(data) >= self.get_min_periods()
    
    def get_min_periods(self) -> int:
        """
        Get minimum number of periods required for strategy
        
        Returns:
            Minimum periods needed
        """
        return self.config.get('min_periods', 50)
    
    def update_position(self, signal: TradingSignal) -> None:
        """
        Update internal position tracking
        
        Args:
            signal: Trading signal that was executed
        """
        if signal.signal_type == SignalType.BUY:
            self.current_position = 1
            self.entry_price = signal.price
            self.entry_timestamp = signal.timestamp
        elif signal.signal_type == SignalType.SELL:
            self.current_position = -1
            self.entry_price = signal.price
            self.entry_timestamp = signal.timestamp
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            self.current_position = 0
            self.entry_price = None
            self.entry_timestamp = None
        
        self.trades_executed += 1
        logger.debug(f"Position updated: {self.current_position}")
    
    def get_current_position(self) -> int:
        """Get current position (-1, 0, 1)"""
        return self.current_position
    
    def is_long(self) -> bool:
        """Check if currently long"""
        return self.current_position == 1
    
    def is_short(self) -> bool:
        """Check if currently short"""
        return self.current_position == -1
    
    def is_flat(self) -> bool:
        """Check if no position"""
        return self.current_position == 0
    
    def get_unrealized_pnl(self, current_price: float) -> Optional[float]:
        """
        Calculate unrealized P&L
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L or None if no position
        """
        if self.entry_price is None or self.current_position == 0:
            return None
        
        if self.current_position == 1:  # Long position
            return current_price - self.entry_price
        else:  # Short position
            return self.entry_price - current_price
    
    def get_unrealized_pnl_pct(self, current_price: float) -> Optional[float]:
        """
        Calculate unrealized P&L percentage
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L percentage or None if no position
        """
        pnl = self.get_unrealized_pnl(current_price)
        if pnl is None or self.entry_price is None:
            return None
        
        return (pnl / self.entry_price) * 100
    
    def should_exit_position(self, data: pd.DataFrame) -> bool:
        """
        Check if current position should be closed
        Override in derived classes for custom exit logic
        
        Args:
            data: Current market data
            
        Returns:
            True if position should be closed
        """
        return False
    
    def log_signal(self, signal: TradingSignal) -> None:
        """
        Log generated signal
        
        Args:
            signal: Trading signal to log
        """
        self.last_signal = signal
        self.signals_generated += 1
        
        logger.info(f"Signal generated: {signal.signal_type.value} {signal.symbol} "
                   f"@ {signal.price:.2f} (confidence: {signal.confidence:.2f}) "
                   f"- {signal.reason}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics
        
        Returns:
            Dictionary with strategy stats
        """
        return {
            'name': self.name,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'last_signal': self.last_signal.to_dict() if self.last_signal else None
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.current_position = 0
        self.entry_price = None
        self.entry_timestamp = None
        self.last_signal = None
        self.signals_generated = 0
        self.trades_executed = 0
        
        logger.info(f"Strategy {self.name} reset")
    
    def __str__(self) -> str:
        return f"Strategy(name={self.name}, position={self.current_position})"
    
    def __repr__(self) -> str:
        return self.__str__()