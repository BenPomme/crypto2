"""
Moving Average Crossover Strategy
Implementation of the baseline MA crossover strategy as described in project.md
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Generates signals based on fast and slow moving average crossovers
    Enhanced with RSI and volume confirmation filters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 parameter_manager=None):
        """
        Initialize MA Crossover Strategy
        
        Args:
            config: Static strategy configuration
            parameter_manager: Dynamic parameter manager for ML optimization
        """
        # Static configuration (doesn't change)
        default_config = {
            'ma_type': 'sma',  # 'sma' or 'ema'
            'volume_confirmation': True,
            'min_periods': 50,
            'exit_on_reverse_cross': True,
            'stop_loss_pct': None,  # Optional stop loss percentage
            'take_profit_pct': None,  # Optional take profit percentage
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("MA_Crossover", default_config)
        
        # Use parameter manager for dynamic parameters
        self.parameter_manager = parameter_manager
        self.ma_type = self.config['ma_type']
        
        # Get initial dynamic parameters
        if self.parameter_manager:
            params = self.parameter_manager.get_current_parameters()
            self.fast_period = params.fast_ma_period
            self.slow_period = params.slow_ma_period
        else:
            # Fallback to static config
            self.fast_period = 12
            self.slow_period = 24
        
        logger.info(f"MA Crossover Strategy initialized: "
                   f"{self.fast_period}/{self.slow_period} {self.ma_type.upper()}")
    
    def get_required_indicators(self) -> List[str]:
        """Get required indicators for this strategy"""
        indicators = [
            f'{self.ma_type}_fast',
            f'{self.ma_type}_slow',
            'rsi'
        ]
        
        if self.config['volume_confirmation']:
            indicators.extend(['volume', 'volume_ma'])
        
        return indicators
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = None) -> Optional[TradingSignal]:
        """
        Generate trading signal based on MA crossover
        
        Args:
            data: DataFrame with market data and indicators
            symbol: Trading symbol for position tracking
            
        Returns:
            TradingSignal or None
        """
        if not self.can_generate_signal(data):
            return None
        
        try:
            # Get latest data point
            latest = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else latest
            
            # Get moving averages
            fast_ma_col = f'{self.ma_type}_fast'
            slow_ma_col = f'{self.ma_type}_slow'
            
            current_fast_ma = latest[fast_ma_col]
            current_slow_ma = latest[slow_ma_col]
            prev_fast_ma = previous[fast_ma_col]
            prev_slow_ma = previous[slow_ma_col]
            
            # Check for NaN values
            if pd.isna(current_fast_ma) or pd.isna(current_slow_ma):
                logger.debug("NaN values in moving averages, skipping signal generation")
                return None
            
            # Detect crossover
            golden_cross = (prev_fast_ma <= prev_slow_ma) and (current_fast_ma > current_slow_ma)
            death_cross = (prev_fast_ma >= prev_slow_ma) and (current_fast_ma < current_slow_ma)
            
            current_price = latest['close']
            timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
            
            # Use provided symbol or fallback to config
            trading_symbol = symbol or self.config.get('symbol', 'BTCUSD')
            
            # Check for buy signal (golden cross)
            if golden_cross and self.is_flat(trading_symbol):
                signal = self._evaluate_buy_signal(latest, data, trading_symbol)
                if signal:
                    self.log_signal(signal)
                    return signal
            
            # Check for sell signal (death cross) - only if we have a position to close
            elif death_cross and self.is_long(trading_symbol) and self.config['exit_on_reverse_cross']:
                signal = TradingSignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=trading_symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.7,
                    reason="Death cross detected",
                    metadata={
                        'fast_ma': current_fast_ma,
                        'slow_ma': current_slow_ma,
                        'crossover_type': 'death_cross'
                    }
                )
                self.log_signal(signal)
                return signal
            
            # Check for stop loss or take profit
            if not self.is_flat(trading_symbol):
                exit_signal = self._check_exit_conditions(latest, trading_symbol)
                if exit_signal:
                    self.log_signal(exit_signal)
                    return exit_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating MA crossover signal: {e}")
            return None
    
    def update_dynamic_parameters(self) -> None:
        """Update strategy with latest dynamic parameters from parameter manager"""
        if self.parameter_manager:
            params = self.parameter_manager.get_current_parameters()
            
            # Update periods if they changed
            if (params.fast_ma_period != self.fast_period or 
                params.slow_ma_period != self.slow_period):
                
                self.fast_period = params.fast_ma_period
                self.slow_period = params.slow_ma_period
                
                logger.info(f"Updated MA periods: {self.fast_period}/{self.slow_period}")
    
    def _evaluate_buy_signal(self, latest: pd.Series, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Evaluate buy signal with additional filters
        
        Args:
            latest: Latest data point
            data: Full data DataFrame
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        # Base signal confidence
        confidence = 0.6
        reasons = ["Golden cross detected"]
        
        # RSI confirmation filter
        if 'rsi' in latest:
            rsi = latest['rsi']
            if not pd.isna(rsi):
                if rsi < self.config['rsi_oversold']:
                    confidence += 0.2
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > self.config['rsi_overbought']:
                    confidence -= 0.3
                    reasons.append(f"RSI overbought ({rsi:.1f})")
                elif 30 <= rsi <= 70:
                    confidence += 0.1
                    reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # Volume confirmation filter
        if self.config['volume_confirmation']:
            if 'volume_ratio' in latest:
                volume_ratio = latest['volume_ratio']
                if not pd.isna(volume_ratio) and volume_ratio >= self.config['volume_threshold']:
                    confidence += 0.15
                    reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
                elif not pd.isna(volume_ratio) and volume_ratio < 0.5:
                    confidence -= 0.2
                    reasons.append(f"Low volume ({volume_ratio:.1f}x avg)")
        
        # Trend strength filter
        if 'trend_strength' in latest:
            trend_strength = latest['trend_strength']
            if not pd.isna(trend_strength):
                if trend_strength > 0.6:
                    confidence += 0.1
                    reasons.append(f"Strong uptrend ({trend_strength:.2f})")
                elif trend_strength < 0.4:
                    confidence -= 0.1
                    reasons.append(f"Weak trend ({trend_strength:.2f})")
        
        # Check if signal meets minimum confidence threshold
        min_confidence = self.config.get('min_confidence', 0.5)
        if confidence < min_confidence:
            logger.debug(f"Signal confidence {confidence:.2f} below threshold {min_confidence}")
            return None
        
        # Ensure confidence doesn't exceed 1.0
        confidence = min(confidence, 1.0)
        
        return TradingSignal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            timestamp=timestamp,
            price=current_price,
            confidence=confidence,
            reason="; ".join(reasons),
            metadata={
                'fast_ma': latest[f'{self.ma_type}_fast'],
                'slow_ma': latest[f'{self.ma_type}_slow'],
                'rsi': latest.get('rsi'),
                'volume_ratio': latest.get('volume_ratio'),
                'crossover_type': 'golden_cross'
            }
        )
    
    def _check_exit_conditions(self, latest: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """
        Check for stop loss or take profit conditions
        
        Args:
            latest: Latest data point
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        # Use symbol-specific position tracking
        entry_price = self.get_symbol_entry_price(symbol)
        if entry_price is None or self.is_flat(symbol):
            return None
        
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        # Calculate current P&L percentage for this symbol
        pnl_pct = self.get_unrealized_pnl_pct(current_price, symbol)
        if pnl_pct is None:
            return None
        
        # Check stop loss
        if self.config['stop_loss_pct'] and pnl_pct <= -abs(self.config['stop_loss_pct']):
            return TradingSignal(
                signal_type=SignalType.CLOSE_LONG if self.is_long(symbol) else SignalType.CLOSE_SHORT,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.9,
                reason=f"Stop loss triggered ({pnl_pct:.2f}%)",
                metadata={
                    'exit_type': 'stop_loss',
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price
                }
            )
        
        # Check take profit
        if self.config['take_profit_pct'] and pnl_pct >= self.config['take_profit_pct']:
            return TradingSignal(
                signal_type=SignalType.CLOSE_LONG if self.is_long(symbol) else SignalType.CLOSE_SHORT,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.8,
                reason=f"Take profit triggered ({pnl_pct:.2f}%)",
                metadata={
                    'exit_type': 'take_profit',
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price
                }
            )
        
        return None
    
    def should_exit_position(self, data: pd.DataFrame) -> bool:
        """
        Check if current position should be closed based on strategy logic
        
        Args:
            data: Current market data
            
        Returns:
            True if position should be closed
        """
        if self.is_flat() or data.empty:
            return False
        
        try:
            latest = data.iloc[-1]
            
            # Check technical exit conditions
            exit_signal = self._check_exit_conditions(latest)
            if exit_signal:
                return True
            
            # Check for reverse crossover
            if self.config['exit_on_reverse_cross'] and len(data) > 1:
                previous = data.iloc[-2]
                
                fast_ma_col = f'{self.ma_type}_fast'
                slow_ma_col = f'{self.ma_type}_slow'
                
                current_fast = latest[fast_ma_col]
                current_slow = latest[slow_ma_col]
                prev_fast = previous[fast_ma_col]
                prev_slow = previous[slow_ma_col]
                
                # Death cross while long
                death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)
                if death_cross and self.is_long():
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return (f"MA Crossover Strategy using {self.fast_period}/{self.slow_period} "
                f"{self.ma_type.upper()} with RSI and volume filters")