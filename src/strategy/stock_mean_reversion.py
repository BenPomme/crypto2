"""
Stock Mean Reversion Strategy
Trades stocks based on mean reversion principles with support for short selling
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class StockMeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy for Stocks
    
    - Enters long when oversold (RSI < 30, price at lower BB)
    - Enters short when overbought (RSI > 70, price at upper BB)
    - Uses tight stops and profit targets
    - Volume confirmation for entry signals
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 parameter_manager=None):
        """
        Initialize Stock Mean Reversion Strategy
        
        Args:
            config: Strategy configuration
            parameter_manager: Parameter manager for optimization
        """
        # Default configuration
        default_config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_ma_period': 20,
            'min_volume': 1000000,      # Minimum daily volume ($1M)
            'volume_surge': 1.5,         # 1.5x average volume
            'enable_shorts': True,       # Enable short selling
            'profit_target': 0.02,       # 2% profit target
            'stop_loss': 0.01,          # 1% stop loss
            'trailing_stop': 0.015,      # 1.5% trailing stop
            'min_bb_width': 0.01,        # Minimum 1% BB width
            'max_bb_width': 0.10,        # Maximum 10% BB width
            'position_hold_bars': 20,    # Maximum bars to hold position
            'use_time_exits': True,      # Exit before market close
            'exit_minutes_before_close': 15,  # Exit 15 min before close
            'max_position_pct': 0.20,    # Max 20% per position
            'min_confidence': 0.6,       # Minimum confidence threshold
            'macd_confirmation': True,   # Use MACD for confirmation
            'enable_premarket': False,   # Trade in premarket
            'enable_afterhours': False,  # Trade in after hours
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("Stock_Mean_Reversion", default_config)
        
        self.parameter_manager = parameter_manager
        self.entry_prices = {}  # Track entry prices for profit targets
        self.entry_times = {}   # Track entry times for time-based exits
        self.highest_profits = {}  # Track highest profit for trailing stops
        
        logger.info(f"Stock Mean Reversion Strategy initialized with config: {self.config}")
    
    def get_required_indicators(self) -> List[str]:
        """Get required indicators for this strategy"""
        return [
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'rsi',
            'volume', 'volume_ma', 'volume_ratio',
            'macd', 'macd_signal', 'macd_histogram',
            'atr',  # For volatility-adjusted stops
            'typical_price',  # For better BB calculation
            'obv', 'obv_ma'  # Volume confirmation
        ]
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = None) -> Optional[TradingSignal]:
        """
        Generate trading signal based on mean reversion
        
        Args:
            data: DataFrame with market data and indicators
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        if not self.can_generate_signal(data):
            return None
        
        try:
            latest = data.iloc[-1]
            current_price = latest['close']
            timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
            
            # Skip if low volume
            if latest['volume'] * current_price < self.config['min_volume']:
                return None
            
            # Check for existing position
            if not self.is_flat(symbol):
                # Check exit conditions for existing position
                return self._check_exit_conditions(latest, data, symbol)
            
            # Calculate Bollinger Band metrics
            bb_position = self._calculate_bb_position(latest)
            bb_width_pct = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
            
            # Skip if BB too narrow or too wide (low volatility or extreme volatility)
            if bb_width_pct < self.config['min_bb_width'] or bb_width_pct > self.config['max_bb_width']:
                return None
            
            # Check for long entry (oversold)
            long_signal = self._evaluate_long_entry(latest, data, symbol, bb_position)
            if long_signal:
                return long_signal
            
            # Check for short entry (overbought)
            if self.config['enable_shorts']:
                short_signal = self._evaluate_short_entry(latest, data, symbol, bb_position)
                if short_signal:
                    return short_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    def _calculate_bb_position(self, latest: pd.Series) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        bb_range = latest['bb_upper'] - latest['bb_lower']
        if bb_range <= 0:
            return 0.5
        return (latest['close'] - latest['bb_lower']) / bb_range
    
    def _evaluate_long_entry(self, latest: pd.Series, data: pd.DataFrame, 
                           symbol: str, bb_position: float) -> Optional[TradingSignal]:
        """Evaluate long entry conditions"""
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        confidence = 0.0
        reasons = []
        
        # Primary condition: Price at lower band
        if bb_position < 0.15:  # Within 15% of lower band
            confidence += 0.3
            reasons.append(f"Price near lower BB ({bb_position:.1%})")
            
            # Extra boost if touching or below lower band
            if bb_position <= 0:
                confidence += 0.1
                reasons.append("Price below lower BB")
        else:
            return None  # Not oversold enough
        
        # RSI confirmation
        rsi = latest.get('rsi', 50)
        if rsi < self.config['rsi_oversold']:
            confidence += 0.2
            reasons.append(f"RSI oversold ({rsi:.1f})")
            
            # Extra boost for extreme oversold
            if rsi < 20:
                confidence += 0.1
                reasons.append("RSI extremely oversold")
        elif rsi < 40:
            confidence += 0.1
            reasons.append(f"RSI mildly oversold ({rsi:.1f})")
        
        # Volume confirmation
        volume_ratio = latest.get('volume_ratio', 1.0)
        if volume_ratio >= self.config['volume_surge']:
            confidence += 0.15
            reasons.append(f"Volume surge ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:
            confidence += 0.05
            reasons.append(f"Above average volume ({volume_ratio:.1f}x)")
        
        # OBV trend confirmation
        if 'obv' in latest and 'obv_ma' in latest:
            if latest['obv'] > latest['obv_ma']:
                confidence += 0.1
                reasons.append("OBV trending up")
        
        # MACD confirmation (optional)
        if self.config['macd_confirmation'] and 'macd' in latest:
            macd_hist = latest.get('macd_histogram', 0)
            if macd_hist > 0 or (len(data) > 1 and macd_hist > data.iloc[-2].get('macd_histogram', 0)):
                confidence += 0.1
                reasons.append("MACD improving")
        
        # Price action confirmation - look for reversal patterns
        if len(data) >= 3:
            # Bullish reversal: lower low but RSI higher low (divergence)
            if (data.iloc[-1]['low'] < data.iloc[-2]['low'] and 
                data.iloc[-1]['rsi'] > data.iloc[-2]['rsi']):
                confidence += 0.15
                reasons.append("Bullish RSI divergence")
        
        # Check minimum confidence
        if confidence < self.config['min_confidence']:
            return None
        
        # Store entry data
        self.entry_prices[symbol] = current_price
        self.entry_times[symbol] = timestamp
        self.highest_profits[symbol] = 0
        
        return TradingSignal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            timestamp=timestamp,
            price=current_price,
            confidence=min(confidence, 1.0),
            reason="; ".join(reasons),
            metadata={
                'strategy': 'mean_reversion',
                'bb_position': bb_position,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'entry_type': 'oversold_reversal'
            }
        )
    
    def _evaluate_short_entry(self, latest: pd.Series, data: pd.DataFrame, 
                            symbol: str, bb_position: float) -> Optional[TradingSignal]:
        """Evaluate short entry conditions"""
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        confidence = 0.0
        reasons = []
        
        # Primary condition: Price at upper band
        if bb_position > 0.85:  # Within 15% of upper band
            confidence += 0.3
            reasons.append(f"Price near upper BB ({bb_position:.1%})")
            
            # Extra boost if touching or above upper band
            if bb_position >= 1.0:
                confidence += 0.1
                reasons.append("Price above upper BB")
        else:
            return None  # Not overbought enough
        
        # RSI confirmation
        rsi = latest.get('rsi', 50)
        if rsi > self.config['rsi_overbought']:
            confidence += 0.2
            reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # Extra boost for extreme overbought
            if rsi > 80:
                confidence += 0.1
                reasons.append("RSI extremely overbought")
        elif rsi > 60:
            confidence += 0.1
            reasons.append(f"RSI mildly overbought ({rsi:.1f})")
        
        # Volume confirmation
        volume_ratio = latest.get('volume_ratio', 1.0)
        if volume_ratio >= self.config['volume_surge']:
            confidence += 0.15
            reasons.append(f"Volume surge ({volume_ratio:.1f}x)")
        elif volume_ratio >= 1.2:
            confidence += 0.05
            reasons.append(f"Above average volume ({volume_ratio:.1f}x)")
        
        # OBV trend confirmation (bearish)
        if 'obv' in latest and 'obv_ma' in latest:
            if latest['obv'] < latest['obv_ma']:
                confidence += 0.1
                reasons.append("OBV trending down")
        
        # MACD confirmation (optional)
        if self.config['macd_confirmation'] and 'macd' in latest:
            macd_hist = latest.get('macd_histogram', 0)
            if macd_hist < 0 or (len(data) > 1 and macd_hist < data.iloc[-2].get('macd_histogram', 0)):
                confidence += 0.1
                reasons.append("MACD weakening")
        
        # Price action confirmation - look for reversal patterns
        if len(data) >= 3:
            # Bearish reversal: higher high but RSI lower high (divergence)
            if (data.iloc[-1]['high'] > data.iloc[-2]['high'] and 
                data.iloc[-1]['rsi'] < data.iloc[-2]['rsi']):
                confidence += 0.15
                reasons.append("Bearish RSI divergence")
        
        # Check minimum confidence
        if confidence < self.config['min_confidence']:
            return None
        
        # Store entry data
        self.entry_prices[symbol] = current_price
        self.entry_times[symbol] = timestamp
        self.highest_profits[symbol] = 0
        
        # Create SELL signal (which will be interpreted as SELL_SHORT for opening position)
        return TradingSignal(
            signal_type=SignalType.SELL,  # This will open a short position
            symbol=symbol,
            timestamp=timestamp,
            price=current_price,
            confidence=min(confidence, 1.0),
            reason="; ".join(reasons),
            metadata={
                'strategy': 'mean_reversion',
                'bb_position': bb_position,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'entry_type': 'overbought_reversal',
                'is_short': True  # Flag to indicate short position
            }
        )
    
    def _check_exit_conditions(self, latest: pd.Series, data: pd.DataFrame, 
                             symbol: str) -> Optional[TradingSignal]:
        """Check exit conditions for existing positions"""
        if symbol not in self.entry_prices:
            return None
        
        current_price = latest['close']
        entry_price = self.entry_prices[symbol]
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        # Calculate P&L
        if self.is_long(symbol):
            pnl_pct = (current_price - entry_price) / entry_price
            exit_signal_type = SignalType.CLOSE_LONG
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
            exit_signal_type = SignalType.CLOSE_SHORT
        
        # Update highest profit for trailing stop
        if symbol in self.highest_profits:
            self.highest_profits[symbol] = max(self.highest_profits[symbol], pnl_pct)
        
        # 1. Fixed profit target
        if pnl_pct >= self.config['profit_target']:
            return TradingSignal(
                signal_type=exit_signal_type,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.9,
                reason=f"Profit target reached ({pnl_pct:.1%})",
                metadata={
                    'exit_type': 'profit_target',
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price
                }
            )
        
        # 2. Stop loss
        if pnl_pct <= -self.config['stop_loss']:
            return TradingSignal(
                signal_type=exit_signal_type,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.95,
                reason=f"Stop loss triggered ({pnl_pct:.1%})",
                metadata={
                    'exit_type': 'stop_loss',
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price
                }
            )
        
        # 3. Trailing stop
        if self.highest_profits[symbol] > 0.005:  # Only trail after 0.5% profit
            drawdown = self.highest_profits[symbol] - pnl_pct
            if drawdown >= self.config['trailing_stop']:
                return TradingSignal(
                    signal_type=exit_signal_type,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.85,
                    reason=f"Trailing stop triggered (peak: {self.highest_profits[symbol]:.1%}, current: {pnl_pct:.1%})",
                    metadata={
                        'exit_type': 'trailing_stop',
                        'pnl_pct': pnl_pct,
                        'peak_pnl': self.highest_profits[symbol],
                        'entry_price': entry_price
                    }
                )
        
        # 4. Mean reversion completion
        bb_position = self._calculate_bb_position(latest)
        
        # Exit long if price reaches middle or upper band
        if self.is_long(symbol):
            if bb_position >= 0.5:  # At or above middle band
                return TradingSignal(
                    signal_type=exit_signal_type,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.7,
                    reason=f"Mean reversion target reached (BB position: {bb_position:.1%})",
                    metadata={
                        'exit_type': 'mean_reversion',
                        'bb_position': bb_position,
                        'pnl_pct': pnl_pct,
                        'entry_price': entry_price
                    }
                )
        
        # Exit short if price reaches middle or lower band
        else:  # Short position
            if bb_position <= 0.5:  # At or below middle band
                return TradingSignal(
                    signal_type=exit_signal_type,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.7,
                    reason=f"Mean reversion target reached (BB position: {bb_position:.1%})",
                    metadata={
                        'exit_type': 'mean_reversion',
                        'bb_position': bb_position,
                        'pnl_pct': pnl_pct,
                        'entry_price': entry_price
                    }
                )
        
        # 5. Time-based exit
        if self.config['position_hold_bars'] and symbol in self.entry_times:
            bars_held = len(data.loc[self.entry_times[symbol]:])
            if bars_held >= self.config['position_hold_bars']:
                return TradingSignal(
                    signal_type=exit_signal_type,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.6,
                    reason=f"Maximum holding period reached ({bars_held} bars)",
                    metadata={
                        'exit_type': 'time_exit',
                        'bars_held': bars_held,
                        'pnl_pct': pnl_pct,
                        'entry_price': entry_price
                    }
                )
        
        # 6. Exit before market close
        if self.config['use_time_exits']:
            # This would need market hours info from data provider
            # Placeholder for now
            pass
        
        return None
    
    def update_position(self, signal: TradingSignal) -> None:
        """Update position tracking after trade execution"""
        super().update_position(signal)
        
        # Clean up tracking data on position close
        if signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            symbol = signal.symbol
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
            if symbol in self.entry_times:
                del self.entry_times[symbol]
            if symbol in self.highest_profits:
                del self.highest_profits[symbol]
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific metrics"""
        base_metrics = super().get_strategy_metrics()
        
        # Add mean reversion specific metrics
        mean_rev_metrics = {
            'active_positions': len(self.entry_prices),
            'average_holding_period': 0,  # Would calculate from historical trades
            'profit_target_hits': 0,      # Would track in real implementation
            'stop_loss_hits': 0,          # Would track in real implementation
            'mean_reversion_exits': 0,    # Would track in real implementation
        }
        
        return {**base_metrics, **mean_rev_metrics}