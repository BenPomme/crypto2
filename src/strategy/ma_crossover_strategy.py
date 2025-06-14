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
            'rsi_oversold': 30.0,
            'rsi_overbought': 70.0,
            'min_confidence': 0.2,  # Reasonable threshold with working volume indicators
            'volume_threshold': 0.8,  # Lower volume requirement
            'enable_trend_following': True,
            'enable_momentum_entry': True,  # New: Enter on momentum signals
            'enable_breakout_entry': True,  # New: Enter on Bollinger Band breakouts
            'leverage': 3.0,  # 3x leverage for crypto
            'leverage_take_profit_pct': 1.5,  # Take profit at 1.5% with 3x leverage = 4.5% effective
            'leverage_stop_loss_pct': 1.0,  # Stop loss at 1% with 3x leverage = 3% loss
            'quick_profit_pct': 0.8,  # Quick scalp profits at 0.8% (2.4% effective with 3x)
        }
        
        if config:
            default_config.update(config)
        
        # Set slow MA period for base class minimum periods calculation
        if 'slow_ma_period' not in default_config:
            default_config['slow_ma_period'] = 24  # Default slow MA period
        
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
            indicators.extend(['volume', 'volume_ma', 'volume_ratio', 'obv', 'obv_trend', 'mfi'])
        
        # Add Bollinger Bands for volatility analysis
        indicators.extend(['bb_upper', 'bb_lower', 'bb_middle'])
        
        # Add MACD for trend confirmation
        indicators.extend(['macd', 'macd_signal', 'macd_histogram', 'macd_bullish', 'macd_momentum'])
        
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
            
            # Debug logging for signal analysis (every 10 cycles to avoid spam)
            if hasattr(self, '_debug_cycle_counter'):
                self._debug_cycle_counter += 1
            else:
                self._debug_cycle_counter = 1
                
            if self._debug_cycle_counter % 10 == 0:
                logger.info(f"🔍 {trading_symbol} Signal Analysis:")
                logger.info(f"  Fast MA: {current_fast_ma:.4f} (prev: {prev_fast_ma:.4f})")
                logger.info(f"  Slow MA: {current_slow_ma:.4f} (prev: {prev_slow_ma:.4f})")
                logger.info(f"  Golden Cross: {golden_cross}, Death Cross: {death_cross}")
                logger.info(f"  Position: {self.is_flat(trading_symbol)} (flat), {self.is_long(trading_symbol)} (long)")
            
            # Check for buy signal (golden cross)
            if golden_cross and self.is_flat(trading_symbol):
                signal = self._evaluate_buy_signal(latest, data, trading_symbol)
                if signal:
                    self.log_signal(signal)
                    return signal
            
            # Check for trend continuation signal (fast MA well above slow MA)
            elif (current_fast_ma > current_slow_ma and 
                  self.is_flat(trading_symbol) and 
                  self.config.get('enable_trend_following', True)):
                
                # Much more aggressive trend following
                ma_spread_pct = ((current_fast_ma - current_slow_ma) / current_slow_ma) * 100
                prev_ma_spread_pct = ((prev_fast_ma - prev_slow_ma) / prev_slow_ma) * 100
                
                # AGGRESSIVE: Enter on ANY meaningful uptrend
                if (ma_spread_pct > 0.1 or  # Even small spread (0.1% vs 0.5%)
                    (ma_spread_pct > prev_ma_spread_pct and ma_spread_pct > 0.05) or  # Accelerating
                    current_price > current_fast_ma):  # Price momentum
                    
                    signal = self._evaluate_trend_continuation_signal(latest, data, trading_symbol)
                    if signal:
                        self.log_signal(signal)
                        return signal
            
            # NEW: Momentum entry for strong trends (even without acceleration)
            elif (current_fast_ma > current_slow_ma and 
                  self.is_flat(trading_symbol) and 
                  self.config.get('enable_momentum_entry', True)):
                
                signal = self._evaluate_momentum_entry_signal(latest, data, trading_symbol)
                if signal:
                    self.log_signal(signal)
                    return signal
            
            # NEW: Bollinger Band breakout entry
            elif (self.is_flat(trading_symbol) and 
                  self.config.get('enable_breakout_entry', True)):
                
                signal = self._evaluate_breakout_signal(latest, data, trading_symbol)
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
    
    def update_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """
        Update strategy with optimized parameters from optimization results
        
        Args:
            optimized_params: Dictionary of optimized parameters
        """
        # Update MA periods if provided
        if 'fast_ma_period' in optimized_params:
            old_fast = self.fast_period
            self.fast_period = int(optimized_params['fast_ma_period'])
            if old_fast != self.fast_period:
                logger.info(f"Updated fast MA period: {old_fast} -> {self.fast_period}")
        
        if 'slow_ma_period' in optimized_params:
            old_slow = self.slow_period
            self.slow_period = int(optimized_params['slow_ma_period'])
            if old_slow != self.slow_period:
                logger.info(f"Updated slow MA period: {old_slow} -> {self.slow_period}")
        
        # Update other strategy parameters
        if 'confidence_threshold' in optimized_params:
            self.config['min_confidence'] = float(optimized_params['confidence_threshold'])
            
        if 'risk_per_trade' in optimized_params:
            self.config['risk_per_trade'] = float(optimized_params['risk_per_trade'])
        
        if 'volume_threshold' in optimized_params:
            self.config['volume_threshold'] = float(optimized_params['volume_threshold'])
        
        logger.info(f"Applied optimized parameters: fast={self.fast_period}, slow={self.slow_period}")

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
        
        # Enhanced volume confirmation filter with OBV and MFI
        if self.config['volume_confirmation']:
            volume_score = 0
            volume_reasons = []
            
            # Volume ratio confirmation
            if 'volume_ratio' in latest:
                volume_ratio = latest['volume_ratio']
                if not pd.isna(volume_ratio):
                    if volume_ratio >= self.config.get('volume_threshold', 1.2):
                        volume_score += 0.15
                        volume_reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
                    elif volume_ratio < 0.5:
                        volume_score -= 0.2
                        volume_reasons.append(f"Low volume ({volume_ratio:.1f}x avg)")
            
            # OBV trend confirmation
            if 'obv_trend' in latest and len(data) > 1:
                obv_trend = latest['obv_trend']
                if not pd.isna(obv_trend) and obv_trend == 1:  # OBV trending up
                    volume_score += 0.1
                    volume_reasons.append("OBV uptrend")
                elif not pd.isna(obv_trend) and obv_trend == 0:  # OBV trending down
                    volume_score -= 0.1
                    volume_reasons.append("OBV downtrend")
            
            # MFI confirmation (Money Flow Index)
            if 'mfi' in latest:
                mfi = latest['mfi']
                if not pd.isna(mfi):
                    if mfi > 50 and mfi < 80:  # Positive money flow, not overbought
                        volume_score += 0.1
                        volume_reasons.append(f"Positive money flow (MFI={mfi:.1f})")
                    elif mfi < 20:  # Oversold on money flow
                        volume_score += 0.15
                        volume_reasons.append(f"MFI oversold ({mfi:.1f})")
                    elif mfi > 80:  # Overbought on money flow
                        volume_score -= 0.15
                        volume_reasons.append(f"MFI overbought ({mfi:.1f})")
            
            confidence += volume_score
            if volume_reasons:
                reasons.extend(volume_reasons)
        
        # Bollinger Bands volatility filter
        if all(col in latest for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            bb_middle = latest['bb_middle']
            
            if not any(pd.isna([bb_upper, bb_lower, bb_middle])):
                # Calculate position within Bollinger Bands
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_width_pct = ((bb_upper - bb_lower) / bb_middle) * 100
                
                # Bollinger Band squeeze detection (low volatility -> potential breakout)
                if bb_width_pct < 4:  # Tight bands indicate low volatility
                    confidence += 0.05
                    reasons.append(f"BB squeeze ({bb_width_pct:.1f}% width)")
                
                # Support from lower band
                if bb_position < 0.2:  # Near lower band
                    confidence += 0.15
                    reasons.append(f"Near BB lower band (oversold)")
                elif bb_position > 0.8:  # Near upper band
                    confidence -= 0.1
                    reasons.append(f"Near BB upper band")
                elif 0.4 <= bb_position <= 0.6:  # Near middle
                    confidence += 0.05
                    reasons.append(f"BB middle support")
        
        # MACD trend confirmation
        if all(col in latest for col in ['macd', 'macd_signal', 'macd_histogram', 'macd_bullish']):
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            macd_histogram = latest['macd_histogram']
            macd_bullish = latest['macd_bullish']
            
            if not any(pd.isna([macd, macd_signal, macd_histogram])):
                # MACD bullish crossover (MACD above signal line)
                if macd_bullish == 1:
                    confidence += 0.15
                    reasons.append("MACD bullish")
                    
                    # Extra boost if MACD is accelerating upward
                    if 'macd_momentum' in latest and not pd.isna(latest['macd_momentum']):
                        if latest['macd_momentum'] > 0:
                            confidence += 0.05
                            reasons.append("MACD accelerating")
                else:
                    confidence -= 0.1
                    reasons.append("MACD bearish")
                
                # MACD zero line confirmation
                if macd > 0:
                    confidence += 0.05
                    reasons.append("MACD above zero")
                
                # MACD histogram growing (momentum building)
                if macd_histogram > 0:
                    confidence += 0.05
                    reasons.append("MACD histogram positive")
        
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
    
    def _evaluate_trend_continuation_signal(self, latest: pd.Series, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Evaluate trend continuation signal for ongoing uptrends
        
        Args:
            latest: Latest data point
            data: Full data DataFrame
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        # AGGRESSIVE: Start with higher confidence for trend continuation
        confidence = 0.35  # Easier to reach threshold
        reasons = ["Trend continuation signal"]
        
        # Calculate MA spread strength
        fast_ma = latest[f'{self.ma_type}_fast']
        slow_ma = latest[f'{self.ma_type}_slow']
        ma_spread_pct = ((fast_ma - slow_ma) / slow_ma) * 100
        
        if ma_spread_pct > 1.0:  # Strong trend
            confidence += 0.1
            reasons.append(f"Strong MA spread ({ma_spread_pct:.1f}%)")
        
        # Price momentum check
        if current_price > fast_ma:
            price_above_fast_pct = ((current_price - fast_ma) / fast_ma) * 100
            if price_above_fast_pct > 0.5:
                confidence += 0.1
                reasons.append(f"Price above fast MA ({price_above_fast_pct:.1f}%)")
        
        # RSI confirmation (less strict for trend continuation)
        if 'rsi' in latest:
            rsi = latest['rsi']
            if not pd.isna(rsi):
                if rsi < self.config['rsi_oversold']:
                    confidence += 0.2
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > self.config['rsi_overbought']:
                    confidence -= 0.1  # Less penalty for trend continuation
                    reasons.append(f"RSI overbought ({rsi:.1f})")
                elif 40 <= rsi <= 70:
                    confidence += 0.1
                    reasons.append(f"RSI healthy ({rsi:.1f})")
        
        # Enhanced volume confirmation for trend continuation
        if self.config['volume_confirmation']:
            volume_score = 0
            volume_reasons = []
            
            # Volume ratio confirmation (less strict for trend continuation)
            if 'volume_ratio' in latest:
                volume_ratio = latest['volume_ratio']
                if not pd.isna(volume_ratio):
                    if volume_ratio >= self.config.get('volume_threshold', 1.2):
                        volume_score += 0.1  # Slightly less boost for continuation
                        volume_reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
                    elif volume_ratio < 0.3:  # Only penalize very low volume
                        volume_score -= 0.05
                        volume_reasons.append(f"Very low volume ({volume_ratio:.1f}x avg)")
            
            # OBV trend confirmation (important for trend continuation)
            if 'obv_trend' in latest and len(data) > 1:
                obv_trend = latest['obv_trend']
                if not pd.isna(obv_trend) and obv_trend == 1:  # OBV trending up
                    volume_score += 0.15  # Higher weight for trend continuation
                    volume_reasons.append("OBV supports trend")
                elif not pd.isna(obv_trend) and obv_trend == 0:  # OBV trending down
                    volume_score -= 0.2  # Significant penalty if volume doesn't support trend
                    volume_reasons.append("OBV divergence")
            
            # MFI confirmation (more lenient for trend continuation)
            if 'mfi' in latest:
                mfi = latest['mfi']
                if not pd.isna(mfi):
                    if mfi > 40 and mfi < 85:  # Broader range for trend continuation
                        volume_score += 0.05
                        volume_reasons.append(f"MFI supportive ({mfi:.1f})")
                    elif mfi > 85:  # Very overbought
                        volume_score -= 0.1
                        volume_reasons.append(f"MFI extremely overbought ({mfi:.1f})")
            
            confidence += volume_score
            if volume_reasons:
                reasons.extend(volume_reasons)
        
        # MACD trend confirmation for continuation (more lenient)
        if all(col in latest for col in ['macd', 'macd_signal', 'macd_bullish']):
            macd = latest['macd']
            macd_bullish = latest['macd_bullish']
            
            if not pd.isna(macd) and not pd.isna(macd_bullish):
                # MACD still bullish supports trend continuation
                if macd_bullish == 1:
                    confidence += 0.1
                    reasons.append("MACD supports trend")
                else:
                    confidence -= 0.15  # Stronger penalty for trend continuation
                    reasons.append("MACD bearish (trend weakening)")
                
                # MACD above zero line still supportive
                if macd > 0:
                    confidence += 0.05
                    reasons.append("MACD above zero")
        
        # Check if signal meets minimum confidence threshold
        min_confidence = self.config.get('min_confidence', 0.5)
        if confidence < min_confidence:
            logger.debug(f"Trend continuation confidence {confidence:.2f} below threshold {min_confidence}")
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
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'ma_spread_pct': ma_spread_pct,
                'rsi': latest.get('rsi'),
                'volume_ratio': latest.get('volume_ratio'),
                'signal_type': 'trend_continuation'
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
        
        # LEVERAGE-AWARE EXIT LOGIC
        leverage = self.config.get('leverage', 1.0)
        
        # Leverage stop loss (tighter for leveraged positions)
        leverage_stop_loss = self.config.get('leverage_stop_loss_pct')
        if leverage_stop_loss and pnl_pct <= -abs(leverage_stop_loss):
            return TradingSignal(
                signal_type=SignalType.CLOSE_LONG if self.is_long(symbol) else SignalType.CLOSE_SHORT,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.95,
                reason=f"Leverage stop loss triggered ({pnl_pct:.2f}%, {leverage}x effective: {pnl_pct*leverage:.2f}%)",
                metadata={
                    'exit_type': 'leverage_stop_loss',
                    'pnl_pct': pnl_pct,
                    'effective_pnl': pnl_pct * leverage,
                    'leverage': leverage,
                    'entry_price': entry_price
                }
            )
        
        # Quick scalp profits (very tight for frequent wins)
        quick_profit = self.config.get('quick_profit_pct')
        if quick_profit and pnl_pct >= quick_profit:
            return TradingSignal(
                signal_type=SignalType.CLOSE_LONG if self.is_long(symbol) else SignalType.CLOSE_SHORT,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.85,
                reason=f"Quick scalp profit ({pnl_pct:.2f}%, {leverage}x effective: {pnl_pct*leverage:.2f}%)",
                metadata={
                    'exit_type': 'quick_scalp',
                    'pnl_pct': pnl_pct,
                    'effective_pnl': pnl_pct * leverage,
                    'leverage': leverage,
                    'entry_price': entry_price
                }
            )
        
        # Main leverage take profit
        leverage_take_profit = self.config.get('leverage_take_profit_pct')
        if leverage_take_profit and pnl_pct >= leverage_take_profit:
            return TradingSignal(
                signal_type=SignalType.CLOSE_LONG if self.is_long(symbol) else SignalType.CLOSE_SHORT,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=0.9,
                reason=f"Leverage take profit ({pnl_pct:.2f}%, {leverage}x effective: {pnl_pct*leverage:.2f}%)",
                metadata={
                    'exit_type': 'leverage_take_profit',
                    'pnl_pct': pnl_pct,
                    'effective_pnl': pnl_pct * leverage,
                    'leverage': leverage,
                    'entry_price': entry_price
                }
            )
        
        # Fallback to original stop/take profit if configured
        if self.config.get('stop_loss_pct') and pnl_pct <= -abs(self.config['stop_loss_pct']):
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
        
        if self.config.get('take_profit_pct') and pnl_pct >= self.config['take_profit_pct']:
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
        
        # Bollinger Bands exit conditions
        if all(col in latest for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            bb_middle = latest['bb_middle']
            
            if not any(pd.isna([bb_upper, bb_lower, bb_middle])):
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                
                # LEVERAGE-AWARE BB exits (more aggressive for leveraged positions)
                leverage = self.config.get('leverage', 1.0)
                
                # Exit at lower BB threshold for leveraged positions
                bb_exit_threshold = 0.95 if leverage <= 1.0 else 0.85  # Earlier exit with leverage
                
                if self.is_long(symbol) and bb_position > bb_exit_threshold:
                    return TradingSignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        confidence=0.8,
                        reason=f"Leverage BB exit ({bb_position:.2f} position, {leverage}x leverage)",
                        metadata={
                            'exit_type': 'leverage_bb_exit',
                            'bb_position': bb_position,
                            'leverage': leverage,
                            'pnl_pct': pnl_pct,
                            'entry_price': entry_price
                        }
                    )
                
                # Exit if price breaks below middle BB after being above (trend weakening)
                if (self.is_long(symbol) and current_price < bb_middle and 
                    pnl_pct < -2):  # Only if losing money
                    return TradingSignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        confidence=0.6,
                        reason=f"BB middle break with loss ({pnl_pct:.2f}%)",
                        metadata={
                            'exit_type': 'bb_middle_break',
                            'bb_position': bb_position,
                            'pnl_pct': pnl_pct,
                            'entry_price': entry_price
                        }
                    )
        
        # MACD exit conditions
        if all(col in latest for col in ['macd_bullish', 'macd', 'macd_signal']):
            if len(data) > 1:
                current_macd_bullish = latest['macd_bullish']
                previous_macd_bullish = data.iloc[-2]['macd_bullish'] if 'macd_bullish' in data.iloc[-2] else current_macd_bullish
                
                # MACD bearish crossover (trend change)
                if (self.is_long(symbol) and 
                    previous_macd_bullish == 1 and current_macd_bullish == 0):
                    return TradingSignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        confidence=0.8,
                        reason="MACD bearish crossover",
                        metadata={
                            'exit_type': 'macd_bearish_cross',
                            'macd': latest['macd'],
                            'macd_signal': latest['macd_signal'],
                            'pnl_pct': pnl_pct,
                            'entry_price': entry_price
                        }
                    )
        
        # LEVERAGE TRAILING STOP (dynamic exit for maximizing leveraged gains)
        leverage = self.config.get('leverage', 1.0)
        if leverage > 1.0 and self.is_long(symbol) and pnl_pct > 0.5:  # Only if profitable
            # Calculate trailing stop based on recent high
            if hasattr(self, f'_max_pnl_{symbol}'):
                max_pnl = getattr(self, f'_max_pnl_{symbol}')
                max_pnl = max(max_pnl, pnl_pct)
                setattr(self, f'_max_pnl_{symbol}', max_pnl)
            else:
                max_pnl = pnl_pct
                setattr(self, f'_max_pnl_{symbol}', max_pnl)
            
            # Trailing stop at 0.3% below peak for leveraged positions
            trailing_distance = 0.3
            if pnl_pct <= (max_pnl - trailing_distance):
                return TradingSignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.9,
                    reason=f"Leverage trailing stop (peak: {max_pnl:.2f}%, current: {pnl_pct:.2f}%)",
                    metadata={
                        'exit_type': 'leverage_trailing_stop',
                        'max_pnl': max_pnl,
                        'pnl_pct': pnl_pct,
                        'leverage': leverage,
                        'effective_max': max_pnl * leverage,
                        'entry_price': entry_price
                    }
                )
        
        return None
    
    def _evaluate_momentum_entry_signal(self, latest: pd.Series, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Evaluate momentum entry for existing uptrends
        """
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        fast_ma = latest[f'{self.ma_type}_fast']
        slow_ma = latest[f'{self.ma_type}_slow']
        
        # Only enter if in clear uptrend
        if fast_ma <= slow_ma:
            return None
            
        # Base confidence for momentum entry
        confidence = 0.3  # Lower starting point
        reasons = ["Momentum entry on uptrend"]
        
        # MA spread strength
        ma_spread_pct = ((fast_ma - slow_ma) / slow_ma) * 100
        if ma_spread_pct > 0.2:
            confidence += 0.1
            reasons.append(f"Strong MA spread ({ma_spread_pct:.2f}%)")
        
        # Price above fast MA
        if current_price > fast_ma:
            price_momentum = ((current_price - fast_ma) / fast_ma) * 100
            if price_momentum > 0.1:
                confidence += 0.1
                reasons.append(f"Price momentum ({price_momentum:.2f}%)")
        
        # RSI not overbought
        if 'rsi' in latest:
            rsi = latest['rsi']
            if not pd.isna(rsi) and rsi < 75:  # More lenient
                confidence += 0.1
                reasons.append(f"RSI healthy ({rsi:.1f})")
        
        # MACD bullish
        if 'macd_bullish' in latest and latest['macd_bullish'] == 1:
            confidence += 0.1
            reasons.append("MACD bullish")
        
        if confidence >= self.config.get('min_confidence', 0.3):
            return TradingSignal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                confidence=min(confidence, 1.0),
                reason="; ".join(reasons),
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'ma_spread_pct': ma_spread_pct,
                    'signal_type': 'momentum_entry'
                }
            )
        return None
    
    def _evaluate_breakout_signal(self, latest: pd.Series, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Evaluate Bollinger Band breakout signals
        """
        if not all(col in latest for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            return None
            
        current_price = latest['close']
        timestamp = latest.name if hasattr(latest, 'name') else datetime.now()
        
        bb_upper = latest['bb_upper']
        bb_lower = latest['bb_lower']
        bb_middle = latest['bb_middle']
        
        if any(pd.isna([bb_upper, bb_lower, bb_middle])):
            return None
        
        # Calculate BB position
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Breakout above upper band or strong move above middle
        if bb_position > 0.8:  # Near or above upper band
            confidence = 0.4
            reasons = ["BB breakout signal"]
            
            # Volume confirmation
            if 'volume_ratio' in latest and not pd.isna(latest['volume_ratio']):
                if latest['volume_ratio'] > 1.0:
                    confidence += 0.1
                    reasons.append("Volume support")
            
            # RSI confirmation
            if 'rsi' in latest:
                rsi = latest['rsi']
                if not pd.isna(rsi) and rsi > 50 and rsi < 80:
                    confidence += 0.1
                    reasons.append(f"RSI momentum ({rsi:.1f})")
            
            if confidence >= self.config.get('min_confidence', 0.3):
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=min(confidence, 1.0),
                    reason="; ".join(reasons),
                    metadata={
                        'bb_position': bb_position,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'signal_type': 'breakout_entry'
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