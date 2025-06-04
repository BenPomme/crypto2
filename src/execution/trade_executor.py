"""
Trade Executor Module
High-level interface for executing trading signals
Integrates order management with risk management and monitoring
"""
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from ..strategy.base_strategy import TradingSignal, SignalType
from ..risk.risk_manager import RiskManager, RiskCheckResult
from .order_manager import OrderManager, OrderResult, OrderSide, OrderType, OrderStatus
from ..utils.fee_calculator import FeeCalculator

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    High-level trade execution engine
    Coordinates signal execution with risk management and order management
    """
    
    def __init__(self, risk_manager: RiskManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trade executor
        
        Args:
            risk_manager: Risk manager instance
            config: Executor configuration
        """
        self.risk_manager = risk_manager
        self.order_manager = OrderManager()
        self.fee_calculator = FeeCalculator()
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'enable_stop_loss': True,
            'enable_take_profit': True,
            'default_stop_loss_pct': 0.05,  # 5%
            'default_take_profit_pct': 0.10,  # 10%
            'use_limit_orders': False,
            'limit_order_offset_pct': 0.001,  # 0.1% limit order offset
            'order_timeout': 300,  # 5 minutes
            'max_slippage_pct': 0.005,  # 0.5% max slippage
            'enable_partial_fills': True,
        }
        
        self.config = {**self.default_config, **self.config}
        
        # Execution tracking
        self.execution_history = []
        self.failed_executions = []
        
        logger.info("TradeExecutor initialized")
    
    def execute_signal(self, 
                      signal: TradingSignal,
                      account_info: Dict[str, Any],
                      current_positions: List[Dict[str, Any]],
                      market_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal to execute
            account_info: Current account information
            current_positions: List of current positions
            market_data: Current market data
            
        Returns:
            Execution result dictionary
        """
        execution_start = datetime.now()
        
        logger.info(f"Executing signal: {signal.signal_type.value} {signal.symbol} @ {signal.price}")
        
        try:
            # Step 1: Risk evaluation
            risk_result = self.risk_manager.evaluate_signal(
                signal=signal,
                account_info=account_info,
                current_positions=current_positions,
                market_data=market_data
            )
            
            if not risk_result.approved:
                result = {
                    'success': False,
                    'reason': 'Risk check failed',
                    'signal': signal.to_dict(),
                    'risk_result': risk_result.to_dict(),
                    'execution_time': (datetime.now() - execution_start).total_seconds()
                }
                
                self.failed_executions.append(result)
                logger.warning(f"Signal rejected by risk manager: {risk_result.recommendation}")
                return result
            
            # Step 2: Execute the trade
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                execution_result = self._execute_entry_signal(signal, risk_result)
            elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                execution_result = self._execute_exit_signal(signal, current_positions)
            else:
                execution_result = {
                    'success': False,
                    'reason': f'Unsupported signal type: {signal.signal_type.value}'
                }
            
            # Step 3: Process execution result
            if execution_result['success']:
                # Update risk manager with trade result (if applicable)
                if 'order_result' in execution_result:
                    order_result = execution_result['order_result']
                    if order_result.status == OrderStatus.FILLED:
                        # For now, we'll update with zero P&L since the trade just started
                        # Real P&L tracking happens when positions are closed
                        if signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                            # Calculate P&L for exit trades
                            pnl = self._calculate_trade_pnl(signal, order_result, current_positions)
                            self.risk_manager.update_trade_result(pnl, execution_start)
                
                logger.info(f"Signal executed successfully: {signal.signal_type.value} {signal.symbol}")
            else:
                logger.error(f"Signal execution failed: {execution_result.get('reason', 'Unknown error')}")
            
            # Prepare final result
            result = {
                'success': execution_result['success'],
                'signal': signal.to_dict(),
                'risk_result': risk_result.to_dict(),
                'execution_result': execution_result,
                'execution_time': (datetime.now() - execution_start).total_seconds(),
                'timestamp': execution_start.isoformat()
            }
            
            # Add to execution history
            self.execution_history.append(result)
            
            # Keep history limited
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing signal: {e}"
            logger.error(error_msg)
            
            result = {
                'success': False,
                'reason': error_msg,
                'signal': signal.to_dict(),
                'execution_time': (datetime.now() - execution_start).total_seconds(),
                'timestamp': execution_start.isoformat()
            }
            
            self.failed_executions.append(result)
            return result
    
    def _execute_entry_signal(self, signal: TradingSignal, 
                             risk_result: RiskCheckResult) -> Dict[str, Any]:
        """
        Execute entry signal (BUY/SELL)
        
        Args:
            signal: Trading signal
            risk_result: Risk check result with position sizing
            
        Returns:
            Execution result
        """
        if not risk_result.position_size:
            return {
                'success': False,
                'reason': 'No position size calculated'
            }
        
        position_size = risk_result.position_size
        
        # Determine order side
        order_side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
        
        # Determine order type
        if self.config['use_limit_orders']:
            order_type = OrderType.LIMIT
            # Calculate limit price with small offset
            offset_pct = self.config['limit_order_offset_pct']
            if order_side == OrderSide.BUY:
                limit_price = signal.price * (1 - offset_pct)  # Buy slightly below market
            else:
                limit_price = signal.price * (1 + offset_pct)  # Sell slightly above market
        else:
            order_type = OrderType.MARKET
            limit_price = None
        
        # Place primary order
        order_result = self.order_manager.place_order(
            symbol=signal.symbol,
            side=order_side,
            quantity=position_size.size_units,
            order_type=order_type,
            limit_price=limit_price,
            client_order_id=f"signal_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        
        if order_result.status == OrderStatus.REJECTED:
            return {
                'success': False,
                'reason': f'Order rejected: {order_result.error_message}',
                'order_result': order_result
            }
        
        # Wait for fill if market order or timeout for limit orders
        final_order = self.order_manager.wait_for_fill(
            order_result.order_id, 
            timeout=self.config['order_timeout']
        )
        
        if final_order.status != OrderStatus.FILLED:
            return {
                'success': False,
                'reason': f'Order not filled: {final_order.status.value}',
                'order_result': final_order
            }
        
        # Check for excessive slippage
        if final_order.filled_price:
            slippage = abs(final_order.filled_price - signal.price) / signal.price
            max_slippage = self.config['max_slippage_pct']
            
            if slippage > max_slippage:
                logger.warning(f"High slippage detected: {slippage:.3%} > {max_slippage:.3%}")
        
        # Calculate fee-adjusted targets
        position_value = final_order.filled_price * (final_order.filled_quantity or final_order.quantity)
        original_tp_pct = self.config['default_take_profit_pct']
        original_sl_pct = self.config['default_stop_loss_pct']
        
        # Adjust targets for trading fees
        adjusted_tp_pct, adjusted_sl_pct = self.fee_calculator.adjust_targets_for_fees(
            take_profit_pct=original_tp_pct,
            stop_loss_pct=original_sl_pct,
            position_value=position_value,
            symbol=signal.symbol
        )
        
        logger.info(f"Fee adjustment: TP {original_tp_pct:.3f} → {adjusted_tp_pct:.3f}, "
                   f"SL {original_sl_pct:.3f} → {adjusted_sl_pct:.3f}")
        
        # Place stop loss and take profit orders with adjusted targets
        protective_orders = []
        
        if self.config['enable_stop_loss'] and final_order.filled_price:
            stop_order = self._place_stop_loss_order(
                signal, final_order, position_size, stop_loss_pct=adjusted_sl_pct
            )
            if stop_order:
                protective_orders.append(stop_order)
        
        if self.config['enable_take_profit'] and final_order.filled_price:
            tp_order = self._place_take_profit_order(
                signal, final_order, position_size, take_profit_pct=adjusted_tp_pct
            )
            if tp_order:
                protective_orders.append(tp_order)
        
        return {
            'success': True,
            'order_result': final_order,
            'protective_orders': protective_orders,
            'position_size': position_size.to_dict(),
            'slippage': slippage if final_order.filled_price else None
        }
    
    def _execute_exit_signal(self, signal: TradingSignal, 
                            current_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute exit signal (CLOSE_LONG/CLOSE_SHORT)
        
        Args:
            signal: Trading signal
            current_positions: Current positions
            
        Returns:
            Execution result
        """
        # Find position to close
        target_position = None
        for position in current_positions:
            if position.get('symbol') == signal.symbol:
                target_position = position
                break
        
        if not target_position:
            return {
                'success': False,
                'reason': f'No position found for {signal.symbol}'
            }
        
        # Determine order details
        position_qty = abs(float(target_position.get('qty', 0)))
        position_side = target_position.get('side', 'long')
        
        if position_qty <= 0:
            return {
                'success': False,
                'reason': f'Invalid position quantity: {position_qty}'
            }
        
        # Determine order side (opposite of position)
        if (signal.signal_type == SignalType.CLOSE_LONG and position_side == 'long') or \
           (signal.signal_type == SignalType.CLOSE_SHORT and position_side == 'short'):
            order_side = OrderSide.SELL if position_side == 'long' else OrderSide.BUY
        else:
            return {
                'success': False,
                'reason': f'Signal {signal.signal_type.value} does not match position side {position_side}'
            }
        
        # Cancel any existing protective orders for this symbol
        self._cancel_protective_orders(signal.symbol)
        
        # Place exit order
        order_result = self.order_manager.place_order(
            symbol=signal.symbol,
            side=order_side,
            quantity=position_qty,
            order_type=OrderType.MARKET,  # Always use market orders for exits
            client_order_id=f"exit_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        
        if order_result.status == OrderStatus.REJECTED:
            return {
                'success': False,
                'reason': f'Exit order rejected: {order_result.error_message}',
                'order_result': order_result
            }
        
        # Wait for fill
        final_order = self.order_manager.wait_for_fill(order_result.order_id)
        
        success = final_order.status == OrderStatus.FILLED
        
        return {
            'success': success,
            'order_result': final_order,
            'closed_position': target_position,
            'reason': None if success else f'Exit order not filled: {final_order.status.value}'
        }
    
    def _place_stop_loss_order(self, signal: TradingSignal, 
                              entry_order: OrderResult,
                              position_size: Any,
                              stop_loss_pct: Optional[float] = None) -> Optional[OrderResult]:
        """Place stop loss order"""
        try:
            if not entry_order.filled_price:
                return None
            
            # Calculate stop loss price
            stop_loss_pct = stop_loss_pct or self.config['default_stop_loss_pct']
            
            if signal.signal_type == SignalType.BUY:
                stop_price = entry_order.filled_price * (1 - stop_loss_pct)
                order_side = OrderSide.SELL
            else:  # SELL
                stop_price = entry_order.filled_price * (1 + stop_loss_pct)
                order_side = OrderSide.BUY
            
            stop_order = self.order_manager.place_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=entry_order.filled_quantity or entry_order.quantity,
                order_type=OrderType.STOP,
                stop_price=stop_price,
                client_order_id=f"sl_{entry_order.order_id}"
            )
            
            logger.info(f"Stop loss placed: {stop_order.order_id} @ {stop_price:.2f}")
            return stop_order
            
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
            return None
    
    def _place_take_profit_order(self, signal: TradingSignal,
                                entry_order: OrderResult,
                                position_size: Any,
                                take_profit_pct: Optional[float] = None) -> Optional[OrderResult]:
        """Place take profit order"""
        try:
            if not entry_order.filled_price:
                return None
            
            # Calculate take profit price
            take_profit_pct = take_profit_pct or self.config['default_take_profit_pct']
            
            if signal.signal_type == SignalType.BUY:
                limit_price = entry_order.filled_price * (1 + take_profit_pct)
                order_side = OrderSide.SELL
            else:  # SELL
                limit_price = entry_order.filled_price * (1 - take_profit_pct)
                order_side = OrderSide.BUY
            
            tp_order = self.order_manager.place_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=entry_order.filled_quantity or entry_order.quantity,
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                client_order_id=f"tp_{entry_order.order_id}"
            )
            
            logger.info(f"Take profit placed: {tp_order.order_id} @ {limit_price:.2f}")
            return tp_order
            
        except Exception as e:
            logger.error(f"Failed to place take profit: {e}")
            return None
    
    def _cancel_protective_orders(self, symbol: str) -> None:
        """Cancel existing stop loss and take profit orders for symbol"""
        active_orders = self.order_manager.get_active_orders()
        
        for order in active_orders:
            if (order.symbol == symbol and 
                order.metadata and 
                ('sl_' in order.metadata.get('client_order_id', '') or 
                 'tp_' in order.metadata.get('client_order_id', ''))):
                
                self.order_manager.cancel_order(order.order_id)
                logger.info(f"Cancelled protective order: {order.order_id}")
    
    def _calculate_trade_pnl(self, signal: TradingSignal, 
                            order_result: OrderResult,
                            current_positions: List[Dict[str, Any]]) -> float:
        """Calculate P&L for a completed trade"""
        try:
            # Find the position that was closed
            for position in current_positions:
                if position.get('symbol') == signal.symbol:
                    unrealized_pl = float(position.get('unrealized_pl', 0))
                    return unrealized_pl
            
            # If no position found, return 0
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trade P&L: {e}")
            return 0.0
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e['success']])
        failed_executions = len(self.failed_executions)
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'order_manager_stats': self.order_manager.get_stats()
        }
    
    def cancel_all_orders(self) -> int:
        """Cancel all active orders"""
        return self.order_manager.cancel_all_orders()