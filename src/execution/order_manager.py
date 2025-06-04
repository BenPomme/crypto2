"""
Order Management Module
Handles order placement, tracking, and management via Alpaca API
"""
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from alpaca_trade_api import REST
from alpaca_trade_api.common import URL

from config.settings import get_settings
from ..strategy.base_strategy import TradingSignal, SignalType

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderResult:
    """Result of order execution"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'status': self.status.value,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'error_message': self.error_message,
            'metadata': self.metadata or {}
        }

class OrderManager:
    """
    Order management system for Alpaca trading
    Handles order placement, tracking, and lifecycle management
    """
    
    def __init__(self):
        """Initialize order manager"""
        settings = get_settings()
        self.api = REST(
            key_id=settings.alpaca.key,
            secret_key=settings.alpaca.secret,
            base_url=URL(settings.alpaca.endpoint)
        )
        
        # Order tracking
        self.active_orders = {}  # order_id -> OrderResult
        self.order_history = []  # List of completed orders
        
        # Configuration
        self.config = {
            'default_order_type': OrderType.MARKET,
            'order_timeout': 300,  # 5 minutes
            'max_retries': 3,
            'retry_delay': 1,  # seconds
            'fill_check_interval': 2,  # seconds
        }
        
        logger.info("OrderManager initialized")
    
    def place_order(self, 
                   symbol: str,
                   side: OrderSide,
                   quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   client_order_id: Optional[str] = None) -> OrderResult:
        """
        Place a crypto order with Alpaca (time_in_force not supported for crypto)
        
        Args:
            symbol: Trading symbol (e.g., BTC/USD)
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type (MARKET recommended for crypto)
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            client_order_id: Client-side order ID
            
        Returns:
            OrderResult with order details
        """
        try:
            # Validate inputs
            if quantity <= 0:
                return OrderResult(
                    order_id="",
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    status=OrderStatus.REJECTED,
                    error_message="Invalid quantity: must be positive",
                    timestamp=datetime.now()
                )
            
            # Prepare order parameters for crypto trading
            order_params = {
                'symbol': symbol,
                'qty': quantity,
                'side': side.value,
                'type': order_type.value
            }
            
            # Note: Crypto orders on Alpaca don't support time_in_force parameter
            # Only add time_in_force for non-crypto symbols if needed in the future
            
            if client_order_id:
                order_params['client_order_id'] = client_order_id
            
            if order_type == OrderType.LIMIT and limit_price:
                order_params['limit_price'] = limit_price
            elif order_type == OrderType.STOP and stop_price:
                order_params['stop_price'] = stop_price
            elif order_type == OrderType.STOP_LIMIT and limit_price and stop_price:
                order_params['limit_price'] = limit_price
                order_params['stop_price'] = stop_price
            
            # Submit order to Alpaca
            logger.info(f"Placing {side.value} order: {quantity} {symbol} @ {order_type.value}")
            
            alpaca_order = self.api.submit_order(**order_params)
            
            # Create order result
            result = OrderResult(
                order_id=alpaca_order.id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.SUBMITTED,
                timestamp=datetime.now(),
                metadata={
                    'alpaca_order_id': alpaca_order.id,
                    'client_order_id': alpaca_order.client_order_id,
                    'order_class': 'crypto'
                }
            )
            
            # Track active order
            self.active_orders[alpaca_order.id] = result
            
            logger.info(f"Order submitted successfully: {alpaca_order.id}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to place order: {e}"
            logger.error(error_msg)
            
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.REJECTED,
                error_message=error_msg,
                timestamp=datetime.now()
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            self.api.cancel_order(order_id)
            
            # Update order status
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED
                self._move_to_history(order_id)
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """
        Get current status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            OrderResult with current status or None if not found
        """
        try:
            # Check if we have it locally first
            if order_id in self.active_orders:
                local_order = self.active_orders[order_id]
                
                # Update from Alpaca if not final status
                if local_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                            OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    self._update_order_from_alpaca(order_id)
                
                return self.active_orders.get(order_id, local_order)
            
            # Check order history
            for order in self.order_history:
                if order.order_id == order_id:
                    return order
            
            # Try to get from Alpaca directly
            alpaca_order = self.api.get_order(order_id)
            return self._convert_alpaca_order(alpaca_order)
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    def wait_for_fill(self, order_id: str, timeout: Optional[int] = None) -> OrderResult:
        """
        Wait for order to fill
        
        Args:
            order_id: Order ID to wait for
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            OrderResult with final status
        """
        timeout = timeout or self.config['order_timeout']
        start_time = time.time()
        check_interval = self.config['fill_check_interval']
        
        while time.time() - start_time < timeout:
            order_result = self.get_order_status(order_id)
            
            if order_result is None:
                logger.error(f"Order {order_id} not found")
                break
            
            if order_result.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                     OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.info(f"Order {order_id} final status: {order_result.status.value}")
                return order_result
            
            time.sleep(check_interval)
        
        # Timeout reached
        logger.warning(f"Order {order_id} wait timeout after {timeout}s")
        order_result = self.get_order_status(order_id)
        return order_result or OrderResult(
            order_id=order_id,
            symbol="UNKNOWN",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.MARKET,
            status=OrderStatus.EXPIRED,
            error_message="Wait timeout",
            timestamp=datetime.now()
        )
    
    def _update_order_from_alpaca(self, order_id: str) -> None:
        """Update local order status from Alpaca"""
        try:
            alpaca_order = self.api.get_order(order_id)
            updated_order = self._convert_alpaca_order(alpaca_order)
            
            if order_id in self.active_orders:
                # Update existing order
                self.active_orders[order_id].status = updated_order.status
                self.active_orders[order_id].filled_price = updated_order.filled_price
                self.active_orders[order_id].filled_quantity = updated_order.filled_quantity
                
                # Move to history if completed
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                          OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    self._move_to_history(order_id)
            
        except Exception as e:
            logger.error(f"Failed to update order {order_id} from Alpaca: {e}")
    
    def _convert_alpaca_order(self, alpaca_order) -> OrderResult:
        """Convert Alpaca order object to OrderResult"""
        # Map Alpaca status to our status
        status_map = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.CANCELLED,
            'pending_cancel': OrderStatus.CANCELLED,
            'pending_replace': OrderStatus.SUBMITTED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.REJECTED,
            'calculated': OrderStatus.SUBMITTED,
        }
        
        status = status_map.get(alpaca_order.status, OrderStatus.PENDING)
        
        # Extract filled information
        filled_price = None
        filled_qty = None
        
        if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price:
            filled_price = float(alpaca_order.filled_avg_price)
        
        if hasattr(alpaca_order, 'filled_qty') and alpaca_order.filled_qty:
            filled_qty = float(alpaca_order.filled_qty)
        
        return OrderResult(
            order_id=alpaca_order.id,
            symbol=alpaca_order.symbol,
            side=OrderSide.BUY if alpaca_order.side == 'buy' else OrderSide.SELL,
            quantity=float(alpaca_order.qty),
            order_type=OrderType(alpaca_order.order_type),
            status=status,
            filled_price=filled_price,
            filled_quantity=filled_qty,
            timestamp=alpaca_order.created_at,
            metadata={
                'alpaca_status': alpaca_order.status,
                'client_order_id': alpaca_order.client_order_id
            }
        )
    
    def _move_to_history(self, order_id: str) -> None:
        """Move completed order from active to history"""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)
            
            # Keep history limited
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]
    
    def get_active_orders(self) -> List[OrderResult]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: Optional[int] = None) -> List[OrderResult]:
        """Get order history"""
        history = self.order_history
        if limit:
            history = history[-limit:]
        return history
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all active orders
        
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        for order_id in list(self.active_orders.keys()):
            if self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} active orders")
        return cancelled_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get order manager statistics"""
        total_orders = len(self.order_history) + len(self.active_orders)
        filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
        
        return {
            'active_orders': len(self.active_orders),
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'order_history_size': len(self.order_history)
        }