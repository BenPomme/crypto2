"""
Advanced Backtesting Engine
High-fidelity simulation of trading strategies with realistic order execution
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..strategy.base_strategy import BaseStrategy, TradingSignal, SignalType
from ..strategy.indicators import TechnicalIndicators
from ..risk.position_sizer import PositionSizer, PositionSizeResult

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

@dataclass
class BacktestOrder:
    """Represents an order in the backtest"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None  # For limit orders
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestPosition:
    """Represents a position in the backtest"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestTrade:
    """Represents a completed trade in the backtest"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    commission: float
    hold_time: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)

class BacktestEngine:
    """
    Advanced backtesting engine with realistic order execution simulation
    Features: Slippage modeling, commission calculation, position tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_model: str = 'linear',
        max_slippage: float = 0.0005,  # 0.05% max slippage
        margin_rate: float = 1.0  # No margin for crypto
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital for backtest
            commission_rate: Commission rate (as decimal)
            slippage_model: Slippage calculation model ('linear', 'sqrt', 'fixed')
            max_slippage: Maximum slippage rate
            margin_rate: Margin multiplier (1.0 = no margin)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.max_slippage = max_slippage
        self.margin_rate = margin_rate
        
        # Initialize backtesting state
        self.reset_backtest()
        
        logger.info(f"Backtest engine initialized: ${initial_capital:,.2f} capital, {commission_rate*100:.2f}% commission")
    
    def reset_backtest(self):
        """Reset backtest state for new run"""
        self.current_capital = self.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: List[BacktestOrder] = []
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_time: Optional[datetime] = None
        
        # Performance tracking
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run complete backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            data: Historical OHLCV data with indicators
            parameters: Strategy parameters to override
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        try:
            self.reset_backtest()
            
            # Apply parameters to strategy if provided
            if parameters:
                self._apply_parameters_to_strategy(strategy, parameters)
            
            # Calculate indicators for the data
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            logger.info(f"Starting backtest: {len(data_with_indicators)} bars, strategy: {strategy.name}")
            
            # Initialize position sizer
            position_sizer = PositionSizer()
            
            # Iterate through historical data
            for i, (timestamp, row) in enumerate(data_with_indicators.iterrows()):
                self.current_time = timestamp
                
                # Update current prices and positions
                self._update_positions(row)
                
                # Process pending orders
                self._process_pending_orders(row)
                
                # Generate trading signals
                if i >= 50:  # Need enough data for indicators
                    window_data = data_with_indicators.iloc[:i+1]
                    signal = strategy.generate_signal(window_data)
                    
                    if signal:
                        # Calculate position size
                        position_size = self._calculate_position_size(
                            signal, row, position_sizer
                        )
                        
                        # Execute signal
                        self._execute_signal(signal, row, position_size)
                
                # Record equity curve
                current_equity = self._calculate_total_equity(row)
                self.equity_curve.append((timestamp, current_equity))
                
                # Update performance tracking
                self._update_performance_tracking(current_equity)
            
            # Calculate final performance metrics
            results = self._calculate_performance_metrics(data_with_indicators)
            
            logger.info(f"Backtest completed: {len(self.trades)} trades, "
                       f"Final equity: ${self._calculate_total_equity(data_with_indicators.iloc[-1]):,.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def _apply_parameters_to_strategy(self, strategy: BaseStrategy, parameters: Dict[str, Any]):
        """Apply parameter overrides to strategy"""
        for param_name, param_value in parameters.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
            elif hasattr(strategy, 'config') and param_name in strategy.config:
                strategy.config[param_name] = param_value
    
    def _calculate_position_size(
        self,
        signal: TradingSignal,
        current_data: pd.Series,
        position_sizer: PositionSizer
    ) -> PositionSizeResult:
        """Calculate position size for signal"""
        # Calculate current total equity
        current_equity = self._calculate_total_equity(current_data)
        
        # Available cash for new position
        available_cash = self.current_capital
        
        # Calculate position size
        position_size = position_sizer.calculate_position_size(
            account_value=current_equity,
            entry_price=signal.price,
            buying_power=available_cash,
            is_crypto=True
        )
        
        return position_size
    
    def _execute_signal(
        self,
        signal: TradingSignal,
        current_data: pd.Series,
        position_size: PositionSizeResult
    ):
        """Execute trading signal"""
        symbol = signal.symbol
        
        if signal.signal_type == SignalType.BUY:
            self._open_position(symbol, position_size.size_units, signal.price, current_data)
            
        elif signal.signal_type == SignalType.SELL:
            self._open_short_position(symbol, position_size.size_units, signal.price, current_data)
            
        elif signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            self._close_position(symbol, signal.price, current_data)
    
    def _open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        current_data: pd.Series
    ):
        """Open a long position"""
        # Calculate costs
        position_value = quantity * price
        commission = position_value * self.commission_rate
        slippage = self._calculate_slippage(position_value, current_data)
        execution_price = price * (1 + slippage)
        
        total_cost = position_value + commission
        
        # Check if we have enough capital
        if total_cost > self.current_capital:
            logger.debug(f"Insufficient capital for {symbol} position: need ${total_cost:.2f}, have ${self.current_capital:.2f}")
            return
        
        # Close existing position if any
        if symbol in self.positions:
            self._close_position(symbol, price, current_data)
        
        # Create new position
        self.positions[symbol] = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=execution_price,
            entry_time=self.current_time,
            current_price=price
        )
        
        # Update capital
        self.current_capital -= total_cost
        self.total_commission += commission
        self.total_slippage += slippage * position_value
        
        logger.debug(f"Opened position: {quantity:.6f} {symbol} @ ${execution_price:.2f}")
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        current_data: pd.Series
    ):
        """Close existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate costs
        position_value = position.quantity * price
        commission = position_value * self.commission_rate
        slippage = self._calculate_slippage(position_value, current_data)
        execution_price = price * (1 - slippage)  # Negative slippage for sells
        
        # Calculate P&L
        gross_proceeds = position.quantity * execution_price
        net_proceeds = gross_proceeds - commission
        total_cost = position.quantity * position.entry_price
        pnl = net_proceeds - total_cost
        pnl_pct = (pnl / total_cost) * 100
        
        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            side='long',
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            hold_time=self.current_time - position.entry_time
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.current_capital += net_proceeds
        self.total_commission += commission
        self.total_slippage += slippage * position_value
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed position: {position.quantity:.6f} {symbol} @ ${execution_price:.2f}, P&L: ${pnl:.2f}")
    
    def _calculate_slippage(self, position_value: float, current_data: pd.Series) -> float:
        """Calculate slippage based on position size and market conditions"""
        if self.slippage_model == 'fixed':
            return self.max_slippage
        
        # Get volume-based slippage
        volume = current_data.get('volume', 1000000)
        volume_ratio = position_value / (volume * current_data['close'])
        
        if self.slippage_model == 'linear':
            slippage = min(volume_ratio * 0.1, self.max_slippage)
        elif self.slippage_model == 'sqrt':
            slippage = min(np.sqrt(volume_ratio) * 0.05, self.max_slippage)
        else:
            slippage = self.max_slippage
        
        return slippage
    
    def _update_positions(self, current_data: pd.Series):
        """Update current prices and unrealized P&L for all positions"""
        for symbol, position in self.positions.items():
            position.current_price = current_data['close']
            unrealized_value = position.quantity * position.current_price
            cost_basis = position.quantity * position.entry_price
            position.unrealized_pnl = unrealized_value - cost_basis
    
    def _process_pending_orders(self, current_data: pd.Series):
        """Process any pending orders (for future limit order support)"""
        # TODO: Implement limit order processing
        pass
    
    def _calculate_total_equity(self, current_data: pd.Series) -> float:
        """Calculate total portfolio equity"""
        equity = self.current_capital
        
        for position in self.positions.values():
            equity += position.quantity * current_data['close']
        
        return equity
    
    def _update_performance_tracking(self, current_equity: float):
        """Update performance tracking metrics"""
        # Update peak equity and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        else:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return self._empty_results()
        
        # Close any remaining positions at final price
        if self.positions:
            final_price = data.iloc[-1]
            for symbol in list(self.positions.keys()):
                self._close_position(symbol, final_price['close'], final_price)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        final_equity = self.initial_capital + total_pnl
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([eq[1] for eq in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = total_return / (self.max_drawdown + 1e-10)
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        # Trading metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        avg_hold_time = np.mean([t.hold_time.total_seconds() / 3600 for t in self.trades])  # hours
        
        return {
            # Basic Performance
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            
            # Risk Metrics
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trading Metrics
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_time_hours': avg_hold_time,
            
            # Cost Analysis
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'commission_pct': (self.total_commission / self.initial_capital) * 100,
            
            # Detailed Data
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            
            # Validation
            'backtest_valid': True,
            'error': None
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        
        annualized_return = returns.mean() * 252  # Assuming daily returns
        annualized_vol = returns.std() * np.sqrt(252)
        
        return (annualized_return - risk_free_rate) / annualized_vol
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        annualized_return = returns.mean() * 252
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        return (annualized_return - risk_free_rate) / downside_vol
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results for failed backtests"""
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital,
            'total_return': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_hold_time_hours': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'commission_pct': 0.0,
            'trades': [],
            'equity_curve': [],
            'backtest_valid': False,
            'error': 'No trades generated'
        }