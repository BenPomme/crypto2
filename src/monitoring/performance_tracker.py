"""
Performance Tracker Module
Main performance tracking and monitoring system
Coordinates metrics calculation, logging, and real-time monitoring
"""
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import threading
from dataclasses import dataclass

from .firebase_logger import FirebaseLogger
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    total_return: float
    total_pnl: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_equity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_equity': self.current_equity
        }

class PerformanceTracker:
    """
    Comprehensive performance tracking system
    Tracks trading performance, logs to Firebase, and provides real-time monitoring
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance tracker
        
        Args:
            initial_capital: Starting capital amount
            config: Configuration options
        """
        self.initial_capital = initial_capital
        self.config = config or {}
        
        # Components
        self.firebase_logger = FirebaseLogger()
        self.metrics_calculator = MetricsCalculator()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.trades_log = []
        self.signals_log = []
        self.equity_curve = [initial_capital]
        self.daily_snapshots = {}
        
        # Real-time metrics
        self.current_metrics = {}
        self.last_update = None
        
        # Configuration
        self.default_config = {
            'log_to_firebase': True,
            'snapshot_interval': 3600,  # 1 hour
            'cleanup_days': 90,
            'max_trades_memory': 10000,
            'enable_real_time_logging': True
        }
        
        self.config = {**self.default_config, **self.config}
        
        # Background monitoring
        self._monitoring = True
        self._monitor_thread = None
        
        logger.info(f"PerformanceTracker initialized with ${initial_capital:.2f} starting capital")
        
        # Start monitoring thread
        self._start_monitoring()
    
    def record_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Record a completed trade
        
        Args:
            trade_result: Trade execution result
        """
        try:
            # Extract trade information
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_result.get('signal', {}).get('symbol', 'UNKNOWN'),
                'signal_type': trade_result.get('signal', {}).get('signal_type', 'UNKNOWN'),
                'entry_price': trade_result.get('signal', {}).get('price', 0),
                'success': trade_result.get('success', False),
                'pnl': 0,  # Will be updated when position is closed
                'execution_time': trade_result.get('execution_time', 0),
                'metadata': trade_result
            }
            
            # Add to trades log
            self.trades_log.append(trade_record)
            
            # Keep memory usage reasonable
            if len(self.trades_log) > self.config['max_trades_memory']:
                self.trades_log = self.trades_log[-self.config['max_trades_memory']:]
            
            # Log to Firebase
            if self.config['log_to_firebase']:
                self.firebase_logger.log_trade(trade_record)
            
            # Update real-time metrics
            self._update_metrics()
            
            logger.info(f"Trade recorded: {trade_record['signal_type']} {trade_record['symbol']}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def record_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Record a trading signal
        
        Args:
            signal_data: Signal data
        """
        try:
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'signal_type': signal_data.get('signal_type', 'UNKNOWN'),
                'symbol': signal_data.get('symbol', 'UNKNOWN'),
                'price': signal_data.get('price', 0),
                'confidence': signal_data.get('confidence', 0),
                'reason': signal_data.get('reason', ''),
                'metadata': signal_data
            }
            
            # Add to signals log
            self.signals_log.append(signal_record)
            
            # Keep memory usage reasonable
            if len(self.signals_log) > self.config['max_trades_memory']:
                self.signals_log = self.signals_log[-self.config['max_trades_memory']:]
            
            # Log to Firebase
            if self.config['log_to_firebase']:
                self.firebase_logger.log_signal(signal_record)
            
            logger.debug(f"Signal recorded: {signal_record['signal_type']} {signal_record['symbol']}")
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
    
    def update_portfolio_value(self, current_value: float, 
                              account_info: Dict[str, Any] = None,
                              positions: List[Dict[str, Any]] = None) -> None:
        """
        Update current portfolio value
        
        Args:
            current_value: Current portfolio value
            account_info: Account information
            positions: Current positions
        """
        try:
            # Update equity curve
            self.equity_curve.append(current_value)
            
            # Keep reasonable history
            if len(self.equity_curve) > 10000:
                self.equity_curve = self.equity_curve[-10000:]
            
            # Calculate metrics with current data
            portfolio_metrics = {}
            if account_info and positions:
                portfolio_metrics = self.metrics_calculator.calculate_portfolio_metrics(
                    account_info, positions
                )
            
            # Update system status in Firebase
            if self.config['log_to_firebase']:
                status_data = {
                    'current_equity': current_value,
                    'initial_capital': self.initial_capital,
                    'total_return': (current_value - self.initial_capital) / self.initial_capital,
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'last_trade_count': len(self.trades_log),
                    **portfolio_metrics
                }
                
                self.firebase_logger.update_system_status(status_data)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Current performance data
        """
        try:
            # Calculate metrics from current trades
            metrics = self.metrics_calculator.calculate_returns_metrics(
                self.trades_log, self.initial_capital
            )
            
            # Add trading metrics
            trading_metrics = self.metrics_calculator.calculate_trading_metrics(self.trades_log)
            metrics.update(trading_metrics)
            
            # Add risk metrics
            risk_metrics = self.metrics_calculator.calculate_risk_metrics(self.trades_log)
            metrics.update(risk_metrics)
            
            # Add current status
            current_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
            metrics.update({
                'current_equity': current_equity,
                'initial_capital': self.initial_capital,
                'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'num_signals': len(self.signals_log),
                'last_update': self.last_update.isoformat() if self.last_update else None
            })
            
            # Cache metrics
            self.current_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting current performance: {e}")
            return {}
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get daily performance summary
        
        Args:
            date: Date for summary (today if None)
            
        Returns:
            Daily summary data
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            # Filter trades for the day
            daily_trades = [
                trade for trade in self.trades_log 
                if trade.get('timestamp', '').startswith(date_str)
            ]
            
            # Calculate daily metrics
            daily_metrics = self.metrics_calculator.calculate_returns_metrics(
                daily_trades, self.initial_capital
            )
            
            # Add daily specific data
            daily_metrics.update({
                'date': date_str,
                'daily_trades': len(daily_trades),
                'daily_signals': len([
                    signal for signal in self.signals_log 
                    if signal.get('timestamp', '').startswith(date_str)
                ])
            })
            
            return daily_metrics
            
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            return {}
    
    def create_performance_snapshot(self) -> PerformanceSnapshot:
        """Create a performance snapshot"""
        try:
            current_metrics = self.get_current_performance()
            current_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                total_return=current_metrics.get('total_return', 0),
                total_pnl=current_metrics.get('total_pnl', 0),
                num_trades=current_metrics.get('num_trades', 0),
                win_rate=current_metrics.get('win_rate', 0),
                sharpe_ratio=current_metrics.get('sharpe_ratio', 0),
                max_drawdown=current_metrics.get('max_drawdown', 0),
                current_equity=current_equity
            )
            
            # Store snapshot
            date_str = snapshot.timestamp.strftime('%Y-%m-%d')
            self.daily_snapshots[date_str] = snapshot
            
            # Log to Firebase
            if self.config['log_to_firebase']:
                self.firebase_logger.log_performance(snapshot.to_dict())
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating performance snapshot: {e}")
            return None
    
    def _update_metrics(self) -> None:
        """Update cached metrics"""
        try:
            self.current_metrics = self.get_current_performance()
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Performance monitoring thread started")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # Create periodic snapshots
                self.create_performance_snapshot()
                
                # Sleep for configured interval
                time.sleep(self.config['snapshot_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep 1 minute on error
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def get_equity_curve(self) -> List[float]:
        """Get equity curve data"""
        return self.equity_curve.copy()
    
    def get_monthly_performance(self) -> Dict[str, float]:
        """Get monthly performance breakdown"""
        return self.metrics_calculator.calculate_monthly_returns(
            self.trades_log, self.initial_capital
        )
    
    def export_performance_report(self) -> Dict[str, Any]:
        """
        Export comprehensive performance report
        
        Returns:
            Complete performance report
        """
        try:
            report = {
                'summary': self.get_current_performance(),
                'daily_summary': self.get_daily_summary(),
                'monthly_returns': self.get_monthly_performance(),
                'equity_curve': self.get_equity_curve(),
                'trade_log': self.trades_log[-100:],  # Last 100 trades
                'signal_log': self.signals_log[-100:],  # Last 100 signals
                'snapshots': {k: v.to_dict() for k, v in self.daily_snapshots.items()},
                'report_timestamp': datetime.now().isoformat(),
                'runtime_info': {
                    'start_time': self.start_time.isoformat(),
                    'initial_capital': self.initial_capital,
                    'total_runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return {}
    
    def reset_performance(self, new_initial_capital: Optional[float] = None) -> None:
        """
        Reset performance tracking
        
        Args:
            new_initial_capital: New starting capital (keeps current if None)
        """
        if new_initial_capital is not None:
            self.initial_capital = new_initial_capital
        
        self.start_time = datetime.now()
        self.trades_log.clear()
        self.signals_log.clear()
        self.equity_curve = [self.initial_capital]
        self.daily_snapshots.clear()
        self.current_metrics.clear()
        self.last_update = None
        
        logger.info(f"Performance tracker reset with ${self.initial_capital:.2f} capital")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except:
            pass