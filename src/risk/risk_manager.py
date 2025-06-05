"""
Risk Management Module
Implements comprehensive risk checks and position limits
"""
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from ..strategy.base_strategy import TradingSignal, SignalType
from .position_sizer import PositionSizer, PositionSizeResult

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskCheck:
    """Individual risk check result"""
    name: str
    passed: bool
    risk_level: RiskLevel
    message: str
    value: Optional[float] = None
    limit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'passed': self.passed,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'value': self.value,
            'limit': self.limit
        }

@dataclass
class RiskCheckResult:
    """Complete risk assessment result"""
    signal: TradingSignal
    approved: bool
    position_size: Optional[PositionSizeResult]
    checks: List[RiskCheck]
    overall_risk_level: RiskLevel
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal': self.signal.to_dict(),
            'approved': self.approved,
            'position_size': self.position_size.to_dict() if self.position_size else None,
            'checks': [check.to_dict() for check in self.checks],
            'overall_risk_level': self.overall_risk_level.value,
            'recommendation': self.recommendation,
            'timestamp': datetime.now().isoformat()
        }

class RiskManager:
    """
    Comprehensive risk management system
    Implements multiple layers of risk checks and position sizing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config or {}
        
        # Default risk configuration
        self.default_config = {
            # Account limits
            'max_position_size': 0.25,  # 25% of account per position
            'max_total_exposure': 0.8,  # 80% of account total exposure
            'max_daily_loss': 0.05,  # 5% max daily loss
            'max_weekly_loss': 0.15,  # 15% max weekly loss
            'max_monthly_loss': 0.30,  # 30% max monthly loss
            
            # Position limits
            'max_concurrent_positions': 3,
            'max_positions_same_asset': 1,
            'min_position_size': 100,  # Minimum $100 position
            
            # Volatility limits
            'max_volatility': 0.15,  # 15% daily volatility limit
            'volatility_lookback': 20,
            
            # Correlation limits
            'max_correlation': 0.7,  # Maximum correlation between positions
            
            # Time-based limits
            'max_trades_per_day': 10,
            'max_trades_per_hour': 3,
            'min_time_between_trades': 300,  # 5 minutes
            
            # Circuit breakers
            'daily_loss_circuit_breaker': 0.05,  # Stop trading at 5% daily loss
            'consecutive_loss_limit': 5,  # Stop after 5 consecutive losses
            
            # Drawdown limits
            'max_drawdown': 0.20,  # 20% max drawdown from peak
            'drawdown_cooldown': 86400,  # 24 hours cooldown after max drawdown
        }
        
        self.config = {**self.default_config, **self.config}
        
        # Initialize position sizer
        self.position_sizer = PositionSizer(self.config)
        
        # Track recent trades and performance
        self.recent_trades = []
        self.daily_pnl = {}
        self.consecutive_losses = 0
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.last_drawdown_breach = None
        
        logger.info("RiskManager initialized")
    
    def evaluate_signal(self, 
                       signal: TradingSignal,
                       account_info: Dict[str, Any],
                       current_positions: List[Dict[str, Any]],
                       market_data: Optional[pd.DataFrame] = None) -> RiskCheckResult:
        """
        Comprehensive risk evaluation of a trading signal
        
        Args:
            signal: Trading signal to evaluate
            account_info: Current account information
            current_positions: List of current positions
            market_data: Recent market data for volatility calculations
            
        Returns:
            RiskCheckResult with approval decision
        """
        checks = []
        approved = True
        overall_risk = RiskLevel.LOW
        
        # Extract account values
        account_value = account_info.get('portfolio_value', 0)
        cash_available = account_info.get('cash', 0)
        buying_power = account_info.get('buying_power', account_value)
        is_pattern_day_trader = account_info.get('pattern_day_trader', False)
        
        # Debug account info for cash availability issues
        logger.info(f"Account info: portfolio_value=${account_value:.2f}, cash=${cash_available:.2f}, buying_power=${buying_power:.2f}")
        
        # For paper trading crypto, use portfolio_value as available cash if cash is 0
        if cash_available == 0 and account_value > 0:
            cash_available = account_value
            logger.info(f"Using portfolio_value as available cash for crypto trading: ${cash_available:.2f}")
        
        # Perform risk checks
        checks.extend(self._check_account_limits(signal, account_value, cash_available))
        checks.extend(self._check_position_limits(signal, current_positions, account_value))
        checks.extend(self._check_volatility_limits(signal, market_data))
        checks.extend(self._check_correlation_limits(signal, current_positions))
        checks.extend(self._check_time_based_limits(signal))
        checks.extend(self._check_performance_limits(signal, account_value))
        checks.extend(self._check_drawdown_limits(signal, account_value))
        
        # Determine overall approval and risk level
        failed_checks = [check for check in checks if not check.passed]
        
        if failed_checks:
            # Check if any critical failures
            critical_failures = [check for check in failed_checks 
                                if check.risk_level == RiskLevel.CRITICAL]
            
            if critical_failures:
                approved = False
                overall_risk = RiskLevel.CRITICAL
            else:
                # Check high risk failures
                high_risk_failures = [check for check in failed_checks 
                                    if check.risk_level == RiskLevel.HIGH]
                
                if len(high_risk_failures) > 2:  # Multiple high risk failures
                    approved = False
                    overall_risk = RiskLevel.HIGH
                elif len(failed_checks) > 5:  # Too many failures overall
                    approved = False
                    overall_risk = RiskLevel.HIGH
                else:
                    # Medium risk - approve with caution
                    overall_risk = RiskLevel.MEDIUM
        
        # Calculate position size if approved
        position_size = None
        if approved and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            try:
                # Calculate volatility if market data available
                volatility = None
                if market_data is not None and not market_data.empty:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) > 5:
                        volatility = returns.std()
                
                # All symbols are crypto now
                is_crypto = True
                
                # Calculate stop loss price for risk-based sizing
                stop_loss_price = None
                if hasattr(signal, 'metadata') and signal.metadata:
                    stop_loss_price = signal.metadata.get('stop_loss_price')
                
                # For crypto trading, pass the actual available cash instead of buying_power
                effective_buying_power = cash_available if is_crypto else buying_power
                
                position_size = self.position_sizer.calculate_position_size(
                    account_value=account_value,
                    entry_price=signal.price,
                    volatility=volatility,
                    buying_power=effective_buying_power,
                    is_crypto=is_crypto,
                    is_pattern_day_trader=is_pattern_day_trader,
                    stop_loss_price=stop_loss_price
                )
                
                # Crypto position validation (cash only, no margin)
                required_cash = position_size.size_usd
                available_cash = cash_available
                
                if required_cash > available_cash:
                    approved = False
                    checks.append(RiskCheck(
                        name="cash_availability_check",
                        passed=False,
                        risk_level=RiskLevel.CRITICAL,
                        message=f"Insufficient cash for crypto trade: need ${required_cash:.2f}, have ${available_cash:.2f}",
                        value=required_cash,
                        limit=available_cash
                    ))
                
            except Exception as e:
                logger.error(f"Error calculating position size: {e}")
                approved = False
                checks.append(RiskCheck(
                    name="position_sizing_error",
                    passed=False,
                    risk_level=RiskLevel.HIGH,
                    message=f"Position sizing failed: {e}"
                ))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(approved, overall_risk, failed_checks)
        
        result = RiskCheckResult(
            signal=signal,
            approved=approved,
            position_size=position_size,
            checks=checks,
            overall_risk_level=overall_risk,
            recommendation=recommendation
        )
        
        # Log result
        if approved:
            logger.info(f"Signal APPROVED: {signal.signal_type.value} {signal.symbol} "
                       f"(Risk: {overall_risk.value})")
        else:
            # Count all failed checks including those added during position sizing
            all_failed_checks = [check for check in checks if not check.passed]
            logger.warning(f"Signal REJECTED: {signal.signal_type.value} {signal.symbol} "
                          f"(Risk: {overall_risk.value}, Failures: {len(all_failed_checks)})")
            
            # Log specific failure reasons for debugging
            for check in all_failed_checks:
                logger.warning(f"  - {check.name}: {check.message}")
        
        return result
    
    def _check_account_limits(self, signal: TradingSignal, 
                             account_value: float, 
                             cash_available: float) -> List[RiskCheck]:
        """Check account-level risk limits"""
        checks = []
        
        # Check minimum account value
        min_account_value = self.config.get('min_account_value', 1000)
        checks.append(RiskCheck(
            name="min_account_value",
            passed=account_value >= min_account_value,
            risk_level=RiskLevel.CRITICAL,
            message=f"Account value: ${account_value:.2f}",
            value=account_value,
            limit=min_account_value
        ))
        
        # Check cash availability for new positions
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            min_cash_reserve = account_value * 0.1  # Keep 10% cash reserve
            checks.append(RiskCheck(
                name="cash_reserve",
                passed=cash_available >= min_cash_reserve,
                risk_level=RiskLevel.MEDIUM,
                message=f"Cash available: ${cash_available:.2f}",
                value=cash_available,
                limit=min_cash_reserve
            ))
        
        return checks
    
    def _check_position_limits(self, signal: TradingSignal,
                              current_positions: List[Dict[str, Any]],
                              account_value: float) -> List[RiskCheck]:
        """Check position-related limits"""
        checks = []
        
        # Check maximum concurrent positions
        max_positions = self.config['max_concurrent_positions']
        current_position_count = len(current_positions)
        
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            checks.append(RiskCheck(
                name="max_concurrent_positions",
                passed=current_position_count < max_positions,
                risk_level=RiskLevel.HIGH,
                message=f"Current positions: {current_position_count}/{max_positions}",
                value=current_position_count,
                limit=max_positions
            ))
        
        # Check same asset position limit
        same_asset_positions = [pos for pos in current_positions 
                               if pos.get('symbol') == signal.symbol]
        max_same_asset = self.config['max_positions_same_asset']
        
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            checks.append(RiskCheck(
                name="max_same_asset_positions",
                passed=len(same_asset_positions) < max_same_asset,
                risk_level=RiskLevel.HIGH,
                message=f"Positions in {signal.symbol}: {len(same_asset_positions)}/{max_same_asset}",
                value=len(same_asset_positions),
                limit=max_same_asset
            ))
        
        # Check total exposure
        total_exposure = sum(abs(float(pos.get('market_value', 0))) for pos in current_positions)
        max_exposure = account_value * self.config['max_total_exposure']
        
        checks.append(RiskCheck(
            name="total_exposure",
            passed=total_exposure <= max_exposure,
            risk_level=RiskLevel.HIGH,
            message=f"Total exposure: ${total_exposure:.2f} (Max: ${max_exposure:.2f})",
            value=total_exposure,
            limit=max_exposure
        ))
        
        return checks
    
    def _check_volatility_limits(self, signal: TradingSignal,
                                market_data: Optional[pd.DataFrame]) -> List[RiskCheck]:
        """Check volatility-based limits"""
        checks = []
        
        if market_data is None or market_data.empty:
            checks.append(RiskCheck(
                name="volatility_data",
                passed=True,  # Pass if no data available
                risk_level=RiskLevel.LOW,
                message="No market data for volatility check"
            ))
            return checks
        
        # Calculate recent volatility
        returns = market_data['close'].pct_change().dropna()
        if len(returns) >= 5:
            volatility = returns.tail(self.config['volatility_lookback']).std()
            max_vol = self.config['max_volatility']
            
            checks.append(RiskCheck(
                name="volatility_limit",
                passed=volatility <= max_vol,
                risk_level=RiskLevel.MEDIUM,
                message=f"Volatility: {volatility:.3f} (Max: {max_vol:.3f})",
                value=volatility,
                limit=max_vol
            ))
        
        return checks
    
    def _check_correlation_limits(self, signal: TradingSignal,
                                 current_positions: List[Dict[str, Any]]) -> List[RiskCheck]:
        """Check correlation limits between positions"""
        checks = []
        
        # Simplified correlation check - in practice would need historical correlation data
        # For now, just check if adding same asset type
        crypto_positions = [pos for pos in current_positions 
                           if pos.get('symbol', '').endswith('USD')]
        
        max_corr = self.config['max_correlation']
        
        if signal.symbol.endswith('USD') and len(crypto_positions) > 0:
            # Assume high correlation between crypto assets
            estimated_correlation = 0.8
            checks.append(RiskCheck(
                name="correlation_limit",
                passed=estimated_correlation <= max_corr or len(crypto_positions) < 2,
                risk_level=RiskLevel.MEDIUM,
                message=f"Estimated crypto correlation: {estimated_correlation:.2f}",
                value=estimated_correlation,
                limit=max_corr
            ))
        
        return checks
    
    def _check_time_based_limits(self, signal: TradingSignal) -> List[RiskCheck]:
        """Check time-based trading limits"""
        checks = []
        
        now = datetime.now()
        
        # Check trades per day
        today_trades = [trade for trade in self.recent_trades 
                       if trade.get('date', '').startswith(now.strftime('%Y-%m-%d'))]
        
        max_daily_trades = self.config['max_trades_per_day']
        checks.append(RiskCheck(
            name="daily_trade_limit",
            passed=len(today_trades) < max_daily_trades,
            risk_level=RiskLevel.MEDIUM,
            message=f"Trades today: {len(today_trades)}/{max_daily_trades}",
            value=len(today_trades),
            limit=max_daily_trades
        ))
        
        # Check minimum time between trades
        if self.recent_trades:
            last_trade_time = self.recent_trades[-1].get('timestamp')
            if last_trade_time:
                try:
                    last_time = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00'))
                    time_diff = (now - last_time).total_seconds()
                    min_time = self.config['min_time_between_trades']
                    
                    checks.append(RiskCheck(
                        name="min_time_between_trades",
                        passed=time_diff >= min_time,
                        risk_level=RiskLevel.LOW,
                        message=f"Time since last trade: {time_diff:.0f}s",
                        value=time_diff,
                        limit=min_time
                    ))
                except:
                    pass  # Skip if time parsing fails
        
        return checks
    
    def _check_performance_limits(self, signal: TradingSignal, 
                                 account_value: float) -> List[RiskCheck]:
        """Check performance-based limits"""
        checks = []
        
        # Check daily loss limit
        today = datetime.now().strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0)
        max_daily_loss = account_value * self.config['max_daily_loss']
        
        checks.append(RiskCheck(
            name="daily_loss_limit",
            passed=abs(daily_loss) <= max_daily_loss if daily_loss < 0 else True,
            risk_level=RiskLevel.CRITICAL,
            message=f"Daily P&L: ${daily_loss:.2f}",
            value=abs(daily_loss) if daily_loss < 0 else 0,
            limit=max_daily_loss
        ))
        
        # Check consecutive losses
        max_consecutive = self.config['consecutive_loss_limit']
        checks.append(RiskCheck(
            name="consecutive_losses",
            passed=self.consecutive_losses < max_consecutive,
            risk_level=RiskLevel.HIGH,
            message=f"Consecutive losses: {self.consecutive_losses}",
            value=self.consecutive_losses,
            limit=max_consecutive
        ))
        
        return checks
    
    def _check_drawdown_limits(self, signal: TradingSignal, 
                              account_value: float) -> List[RiskCheck]:
        """Check drawdown limits"""
        checks = []
        
        # Update peak equity
        if self.peak_equity is None or account_value > self.peak_equity:
            self.peak_equity = account_value
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - account_value) / self.peak_equity
        
        max_dd = self.config['max_drawdown']
        checks.append(RiskCheck(
            name="max_drawdown",
            passed=self.current_drawdown <= max_dd,
            risk_level=RiskLevel.CRITICAL,
            message=f"Current drawdown: {self.current_drawdown:.2%}",
            value=self.current_drawdown,
            limit=max_dd
        ))
        
        # Check cooldown period after drawdown breach
        if self.last_drawdown_breach:
            cooldown = self.config['drawdown_cooldown']
            time_since_breach = (datetime.now() - self.last_drawdown_breach).total_seconds()
            
            checks.append(RiskCheck(
                name="drawdown_cooldown",
                passed=time_since_breach >= cooldown,
                risk_level=RiskLevel.HIGH,
                message=f"Time since drawdown breach: {time_since_breach/3600:.1f}h",
                value=time_since_breach,
                limit=cooldown
            ))
        
        return checks
    
    def _generate_recommendation(self, approved: bool, 
                               risk_level: RiskLevel,
                               failed_checks: List[RiskCheck]) -> str:
        """Generate risk recommendation text"""
        if approved:
            if risk_level == RiskLevel.LOW:
                return "Signal approved with low risk. Proceed with full position size."
            elif risk_level == RiskLevel.MEDIUM:
                return "Signal approved with medium risk. Consider reduced position size."
            else:
                return "Signal approved but high risk detected. Use caution."
        else:
            critical_failures = [c for c in failed_checks if c.risk_level == RiskLevel.CRITICAL]
            if critical_failures:
                return f"Signal rejected due to critical risk: {critical_failures[0].message}"
            else:
                return f"Signal rejected due to {len(failed_checks)} risk violations."
    
    def update_trade_result(self, trade_pnl: float, trade_timestamp: datetime) -> None:
        """
        Update risk manager with trade result
        
        Args:
            trade_pnl: Trade profit/loss
            trade_timestamp: Trade timestamp
        """
        # Update recent trades
        trade_record = {
            'pnl': trade_pnl,
            'timestamp': trade_timestamp.isoformat(),
            'date': trade_timestamp.strftime('%Y-%m-%d')
        }
        self.recent_trades.append(trade_record)
        
        # Keep only recent trades (last 100)
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]
        
        # Update daily P&L
        date_str = trade_timestamp.strftime('%Y-%m-%d')
        self.daily_pnl[date_str] = self.daily_pnl.get(date_str, 0) + trade_pnl
        
        # Update consecutive losses
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update drawdown breach timestamp
        if self.current_drawdown > self.config['max_drawdown']:
            self.last_drawdown_breach = trade_timestamp
        
        logger.info(f"Trade result updated: P&L ${trade_pnl:.2f}, "
                   f"Consecutive losses: {self.consecutive_losses}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        return {
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'daily_pnl': dict(list(self.daily_pnl.items())[-7:]),  # Last 7 days
            'recent_trades_count': len(self.recent_trades),
            'last_drawdown_breach': self.last_drawdown_breach.isoformat() if self.last_drawdown_breach else None
        }