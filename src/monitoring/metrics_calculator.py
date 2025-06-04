"""
Metrics Calculator Module
Calculates trading performance metrics and statistics
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculate comprehensive trading performance metrics
    Provides standard financial metrics for strategy evaluation
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        logger.info("MetricsCalculator initialized")
    
    def calculate_returns_metrics(self, trades: List[Dict[str, Any]], 
                                 initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Calculate return-based performance metrics
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Starting capital amount
            
        Returns:
            Dictionary of return metrics
        """
        if not trades:
            return self._empty_metrics()
        
        try:
            # Convert trades to DataFrame
            df = pd.DataFrame(trades)
            
            # Extract P&L values
            if 'pnl' in df.columns:
                pnl_series = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
            elif 'realized_pl' in df.columns:
                pnl_series = pd.to_numeric(df['realized_pl'], errors='coerce').fillna(0)
            else:
                logger.warning("No P&L column found in trades data")
                return self._empty_metrics()
            
            # Calculate cumulative P&L and equity curve
            cumulative_pnl = pnl_series.cumsum()
            equity_curve = initial_capital + cumulative_pnl
            
            # Calculate returns
            returns = pnl_series / initial_capital
            cumulative_returns = cumulative_pnl / initial_capital
            
            # Basic metrics
            total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
            total_pnl = cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0
            
            # Win/Loss metrics
            winning_trades = pnl_series[pnl_series > 0]
            losing_trades = pnl_series[pnl_series < 0]
            
            num_trades = len(pnl_series)
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            win_rate = num_winning / num_trades if num_trades > 0 else 0
            
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            annual_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'annualized_return': annual_return,
                'num_trades': num_trades,
                'num_winning': num_winning,
                'num_losing': num_losing,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'final_equity': equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
            }
            
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
            return self._empty_metrics()
    
    def calculate_risk_metrics(self, trades: List[Dict[str, Any]], 
                              positions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate risk-related metrics
        
        Args:
            trades: List of trade dictionaries
            positions: Current positions (optional)
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if not trades:
                return {
                    'value_at_risk_95': 0,
                    'value_at_risk_99': 0,
                    'expected_shortfall': 0,
                    'current_exposure': 0,
                    'max_position_size': 0
                }
            
            df = pd.DataFrame(trades)
            
            # Extract P&L
            if 'pnl' in df.columns:
                pnl_series = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
            else:
                return self._empty_risk_metrics()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(pnl_series, 5) if len(pnl_series) > 0 else 0
            var_99 = np.percentile(pnl_series, 1) if len(pnl_series) > 0 else 0
            
            # Expected Shortfall (Conditional VaR)
            losses_below_var = pnl_series[pnl_series <= var_95]
            expected_shortfall = losses_below_var.mean() if len(losses_below_var) > 0 else 0
            
            # Current exposure
            current_exposure = 0
            max_position_size = 0
            
            if positions:
                for pos in positions:
                    market_value = abs(float(pos.get('market_value', 0)))
                    current_exposure += market_value
                    max_position_size = max(max_position_size, market_value)
            
            return {
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'expected_shortfall': expected_shortfall,
                'current_exposure': current_exposure,
                'max_position_size': max_position_size
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._empty_risk_metrics()
    
    def calculate_trading_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trading activity metrics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trading metrics
        """
        try:
            if not trades:
                return {
                    'trades_per_day': 0,
                    'avg_hold_time': 0,
                    'turnover_rate': 0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'largest_win': 0,
                    'largest_loss': 0
                }
            
            df = pd.DataFrame(trades)
            
            # Convert timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
            
            # Extract P&L
            pnl_series = pd.Series()
            if 'pnl' in df.columns:
                pnl_series = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
            
            # Trading frequency
            if len(df) > 0 and 'timestamp' in df.columns:
                date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                trades_per_day = len(df) / max(date_range, 1)
            else:
                trades_per_day = 0
            
            # Hold time (simplified - would need entry/exit timestamps for accuracy)
            avg_hold_time = 0  # Placeholder
            
            # Win/Loss streaks
            if len(pnl_series) > 0:
                wins = (pnl_series > 0).astype(int)
                losses = (pnl_series < 0).astype(int)
                
                consecutive_wins = self._max_consecutive(wins)
                consecutive_losses = self._max_consecutive(losses)
                
                largest_win = pnl_series.max()
                largest_loss = pnl_series.min()
            else:
                consecutive_wins = 0
                consecutive_losses = 0
                largest_win = 0
                largest_loss = 0
            
            return {
                'trades_per_day': trades_per_day,
                'avg_hold_time': avg_hold_time,
                'turnover_rate': 0,  # Would need position data
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}
    
    def calculate_monthly_returns(self, trades: List[Dict[str, Any]], 
                                 initial_capital: float = 10000.0) -> Dict[str, float]:
        """
        Calculate monthly returns
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Starting capital
            
        Returns:
            Dictionary of monthly returns
        """
        try:
            if not trades:
                return {}
            
            df = pd.DataFrame(trades)
            
            # Convert timestamps and extract P&L
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
            else:
                return {}
            
            if 'pnl' in df.columns:
                df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
            else:
                return {}
            
            # Group by month and sum P&L
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_pnl = df.groupby('month')['pnl'].sum()
            
            # Convert to returns
            monthly_returns = {}
            cumulative_capital = initial_capital
            
            for month, pnl in monthly_pnl.items():
                monthly_return = pnl / cumulative_capital
                monthly_returns[str(month)] = monthly_return
                cumulative_capital += pnl
            
            return monthly_returns
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return {}
    
    def calculate_portfolio_metrics(self, account_info: Dict[str, Any], 
                                   positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate current portfolio metrics
        
        Args:
            account_info: Account information
            positions: Current positions
            
        Returns:
            Dictionary of portfolio metrics
        """
        try:
            total_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            
            # Position metrics
            num_positions = len(positions)
            total_market_value = sum(abs(float(pos.get('market_value', 0))) for pos in positions)
            total_unrealized_pl = sum(float(pos.get('unrealized_pl', 0)) for pos in positions)
            
            # Exposure metrics
            long_exposure = sum(float(pos.get('market_value', 0)) 
                              for pos in positions 
                              if float(pos.get('market_value', 0)) > 0)
            
            short_exposure = sum(abs(float(pos.get('market_value', 0))) 
                               for pos in positions 
                               if float(pos.get('market_value', 0)) < 0)
            
            # Portfolio ratios
            cash_ratio = cash / total_value if total_value > 0 else 0
            exposure_ratio = total_market_value / total_value if total_value > 0 else 0
            
            return {
                'total_portfolio_value': total_value,
                'cash': cash,
                'cash_ratio': cash_ratio,
                'num_positions': num_positions,
                'total_market_value': total_market_value,
                'total_unrealized_pl': total_unrealized_pl,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_exposure': long_exposure - short_exposure,
                'exposure_ratio': exposure_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive occurrences"""
        if len(series) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in series:
            if value == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary"""
        return {
            'total_return': 0,
            'total_pnl': 0,
            'annualized_return': 0,
            'num_trades': 0,
            'num_winning': 0,
            'num_losing': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'final_equity': 0
        }
    
    def _empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics dictionary"""
        return {
            'value_at_risk_95': 0,
            'value_at_risk_99': 0,
            'expected_shortfall': 0,
            'current_exposure': 0,
            'max_position_size': 0
        }
    
    def calculate_benchmark_comparison(self, trades: List[Dict[str, Any]], 
                                     benchmark_returns: List[float],
                                     initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Compare strategy performance to benchmark
        
        Args:
            trades: List of trade dictionaries
            benchmark_returns: List of benchmark returns
            initial_capital: Starting capital
            
        Returns:
            Comparison metrics
        """
        try:
            if not trades or not benchmark_returns:
                return {}
            
            # Calculate strategy returns
            strategy_metrics = self.calculate_returns_metrics(trades, initial_capital)
            strategy_return = strategy_metrics.get('total_return', 0)
            
            # Calculate benchmark metrics
            benchmark_total_return = sum(benchmark_returns)
            benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
            benchmark_sharpe = np.mean(benchmark_returns) * 252 / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Comparison metrics
            excess_return = strategy_return - benchmark_total_return
            strategy_sharpe = strategy_metrics.get('sharpe_ratio', 0)
            
            # Information ratio (excess return / tracking error)
            # Simplified calculation
            tracking_error = abs(strategy_return - benchmark_total_return)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            return {
                'strategy_return': strategy_return,
                'benchmark_return': benchmark_total_return,
                'excess_return': excess_return,
                'strategy_sharpe': strategy_sharpe,
                'benchmark_sharpe': benchmark_sharpe,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            }
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}