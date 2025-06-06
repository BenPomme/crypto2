"""
Advanced Performance Metrics Calculator
Comprehensive analysis of trading strategy performance
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Advanced performance analysis for trading strategies
    Provides comprehensive metrics, statistical analysis, and visualizations
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate
        
    def analyze_backtest_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of backtest results
        
        Args:
            backtest_results: Results from BacktestEngine
            
        Returns:
            Dictionary with detailed performance analysis
        """
        if not backtest_results.get('backtest_valid', False):
            return self._empty_analysis()
        
        trades = backtest_results.get('trades', [])
        equity_curve = backtest_results.get('equity_curve', [])
        
        if not trades or not equity_curve:
            return self._empty_analysis()
        
        # Convert to pandas for analysis
        trades_df = self._trades_to_dataframe(trades)
        equity_df = self._equity_to_dataframe(equity_curve)
        
        analysis = {
            # Core metrics from backtest
            **self._extract_core_metrics(backtest_results),
            
            # Advanced return metrics
            **self._calculate_return_metrics(trades_df, equity_df, backtest_results),
            
            # Risk metrics
            **self._calculate_risk_metrics(trades_df, equity_df),
            
            # Trading behavior metrics
            **self._calculate_trading_metrics(trades_df),
            
            # Statistical analysis
            **self._calculate_statistical_metrics(trades_df, equity_df),
            
            # Market exposure metrics
            **self._calculate_exposure_metrics(trades_df, equity_df),
            
            # Consistency metrics
            **self._calculate_consistency_metrics(trades_df, equity_df),
            
            # Raw data for further analysis
            'trades_df': trades_df,
            'equity_df': equity_df
        }
        
        # Add interpretations and scores
        analysis['performance_score'] = self._calculate_performance_score(analysis)
        analysis['risk_score'] = self._calculate_risk_score(analysis)
        analysis['interpretation'] = self._generate_interpretation(analysis)
        
        return analysis
    
    def _extract_core_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract core metrics from backtest results"""
        return {
            'initial_capital': backtest_results.get('initial_capital', 0),
            'final_equity': backtest_results.get('final_equity', 0),
            'total_return': backtest_results.get('total_return', 0),
            'total_pnl': backtest_results.get('total_pnl', 0),
            'max_drawdown': backtest_results.get('max_drawdown', 0),
            'total_trades': backtest_results.get('total_trades', 0),
            'win_rate': backtest_results.get('win_rate', 0),
            'total_commission': backtest_results.get('total_commission', 0),
            'total_slippage': backtest_results.get('total_slippage', 0)
        }
    
    def _calculate_return_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame, backtest_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive return metrics"""
        if equity_df.empty:
            return {}
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Annualized metrics
        trading_days = len(equity_df)
        years = trading_days / 252  # Assume 252 trading days per year
        
        total_return = backtest_results.get('total_return', 0)
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe and Sortino ratios
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Calmar ratio
        max_drawdown = backtest_results.get('max_drawdown', 0)
        calmar_ratio = annualized_return / (max_drawdown + 1e-10)
        
        # Risk-adjusted returns
        risk_adjusted_return = annualized_return / (annual_volatility + 1e-10)
        
        return {
            'annualized_return': annualized_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'risk_adjusted_return': risk_adjusted_return,
            'trading_period_years': years
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        if equity_df.empty:
            return {}
        
        returns = equity_df['equity'].pct_change().dropna()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns) > 0 else 0
        
        # Maximum consecutive losses
        if not trades_df.empty:
            trades_df['is_loss'] = trades_df['pnl'] < 0
            max_consecutive_losses = self._max_consecutive_true(trades_df['is_loss'])
        else:
            max_consecutive_losses = 0
        
        # Drawdown analysis
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown_series = (equity_series - running_max) / running_max
        
        avg_drawdown = drawdown_series[drawdown_series < 0].mean()
        drawdown_duration = self._calculate_avg_drawdown_duration(drawdown_series)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'conditional_var_95': cvar_95,
            'conditional_var_99': cvar_99,
            'max_consecutive_losses': max_consecutive_losses,
            'average_drawdown': avg_drawdown,
            'average_drawdown_duration_days': drawdown_duration,
            'downside_deviation': downside_deviation
        }
    
    def _calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading behavior metrics"""
        if trades_df.empty:
            return {}
        
        # Basic trading stats
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 1e-10
        profit_factor = gross_profit / gross_loss
        
        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Hold time analysis
        trades_df['hold_time_hours'] = trades_df['hold_time'].dt.total_seconds() / 3600
        avg_hold_time = trades_df['hold_time_hours'].mean()
        median_hold_time = trades_df['hold_time_hours'].median()
        
        # Win/loss streaks
        trades_df['is_win'] = trades_df['pnl'] > 0
        max_win_streak = self._max_consecutive_true(trades_df['is_win'])
        max_loss_streak = self._max_consecutive_true(~trades_df['is_win'])
        
        # Trade size consistency
        trade_size_std = trades_df['quantity'].std()
        trade_size_cv = trade_size_std / trades_df['quantity'].mean() if trades_df['quantity'].mean() > 0 else 0
        
        return {
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'average_hold_time_hours': avg_hold_time,
            'median_hold_time_hours': median_hold_time,
            'max_winning_streak': max_win_streak,
            'max_losing_streak': max_loss_streak,
            'trade_size_consistency': 1 / (1 + trade_size_cv)  # Higher is more consistent
        }
    
    def _calculate_statistical_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance metrics"""
        if trades_df.empty or equity_df.empty:
            return {}
        
        returns = equity_df['equity'].pct_change().dropna()
        pnl_series = trades_df['pnl']
        
        # Statistical tests
        if len(returns) > 10:
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Limit for computational efficiency
            is_normal = shapiro_p > 0.05
            
            # Skewness and kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
        else:
            shapiro_stat = shapiro_p = skewness = kurtosis = 0
            is_normal = False
        
        # T-test for PnL significance
        if len(pnl_series) > 1:
            t_stat, t_p_value = stats.ttest_1samp(pnl_series, 0)
            is_significantly_profitable = t_p_value < 0.05 and pnl_series.mean() > 0
        else:
            t_stat = t_p_value = 0
            is_significantly_profitable = False
        
        # Confidence intervals
        if len(pnl_series) > 1:
            mean_pnl = pnl_series.mean()
            sem_pnl = stats.sem(pnl_series)
            ci_95 = stats.t.interval(0.95, len(pnl_series)-1, mean_pnl, sem_pnl)
        else:
            ci_95 = (0, 0)
        
        return {
            'returns_skewness': skewness,
            'returns_kurtosis': kurtosis,
            'returns_normal_distribution': is_normal,
            'shapiro_test_p_value': shapiro_p,
            't_test_statistic': t_stat,
            't_test_p_value': t_p_value,
            'significantly_profitable': is_significantly_profitable,
            'pnl_confidence_interval_95': ci_95
        }
    
    def _calculate_exposure_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market exposure metrics"""
        if trades_df.empty or equity_df.empty:
            return {}
        
        # Calculate time in market
        total_time = equity_df.index[-1] - equity_df.index[0]
        total_hold_time = trades_df['hold_time'].sum()
        
        market_exposure = total_hold_time / total_time if total_time.total_seconds() > 0 else 0
        
        # Trade frequency
        trades_per_day = len(trades_df) / (total_time.total_seconds() / 86400) if total_time.total_seconds() > 0 else 0
        
        return {
            'market_exposure_ratio': market_exposure,
            'trades_per_day': trades_per_day,
            'total_trading_days': total_time.total_seconds() / 86400
        }
    
    def _calculate_consistency_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate strategy consistency metrics"""
        if trades_df.empty or equity_df.empty:
            return {}
        
        # Monthly returns consistency
        if len(equity_df) > 30:
            equity_df['month'] = equity_df.index.to_period('M')
            monthly_returns = equity_df.groupby('month')['equity'].apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0
            )
            
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            monthly_win_rate = positive_months / total_months if total_months > 0 else 0
            
            monthly_returns_std = monthly_returns.std()
            monthly_consistency = 1 / (1 + monthly_returns_std) if monthly_returns_std > 0 else 0
        else:
            monthly_win_rate = monthly_consistency = 0
        
        # Rolling Sharpe ratio stability
        if len(equity_df) > 50:
            returns = equity_df['equity'].pct_change().dropna()
            rolling_sharpe = returns.rolling(30).apply(
                lambda x: self._calculate_sharpe_ratio(x) if len(x) == 30 else np.nan
            ).dropna()
            
            sharpe_stability = 1 - (rolling_sharpe.std() / (abs(rolling_sharpe.mean()) + 1e-10))
        else:
            sharpe_stability = 0
        
        return {
            'monthly_win_rate': monthly_win_rate,
            'monthly_consistency_score': monthly_consistency,
            'sharpe_ratio_stability': sharpe_stability
        }
    
    def _calculate_performance_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)"""
        # Weighted scoring of key metrics
        scores = []
        
        # Return component (30%)
        total_return = analysis.get('total_return', 0)
        return_score = min(max(total_return * 100, -50), 100)  # Cap at -50% to 100%
        scores.append(('return', return_score, 0.3))
        
        # Risk component (25%)
        sharpe_ratio = analysis.get('sharpe_ratio', 0)
        risk_score = min(max(sharpe_ratio * 20, -20), 60)  # Convert Sharpe to 0-60 scale
        scores.append(('risk', risk_score, 0.25))
        
        # Win rate component (20%)
        win_rate = analysis.get('win_rate', 0)
        win_score = win_rate * 100
        scores.append(('win_rate', win_score, 0.2))
        
        # Drawdown component (15%)
        max_drawdown = analysis.get('max_drawdown', 1)
        drawdown_score = max(0, 100 * (1 - max_drawdown * 2))  # Penalize drawdowns
        scores.append(('drawdown', drawdown_score, 0.15))
        
        # Consistency component (10%)
        monthly_win_rate = analysis.get('monthly_win_rate', 0)
        consistency_score = monthly_win_rate * 100
        scores.append(('consistency', consistency_score, 0.1))
        
        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        return max(0, min(100, weighted_score))
    
    def _calculate_risk_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate risk score (0-100, higher is riskier)"""
        risk_factors = []
        
        # Drawdown risk
        max_drawdown = analysis.get('max_drawdown', 0)
        drawdown_risk = min(max_drawdown * 200, 100)  # 50% drawdown = 100 risk
        risk_factors.append(drawdown_risk)
        
        # Volatility risk
        annual_volatility = analysis.get('annual_volatility', 0)
        volatility_risk = min(annual_volatility * 100, 100)  # 100% volatility = 100 risk
        risk_factors.append(volatility_risk)
        
        # Consecutive loss risk
        max_consecutive_losses = analysis.get('max_consecutive_losses', 0)
        loss_streak_risk = min(max_consecutive_losses * 10, 100)  # 10 losses = 100 risk
        risk_factors.append(loss_streak_risk)
        
        return np.mean(risk_factors)
    
    def _generate_interpretation(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable interpretation of results"""
        interpretations = {}
        
        # Overall performance
        performance_score = analysis.get('performance_score', 0)
        if performance_score >= 80:
            interpretations['overall'] = "Excellent strategy performance"
        elif performance_score >= 60:
            interpretations['overall'] = "Good strategy performance" 
        elif performance_score >= 40:
            interpretations['overall'] = "Moderate strategy performance"
        else:
            interpretations['overall'] = "Poor strategy performance"
        
        # Risk assessment
        risk_score = analysis.get('risk_score', 0)
        if risk_score <= 20:
            interpretations['risk'] = "Low risk strategy"
        elif risk_score <= 40:
            interpretations['risk'] = "Moderate risk strategy"
        elif risk_score <= 60:
            interpretations['risk'] = "High risk strategy"
        else:
            interpretations['risk'] = "Very high risk strategy"
        
        # Profitability
        total_return = analysis.get('total_return', 0)
        win_rate = analysis.get('win_rate', 0)
        
        if total_return > 0.2 and win_rate > 0.6:
            interpretations['profitability'] = "Highly profitable with good consistency"
        elif total_return > 0.1:
            interpretations['profitability'] = "Profitable strategy"
        elif total_return > 0:
            interpretations['profitability'] = "Marginally profitable"
        else:
            interpretations['profitability'] = "Unprofitable strategy"
        
        return interpretations
    
    # Helper methods
    def _trades_to_dataframe(self, trades: List) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        if not trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'hold_time': trade.hold_time,
                'commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)
    
    def _equity_to_dataframe(self, equity_curve: List[Tuple]) -> pd.DataFrame:
        """Convert equity curve to DataFrame"""
        if not equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) <= 1 or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return excess_returns / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) <= 1:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return excess_returns / downside_returns.std() * np.sqrt(252)
    
    def _max_consecutive_true(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        if series.empty:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_avg_drawdown_duration(self, drawdown_series: pd.Series) -> float:
        """Calculate average drawdown duration in days"""
        if drawdown_series.empty:
            return 0
        
        in_drawdown = drawdown_series < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return np.mean(drawdown_periods) if drawdown_periods else 0
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis for invalid backtests"""
        return {
            'performance_score': 0,
            'risk_score': 100,
            'interpretation': {'overall': 'Invalid backtest results'},
            'error': 'No valid trades or equity data'
        }