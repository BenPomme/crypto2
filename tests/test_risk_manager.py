"""
Tests for RiskManager module
"""
import pytest
from datetime import datetime
from src.risk.risk_manager import RiskManager, RiskLevel
from src.strategy.base_strategy import TradingSignal, SignalType

def test_risk_manager_initialization():
    """Test RiskManager initialization"""
    config = {
        'max_position_size': 0.2,
        'max_daily_loss': 0.03,
        'risk_per_trade': 0.015
    }
    
    risk_manager = RiskManager(config)
    assert risk_manager.config['max_position_size'] == 0.2
    assert risk_manager.config['max_daily_loss'] == 0.03
    assert risk_manager.config['risk_per_trade'] == 0.015

def test_evaluate_signal_approval(sample_account_info, sample_positions):
    """Test signal evaluation and approval"""
    risk_manager = RiskManager()
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8,
        reason="Test signal"
    )
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=[],  # No current positions
        market_data=None
    )
    
    assert result.signal == signal
    assert result.approved  # Should be approved with no risk violations
    assert result.position_size is not None
    assert len(result.checks) > 0

def test_position_limit_check(sample_account_info):
    """Test position limit checks"""
    config = {
        'max_concurrent_positions': 2,
        'max_same_asset_positions': 1
    }
    risk_manager = RiskManager(config)
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    # Too many concurrent positions
    current_positions = [
        {'symbol': 'ETHUSD', 'qty': 1.0, 'market_value': 3000.0},
        {'symbol': 'ADAUSD', 'qty': 100.0, 'market_value': 2000.0},
        {'symbol': 'SOLUSD', 'qty': 10.0, 'market_value': 1500.0}
    ]
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=current_positions
    )
    
    # Should be rejected due to too many positions
    position_check = next((check for check in result.checks 
                          if check.name == 'max_concurrent_positions'), None)
    assert position_check is not None
    assert not position_check.passed

def test_same_asset_limit(sample_account_info):
    """Test same asset position limit"""
    config = {'max_positions_same_asset': 1}
    risk_manager = RiskManager(config)
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    # Already have position in same asset
    current_positions = [
        {'symbol': 'BTCUSD', 'qty': 0.1, 'market_value': 5000.0}
    ]
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=current_positions
    )
    
    # Should be rejected due to existing position in same asset
    asset_check = next((check for check in result.checks 
                       if check.name == 'max_same_asset_positions'), None)
    assert asset_check is not None
    assert not asset_check.passed

def test_daily_loss_limit(sample_account_info):
    """Test daily loss limit"""
    config = {'max_daily_loss': 0.02}  # 2% max daily loss
    risk_manager = RiskManager(config)
    
    # Simulate daily loss
    account_value = sample_account_info['portfolio_value']
    daily_loss = account_value * 0.025  # 2.5% loss (exceeds limit)
    
    today = datetime.now().strftime('%Y-%m-%d')
    risk_manager.daily_pnl[today] = -daily_loss
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=[]
    )
    
    # Should be rejected due to daily loss limit
    loss_check = next((check for check in result.checks 
                      if check.name == 'daily_loss_limit'), None)
    assert loss_check is not None
    assert not loss_check.passed

def test_consecutive_losses_limit(sample_account_info):
    """Test consecutive losses limit"""
    config = {'consecutive_loss_limit': 3}
    risk_manager = RiskManager(config)
    
    # Simulate consecutive losses
    risk_manager.consecutive_losses = 4  # Exceeds limit of 3
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=[]
    )
    
    # Should be rejected due to consecutive losses
    loss_check = next((check for check in result.checks 
                      if check.name == 'consecutive_losses'), None)
    assert loss_check is not None
    assert not loss_check.passed

def test_drawdown_limit(sample_account_info):
    """Test maximum drawdown limit"""
    config = {'max_drawdown': 0.1}  # 10% max drawdown
    risk_manager = RiskManager(config)
    
    # Set peak equity and simulate drawdown
    account_value = sample_account_info['portfolio_value']
    risk_manager.peak_equity = account_value * 1.2  # Previous peak was 20% higher
    risk_manager.current_drawdown = 0.15  # Current drawdown is 15%
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=[]
    )
    
    # Should be rejected due to excessive drawdown
    dd_check = next((check for check in result.checks 
                    if check.name == 'max_drawdown'), None)
    assert dd_check is not None
    assert not dd_check.passed

def test_update_trade_result():
    """Test updating trade results"""
    risk_manager = RiskManager()
    
    # Test winning trade
    trade_time = datetime.now()
    risk_manager.update_trade_result(100.0, trade_time)
    
    assert len(risk_manager.recent_trades) == 1
    assert risk_manager.consecutive_losses == 0
    
    # Test losing trade
    risk_manager.update_trade_result(-50.0, trade_time)
    
    assert len(risk_manager.recent_trades) == 2
    assert risk_manager.consecutive_losses == 1
    
    # Test another losing trade
    risk_manager.update_trade_result(-30.0, trade_time)
    
    assert risk_manager.consecutive_losses == 2

def test_cash_availability_check(sample_account_info):
    """Test cash availability check"""
    risk_manager = RiskManager()
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    # Modify account to have very little cash
    low_cash_account = sample_account_info.copy()
    low_cash_account['cash'] = 100.0  # Very low cash
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=low_cash_account,
        current_positions=[]
    )
    
    # Should either be rejected or have very small position size
    if result.position_size:
        assert result.position_size.size_usd <= low_cash_account['cash']

def test_risk_status():
    """Test risk status reporting"""
    risk_manager = RiskManager()
    
    # Add some trade history
    risk_manager.consecutive_losses = 2
    risk_manager.current_drawdown = 0.05
    risk_manager.peak_equity = 10000.0
    
    today = datetime.now().strftime('%Y-%m-%d')
    risk_manager.daily_pnl[today] = -200.0
    
    status = risk_manager.get_risk_status()
    
    assert 'consecutive_losses' in status
    assert 'current_drawdown' in status
    assert 'peak_equity' in status
    assert 'daily_pnl' in status
    assert status['consecutive_losses'] == 2
    assert status['current_drawdown'] == 0.05

def test_risk_level_escalation(sample_account_info):
    """Test risk level escalation with multiple violations"""
    config = {
        'max_concurrent_positions': 1,
        'max_daily_loss': 0.01,  # Very low limit
        'consecutive_loss_limit': 1  # Very low limit
    }
    risk_manager = RiskManager(config)
    
    # Setup multiple violations
    risk_manager.consecutive_losses = 2
    today = datetime.now().strftime('%Y-%m-%d')
    risk_manager.daily_pnl[today] = -500.0  # Exceeds daily limit
    
    signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    current_positions = [
        {'symbol': 'ETHUSD', 'qty': 1.0, 'market_value': 3000.0}
    ]
    
    result = risk_manager.evaluate_signal(
        signal=signal,
        account_info=sample_account_info,
        current_positions=current_positions
    )
    
    # Should be rejected with high risk level
    assert not result.approved
    assert result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]