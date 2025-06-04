"""
Tests for MACrossoverStrategy
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.strategy.base_strategy import SignalType

def test_strategy_initialization():
    """Test strategy initialization"""
    config = {
        'fast_ma_period': 10,
        'slow_ma_period': 20,
        'ma_type': 'sma'
    }
    
    strategy = MACrossoverStrategy(config)
    assert strategy.name == "MA_Crossover"
    assert strategy.fast_period == 10
    assert strategy.slow_period == 20
    assert strategy.ma_type == 'sma'

def test_required_indicators():
    """Test required indicators"""
    strategy = MACrossoverStrategy()
    required = strategy.get_required_indicators()
    
    assert 'sma_fast' in required
    assert 'sma_slow' in required
    assert 'rsi' in required

def test_generate_signal_golden_cross(sample_ohlcv_data):
    """Test golden cross signal generation"""
    strategy = MACrossoverStrategy({
        'fast_ma_period': 5,
        'slow_ma_period': 10,
        'min_confidence': 0.5
    })
    
    # Create test data with golden cross
    data = sample_ohlcv_data.copy()
    
    # Add moving averages manually for testing
    data['sma_fast'] = data['close'].rolling(5).mean()
    data['sma_slow'] = data['close'].rolling(10).mean()
    data['rsi'] = 50.0  # Neutral RSI
    data['volume_ratio'] = 1.5  # High volume
    
    # Create golden cross scenario
    # Fast MA crosses above slow MA
    data.iloc[-2, data.columns.get_loc('sma_fast')] = 49900  # Below slow MA
    data.iloc[-2, data.columns.get_loc('sma_slow')] = 50000
    
    data.iloc[-1, data.columns.get_loc('sma_fast')] = 50100  # Above slow MA
    data.iloc[-1, data.columns.get_loc('sma_slow')] = 50000
    
    signal = strategy.generate_signal(data)
    
    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.symbol == 'BTCUSD'
    assert signal.confidence > 0.5

def test_generate_signal_death_cross(sample_ohlcv_data):
    """Test death cross signal generation"""
    strategy = MACrossoverStrategy({
        'fast_ma_period': 5,
        'slow_ma_period': 10,
        'exit_on_reverse_cross': True
    })
    
    # Set strategy to have a long position
    strategy.current_position = 1
    strategy.entry_price = 50000.0
    
    # Create test data with death cross
    data = sample_ohlcv_data.copy()
    
    # Add moving averages manually for testing
    data['sma_fast'] = data['close'].rolling(5).mean()
    data['sma_slow'] = data['close'].rolling(10).mean()
    data['rsi'] = 50.0
    
    # Create death cross scenario
    # Fast MA crosses below slow MA
    data.iloc[-2, data.columns.get_loc('sma_fast')] = 50100  # Above slow MA
    data.iloc[-2, data.columns.get_loc('sma_slow')] = 50000
    
    data.iloc[-1, data.columns.get_loc('sma_fast')] = 49900  # Below slow MA
    data.iloc[-1, data.columns.get_loc('sma_slow')] = 50000
    
    signal = strategy.generate_signal(data)
    
    assert signal is not None
    assert signal.signal_type == SignalType.CLOSE_LONG

def test_no_signal_when_no_crossover(sample_ohlcv_data):
    """Test no signal when no crossover occurs"""
    strategy = MACrossoverStrategy()
    
    data = sample_ohlcv_data.copy()
    
    # Add moving averages with no crossover
    data['sma_fast'] = 50100.0  # Consistently above slow MA
    data['sma_slow'] = 50000.0
    data['rsi'] = 50.0
    data['volume_ratio'] = 1.0
    
    signal = strategy.generate_signal(data)
    assert signal is None

def test_insufficient_data():
    """Test strategy with insufficient data"""
    strategy = MACrossoverStrategy()
    
    # Create minimal dataset
    data = pd.DataFrame({
        'close': [100, 101, 102],
        'sma_fast': [100, 101, 102],
        'sma_slow': [100, 101, 102],
        'rsi': [50, 50, 50]
    })
    
    signal = strategy.generate_signal(data)
    assert signal is None  # Not enough data

def test_position_tracking():
    """Test position tracking functionality"""
    strategy = MACrossoverStrategy()
    
    # Initially flat
    assert strategy.is_flat()
    assert not strategy.is_long()
    assert not strategy.is_short()
    
    # Simulate buy signal execution
    from src.strategy.base_strategy import TradingSignal
    buy_signal = TradingSignal(
        signal_type=SignalType.BUY,
        symbol='BTCUSD',
        timestamp=datetime.now(),
        price=50000.0,
        confidence=0.8
    )
    
    strategy.update_position(buy_signal)
    
    assert strategy.is_long()
    assert strategy.entry_price == 50000.0
    
    # Test P&L calculation
    current_price = 51000.0
    pnl = strategy.get_unrealized_pnl(current_price)
    assert pnl == 1000.0
    
    pnl_pct = strategy.get_unrealized_pnl_pct(current_price)
    assert pnl_pct == 2.0

def test_rsi_filters(sample_ohlcv_data):
    """Test RSI confirmation filters"""
    strategy = MACrossoverStrategy({
        'fast_ma_period': 5,
        'slow_ma_period': 10,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    })
    
    data = sample_ohlcv_data.copy()
    
    # Add golden cross setup
    data['sma_fast'] = data['close'].rolling(5).mean()
    data['sma_slow'] = data['close'].rolling(10).mean()
    data['volume_ratio'] = 1.5
    
    # Create golden cross
    data.iloc[-2, data.columns.get_loc('sma_fast')] = 49900
    data.iloc[-2, data.columns.get_loc('sma_slow')] = 50000
    data.iloc[-1, data.columns.get_loc('sma_fast')] = 50100
    data.iloc[-1, data.columns.get_loc('sma_slow')] = 50000
    
    # Test with oversold RSI (should increase confidence)
    data['rsi'] = 25.0
    signal = strategy.generate_signal(data)
    oversold_confidence = signal.confidence if signal else 0
    
    # Test with overbought RSI (should decrease confidence)
    data['rsi'] = 75.0
    signal = strategy.generate_signal(data)
    overbought_confidence = signal.confidence if signal else 0
    
    # Oversold should have higher confidence than overbought
    assert oversold_confidence > overbought_confidence

def test_volume_confirmation(sample_ohlcv_data):
    """Test volume confirmation filter"""
    strategy = MACrossoverStrategy({
        'fast_ma_period': 5,
        'slow_ma_period': 10,
        'volume_confirmation': True,
        'volume_threshold': 1.5
    })
    
    data = sample_ohlcv_data.copy()
    
    # Add golden cross setup
    data['sma_fast'] = data['close'].rolling(5).mean()
    data['sma_slow'] = data['close'].rolling(10).mean()
    data['rsi'] = 50.0
    
    # Create golden cross
    data.iloc[-2, data.columns.get_loc('sma_fast')] = 49900
    data.iloc[-2, data.columns.get_loc('sma_slow')] = 50000
    data.iloc[-1, data.columns.get_loc('sma_fast')] = 50100
    data.iloc[-1, data.columns.get_loc('sma_slow')] = 50000
    
    # Test with high volume (should increase confidence)
    data['volume_ratio'] = 2.0
    signal = strategy.generate_signal(data)
    high_volume_confidence = signal.confidence if signal else 0
    
    # Test with low volume (should decrease confidence)
    data['volume_ratio'] = 0.3
    signal = strategy.generate_signal(data)
    low_volume_confidence = signal.confidence if signal else 0
    
    # High volume should have higher confidence
    assert high_volume_confidence > low_volume_confidence

def test_strategy_reset():
    """Test strategy reset functionality"""
    strategy = MACrossoverStrategy()
    
    # Set some state
    strategy.current_position = 1
    strategy.entry_price = 50000.0
    strategy.signals_generated = 5
    strategy.trades_executed = 3
    
    # Reset
    strategy.reset()
    
    assert strategy.current_position == 0
    assert strategy.entry_price is None
    assert strategy.signals_generated == 0
    assert strategy.trades_executed == 0

def test_strategy_stats():
    """Test strategy statistics"""
    strategy = MACrossoverStrategy()
    
    stats = strategy.get_strategy_stats()
    
    assert 'name' in stats
    assert 'signals_generated' in stats
    assert 'trades_executed' in stats
    assert 'current_position' in stats
    assert stats['name'] == "MA_Crossover"