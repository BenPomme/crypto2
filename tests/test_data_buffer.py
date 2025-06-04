"""
Tests for DataBuffer module
"""
import pytest
import pandas as pd
from datetime import datetime
from src.data.data_buffer import DataBuffer

def test_data_buffer_initialization():
    """Test DataBuffer initialization"""
    buffer = DataBuffer(max_size=100)
    assert buffer.max_size == 100
    assert buffer.size() == 0
    assert not buffer.is_ready(10)

def test_add_bar():
    """Test adding a single bar to buffer"""
    buffer = DataBuffer(max_size=10)
    
    bar_data = {
        'timestamp': datetime.now(),
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000.0
    }
    
    buffer.add_bar(bar_data)
    assert buffer.size() == 1
    
    latest = buffer.get_latest_bar()
    assert latest['close'] == 102.0
    assert buffer.get_latest_price() == 102.0

def test_bulk_add(sample_ohlcv_data):
    """Test adding multiple bars from DataFrame"""
    buffer = DataBuffer(max_size=1000)
    
    # Reset index to have timestamp as column
    df = sample_ohlcv_data.reset_index()
    
    buffer.bulk_add(df)
    assert buffer.size() == len(df)
    
    # Test DataFrame conversion
    result_df = buffer.get_dataframe()
    assert len(result_df) == len(df)
    assert 'close' in result_df.columns

def test_buffer_max_size():
    """Test buffer respects maximum size"""
    buffer = DataBuffer(max_size=5)
    
    # Add more bars than max size
    for i in range(10):
        bar_data = {
            'timestamp': datetime.now(),
            'open': 100.0 + i,
            'high': 105.0 + i,
            'low': 95.0 + i,
            'close': 102.0 + i,
            'volume': 1000.0
        }
        buffer.add_bar(bar_data)
    
    # Should only keep last 5
    assert buffer.size() == 5
    
    # Latest should be the last added
    latest = buffer.get_latest_bar()
    assert latest['close'] == 102.0 + 9

def test_get_price_series():
    """Test getting price series"""
    buffer = DataBuffer(max_size=10)
    
    # Add some bars
    for i in range(5):
        bar_data = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 100.0 + i,
            'volume': 1000.0
        }
        buffer.add_bar(bar_data)
    
    prices = buffer.get_price_series()
    assert len(prices) == 5
    assert prices.iloc[-1] == 104.0  # Last close price

def test_buffer_stats():
    """Test buffer statistics"""
    buffer = DataBuffer(max_size=10)
    
    # Empty buffer stats
    stats = buffer.get_stats()
    assert stats['size'] == 0
    assert stats['max_size'] == 10
    assert not stats['is_full']
    
    # Add some bars
    for i in range(3):
        bar_data = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 100.0 + i,
            'volume': 1000.0
        }
        buffer.add_bar(bar_data)
    
    stats = buffer.get_stats()
    assert stats['size'] == 3
    assert not stats['is_full']
    assert stats['latest_price'] == 102.0

def test_is_ready():
    """Test readiness check"""
    buffer = DataBuffer(max_size=10)
    
    assert not buffer.is_ready(5)
    
    # Add 5 bars
    for i in range(5):
        bar_data = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 100.0 + i,
            'volume': 1000.0
        }
        buffer.add_bar(bar_data)
    
    assert buffer.is_ready(5)
    assert not buffer.is_ready(10)

def test_clear():
    """Test buffer clearing"""
    buffer = DataBuffer(max_size=10)
    
    # Add some bars
    for i in range(3):
        bar_data = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 100.0 + i,
            'volume': 1000.0
        }
        buffer.add_bar(bar_data)
    
    assert buffer.size() == 3
    
    buffer.clear()
    assert buffer.size() == 0
    assert buffer.get_latest_bar() is None