"""
Pytest configuration and fixtures for the trading system tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # Generate realistic crypto price data
    initial_price = 50000.0
    price_changes = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility
    prices = initial_price * np.exp(np.cumsum(price_changes))
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close
        volatility = close * 0.005  # 0.5% intraday volatility
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = low + np.random.uniform(0, high - low)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def sample_trade_data():
    """Generate sample trade data for testing"""
    trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        trade_time = base_time + timedelta(hours=i * 12)
        pnl = np.random.normal(10, 50)  # Random P&L with slight positive bias
        
        trades.append({
            'timestamp': trade_time.isoformat(),
            'symbol': 'BTCUSD',
            'signal_type': 'BUY' if i % 2 == 0 else 'SELL',
            'entry_price': 50000 + np.random.normal(0, 1000),
            'pnl': pnl,
            'success': True,
            'metadata': {}
        })
    
    return trades

@pytest.fixture
def sample_account_info():
    """Sample account information for testing"""
    return {
        'account_id': 'test_account',
        'buying_power': 10000.0,
        'cash': 8000.0,
        'portfolio_value': 12000.0,
        'pattern_day_trader': False,
        'trading_blocked': False,
        'transfers_blocked': False
    }

@pytest.fixture
def sample_positions():
    """Sample positions for testing"""
    return [
        {
            'symbol': 'BTCUSD',
            'qty': 0.2,
            'market_value': 10000.0,
            'avg_entry_price': 50000.0,
            'unrealized_pl': 500.0,
            'unrealized_plpc': 0.05,
            'side': 'long'
        }
    ]

@pytest.fixture
def mock_firebase_logger():
    """Mock Firebase logger for testing"""
    class MockFirebaseLogger:
        def __init__(self):
            self.logs = []
            self.initialized = True
        
        def log_trade(self, trade_data):
            self.logs.append(('trade', trade_data))
            return True
        
        def log_signal(self, signal_data):
            self.logs.append(('signal', signal_data))
            return True
        
        def log_performance(self, performance_data):
            self.logs.append(('performance', performance_data))
            return True
        
        def update_system_status(self, status_data):
            self.logs.append(('status', status_data))
            return True
        
        def is_connected(self):
            return True
    
    return MockFirebaseLogger()

@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API for testing"""
    class MockAlpacaAPI:
        def __init__(self):
            self.account_info = {
                'id': 'test_account',
                'buying_power': 10000.0,
                'cash': 8000.0,
                'portfolio_value': 12000.0,
                'pattern_day_trader': False,
                'trading_blocked': False,
                'transfers_blocked': False
            }
            self.positions = []
            self.orders = []
        
        def get_account(self):
            from types import SimpleNamespace
            return SimpleNamespace(**self.account_info)
        
        def list_positions(self):
            return [SimpleNamespace(**pos) for pos in self.positions]
        
        def submit_order(self, **kwargs):
            from types import SimpleNamespace
            order = SimpleNamespace(
                id=f"order_{len(self.orders)}",
                symbol=kwargs['symbol'],
                qty=kwargs['qty'],
                side=kwargs['side'],
                type=kwargs['type'],
                status='new',
                client_order_id=kwargs.get('client_order_id'),
                created_at=datetime.now()
            )
            self.orders.append(order)
            return order
        
        def get_order(self, order_id):
            for order in self.orders:
                if order.id == order_id:
                    # Simulate filled order
                    order.status = 'filled'
                    order.filled_avg_price = 50000.0
                    order.filled_qty = order.qty
                    return order
            raise ValueError(f"Order {order_id} not found")
        
        def cancel_order(self, order_id):
            for order in self.orders:
                if order.id == order_id:
                    order.status = 'canceled'
                    return
            raise ValueError(f"Order {order_id} not found")
        
        def get_crypto_bars(self, symbol, timeframe, start=None, end=None, limit=None):
            # Return mock DataFrame
            dates = pd.date_range(start='2024-01-01', periods=limit or 100, freq='1min')
            data = {
                'open': np.random.uniform(49000, 51000, len(dates)),
                'high': np.random.uniform(50000, 52000, len(dates)),
                'low': np.random.uniform(48000, 50000, len(dates)),
                'close': np.random.uniform(49500, 50500, len(dates)),
                'volume': np.random.uniform(1000, 10000, len(dates)),
                'exchange': ['CBSE'] * len(dates)
            }
            df = pd.DataFrame(data, index=dates)
            
            from types import SimpleNamespace
            return SimpleNamespace(df=df)
        
        def get_latest_crypto_trade(self, symbol):
            from types import SimpleNamespace
            return SimpleNamespace(price=50000.0)
    
    return MockAlpacaAPI()

# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce noise during tests