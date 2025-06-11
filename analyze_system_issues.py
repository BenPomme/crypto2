#!/usr/bin/env python3
"""
Analyze system issues with stock trading and performance metrics
"""
import os
import sys
sys.path.append('src')

from datetime import datetime, timedelta
from config.settings import get_settings
from config.stock_settings import get_stock_settings, is_stock_trading_enabled
from src.data.market_data import AlpacaDataProvider
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.firebase_logger import FirebaseLogger
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_stock_trading_config():
    """Check stock trading configuration"""
    print("\n=== STOCK TRADING CONFIGURATION ===")
    
    # Check if enabled
    enabled = is_stock_trading_enabled()
    print(f"Stock Trading Enabled: {enabled}")
    
    if enabled:
        stock_settings = get_stock_settings()
        if stock_settings:
            print(f"Stock Symbols: {stock_settings.stock_symbols}")
            print(f"Short Selling Enabled: {stock_settings.enable_short_selling}")
            print(f"Stock Order Type: {stock_settings.stock_order_type}")
            print(f"Stock Stop Loss %: {stock_settings.stock_stop_loss_pct}")
            print(f"Stock Take Profit %: {stock_settings.stock_take_profit_pct}")
        else:
            print("WARNING: Stock settings not loaded!")
    
    # Check main settings
    settings = get_settings()
    print(f"\nMain Trading Symbol(s): {settings.trading.symbol}")
    
def check_market_hours():
    """Check current market status"""
    print("\n=== MARKET STATUS ===")
    
    try:
        data_provider = AlpacaDataProvider()
        
        # Check crypto symbols
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD']
        print("\nCrypto Markets:")
        for symbol in crypto_symbols:
            is_open = data_provider.is_market_open(symbol)
            print(f"  {symbol}: {'OPEN' if is_open else 'CLOSED'}")
        
        # Check stock symbols
        stock_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
        print("\nStock Markets:")
        for symbol in stock_symbols:
            is_open = data_provider.is_market_open(symbol)
            print(f"  {symbol}: {'OPEN' if is_open else 'CLOSED'}")
        
        # Get market hours
        print("\nMarket Hours (ET):")
        print(f"  Current Time: {datetime.now()}")
        
    except Exception as e:
        print(f"Error checking market status: {e}")

def check_recent_trades():
    """Check recent trading activity"""
    print("\n=== RECENT TRADING ACTIVITY ===")
    
    try:
        data_provider = AlpacaDataProvider()
        
        # Get account info
        account_info = data_provider.get_account_info()
        print(f"\nAccount Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        print(f"Account Cash: ${account_info.get('cash', 0):,.2f}")
        print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        
        # Get positions
        positions = data_provider.get_positions()
        print(f"\nCurrent Positions: {len(positions)}")
        
        for pos in positions:
            symbol = pos.get('symbol', 'UNKNOWN')
            qty = pos.get('qty', 0)
            market_value = pos.get('market_value', 0)
            unrealized_pl = pos.get('unrealized_pl', 0)
            print(f"  {symbol}: {qty} shares, Value: ${market_value:,.2f}, P&L: ${unrealized_pl:,.2f}")
        
        # Get recent orders
        print("\nRecent Orders (last 24 hours):")
        orders = data_provider.api.list_orders(
            status='all',
            after=(datetime.now() - timedelta(hours=24)).isoformat()
        )
        
        order_count = 0
        for order in orders[:10]:  # Show last 10
            order_count += 1
            print(f"  {order.symbol} - {order.side} {order.qty} @ ${order.filled_avg_price or order.limit_price or 0:.2f} - Status: {order.status}")
        
        if order_count == 0:
            print("  No orders in the last 24 hours")
            
    except Exception as e:
        print(f"Error checking trading activity: {e}")

def check_performance_metrics():
    """Check why performance metrics show zero"""
    print("\n=== PERFORMANCE METRICS ANALYSIS ===")
    
    try:
        # Initialize performance tracker
        tracker = PerformanceTracker()
        
        # Get current performance
        performance = tracker.get_current_performance()
        
        print(f"\nTracker State:")
        print(f"  Initial Capital: ${tracker.initial_capital:,.2f}")
        print(f"  Trades in Memory: {len(tracker.trades_log)}")
        print(f"  Signals in Memory: {len(tracker.signals_log)}")
        print(f"  Equity Curve Points: {len(tracker.equity_curve)}")
        
        print(f"\nCalculated Metrics:")
        print(f"  Total Return: {performance.get('total_return', 0):.2%}")
        print(f"  Total P&L: ${performance.get('total_pnl', 0):,.2f}")
        print(f"  Num Trades: {performance.get('num_trades', 0)}")
        print(f"  Win Rate: {performance.get('win_rate', 0):.1%}")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        
        # Check if trades have PnL
        print(f"\nTrade P&L Analysis:")
        if tracker.trades_log:
            for i, trade in enumerate(tracker.trades_log[-5:]):  # Last 5 trades
                pnl = trade.get('pnl', 'NOT SET')
                symbol = trade.get('symbol', 'UNKNOWN')
                print(f"  Trade {i+1}: {symbol} - P&L: {pnl}")
        else:
            print("  No trades in memory")
        
        # Check Firebase logging
        print(f"\nFirebase Logger Status:")
        try:
            firebase_logger = FirebaseLogger()
            print(f"  Firebase Initialized: {firebase_logger.db is not None}")
        except Exception as e:
            print(f"  Firebase Error: {e}")
            
    except Exception as e:
        print(f"Error checking performance metrics: {e}")

def main():
    """Run all diagnostics"""
    print("=== CRYPTO TRADING SYSTEM DIAGNOSTICS ===")
    print(f"Run Time: {datetime.now()}")
    
    check_stock_trading_config()
    check_market_hours()
    check_recent_trades()
    check_performance_metrics()
    
    print("\n=== IDENTIFIED ISSUES ===")
    print("1. Performance metrics show zero because trades are initialized with pnl=0")
    print("2. P&L should be calculated from closed positions, not set at trade creation")
    print("3. Stock trading may not be happening due to:")
    print("   - Market hours check (stocks only trade 9:30 AM - 4:00 PM ET)")
    print("   - Volume requirements ($1M minimum)")
    print("   - Strategy conditions not met")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Fix performance tracker to calculate P&L from closed positions")
    print("2. Add more logging for stock strategy signal generation")
    print("3. Monitor during market hours to see stock activity")

if __name__ == "__main__":
    main()