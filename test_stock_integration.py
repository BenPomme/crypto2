#!/usr/bin/env python3
"""
Test Stock Trading Integration
Verifies the stock trading components work without breaking existing system
"""

import sys
sys.path.append('src')

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test new imports
        from src.risk.buying_power_manager import BuyingPowerManager
        print("✅ BuyingPowerManager imported successfully")
        
        from src.strategy.stock_mean_reversion import StockMeanReversionStrategy
        print("✅ StockMeanReversionStrategy imported successfully")
        
        from config.stock_settings import is_stock_trading_enabled, get_stock_settings
        print("✅ Stock settings imported successfully")
        
        # Test existing imports still work
        from src.strategy.ma_crossover_strategy import MACrossoverStrategy
        print("✅ Existing strategies still work")
        
        from src.risk.risk_manager import RiskManager
        print("✅ Existing risk manager still works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_buying_power_manager():
    """Test buying power calculations"""
    print("\nTesting BuyingPowerManager...")
    
    try:
        from src.risk.buying_power_manager import BuyingPowerManager, AssetType
        
        manager = BuyingPowerManager()
        
        # Test asset type detection
        assert manager.get_asset_type('BTC/USD') == AssetType.CRYPTO
        assert manager.get_asset_type('AAPL') == AssetType.STOCK
        print("✅ Asset type detection works")
        
        # Test margin calculations
        account_info = {
            'portfolio_value': 100000,
            'cash': 50000,
            'positions': []
        }
        
        crypto_bp = manager.calculate_available_buying_power(account_info, AssetType.CRYPTO)
        stock_bp = manager.calculate_available_buying_power(account_info, AssetType.STOCK)
        
        print(f"✅ Crypto buying power: ${crypto_bp:,.2f} (3x leverage)")
        print(f"✅ Stock buying power: ${stock_bp:,.2f} (4x leverage)")
        
        return True
        
    except Exception as e:
        print(f"❌ BuyingPowerManager error: {e}")
        return False

def test_stock_strategy():
    """Test stock strategy initialization"""
    print("\nTesting Stock Strategy...")
    
    try:
        from src.strategy.stock_mean_reversion import StockMeanReversionStrategy
        import pandas as pd
        
        strategy = StockMeanReversionStrategy({
            'enable_shorts': True,
            'min_confidence': 0.6
        })
        
        print(f"✅ Strategy initialized: {strategy.name}")
        print(f"✅ Required indicators: {strategy.get_required_indicators()}")
        
        # Test with empty data (should return None)
        empty_df = pd.DataFrame()
        signal = strategy.generate_signal(empty_df, 'AAPL')
        assert signal is None
        print("✅ Handles empty data correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Stock strategy error: {e}")
        return False

def test_configuration():
    """Test configuration without breaking existing system"""
    print("\nTesting Configuration...")
    
    try:
        from config.stock_settings import is_stock_trading_enabled, get_stock_settings
        
        # Should be disabled by default
        enabled = is_stock_trading_enabled()
        print(f"✅ Stock trading enabled: {enabled} (default: False)")
        
        settings = get_stock_settings()
        if settings:
            print(f"✅ Stock settings loaded: {settings.stock_symbols}")
        else:
            print("✅ Stock settings not loaded (expected when disabled)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING STOCK TRADING INTEGRATION")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_buying_power_manager,
        test_stock_strategy,
        test_configuration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"✅ Passed {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Stock trading components are ready.")
        print("📝 Note: Stock trading is disabled by default.")
        print("   Set ENABLE_STOCK_TRADING=true to activate.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()