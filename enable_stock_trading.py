#!/usr/bin/env python3
"""
Enable Stock Trading Integration
This script shows the minimal changes needed to enable stock trading in main.py
"""

import sys
sys.path.append('src')

from config.settings import get_settings
from config.stock_settings import get_stock_settings, is_stock_trading_enabled

def check_stock_configuration():
    """Check current stock trading configuration"""
    print("🔍 Checking Stock Trading Configuration...")
    print("=" * 50)
    
    # Check if stock trading is enabled
    stock_enabled = is_stock_trading_enabled()
    print(f"Stock Trading Enabled: {stock_enabled}")
    
    if stock_enabled:
        stock_settings = get_stock_settings()
        print(f"Stock Symbols: {stock_settings.stock_symbols}")
        print(f"Short Selling: {stock_settings.enable_short_selling}")
        print(f"Risk Per Trade: {stock_settings.stock_risk_per_trade}")
    else:
        print("⚠️  Stock trading is disabled")
    
    # Check current trading symbols
    settings = get_settings()
    print(f"\nCurrent Trading Symbols: {settings.trading.symbol}")
    print("=" * 50)
    
    return stock_enabled

def show_integration_changes():
    """Show the changes needed in main.py"""
    
    changes = '''
REQUIRED CHANGES TO main.py:

1. Import stock settings (after line 16):
```python
from config.stock_settings import get_stock_settings, is_stock_trading_enabled
from src.strategy.stock_mean_reversion import StockMeanReversionStrategy
```

2. Update _initialize_components() to include stock symbols (around line 78):
```python
# Parse trading symbols - Support multiple symbols
symbols = [s.strip() for s in self.settings.trading.symbol.split(',')]

# Add stock symbols if enabled
if is_stock_trading_enabled():
    stock_settings = get_stock_settings()
    stock_symbols = [s.strip() for s in stock_settings.stock_symbols.split(',')]
    symbols.extend(stock_symbols)
    self.logger.info(f"Added {len(stock_symbols)} stock symbols")

self.trading_symbols = symbols
self.logger.info(f"Total trading symbols: {symbols}")
```

3. Add stock strategy initialization (after line 154):
```python
# Initialize stock strategy if enabled
self.stock_strategy = None
if is_stock_trading_enabled():
    stock_config = {
        'enable_shorts': get_stock_settings().enable_short_selling,
        'profit_target': 0.02,
        'stop_loss': 0.01,
        'min_confidence': 0.6
    }
    self.stock_strategy = StockMeanReversionStrategy(stock_config)
    self.logger.info("Stock mean reversion strategy initialized")
```

4. Update generate_signal() in _execute_symbol_cycle (around line 376):
```python
# Step 5: Generate trading signal for this symbol
if self.data_provider.is_crypto_symbol(symbol):
    signal = self.strategy.generate_signal(featured_data, symbol)
else:
    # Use stock strategy for stock symbols
    if self.stock_strategy:
        signal = self.stock_strategy.generate_signal(featured_data, symbol)
    else:
        signal = None
```
'''
    
    print(changes)

def create_integration_patch():
    """Create a patch file for the integration"""
    
    patch_content = '''--- a/main.py
+++ b/main.py
@@ -16,6 +16,8 @@ from config.settings import get_settings
 from src.utils.logger import setup_logging
 from src.data.market_data import AlpacaDataProvider
 from src.data.data_buffer import DataBuffer
+from config.stock_settings import get_stock_settings, is_stock_trading_enabled
+from src.strategy.stock_mean_reversion import StockMeanReversionStrategy
 from src.data.volume_data_manager import VolumeDataManager
 from src.strategy.feature_engineering import FeatureEngineer
 from src.strategy.ma_crossover_strategy import MACrossoverStrategy
@@ -78,6 +80,14 @@ class CryptoTradingBot:
             # Parse trading symbols - Support multiple symbols
             symbols = [s.strip() for s in self.settings.trading.symbol.split(',')]
+            
+            # Add stock symbols if enabled
+            if is_stock_trading_enabled():
+                stock_settings = get_stock_settings()
+                stock_symbols = [s.strip() for s in stock_settings.stock_symbols.split(',')]
+                symbols.extend(stock_symbols)
+                self.logger.info(f"Added {len(stock_symbols)} stock symbols")
+            
             self.trading_symbols = symbols
             self.logger.info(f"Trading symbols: {symbols}")
             
@@ -154,6 +164,17 @@ class CryptoTradingBot:
             }
             self.strategy = MACrossoverStrategy(strategy_config, self.parameter_manager)
             
+            # Initialize stock strategy if enabled
+            self.stock_strategy = None
+            if is_stock_trading_enabled():
+                stock_config = {
+                    'enable_shorts': get_stock_settings().enable_short_selling,
+                    'profit_target': 0.02,
+                    'stop_loss': 0.01,
+                    'min_confidence': 0.6
+                }
+                self.stock_strategy = StockMeanReversionStrategy(stock_config)
+                self.logger.info("Stock mean reversion strategy initialized")
+            
             self.logger.info("All components initialized successfully")
             
@@ -374,7 +395,13 @@ class CryptoTradingBot:
             featured_data = self.feature_engineer.engineer_features(enhanced_data, custom_config=feature_config)
             
             # Step 5: Generate trading signal for this symbol
-            signal = self.strategy.generate_signal(featured_data, symbol)
+            if self.data_provider.is_crypto_symbol(symbol):
+                signal = self.strategy.generate_signal(featured_data, symbol)
+            else:
+                # Use stock strategy for stock symbols
+                if self.stock_strategy:
+                    signal = self.stock_strategy.generate_signal(featured_data, symbol)
+                else:
+                    signal = None
             
             # Log signal generation details
'''
    
    with open('stock_integration.patch', 'w') as f:
        f.write(patch_content)
    
    print("\n✅ Created stock_integration.patch")
    print("To apply: git apply stock_integration.patch")

if __name__ == "__main__":
    print("🏦 STOCK TRADING INTEGRATION CHECK")
    print("=" * 50)
    
    # Check configuration
    stock_enabled = check_stock_configuration()
    
    if not stock_enabled:
        print("\n⚠️  Stock trading is not enabled!")
        print("Set ENABLE_STOCK_TRADING=true in Railway to enable")
    else:
        print("\n✅ Stock trading is enabled in configuration")
        print("❌ But NOT integrated in main.py!")
    
    # Show required changes
    show_integration_changes()
    
    # Create patch file
    create_integration_patch()