#!/usr/bin/env python3
"""
Debug script to test if enhanced indicators are working
"""
import sys
import os
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_indicators():
    """Test if enhanced indicators are calculated"""
    try:
        # Create simple test data
        dates = pd.date_range(start='2025-01-01', periods=50, freq='1min')
        np.random.seed(42)
        
        # Create OHLCV data
        close_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 50))
        data = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.2, 50),
            'high': close_prices + np.random.uniform(0, 1, 50),
            'low': close_prices - np.random.uniform(0, 1, 50),
            'close': close_prices,
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=dates)
        
        print("🧪 Testing Enhanced Indicators...")
        print(f"Input data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Try to import and use our enhanced indicators
        from strategy.indicators import TechnicalIndicators
        
        print("✅ Successfully imported TechnicalIndicators")
        
        # Calculate indicators
        result = TechnicalIndicators.calculate_all_indicators(data)
        
        print(f"✅ Enhanced indicators calculated, result shape: {result.shape}")
        print(f"Result columns: {list(result.columns)}")
        
        # Check for our enhanced indicators
        enhanced_indicators = [
            'obv', 'obv_trend', 'mfi',  # Volume indicators
            'bb_upper', 'bb_lower', 'bb_middle',  # Bollinger Bands
            'macd', 'macd_signal', 'macd_histogram', 'macd_bullish'  # MACD
        ]
        
        missing = []
        present = []
        
        for indicator in enhanced_indicators:
            if indicator in result.columns:
                present.append(indicator)
                print(f"✅ {indicator}: {result[indicator].iloc[-1]:.3f}")
            else:
                missing.append(indicator)
                print(f"❌ {indicator}: MISSING")
        
        print(f"\n📊 Summary:")
        print(f"   Present indicators: {len(present)}/{len(enhanced_indicators)}")
        print(f"   Missing indicators: {missing}")
        
        if len(present) == len(enhanced_indicators):
            print("🎉 All enhanced indicators are working!")
            return True
        else:
            print("⚠️  Some enhanced indicators are missing!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing indicators: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_indicators()
    sys.exit(0 if success else 1)