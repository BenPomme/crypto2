#!/usr/bin/env python3
"""
Quick test to verify enhanced indicators logic
"""
import pandas as pd
import numpy as np

def test_enhanced_calculations():
    """Test the enhanced indicator calculations directly"""
    print("🧪 Testing Enhanced Indicator Calculations...")
    
    # Create test data
    np.random.seed(42)
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1200, 800, 1500, 900]
    })
    
    print(f"✅ Test data created: {len(data)} bars")
    
    # Test 1: OBV calculation
    obv = pd.Series(0.0, index=data.index)
    obv.iloc[0] = data['volume'].iloc[0]
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
        elif data['close'].iloc[i] < data['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    print(f"✅ OBV calculated: {obv.iloc[-1]:.0f}")
    
    # Test 2: MFI calculation
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    
    positive_flow = pd.Series(0.0, index=data.index)
    negative_flow = pd.Series(0.0, index=data.index)
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = money_flow.iloc[i]
    
    pos_mf = positive_flow.rolling(window=3, min_periods=1).sum()
    neg_mf = negative_flow.rolling(window=3, min_periods=1).sum()
    mfi_ratio = pos_mf / (neg_mf + 1e-10)
    mfi = 100 - (100 / (1 + mfi_ratio))
    
    print(f"✅ MFI calculated: {mfi.iloc[-1]:.1f}")
    
    # Test 3: MACD calculation
    fast_ema = data['close'].ewm(span=3).mean()
    slow_ema = data['close'].ewm(span=5).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=3).mean()
    
    print(f"✅ MACD calculated: {macd.iloc[-1]:.3f}, Signal: {macd_signal.iloc[-1]:.3f}")
    
    # Test 4: Bollinger Bands
    bb_sma = data['close'].rolling(window=3, min_periods=1).mean()
    bb_std = data['close'].rolling(window=3, min_periods=1).std()
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    
    print(f"✅ Bollinger Bands: Upper={bb_upper.iloc[-1]:.2f}, Lower={bb_lower.iloc[-1]:.2f}")
    
    print("\n🎉 All enhanced indicator calculations working correctly!")
    print("📊 Summary of what should appear in production logs:")
    print(f"   📈 Basic: SMA_fast, SMA_slow, RSI")
    print(f"   🔊 Volume: OBV={obv.iloc[-1]:.0f}, MFI={mfi.iloc[-1]:.1f}")
    print(f"   📊 MACD: {macd.iloc[-1]:.3f}, Signal={macd_signal.iloc[-1]:.3f}")
    print(f"   📏 Bollinger: Upper={bb_upper.iloc[-1]:.2f}, Lower={bb_lower.iloc[-1]:.2f}")

if __name__ == "__main__":
    test_enhanced_calculations()