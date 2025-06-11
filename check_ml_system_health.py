#!/usr/bin/env python3
"""
Check if ML systems are working properly
"""

print("=== ML SYSTEM HEALTH CHECK ===\n")

# Test 1: Check if settings can be loaded
print("1. Testing configuration loading...")
try:
    # Try the actual import that's failing
    from pydantic_settings import BaseSettings
    print("✅ pydantic_settings import successful")
except ImportError as e:
    print(f"❌ CRITICAL: pydantic_settings import failed: {e}")
    print("   This will break the entire system!")
    print("\n   Fix: pip install pydantic-settings")

# Test 2: Check if config modules load
print("\n2. Testing config modules...")
try:
    import sys
    sys.path.append('.')
    from config.settings import get_settings
    settings = get_settings()
    print("✅ Settings loaded successfully")
except Exception as e:
    print(f"❌ Settings loading failed: {e}")

# Test 3: Check ML components
print("\n3. Testing ML components...")
try:
    from integrate_optimization import OptimizedParameterManager
    param_manager = OptimizedParameterManager("optimized_parameters.json")
    print("✅ OptimizedParameterManager loaded")
    
    # Check if parameters exist
    test_params = param_manager.get_parameters_for_symbol("BTC/USD")
    if test_params:
        print(f"✅ Found optimized parameters for BTC/USD: {list(test_params.keys())}")
    else:
        print("⚠️  No optimized parameters found for BTC/USD")
        
except Exception as e:
    print(f"❌ ML parameter loading failed: {e}")

# Test 4: Check strategy parameter manager
print("\n4. Testing strategy parameter manager...")
try:
    from src.strategy.parameter_manager import ParameterManager
    # Note: This requires Firebase, so might fail locally
    print("✅ ParameterManager import successful")
except Exception as e:
    print(f"⚠️  ParameterManager import failed: {e}")
    print("   (This might be OK if Firebase isn't configured locally)")

print("\n=== SUMMARY ===")
print("""
If pydantic_settings import fails:
1. The entire system cannot start
2. No configuration can be loaded
3. ML optimization parameters cannot be used
4. Trading strategies cannot initialize

This is a CRITICAL dependency issue that would prevent
the bot from running at all on Railway.

To verify Railway status:
```bash
railway logs | grep -i "error\\|import\\|pydantic"
```

If the bot is running on Railway, then pydantic-settings
must be installed correctly there.
""")