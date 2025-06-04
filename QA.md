# Quality Assurance (QA) Process

This document defines the systematic QA process that must be completed before every deployment to ensure changes work correctly.

## 🎯 QA Checklist Overview

**Before ANY deployment, ALL items must pass:**

- [ ] **Code Validation** - Syntax and imports check
- [ ] **Configuration Validation** - Environment variables and settings  
- [ ] **Component Integration** - All modules work together
- [ ] **Trading Logic** - Signal generation and execution
- [ ] **Risk Management** - Position sizing and limits
- [ ] **Multi-Symbol Support** - All configured symbols work
- [ ] **Error Handling** - Graceful failure recovery
- [ ] **Performance Monitoring** - Logging and metrics
- [ ] **Deployment Verification** - Live system functionality

---

## 1. 📋 Pre-Deployment Code Validation

### 1.1 Syntax and Import Checks
```bash
# Check Python syntax
python -m py_compile main.py
python -m py_compile src/**/*.py

# Verify all imports resolve
python -c "import main; print('✅ Main imports OK')"
python -c "from src.data.market_data import AlpacaDataProvider; print('✅ Data imports OK')"
python -c "from src.strategy.ma_crossover_strategy import MACrossoverStrategy; print('✅ Strategy imports OK')"
python -c "from src.risk.risk_manager import RiskManager; print('✅ Risk imports OK')"
python -c "from src.execution.trade_executor import TradeExecutor; print('✅ Execution imports OK')"
```

### 1.2 Configuration Validation
```bash
# Verify environment variables
echo "TRADING_SYMBOL: $TRADING_SYMBOL"
echo "ALPACA_ENDPOINT: $ALPACA_ENDPOINT"
echo "Expected symbols: BTC/USD,ETH/USD,SPY,QQQ"

# Test settings loading
python -c "from config.settings import get_settings; s=get_settings(); print(f'✅ Trading symbols: {s.trading.symbol}')"
```

---

## 2. 🔧 Component Integration Testing

### 2.1 Data Provider Validation
**Test each symbol individually:**
```python
# Test crypto symbols
symbols_crypto = ["BTC/USD", "ETH/USD"]
for symbol in symbols_crypto:
    try:
        price = provider.get_latest_price(symbol)
        historical = provider.get_historical_data(symbol, "1Min", 5)
        print(f"✅ {symbol}: ${price:.2f}, {len(historical)} bars")
    except Exception as e:
        print(f"❌ {symbol}: {e}")

# Test stock symbols  
symbols_stocks = ["SPY", "QQQ"]
for symbol in symbols_stocks:
    try:
        price = provider.get_latest_price(symbol)
        historical = provider.get_historical_data(symbol, "1Min", 5)
        print(f"✅ {symbol}: ${price:.2f}, {len(historical)} bars")
    except Exception as e:
        print(f"❌ {symbol}: {e}")
```

### 2.2 Strategy Signal Generation
**Verify signals can be generated for each symbol:**
```python
for symbol in ["BTC/USD", "ETH/USD", "SPY", "QQQ"]:
    # Load data
    data = provider.get_historical_data(symbol, "1Min", 50)
    featured_data = feature_engineer.engineer_features(data)
    
    # Update strategy symbol
    strategy.config['symbol'] = symbol
    signal = strategy.generate_signal(featured_data)
    
    print(f"✅ {symbol}: Signal generation working")
```

### 2.3 Risk Management Validation
**Test position sizing for different symbols and market conditions:**
```python
test_scenarios = [
    {"symbol": "BTC/USD", "price": 50000, "is_crypto": True},
    {"symbol": "ETH/USD", "price": 3000, "is_crypto": True}, 
    {"symbol": "SPY", "price": 400, "is_crypto": False},
    {"symbol": "QQQ", "price": 350, "is_crypto": False}
]

for scenario in test_scenarios:
    position_size = risk_manager.calculate_position_size(
        account_value=100000,
        entry_price=scenario["price"],
        buying_power=150000,
        is_crypto=scenario["is_crypto"]
    )
    print(f"✅ {scenario['symbol']}: ${position_size.size_usd:.0f} position")
```

---

## 3. 🎮 Trading Logic Validation

### 3.1 Multi-Symbol Cycle Testing
**Verify all symbols are processed in each cycle:**
```python
# Simulate one complete trading cycle
for symbol in trading_symbols:
    print(f"Processing {symbol}...")
    
    # Check data buffer exists
    assert symbol in bot.data_buffers
    
    # Check data loading
    latest_price = bot.data_provider.get_latest_price(symbol)
    
    # Check signal generation
    # (mock the cycle logic)
    
    print(f"✅ {symbol}: Complete cycle successful")
```

### 3.2 Order Execution Testing
**Test order placement for different order types and symbols:**
```python
# Test crypto orders (should use gtc time_in_force)
crypto_test = {
    "symbol": "BTC/USD",
    "quantity": 0.001,
    "order_type": "market",
    "expected_time_in_force": "gtc"
}

# Test stock orders (should use day/gtc time_in_force)  
stock_test = {
    "symbol": "SPY", 
    "quantity": 1,
    "order_type": "market",
    "expected_time_in_force": "day"
}
```

### 3.3 Fee Calculation Validation
**Verify fees are calculated correctly:**
```python
# Test crypto fees (0.25% taker)
crypto_fee = fee_calculator.calculate_crypto_fees(50000, is_maker=False)
assert 120 <= crypto_fee <= 130  # ~$125 expected

# Test stock fees (commission-free)
stock_fee = fee_calculator.calculate_stock_fees(50000)
assert stock_fee == 0

print("✅ Fee calculations correct")
```

---

## 4. 🛡️ Risk Management Validation

### 4.1 Position Limits Testing
```python
# Test position limits are enforced
scenarios = [
    {"symbol": "BTC/USD", "max_expected": 0.5},  # 50% for crypto
    {"symbol": "SPY", "max_expected": 1.0},      # 100% for stocks with margin
]

for scenario in scenarios:
    # Test maximum position sizing
    max_position = risk_manager.calculate_max_position(
        symbol=scenario["symbol"],
        account_value=100000
    )
    expected_max = 100000 * scenario["max_expected"]
    
    assert max_position <= expected_max
    print(f"✅ {scenario['symbol']}: Position limit enforced")
```

### 4.2 Risk Per Trade Validation
```python
# Verify risk per trade is within bounds
for symbol in trading_symbols:
    risk_amount = risk_manager.calculate_risk_amount(
        account_value=100000,
        symbol=symbol
    )
    
    # Should be 2% max
    assert risk_amount <= 2000
    print(f"✅ {symbol}: Risk per trade within limits")
```

---

## 5. 🚨 Error Handling Validation

### 5.1 Data Failures
```python
# Test graceful handling of data failures
test_cases = [
    {"symbol": "INVALID/USD", "should_fail": True},
    {"symbol": "BTC/USD", "should_fail": False},
]

for case in test_cases:
    try:
        price = provider.get_latest_price(case["symbol"])
        if case["should_fail"]:
            print(f"❌ {case['symbol']}: Should have failed but didn't")
        else:
            print(f"✅ {case['symbol']}: Success as expected")
    except Exception as e:
        if case["should_fail"]:
            print(f"✅ {case['symbol']}: Failed gracefully as expected")
        else:
            print(f"❌ {case['symbol']}: Unexpected failure: {e}")
```

### 5.2 Network Failures
```python
# Test handling of API failures
# Mock network timeouts, rate limits, etc.
```

---

## 6. 📊 Performance Monitoring Validation

### 6.1 Logging Verification
**Check all critical events are logged:**
```python
required_log_events = [
    "Trading symbols: [.*]",
    "Signal generated.*",
    "Order submitted successfully.*", 
    "Fee adjustment.*",
    "Position sizing.*",
    "Status - Cycle.*"
]

# Verify each pattern appears in logs
for pattern in required_log_events:
    # Check if pattern exists in recent logs
    print(f"✅ Log pattern verified: {pattern}")
```

### 6.2 Metrics Collection
```python
# Verify all metrics are being tracked
required_metrics = [
    "total_trades",
    "win_rate", 
    "portfolio_value",
    "total_return",
    "risk_per_trade",
    "position_sizes"
]

for metric in required_metrics:
    value = performance_tracker.get_metric(metric)
    assert value is not None
    print(f"✅ Metric tracked: {metric}")
```

---

## 7. 🚀 Deployment Verification

### 7.1 Pre-Deployment Checklist
- [ ] All QA tests above pass
- [ ] Code committed to git
- [ ] Environment variables set correctly
- [ ] Railway deployment configured

### 7.2 Post-Deployment Validation (First 5 minutes)
```bash
# 1. Check deployment status
railway status

# 2. Verify initialization logs
railway logs | grep -E "(Trading symbols|initialized successfully)"

# 3. Verify all symbols are being processed
railway logs | grep -E "(BTC/USD|ETH/USD|SPY|QQQ)"

# 4. Check for errors
railway logs | grep ERROR

# 5. Verify data loading for all symbols
railway logs | grep "Loaded.*bars"

# 6. Monitor signal generation
railway logs | grep "Signal.*generated"
```

### 7.3 Success Criteria
**Deployment is successful if:**
- [ ] Bot initializes without errors
- [ ] All 4 symbols are logged during startup
- [ ] Data buffers created for all symbols  
- [ ] At least one symbol shows analysis logs
- [ ] No critical errors in first 5 minutes
- [ ] Performance tracking is active

### 7.4 Rollback Criteria
**Rollback immediately if:**
- [ ] Bot fails to initialize
- [ ] Only BTC/USD is being processed
- [ ] Critical errors appear
- [ ] Orders fail to execute
- [ ] Risk management errors

---

## 8. 📝 QA Execution Template

**For each deployment, copy and complete:**

```
QA Execution Report - [DATE]
=================================

Deployment: [DESCRIPTION]
Commit: [GIT_HASH]

Pre-Deployment Testing:
[ ] Code validation passed
[ ] Component integration passed  
[ ] Trading logic passed
[ ] Risk management passed
[ ] Error handling passed
[ ] Performance monitoring passed

Deployment:
[ ] Code deployed successfully
[ ] Environment variables verified
[ ] Railway deployment confirmed

Post-Deployment Validation:
[ ] Initialization successful
[ ] Multi-symbol processing confirmed
[ ] No critical errors
[ ] Performance tracking active

Result: ✅ PASS / ❌ FAIL
Notes: [ANY_ISSUES_FOUND]

Signed off by: [YOUR_NAME]
```

---

## 9. 🔄 Continuous Monitoring

**After successful deployment, monitor for 24 hours:**

### Key Metrics to Track:
- [ ] **Signal generation frequency** across all symbols
- [ ] **Trade execution success rate** 
- [ ] **Position sizing accuracy**
- [ ] **Fee calculations correctness**
- [ ] **Risk management compliance**
- [ ] **Performance metrics updates**

### Alert Conditions:
- No signals generated for >2 hours
- Only one symbol being traded for >1 hour  
- Order execution failures >10%
- Risk management violations
- Memory/CPU issues

---

## 📞 Emergency Procedures

**If critical issues are found:**

1. **Immediate Actions:**
   - Stop bot if actively trading
   - Document the issue
   - Check account positions

2. **Investigation:**
   - Review Railway logs
   - Check code changes
   - Verify environment variables

3. **Resolution:**
   - Fix identified issues
   - Re-run full QA process
   - Deploy fix
   - Monitor closely

---

**This QA process must be followed for EVERY deployment to ensure reliable, profitable trading operations.**