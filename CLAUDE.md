# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a sophisticated crypto and stock trading algorithm targeting 5-10% monthly ROI. The system integrates with:

- **Alpaca API** for paper trading, market data, and order execution
- **Binance WebSocket** for real-time crypto volume data
- **Railway** for cloud deployment with GitHub CI/CD
- **Firebase** for data storage, logging, and real-time monitoring
- **Python 3.11+** as the primary development language

## GOLDEN PROJECT RULES

1. **No Hardcoded Data** - Never hardcode data, use fake data, or use hardcoded data as fallback for trading. All data and numbers and calculations must be real and functional.
2. **Deployments** - Handle deployments yourself by pushing to main branch in GitHub, which auto-deploys to Railway.
3. **Position Sizing** - Always respect buying power limits. Use 50% max of available buying power per position.
4. **Multi-Asset Support** - System trades both crypto (24/7) and stocks (market hours only). 

## Architecture Components

### Core Trading System
1. **Data Ingestion Module** (`src/data/`)
   - `market_data.py` - Alpaca REST API and WebSocket integration
   - `data_buffer.py` - Efficient circular buffer for OHLCV data
   - `volume_data_manager.py` - Binance WebSocket for real-time crypto volumes

2. **Strategy Framework** (`src/strategy/`)
   - `base_strategy.py` - Abstract base class for all strategies
   - `ma_crossover_strategy.py` - Crypto strategy with multiple entry signals
   - `stock_mean_reversion.py` - Stock strategy supporting long/short positions
   - `indicators.py` - 20+ technical indicators via pandas-ta
   - `feature_engineering.py` - Market feature creation
   - `parameter_manager.py` - ML-optimizable parameters

3. **Risk Management** (`src/risk/`)
   - `risk_manager.py` - Multi-layer risk checks and circuit breakers
   - `position_sizer.py` - Advanced sizing (percent risk, volatility-adjusted, Kelly)
   - `buying_power_manager.py` - Unified margin management for crypto/stocks

4. **Execution Engine** (`src/execution/`)
   - `order_manager.py` - Alpaca order placement and lifecycle
   - `trade_executor.py` - High-level execution with risk integration

5. **Monitoring** (`src/monitoring/`)
   - `performance_tracker.py` - Real-time P&L and metrics
   - `firebase_logger.py` - Firebase integration with multi-line JSON support
   - `metrics_calculator.py` - Sharpe, drawdown, and other metrics

## Trading Configuration

### Crypto Trading (24/7)
- **Symbols**: BTC/USD, ETH/USD, SOL/USD, AVAX/USD
- **Strategy**: MA Crossover with multiple signals
  - Golden/Death crosses
  - Trend continuation
  - Momentum entries
  - Breakout signals
- **Leverage**: 3x available
- **Volume**: Real-time Binance data
- **Targets**: 6% profit, 2% stop loss

### Stock Trading (Market Hours)
- **Symbols**: SPY, QQQ, AAPL, TSLA
- **Strategy**: Mean Reversion
  - Long positions on oversold bounce
  - Short positions on overbought reversal
- **Leverage**: 4x available (Reg T)
- **Volume**: Alpaca volume data
- **Targets**: 2% profit, 1% stop loss

## Recent Critical Fixes

### 1. Position Sizing Fix (Commit: 65c1c63)
**Problem**: Positions exceeding available buying power
```python
# OLD: Could create massive positions with tiny stops
position_size = (risk_amount / price_risk) * current_price

# NEW: Added minimum price risk constraint
min_price_risk = 0.01  # At least 1% price movement
price_risk = max(price_risk, min_price_risk)
buying_power_usage = 0.5  # Use max 50% of available
```

### 2. Firebase Multi-line JSON Fix (Commit: 94f962f)
**Problem**: Firebase credentials with multi-line private keys failing to parse
```python
# NEW: Robust JSON parser in src/utils/firebase_helper.py
def parse_firebase_credentials():
    # Handles escaped newlines, multi-line keys, malformed JSON
    # Multiple fallback parsing methods
```

### 3. Stock Trading Integration (Commit: 6b233ac)
**Problem**: Stock strategy designed but not connected to main loop
```python
# main.py changes:
1. Import stock modules
2. Add stock symbols to trading list
3. Route symbols by format (crypto has '/', stocks don't)
4. Use appropriate strategy per symbol type
```

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src

# Run main bot
python main.py

# Test specific components
python quick_indicator_test.py
python simple_backtest_demo.py
```

### Railway Deployment & Monitoring
```bash
# Environment setup (already configured)
export RAILWAY_TOKEN=800e9ae8-5270-4fef-821c-3038a351a89b

# Deploy to production
git add .
git commit -m "Deploy: description"
git push origin main  # Auto-deploys via GitHub integration

# Monitor live logs
railway logs --follow

# Check recent logs
railway logs | tail -200

# Search logs
railway logs | grep -i "signal\|error\|warning"

# Check system status
railway status

# Redeploy if needed
railway redeploy
```

## Environment Variables

### Required in Railway
```env
# Alpaca Configuration
ALPACA_ENDPOINT=https://paper-api.alpaca.markets
ALPACA_KEY=your_key
ALPACA_SECRET=your_secret

# Firebase Configuration (supports multi-line JSON)
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_PRIVATE_KEY_ID=your_key_id
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n
FIREBASE_CLIENT_EMAIL=your_email
FIREBASE_CLIENT_ID=your_client_id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
FIREBASE_CLIENT_CERT_URL=your_cert_url

# Trading Configuration
TRADING_SYMBOL=BTC/USD,ETH/USD,SOL/USD,AVAX/USD
ENABLE_STOCK_TRADING=true
STOCK_SYMBOLS=SPY,QQQ,AAPL,TSLA
ENABLE_SHORT_SELLING=true

# Risk Settings
RISK_PER_TRADE=0.02
MAX_POSITION_SIZE=1000.0
MAX_DAILY_LOSS=0.06
```

## Common Issues & Solutions

### 1. "Firebase not initialized"
- Check FIREBASE_* env vars in Railway
- Ensure private key has proper newlines
- Check logs for JSON parsing errors

### 2. "Insufficient balance" errors
- Position sizing now fixed with 50% buying power limit
- Check account balance with `railway logs | grep "account balance"`
- Verify leverage settings

### 3. No stock trades during market hours
- Verify ENABLE_STOCK_TRADING=true
- Check market hours (9:30 AM - 4:00 PM ET)
- Look for "Stock mean reversion strategy initialized" in logs

### 4. Missing crypto volume data
- Check Binance WebSocket connection status
- Look for "Binance WebSocket: ✅ Connected" in logs
- Volume manager only handles crypto symbols

## System Status Checks

### What to Look for in Logs
```bash
# Good signs
"Added 4 stock symbols"
"Stock mean reversion strategy initialized"
"Using Firebase service account key from environment"
"🎯 [SYMBOL] Signal: BUY/SELL"
"Binance WebSocket: ✅ Connected"

# Check trading activity
railway logs | grep -E "Signal:|Position:|Trade executed"

# Check for errors
railway logs | grep -i "error\|warning\|failed"

# Market status
railway logs | grep -i "market\|open\|closed"
```

### Performance Monitoring
The system logs comprehensive status every 10 cycles including:
- Portfolio value and P&L
- Win rate and Sharpe ratio
- Active positions and pending orders
- Market status for all symbols
- Volume data status

## Architecture Best Practices

1. **Symbol Routing**: Symbols with '/' are crypto, without are stocks
2. **Volume Data**: Binance for crypto only, Alpaca for stocks
3. **Market Hours**: Use `is_market_open()` before trading stocks
4. **Position Management**: Track positions per symbol, not globally
5. **Strategy Selection**: Route to appropriate strategy by symbol type

## Future Enhancements

### In Progress
- [ ] Backtesting system with walk-forward optimization
- [ ] ML-based signal prediction
- [ ] Portfolio rebalancing logic

### Planned
- [ ] Options trading integration
- [ ] Sentiment analysis from news/social
- [ ] Advanced order types (trailing stops, icebergs)
- [ ] Web dashboard for monitoring
- [ ] Multi-exchange arbitrage

## Optimization System

The system includes sophisticated parameter optimization:

1. **Parameter Files**:
   - `optimized_parameters.json` - Production parameters
   - `demo_optimized_parameters.json` - Demo parameters with more symbols

2. **Optimization Scripts**:
   - `optimize.py` - Main optimization engine
   - `optimize_simple.py` - Quick optimization
   - `optimize_trading_pairs.py` - Multi-symbol optimization

3. **Integration**:
   - `OptimizedParameterManager` loads parameters per symbol
   - Parameters updated every 100 trading cycles
   - ML-based continuous improvement

## Testing Strategy

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Backtesting
python quick_backtesting_demo.py

# Live paper trading validation
python main.py  # Monitor for 24 hours
```

## Critical Safety Features

1. **Circuit Breakers**:
   - Daily loss limit: 6%
   - Consecutive loss limit: 5 trades
   - Maximum drawdown: 20%

2. **Position Limits**:
   - 50% max buying power per position
   - 1% minimum price risk (prevents huge positions)
   - $100 minimum position size

3. **Risk Management**:
   - 2% risk per trade
   - Stop losses on all positions
   - Real-time margin monitoring

## Support & Troubleshooting

1. Check comprehensive test examples in `tests/`
2. Review `project.md` for design details
3. Examine module docstrings in `src/`
4. Use `check_system_status.py` for diagnostics

---

Last Updated: June 2025
System Version: 2.0 (Crypto + Stocks)