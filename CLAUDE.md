# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the design and implementation plan for a crypto trading algorithm targeting 5-10% monthly ROI. The system is designed to integrate with:

- **Alpaca API** for paper trading and market data
- **Railway** for cloud deployment with GitHub CI/CD
- **Firebase** for data storage, logging, and real-time monitoring
- **Python** as the primary development language

## GOLDEN PROJECT RULE

**No Hardcoded Data** Never hardcode data, or use fake data, or use hardcoded data as fallback for trading. All data and numbers and calculations must be real and functional. 
**Deployments** Handle Deployments yourself by deploying to Main in Github, which will push to railway. 

## Architecture Components

The system follows a modular architecture with clear separation of concerns:

1. **Data Ingestion Module** - Alpaca market data via REST API and WebSocket
2. **Feature Engineering Module** - Technical indicators using TA-Lib or pandas-ta
3. **Signal Generation Module** - Rule-based and ML-based trading decisions
4. **Risk Management Module** - Position sizing, stop-loss, and safety filters
5. **Execution Module** - Order placement via Alpaca API
6. **Performance Tracking Module** - Firebase logging and monitoring

## Key Technologies and Libraries

- `alpaca_trade_api` - Primary trading API integration
- `TA-Lib` or `pandas-ta` - Technical indicator calculations
- `scikit-learn`, `XGBoost` - Machine learning models
- `stable-baselines3` or `FinRL` - Reinforcement learning (if implemented)
- `firebase-admin` - Database and logging integration
- `River` - Online learning capabilities

## Trading Strategy Approach

The strategy combines multiple signal types:
- **Trend-following** (Moving Averages)
- **Momentum** (RSI, other oscillators)
- **Volume** (OBV, MFI)
- **Volatility** (Bollinger Bands, ATR)
- **Machine Learning** (supervised/reinforcement learning)

## Development Patterns

- Start with simple strategies (MA crossover) before adding complexity
- Use modular, event-driven architecture for scalability
- Implement comprehensive logging to Firebase for monitoring
- Separate model logic from execution agent
- Apply strict risk management (1-2% risk rule, stop-losses)
- Use paper trading for validation before live deployment

## Deployment Strategy

- **Railway** hosting with GitHub integration for CI/CD
- **Docker** containerization for consistent environments
- Environment variables for API keys and secrets
- 24/7 operation for crypto market hours
- Real-time monitoring via Firebase dashboard

## Risk Management Principles

- Never risk more than 1-2% of capital per trade
- Implement automatic stop-losses and take-profits
- Use position sizing based on volatility
- Include circuit breakers for excessive drawdowns
- Maintain diversification across uncorrelated assets (when scaling)

## Current Implementation Status

### Completed Core Modules ✅

1. **Data Ingestion** (`src/data/`)
   - `market_data.py` - Alpaca API integration for real-time and historical data
   - `data_buffer.py` - Efficient circular buffer for OHLCV data management

2. **Strategy Framework** (`src/strategy/`)
   - `base_strategy.py` - Abstract base class for all trading strategies
   - `indicators.py` - Technical indicators using pandas-ta (MA, RSI, Bollinger Bands, etc.)
   - `feature_engineering.py` - Comprehensive feature creation from market data
   - `ma_crossover_strategy.py` - Baseline MA crossover strategy with filters
   - `parameter_manager.py` - ML-optimizable parameter management system

3. **Risk Management** (`src/risk/`)
   - `risk_manager.py` - Multi-layer risk checking and position limits
   - `position_sizer.py` - Advanced position sizing (percent risk, volatility-adjusted, Kelly)

4. **Execution Engine** (`src/execution/`)
   - `order_manager.py` - Order placement and lifecycle management via Alpaca
   - `trade_executor.py` - High-level trade execution with risk integration

5. **Performance Monitoring** (`src/monitoring/`)
   - `performance_tracker.py` - Real-time performance tracking and analytics
   - `firebase_logger.py` - Firebase integration for data persistence
   - `metrics_calculator.py` - Comprehensive trading metrics (Sharpe, drawdown, etc.)

6. **Main System** 
   - `main.py` - Complete trading bot orchestrating all modules
   - `config/settings.py` - Environment-based configuration management

### Recent Fixes ✅

**Alpaca API Connection Issue (RESOLVED)**
- **Problem**: API URLs were duplicated (/v2/v2/account) causing 404 errors
- **Root Cause**: Both `base_url` (containing /v2) and `api_version='v2'` specified
- **Fix**: Removed `api_version='v2'` parameter from AlpacaDataProvider initialization
- **Status**: Fixed in commit `8dd6a92` and deployed to Railway

**Missing Dependency Issue (RESOLVED)**
- **Problem**: ModuleNotFoundError: No module named 'pydantic_settings'
- **Root Cause**: pydantic-settings package missing from requirements.txt
- **Fix**: Added pydantic-settings to requirements.txt
- **Status**: Fixed in commit `edfc1e3` and deployed to Railway

**pandas_ta/numpy Compatibility Issue (RESOLVED)**
- **Problem**: ImportError: cannot import name 'NaN' from 'numpy'
- **Root Cause**: pandas_ta incompatible with numpy 2.0+ (NaN deprecated → nan)
- **Fix**: Pin numpy<2.0.0 and use pandas-ta>=0.3.14b0 for compatibility
- **Status**: Fixed in commit `281b3f9` and deploying to Railway

### Testing Framework ✅
- Comprehensive test suite with pytest
- Mock fixtures for Alpaca API and Firebase
- Unit tests for core components

### Key Features Implemented

- **Real-time data ingestion** from Alpaca with proper error handling
- **Multi-signal strategy framework** with confidence scoring
- **Advanced risk management** with multiple circuit breakers
- **Automated order execution** with stop-loss and take-profit
- **Real-time performance tracking** with Firebase logging
- **Comprehensive metrics calculation** including risk-adjusted returns
- **Modular architecture** allowing easy strategy additions

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run the trading bot
python main.py

# Check code quality
black src/ tests/
flake8 src/ tests/
```

## Railway Deployment & Monitoring

**Railway CLI Setup:**
```bash
# CLI is installed, token is configured in .env.local
export RAILWAY_TOKEN=800e9ae8-5270-4fef-821c-3038a351a89b
railway link --project satisfied-commitment
```

**Monitoring Commands:**
```bash
# Check deployment status
railway status

# Stream live logs
railway logs --follow

# Check build logs
railway logs --build

# Check deployment logs  
railway logs --deployment

# Redeploy latest
railway redeploy
```

**Deployment Process:**
1. Make changes to code
2. Commit and push to main branch
3. Railway auto-deploys via GitHub integration
4. Monitor with `railway logs --follow`

## File Structure Context

- `project.md` - Comprehensive design document and implementation guide
- `cryptotradingstrategy.md` - shares some best practice on working crypto trading strategy on Alpaca that we can replicate
- `main.py` - Entry point for the complete trading system
- `src/` - All source code organized by module (data, strategy, risk, execution, monitoring)
- `tests/` - Comprehensive test suite for all modules
- `config/` - Configuration management with environment variables# Environment variables updated Wed Jun  4 12:20:42 CEST 2025
