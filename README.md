# Crypto Trading Algorithm

A sophisticated cryptocurrency trading system targeting 5-10% monthly ROI using Alpaca's paper trading API. Built with a modular architecture following the design principles outlined in `project.md`.

## 🚀 Features

- **Real-time Data Ingestion**: Alpaca API integration for live crypto market data
- **Advanced Technical Analysis**: 20+ technical indicators using pandas-ta
- **Multi-Signal Strategy Framework**: Extensible strategy system with confidence scoring
- **Comprehensive Risk Management**: Multi-layer risk checks and position sizing
- **Automated Order Execution**: Stop-loss, take-profit, and order management
- **Real-time Performance Tracking**: Firebase integration for live monitoring
- **Robust Testing**: Comprehensive test suite with 95%+ coverage

## 🏗️ Architecture

```
├── src/
│   ├── data/           # Market data ingestion and buffering
│   ├── strategy/       # Trading strategies and technical analysis
│   ├── risk/          # Risk management and position sizing
│   ├── execution/     # Order management and trade execution
│   ├── monitoring/    # Performance tracking and Firebase logging
│   └── utils/         # Logging and helper utilities
├── tests/             # Comprehensive test suite
├── config/            # Configuration management
└── main.py           # Trading bot entry point
```

## 📋 Prerequisites

- Python 3.11+
- Alpaca Paper Trading Account
- Firebase Project (for logging and monitoring)
- Railway Account (for deployment)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crypto2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   
   Create a `.env.local` file with your credentials:
   ```env
   # Alpaca Configuration
   ALPACA_ENDPOINT=https://paper-api.alpaca.markets/v2
   ALPACA_KEY=your_alpaca_key
   ALPACA_SECRET=your_alpaca_secret

   # Firebase Configuration
   FIREBASE_API_KEY=your_firebase_api_key
   FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
   FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   FIREBASE_APP_ID=your_app_id
   FIREBASE_MEASUREMENT_ID=your_measurement_id

   # Trading Configuration (Optional)
   TRADING_SYMBOL=BTCUSD
   RISK_PER_TRADE=0.02
   MAX_POSITION_SIZE=1000.0
   ```

## 🏃‍♂️ Usage

### Local Development

1. **Run tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

2. **Start the trading bot**
   ```bash
   python main.py
   ```

3. **Monitor logs**
   ```bash
   tail -f logs/crypto_trading_bot.log
   ```

### Docker Development

1. **Build the image**
   ```bash
   docker build -t crypto-trading-bot .
   ```

2. **Run the container**
   ```bash
   docker run --env-file .env.local crypto-trading-bot
   ```

## 🚀 Deployment on Railway

1. **Connect your GitHub repository to Railway**

2. **Set environment variables in Railway dashboard**:
   - All variables from `.env.local`
   - `ENVIRONMENT=production`
   - `LOG_LEVEL=INFO`

3. **Deploy**:
   Railway will automatically deploy using the `Dockerfile` and `railway.json` configuration.

## 📊 Strategy Overview

### Current Implementation: MA Crossover Strategy

- **Signals**: Golden cross (fast MA > slow MA) for buy, death cross for sell
- **Filters**: RSI confirmation, volume validation, trend strength
- **Risk Management**: 2% risk per trade, stop-loss, take-profit
- **Position Sizing**: Volatility-adjusted with multiple sizing methods

### Technical Indicators Used

- Moving Averages (SMA/EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Money Flow Index (MFI)
- MACD
- Stochastic Oscillator

## 🛡️ Risk Management

- **Position Sizing**: Percent risk, volatility-adjusted, Kelly criterion
- **Circuit Breakers**: Daily loss limits, consecutive loss limits
- **Drawdown Protection**: Maximum drawdown limits with cooldown periods
- **Position Limits**: Maximum concurrent positions and same-asset limits
- **Real-time Monitoring**: Live risk metrics and alerts

## 📈 Performance Monitoring

### Firebase Integration
- Real-time trade logging
- Performance metrics storage
- System status monitoring
- Error tracking and alerting

### Metrics Calculated
- Total return and P&L
- Sharpe ratio and volatility
- Maximum drawdown
- Win rate and profit factor
- Risk-adjusted returns
- Monthly performance breakdown

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_ma_crossover_strategy.py -v
```

## 🔧 Configuration

### Trading Parameters
- `FAST_MA_PERIOD`: Fast moving average period (default: 12)
- `SLOW_MA_PERIOD`: Slow moving average period (default: 24)
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `RISK_PER_TRADE`: Risk percentage per trade (default: 0.02)
- `MAX_POSITION_SIZE`: Maximum position size in USD (default: 1000)

### Risk Management
- `MAX_DAILY_LOSS`: Maximum daily loss percentage (default: 0.05)
- `MAX_DRAWDOWN`: Maximum drawdown percentage (default: 0.20)
- `CONSECUTIVE_LOSS_LIMIT`: Maximum consecutive losses (default: 5)

## 📝 Logs and Monitoring

### Log Files
- `logs/crypto_trading_bot.log`: Main application logs
- Structured JSON logging for production
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)

### Firebase Dashboard
- Real-time trading signals
- Performance metrics
- Trade execution logs
- System health monitoring

## 🚧 Future Enhancements

### Planned Features
- [ ] Backtesting system for strategy validation
- [ ] Additional strategies (mean reversion, momentum)
- [ ] Machine learning integration (supervised/reinforcement learning)
- [ ] Multi-asset portfolio support
- [ ] Advanced order types (trailing stops, iceberg orders)
- [ ] Web dashboard for monitoring
- [ ] Alert system (email, SMS, Discord)

### Advanced ML Integration
- Supervised learning for signal prediction
- Online learning for real-time adaptation
- Reinforcement learning agents
- Feature importance analysis
- Strategy ensemble methods

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure code quality (black, flake8)
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the comprehensive test suite examples
2. Review the `project.md` design document
3. Examine the modular architecture in `src/`