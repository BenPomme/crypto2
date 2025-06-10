# Stock Trading System Design

## Overview
This document outlines the design for integrating stock trading capabilities into the existing crypto trading system, supporting both long and short positions during US market hours.

## Key Design Principles
1. **Non-Breaking Integration**: Extend existing infrastructure without disrupting crypto trading
2. **Unified Risk Management**: Share buying power and leverage across crypto and stocks
3. **Market Hours Awareness**: Respect US market hours for stock trading
4. **Commission Optimization**: Handle different fee structures for stocks vs crypto

## Architecture Components

### 1. Extended Symbol Management
```python
# config/settings.py - Add stock symbols
TRADING_SYMBOLS = "BTC/USD,ETH/USD,SOL/USD,AVAX/USD,AAPL,TSLA,SPY,QQQ"
```

### 2. Enhanced Position Management

#### A. Short Selling Support
```python
# src/execution/order_manager.py
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"  # New for initiating short positions
    BUY_TO_COVER = "buy_to_cover"  # New for closing short positions
```

#### B. Position Type Detection
```python
# src/strategy/base_strategy.py
def is_short(self, symbol: str) -> bool:
    """Check if we have a short position in symbol"""
    position = self.position_tracking.get(symbol, {})
    return position.get('position_type') == 'short'
```

### 3. Buying Power & Margin Management

#### A. Unified Buying Power Calculation
```python
# src/risk/buying_power_manager.py
class BuyingPowerManager:
    def __init__(self):
        self.crypto_leverage = 3.0  # Max 3x for crypto
        self.stock_leverage = 4.0   # Max 4x for stocks (Reg T)
        self.maintenance_margin = {
            'crypto': 0.35,  # 35% maintenance
            'stock': 0.25    # 25% maintenance
        }
    
    def calculate_available_buying_power(self, account_info, asset_type='crypto'):
        """Calculate available buying power considering all positions"""
        total_equity = account_info['portfolio_value']
        
        # Calculate margin used by existing positions
        margin_used = self.calculate_margin_used(account_info['positions'])
        
        # Available margin
        if asset_type == 'crypto':
            max_buying_power = total_equity * self.crypto_leverage
        else:
            max_buying_power = total_equity * self.stock_leverage
            
        return max_buying_power - margin_used
    
    def can_open_position(self, symbol, size, side, account_info):
        """Check if we have enough buying power for new position"""
        asset_type = 'stock' if self.is_stock(symbol) else 'crypto'
        required_margin = self.calculate_required_margin(symbol, size, side, asset_type)
        available = self.calculate_available_buying_power(account_info, asset_type)
        
        return required_margin <= available
```

### 4. Commission-Aware Position Sizing

#### A. Fee Structure
```python
# src/utils/fee_calculator.py - Extended for stocks
class FeeCalculator:
    def __init__(self):
        self.fee_structures = {
            'crypto': {
                'maker': 0.0025,  # 0.25%
                'taker': 0.0025   # 0.25%
            },
            'stock': {
                'commission': 0.0,  # $0 commission on Alpaca
                'sec_fee': 0.0000278,  # SEC fee on sales
                'taf_fee': 0.000119,   # FINRA TAF
                'locate_fee': 0.01     # Per share short locate fee
            }
        }
    
    def calculate_stock_fees(self, quantity, price, side, is_short=False):
        """Calculate total fees for stock trades"""
        value = quantity * price
        fees = 0
        
        if side == 'sell' or side == 'buy_to_cover':
            # SEC and TAF fees on sales
            fees += value * self.fee_structures['stock']['sec_fee']
            fees += quantity * self.fee_structures['stock']['taf_fee']
        
        if is_short:
            # Add locate fees for short positions
            fees += quantity * self.fee_structures['stock']['locate_fee']
            
        return fees
```

### 5. Stock Trading Strategies

#### A. Mean Reversion Strategy for Stocks
```python
# src/strategy/stock_mean_reversion.py
class StockMeanReversionStrategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__("Stock_Mean_Reversion", config)
        self.default_config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_volume': 1000000,  # Min daily volume
            'enable_shorts': True,
            'profit_target': 0.02,   # 2% profit target
            'stop_loss': 0.01,       # 1% stop loss
            'max_position_pct': 0.20 # Max 20% per position
        }
    
    def generate_signal(self, data, symbol):
        """Generate long/short signals based on mean reversion"""
        if not self.can_generate_signal(data):
            return None
            
        latest = data.iloc[-1]
        
        # Skip if low volume
        if latest['volume'] < self.config['min_volume']:
            return None
        
        # Bollinger Band position
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        
        # Long signal: Oversold + touching lower band
        if (bb_position < 0.1 and 
            latest['rsi'] < self.config['rsi_oversold'] and
            self.is_flat(symbol)):
            
            return TradingSignal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(),
                price=latest['close'],
                confidence=0.7,
                reason="Mean reversion long: BB oversold + RSI oversold"
            )
        
        # Short signal: Overbought + touching upper band
        elif (bb_position > 0.9 and 
              latest['rsi'] > self.config['rsi_overbought'] and
              self.is_flat(symbol) and
              self.config['enable_shorts']):
            
            return TradingSignal(
                signal_type=SignalType.SELL_SHORT,  # New signal type
                symbol=symbol,
                timestamp=datetime.now(),
                price=latest['close'],
                confidence=0.7,
                reason="Mean reversion short: BB overbought + RSI overbought"
            )
        
        # Exit conditions
        return self._check_exit_conditions(latest, symbol)
```

#### B. Momentum Strategy for Stocks
```python
# src/strategy/stock_momentum.py
class StockMomentumStrategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__("Stock_Momentum", config)
        self.default_config = {
            'lookback_period': 20,
            'min_momentum': 0.10,    # 10% minimum move
            'volume_surge': 2.0,     # 2x average volume
            'atr_multiplier': 2.0,   # ATR-based stops
            'enable_shorts': True,
            'trailing_stop_pct': 0.02 # 2% trailing stop
        }
```

### 6. Market Hours Integration

#### A. Enhanced Main Loop
```python
# main.py modifications
def _execute_trading_cycle(self):
    """Execute one complete trading cycle for all symbols"""
    for symbol in self.trading_symbols:
        try:
            # Check if market is open for this symbol
            if not self.data_provider.is_market_open(symbol):
                is_crypto = self.data_provider.is_crypto_symbol(symbol)
                symbol_type = "crypto" if is_crypto else "stock"
                
                # Log only every 10 cycles
                if self.cycle_count % 10 == 0:
                    self.logger.info(f"⏰ Market closed for {symbol_type} {symbol}")
                continue
            
            # Apply appropriate strategy based on symbol type
            if self.data_provider.is_crypto_symbol(symbol):
                self._execute_crypto_cycle(symbol)
            else:
                self._execute_stock_cycle(symbol)
                
        except Exception as e:
            self.logger.error(f"Error in cycle for {symbol}: {e}")
```

### 7. Risk Management Extensions

#### A. PDT Rule Compliance
```python
# src/risk/pdt_manager.py
class PDTManager:
    def __init__(self):
        self.day_trades = []
        self.pdt_threshold = 25000  # $25k minimum
    
    def check_pdt_compliance(self, account_info, new_trade):
        """Ensure PDT rules are followed"""
        if account_info['portfolio_value'] >= self.pdt_threshold:
            return True  # No PDT restrictions
        
        # Count day trades in rolling 5 days
        day_trades_count = self.count_day_trades()
        
        if day_trades_count >= 3 and self.is_day_trade(new_trade):
            return False  # Would violate PDT rule
            
        return True
```

#### B. Short Selling Risk Management
```python
# src/risk/short_risk_manager.py
class ShortRiskManager:
    def __init__(self):
        self.max_short_exposure = 0.5  # Max 50% short exposure
        self.hard_to_borrow_list = []  # HTB stocks to avoid
        
    def can_short(self, symbol, size, account_info):
        """Check if short selling is allowed"""
        # Check if stock is available to short
        if symbol in self.hard_to_borrow_list:
            return False, "Stock is hard to borrow"
        
        # Check short exposure limits
        current_short_exposure = self.calculate_short_exposure(account_info)
        new_exposure = (size * self.get_current_price(symbol)) / account_info['portfolio_value']
        
        if current_short_exposure + new_exposure > self.max_short_exposure:
            return False, "Exceeds maximum short exposure"
            
        return True, "Short allowed"
```

### 8. Configuration Updates

#### A. Environment Variables
```env
# Stock Trading Configuration
ENABLE_STOCK_TRADING=true
STOCK_SYMBOLS=AAPL,TSLA,SPY,QQQ,NVDA
STOCK_STRATEGY=mean_reversion,momentum
ENABLE_SHORT_SELLING=true
MAX_SHORT_EXPOSURE=0.5
STOCK_RISK_PER_TRADE=0.01
```

#### B. Settings Extension
```python
# config/settings.py
class StockTradingSettings(BaseSettings):
    """Stock trading configuration"""
    enable_stock_trading: bool = Field(default=False, env="ENABLE_STOCK_TRADING")
    stock_symbols: str = Field(default="SPY,QQQ", env="STOCK_SYMBOLS")
    stock_strategies: str = Field(default="mean_reversion", env="STOCK_STRATEGY")
    enable_short_selling: bool = Field(default=True, env="ENABLE_SHORT_SELLING")
    max_short_exposure: float = Field(default=0.5, env="MAX_SHORT_EXPOSURE")
    stock_risk_per_trade: float = Field(default=0.01, env="STOCK_RISK_PER_TRADE")
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
1. Extend symbol management to support stocks
2. Add short selling to order types
3. Implement buying power manager
4. Add stock fee calculations

### Phase 2: Strategies (Week 2)
1. Implement mean reversion strategy
2. Add momentum strategy
3. Integrate with existing signal generation
4. Test paper trading

### Phase 3: Risk & Compliance (Week 3)
1. Implement PDT rule checking
2. Add short selling risk management
3. Enhance position tracking for shorts
4. Add comprehensive logging

### Phase 4: Integration & Testing (Week 4)
1. Full integration testing
2. Performance optimization
3. Deploy to paper trading
4. Monitor and adjust

## Key Benefits

1. **Unified System**: One codebase for both crypto and stocks
2. **Market Diversification**: Trade stocks during day, crypto 24/7
3. **Capital Efficiency**: Shared buying power and risk management
4. **Strategy Flexibility**: Different strategies for different markets
5. **Compliance Built-in**: PDT rules and short selling restrictions

## Risk Considerations

1. **Leverage Risk**: Higher leverage on stocks (4x vs 3x crypto)
2. **Short Squeeze Risk**: Unlimited loss potential on shorts
3. **Gap Risk**: Stocks can gap overnight
4. **Regulatory Risk**: PDT rules and other regulations
5. **Liquidity Risk**: Some stocks may have low liquidity

## Monitoring & Alerts

1. **Position Monitoring**: Track long/short exposure by asset class
2. **Margin Monitoring**: Alert when approaching margin calls
3. **PDT Monitoring**: Track day trades for accounts under $25k
4. **Performance Tracking**: Separate metrics for stocks vs crypto
5. **Risk Alerts**: Real-time alerts for risk limit breaches