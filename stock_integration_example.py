#!/usr/bin/env python3
"""
Stock Trading Integration Example
Shows how to integrate stock trading into the existing main.py
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

# Import new components
from src.risk.buying_power_manager import BuyingPowerManager
from src.strategy.stock_mean_reversion import StockMeanReversionStrategy

logger = logging.getLogger(__name__)

class EnhancedCryptoTradingBot:
    """
    Enhanced trading bot with stock trading capabilities
    This shows the key modifications needed to main.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced trading bot"""
        # ... existing initialization ...
        
        # Add stock symbols to trading symbols
        self._initialize_trading_symbols()
        
        # Initialize buying power manager
        self.buying_power_manager = BuyingPowerManager({
            'crypto_leverage': 3.0,
            'stock_leverage': 4.0,
            'max_short_exposure': 0.5,
            'max_position_pct': 0.25
        })
        
        # Initialize strategies for different asset types
        self._initialize_strategies()
        
        logger.info(f"Enhanced bot initialized with {len(self.trading_symbols)} symbols")
    
    def _initialize_trading_symbols(self):
        """Initialize both crypto and stock symbols"""
        # Parse crypto symbols from settings
        crypto_symbols = [s.strip() for s in self.settings.trading.symbol.split(',')]
        
        # Add stock symbols if enabled
        if self.settings.stock_trading.enable_stock_trading:
            stock_symbols = [s.strip() for s in self.settings.stock_trading.stock_symbols.split(',')]
            self.trading_symbols = crypto_symbols + stock_symbols
        else:
            self.trading_symbols = crypto_symbols
        
        # Separate symbols by type for easier processing
        self.crypto_symbols = [s for s in self.trading_symbols if '/' in s]
        self.stock_symbols = [s for s in self.trading_symbols if '/' not in s]
        
        logger.info(f"Trading symbols: {len(self.crypto_symbols)} crypto, {len(self.stock_symbols)} stocks")
    
    def _initialize_strategies(self):
        """Initialize strategies for different asset types"""
        # Existing crypto strategy
        self.crypto_strategy = MACrossoverStrategy(self.config, self.parameter_manager)
        
        # New stock strategies
        self.stock_strategies = {}
        
        if 'mean_reversion' in self.settings.stock_trading.stock_strategies:
            self.stock_strategies['mean_reversion'] = StockMeanReversionStrategy({
                'enable_shorts': self.settings.stock_trading.enable_short_selling,
                'profit_target': 0.02,
                'stop_loss': 0.01,
                'min_confidence': 0.6
            })
        
        # Could add more strategies here (momentum, pairs trading, etc.)
        
    def _execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle for all symbols"""
        try:
            # Process all symbols
            for symbol in self.trading_symbols:
                try:
                    # Check if market is open
                    if not self.data_provider.is_market_open(symbol):
                        continue
                    
                    # Route to appropriate execution method
                    if self.data_provider.is_crypto_symbol(symbol):
                        self._execute_crypto_cycle(symbol)
                    else:
                        self._execute_stock_cycle(symbol)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _execute_stock_cycle(self, symbol: str) -> None:
        """Execute trading cycle for a stock symbol"""
        try:
            # Check if we have data
            if symbol not in self.data_buffers:
                logger.warning(f"No data buffer for stock {symbol}")
                return
            
            # Get market data
            market_data = self.data_buffers[symbol].get_dataframe()
            if market_data.empty:
                return
            
            # Engineer features (same as crypto)
            featured_data = self.feature_engineer.engineer_features(market_data)
            
            # Try each stock strategy
            for strategy_name, strategy in self.stock_strategies.items():
                # Check if strategy has enough data
                if not self.data_buffers[symbol].is_ready(strategy.get_min_periods()):
                    continue
                
                # Generate signal
                signal = strategy.generate_signal(featured_data, symbol)
                
                if signal:
                    logger.info(f"📈 Stock signal from {strategy_name}: {signal.signal_type.value} {symbol}")
                    
                    # Execute with enhanced risk management
                    self._execute_stock_signal(signal, strategy_name)
                    
                    # Only execute one signal per symbol per cycle
                    break
                    
        except Exception as e:
            logger.error(f"Error in stock cycle for {symbol}: {e}")
    
    def _execute_stock_signal(self, signal: TradingSignal, strategy_name: str) -> Dict[str, Any]:
        """Execute stock trading signal with proper risk management"""
        try:
            # Get account info
            account_info = self.data_provider.get_account_info()
            current_positions = self.data_provider.get_positions()
            
            # Check PDT rules if applicable
            if not self._check_pdt_compliance(signal, account_info):
                logger.warning(f"Signal rejected due to PDT rules: {signal.symbol}")
                return {'success': False, 'reason': 'PDT rule violation'}
            
            # Check buying power
            current_price = signal.price
            
            # Get position sizing recommendation
            sizing = self.buying_power_manager.get_position_sizing_recommendation(
                symbol=signal.symbol,
                side='sell_short' if signal.metadata.get('is_short') else 'buy',
                account_info=account_info,
                risk_pct=self.settings.stock_trading.stock_risk_per_trade
            )
            
            if sizing['recommended_value'] <= 0:
                logger.warning(f"Insufficient buying power for {signal.symbol}")
                return {'success': False, 'reason': 'Insufficient buying power'}
            
            # Calculate shares
            position_value = min(sizing['recommended_value'], account_info['buying_power'] * 0.9)
            shares = int(position_value / current_price)
            
            if shares < 1:
                logger.warning(f"Position too small for {signal.symbol}")
                return {'success': False, 'reason': 'Position size too small'}
            
            # Additional checks for short selling
            if signal.metadata.get('is_short'):
                can_short, reason = self._check_short_availability(signal.symbol, shares, account_info)
                if not can_short:
                    logger.warning(f"Cannot short {signal.symbol}: {reason}")
                    return {'success': False, 'reason': reason}
            
            # Enhanced signal with position sizing
            signal.metadata['position_size'] = shares
            signal.metadata['position_value'] = shares * current_price
            signal.metadata['strategy'] = strategy_name
            
            # Execute through existing infrastructure
            execution_result = self.trade_executor.execute_signal(
                signal=signal,
                account_info=account_info,
                current_positions=current_positions,
                market_data=self.data_buffers[signal.symbol].get_dataframe()
            )
            
            # Update strategy position tracking
            if execution_result.get('success'):
                self.stock_strategies[strategy_name].update_position(signal)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing stock signal: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _check_pdt_compliance(self, signal: TradingSignal, account_info: Dict[str, Any]) -> bool:
        """Check Pattern Day Trader compliance"""
        # Only applies to accounts under $25k
        if account_info['portfolio_value'] >= 25000:
            return True
        
        # Check if this would be a day trade
        # This is simplified - real implementation would track trades properly
        if signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
            # Check if position was opened today
            # Would need to track this properly
            pass
        
        return True  # Simplified for example
    
    def _check_short_availability(self, symbol: str, shares: int, 
                                 account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if stock is available for shorting"""
        # This would integrate with Alpaca's API to check:
        # 1. If stock is shortable
        # 2. If shares are available to borrow
        # 3. Borrow fees
        
        # Simplified check
        hard_to_borrow = ['GME', 'AMC', 'BBBY']  # Example HTB list
        if symbol in hard_to_borrow:
            return False, "Stock is hard to borrow"
        
        return True, "Shortable"
    
    def _log_status(self) -> None:
        """Enhanced status logging with stock positions"""
        try:
            # Get existing performance
            performance = self.performance_tracker.get_current_performance()
            account_info = self.data_provider.get_account_info()
            
            # Get leverage summary
            leverage_summary = self.buying_power_manager.get_leverage_summary(account_info)
            
            # Enhanced logging
            logger.info("=" * 80)
            logger.info(f"🚀 ENHANCED TRADING BOT STATUS - Cycle {self.cycle_count}")
            logger.info("=" * 80)
            logger.info(f"💰 PERFORMANCE METRICS:")
            logger.info(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"   Total Return: {performance.get('total_return', 0):.2%}")
            logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            
            logger.info(f"📊 LEVERAGE SUMMARY:")
            logger.info(f"   Gross Leverage: {leverage_summary['gross_leverage']:.2f}x")
            logger.info(f"   Net Leverage: {leverage_summary['net_leverage']:.2f}x")
            logger.info(f"   Crypto Long: ${leverage_summary['crypto_long']:,.2f}")
            logger.info(f"   Crypto Short: ${leverage_summary['crypto_short']:,.2f}")
            logger.info(f"   Stock Long: ${leverage_summary['stock_long']:,.2f}")
            logger.info(f"   Stock Short: ${leverage_summary['stock_short']:,.2f}")
            
            logger.info(f"📈 POSITION BREAKDOWN:")
            positions = self.data_provider.get_positions()
            for position in positions:
                symbol = position['symbol']
                side = position['side']
                value = position['market_value']
                pnl_pct = position['unrealized_plpc'] * 100
                asset_type = "crypto" if '/' in symbol else "stock"
                logger.info(f"   {symbol} ({asset_type}): {side} ${value:,.2f} ({pnl_pct:+.2f}%)")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error logging enhanced status: {e}")


def show_configuration_example():
    """Show example configuration for stock trading"""
    config_example = """
# Example Railway environment variables for stock trading

# Enable stock trading
ENABLE_STOCK_TRADING=true

# Stock symbols to trade (high volume, liquid stocks)
STOCK_SYMBOLS=SPY,QQQ,AAPL,MSFT,TSLA,NVDA,AMD,META,AMZN,GOOGL

# Stock trading strategies
STOCK_STRATEGY=mean_reversion,momentum

# Risk management
ENABLE_SHORT_SELLING=true
MAX_SHORT_EXPOSURE=0.5
STOCK_RISK_PER_TRADE=0.01
MAX_POSITION_PCT=0.20

# Market hours settings
ENABLE_PREMARKET=false
ENABLE_AFTERHOURS=false

# PDT settings
CHECK_PDT_RULES=true
MIN_EQUITY_FOR_DAYTRADING=25000

# Commission settings
STOCK_COMMISSION=0.0
STOCK_LOCATE_FEE=0.01
"""
    
    print("\n📝 STOCK TRADING CONFIGURATION:")
    print("=" * 50)
    print(config_example)


def show_deployment_notes():
    """Show deployment considerations"""
    notes = """
🚀 DEPLOYMENT CONSIDERATIONS FOR STOCK TRADING
=============================================

1. API Permissions:
   ✅ Ensure Alpaca account has stock trading enabled
   ✅ Enable short selling permissions if needed
   ✅ Request extended hours trading if desired

2. Risk Management:
   ⚠️  Stocks have different margin requirements
   ⚠️  PDT rules apply to accounts under $25k
   ⚠️  Short selling has unlimited risk
   ⚠️  Consider circuit breakers for volatility

3. Data Considerations:
   📊 Stock data may have delays during market hours
   📊 Volume data is critical for liquidity
   📊 Corporate actions (splits, dividends) affect positions
   📊 Halt/resume events need handling

4. Regulatory Compliance:
   📋 Pattern Day Trader (PDT) rules
   📋 Reg T margin requirements
   📋 Short sale restrictions (SSR)
   📋 Wash sale rules for taxes

5. Performance Optimization:
   ⚡ Separate threads for crypto vs stock processing
   ⚡ Cache market hours to reduce API calls
   ⚡ Batch order submissions when possible
   ⚡ Monitor API rate limits carefully

6. Monitoring:
   📈 Track P&L by asset class
   📈 Monitor margin usage closely
   📈 Alert on failed short locates
   📈 Track regulatory warnings
"""
    print(notes)


if __name__ == "__main__":
    print("🎯 STOCK TRADING INTEGRATION GUIDE")
    print("=" * 50)
    
    # Show configuration
    show_configuration_example()
    
    # Show deployment notes
    show_deployment_notes()
    
    print("\n✅ Integration guide complete!")
    print("\n📝 Next steps:")
    print("   1. Update your main.py with the enhanced bot class")
    print("   2. Add stock symbols to your configuration")
    print("   3. Deploy and test with paper trading first")
    print("   4. Monitor carefully during market hours")
    print("   5. Adjust parameters based on performance")