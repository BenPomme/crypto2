"""
Main Trading Bot Entry Point
Orchestrates all components of the crypto trading system
"""
import sys
import time
import signal
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Add src to path
sys.path.append('src')

from config.settings import get_settings
from src.utils.logger import setup_logging
from src.data.market_data import AlpacaDataProvider
from src.data.data_buffer import DataBuffer
from src.strategy.feature_engineering import FeatureEngineer
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.strategy.parameter_manager import ParameterManager
from src.risk.risk_manager import RiskManager
from src.execution.trade_executor import TradeExecutor
from src.monitoring.performance_tracker import PerformanceTracker

class CryptoTradingBot:
    """
    Main crypto trading bot class
    Orchestrates all trading system components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading bot
        
        Args:
            config: Bot configuration
        """
        # Load settings
        self.settings = get_settings()
        self.config = config or {}
        
        # Setup logging
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', 'logs/trading_bot.log')
        setup_logging(log_level=log_level, log_file=log_file)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Crypto Trading Bot")
        
        # Trading state
        self.running = False
        self.last_signal_time = None
        self.cycle_count = 0
        
        # Initialize components
        self._initialize_components()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Crypto Trading Bot initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all trading system components"""
        try:
            # Data provider
            self.data_provider = AlpacaDataProvider()
            
            # Validate connection
            if not self.data_provider.validate_connection():
                raise RuntimeError("Failed to connect to Alpaca API")
            
            # Data buffer
            buffer_size = self.config.get('buffer_size', 1000)
            self.data_buffer = DataBuffer(max_size=buffer_size)
            
            # Parameter manager for ML-optimizable parameters
            self.parameter_manager = ParameterManager()
            
            # Feature engineer
            feature_config = {
                'ma_fast': self.settings.trading.fast_ma_period,
                'ma_slow': self.settings.trading.slow_ma_period,
                'rsi_period': self.settings.trading.rsi_period,
            }
            self.feature_engineer = FeatureEngineer(feature_config)
            
            # Strategy with dynamic parameters - Support multiple symbols
            symbols = [s.strip() for s in self.settings.trading.symbol.split(',')]
            self.trading_symbols = symbols
            logger.info(f"Trading symbols: {symbols}")
            
            strategy_config = {
                'symbol': symbols[0],  # Primary symbol for strategy
                'volume_confirmation': True,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'min_periods': 30,  # Reduce from 50 to start trading sooner
                'min_confidence': 0.4,  # Lower threshold for more signals
                'rsi_oversold': self.settings.trading.rsi_oversold,
                'rsi_overbought': self.settings.trading.rsi_overbought,
                'volume_threshold': 1.2,  # 20% above average volume
            }
            self.strategy = MACrossoverStrategy(strategy_config, self.parameter_manager)
            
            # Risk manager - More aggressive for 5-10% monthly target
            risk_config = {
                'max_position_size': 0.5,  # 50% max position for single pair strategy
                'max_daily_loss': 0.08,    # 8% max daily loss
                'risk_per_trade': 0.04,    # 4% risk per trade (higher for target ROI)
            }
            self.risk_manager = RiskManager(risk_config)
            
            # Trade executor - Optimized for target ROI
            executor_config = {
                'enable_stop_loss': True,
                'enable_take_profit': True,
                'default_stop_loss_pct': 0.04,  # Tighter stop loss (4%)
                'default_take_profit_pct': 0.15, # Higher take profit (15%)
            }
            self.trade_executor = TradeExecutor(self.risk_manager, executor_config)
            
            # Performance tracker - get actual account balance
            try:
                account_info = self.data_provider.get_account_info()
                initial_capital = account_info.get('portfolio_value', 10000.0)
                self.logger.info(f"Using actual account balance: ${initial_capital:,.2f}")
            except Exception as e:
                self.logger.warning(f"Could not get account balance, using default: {e}")
                initial_capital = self.config.get('initial_capital', 10000.0)
            
            self.performance_tracker = PerformanceTracker(initial_capital)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start(self) -> None:
        """Start the trading bot"""
        self.logger.info("Starting Crypto Trading Bot")
        
        try:
            # Load initial market data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._load_initial_data()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Data loading attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(10)
                    else:
                        raise
            
            self.logger.info("✅ Trading bot successfully initialized and ready")
            
            # Start main trading loop
            self.running = True
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Critical error in trading bot: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self._shutdown()
    
    def _load_initial_data(self) -> None:
        """Load initial historical data"""
        try:
            symbol = self.settings.trading.symbol
            timeframe = self.settings.trading.data_timeframe
            periods = self.settings.trading.lookback_periods
            
            self.logger.info(f"Loading initial data: {periods} periods of {timeframe} for {symbol}")
            
            # Get historical data
            historical_data = self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                periods=periods
            )
            
            if historical_data.empty:
                raise RuntimeError("No historical data received")
            
            # Add to buffer
            self.data_buffer.bulk_add(historical_data)
            
            self.logger.info(f"Loaded {len(historical_data)} bars of historical data")
            
        except Exception as e:
            self.logger.error(f"Failed to load initial data: {e}")
            raise
    
    def _main_loop(self) -> None:
        """Main trading loop"""
        self.logger.info("Starting main trading loop")
        
        # Trading loop configuration
        loop_interval = self.config.get('loop_interval', 60)  # 1 minute
        self.logger.info(f"Trading loop will run every {loop_interval} seconds")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Execute one trading cycle
                self.logger.debug(f"Starting trading cycle {self.cycle_count + 1}")
                self._execute_trading_cycle()
                
                # Increment cycle counter
                self.cycle_count += 1
                self.logger.info(f"Completed trading cycle {self.cycle_count}")
                
                # Log periodic status
                if self.cycle_count % 10 == 0:
                    self._log_status()
                
                # Optimize parameters periodically (every 100 cycles)
                if self.cycle_count % 100 == 0 and self.cycle_count > 0:
                    self._optimize_parameters()
                
            except Exception as e:
                self.logger.error(f"Error in trading cycle {self.cycle_count}: {e}")
                self.logger.error(traceback.format_exc())
            
            # Sleep until next cycle
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - elapsed)
            
            self.logger.debug(f"Cycle took {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle for all symbols"""
        try:
            # Cycle through all trading symbols
            for symbol in self.trading_symbols:
                self._execute_symbol_cycle(symbol)
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            raise
    
    def _execute_symbol_cycle(self, symbol: str) -> None:
        """Execute trading cycle for a specific symbol"""
        try:
            # Step 1: Get latest market data
            self.logger.debug(f"Getting latest price for {symbol}")
            latest_price = self.data_provider.get_latest_price(symbol)
            self.logger.debug(f"{symbol} latest price: ${latest_price:.2f}")
            
            # Create synthetic bar (in production, would get actual OHLCV bar)
            # Use timezone-aware datetime to match Alpaca data
            from datetime import timezone
            current_time = datetime.now(timezone.utc)
            latest_bar = {
                'timestamp': current_time,
                'open': latest_price,
                'high': latest_price,
                'low': latest_price,
                'close': latest_price,
                'volume': 1000  # Placeholder volume
            }
            
            # Add to buffer
            self.data_buffer.add_bar(latest_bar)
            
            # Step 2: Check if we have enough data
            min_periods = self.strategy.get_min_periods()
            if not self.data_buffer.is_ready(min_periods):
                self.logger.debug(f"Insufficient data: {self.data_buffer.size()}/{min_periods}")
                return
            
            # Step 3: Get data and engineer features
            market_data = self.data_buffer.get_dataframe()
            featured_data = self.feature_engineer.engineer_features(market_data)
            
            # Step 4: Generate trading signal
            signal = self.strategy.generate_signal(featured_data)
            
            # Log signal generation details
            if signal:
                self.logger.info(f"🎯 Signal generated: {signal.signal_type.value} @ ${signal.price:.2f} (confidence: {signal.confidence:.2f})")
                self.logger.info(f"Signal reason: {signal.reason}")
            else:
                # Log current MA status for debugging (use INFO level to see in logs)
                if len(featured_data) > 0:
                    latest = featured_data.iloc[-1]
                    if 'sma_fast' in latest and 'sma_slow' in latest:
                        fast_ma = latest['sma_fast']
                        slow_ma = latest['sma_slow']
                        rsi = latest.get('rsi', 'N/A')
                        position = "FLAT" if self.strategy.is_flat() else ("LONG" if self.strategy.is_long() else "SHORT")
                        
                        # Log every 5 cycles to avoid spam
                        if self.cycle_count % 5 == 0:
                            self.logger.info(f"📊 Analysis: Fast MA=${fast_ma:.2f}, Slow MA=${slow_ma:.2f}, RSI={rsi}, Position={position}")
                            
                            # Check if we're close to a crossover
                            ma_diff = fast_ma - slow_ma
                            if abs(ma_diff) < 5:  # Within $5 of crossover
                                self.logger.info(f"🔥 Close to crossover! MA difference: ${ma_diff:.2f}")
                else:
                    self.logger.debug("No featured data available for analysis")
            
            if signal:
                # Record signal
                self.performance_tracker.record_signal(signal.to_dict())
                self.last_signal_time = current_time
                
                # Step 5: Execute signal if generated
                account_info = self.data_provider.get_account_info()
                current_positions = self.data_provider.get_positions()
                
                execution_result = self.trade_executor.execute_signal(
                    signal=signal,
                    account_info=account_info,
                    current_positions=current_positions,
                    market_data=featured_data
                )
                
                # Record trade result
                self.performance_tracker.record_trade(execution_result)
                
                # Update strategy position if trade was successful
                if execution_result.get('success'):
                    self.strategy.update_position(signal)
            
            # Step 6: Update performance tracking
            account_info = self.data_provider.get_account_info()
            current_positions = self.data_provider.get_positions()
            current_portfolio_value = account_info.get('portfolio_value', 0)
            
            self.performance_tracker.update_portfolio_value(
                current_portfolio_value, account_info, current_positions
            )
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            raise
    
    def _log_status(self) -> None:
        """Log current bot status"""
        try:
            # Get current performance
            performance = self.performance_tracker.get_current_performance()
            
            # Get account info
            account_info = self.data_provider.get_account_info()
            
            # Get buffer stats
            buffer_stats = self.data_buffer.get_stats()
            
            # Get execution stats
            execution_stats = self.trade_executor.get_execution_stats()
            
            self.logger.info(
                f"Status - Cycle: {self.cycle_count}, "
                f"Portfolio: ${account_info.get('portfolio_value', 0):.2f}, "
                f"Return: {performance.get('total_return', 0):.2%}, "
                f"Trades: {performance.get('num_trades', 0)}, "
                f"Win Rate: {performance.get('win_rate', 0):.1%}, "
                f"Buffer: {buffer_stats['size']}/{buffer_stats['max_size']}, "
                f"Success Rate: {execution_stats.get('success_rate', 0):.1%}"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
    
    def _optimize_parameters(self) -> None:
        """Optimize trading parameters based on recent performance"""
        try:
            # Get recent performance data
            recent_trades = self.performance_tracker.trades_log[-50:]  # Last 50 trades
            
            if len(recent_trades) < 10:
                self.logger.info("Insufficient trades for parameter optimization")
                return
            
            # Get current performance metrics
            current_performance = self.performance_tracker.get_current_performance()
            
            # Optimize parameters
            optimized_params = self.parameter_manager.optimize_parameters(recent_trades)
            
            # Update parameter manager with new parameters and performance
            self.parameter_manager.update_parameters(optimized_params, current_performance)
            
            # Update strategy with new parameters
            self.strategy.update_dynamic_parameters()
            
            self.logger.info(f"Parameters optimized - Cycle: {self.cycle_count}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown")
        self.running = False
    
    def _shutdown(self) -> None:
        """Graceful shutdown"""
        self.logger.info("Shutting down trading bot")
        
        try:
            # Cancel all open orders
            cancelled_orders = self.trade_executor.cancel_all_orders()
            if cancelled_orders > 0:
                self.logger.info(f"Cancelled {cancelled_orders} open orders")
            
            # Stop performance monitoring
            self.performance_tracker.stop_monitoring()
            
            # Export final performance report
            final_report = self.performance_tracker.export_performance_report()
            
            # Log final statistics
            summary = final_report.get('summary', {})
            self.logger.info(
                f"Final Performance - "
                f"Return: {summary.get('total_return', 0):.2%}, "
                f"P&L: ${summary.get('total_pnl', 0):.2f}, "
                f"Trades: {summary.get('num_trades', 0)}, "
                f"Win Rate: {summary.get('win_rate', 0):.1%}, "
                f"Sharpe: {summary.get('sharpe_ratio', 0):.2f}, "
                f"Max DD: {summary.get('max_drawdown', 0):.2%}"
            )
            
            self.logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'running': self.running,
            'cycle_count': self.cycle_count,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'buffer_size': self.data_buffer.size() if hasattr(self, 'data_buffer') else 0,
            'performance': self.performance_tracker.get_current_performance() if hasattr(self, 'performance_tracker') else {}
        }

def main():
    """Main entry point"""
    # Configuration
    config = {
        'log_level': 'INFO',
        'log_file': 'logs/crypto_trading_bot.log',
        'loop_interval': 60,  # 1 minute cycles
        'buffer_size': 1000,
        'initial_capital': 10000.0,
    }
    
    # Create and start bot
    bot = CryptoTradingBot(config)
    bot.start()

if __name__ == "__main__":
    main()