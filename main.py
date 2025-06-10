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
from src.data.volume_data_manager import VolumeDataManager
from src.strategy.feature_engineering import FeatureEngineer
from src.strategy.ma_crossover_strategy import MACrossoverStrategy
from src.strategy.parameter_manager import ParameterManager
from integrate_optimization import OptimizedParameterManager
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
            
            # Parse trading symbols - Support multiple symbols
            symbols = [s.strip() for s in self.settings.trading.symbol.split(',')]
            self.trading_symbols = symbols
            self.logger.info(f"Trading symbols: {symbols}")
            
            # Volume data manager for accurate crypto volume
            self.volume_manager = VolumeDataManager(trading_symbols=symbols)
            
            # Data buffers - one per symbol
            buffer_size = self.config.get('buffer_size', 1000)
            self.data_buffers = {}
            for symbol in symbols:
                self.data_buffers[symbol] = DataBuffer(max_size=buffer_size)
                self.logger.info(f"Created data buffer for {symbol}")
            
            # Feature engineer
            feature_config = {
                'ma_fast': self.settings.trading.fast_ma_period,
                'ma_slow': self.settings.trading.slow_ma_period,
                'rsi_period': self.settings.trading.rsi_period,
            }
            self.feature_engineer = FeatureEngineer(feature_config)
            
            # Strategy will be initialized after parameter manager
            
            # Risk manager - Multi-symbol risk settings (4 crypto pairs)
            risk_config = {
                'max_position_size': 0.4,  # 40% max position per symbol (4 symbols = 160% total exposure)
                'max_daily_loss': 0.06,    # 6% max daily loss
                'risk_per_trade': 0.02,    # 2% risk per trade
            }
            self.risk_manager = RiskManager(risk_config)
            
            # Trade executor - Leverage-optimized targets
            executor_config = {
                'enable_stop_loss': True,
                'enable_take_profit': True,
                'default_stop_loss_pct': 0.02,  # Very tight stop loss (2%)
                'default_take_profit_pct': 0.06, # Conservative take profit (6%)
                # With leverage: 6% on $100k = $6k profit, 2% risk = $2k loss
                # Risk/Reward = 3:1 ratio, much higher win probability
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
            
            # Parameter manager for ML-optimizable parameters (after Firebase initialization)
            self.parameter_manager = ParameterManager(firebase_logger=self.performance_tracker.firebase_logger)
            
            # Optimized parameter manager - loads parameters from optimization results
            # Use demo file if available (has BTC, ETH, SOL, DOGE), otherwise use main file
            param_file = "demo_optimized_parameters.json" if self.config.get('use_demo_params', True) else "optimized_parameters.json"
            self.optimized_param_manager = OptimizedParameterManager(param_file)
            self.logger.info(f"Using optimization parameters from: {param_file}")
            
            # Strategy initialization (after parameter manager)
            strategy_config = {
                'volume_confirmation': True,  # Re-enable volume confirmation with fixed indicators
                'stop_loss_pct': 0.02,  # Match executor config
                'take_profit_pct': 0.06,  # Match executor config
                'min_periods': 20,  # Faster signal generation
                'min_confidence': 0.2,  # Reasonable threshold with working volume indicators
                'rsi_oversold': 30.0,  # Standard RSI levels
                'rsi_overbought': 80.0,  # Higher threshold for strong uptrends
                'volume_threshold': 1.2,  # Standard volume requirement (should work now)
                'enable_trend_following': True,  # Enable trend continuation signals
            }
            self.strategy = MACrossoverStrategy(strategy_config, self.parameter_manager)
            
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
            
            # Start volume data collection
            self.volume_manager.start()
            
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
    
    def _shutdown(self) -> None:
        """Gracefully shutdown the trading bot"""
        self.logger.info("Shutting down trading bot...")
        self.running = False
        
        # Stop volume data collection
        if hasattr(self, 'volume_manager'):
            self.volume_manager.stop()
        
        self.logger.info("Trading bot shutdown complete")
    
    def _log_volume_status(self) -> None:
        """Log volume data status for monitoring"""
        try:
            if not hasattr(self, 'volume_manager'):
                return
                
            status = self.volume_manager.get_status()
            summary = self.volume_manager.get_volume_summary()
            
            self.logger.info("📊 REAL-TIME VOLUME STATUS:")
            self.logger.info(f"   Binance WebSocket: {'✅ Connected' if status['binance_provider']['connected'] else '❌ Disconnected'}")
            self.logger.info(f"   Symbols Tracking: {status['binance_provider']['symbols_tracking']}")
            
            for symbol, data in summary.items():
                if data['data_available']:
                    volume_24h = data['volume_24h_usd']
                    volume_min = data['volume_per_minute_usd']
                    self.logger.info(f"   {symbol}: ${volume_24h:,.0f}/24h (${volume_min:,.0f}/min)")
                else:
                    self.logger.warning(f"   {symbol}: No volume data available")
                    
        except Exception as e:
            self.logger.error(f"Error logging volume status: {e}")
    
    def _load_initial_data(self) -> None:
        """Load initial historical data for all trading symbols"""
        try:
            timeframe = self.settings.trading.data_timeframe
            periods = self.settings.trading.lookback_periods
            
            # Load historical data for each symbol
            for symbol in self.trading_symbols:
                self.logger.info(f"Loading initial data: {periods} periods of {timeframe} for {symbol}")
                
                try:
                    # Get historical data for this symbol
                    historical_data = self.data_provider.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        periods=periods
                    )
                    
                    if historical_data.empty:
                        self.logger.warning(f"No historical data received for {symbol}")
                        continue
                    
                    # Add to symbol-specific buffer
                    self.data_buffers[symbol].bulk_add(historical_data)
                    
                    self.logger.info(f"Loaded {len(historical_data)} bars for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load data for {symbol}: {e}")
                    # Continue with other symbols
            
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
                
                # Log volume data status every 50 cycles
                if self.cycle_count % 50 == 0:
                    self._log_volume_status()
                
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
            # Log symbol processing every 20 cycles to track missing symbols
            if self.cycle_count % 20 == 0:
                self.logger.info(f"🔄 Processing {len(self.trading_symbols)} symbols: {self.trading_symbols}")
            
            # Cycle through all trading symbols
            for symbol in self.trading_symbols:
                try:
                    self._execute_symbol_cycle(symbol)
                except Exception as e:
                    # Log error but continue with other symbols
                    self.logger.error(f"❌ Error in trading cycle for {symbol}: {e}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            raise
    
    def _execute_symbol_cycle(self, symbol: str) -> None:
        """Execute trading cycle for a specific symbol"""
        try:
            # Step 0: Check if market is open for this symbol
            if not self.data_provider.is_market_open(symbol):
                is_crypto = self.data_provider.is_crypto_symbol(symbol)
                symbol_type = "crypto" if is_crypto else "stock"
                
                # Log market closure (but only every 10 cycles to avoid spam)
                if self.cycle_count % 10 == 0:
                    self.logger.info(f"⏰ Market closed for {symbol_type} symbol {symbol}, skipping trading cycle")
                
                return
            
            # Step 1: Use existing buffer data (no need to fetch new bars every cycle)
            # The buffers already have historical data from initialization
            if symbol not in self.data_buffers:
                self.logger.warning(f"No data buffer for {symbol}, skipping")
                return
            
            # Step 2: Check if we have enough data for this symbol
            min_periods = self.strategy.get_min_periods()
            if not self.data_buffers[symbol].is_ready(min_periods):
                self.logger.debug(f"Insufficient data for {symbol}: {self.data_buffers[symbol].size()}/{min_periods}")
                return
            
            # Step 3: Apply optimized parameters for this symbol (if available)
            optimized_params = self.optimized_param_manager.get_parameters_for_symbol(symbol)
            if optimized_params:
                self.strategy.update_parameters(optimized_params)
            
            # Step 4: Get data and enhance with real volume data
            market_data = self.data_buffers[symbol].get_dataframe()
            
            # Enhance volume data with Binance real-time data
            enhanced_data = self.volume_manager.enhance_dataframe_volume(market_data, symbol)
            
            # Engineer features with enhanced data
            feature_config = {
                'ma_fast': self.strategy.fast_period,
                'ma_slow': self.strategy.slow_period
            }
            featured_data = self.feature_engineer.engineer_features(enhanced_data, custom_config=feature_config)
            
            # Step 5: Generate trading signal for this symbol
            signal = self.strategy.generate_signal(featured_data, symbol)
            
            # Log signal generation details
            if signal:
                self.logger.info(f"🎯 {symbol} Signal: {signal.signal_type.value} @ ${signal.price:.2f} (confidence: {signal.confidence:.2f})")
                self.logger.info(f"Signal reason: {signal.reason}")
            else:
                # Log current MA status for debugging (use INFO level to see in logs)
                if len(featured_data) > 0:
                    latest = featured_data.iloc[-1]
                    if 'sma_fast' in latest and 'sma_slow' in latest:
                        fast_ma = latest['sma_fast']
                        slow_ma = latest['sma_slow']
                        rsi = latest.get('rsi', 'N/A')
                        position = "FLAT" if self.strategy.is_flat(symbol) else ("LONG" if self.strategy.is_long(symbol) else "SHORT")
                        
                        # Enhanced indicators status
                        obv = latest.get('obv', 'N/A')
                        obv_trend = "UP" if latest.get('obv_trend', 0) == 1 else "DOWN"
                        mfi = latest.get('mfi', 'N/A')
                        macd = latest.get('macd', 'N/A')
                        macd_signal = latest.get('macd_signal', 'N/A')
                        macd_bullish = "BULL" if latest.get('macd_bullish', 0) == 1 else "BEAR"
                        bb_upper = latest.get('bb_upper', 'N/A')
                        bb_lower = latest.get('bb_lower', 'N/A')
                        
                        # Log every 5 cycles to avoid spam
                        if self.cycle_count % 5 == 0:
                            latest_price = latest['close']
                            self.logger.info(f"📊 {symbol} Technical Analysis:")
                            self.logger.info(f"   Price: ${latest_price:.2f}, Position: {position}")
                            self.logger.info(f"   Moving Averages: Fast=${fast_ma:.2f}, Slow=${slow_ma:.2f}")
                            self.logger.info(f"   RSI: {rsi}")
                            self.logger.info(f"   Volume: OBV={obv}, OBV Trend={obv_trend}, MFI={mfi}")
                            self.logger.info(f"   MACD: {macd:.3f} / Signal={macd_signal:.3f} ({macd_bullish})")
                            self.logger.info(f"   Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f}")
                            
                            # Check if we're close to a crossover
                            ma_diff = fast_ma - slow_ma
                            if abs(ma_diff) < (latest_price * 0.001):  # Within 0.1% of crossover
                                self.logger.info(f"🔥 {symbol} Close to crossover! MA difference: ${ma_diff:.2f}")
                else:
                    self.logger.debug("No featured data available for analysis")
            
            if signal:
                # Record signal
                self.performance_tracker.record_signal(signal.to_dict())
                self.last_signal_time = datetime.now()
                
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
            
            # Get buffer stats for all symbols
            total_buffer_size = sum(buf.size() for buf in self.data_buffers.values())
            buffer_info = ", ".join([f"{sym}:{buf.size()}" for sym, buf in self.data_buffers.items()])
            
            # Get execution stats
            execution_stats = self.trade_executor.get_execution_stats()
            
            # Get market status for symbols
            market_status = []
            for symbol in self.trading_symbols:
                is_open = self.data_provider.is_market_open(symbol)
                is_crypto = self.data_provider.is_crypto_symbol(symbol)
                symbol_type = "crypto" if is_crypto else "stock"
                status = "open" if is_open else "closed"
                market_status.append(f"{symbol}({symbol_type}):{status}")
            
            market_info = ", ".join(market_status)
            
            # Calculate additional metrics
            total_pnl = performance.get('total_pnl', 0)
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = performance.get('max_drawdown', 0)
            
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 TRADING BOT STATUS - Cycle {self.cycle_count}")
            self.logger.info("=" * 80)
            self.logger.info(f"💰 PERFORMANCE METRICS:")
            self.logger.info(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            self.logger.info(f"   Total Return (ROI): {performance.get('total_return', 0):.2%}")
            self.logger.info(f"   Total P&L: ${total_pnl:,.2f}")
            self.logger.info(f"   Total Trades: {performance.get('num_trades', 0)}")
            self.logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1%}")
            self.logger.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            self.logger.info(f"   Max Drawdown: {max_drawdown:.2%}")
            self.logger.info(f"📊 SYSTEM STATUS:")
            self.logger.info(f"   Data Buffers: [{buffer_info}]")
            self.logger.info(f"   Markets: [{market_info}]")
            self.logger.info(f"   Execution Success Rate: {execution_stats.get('success_rate', 0):.1%}")
            self.logger.info("=" * 80)
            
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