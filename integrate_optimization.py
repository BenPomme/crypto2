#!/usr/bin/env python3
"""
Integration Helper for Live Trading System
Shows how to integrate optimized parameters with your existing main.py
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OptimizedParameterManager:
    """
    Manages optimized parameters for live trading
    Integrates with your existing trading system
    """
    
    def __init__(self, parameters_file: str = "optimized_parameters.json"):
        self.parameters_file = parameters_file
        self.parameters = {}
        self.last_update = None
        self.load_parameters()
    
    def load_parameters(self):
        """Load optimized parameters from file"""
        try:
            with open(self.parameters_file, 'r') as f:
                data = json.load(f)
                
                # Support both formats
                if 'live_trading_parameters' in data:
                    # New format (optimized_parameters.json)
                    self.parameters = data.get('live_trading_parameters', {})
                    self.last_update = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
                elif 'pairs' in data:
                    # Demo format (demo_optimized_parameters.json)
                    pairs = data.get('pairs', {})
                    self.parameters = {}
                    for symbol, pair_data in pairs.items():
                        self.parameters[symbol] = {
                            'parameters': pair_data.get('parameters', {}),
                            'confidence': pair_data.get('confidence', 0.0),
                            'sharpe_ratio': pair_data.get('performance', {}).get('sharpe_ratio', 0.0),
                            'source': 'optimization'
                        }
                    self.last_update = datetime.fromisoformat(data.get('optimization_date', datetime.now().isoformat()))
                else:
                    logger.warning("Unknown parameter file format")
                    self.parameters = {}
                    
                logger.info(f"Loaded parameters for {len(self.parameters)} symbols")
        except FileNotFoundError:
            logger.warning(f"Parameters file {self.parameters_file} not found")
            self.parameters = {}
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.parameters = {}
    
    def get_parameters_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get optimized parameters for a specific symbol"""
        if symbol in self.parameters:
            params = self.parameters[symbol]
            logger.info(f"Using optimized parameters for {symbol} (confidence: {params['confidence']:.2f})")
            return params['parameters']
        else:
            logger.warning(f"No optimized parameters for {symbol}, using defaults")
            return self.get_default_parameters()
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Fallback default parameters"""
        return {
            'fast_ma_period': 10,
            'slow_ma_period': 30,
            'risk_per_trade': 0.015,
            'confidence_threshold': 0.6,
            'volume_threshold': 1.5,
            'mfi_oversold': 20,
            'mfi_overbought': 80,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0
        }
    
    def should_update_parameters(self) -> bool:
        """Check if parameters need updating (older than 1 week)"""
        if not self.last_update:
            return True
        
        age = datetime.now() - self.last_update
        return age > timedelta(days=7)
    
    def get_confidence_score(self, symbol: str) -> float:
        """Get confidence score for symbol's parameters"""
        if symbol in self.parameters:
            return self.parameters[symbol]['confidence']
        return 0.0

# Example integration with your existing main.py
def integrate_with_main_trading_system():
    """
    Example of how to integrate with your existing main.py
    """
    print("🔗 Integration Example with Live Trading System")
    print("=" * 55)
    
    # Initialize parameter manager
    param_manager = OptimizedParameterManager()
    
    # Example symbols from your trading system
    trading_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    print("📊 Current Optimized Parameters:")
    print("-" * 40)
    
    for symbol in trading_symbols:
        params = param_manager.get_parameters_for_symbol(symbol)
        confidence = param_manager.get_confidence_score(symbol)
        
        print(f"\n🪙 {symbol}:")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Fast MA: {params['fast_ma_period']}")
        print(f"   Slow MA: {params['slow_ma_period']}")
        print(f"   Risk per Trade: {params['risk_per_trade']:.3f}")
        print(f"   Confidence Threshold: {params['confidence_threshold']:.2f}")
    
    # Check if parameters need updating
    if param_manager.should_update_parameters():
        print(f"\n⚠️  Parameters are older than 1 week - consider re-optimization")
    else:
        print(f"\n✅ Parameters are up to date")
    
    return param_manager

def show_main_py_integration():
    """Show how to modify main.py to use optimized parameters"""
    
    integration_code = '''
# Add this to your main.py

from integrate_optimization import OptimizedParameterManager

class EnhancedTradingBot:
    def __init__(self):
        # Your existing initialization
        self.data_provider = AlpacaDataProvider()
        self.strategy = MACrossoverStrategy()
        # ... other components ...
        
        # Add parameter manager
        self.param_manager = OptimizedParameterManager()
        self.last_param_update = datetime.now()
    
    async def run_trading_session(self):
        """Enhanced trading session with adaptive parameters"""
        
        # Update parameters weekly
        if self.should_update_parameters():
            await self.update_strategy_parameters()
        
        # Your existing trading logic
        for symbol in self.symbols:
            # Get optimized parameters for this symbol
            params = self.param_manager.get_parameters_for_symbol(symbol)
            confidence = self.param_manager.get_confidence_score(symbol)
            
            # Update strategy with optimized parameters
            self.strategy.update_parameters(params)
            
            # Adjust risk based on parameter confidence
            if confidence > 0.7:
                # High confidence - use full risk
                risk_multiplier = 1.0
            elif confidence > 0.5:
                # Medium confidence - reduce risk
                risk_multiplier = 0.7
            else:
                # Low confidence - conservative risk
                risk_multiplier = 0.5
            
            # Apply risk adjustment
            adjusted_params = params.copy()
            adjusted_params['risk_per_trade'] *= risk_multiplier
            self.strategy.update_parameters(adjusted_params)
            
            # Continue with your existing trading logic
            await self.execute_trading_logic(symbol)
    
    def should_update_parameters(self) -> bool:
        """Check if we should run new optimization"""
        age = datetime.now() - self.last_param_update
        return age > timedelta(days=7)  # Weekly updates
    
    async def update_strategy_parameters(self):
        """Run optimization and update parameters"""
        logger.info("🔧 Updating strategy parameters...")
        
        # Run optimization (this could be in a separate process/container)
        from run_optimization import run_crypto_optimization
        await run_crypto_optimization()
        
        # Reload parameters
        self.param_manager.load_parameters()
        self.last_param_update = datetime.now()
        
        logger.info("✅ Strategy parameters updated")

# Usage in your main function:
async def main():
    bot = EnhancedTradingBot()
    await bot.run_trading_session()
'''
    
    print("\n📝 MAIN.PY INTEGRATION CODE:")
    print("=" * 50)
    print(integration_code)

def show_deployment_strategy():
    """Show deployment strategy for production"""
    
    deployment_info = '''
🚀 PRODUCTION DEPLOYMENT STRATEGY
================================

1. Parameter Optimization Schedule:
   ⏰ Run optimization weekly (Sunday nights)
   📊 Use 3-6 months of historical data
   🧠 Train ML models on results
   💾 Save parameters to shared storage

2. Live Trading Integration:
   📈 Load optimized parameters at startup
   🔄 Check for parameter updates every hour
   ⚡ Apply new parameters during low-activity periods
   📊 Monitor performance continuously

3. Railway Deployment:
   🔧 Deploy optimization as separate service
   ⏰ Schedule with cron jobs or Railway schedules
   💾 Store results in Firebase/database
   📡 Notify main trading bot of updates

4. Risk Management:
   🛡️  Validate parameters before applying
   📉 Reduce position sizes for low-confidence parameters
   🚨 Fall back to proven defaults if needed
   📊 Monitor performance and rollback if necessary

5. Monitoring:
   📈 Track parameter performance in real-time
   🧠 Monitor ML model confidence scores
   ⚠️  Alert on parameter degradation
   📊 Generate performance reports

Example Railway Deployment:
---------------------------
# railway.toml
[environments.optimization]
  [environments.optimization.variables]
    OPTIMIZATION_SCHEDULE = "0 2 * * 0"  # Sunday 2 AM
    SYMBOLS = "BTC/USD,ETH/USD,SOL/USD"
    
[environments.trading]
  [environments.trading.variables]
    PARAMETER_UPDATE_INTERVAL = "3600"  # Check hourly
'''
    
    print(deployment_info)

if __name__ == "__main__":
    print("🔗 Live Trading System Integration Guide")
    print("=" * 55)
    
    # Show current parameters
    integrate_with_main_trading_system()
    
    # Show integration code
    show_main_py_integration()
    
    # Show deployment strategy
    show_deployment_strategy()
    
    print("\n✅ Integration guide complete!")
    print("📝 Next steps:")
    print("   1. Run 'python3 run_optimization.py' to generate parameters")
    print("   2. Update your main.py with the integration code above")
    print("   3. Deploy optimization service to Railway")
    print("   4. Monitor performance and adjust as needed")