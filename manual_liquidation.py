#!/usr/bin/env python3
"""
Manual Position Liquidation Script
Uses existing trading system components to safely liquidate current positions
"""
import logging
import sys
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data.market_data import AlpacaDataProvider
from src.execution.order_manager import OrderManager, OrderSide, OrderType
from config.settings import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionLiquidator:
    """Manual position liquidation manager"""
    
    def __init__(self):
        """Initialize liquidator with market data and order manager"""
        logger.info("Initializing Position Liquidator...")
        
        try:
            self.market_data = AlpacaDataProvider()
            self.order_manager = OrderManager()
            logger.info("Successfully initialized components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current account positions"""
        try:
            logger.info("Fetching current positions...")
            positions = self.market_data.get_positions()
            
            if not positions:
                logger.info("No positions found")
                return []
            
            logger.info(f"Found {len(positions)} positions:")
            for pos in positions:
                logger.info(f"  {pos['symbol']}: {pos['qty']} shares, "
                          f"Market Value: ${pos['market_value']:.2f}, "
                          f"P&L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2%})")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            logger.info("Fetching account information...")
            account_info = self.market_data.get_account_info()
            
            logger.info(f"Account Info:")
            logger.info(f"  Portfolio Value: ${account_info['portfolio_value']:.2f}")
            logger.info(f"  Cash: ${account_info['cash']:.2f}")
            logger.info(f"  Buying Power: ${account_info['buying_power']:.2f}")
            logger.info(f"  Trading Blocked: {account_info['trading_blocked']}")
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
    
    def liquidate_position(self, symbol: str, quantity: float) -> bool:
        """
        Liquidate a specific position
        
        Args:
            symbol: Symbol to liquidate
            quantity: Quantity to sell (should be positive)
            
        Returns:
            True if liquidation successful
        """
        try:
            logger.info(f"Starting liquidation of {quantity} shares of {symbol}")
            
            # Place market sell order
            order_result = self.order_manager.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=abs(quantity),  # Ensure positive quantity for sell order
                order_type=OrderType.MARKET
            )
            
            if order_result.status.value in ['rejected']:
                logger.error(f"Order rejected: {order_result.error_message}")
                return False
            
            logger.info(f"Sell order placed: {order_result.order_id}")
            logger.info(f"Waiting for order to fill...")
            
            # Wait for order to complete
            final_result = self.order_manager.wait_for_fill(order_result.order_id, timeout=60)
            
            if final_result.status.value == 'filled':
                logger.info(f"Position liquidated successfully!")
                logger.info(f"  Filled Price: ${final_result.filled_price:.4f}")
                logger.info(f"  Filled Quantity: {final_result.filled_quantity}")
                return True
            else:
                logger.warning(f"Order not filled. Final status: {final_result.status.value}")
                if final_result.error_message:
                    logger.warning(f"Error: {final_result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error liquidating position {symbol}: {e}")
            return False
    
    def liquidate_all_positions(self, confirm: bool = False) -> bool:
        """
        Liquidate all current positions
        
        Args:
            confirm: Must be True to actually execute liquidation
            
        Returns:
            True if all positions liquidated successfully
        """
        if not confirm:
            logger.warning("liquidate_all_positions called without confirm=True. Set confirm=True to execute.")
            return False
        
        logger.info("Starting liquidation of ALL positions...")
        
        positions = self.get_current_positions()
        if not positions:
            logger.info("No positions to liquidate")
            return True
        
        liquidation_results = []
        
        for position in positions:
            symbol = position['symbol']
            quantity = abs(float(position['qty']))  # Ensure positive quantity
            
            if quantity > 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Liquidating {symbol}: {quantity} shares")
                logger.info(f"Current Value: ${position['market_value']:.2f}")
                logger.info(f"{'='*50}")
                
                success = self.liquidate_position(symbol, quantity)
                liquidation_results.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'success': success
                })
            else:
                logger.info(f"Skipping {symbol} - zero quantity")
        
        # Summary
        successful = [r for r in liquidation_results if r['success']]
        failed = [r for r in liquidation_results if not r['success']]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"LIQUIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total positions processed: {len(liquidation_results)}")
        logger.info(f"Successfully liquidated: {len(successful)}")
        logger.info(f"Failed to liquidate: {len(failed)}")
        
        if successful:
            logger.info(f"\nSuccessful liquidations:")
            for result in successful:
                logger.info(f"  ✓ {result['symbol']}: {result['quantity']} shares")
        
        if failed:
            logger.warning(f"\nFailed liquidations:")
            for result in failed:
                logger.warning(f"  ✗ {result['symbol']}: {result['quantity']} shares")
        
        return len(failed) == 0
    
    def check_status(self) -> None:
        """Check current account and position status"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ACCOUNT STATUS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # Get account info
        account_info = self.get_account_info()
        
        logger.info(f"\n{'='*30}")
        logger.info(f"CURRENT POSITIONS")
        logger.info(f"{'='*30}")
        
        # Get positions
        positions = self.get_current_positions()
        
        if positions:
            total_value = sum(float(pos['market_value']) for pos in positions)
            total_pl = sum(float(pos['unrealized_pl']) for pos in positions)
            logger.info(f"\nTotal Position Value: ${total_value:.2f}")
            logger.info(f"Total Unrealized P&L: ${total_pl:.2f}")
        
        logger.info(f"\n{'='*60}")

def main():
    """Main liquidation script"""
    liquidator = PositionLiquidator()
    
    print("\n" + "="*60)
    print("MANUAL POSITION LIQUIDATION TOOL")
    print("="*60)
    print("\nThis script will help you liquidate your current positions.")
    print("WARNING: This will sell ALL current positions immediately!")
    print("\nFirst, let's check your current status...\n")
    
    # Check current status
    liquidator.check_status()
    
    # Ask for confirmation
    print("\n" + "="*60)
    print("LIQUIDATION CONFIRMATION")
    print("="*60)
    
    response = input("\nDo you want to liquidate ALL current positions? (type 'YES' to confirm): ")
    
    if response.strip().upper() == 'YES':
        print("\nProceeding with liquidation...")
        success = liquidator.liquidate_all_positions(confirm=True)
        
        if success:
            print("\n✓ All positions liquidated successfully!")
        else:
            print("\n✗ Some positions failed to liquidate. Check logs above.")
        
        # Final status check
        print("\nFinal status check...")
        liquidator.check_status()
        
    else:
        print("\nLiquidation cancelled. No positions were sold.")
        print("Run the script again if you want to liquidate positions.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nError: {e}")
        print("Check the logs above for more details.")