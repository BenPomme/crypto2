#!/usr/bin/env python3
"""
Simple Position Liquidation Script
Direct Alpaca API calls without complex imports
"""
import os
import logging
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.local')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from alpaca_trade_api import REST
    from alpaca_trade_api.common import URL
except ImportError:
    print("Error: alpaca_trade_api not installed. Installing...")
    os.system("pip3 install alpaca-trade-api")
    from alpaca_trade_api import REST
    from alpaca_trade_api.common import URL

class SimpleLiquidator:
    """Simple position liquidator using direct Alpaca API"""
    
    def __init__(self):
        """Initialize with Alpaca API"""
        self.api = REST(
            key_id=os.getenv('ALPACA_KEY'),
            secret_key=os.getenv('ALPACA_SECRET'),
            base_url=URL(os.getenv('ALPACA_ENDPOINT', 'https://paper-api.alpaca.markets'))
        )
        logger.info("Simple Liquidator initialized")
    
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            logger.info(f"Account Info:")
            logger.info(f"  Portfolio Value: ${float(account.portfolio_value):.2f}")
            logger.info(f"  Cash: ${float(account.cash):.2f}")
            logger.info(f"  Buying Power: ${float(account.buying_power):.2f}")
            logger.info(f"  Trading Blocked: {account.trading_blocked}")
            return account
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            logger.info(f"Found {len(positions)} positions:")
            
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.qty} shares")
                logger.info(f"    Market Value: ${float(pos.market_value):.2f}")
                logger.info(f"    P&L: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc):.2%})")
                logger.info(f"    Entry Price: ${float(pos.avg_entry_price):.4f}")
                logger.info(f"    Current Price: ${float(pos.market_value) / abs(float(pos.qty)):.4f}")
                logger.info(f"    Side: {pos.side}")
                logger.info(f"")
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def liquidate_position(self, symbol, quantity):
        """Liquidate a specific position"""
        try:
            logger.info(f"Placing SELL order for {quantity} shares of {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(float(quantity)),
                side='sell',
                type='market',
                time_in_force='gtc'  # Good Till Cancelled for crypto
            )
            
            logger.info(f"Order submitted: {order.id}")
            logger.info(f"Status: {order.status}")
            
            # Wait a moment for order to process
            import time
            time.sleep(2)
            
            # Check order status
            updated_order = self.api.get_order(order.id)
            logger.info(f"Final order status: {updated_order.status}")
            
            if updated_order.status == 'filled':
                logger.info(f"✓ Position liquidated successfully!")
                if hasattr(updated_order, 'filled_avg_price'):
                    logger.info(f"  Fill Price: ${float(updated_order.filled_avg_price):.4f}")
                return True
            else:
                logger.warning(f"Order not filled yet. Status: {updated_order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error liquidating position: {e}")
            return False
    
    def liquidate_all(self):
        """Liquidate all positions"""
        logger.info("Starting liquidation of ALL positions...")
        
        positions = self.get_positions()
        if not positions:
            logger.info("No positions to liquidate")
            return True
        
        success_count = 0
        for position in positions:
            symbol = position.symbol
            qty = abs(float(position.qty))
            
            if qty > 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Liquidating {symbol}")
                logger.info(f"{'='*50}")
                
                if self.liquidate_position(symbol, qty):
                    success_count += 1
                    
        logger.info(f"\n{'='*50}")
        logger.info(f"SUMMARY: {success_count}/{len(positions)} positions liquidated")
        logger.info(f"{'='*50}")
        
        return success_count == len(positions)

def main():
    """Main function"""
    print("\n" + "="*60)
    print("SIMPLE POSITION LIQUIDATION TOOL")
    print("="*60)
    
    try:
        liquidator = SimpleLiquidator()
        
        print("\nChecking current status...")
        account = liquidator.get_account_info()
        if not account:
            print("Failed to connect to Alpaca API")
            return
        
        print("\nCurrent positions:")
        positions = liquidator.get_positions()
        
        if not positions:
            print("No positions to liquidate!")
            return
        
        print(f"\nFound {len(positions)} positions to liquidate")
        response = input("\nProceed with liquidation? (type 'YES' to confirm): ")
        
        if response.strip().upper() == 'YES':
            print("\nStarting liquidation...")
            liquidator.liquidate_all()
            
            print("\nFinal check...")
            liquidator.get_account_info()
            liquidator.get_positions()
        else:
            print("Liquidation cancelled")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()