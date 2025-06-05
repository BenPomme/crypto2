#!/usr/bin/env python3
"""
Direct Position Liquidation Script
Uses hardcoded credentials from env file
"""
import logging

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
    print("Installing alpaca_trade_api...")
    import subprocess
    subprocess.run(["pip3", "install", "alpaca-trade-api"])
    from alpaca_trade_api import REST
    from alpaca_trade_api.common import URL

# Credentials from .env.local file
ALPACA_KEY = "PK3KOPY1SYF1OXSDDYPG"
ALPACA_SECRET = "eCcig7euiFP1eMSmCigHDwVFZDUax5aoRiDevCog"
ALPACA_ENDPOINT = "https://paper-api.alpaca.markets"

class DirectLiquidator:
    """Direct position liquidator using Alpaca API"""
    
    def __init__(self):
        """Initialize with Alpaca API"""
        self.api = REST(
            key_id=ALPACA_KEY,
            secret_key=ALPACA_SECRET,
            base_url=URL(ALPACA_ENDPOINT)
        )
        logger.info("Direct Liquidator initialized")
    
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            logger.info(f"Account Info:")
            logger.info(f"  Account ID: {account.id}")
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
            
            if not positions:
                logger.info("  No positions found")
                return positions
            
            total_value = 0
            total_pl = 0
            
            for pos in positions:
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)
                
                total_value += market_value
                total_pl += unrealized_pl
                
                logger.info(f"  {pos.symbol}:")
                logger.info(f"    Quantity: {pos.qty} shares")
                logger.info(f"    Market Value: ${market_value:.2f}")
                logger.info(f"    P&L: ${unrealized_pl:.2f} ({unrealized_plpc:.2%})")
                logger.info(f"    Entry Price: ${float(pos.avg_entry_price):.4f}")
                logger.info(f"    Side: {pos.side}")
                logger.info(f"")
            
            logger.info(f"Total Portfolio:")
            logger.info(f"  Total Position Value: ${total_value:.2f}")
            logger.info(f"  Total Unrealized P&L: ${total_pl:.2f}")
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def liquidate_position(self, symbol, quantity):
        """Liquidate a specific position"""
        try:
            qty = abs(float(quantity))
            logger.info(f"Placing SELL order for {qty} shares of {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'  # Good Till Cancelled for crypto
            )
            
            logger.info(f"Order submitted successfully!")
            logger.info(f"  Order ID: {order.id}")
            logger.info(f"  Status: {order.status}")
            
            # Wait for order to process
            import time
            logger.info("Waiting for order to fill...")
            
            for i in range(10):  # Check for up to 10 seconds
                time.sleep(1)
                try:
                    updated_order = self.api.get_order(order.id)
                    logger.info(f"  Check {i+1}: Status = {updated_order.status}")
                    
                    if updated_order.status == 'filled':
                        logger.info(f"✓ Position liquidated successfully!")
                        if hasattr(updated_order, 'filled_avg_price') and updated_order.filled_avg_price:
                            logger.info(f"  Fill Price: ${float(updated_order.filled_avg_price):.4f}")
                        return True
                    elif updated_order.status in ['canceled', 'rejected']:
                        logger.error(f"✗ Order {updated_order.status}")
                        return False
                except Exception as e:
                    logger.warning(f"Error checking order status: {e}")
            
            logger.warning(f"Order may still be processing. Final status check...")
            final_order = self.api.get_order(order.id)
            logger.info(f"Final status: {final_order.status}")
            return final_order.status == 'filled'
                
        except Exception as e:
            logger.error(f"Error liquidating position {symbol}: {e}")
            return False
    
    def liquidate_all(self):
        """Liquidate all positions"""
        logger.info("Starting liquidation of ALL positions...")
        
        positions = self.get_positions()
        if not positions:
            logger.info("No positions to liquidate")
            return True
        
        results = []
        for position in positions:
            symbol = position.symbol
            qty = abs(float(position.qty))
            
            if qty > 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Liquidating {symbol}: {qty} shares")
                logger.info(f"Current Value: ${float(position.market_value):.2f}")
                logger.info(f"{'='*50}")
                
                success = self.liquidate_position(symbol, qty)
                results.append({
                    'symbol': symbol,
                    'quantity': qty,
                    'success': success
                })
        
        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"LIQUIDATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total positions: {len(results)}")
        logger.info(f"Successfully liquidated: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            logger.info(f"\n✓ Successful liquidations:")
            for result in successful:
                logger.info(f"  {result['symbol']}: {result['quantity']} shares")
        
        if failed:
            logger.info(f"\n✗ Failed liquidations:")
            for result in failed:
                logger.info(f"  {result['symbol']}: {result['quantity']} shares")
        
        return len(failed) == 0

def main():
    """Main function"""
    print("\n" + "="*60)
    print("DIRECT POSITION LIQUIDATION TOOL")
    print("="*60)
    print("WARNING: This will liquidate ALL current positions!")
    print("="*60)
    
    try:
        liquidator = DirectLiquidator()
        
        print("\nStep 1: Connecting to Alpaca API...")
        account = liquidator.get_account_info()
        if not account:
            print("✗ Failed to connect to Alpaca API")
            return
        print("✓ Connected successfully")
        
        print("\nStep 2: Checking current positions...")
        positions = liquidator.get_positions()
        
        if not positions:
            print("✓ No positions found - account is already flat")
            return
        
        print(f"\nFound {len(positions)} positions that will be liquidated:")
        for pos in positions:
            print(f"  - {pos.symbol}: {pos.qty} shares (${float(pos.market_value):.2f})")
        
        print("\n" + "="*60)
        response = input("Are you sure you want to liquidate ALL positions? (type 'YES' to confirm): ")
        
        if response.strip().upper() == 'YES':
            print("\nProceeding with liquidation...")
            success = liquidator.liquidate_all()
            
            print("\nStep 3: Final status check...")
            liquidator.get_account_info()
            final_positions = liquidator.get_positions()
            
            if not final_positions:
                print("\n✓ ALL POSITIONS SUCCESSFULLY LIQUIDATED!")
                print("✓ Account is now flat - ready for new trading")
            else:
                print(f"\n⚠ Warning: {len(final_positions)} positions still remain")
                
        else:
            print("\nLiquidation cancelled - no positions were sold")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()