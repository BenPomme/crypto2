"""
Fee Calculator Module
Calculates trading fees for Alpaca crypto and stock orders
"""
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FeeCalculator:
    """
    Calculate trading fees for Alpaca orders
    """
    
    def __init__(self):
        """Initialize fee calculator with Alpaca fee structure"""
        
        # Alpaca crypto fee tiers (maker/taker % fees)
        self.crypto_fee_tiers = [
            {'min_volume': 0, 'max_volume': 100000, 'maker': 0.0015, 'taker': 0.0025},
            {'min_volume': 100000, 'max_volume': 500000, 'maker': 0.0012, 'taker': 0.0020},
            {'min_volume': 500000, 'max_volume': 1000000, 'maker': 0.0010, 'taker': 0.0018},
            {'min_volume': 1000000, 'max_volume': 5000000, 'maker': 0.0008, 'taker': 0.0015},
            {'min_volume': 5000000, 'max_volume': 10000000, 'maker': 0.0005, 'taker': 0.0012},
            {'min_volume': 10000000, 'max_volume': 25000000, 'maker': 0.0003, 'taker': 0.0010},
            {'min_volume': 25000000, 'max_volume': 100000000, 'maker': 0.0001, 'taker': 0.0008},
            {'min_volume': 100000000, 'max_volume': float('inf'), 'maker': 0.0000, 'taker': 0.0010}
        ]
        
        # Stock trading is commission-free on Alpaca
        self.stock_commission = 0.0
        
        logger.info("FeeCalculator initialized")
    
    def calculate_crypto_fees(self, trade_value: float, is_maker: bool = False, 
                             monthly_volume: float = 0) -> float:
        """
        Calculate crypto trading fees based on trade value and volume tier
        
        Args:
            trade_value: Dollar value of the trade
            is_maker: Whether this is a maker order (limit orders that add liquidity)
            monthly_volume: 30-day trading volume for tier calculation
            
        Returns:
            Fee amount in USD
        """
        # Find appropriate fee tier
        fee_rate = 0.0025  # Default to highest taker fee
        
        for tier in self.crypto_fee_tiers:
            if tier['min_volume'] <= monthly_volume < tier['max_volume']:
                fee_rate = tier['maker'] if is_maker else tier['taker']
                break
        
        fee = trade_value * fee_rate
        
        logger.debug(f"Crypto fee: ${fee:.2f} (rate: {fee_rate:.4f}, "
                    f"value: ${trade_value:.2f}, maker: {is_maker})")
        
        return fee
    
    def calculate_stock_fees(self, trade_value: float) -> float:
        """
        Calculate stock trading fees (commission-free on Alpaca)
        
        Args:
            trade_value: Dollar value of the trade
            
        Returns:
            Fee amount (always 0 for stocks)
        """
        return 0.0
    
    def calculate_round_trip_fees(self, position_value: float, symbol: str,
                                 is_maker: bool = False, monthly_volume: float = 0) -> float:
        """
        Calculate total fees for a complete round trip (buy + sell)
        
        Args:
            position_value: Dollar value of position
            symbol: Trading symbol
            is_maker: Whether orders are typically maker orders
            monthly_volume: 30-day trading volume
            
        Returns:
            Total round trip fee amount
        """
        is_crypto = '/' in symbol and 'USD' in symbol
        
        if is_crypto:
            # Crypto: entry fee + exit fee
            entry_fee = self.calculate_crypto_fees(position_value, is_maker, monthly_volume)
            exit_fee = self.calculate_crypto_fees(position_value, is_maker, monthly_volume)
            total_fee = entry_fee + exit_fee
        else:
            # Stocks: commission-free
            total_fee = 0.0
        
        logger.info(f"Round trip fees for {symbol}: ${total_fee:.2f} "
                   f"(position: ${position_value:.2f})")
        
        return total_fee
    
    def get_minimum_profitable_move(self, position_value: float, symbol: str,
                                   is_maker: bool = False, monthly_volume: float = 0) -> float:
        """
        Calculate minimum price move needed to cover round trip fees
        
        Args:
            position_value: Dollar value of position
            symbol: Trading symbol
            is_maker: Whether orders are typically maker orders
            monthly_volume: 30-day trading volume
            
        Returns:
            Minimum percentage move needed to break even on fees
        """
        total_fees = self.calculate_round_trip_fees(position_value, symbol, is_maker, monthly_volume)
        
        if position_value > 0:
            min_move_pct = total_fees / position_value
        else:
            min_move_pct = 0.0
        
        logger.info(f"Minimum profitable move for {symbol}: {min_move_pct:.4f} "
                   f"({min_move_pct*100:.2f}%)")
        
        return min_move_pct
    
    def adjust_targets_for_fees(self, take_profit_pct: float, stop_loss_pct: float,
                               position_value: float, symbol: str,
                               is_maker: bool = False, monthly_volume: float = 0) -> Tuple[float, float]:
        """
        Adjust take profit and stop loss targets to account for fees
        
        Args:
            take_profit_pct: Original take profit percentage
            stop_loss_pct: Original stop loss percentage
            position_value: Dollar value of position
            symbol: Trading symbol
            is_maker: Whether orders are typically maker orders
            monthly_volume: 30-day trading volume
            
        Returns:
            Tuple of (adjusted_take_profit_pct, adjusted_stop_loss_pct)
        """
        min_move = self.get_minimum_profitable_move(position_value, symbol, is_maker, monthly_volume)
        
        # Add fee buffer to take profit target
        adjusted_take_profit = take_profit_pct + min_move + 0.001  # Extra 0.1% safety margin
        
        # Stop loss doesn't need adjustment (we lose the fees anyway)
        adjusted_stop_loss = stop_loss_pct
        
        logger.info(f"Fee-adjusted targets for {symbol}: "
                   f"TP: {take_profit_pct:.3f} → {adjusted_take_profit:.3f}, "
                   f"SL: {stop_loss_pct:.3f} (unchanged)")
        
        return adjusted_take_profit, adjusted_stop_loss