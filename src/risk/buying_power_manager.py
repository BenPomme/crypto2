"""
Buying Power Manager
Unified buying power and margin management for crypto and stocks
"""
import logging
from typing import Dict, Any, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class AssetType(Enum):
    CRYPTO = "crypto"
    STOCK = "stock"

class BuyingPowerManager:
    """
    Manages buying power and margin requirements across crypto and stocks
    Ensures proper leverage limits and margin compliance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize buying power manager"""
        self.config = config or {}
        
        # Leverage limits
        self.crypto_leverage = self.config.get('crypto_leverage', 3.0)
        self.stock_leverage = self.config.get('stock_leverage', 4.0)  # Reg T
        
        # Maintenance margin requirements
        self.maintenance_margin = {
            AssetType.CRYPTO: 0.35,  # 35% maintenance for crypto
            AssetType.STOCK: 0.25    # 25% maintenance for stocks
        }
        
        # Initial margin requirements
        self.initial_margin = {
            AssetType.CRYPTO: 0.50,  # 50% initial for crypto (2x leverage)
            AssetType.STOCK: 0.50   # 50% initial for stocks (Reg T)
        }
        
        # Short selling specific
        self.short_margin_requirement = 1.5  # 150% for short positions
        
        logger.info(f"BuyingPowerManager initialized - Crypto leverage: {self.crypto_leverage}x, Stock leverage: {self.stock_leverage}x")
    
    def get_asset_type(self, symbol: str) -> AssetType:
        """Determine if symbol is crypto or stock"""
        # Crypto symbols contain '/'
        if '/' in symbol:
            return AssetType.CRYPTO
        return AssetType.STOCK
    
    def calculate_available_buying_power(self, account_info: Dict[str, Any], 
                                       asset_type: AssetType = None) -> float:
        """
        Calculate available buying power for a specific asset type
        
        Args:
            account_info: Account information including positions
            asset_type: Type of asset (crypto/stock) or None for total
            
        Returns:
            Available buying power in USD
        """
        try:
            total_equity = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            
            # Calculate margin used by existing positions
            positions = account_info.get('positions', [])
            margin_used = self.calculate_total_margin_used(positions)
            
            # Calculate maximum buying power based on leverage
            if asset_type == AssetType.CRYPTO:
                max_buying_power = total_equity * self.crypto_leverage
            elif asset_type == AssetType.STOCK:
                max_buying_power = total_equity * self.stock_leverage
            else:
                # Return minimum of both if no specific type
                crypto_bp = total_equity * self.crypto_leverage - margin_used
                stock_bp = total_equity * self.stock_leverage - margin_used
                return min(crypto_bp, stock_bp)
            
            # Available buying power = max allowed - currently used
            available = max_buying_power - margin_used
            
            # Can't exceed available cash for initial positions
            available = min(available, cash * 2)  # Assuming 50% margin requirement
            
            logger.debug(f"Buying power calculation: Equity=${total_equity:,.2f}, "
                        f"Margin used=${margin_used:,.2f}, Available=${available:,.2f}")
            
            return max(0, available)
            
        except Exception as e:
            logger.error(f"Error calculating buying power: {e}")
            return 0
    
    def calculate_total_margin_used(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate total margin used by all positions"""
        total_margin = 0
        
        for position in positions:
            symbol = position.get('symbol', '')
            market_value = abs(float(position.get('market_value', 0)))
            side = position.get('side', 'long')
            
            asset_type = self.get_asset_type(symbol)
            
            # Calculate margin requirement based on position type
            if side == 'short':
                # Short positions require more margin
                margin_req = market_value / self.short_margin_requirement
            else:
                # Long positions use maintenance margin
                margin_req = market_value * self.maintenance_margin[asset_type]
            
            total_margin += margin_req
            
        return total_margin
    
    def calculate_required_margin(self, symbol: str, quantity: float, 
                                price: float, side: str) -> float:
        """
        Calculate margin required for a new position
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares/units
            price: Current price
            side: 'buy', 'sell', 'sell_short', 'buy_to_cover'
            
        Returns:
            Required margin in USD
        """
        position_value = quantity * price
        asset_type = self.get_asset_type(symbol)
        
        # Determine if this is opening or closing a position
        if side in ['sell_short']:
            # Opening short position - use short margin requirement
            required_margin = position_value / self.short_margin_requirement
        elif side in ['buy']:
            # Opening long position - use initial margin
            required_margin = position_value * self.initial_margin[asset_type]
        else:
            # Closing position - no additional margin required
            required_margin = 0
        
        return required_margin
    
    def can_open_position(self, symbol: str, quantity: float, price: float, 
                         side: str, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if we have enough buying power for new position
        
        Returns:
            Tuple of (can_open, reason)
        """
        try:
            asset_type = self.get_asset_type(symbol)
            
            # Calculate required margin
            required_margin = self.calculate_required_margin(symbol, quantity, price, side)
            
            # Get available buying power
            available_bp = self.calculate_available_buying_power(account_info, asset_type)
            
            # Check if we have enough
            if required_margin > available_bp:
                return False, f"Insufficient buying power: Required ${required_margin:,.2f}, Available ${available_bp:,.2f}"
            
            # Additional checks for short selling
            if side == 'sell_short':
                # Check if shortable
                if not self._is_shortable(symbol, account_info):
                    return False, f"{symbol} is not available for short selling"
                
                # Check short exposure limits
                short_exposure = self._calculate_short_exposure(account_info)
                max_short_exposure = self.config.get('max_short_exposure', 0.5)
                
                new_short_value = quantity * price
                total_portfolio = float(account_info.get('portfolio_value', 0))
                new_exposure = (short_exposure + new_short_value) / total_portfolio
                
                if new_exposure > max_short_exposure:
                    return False, f"Would exceed maximum short exposure ({new_exposure:.1%} > {max_short_exposure:.1%})"
            
            return True, "Sufficient buying power"
            
        except Exception as e:
            logger.error(f"Error checking position feasibility: {e}")
            return False, f"Error: {str(e)}"
    
    def _is_shortable(self, symbol: str, account_info: Dict[str, Any]) -> bool:
        """Check if a symbol is available for short selling"""
        # For crypto, short selling is generally available
        if self.get_asset_type(symbol) == AssetType.CRYPTO:
            return True
        
        # For stocks, would need to check with broker API
        # This is a placeholder - actual implementation would query Alpaca
        # to check if the stock is easy-to-borrow (ETB)
        hard_to_borrow = self.config.get('hard_to_borrow_list', [])
        return symbol not in hard_to_borrow
    
    def _calculate_short_exposure(self, account_info: Dict[str, Any]) -> float:
        """Calculate current short exposure in portfolio"""
        positions = account_info.get('positions', [])
        total_short_value = 0
        
        for position in positions:
            if position.get('side') == 'short':
                total_short_value += abs(float(position.get('market_value', 0)))
        
        return total_short_value
    
    def get_position_sizing_recommendation(self, symbol: str, side: str, 
                                         account_info: Dict[str, Any],
                                         risk_pct: float = 0.01) -> Dict[str, Any]:
        """
        Get recommended position size based on buying power and risk
        
        Args:
            symbol: Trading symbol
            side: Trade side
            account_info: Account information
            risk_pct: Risk percentage per trade
            
        Returns:
            Dictionary with sizing recommendations
        """
        try:
            asset_type = self.get_asset_type(symbol)
            available_bp = self.calculate_available_buying_power(account_info, asset_type)
            portfolio_value = float(account_info.get('portfolio_value', 0))
            
            # Risk-based position size
            risk_amount = portfolio_value * risk_pct
            
            # Buying power based position size
            if side == 'sell_short':
                # For shorts, account for higher margin requirement
                bp_based_size = available_bp * self.short_margin_requirement
            else:
                # For longs, use leverage
                leverage = self.crypto_leverage if asset_type == AssetType.CRYPTO else self.stock_leverage
                bp_based_size = available_bp
            
            # Take the minimum of risk-based and BP-based
            max_position_value = min(risk_amount * 100, bp_based_size)  # 100x assumes 1% stop loss
            
            # Apply maximum position size limit
            max_position_pct = self.config.get('max_position_pct', 0.25)
            max_position_value = min(max_position_value, portfolio_value * max_position_pct)
            
            return {
                'recommended_value': max_position_value,
                'risk_based_value': risk_amount * 100,
                'bp_based_value': bp_based_size,
                'available_buying_power': available_bp,
                'leverage_used': max_position_value / (max_position_value * self.initial_margin[asset_type])
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'recommended_value': 0,
                'error': str(e)
            }
    
    def check_margin_call(self, account_info: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if account is in margin call
        
        Returns:
            Tuple of (is_margin_call, margin_deficit)
        """
        try:
            positions = account_info.get('positions', [])
            total_equity = float(account_info.get('portfolio_value', 0))
            
            # Calculate total margin requirement
            total_margin_req = 0
            for position in positions:
                symbol = position.get('symbol', '')
                market_value = abs(float(position.get('market_value', 0)))
                side = position.get('side', 'long')
                asset_type = self.get_asset_type(symbol)
                
                if side == 'short':
                    margin_req = market_value * 1.3  # 130% for shorts
                else:
                    margin_req = market_value * self.maintenance_margin[asset_type]
                
                total_margin_req += margin_req
            
            # Check if equity covers margin requirement
            if total_equity < total_margin_req:
                deficit = total_margin_req - total_equity
                return True, deficit
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Error checking margin call: {e}")
            return False, 0
    
    def get_leverage_summary(self, account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get current leverage usage summary"""
        try:
            positions = account_info.get('positions', [])
            portfolio_value = float(account_info.get('portfolio_value', 0))
            
            # Calculate exposure by asset type
            crypto_long = 0
            crypto_short = 0
            stock_long = 0
            stock_short = 0
            
            for position in positions:
                symbol = position.get('symbol', '')
                market_value = float(position.get('market_value', 0))
                side = position.get('side', 'long')
                asset_type = self.get_asset_type(symbol)
                
                if asset_type == AssetType.CRYPTO:
                    if side == 'long':
                        crypto_long += market_value
                    else:
                        crypto_short += abs(market_value)
                else:
                    if side == 'long':
                        stock_long += market_value
                    else:
                        stock_short += abs(market_value)
            
            # Calculate leverage ratios
            total_exposure = crypto_long + crypto_short + stock_long + stock_short
            gross_leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            net_leverage = (crypto_long + stock_long - crypto_short - stock_short) / portfolio_value if portfolio_value > 0 else 0
            
            return {
                'portfolio_value': portfolio_value,
                'total_exposure': total_exposure,
                'gross_leverage': gross_leverage,
                'net_leverage': net_leverage,
                'crypto_long': crypto_long,
                'crypto_short': crypto_short,
                'stock_long': stock_long,
                'stock_short': stock_short,
                'crypto_leverage_used': (crypto_long + crypto_short) / (portfolio_value * self.crypto_leverage) if portfolio_value > 0 else 0,
                'stock_leverage_used': (stock_long + stock_short) / (portfolio_value * self.stock_leverage) if portfolio_value > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating leverage summary: {e}")
            return {}