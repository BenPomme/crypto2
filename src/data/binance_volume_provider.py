"""
Binance Volume Data Provider
Real-time volume data from Binance public WebSocket API to supplement Alpaca price data
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
import websocket
import pandas as pd

logger = logging.getLogger(__name__)

class BinanceVolumeProvider:
    """
    Real-time volume data provider using Binance public WebSocket API
    Provides accurate volume data to supplement Alpaca's low-volume crypto data
    """
    
    def __init__(self, trading_symbols=None):
        """Initialize Binance volume provider"""
        self.ws = None
        self.volume_data = {}  # Symbol -> latest volume data
        self.last_update = {}  # Symbol -> last update timestamp
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Complete symbol mapping: Our symbols -> Binance symbols
        self.all_symbol_mapping = {
            'BTC/USD': 'btcusdt',
            'ETH/USD': 'ethusdt', 
            'SOL/USD': 'solusdt',
            'AVAX/USD': 'avaxusdt',
            'MATIC/USD': 'maticusdt',
            'DOGE/USD': 'dogeusdt'
        }
        
        # Filter to only trading symbols if provided
        if trading_symbols:
            self.symbol_mapping = {
                symbol: self.all_symbol_mapping[symbol] 
                for symbol in trading_symbols 
                if symbol in self.all_symbol_mapping
            }
        else:
            self.symbol_mapping = self.all_symbol_mapping
        
        # Reverse mapping for callbacks
        self.reverse_mapping = {v: k for k, v in self.symbol_mapping.items()}
        
        # WebSocket URL (free public API, no auth required)
        self.ws_url = "wss://data-stream.binance.vision/ws"
        
        logger.info(f"Binance Volume Provider initialized for symbols: {list(self.symbol_mapping.keys())}")
    
    def start(self) -> None:
        """Start the WebSocket connection and volume streaming"""
        if self.running:
            logger.warning("Volume provider already running")
            return
            
        self.running = True
        self.reconnect_attempts = 0
        
        # Start WebSocket in background thread
        threading.Thread(target=self._start_websocket, daemon=True).start()
        logger.info("🚀 Starting Binance volume data stream...")
    
    def stop(self) -> None:
        """Stop the WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("Binance volume provider stopped")
    
    def get_volume_data(self, symbol: str) -> Optional[Dict]:
        """
        Get latest volume data for a symbol
        
        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')
            
        Returns:
            Dictionary with volume data or None if not available
        """
        binance_symbol = self.symbol_mapping.get(symbol)
        if not binance_symbol:
            return None
            
        data = self.volume_data.get(binance_symbol)
        if not data:
            return None
        
        # Check if data is recent (within last 2 minutes)
        last_update = self.last_update.get(binance_symbol)
        if last_update and (datetime.now() - last_update).seconds > 120:
            logger.warning(f"Volume data for {symbol} is stale ({last_update})")
            return None
            
        return data
    
    def get_24h_volume_usd(self, symbol: str) -> Optional[float]:
        """
        Get 24h USD volume for a symbol
        
        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')
            
        Returns:
            24h volume in USD or None if not available
        """
        data = self.get_volume_data(symbol)
        if not data:
            return None
            
        # Return quote asset volume (USD volume)
        return float(data.get('q', 0))
    
    def get_current_volume_rate(self, symbol: str) -> Optional[float]:
        """
        Estimate current volume rate (USD per minute) based on 24h data
        
        Args:
            symbol: Symbol in our format (e.g., 'BTC/USD')
            
        Returns:
            Estimated volume per minute in USD
        """
        volume_24h = self.get_24h_volume_usd(symbol)
        if not volume_24h:
            return None
            
        # Convert to per-minute rate
        return volume_24h / (24 * 60)
    
    def _start_websocket(self) -> None:
        """Start WebSocket connection with automatic reconnection"""
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to Binance WebSocket (attempt {self.reconnect_attempts + 1})")
                
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                self.ws.run_forever()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.reconnect_attempts += 1
                
                if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                    wait_time = min(30, 2 ** self.reconnect_attempts)
                    logger.info(f"Reconnecting in {wait_time} seconds...")
                    time.sleep(wait_time)
    
    def _on_open(self, ws) -> None:
        """WebSocket connection opened"""
        logger.info("✅ Connected to Binance WebSocket")
        self.reconnect_attempts = 0
        
        # Subscribe to 24hr mini ticker for all our symbols
        streams = [f"{symbol}@miniTicker" for symbol in self.symbol_mapping.values()]
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        logger.info(f"📡 Subscribing to {len(streams)} volume streams:")
        for our_symbol, binance_symbol in self.symbol_mapping.items():
            logger.info(f"   {our_symbol} -> {binance_symbol}@miniTicker")
        
        ws.send(json.dumps(subscribe_msg))
        logger.info(f"✅ Subscription request sent for: {streams}")
    
    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle mini ticker updates
            if data.get('e') == '24hrMiniTicker':
                symbol = data['s'].lower()  # e.g., 'btcusdt'
                
                if symbol in self.reverse_mapping:
                    # Store volume data
                    self.volume_data[symbol] = {
                        'symbol': symbol,
                        'price': float(data['c']),      # Close price
                        'open': float(data['o']),       # Open price
                        'high': float(data['h']),       # High price
                        'low': float(data['l']),        # Low price
                        'volume': float(data['v']),     # Base asset volume (e.g., BTC)
                        'volume_usd': float(data['q']), # Quote asset volume (USD)
                        'timestamp': datetime.now()
                    }
                    
                    self.last_update[symbol] = datetime.now()
                    
                    our_symbol = self.reverse_mapping[symbol]
                    volume_usd = float(data['q'])
                    
                    # Log significant volume updates (every 100 updates to avoid spam)
                    if not hasattr(self, f'_log_counter_{symbol}'):
                        setattr(self, f'_log_counter_{symbol}', 0)
                    
                    counter = getattr(self, f'_log_counter_{symbol}') + 1
                    setattr(self, f'_log_counter_{symbol}', counter)
                    
                    if counter % 100 == 0:
                        logger.info(f"📊 {our_symbol} Volume: ${volume_usd:,.0f} (24h), Price: ${float(data['c']):.2f}")
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            logger.debug(f"Message: {message}")
    
    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket close"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        if self.running:
            logger.info("Attempting to reconnect...")
            self.reconnect_attempts += 1
    
    def get_status(self) -> Dict:
        """Get provider status for monitoring"""
        return {
            'running': self.running,
            'connected': self.ws and self.ws.sock and self.ws.sock.connected if self.ws else False,
            'symbols_tracking': len(self.volume_data),
            'last_updates': {
                self.reverse_mapping.get(symbol, symbol): update.isoformat() 
                for symbol, update in self.last_update.items()
            },
            'reconnect_attempts': self.reconnect_attempts
        }