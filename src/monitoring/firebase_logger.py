"""
Firebase Logger Module
Handles logging and data persistence to Firebase Firestore
"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import traceback

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)

class FirebaseLogger:
    """
    Firebase integration for logging trading data
    Handles real-time data persistence and retrieval
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Firebase logger
        
        Args:
            config: Firebase configuration (uses settings if None)
        """
        self.config = config
        self.db = None
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase Admin SDK not available. Logging disabled.")
            return
        
        try:
            self._initialize_firebase()
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            settings = get_settings()
            
            # Check if we have Firebase credentials available
            import os
            has_service_account = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') is not None
            has_firebase_key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY') is not None
            
            if not has_service_account and not has_firebase_key:
                logger.warning("No Firebase credentials found, running without Firebase logging")
                self.initialized = False
                return
            
            # Check if Firebase app is already initialized
            try:
                app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            except ValueError:
                # Try to initialize with available credentials
                if has_firebase_key:
                    # Use service account key from environment
                    import json
                    service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
                    cred = credentials.Certificate(service_account_info)
                else:
                    # Use application default credentials (if available)
                    cred = credentials.ApplicationDefault()
                
                app = firebase_admin.initialize_app(cred, {
                    'projectId': settings.firebase.project_id
                })
                
                logger.info("Firebase app initialized")
            
            # Get Firestore client
            self.db = firestore.client()
            self.initialized = True
            
            logger.info("Firebase logger initialized successfully")
            
        except Exception as e:
            logger.warning(f"Firebase initialization failed, continuing without Firebase: {e}")
            self.initialized = False
    
    def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Log trade execution to Firebase
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping trade log")
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            # Add to trades collection
            doc_ref = self.db.collection('trades').add(trade_data)
            logger.debug(f"Trade logged to Firebase: {doc_ref[1].id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade to Firebase: {e}")
            return False
    
    def log_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Log trading signal to Firebase
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping signal log")
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in signal_data:
                signal_data['timestamp'] = datetime.now().isoformat()
            
            # Add to signals collection
            doc_ref = self.db.collection('signals').add(signal_data)
            logger.debug(f"Signal logged to Firebase: {doc_ref[1].id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log signal to Firebase: {e}")
            return False
    
    def log_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        Log performance metrics to Firebase
        
        Args:
            performance_data: Performance data dictionary
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping performance log")
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in performance_data:
                performance_data['timestamp'] = datetime.now().isoformat()
            
            # Use date as document ID for daily performance
            date_str = datetime.now().strftime('%Y-%m-%d')
            
            # Update or create performance document
            doc_ref = self.db.collection('performance').document(date_str)
            doc_ref.set(performance_data, merge=True)
            
            logger.debug(f"Performance logged to Firebase: {date_str}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log performance to Firebase: {e}")
            return False
    
    def log_error(self, error_data: Dict[str, Any]) -> bool:
        """
        Log error to Firebase
        
        Args:
            error_data: Error data dictionary
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping error log")
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in error_data:
                error_data['timestamp'] = datetime.now().isoformat()
            
            # Add to errors collection
            doc_ref = self.db.collection('errors').add(error_data)
            logger.debug(f"Error logged to Firebase: {doc_ref[1].id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log error to Firebase: {e}")
            return False
    
    def update_system_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Update system status in Firebase
        
        Args:
            status_data: System status data
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping status update")
            return False
        
        try:
            # Add timestamp
            status_data['last_updated'] = datetime.now().isoformat()
            
            # Update system_status document
            doc_ref = self.db.collection('system').document('status')
            doc_ref.set(status_data, merge=True)
            
            logger.debug("System status updated in Firebase")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update system status in Firebase: {e}")
            return False
    
    def get_trades(self, limit: int = 100, 
                   start_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve trades from Firebase
        
        Args:
            limit: Maximum number of trades to retrieve
            start_date: Start date filter
            
        Returns:
            List of trade data
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, returning empty trades list")
            return []
        
        try:
            query = self.db.collection('trades').order_by('timestamp', direction=firestore.Query.DESCENDING)
            
            if start_date:
                query = query.where('timestamp', '>=', start_date.isoformat())
            
            docs = query.limit(limit).stream()
            
            trades = []
            for doc in docs:
                trade_data = doc.to_dict()
                trade_data['id'] = doc.id
                trades.append(trade_data)
            
            logger.debug(f"Retrieved {len(trades)} trades from Firebase")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades from Firebase: {e}")
            return []
    
    def get_performance_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Retrieve performance data from Firebase
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            Performance data dictionary
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, returning empty performance data")
            return {}
        
        try:
            # Get recent performance documents
            docs = self.db.collection('performance').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(days).stream()
            
            performance_data = {}
            for doc in docs:
                data = doc.to_dict()
                performance_data[doc.id] = data
            
            logger.debug(f"Retrieved performance data for {len(performance_data)} days")
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve performance data from Firebase: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status from Firebase
        
        Returns:
            System status data
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, returning empty status")
            return {}
        
        try:
            doc_ref = self.db.collection('system').document('status')
            doc = doc_ref.get()
            
            if doc.exists:
                status_data = doc.to_dict()
                logger.debug("Retrieved system status from Firebase")
                return status_data
            else:
                logger.warning("System status document not found in Firebase")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to retrieve system status from Firebase: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """
        Clean up old data from Firebase
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.warning("Firebase not initialized, skipping cleanup")
            return False
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            # Cleanup old trades
            old_trades = self.db.collection('trades').where('timestamp', '<', cutoff_str).stream()
            
            deleted_count = 0
            for doc in old_trades:
                doc.reference.delete()
                deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old trade records from Firebase")
            
            # Cleanup old signals
            old_signals = self.db.collection('signals').where('timestamp', '<', cutoff_str).stream()
            
            deleted_count = 0
            for doc in old_signals:
                doc.reference.delete()
                deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old signal records from Firebase")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data from Firebase: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if Firebase connection is active"""
        return self.initialized and self.db is not None