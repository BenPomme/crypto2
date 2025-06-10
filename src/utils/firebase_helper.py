"""
Firebase Helper Utilities
Handles Firebase credential parsing and initialization
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_firebase_credentials() -> Optional[Dict[str, Any]]:
    """
    Parse Firebase service account credentials from environment
    Handles multi-line private keys and escaped JSON
    
    Returns:
        Parsed service account dictionary or None
    """
    firebase_key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
    if not firebase_key:
        return None
    
    try:
        # First attempt: direct JSON parse
        return json.loads(firebase_key)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parse failed, trying to fix formatting")
        
    try:
        # Second attempt: Handle escaped quotes and newlines
        # Replace escaped quotes
        firebase_key = firebase_key.replace('\\"', '"')
        
        # If the JSON is wrapped in quotes, remove them
        if firebase_key.startswith('"') and firebase_key.endswith('"'):
            firebase_key = firebase_key[1:-1]
        
        return json.loads(firebase_key)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Firebase credentials after cleanup: {e}")
        
    # Third attempt: Try to reconstruct if it's a broken JSON
    try:
        # Extract key components using string manipulation
        if '"type":"service_account"' in firebase_key:
            logger.debug("Attempting to reconstruct Firebase credentials")
            
            # This is a last resort - extract the essential fields
            import re
            
            def extract_field(key: str, text: str) -> Optional[str]:
                """Extract a field value from malformed JSON"""
                pattern = f'"{key}":"([^"]*)"'
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
                
                # Try multi-line pattern for private key
                if key == "private_key":
                    pattern = f'"{key}":"(-----BEGIN[^"]+-----)"'
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        return match.group(1)
                return None
            
            # Build credentials manually
            creds = {
                "type": "service_account",
                "project_id": extract_field("project_id", firebase_key),
                "private_key_id": extract_field("private_key_id", firebase_key),
                "private_key": extract_field("private_key", firebase_key),
                "client_email": extract_field("client_email", firebase_key),
                "client_id": extract_field("client_id", firebase_key),
                "auth_uri": extract_field("auth_uri", firebase_key),
                "token_uri": extract_field("token_uri", firebase_key),
                "auth_provider_x509_cert_url": extract_field("auth_provider_x509_cert_url", firebase_key),
                "client_x509_cert_url": extract_field("client_x509_cert_url", firebase_key)
            }
            
            # Validate required fields
            required_fields = ["project_id", "private_key", "client_email"]
            if all(creds.get(field) for field in required_fields):
                logger.info("Successfully reconstructed Firebase credentials")
                return creds
            else:
                logger.error(f"Missing required fields in reconstructed credentials: {[f for f in required_fields if not creds.get(f)]}")
                
    except Exception as e:
        logger.error(f"Failed to reconstruct Firebase credentials: {e}")
    
    return None

def write_temp_credentials_file() -> Optional[str]:
    """
    Write Firebase credentials to a temporary file if needed
    
    Returns:
        Path to credentials file or None
    """
    creds = parse_firebase_credentials()
    if not creds:
        return None
    
    try:
        # Write to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds, f, indent=2)
            temp_path = f.name
        
        logger.info(f"Wrote Firebase credentials to temporary file: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to write credentials file: {e}")
        return None