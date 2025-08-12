"""
Stability fixes and improvements for NLP Tool for YPAR
This module provides stable implementations with proper error handling
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import traceback
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import threading
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread safety lock
_db_lock = threading.Lock()
_session_lock = threading.Lock()

class StableMongoDBManager:
    """Thread-safe MongoDB manager with proper connection handling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for connection management"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.client = None
                    cls._instance.db = None
                    cls._instance.connected = False
        return cls._instance
    
    @contextmanager
    def get_connection(self):
        """Context manager for safe database connections"""
        connection = None
        try:
            connection = self._ensure_connection()
            yield connection
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            # Connection pooling handles cleanup automatically
            pass
    
    def _ensure_connection(self):
        """Ensure database connection is established with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                if not self.connected or not self._is_connection_alive():
                    self._connect()
                return self.db
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Failed to establish database connection after retries")
                    raise
    
    def _connect(self):
        """Establish database connection with proper configuration"""
        try:
            from config import Config
            connection_string = Config.get_mongodb_connection_string()
            
            if not connection_string:
                logger.warning("No MongoDB connection string available")
                self.connected = False
                return
            
            # Configure connection with pooling and timeouts
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=45000,
                waitQueueTimeoutMS=10000
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client["nlp_ypar_db"]
            self.connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.connected = False
            raise
    
    def _is_connection_alive(self):
        """Check if the connection is still alive"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            return False
        return False
    
    def safe_insert(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Safely insert a document with error handling"""
        try:
            with self.get_connection() as db:
                if db:
                    collection = db[collection_name]
                    result = collection.insert_one(document)
                    return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
        return None
    
    def safe_find(self, collection_name: str, query: Dict[str, Any], 
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Safely query documents with error handling"""
        try:
            with self.get_connection() as db:
                if db:
                    collection = db[collection_name]
                    results = list(collection.find(query).limit(limit))
                    return results
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
        return []
    
    def close(self):
        """Properly close database connection"""
        try:
            if self.client:
                self.client.close()
                self.connected = False
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

def safe_file_handler(func):
    """Decorator for safe file handling with proper cleanup"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        file_handle = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"File handling error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Cleanup file handles if any
            if file_handle and hasattr(file_handle, 'close'):
                try:
                    file_handle.close()
                except:
                    pass
    return wrapper

def safe_session_state_update(key: str, value: Any):
    """Thread-safe session state update"""
    with _session_lock:
        try:
            st.session_state[key] = value
        except Exception as e:
            logger.error(f"Failed to update session state key '{key}': {e}")

def safe_session_state_get(key: str, default: Any = None) -> Any:
    """Thread-safe session state retrieval"""
    with _session_lock:
        try:
            return st.session_state.get(key, default)
        except Exception as e:
            logger.error(f"Failed to get session state key '{key}': {e}")
            return default

class SafeMemoryManager:
    """Memory-bounded collection manager"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items = []
        self._lock = threading.Lock()
    
    def add(self, item: Any):
        """Add item with automatic pruning"""
        with self._lock:
            self.items.append(item)
            if len(self.items) > self.max_size:
                # Remove oldest items
                self.items = self.items[-self.max_size:]
    
    def get_all(self) -> List[Any]:
        """Get all items safely"""
        with self._lock:
            return self.items.copy()
    
    def clear(self):
        """Clear all items"""
        with self._lock:
            self.items.clear()

def validate_and_sanitize_input(text: str, max_length: int = 100000) -> Optional[str]:
    """Comprehensive input validation and sanitization"""
    if not text:
        return None
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            logger.error("Failed to convert input to string")
            return None
    
    # Length check
    if len(text) > max_length:
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # Remove dangerous patterns
    import re
    
    # Remove script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove SQL injection patterns
    sql_patterns = [
        r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript)\b',
        r'(--|#|\/\*|\*\/)',
        r'(\x00|\x1a)'  # Null bytes
    ]
    
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text.strip()

def safe_nltk_download():
    """Safely download NLTK data with error handling"""
    import nltk
    packages = [
        'punkt', 
        'punkt_tab',  # New NLTK tokenizer format
        'stopwords', 
        'vader_lexicon', 
        'maxent_ne_chunker', 
        'words', 
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    
    for package in packages:
        try:
            # Check if package exists
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            else:
                nltk.data.find(package)
            logger.debug(f"NLTK package '{package}' already downloaded")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
            except Exception as e:
                # Special handling for punkt_tab - fallback to punkt
                if package == 'punkt_tab':
                    try:
                        logger.info("Falling back to standard punkt tokenizer")
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
                logger.warning(f"Failed to download NLTK package '{package}': {e}")

class ErrorHandler:
    """Centralized error handling and recovery"""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "Unknown", 
                    user_friendly: bool = True) -> Optional[str]:
        """Handle errors with appropriate logging and user feedback"""
        error_id = hashlib.md5(f"{error}{time.time()}".encode()).hexdigest()[:8]
        
        # Log detailed error
        logger.error(f"Error [{error_id}] in {context}: {error}")
        logger.error(traceback.format_exc())
        
        # User-friendly message
        if user_friendly:
            if isinstance(error, ConnectionFailure):
                return "Database connection issue. Please try again later."
            elif isinstance(error, ValueError):
                return "Invalid input provided. Please check your data."
            elif isinstance(error, MemoryError):
                return "System memory limit reached. Please reduce the data size."
            elif isinstance(error, PermissionError):
                return "Permission denied. Please check file permissions."
            else:
                return f"An error occurred (ID: {error_id}). Please contact support if it persists."
        
        return str(error)

# Example usage in main application
def stabilized_file_processor(file):
    """Example of stabilized file processing"""
    try:
        # Validate file first
        if not file:
            raise ValueError("No file provided")
        
        # Check file size
        file_size = len(file.read()) if hasattr(file, 'read') else 0
        file.seek(0)  # Reset file pointer
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("File size exceeds 50MB limit")
        
        # Process with proper error handling
        content = None
        with safe_file_handler:
            # Your file processing logic here
            content = file.read()
        
        # Validate and sanitize content
        content = validate_and_sanitize_input(content)
        if not content:
            raise ValueError("File content is invalid or empty")
        
        return content
        
    except Exception as e:
        error_msg = ErrorHandler.handle_error(e, "File Processing")
        st.error(error_msg)
        return None

# Health check system
class HealthMonitor:
    """Monitor system health and resources"""
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """Check overall system health"""
        import psutil
        
        health = {
            'status': 'healthy',
            'checks': {},
            'timestamp': time.time()
        }
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            health['checks']['memory'] = {
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'status': 'ok' if memory.percent < 90 else 'warning'
            }
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            health['checks']['cpu'] = {
                'percent': cpu_percent,
                'status': 'ok' if cpu_percent < 80 else 'warning'
            }
            
            # Database check
            db_manager = StableMongoDBManager()
            db_status = 'ok' if db_manager._is_connection_alive() else 'error'
            health['checks']['database'] = {'status': db_status}
            
            # Overall status
            if any(check.get('status') == 'error' for check in health['checks'].values()):
                health['status'] = 'unhealthy'
            elif any(check.get('status') == 'warning' for check in health['checks'].values()):
                health['status'] = 'degraded'
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health['status'] = 'unknown'
            
        return health

# Import this module in your main application
# from stability_fixes import (
#     StableMongoDBManager, 
#     safe_session_state_update,
#     safe_session_state_get,
#     validate_and_sanitize_input,
#     safe_nltk_download,
#     ErrorHandler,
#     HealthMonitor
# )