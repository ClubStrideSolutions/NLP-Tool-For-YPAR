"""
Configuration management for NLP Tool for YPAR
"""
import os
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management class"""
    
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        """Get OpenAI API key from environment or Streamlit secrets"""
        try:
            if hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                return st.secrets['openai_api_key']
            return os.getenv('OPENAI_API_KEY')
        except Exception:
            return os.getenv('OPENAI_API_KEY')
    
    @staticmethod
    def get_mongodb_connection_string() -> Optional[str]:
        """Get MongoDB connection string from environment or Streamlit secrets"""
        try:
            if hasattr(st, 'secrets') and 'mongodb_connection_string' in st.secrets:
                return st.secrets['mongodb_connection_string']
            return os.getenv('CONNECTION_STRING')
        except Exception:
            return os.getenv('CONNECTION_STRING')
    
    @staticmethod
    def get_app_config() -> dict:
        """Get application configuration"""
        return {
            'page_title': 'NLP Tool for YPAR',
            'page_icon': '📊',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded',
            'cache_ttl': 3600,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'supported_file_types': ['txt', 'docx', 'xlsx'],
            'batch_size': 5,
            'theme_colors': {
                'primary': '#2874a6',
                'secondary': '#154360',
                'background': '#eaf2f8',
                'text': '#2c3e50',
                'success': '#27ae60',
                'error': '#e74c3c'
            }
        }
    
    @staticmethod
    def validate_config() -> tuple[bool, str]:
        """Validate configuration"""
        errors = []
        
        if not Config.get_openai_api_key() and not Config.get_mongodb_connection_string():
            errors.append("Neither OpenAI API key nor MongoDB connection configured. At least one is recommended.")
        
        if errors:
            return False, "\n".join(errors)
        return True, "Configuration valid"