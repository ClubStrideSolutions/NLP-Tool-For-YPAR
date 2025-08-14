"""
Test script to verify the NLP YPAR Tool functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    errors = []
    
    # Test core modules
    try:
        from file_handlers import FileHandler
        print("[OK] FileHandler imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] FileHandler import failed: {e}")
    
    try:
        from openai_analyzer import OpenAIAnalyzer
        print("[OK] OpenAIAnalyzer imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] OpenAIAnalyzer import failed: {e}")
    
    try:
        from enhanced_db_manager import EnhancedMongoManager
        print("[OK] EnhancedMongoManager imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] EnhancedMongoManager import failed: {e}")
    
    try:
        from rag_system import RAGSystem
        print("[OK] RAGSystem imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] RAGSystem import failed: {e}")
    
    try:
        from ui_components import UIComponents
        print("[OK] UIComponents imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] UIComponents import failed: {e}")
    
    # Test main app import
    try:
        import main_enhanced_cal
        print("[OK] main_enhanced_cal imported successfully")
    except ImportError as e:
        errors.append(f"[ERROR] main_enhanced_cal import failed: {e}")
    
    return errors

def test_class_instantiation():
    """Test class instantiation"""
    errors = []
    
    try:
        from file_handlers import FileHandler
        handler = FileHandler()
        print("[OK] FileHandler instantiated successfully")
    except Exception as e:
        errors.append(f"[ERROR] FileHandler instantiation failed: {e}")
    
    try:
        from openai_analyzer import OpenAIAnalyzer
        analyzer = OpenAIAnalyzer()
        print("[OK] OpenAIAnalyzer instantiated successfully")
    except Exception as e:
        errors.append(f"[ERROR] OpenAIAnalyzer instantiation failed: {e}")
    
    return errors

if __name__ == "__main__":
    print("=" * 50)
    print("NLP YPAR Tool - System Test")
    print("=" * 50)
    
    print("\n1. Testing imports...")
    import_errors = test_imports()
    
    print("\n2. Testing class instantiation...")
    class_errors = test_class_instantiation()
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_errors = import_errors + class_errors
    
    if not all_errors:
        print("[SUCCESS] All tests passed successfully!")
    else:
        print(f"[FAILED] Found {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  {error}")
    
    print("\n" + "=" * 50)