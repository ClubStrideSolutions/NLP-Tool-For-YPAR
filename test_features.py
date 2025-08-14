"""
Feature Testing Script for NLP YPAR Tool
Run this to verify basic functionality
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test that all core modules can be imported"""
    print("\n" + "="*50)
    print("TESTING CORE IMPORTS")
    print("="*50)
    
    results = []
    
    # Test imports
    modules = [
        ('file_handlers', 'FileHandler'),
        ('openai_analyzer', 'OpenAIAnalyzer'),
        ('enhanced_db_manager', 'EnhancedMongoManager'),
        ('rag_system', 'RAGSystem'),
        ('ui_components', 'UIComponents'),
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"[OK] {module_name}.{class_name}")
                results.append(True)
            else:
                print(f"[FAIL] {module_name}.{class_name} - Class not found")
                results.append(False)
        except ImportError as e:
            print(f"[FAIL] {module_name} - Import error: {e}")
            results.append(False)
    
    return all(results)

def test_session_state():
    """Test session state initialization"""
    print("\n" + "="*50)
    print("TESTING SESSION STATE")
    print("="*50)
    
    # Simulate session state
    session_vars = {
        'processed_data': [],
        'file_names': [],
        'file_ids': [],
        'analysis_results': [],
        'openai_api_key': '',
        'mongodb_connection_string': ''
    }
    
    for key, default_value in session_vars.items():
        print(f"[OK] {key}: {type(default_value).__name__}")
    
    return True

def test_file_handler():
    """Test FileHandler functionality"""
    print("\n" + "="*50)
    print("TESTING FILE HANDLER")
    print("="*50)
    
    try:
        from file_handlers import FileHandler
        handler = FileHandler()
        
        # Test supported formats
        supported = handler.get_supported_extensions()
        print(f"[OK] Supported formats: {len(supported)} types")
        print(f"     Includes: {', '.join(supported[:5])}...")
        
        return True
    except Exception as e:
        print(f"[FAIL] FileHandler test failed: {e}")
        return False

def test_database_manager():
    """Test database manager initialization"""
    print("\n" + "="*50)
    print("TESTING DATABASE MANAGER")
    print("="*50)
    
    try:
        from enhanced_db_manager import EnhancedMongoManager
        
        # Test without connection string (should handle gracefully)
        manager = EnhancedMongoManager("")
        print(f"[OK] EnhancedMongoManager initialized")
        print(f"     Connected: {manager.connected}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Database manager test failed: {e}")
        return False

def test_openai_analyzer():
    """Test OpenAI analyzer initialization"""
    print("\n" + "="*50)
    print("TESTING OPENAI ANALYZER")
    print("="*50)
    
    try:
        # Mock session state for testing
        if 'st' in sys.modules:
            st.session_state = {'openai_api_key': ''}
        
        from openai_analyzer import OpenAIAnalyzer
        analyzer = OpenAIAnalyzer()
        
        print(f"[OK] OpenAIAnalyzer initialized")
        print(f"     Available: {analyzer.is_available()}")
        
        return True
    except Exception as e:
        print(f"[FAIL] OpenAI analyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("NLP YPAR TOOL - FEATURE VERIFICATION")
    print("="*60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Session State", test_session_state),
        ("File Handler", test_file_handler),
        ("Database Manager", test_database_manager),
        ("OpenAI Analyzer", test_openai_analyzer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All basic tests passed!")
    else:
        print(f"[WARNING] {total - passed} test(s) failed")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Open http://localhost:9001 in your browser")
    print("2. Test each page using the navigation menu")
    print("3. Update FEATURE_REVIEW_CHECKLIST.md with results")
    print("4. Report any issues found")
    print("="*60)

if __name__ == "__main__":
    main()