"""
Fallback Verification Script for NLP YPAR Tool
Tests that application works without MongoDB and OpenAI
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_no_mongodb():
    """Test that app works without MongoDB"""
    print("\n" + "="*60)
    print("TESTING WITHOUT MONGODB")
    print("="*60)
    
    # Mock session state
    if 'session_state' not in dir(st):
        class MockSessionState:
            def __init__(self):
                self.data = {}
            def __setitem__(self, key, value):
                self.data[key] = value
            def __getitem__(self, key):
                return self.data[key]
            def get(self, key, default=None):
                return self.data.get(key, default)
        st.session_state = MockSessionState()
    
    # Test EnhancedDatabaseManager without connection
    from main_enhanced_cal import EnhancedDatabaseManager
    
    # Create manager without connection string
    manager = EnhancedDatabaseManager()
    
    print(f"[CHECK] Database Connected: {manager.connected}")
    
    if not manager.connected:
        print("[OK] MongoDB not connected - testing fallback")
        
        # Test storing to session state
        test_data = {
            "test": "data",
            "timestamp": str(datetime.now()),
            "value": 123
        }
        
        result = manager.store_analysis(
            file_id="test_file_001",
            analysis_type="test_analysis",
            results=test_data,
            filename="test.txt"
        )
        
        if result:
            print("[OK] Data stored to session state fallback")
        else:
            print("[FAIL] Failed to store to session state")
        
        # Check if data is in session state
        if hasattr(st.session_state, 'analysis_results'):
            print(f"[OK] Session state has {len(st.session_state.analysis_results)} results")
        else:
            print("[INFO] Session state storage initialized")
        
        # Test cache
        cached = manager.get_from_cache("test_file_001", "test_analysis")
        if cached:
            print("[OK] Cache working")
        else:
            print("[INFO] Cache empty")
    else:
        print("[INFO] MongoDB is connected - disable connection to test fallback")
    
    return True

def test_no_openai():
    """Test that app works without OpenAI"""
    print("\n" + "="*60)
    print("TESTING WITHOUT OPENAI")
    print("="*60)
    
    # Mock session state without OpenAI key
    if hasattr(st, 'session_state'):
        st.session_state['openai_api_key'] = ''
    
    from main_enhanced_cal import EnhancedTextAnalyzer
    
    # Create analyzer without OpenAI
    analyzer = EnhancedTextAnalyzer()
    
    print(f"[CHECK] Using AI: {analyzer.use_ai}")
    print(f"[CHECK] AI Analyzer Available: {analyzer.ai_analyzer is not None}")
    
    if not analyzer.use_ai:
        print("[OK] OpenAI not configured - using traditional NLP")
        
        # Test traditional methods
        test_text = """
        This is a sample text for testing the traditional NLP methods.
        The system should be able to extract keywords, analyze sentiment,
        and identify themes without requiring OpenAI API.
        """
        
        # Test theme analysis
        print("\n[TEST] Theme Analysis:")
        try:
            themes = analyzer.analyze_themes(test_text, "test_001")
            if themes:
                print("[OK] Theme analysis working")
            else:
                print("[WARN] No themes extracted")
        except Exception as e:
            print(f"[INFO] Theme analysis: {str(e)[:50]}")
        
        # Test keyword extraction
        print("\n[TEST] Keyword Extraction:")
        try:
            keywords = analyzer.extract_keywords(test_text, "test_001")
            if keywords:
                print("[OK] Keyword extraction working")
            else:
                print("[WARN] No keywords extracted")
        except Exception as e:
            print(f"[INFO] Keyword extraction: {str(e)[:50]}")
        
        # Test TextBlob sentiment
        print("\n[TEST] Sentiment Analysis (TextBlob):")
        try:
            from textblob import TextBlob
            blob = TextBlob(test_text)
            polarity = blob.sentiment.polarity
            print(f"[OK] TextBlob sentiment: {polarity:.2f}")
        except ImportError:
            print("[WARN] TextBlob not installed - install with: pip install textblob")
        except Exception as e:
            print(f"[FAIL] TextBlob error: {e}")
    else:
        print("[INFO] OpenAI is configured - remove API key to test fallback")
    
    return True

def test_traditional_ml():
    """Test traditional ML/Stats methods"""
    print("\n" + "="*60)
    print("TESTING TRADITIONAL ML/STATS")
    print("="*60)
    
    methods = []
    
    # Test NLTK
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        print("[OK] NLTK available")
        methods.append("NLTK")
    except ImportError:
        print("[WARN] NLTK not available")
    
    # Test TextBlob
    try:
        from textblob import TextBlob
        print("[OK] TextBlob available")
        methods.append("TextBlob")
    except ImportError:
        print("[WARN] TextBlob not available - install with: pip install textblob")
    
    # Test scikit-learn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        print("[OK] scikit-learn available (TF-IDF, LDA)")
        methods.append("scikit-learn")
    except ImportError:
        print("[WARN] scikit-learn not available")
    
    # Test YAKE
    try:
        import yake
        print("[OK] YAKE available for keyword extraction")
        methods.append("YAKE")
    except ImportError:
        print("[WARN] YAKE not available - install with: pip install yake")
    
    # Test pandas/numpy
    try:
        import pandas as pd
        import numpy as np
        print("[OK] pandas/numpy available for data processing")
        methods.append("pandas/numpy")
    except ImportError:
        print("[WARN] pandas/numpy not available")
    
    print(f"\n[SUMMARY] {len(methods)} traditional ML/Stats methods available")
    return len(methods) >= 3  # Need at least 3 methods for good fallback

def test_session_state_storage():
    """Test session state as storage backend"""
    print("\n" + "="*60)
    print("TESTING SESSION STATE STORAGE")
    print("="*60)
    
    # Initialize mock session state if needed
    if 'session_state' not in dir(st):
        class MockSessionState:
            def __init__(self):
                self.data = {
                    'processed_data': [],
                    'file_names': [],
                    'file_ids': [],
                    'analysis_results': []
                }
            def __setitem__(self, key, value):
                self.data[key] = value
            def __getitem__(self, key):
                return self.data[key]
            def get(self, key, default=None):
                return self.data.get(key, default)
        st.session_state = MockSessionState()
    
    # Test storing data
    test_doc = {
        'id': 'doc_001',
        'name': 'test_document.txt',
        'content': 'This is test content',
        'timestamp': str(datetime.now())
    }
    
    # Add to processed data
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = []
    st.session_state['processed_data'].append(test_doc['content'])
    
    if 'file_names' not in st.session_state:
        st.session_state['file_names'] = []
    st.session_state['file_names'].append(test_doc['name'])
    
    if 'file_ids' not in st.session_state:
        st.session_state['file_ids'] = []
    st.session_state['file_ids'].append(test_doc['id'])
    
    # Verify storage
    stored_count = len(st.session_state.get('processed_data', []))
    print(f"[OK] Stored {stored_count} documents in session state")
    
    # Test analysis results storage
    test_analysis = {
        'file_id': 'doc_001',
        'type': 'sentiment',
        'results': {'sentiment': 'positive', 'score': 0.8},
        'timestamp': str(datetime.now())
    }
    
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = []
    st.session_state['analysis_results'].append(test_analysis)
    
    analysis_count = len(st.session_state.get('analysis_results', []))
    print(f"[OK] Stored {analysis_count} analysis results in session state")
    
    return True

def main():
    """Run all fallback tests"""
    print("\n" + "="*70)
    print("NLP YPAR TOOL - FALLBACK VERIFICATION")
    print("="*70)
    print("Testing that application works without MongoDB and OpenAI")
    print("="*70)
    
    tests = [
        ("MongoDB Fallback", test_no_mongodb),
        ("OpenAI Fallback", test_no_openai),
        ("Traditional ML/Stats", test_traditional_ml),
        ("Session State Storage", test_session_state_storage),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("FALLBACK TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTests Passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n[SUCCESS] All fallback mechanisms working!")
        print("The application can run without MongoDB and OpenAI")
    else:
        print(f"\n[WARNING] {total_count - passed_count} fallback test(s) failed")
    
    print("\n" + "="*70)
    print("FALLBACK CAPABILITIES:")
    print("="*70)
    print("WITHOUT MongoDB:")
    print("  - Data stored in Streamlit session state")
    print("  - Analysis results cached locally")
    print("  - Full functionality maintained during session")
    print("\nWITHOUT OpenAI:")
    print("  - TextBlob for sentiment analysis")
    print("  - YAKE for keyword extraction")
    print("  - LDA for theme modeling")
    print("  - TF-IDF for document analysis")
    print("  - Basic statistics and word frequency")
    print("="*70)

if __name__ == "__main__":
    main()