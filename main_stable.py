"""
NLP Tool for YPAR - Stable Version
With comprehensive error handling and stability improvements
"""

import streamlit as st
import logging
import sys
import os
from typing import Optional, Dict, Any, List
import traceback

# Configure page
st.set_page_config(
    page_title="NLP Tool for YPAR - Stable",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import stability fixes first
try:
    from stability_fixes import (
        StableMongoDBManager,
        safe_session_state_update,
        safe_session_state_get,
        validate_and_sanitize_input,
        safe_nltk_download,
        ErrorHandler,
        HealthMonitor,
        SafeMemoryManager
    )
except ImportError as e:
    st.error(f"Failed to import stability modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_ypar.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLTK data safely
safe_nltk_download()

# Safe imports with error handling
def safe_import_modules():
    """Safely import required modules with fallbacks"""
    modules = {}
    
    try:
        import pandas as pd
        modules['pd'] = pd
    except ImportError:
        logger.error("Failed to import pandas")
        st.error("Required module 'pandas' not found. Please install dependencies.")
        st.stop()
    
    try:
        import numpy as np
        modules['np'] = np
    except ImportError:
        logger.error("Failed to import numpy")
        st.error("Required module 'numpy' not found.")
        st.stop()
    
    # Try importing custom modules with fallbacks
    try:
        from config import Config
        modules['Config'] = Config
    except ImportError:
        logger.warning("Config module not found, using defaults")
        modules['Config'] = None
    
    try:
        from file_handlers import FileHandler
        modules['FileHandler'] = FileHandler
    except ImportError:
        logger.warning("FileHandler module not found")
        modules['FileHandler'] = None
    
    try:
        from ui_components_berkeley import UIComponents
        modules['UIComponents'] = UIComponents
    except ImportError:
        logger.warning("UIComponents module not found, using basic UI")
        modules['UIComponents'] = None
    
    try:
        from rag_system import RAGSystem
        modules['RAGSystem'] = RAGSystem
    except ImportError:
        logger.warning("RAG system not available")
        modules['RAGSystem'] = None
    
    return modules

# Initialize modules
modules = safe_import_modules()

# Initialize session state with safety checks
def init_session_state():
    """Initialize session state with thread safety"""
    defaults = {
        'processed_data': [],
        'file_names': [],
        'file_metadata': [],
        'current_file_index': 0,
        'themes': {},
        'sentiments': {},
        'quotes': {},
        'insights': {},
        'file_ids': [],
        'processed_files': set(),
        'analysis_results': [],
        'file_count': 0,
        'analysis_count': 0,
        'memory_manager': SafeMemoryManager(max_size=500),
        'health_status': {},
        'error_log': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            safe_session_state_update(key, value)

# Initialize
init_session_state()

# Database Manager
db_manager = StableMongoDBManager()

class StableNLPProcessor:
    """Stable NLP processor with comprehensive error handling"""
    
    def __init__(self):
        self.db = db_manager
        self.health_monitor = HealthMonitor()
        self.error_handler = ErrorHandler()
    
    def process_file_safely(self, file) -> Optional[Dict[str, Any]]:
        """Process file with full error handling and validation"""
        try:
            # Validate file
            if not file:
                raise ValueError("No file provided")
            
            # Check file size (50MB limit)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > 50 * 1024 * 1024:
                raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds 50MB limit")
            
            # Check system health before processing
            health = self.health_monitor.check_system_health()
            if health['status'] == 'unhealthy':
                raise RuntimeError("System unhealthy, cannot process file")
            
            # Process file based on type
            file_handler = modules.get('FileHandler')
            if file_handler:
                try:
                    content = file_handler.process_file(file)
                except Exception as e:
                    logger.error(f"FileHandler failed: {e}")
                    # Fallback to basic text extraction
                    content = file.read().decode('utf-8', errors='ignore')
            else:
                # Basic fallback
                content = file.read().decode('utf-8', errors='ignore')
            
            # Validate and sanitize content
            content = validate_and_sanitize_input(content)
            if not content:
                raise ValueError("File content is empty or invalid")
            
            # Store in memory manager
            memory_manager = safe_session_state_get('memory_manager')
            if memory_manager:
                memory_manager.add({
                    'filename': file.name,
                    'content': content[:1000],  # Store preview
                    'size': file_size
                })
            
            return {
                'status': 'success',
                'filename': file.name,
                'content': content,
                'size': file_size,
                'preview': content[:500]
            }
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, f"Processing file {getattr(file, 'name', 'unknown')}")
            
            # Log to error history
            error_log = safe_session_state_get('error_log', [])
            error_log.append({
                'file': getattr(file, 'name', 'unknown'),
                'error': str(e),
                'message': error_msg
            })
            safe_session_state_update('error_log', error_log[-50:])  # Keep last 50 errors
            
            return {
                'status': 'error',
                'filename': getattr(file, 'name', 'unknown'),
                'error': error_msg
            }
    
    def analyze_text_safely(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Perform text analysis with error handling"""
        try:
            # Validate input
            text = validate_and_sanitize_input(text)
            if not text:
                raise ValueError("Invalid or empty text")
            
            results = {'type': analysis_type, 'status': 'success'}
            
            if analysis_type == 'sentiment':
                results['data'] = self._analyze_sentiment(text)
            elif analysis_type == 'themes':
                results['data'] = self._extract_themes(text)
            elif analysis_type == 'entities':
                results['data'] = self._extract_entities(text)
            else:
                results['data'] = {'message': 'Analysis type not implemented'}
            
            # Store results safely
            if self.db.connected:
                doc_id = self.db.safe_insert('analysis_results', results)
                results['db_id'] = doc_id
            
            return results
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, f"Analyzing text ({analysis_type})")
            return {
                'type': analysis_type,
                'status': 'error',
                'error': error_msg
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis with fallback"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text[:5000])  # Limit text length
            return {
                'polarity': float(blob.sentiment.polarity),
                'subjectivity': float(blob.sentiment.subjectivity)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'error': 'Sentiment analysis unavailable'}
    
    def _extract_themes(self, text: str) -> Dict[str, Any]:
        """Theme extraction with error handling"""
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=10)
            keywords = kw_extractor.extract_keywords(text[:10000])
            return {'themes': [kw[0] for kw in keywords]}
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return {'themes': []}
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Entity extraction with fallback"""
        try:
            import nltk
            tokens = nltk.word_tokenize(text[:5000])
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags, binary=False)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join(c[0] for c in chunk)
                    entities.append({'text': entity_name, 'type': chunk.label()})
            
            return {'entities': entities[:50]}  # Limit to 50 entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {'entities': []}

# Main Application UI
def main():
    """Main application with stability improvements"""
    try:
        # Apply theme if available
        if modules.get('UIComponents'):
            modules['UIComponents'].apply_berkeley_theme()
            modules['UIComponents'].render_modern_header()
        else:
            st.title("🔬 NLP Tool for YPAR - Stable Version")
            st.write("Advanced Natural Language Processing for Youth Participatory Action Research")
        
        # Initialize processor
        processor = StableNLPProcessor()
        
        # Sidebar with health monitoring
        with st.sidebar:
            st.header("System Health")
            health = processor.health_monitor.check_system_health()
            
            if health['status'] == 'healthy':
                st.success("✅ System Healthy")
            elif health['status'] == 'degraded':
                st.warning("⚠️ System Degraded")
            else:
                st.error("❌ System Unhealthy")
            
            # Show health metrics
            if 'checks' in health:
                for check_name, check_data in health['checks'].items():
                    if 'percent' in check_data:
                        st.metric(check_name.capitalize(), f"{check_data['percent']:.1f}%")
            
            st.divider()
            
            # Navigation
            menu_options = ["📤 Upload & Process", "🔍 Analysis", "📊 Results", "⚙️ Settings"]
            selected = st.selectbox("Navigation", menu_options)
        
        # Main content area
        if selected == "📤 Upload & Process":
            st.header("Upload Files")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['txt', 'pdf', 'docx', 'csv', 'json', 'html', 'md']
            )
            
            if uploaded_files:
                st.write(f"Processing {len(uploaded_files)} file(s)...")
                
                progress_bar = st.progress(0)
                results = []
                
                for idx, file in enumerate(uploaded_files):
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    result = processor.process_file_safely(file)
                    results.append(result)
                    
                    if result['status'] == 'success':
                        st.success(f"✅ {result['filename']} processed successfully")
                        with st.expander(f"Preview: {result['filename']}"):
                            st.text(result['preview'])
                    else:
                        st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")
                
                # Update session state
                safe_session_state_update('file_count', safe_session_state_get('file_count', 0) + len(results))
        
        elif selected == "🔍 Analysis":
            st.header("Text Analysis")
            
            # Get processed files
            memory_manager = safe_session_state_get('memory_manager')
            if memory_manager:
                files = memory_manager.get_all()
                if files:
                    st.write(f"Found {len(files)} processed file(s)")
                    
                    # Analysis options
                    analysis_type = st.selectbox(
                        "Select Analysis Type",
                        ["sentiment", "themes", "entities"]
                    )
                    
                    if st.button("Run Analysis", type="primary"):
                        # Analyze last file for demo
                        last_file = files[-1]
                        if 'content' in last_file:
                            result = processor.analyze_text_safely(
                                last_file['content'],
                                analysis_type
                            )
                            
                            if result['status'] == 'success':
                                st.success("Analysis complete!")
                                st.json(result['data'])
                            else:
                                st.error(f"Analysis failed: {result.get('error')}")
                else:
                    st.info("No files processed yet. Please upload files first.")
            else:
                st.warning("Memory manager not initialized")
        
        elif selected == "📊 Results":
            st.header("Analysis Results")
            
            # Query database for results
            if db_manager.connected:
                results = db_manager.safe_find('analysis_results', {}, limit=10)
                if results:
                    st.write(f"Found {len(results)} result(s)")
                    for result in results:
                        with st.expander(f"{result.get('type', 'Unknown')} Analysis"):
                            st.json(result.get('data', {}))
                else:
                    st.info("No analysis results found")
            else:
                st.warning("Database not connected")
        
        elif selected == "⚙️ Settings":
            st.header("Settings")
            
            # Error log
            st.subheader("Error Log")
            error_log = safe_session_state_get('error_log', [])
            if error_log:
                st.write(f"Recent errors ({len(error_log)})")
                for error in error_log[-5:]:  # Show last 5 errors
                    st.error(f"{error['file']}: {error['message']}")
            else:
                st.success("No errors logged")
            
            # Clear data
            if st.button("Clear Session Data"):
                for key in st.session_state.keys():
                    if key not in ['memory_manager', 'health_status']:
                        del st.session_state[key]
                st.success("Session data cleared")
                st.rerun()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application error: {e}")
        st.error("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical application failure: {e}")
        logger.critical(traceback.format_exc())
    finally:
        # Cleanup
        try:
            db_manager.close()
        except:
            pass