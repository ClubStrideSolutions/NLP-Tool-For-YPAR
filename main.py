"""
NLP Tool for YPAR - Enhanced Cal Edition
Unified version with superior UI, file handling, and RAG system
"""

import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from collections import Counter
import re
from io import BytesIO
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from pymongo import MongoClient
from datetime import datetime
import functools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uuid
import yake
import logging
import sys
import os
import traceback
from streamlit_option_menu import option_menu

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

# Page configuration
st.set_page_config(
    page_title="NLP Tool for YPAR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import enhanced modules
try:
    from ui_components_berkeley import UIComponents
    ui_available = True
except ImportError:
    ui_available = False
    logger.warning("UIComponents not available, using fallback UI")

try:
    from file_handlers import FileHandler
    file_handler_available = True
except ImportError:
    file_handler_available = False
    logger.warning("FileHandler not available, using basic file processing")

try:
    from rag_system import RAGSystem, PersonaManager, ConversationMemory
    rag_available = True
except ImportError:
    rag_available = False
    logger.warning("RAG system not available")

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
    stability_available = True
except ImportError:
    stability_available = False
    logger.warning("Stability modules not available")
    
    # Fallback implementations
    class ErrorHandler:
        def handle_error(self, error, context=""):
            logger.error(f"{context}: {error}")
            return str(error)
    
    class HealthMonitor:
        def check_system_health(self):
            return {'status': 'unknown', 'checks': {}}
    
    def safe_session_state_update(key, value):
        st.session_state[key] = value
    
    def safe_session_state_get(key, default=None):
        return st.session_state.get(key, default)

# Apply enhanced UI theme
if ui_available:
    UIComponents.apply_berkeley_theme()
else:
    # Fallback Cal colors theme
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #001a33 0%, #003262 100%);
    }
    .main .block-container {
        background-color: rgba(0, 50, 98, 0.95);
        padding: 2rem;
        border-radius: 10px;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #FDB515 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #FDB515 0%, #C4820E 100%);
        color: #003262;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(253, 181, 21, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize NLTK
try:
    # Download all required NLTK data
    required_nltk_data = [
        'punkt', 
        'punkt_tab',  # New tokenizer format
        'stopwords', 
        'vader_lexicon', 
        'maxent_ne_chunker', 
        'words', 
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    
    for package in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else package)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                # If specific package fails, try without quiet mode
                if package == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
except Exception as e:
    logger.warning(f"NLTK initialization warning: {e}")

# Configuration
class Config:
    """Application configuration"""
    
    # Cal Colors
    BERKELEY_BLUE = "#003262"
    CALIFORNIA_GOLD = "#FDB515"
    FOUNDERS_ROCK = "#3B7EA1"
    MEDALIST = "#C4820E"
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'xlsx', 'csv', 'json', 'html', 'md', 'rtf', 'tsv']
    BATCH_SIZE = 5
    CACHE_TTL = 3600
    
    @staticmethod
    def get_mongodb_connection_string():
        """Get MongoDB connection string"""
        try:
            return st.secrets.get("mongodb_connection_string") or os.getenv("CONNECTION_STRING")
        except:
            return None

# Enhanced Database Manager
class EnhancedDatabaseManager:
    """Database manager with automatic fallback and caching"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self.cache = {}
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            connection_string = Config.get_mongodb_connection_string()
            if connection_string:
                self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                self.client.server_info()
                self.db = self.client["nlp_tool"]
                self.connected = True
                logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            self.connected = False
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any]) -> Optional[str]:
        """Store analysis with caching"""
        cache_key = f"{file_id}_{analysis_type}"
        self.cache[cache_key] = results
        
        if self.connected and self.db:
            try:
                collection = self.db["analysis_results"]
                document = {
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "results": results,
                    "timestamp": datetime.now()
                }
                result = collection.insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Error storing to MongoDB: {e}")
        
        # Fallback to session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        st.session_state.analysis_results.append({
            "file_id": file_id,
            "analysis_type": analysis_type,
            "results": results,
            "timestamp": datetime.now()
        })
        return file_id
    
    def get_from_cache(self, file_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get from cache first"""
        cache_key = f"{file_id}_{analysis_type}"
        return self.cache.get(cache_key)

# Initialize database
@st.cache_resource
def get_db_manager():
    return EnhancedDatabaseManager()

db_manager = get_db_manager()

# Enhanced File Processing
class EnhancedFileProcessor:
    """Advanced file processing with multiple format support"""
    
    def __init__(self):
        self.file_handler = FileHandler() if file_handler_available else None
        self.error_handler = ErrorHandler() if stability_available else None
    
    def process_file(self, file) -> Optional[Tuple[str, str, str, Dict[str, Any]]]:
        """Process file with enhanced handling"""
        try:
            # Use enhanced file handler if available
            if self.file_handler:
                is_valid, msg = FileHandler.validate_file(file, Config.MAX_FILE_SIZE)
                if not is_valid:
                    st.error(f"File validation failed: {msg}")
                    return None
                
                content, metadata = FileHandler.process_file(file)
                if content:
                    file_id = str(uuid.uuid4())[:8]
                    unique_name = f"{file.name}_{file_id}"
                    return content, file_id, unique_name, metadata
            else:
                # Fallback processing
                file_type = file.type
                file_content = file.read()
                
                if file_type == "text/plain":
                    content = file_content.decode("utf-8", errors='ignore')
                elif "wordprocessingml" in file_type:
                    doc = Document(BytesIO(file_content))
                    content = "\n".join([p.text for p in doc.paragraphs])
                elif "spreadsheetml" in file_type:
                    df = pd.read_excel(BytesIO(file_content))
                    content = df.to_string()
                else:
                    content = file_content.decode("utf-8", errors='ignore')
                
                file_id = str(uuid.uuid4())[:8]
                unique_name = f"{file.name}_{file_id}"
                metadata = {"type": file_type, "size": len(file_content)}
                
                return content, file_id, unique_name, metadata
                
        except Exception as e:
            if self.error_handler:
                error_msg = self.error_handler.handle_error(e, f"Processing {file.name}")
                st.error(error_msg)
            else:
                st.error(f"Error processing file: {str(e)}")
            return None

# Enhanced Text Analyzer with RAG
class EnhancedTextAnalyzer:
    """Text analyzer with RAG support and advanced features"""
    
    def __init__(self):
        self.rag_system = RAGSystem() if rag_available else None
        self.persona_manager = PersonaManager() if rag_available else None
        self.stop_words = set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def analyze_with_persona(_self, text: str, file_id: str, persona_name: str = "researcher") -> str:
        """Analyze text with selected persona"""
        try:
            # Check cache first
            cached = db_manager.get_from_cache(file_id, f"persona_{persona_name}")
            if cached:
                return cached.get("analysis", "")
            
            if _self.rag_system and _self.persona_manager:
                # Use RAG system with persona
                _self.persona_manager.set_active_persona(persona_name)
                persona = _self.persona_manager.active_persona
                
                # Add to RAG knowledge base
                _self.rag_system.add_document(text, {"file_id": file_id, "persona": persona_name})
                
                # Generate analysis with persona context
                query = f"""
                {persona.to_prompt_context()}
                
                Analyze the following text and provide insights based on your persona:
                {text[:5000]}
                """
                
                results = _self.rag_system.query(query, top_k=5)
                analysis = _self.rag_system.generate_response(query, results)
                
                # Store results
                db_manager.store_analysis(file_id, f"persona_{persona_name}", {"analysis": analysis})
                
                return analysis
            else:
                # Fallback to basic analysis
                return _self.analyze_themes(text, file_id)
                
        except Exception as e:
            logger.error(f"Error in persona analysis: {e}")
            return "Analysis failed"
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def analyze_themes(_self, text: str, file_id: str) -> str:
        """Enhanced theme analysis"""
        try:
            # Preprocess
            words = word_tokenize(text.lower())
            filtered = [w for w in words if w.isalnum() and w not in _self.stop_words and len(w) > 2]
            
            if len(filtered) < 10:
                return "Insufficient text for analysis"
            
            # TF-IDF with better parameters
            vectorizer = TfidfVectorizer(
                max_features=min(200, len(filtered)),
                ngram_range=(1, 3),
                min_df=0.01,
                max_df=0.95
            )
            
            doc_matrix = vectorizer.fit_transform([' '.join(filtered)])
            
            # LDA with optimized parameters
            n_topics = min(15, max(5, len(filtered) // 30))
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                learning_method='online',
                random_state=42,
                max_iter=100
            )
            lda.fit(doc_matrix)
            
            # Extract themes
            feature_names = vectorizer.get_feature_names_out()
            themes_list = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-20:][::-1]
                top_words = [feature_names[i] for i in top_indices if i < len(feature_names)]
                themes_list.append({
                    'theme': f"Theme {topic_idx + 1}",
                    'words': top_words[:10],
                    'strength': float(topic[top_indices].sum())
                })
            
            # YAKE keywords
            try:
                kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=20)
                keywords = kw_extractor.extract_keywords(text[:10000])
            except:
                keywords = []
            
            # Format output
            output = "## 📊 Identified Themes\n\n"
            for theme in sorted(themes_list, key=lambda x: x['strength'], reverse=True)[:10]:
                output += f"### {theme['theme']} (Strength: {theme['strength']:.2f})\n"
                output += f"**Key Terms**: {', '.join(theme['words'][:8])}\n\n"
            
            if keywords:
                output += "\n## 🔑 Key Concepts\n\n"
                for kw, score in keywords[:15]:
                    output += f"- **{kw}** (relevance: {1/score:.2f})\n"
            
            # Store and return
            db_manager.store_analysis(file_id, "theme_analysis", {"themes": output})
            return output
            
        except Exception as e:
            logger.error(f"Theme analysis error: {e}")
            return f"Error: {str(e)}"

# Initialize components
file_processor = EnhancedFileProcessor()
text_analyzer = EnhancedTextAnalyzer()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'processed_data': [],
        'file_names': [],
        'file_metadata': [],
        'current_file_index': 0,
        'themes': {},
        'quotes': {},
        'insights': {},
        'file_ids': [],
        'processed_files': set(),
        'analysis_results': [],
        'rag_system': RAGSystem() if rag_available else None,
        'persona_manager': PersonaManager() if rag_available else None,
        'conversation_memory': ConversationMemory() if rag_available else None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Main Application
def main():
    """Enhanced main application"""
    
    # Header with Cal theme
    if ui_available:
        UIComponents.render_modern_header()
    else:
        st.markdown("""
        <h1 style='text-align: center; color: #FDB515; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🔬 NLP Tool for YPAR
        </h1>
        <p style='text-align: center; color: #ffffff; font-size: 1.2em;'>
            Advanced Text Analysis with Cal Excellence
        </p>
        """, unsafe_allow_html=True)
    
    # Enhanced navigation
    with st.sidebar:
        if ui_available:
            selected = option_menu(
                menu_title="Navigation",
                options=[
                    "Home", "Upload Data", "Text Analysis", "Theme Modeling",
                    "Quote Extraction", "Insights", "Visualizations",
                    "RAG Analysis", "Settings"
                ],
                icons=[
                    "house", "cloud-upload", "search", "diagram-3",
                    "chat-quote", "lightbulb", "graph-up",
                    "robot", "gear"
                ],
                menu_icon="list",
                default_index=0,
                styles={
                    "container": {"background-color": "#003262"},
                    "icon": {"color": "#FDB515", "font-size": "20px"},
                    "nav-link": {
                        "color": "#ffffff",
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#1a4a7a"
                    },
                    "nav-link-selected": {"background-color": "#FDB515", "color": "#003262"}
                }
            )
        else:
            selected = st.selectbox(
                "Navigation",
                ["Home", "Upload Data", "Text Analysis", "Theme Modeling",
                 "Quote Extraction", "Insights", "Visualizations",
                 "RAG Analysis", "Settings"]
            )
        
        # System status
        st.divider()
        st.markdown("### 📊 System Status")
        
        # Health monitoring
        if stability_available:
            health_monitor = HealthMonitor()
            health = health_monitor.check_system_health()
            
            if health['status'] == 'healthy':
                st.success("✅ System Healthy")
            elif health['status'] == 'degraded':
                st.warning("⚠️ System Degraded")
            else:
                st.error("❌ System Issues")
        
        # Database status
        if db_manager.connected:
            st.success("✅ Database Connected")
        else:
            st.info("💾 Using Local Storage")
        
        # Files processed
        st.metric("Files Processed", len(st.session_state.processed_data))
    
    # Page routing
    if selected == "Home":
        show_home()
    elif selected == "Upload Data":
        show_upload()
    elif selected == "Text Analysis":
        show_analysis()
    elif selected == "Theme Modeling":
        show_themes()
    elif selected == "Quote Extraction":
        show_quotes()
    elif selected == "Insights":
        show_insights()
    elif selected == "Visualizations":
        show_visualizations()
    elif selected == "RAG Analysis":
        show_rag_analysis()
    elif selected == "Settings":
        show_settings()

def show_home():
    """Enhanced home page"""
    st.markdown("## Welcome to the Enhanced NLP Tool")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ui_available:
            UIComponents.render_feature_card(
                "🚀 Advanced Processing",
                "Support for 10+ file formats including PDF, Word, Excel, HTML, and more",
                "#003262"
            )
        else:
            st.info("🚀 **Advanced Processing**\nMultiple file format support")
    
    with col2:
        if ui_available:
            UIComponents.render_feature_card(
                "🤖 RAG System",
                "Retrieval-Augmented Generation with personas and conversation memory",
                "#003262"
            )
        else:
            st.info("🤖 **RAG System**\nAdvanced AI analysis")
    
    with col3:
        if ui_available:
            UIComponents.render_feature_card(
                "📊 Rich Visualizations",
                "Interactive charts, word clouds, and network graphs",
                "#003262"
            )
        else:
            st.info("📊 **Rich Visualizations**\nInteractive data exploration")
    
    # Stats
    if ui_available:
        UIComponents.render_stats_dashboard({
            "Files Processed": len(st.session_state.processed_data),
            "Analyses Run": len(st.session_state.get('analysis_results', [])),
            "Active Personas": len(st.session_state.get('persona_manager', {}).personas) if rag_available else 0,
            "Knowledge Base Size": st.session_state.get('rag_system', {}).document_store.shape[0] if rag_available and hasattr(st.session_state.get('rag_system', {}), 'document_store') else 0
        })

def show_upload():
    """Enhanced upload page"""
    st.markdown("## 📤 Upload Your Data")
    
    # File uploader with enhanced support
    supported_types = Config.SUPPORTED_FILE_TYPES if file_handler_available else ['txt', 'docx', 'xlsx']
    
    st.info(f"""
    ### Supported Formats
    {', '.join([f'.{ext}' for ext in supported_types])}
    
    Maximum file size: {Config.MAX_FILE_SIZE / (1024*1024):.0f}MB
    """)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=supported_types,
        accept_multiple_files=True
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if st.button("🚀 Process Files", type="primary"):
                progress = st.progress(0)
                
                for idx, file in enumerate(new_files):
                    progress.progress((idx + 1) / len(new_files))
                    
                    result = file_processor.process_file(file)
                    if result:
                        content, file_id, unique_name, metadata = result
                        
                        st.session_state.processed_data.append(content)
                        st.session_state.file_names.append(unique_name)
                        st.session_state.file_ids.append(file_id)
                        st.session_state.file_metadata.append(metadata)
                        st.session_state.processed_files.add(file.name)
                        
                        # Add to RAG system if available
                        if rag_available and st.session_state.rag_system:
                            st.session_state.rag_system.add_document(
                                content,
                                {"file_id": file_id, "filename": file.name, **metadata}
                            )
                        
                        st.success(f"✅ {file.name} processed successfully")
                
                st.balloons()
        
        # Display processed files
        if st.session_state.file_names:
            st.markdown("### 📁 Processed Files")
            
            for i, (name, metadata) in enumerate(zip(st.session_state.file_names, st.session_state.file_metadata)):
                with st.expander(f"📄 {name.split('_')[0]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Type**: {metadata.get('type', 'Unknown')}")
                        st.write(f"**Size**: {metadata.get('size', 0) / 1024:.1f}KB")
                    
                    with col2:
                        st.write(f"**Pages**: {metadata.get('pages', 'N/A')}")
                        st.write(f"**Tables**: {metadata.get('tables', 'N/A')}")
                    
                    with col3:
                        if st.button(f"Remove", key=f"remove_{i}"):
                            st.session_state.processed_data.pop(i)
                            st.session_state.file_names.pop(i)
                            st.session_state.file_ids.pop(i)
                            st.session_state.file_metadata.pop(i)
                            st.rerun()

def show_analysis():
    """Enhanced text analysis page"""
    st.markdown("## 🔍 Text Analysis")
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    selected_file = st.selectbox(
        "Select a file",
        st.session_state.file_names
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Standard", "With Persona", "Comparative", "Deep Dive"]
            )
        
        with col2:
            if analysis_type == "With Persona" and rag_available:
                persona = st.selectbox(
                    "Select Persona",
                    ["researcher", "educator", "youth_advocate", "data_analyst", "policy_maker"]
                )
            else:
                persona = "researcher"
        
        if st.button("🔬 Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                if analysis_type == "With Persona" and rag_available:
                    result = text_analyzer.analyze_with_persona(text, file_id, persona)
                else:
                    result = text_analyzer.analyze_themes(text, file_id)
                
                st.markdown(result)

def show_themes():
    """Theme modeling page"""
    st.markdown("## 📊 Theme Modeling")
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    selected_file = st.selectbox(
        "Select a file",
        st.session_state.file_names
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        if st.button("🎯 Extract Themes", type="primary"):
            with st.spinner("Extracting themes..."):
                themes = text_analyzer.analyze_themes(text, file_id)
                st.session_state.themes[selected_file] = themes
                st.markdown(themes)

def show_quotes():
    """Quote extraction page"""
    st.markdown("## 💬 Quote Extraction")
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.info("Quote extraction with sentiment analysis and context")
    
    selected_file = st.selectbox(
        "Select a file",
        st.session_state.file_names
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        
        if st.button("📝 Extract Quotes", type="primary"):
            with st.spinner("Extracting quotes..."):
                sentences = sent_tokenize(text)[:100]
                quotes = []
                
                for sent in sentences:
                    if len(sent.split()) > 10 and len(sent.split()) < 50:
                        try:
                            blob = TextBlob(sent)
                            if abs(blob.sentiment.polarity) > 0.3:
                                quotes.append({
                                    'text': sent,
                                    'sentiment': blob.sentiment.polarity,
                                    'subjectivity': blob.sentiment.subjectivity
                                })
                        except:
                            pass
                
                if quotes:
                    st.success(f"Found {len(quotes)} significant quotes")
                    for i, quote in enumerate(quotes[:10], 1):
                        sentiment_color = "#27ae60" if quote['sentiment'] > 0 else "#e74c3c"
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {sentiment_color};'>
                            <h4>Quote {i}</h4>
                            <p style='font-style: italic;'>"{quote['text']}"</p>
                            <p><strong>Sentiment:</strong> {quote['sentiment']:.2f} | <strong>Subjectivity:</strong> {quote['subjectivity']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)

def show_insights():
    """Insights generation page"""
    st.markdown("## 💡 Insight Generation")
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    selected_file = st.selectbox(
        "Select a file",
        st.session_state.file_names
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        
        if st.button("🧠 Generate Insights", type="primary"):
            with st.spinner("Generating insights..."):
                # Text statistics
                words = text.split()
                sentences = text.split('.')
                
                # Sentiment analysis
                try:
                    blob = TextBlob(text[:5000])
                    sentiment = blob.sentiment
                except:
                    sentiment = None
                
                # Display insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Words", f"{len(words):,}")
                    st.metric("Vocabulary Size", f"{len(set(words)):,}")
                
                with col2:
                    st.metric("Total Sentences", f"{len(sentences):,}")
                    st.metric("Avg Sentence Length", f"{len(words)/max(len(sentences),1):.1f}")
                
                with col3:
                    if sentiment:
                        st.metric("Sentiment", f"{sentiment.polarity:.2f}")
                        st.metric("Subjectivity", f"{sentiment.subjectivity:.2f}")
                
                # Key patterns
                st.markdown("### 🔍 Detected Patterns")
                
                patterns = []
                if text.count('?') > 5:
                    patterns.append("❓ Questioning/Exploratory tone")
                if any(word in text.lower() for word in ['however', 'but', 'although']):
                    patterns.append("🔄 Contrasting viewpoints")
                if any(word in text.lower() for word in ['therefore', 'thus', 'consequently']):
                    patterns.append("➡️ Causal relationships")
                
                for pattern in patterns:
                    st.write(pattern)

def show_visualizations():
    """Enhanced visualizations page"""
    st.markdown("## 📈 Visualizations")
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["Word Cloud", "Theme Network", "Sentiment Timeline", "Topic Distribution"]
    )
    
    if viz_type == "Word Cloud":
        selected_file = st.selectbox(
            "Select a file",
            st.session_state.file_names + (["All Files"] if len(st.session_state.file_names) > 1 else [])
        )
        
        if selected_file == "All Files":
            text = " ".join(st.session_state.processed_data)
        else:
            file_index = st.session_state.file_names.index(selected_file)
            text = st.session_state.processed_data[file_index]
        
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("Max words", 50, 300, 150)
        with col2:
            colormap = st.selectbox("Color scheme", ["YlOrRd", "Blues", "Greens", "plasma"])
        
        if st.button("Generate Word Cloud", type="primary"):
            with st.spinner("Generating..."):
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap=colormap,
                    max_words=max_words,
                    relative_scaling=0.5
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title("Word Cloud Analysis", fontsize=16, color='#003262')
                st.pyplot(fig)
    
    elif viz_type == "Theme Network":
        if st.session_state.themes:
            st.info("Building theme network from analyzed files...")
            
            # Create network graph
            G = nx.Graph()
            
            for file_name, themes in st.session_state.themes.items():
                # Extract theme words
                theme_words = re.findall(r'\*\*([^*]+)\*\*', themes)
                
                # Add nodes and edges
                for i, word1 in enumerate(theme_words):
                    G.add_node(word1)
                    for word2 in theme_words[i+1:]:
                        G.add_edge(word1, word2)
            
            if G.nodes():
                # Create interactive plot
                pos = nx.spring_layout(G)
                
                edge_trace = go.Scatter(
                    x=[], y=[],
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace['x'] += tuple([x0, x1, None])
                    edge_trace['y'] += tuple([y0, y1, None])
                
                node_trace = go.Scatter(
                    x=[], y=[],
                    text=[],
                    mode='markers+text',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlOrRd',
                        size=10,
                        colorbar=dict(
                            thickness=15,
                            title='Connections',
                            xanchor='left',
                            titleside='right'
                        )
                    )
                )
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y])
                    node_trace['text'] += tuple([node])
                    node_trace['marker']['color'] += tuple([len(G[node])])
                
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Theme Network Analysis',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please analyze some files first to build the theme network")

def show_rag_analysis():
    """RAG system analysis page"""
    st.markdown("## 🤖 RAG Analysis")
    
    if not rag_available:
        st.warning("RAG system not available. Please install required dependencies.")
        return
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    # Persona selection
    col1, col2 = st.columns(2)
    
    with col1:
        persona = st.selectbox(
            "Select Analysis Persona",
            ["researcher", "educator", "youth_advocate", "data_analyst", "policy_maker"]
        )
    
    with col2:
        analysis_depth = st.select_slider(
            "Analysis Depth",
            ["Quick", "Standard", "Deep", "Comprehensive"]
        )
    
    # Query input
    query = st.text_area(
        "Ask a question about your data",
        placeholder="What are the main themes across all documents?",
        height=100
    )
    
    if st.button("🔮 Analyze with RAG", type="primary"):
        if query:
            with st.spinner("Analyzing with RAG system..."):
                try:
                    # Set persona
                    st.session_state.persona_manager.set_active_persona(persona)
                    
                    # Query RAG system
                    results = st.session_state.rag_system.query(query, top_k=10)
                    
                    # Generate response
                    response = st.session_state.rag_system.generate_response(query, results)
                    
                    # Add to conversation memory
                    st.session_state.conversation_memory.add_exchange(query, response)
                    
                    # Display results
                    st.markdown("### 🎯 Analysis Results")
                    st.markdown(response)
                    
                    # Show retrieved documents
                    with st.expander("📚 Retrieved Context"):
                        for i, result in enumerate(results[:5], 1):
                            st.write(f"**Source {i}**: {result.get('metadata', {}).get('filename', 'Unknown')}")
                            st.write(f"Relevance: {result.get('score', 0):.2f}")
                            st.write(result.get('text', '')[:200] + "...")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"RAG analysis failed: {str(e)}")
        else:
            st.warning("Please enter a question")
    
    # Conversation history
    if st.session_state.conversation_memory and hasattr(st.session_state.conversation_memory, 'get_history'):
        with st.expander("💬 Conversation History"):
            history = st.session_state.conversation_memory.get_history()
            for exchange in history[-5:]:
                st.write(f"**Q**: {exchange['query']}")
                st.write(f"**A**: {exchange['response'][:200]}...")
                st.divider()

def show_settings():
    """Settings page"""
    st.markdown("## ⚙️ Settings")
    
    # System info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Information")
        st.write(f"**Files Processed**: {len(st.session_state.processed_data)}")
        st.write(f"**Database**: {'Connected' if db_manager.connected else 'Local Storage'}")
        st.write(f"**RAG System**: {'Available' if rag_available else 'Not Available'}")
        st.write(f"**Enhanced UI**: {'Available' if ui_available else 'Fallback'}")
    
    with col2:
        st.subheader("🔧 Actions")
        
        if st.button("Clear All Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("All data cleared")
            st.rerun()
        
        if st.button("Export Analysis Results"):
            if st.session_state.get('analysis_results'):
                df = pd.DataFrame(st.session_state.analysis_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "analysis_results.csv",
                    "text/csv"
                )
    
    # Theme settings
    st.subheader("🎨 Theme Settings")
    theme_mode = st.radio(
        "Color Scheme",
        ["Cal Colors (Default)", "Dark Mode", "Light Mode"]
    )
    
    if theme_mode != "Cal Colors (Default)":
        st.info("Custom themes coming soon!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or contact support")