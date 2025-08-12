"""
NLP Tool for YPAR - Unified Version
Combines stability features from main_stable.py with advanced analysis from main_improved.py
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

# Configure logging with both file and console output
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
    page_title="NLP Tool for YPAR - Unified",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration class
class Config:
    """Application configuration"""
    
    @staticmethod
    def get_app_config():
        return {
            'page_title': 'NLP Tool for YPAR',
            'page_icon': '🔬',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'supported_file_types': ['txt', 'docx', 'xlsx', 'pdf', 'csv', 'json'],
            'batch_size': 3,
            'cache_ttl': 3600,
            'theme_colors': {
                'primary': '#2874a6',
                'secondary': '#154360',
                'background': '#eaf2f8',
                'text': '#2c3e50',
                'success': '#27ae60'
            }
        }
    
    @staticmethod
    def get_mongodb_connection_string():
        """Get MongoDB connection string from environment or Streamlit secrets"""
        try:
            return st.secrets.get("mongodb_connection_string") or os.getenv("CONNECTION_STRING")
        except:
            return None

config = Config.get_app_config()

# Import stability features
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
    stability_modules_available = True
    logger.info("Stability modules loaded successfully")
except ImportError as e:
    logger.warning(f"Stability modules not available: {e}. Using fallback implementations.")
    stability_modules_available = False
    
    # Fallback implementations
    class ErrorHandler:
        def handle_error(self, error, context=""):
            logger.error(f"{context}: {error}")
            return str(error)
    
    class HealthMonitor:
        def check_system_health(self):
            return {'status': 'unknown', 'checks': {}}
    
    class SafeMemoryManager:
        def __init__(self, max_size=500):
            self.data = []
            self.max_size = max_size
        
        def add(self, item):
            self.data.append(item)
            if len(self.data) > self.max_size:
                self.data.pop(0)
        
        def get_all(self):
            return self.data
    
    def safe_session_state_update(key, value):
        st.session_state[key] = value
    
    def safe_session_state_get(key, default=None):
        return st.session_state.get(key, default)
    
    def validate_and_sanitize_input(text):
        if not text:
            return ""
        return str(text).strip()
    
    def safe_nltk_download():
        packages = ['punkt', 'stopwords', 'vader_lexicon', 'maxent_ne_chunker', 'words', 'averaged_perceptron_tagger']
        for package in packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    pass

# Import utility functions
try:
    from utils import (
        sanitize_input, validate_file, generate_file_id,
        chunk_text, format_results, cache_key_generator,
        safe_json_parse, display_progress, get_word_statistics
    )
    utils_available = True
except ImportError:
    utils_available = False
    
    # Fallback implementations
    def sanitize_input(text):
        if not text:
            return ""
        return str(text).strip()
    
    def validate_file(file, max_size):
        if not file:
            return False, "No file provided"
        return True, "OK"
    
    def generate_file_id(content, filename):
        return str(uuid.uuid4())[:8]
    
    def display_progress(operation, total):
        progress_bar = st.progress(0)
        def update(completed):
            progress_bar.progress(completed / total)
        return update
    
    def get_word_statistics(text):
        words = text.split()
        sentences = text.split('.')
        unique_words = set(words)
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_words': len(unique_words),
            'vocabulary_richness': len(unique_words) / max(len(words), 1)
        }

# Download NLTK data
safe_nltk_download() if stability_modules_available else None

# Custom CSS
def apply_custom_css():
    """Apply custom CSS styling"""
    colors = config['theme_colors']
    st.markdown(f"""
    <style>
    .main-title {{
        font-size: 2.8em;
        color: {colors['secondary']};
        text-align: center;
        margin-bottom: 1em;
        font-weight: bold;
    }}
    .section-title {{
        font-size: 2em;
        color: {colors['secondary']};
        margin-top: 1em;
        margin-bottom: 0.5em;
        border-bottom: 3px solid {colors['primary']};
        padding-bottom: 0.3em;
    }}
    .info-box {{
        background-color: {colors['background']};
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid {colors['primary']};
        color: {colors['text']};
    }}
    .feature-card {{
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        border: 1px solid #bdc3c7;
        color: {colors['text']};
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }}
    .stButton>button {{
        background-color: {colors['primary']};
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {colors['secondary']};
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    .result-box {{
        background-color: {colors['background']};
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        border-left: 5px solid {colors['success']};
        color: {colors['text']};
    }}
    .health-good {{
        color: #27ae60;
        font-weight: bold;
    }}
    .health-warning {{
        color: #f39c12;
        font-weight: bold;
    }}
    .health-error {{
        color: #e74c3c;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Enhanced Database Manager with stability features
class UnifiedDatabaseManager:
    """Unified database manager with stability and fallback mechanisms"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self.error_handler = ErrorHandler() if stability_modules_available else None
        self._connect()
    
    def _connect(self):
        """Establish database connection with error handling"""
        try:
            if stability_modules_available:
                # Try using stable MongoDB manager
                from stability_fixes import StableMongoDBManager
                stable_manager = StableMongoDBManager()
                if stable_manager.connected:
                    self.db = stable_manager.db
                    self.connected = True
                    logger.info("Connected using StableMongoDBManager")
                    return
            
            # Fallback to direct connection
            connection_string = Config.get_mongodb_connection_string()
            if connection_string:
                self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                self.client.server_info()  # Force connection test
                self.db = self.client["nlp_tool"]
                self.connected = True
                logger.info("MongoDB connection established")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using local storage.")
            self.connected = False
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any]) -> Optional[str]:
        """Store analysis results with automatic fallback"""
        try:
            if self.connected and self.db:
                collection = self.db["analysis_results"]
                document = {
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "results": results,
                    "timestamp": datetime.now()
                }
                result = collection.insert_one(document)
                return str(result.inserted_id) if result.inserted_id else None
        except Exception as e:
            logger.error(f"Error storing to MongoDB: {e}")
        
        # Fallback to session state
        return self._store_local(file_id, analysis_type, results)
    
    def _store_local(self, file_id: str, analysis_type: str, results: Dict[str, Any]) -> str:
        """Store results in session state as fallback"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        st.session_state.analysis_results.append({
            "file_id": file_id,
            "analysis_type": analysis_type,
            "results": results,
            "timestamp": datetime.now()
        })
        return file_id
    
    def get_analysis_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Retrieve analysis history with fallback"""
        try:
            if self.connected and self.db:
                collection = self.db["analysis_results"]
                return list(collection.find({"file_id": file_id}).sort("timestamp", -1))
        except Exception as e:
            logger.error(f"Error retrieving from MongoDB: {e}")
        
        # Fallback to session state
        if 'analysis_results' in st.session_state:
            return [r for r in st.session_state.analysis_results if r['file_id'] == file_id]
        return []
    
    def close(self):
        """Close database connection safely"""
        try:
            if self.client:
                self.client.close()
        except:
            pass

# Initialize database manager
@st.cache_resource
def get_db_manager():
    """Get cached database manager instance"""
    return UnifiedDatabaseManager()

db_manager = get_db_manager()

# Enhanced Text Analyzer with stability features
class UnifiedTextAnalyzer:
    """Unified text analyzer with error handling and performance optimizations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
        self.error_handler = ErrorHandler() if stability_modules_available else None
        self.health_monitor = HealthMonitor() if stability_modules_available else None
    
    @st.cache_data(ttl=config['cache_ttl'])
    def analyze_themes(_self, text: str, file_id: str) -> str:
        """Analyze themes with error recovery"""
        try:
            # Check system health before heavy processing
            if _self.health_monitor:
                health = _self.health_monitor.check_system_health()
                if health['status'] == 'unhealthy':
                    return "System resources insufficient for theme analysis"
            
            text = sanitize_input(text) if utils_available else text.strip()
            if not text or len(text) < 50:
                return "Insufficient text for theme analysis"
            
            # Limit text length for stability
            max_text_length = 50000
            if len(text) > max_text_length:
                text = text[:max_text_length]
                logger.warning(f"Text truncated to {max_text_length} characters for analysis")
            
            # Enhanced preprocessing
            word_tokens = word_tokenize(text.lower())
            filtered_text = [
                w for w in word_tokens 
                if w.isalnum() and w not in _self.stop_words and len(w) > 2
            ]
            
            if len(filtered_text) < 10:
                return "Insufficient meaningful words for theme analysis"
            
            # Create document-term matrix with error handling
            try:
                vectorizer = TfidfVectorizer(
                    max_features=min(100, len(filtered_text)),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=0.01,
                    max_df=0.95
                )
                
                doc_term_matrix = vectorizer.fit_transform([' '.join(filtered_text)])
                
                # Apply LDA with stability checks
                n_topics = min(10, max(3, len(filtered_text) // 20))
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    learning_method='online',
                    random_state=42,
                    max_iter=50
                )
                lda.fit(doc_term_matrix)
                
                # Extract themes
                feature_names = vectorizer.get_feature_names_out()
                themes_list = []
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-15:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx if i < len(feature_names)]
                    theme_strength = topic[top_words_idx].sum() if len(top_words_idx) > 0 else 0
                    themes_list.append({
                        'theme': f"Theme {topic_idx + 1}",
                        'words': top_words[:8],
                        'strength': float(theme_strength)
                    })
            except Exception as e:
                logger.error(f"LDA analysis failed: {e}")
                themes_list = []
            
            # Use YAKE for keyword extraction as backup/enhancement
            try:
                kw_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,
                    dedupLim=0.7,
                    top=15,
                    features=None
                )
                keywords = kw_extractor.extract_keywords(text[:10000])
            except Exception as e:
                logger.error(f"YAKE extraction failed: {e}")
                keywords = []
            
            # Format results
            themes_text = "## Identified Themes\n\n"
            
            if themes_list:
                for theme in sorted(themes_list, key=lambda x: x['strength'], reverse=True):
                    themes_text += f"### {theme['theme']} (Strength: {theme['strength']:.2f})\n"
                    themes_text += f"**Key Terms**: {', '.join(theme['words'])}\n\n"
            else:
                themes_text += "Could not extract themes using topic modeling.\n\n"
            
            if keywords:
                themes_text += "\n## Key Concepts\n\n"
                for kw, score in keywords[:10]:
                    themes_text += f"- **{kw}** (relevance: {1/score:.2f})\n"
            
            # Add statistics
            stats = get_word_statistics(text) if utils_available else {
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'vocabulary_richness': 0.3
            }
            themes_text += f"\n## Text Statistics\n"
            themes_text += f"- Words: {stats['word_count']}\n"
            themes_text += f"- Sentences: {stats['sentence_count']}\n"
            themes_text += f"- Vocabulary Richness: {stats['vocabulary_richness']:.2%}\n"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="theme_analysis",
                results={"themes": themes_text}
            )
            
            return themes_text
            
        except Exception as e:
            error_msg = _self.error_handler.handle_error(e, "Theme analysis") if _self.error_handler else str(e)
            logger.error(f"Theme analysis error: {error_msg}")
            return f"Error analyzing themes: {error_msg}"
    
    @st.cache_data(ttl=config['cache_ttl'])
    def extract_quotes(_self, text: str, file_id: str) -> str:
        """Extract quotes with stability improvements"""
        try:
            text = sanitize_input(text) if utils_available else text.strip()
            if not text:
                return "No text provided for analysis"
            
            # Limit text for stability
            if len(text) > 50000:
                text = text[:50000]
            
            sentences = sent_tokenize(text)
            if not sentences:
                return "No sentences found in text"
            
            scored_sentences = []
            
            for sent in sentences[:500]:  # Limit number of sentences
                try:
                    score = 0
                    sent_clean = sent.strip()
                    
                    # Skip very short or very long sentences
                    word_count = len(sent_clean.split())
                    if word_count < 5 or word_count > 100:
                        continue
                    
                    # Scoring logic
                    if 10 <= word_count <= 50:
                        score += 3
                    
                    if '"' in sent or "'" in sent or '"' in sent:
                        score += 5
                    
                    # Safe sentiment analysis
                    try:
                        blob = TextBlob(sent)
                        polarity = abs(blob.sentiment.polarity)
                        subjectivity = blob.sentiment.subjectivity
                        
                        if polarity > 0.3:
                            score += 3
                        if subjectivity > 0.5:
                            score += 2
                    except:
                        pass
                    
                    # Key phrases
                    key_phrases = [
                        'believe', 'think', 'feel', 'important', 'significant',
                        'challenge', 'opportunity', 'experience', 'understand'
                    ]
                    
                    sent_lower = sent.lower()
                    for phrase in key_phrases:
                        if phrase in sent_lower:
                            score += 1
                    
                    if score > 0:
                        scored_sentences.append((sent_clean, score))
                except Exception as e:
                    logger.warning(f"Error scoring sentence: {e}")
                    continue
            
            # Sort and select top quotes
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_quotes = scored_sentences[:min(15, len(scored_sentences))]
            
            if not top_quotes:
                return "No significant quotes found"
            
            quotes_text = "## Significant Quotes\n\n"
            
            for i, (quote, score) in enumerate(top_quotes, 1):
                quotes_text += f"### Quote {i}\n"
                quotes_text += f"> {quote}\n\n"
                quotes_text += f"**Relevance Score**: {score}/20\n\n"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="quote_extraction",
                results={"quotes": quotes_text}
            )
            
            return quotes_text
            
        except Exception as e:
            error_msg = _self.error_handler.handle_error(e, "Quote extraction") if _self.error_handler else str(e)
            logger.error(f"Quote extraction error: {error_msg}")
            return f"Error extracting quotes: {error_msg}"
    
    @st.cache_data(ttl=config['cache_ttl'])
    def generate_insights(_self, text: str, file_id: str) -> str:
        """Generate insights with comprehensive error handling"""
        try:
            text = sanitize_input(text) if utils_available else text.strip()
            if not text:
                return "No text provided for analysis"
            
            # Limit text for stability
            if len(text) > 50000:
                text = text[:50000]
            
            # Get basic statistics
            stats = get_word_statistics(text) if utils_available else {
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'unique_words': len(set(text.split())),
                'vocabulary_richness': 0.3
            }
            
            sentences = sent_tokenize(text)[:100]  # Limit sentences
            
            # Word frequency analysis
            words = word_tokenize(text.lower())
            filtered_words = [w for w in words if w.isalnum() and len(w) > 3]
            if _self.stop_words:
                filtered_words = [w for w in filtered_words if w not in _self.stop_words]
            
            word_freq = Counter(filtered_words)
            common_words = word_freq.most_common(15)
            
            # Safe sentiment analysis
            sentiment_info = {
                'polarity': 0,
                'subjectivity': 0,
                'label': 'Neutral',
                'trend': 'stable'
            }
            
            try:
                blob = TextBlob(text[:5000])  # Limit text for sentiment
                sentiment_info['polarity'] = blob.sentiment.polarity
                sentiment_info['subjectivity'] = blob.sentiment.subjectivity
                sentiment_info['label'] = 'Positive' if blob.sentiment.polarity > 0.1 else \
                                         'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral'
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
            
            # Build insights
            insights_text = "## Key Insights\n\n"
            
            # Text characteristics
            insights_text += "### Text Characteristics\n"
            insights_text += f"- **Total words**: {stats['word_count']:,}\n"
            insights_text += f"- **Total sentences**: {stats['sentence_count']:,}\n"
            insights_text += f"- **Average sentence length**: {stats['word_count']/max(stats['sentence_count'], 1):.1f} words\n"
            insights_text += f"- **Vocabulary richness**: {stats['vocabulary_richness']:.2%}\n"
            insights_text += f"- **Unique words**: {stats.get('unique_words', 'N/A')}\n\n"
            
            # Sentiment insights
            insights_text += "### Sentiment Analysis\n"
            insights_text += f"- **Overall sentiment**: {sentiment_info['label']}\n"
            insights_text += f"- **Sentiment strength**: {abs(sentiment_info['polarity']):.2f}\n"
            insights_text += f"- **Subjectivity**: {sentiment_info['subjectivity']:.2f}\n\n"
            
            # Key terms
            if common_words:
                insights_text += "### Most Frequent Terms\n"
                for word, count in common_words[:10]:
                    frequency_pct = (count / len(filtered_words)) * 100 if filtered_words else 0
                    insights_text += f"- **{word}**: {count} occurrences ({frequency_pct:.1f}%)\n"
            
            # Pattern observations
            insights_text += "\n### Patterns Observed\n"
            
            patterns_found = []
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['however', 'but', 'although', 'despite']):
                patterns_found.append("- **Contrasting viewpoints** detected")
            
            if any(word in text_lower for word in ['therefore', 'thus', 'consequently', 'because']):
                patterns_found.append("- **Causal relationships** detected")
            
            if text.count('?') > 2:
                patterns_found.append("- **Questioning/exploratory** tone detected")
            
            if patterns_found:
                insights_text += "\n".join(patterns_found)
            else:
                insights_text += "- No significant rhetorical patterns detected"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="insight_generation",
                results={"insights": insights_text}
            )
            
            return insights_text
            
        except Exception as e:
            error_msg = _self.error_handler.handle_error(e, "Insight generation") if _self.error_handler else str(e)
            logger.error(f"Insight generation error: {error_msg}")
            return f"Error generating insights: {error_msg}"

# Initialize text analyzer
text_analyzer = UnifiedTextAnalyzer()

# Safe file processing
@st.cache_data(ttl=config['cache_ttl'])
def process_file_safely(file_content: bytes, file_name: str, file_type: str) -> Optional[Tuple[str, str, str]]:
    """Process file with comprehensive error handling"""
    try:
        content = ""
        
        # Process based on file type
        if file_type == "text/plain":
            content = file_content.decode("utf-8", errors='ignore')
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(BytesIO(file_content))
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except:
                content = "Error reading Word document"
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            try:
                df = pd.read_excel(BytesIO(file_content))
                content = df.to_string()
            except:
                content = "Error reading Excel file"
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return None
        
        # Validate content
        if stability_modules_available:
            content = validate_and_sanitize_input(content)
        else:
            content = content.strip()
        
        if not content or len(content) < 10:
            logger.warning(f"File {file_name} has insufficient content")
            return None
        
        # Generate unique identifier
        file_id = generate_file_id(content, file_name) if utils_available else str(uuid.uuid4())[:8]
        unique_file_name = f"{file_name}_{file_id[:8]}"
        
        return content, file_id, unique_file_name
        
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")
        return None

# Initialize session state
def init_session_state():
    """Initialize session state with all required variables"""
    defaults = {
        'processed_data': [],
        'file_names': [],
        'current_file_index': 0,
        'themes': {},
        'sentiments': {},
        'quotes': {},
        'insights': {},
        'file_ids': [],
        'processed_files': set(),
        'analysis_results': [],
        'error_log': [],
        'file_count': 0,
        'analysis_count': 0
    }
    
    # Add stability features if available
    if stability_modules_available:
        defaults['memory_manager'] = SafeMemoryManager(max_size=500)
        defaults['health_status'] = {}
    
    for key, value in defaults.items():
        if key not in st.session_state:
            if stability_modules_available:
                safe_session_state_update(key, value)
            else:
                st.session_state[key] = value

init_session_state()

# Main application
def main():
    """Main application with unified features"""
    try:
        # Title
        st.markdown('<h1 class="main-title">NLP Tool for YPAR - Unified Edition</h1>', unsafe_allow_html=True)
        
        # Initialize health monitor if available
        health_monitor = HealthMonitor() if stability_modules_available else None
        error_handler = ErrorHandler() if stability_modules_available else None
        
        # Sidebar with system status
        with st.sidebar:
            st.markdown('<h2 class="sidebar-title">Navigation & Status</h2>', unsafe_allow_html=True)
            
            # System health display
            if health_monitor:
                health = health_monitor.check_system_health()
                
                st.subheader("System Health")
                if health['status'] == 'healthy':
                    st.markdown('<span class="health-good">✅ System Healthy</span>', unsafe_allow_html=True)
                elif health['status'] == 'degraded':
                    st.markdown('<span class="health-warning">⚠️ System Degraded</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="health-error">❌ System Unhealthy</span>', unsafe_allow_html=True)
                
                # Show metrics if available
                if 'checks' in health:
                    for check_name, check_data in health['checks'].items():
                        if 'percent' in check_data:
                            st.metric(check_name.capitalize(), f"{check_data['percent']:.1f}%")
            
            # Database status
            st.subheader("Database Status")
            if db_manager.connected:
                st.success("✅ MongoDB Connected")
            else:
                st.warning("⚠️ Using Local Storage")
            
            st.divider()
            
            # Navigation
            pages = [
                "🏠 Home",
                "📤 Data Upload",
                "🔍 Text Processing",
                "📊 Topic Modeling",
                "💬 Quote Extraction",
                "💡 Insight Generation",
                "📈 Visualization",
                "🏷️ Named Entity Recognition",
                "🎯 Intent Detection",
                "⚖️ Ethics & Bias",
                "📜 Analysis History",
                "⚙️ Settings"
            ]
            
            page = st.radio("Navigate to:", pages, label_visibility="collapsed")
        
        # Main content area
        if page == "🏠 Home":
            show_home_page()
        elif page == "📤 Data Upload":
            show_upload_page()
        elif page == "🔍 Text Processing":
            show_processing_page()
        elif page == "📊 Topic Modeling":
            show_topic_modeling_page()
        elif page == "💬 Quote Extraction":
            show_quote_extraction_page()
        elif page == "💡 Insight Generation":
            show_insight_generation_page()
        elif page == "📈 Visualization":
            show_visualization_page()
        elif page == "🏷️ Named Entity Recognition":
            show_ner_page()
        elif page == "🎯 Intent Detection":
            show_intent_detection_page()
        elif page == "⚖️ Ethics & Bias":
            show_ethics_page()
        elif page == "📜 Analysis History":
            show_history_page()
        elif page == "⚙️ Settings":
            show_settings_page()
        
        # Footer
        show_footer()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application error: {e}")
        st.error("Please refresh the page or contact support if the issue persists.")

def show_home_page():
    """Display home page"""
    st.markdown('<h2 class="section-title">Welcome to the NLP Tool for YPAR</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Advanced Analysis</h3>
            <p>Powerful NLP tools for analyzing qualitative data with educational context.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Quote Extraction</h3>
            <p>Identify and analyze representative quotes with context and significance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🛡️ Stability Features</h3>
            <p>Enhanced error handling, health monitoring, and automatic recovery.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Visualization</h3>
            <p>Interactive visualizations to understand and communicate findings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⚖️ Ethics & Bias</h3>
            <p>Comprehensive analysis of potential biases and ethical considerations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>💾 Flexible Storage</h3>
            <p>Automatic fallback between MongoDB and local storage.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h3>About This Tool</h3>
        <p>This unified version combines the best features from multiple implementations:</p>
        <ul>
            <li>✅ Stability features from main_stable.py</li>
            <li>✅ Advanced analysis from main_improved.py</li>
            <li>✅ Comprehensive error handling and recovery</li>
            <li>✅ Health monitoring and resource management</li>
            <li>✅ Flexible database connectivity with fallback</li>
        </ul>
        <p><strong>Version:</strong> 3.0 Unified</p>
        <p><strong>Status:</strong> {'✅ All Systems Operational' if db_manager.connected else '⚠️ Running in Offline Mode'}</p>
        <p><strong>Files Processed:</strong> {len(st.session_state.processed_data)}</p>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    """Display data upload page with error handling"""
    st.markdown('<h2 class="section-title">Upload Your Data</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>📁 Data Upload Guidelines</h3>
        <p>Supported file formats:</p>
        <ul>
            <li>Text files (.txt)</li>
            <li>Word documents (.docx)</li>
            <li>Excel spreadsheets (.xlsx)</li>
        </ul>
        <p>Maximum file size: 10MB per file</p>
        <p>Files are processed with automatic error recovery and validation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'docx', 'xlsx'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if st.button("Process Files", type="primary"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                for idx, file in enumerate(new_files):
                    progress_bar.progress((idx + 1) / len(new_files))
                    
                    try:
                        # Validate file
                        if utils_available:
                            is_valid, msg = validate_file(file, config['max_file_size'])
                            if not is_valid:
                                results_container.warning(f"⚠️ {file.name}: {msg}")
                                continue
                        
                        # Read and process file
                        file_content = file.read()
                        result = process_file_safely(file_content, file.name, file.type)
                        
                        if result:
                            content, file_id, unique_name = result
                            st.session_state.processed_data.append(content)
                            st.session_state.file_names.append(unique_name)
                            st.session_state.file_ids.append(file_id)
                            st.session_state.processed_files.add(file.name)
                            
                            # Store in memory manager if available
                            if stability_modules_available:
                                memory_manager = safe_session_state_get('memory_manager')
                                if memory_manager:
                                    memory_manager.add({
                                        'filename': file.name,
                                        'content': content[:1000],
                                        'size': len(file_content)
                                    })
                            
                            results_container.success(f"✅ {file.name} processed successfully")
                        else:
                            results_container.error(f"❌ Failed to process {file.name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file.name}: {e}")
                        results_container.error(f"❌ {file.name}: {str(e)}")
                        
                        # Log error if error logging is available
                        if 'error_log' in st.session_state:
                            st.session_state.error_log.append({
                                'file': file.name,
                                'error': str(e),
                                'timestamp': datetime.now()
                            })
                
                progress_bar.progress(1.0)
                st.balloons()
        else:
            st.info("All selected files have already been processed.")
        
        # Display processed files
        if st.session_state.file_names:
            st.markdown("""
            <div class="result-box">
                <h3>📋 Processed Files</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (name, content) in enumerate(zip(st.session_state.file_names, st.session_state.processed_data)):
                with st.expander(f"📄 {name.split('_')[0]}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        stats = get_word_statistics(content) if utils_available else {
                            'word_count': len(content.split()),
                            'sentence_count': len(content.split('.'))
                        }
                        st.write(f"**Words:** {stats['word_count']:,}")
                        st.write(f"**Sentences:** {stats['sentence_count']:,}")
                    
                    with col2:
                        if st.button(f"View", key=f"view_{i}"):
                            st.text_area("Content Preview", content[:1000] + "...", height=200)
                    
                    with col3:
                        if st.button(f"Remove", key=f"remove_{i}", type="secondary"):
                            st.session_state.processed_data.pop(i)
                            st.session_state.file_names.pop(i)
                            st.session_state.file_ids.pop(i)
                            st.rerun()

def show_processing_page():
    """Display text processing page"""
    st.markdown('<h2 class="section-title">Text Processing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    selected_file = st.selectbox(
        "Select a file to process",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        st.session_state.current_file_index = file_index
        
        st.markdown("""
        <div class="info-box">
            <h3>About Text Processing</h3>
            <p>Prepare your text for analysis through cleaning, standardization, and pattern identification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        text = st.session_state.processed_data[file_index]
        stats = get_word_statistics(text) if utils_available else {
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'unique_words': len(set(text.split())),
            'vocabulary_richness': 0.3
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", f"{stats['word_count']:,}")
        with col2:
            st.metric("Unique Words", f"{stats.get('unique_words', 'N/A')}")
        with col3:
            st.metric("Vocabulary Richness", f"{stats['vocabulary_richness']:.1%}")
        
        with st.expander("View Text Sample"):
            st.text_area("Text", text[:2000] + "...", height=200)

def show_topic_modeling_page():
    """Display topic modeling page"""
    st.markdown('<h2 class="section-title">Topic Modeling</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Topic Modeling</h3>
        <p>Identify key themes using LDA and keyword extraction with automatic error recovery.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        if st.button("Analyze Themes", type="primary"):
            with st.spinner("Analyzing themes..."):
                try:
                    themes = text_analyzer.analyze_themes(text, file_id)
                    if themes:
                        st.session_state.themes[selected_file] = themes
                        st.markdown(themes)
                except Exception as e:
                    st.error(f"Error analyzing themes: {str(e)}")

def show_quote_extraction_page():
    """Display quote extraction page"""
    st.markdown('<h2 class="section-title">Quote Extraction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Quote Extraction</h3>
        <p>Identify significant quotes using sentence scoring and sentiment analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        if st.button("Extract Quotes", type="primary"):
            with st.spinner("Extracting quotes..."):
                try:
                    quotes = text_analyzer.extract_quotes(text, file_id)
                    if quotes:
                        st.session_state.quotes[selected_file] = quotes
                        st.markdown(quotes)
                except Exception as e:
                    st.error(f"Error extracting quotes: {str(e)}")

def show_insight_generation_page():
    """Display insight generation page"""
    st.markdown('<h2 class="section-title">Insight Generation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Insight Generation</h3>
        <p>Generate insights using statistical analysis and pattern recognition.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Generating insights..."):
                try:
                    insights = text_analyzer.generate_insights(text, file_id)
                    if insights:
                        st.session_state.insights[selected_file] = insights
                        st.markdown(insights)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")

def show_visualization_page():
    """Display visualization page"""
    st.markdown('<h2 class="section-title">Data Visualization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    viz_option = st.radio(
        "Select visualization type",
        ["Word Cloud", "Theme Network", "Combined Analysis"],
        horizontal=True
    )
    
    if viz_option == "Word Cloud":
        selected_file = st.selectbox(
            "Select a file to visualize",
            st.session_state.file_names + (["All Files"] if len(st.session_state.file_names) > 1 else [])
        )
        
        if selected_file == "All Files":
            text_to_visualize = "\n".join(st.session_state.processed_data)
        else:
            file_index = st.session_state.file_names.index(selected_file)
            text_to_visualize = st.session_state.processed_data[file_index]
        
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("Maximum words", 50, 200, 100)
        with col2:
            colormap = st.selectbox("Color scheme", ["viridis", "plasma", "inferno", "magma"])
        
        if st.button("Generate Word Cloud", type="primary"):
            with st.spinner("Generating word cloud..."):
                try:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap=colormap,
                        max_words=max_words
                    ).generate(text_to_visualize)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")

def show_ner_page():
    """Display named entity recognition page"""
    st.markdown('<h2 class="section-title">Named Entity Recognition</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Named Entity Recognition</h3>
        <p>Identify named entities using NLTK's NER capabilities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        
        if st.button("Analyze Entities", type="primary"):
            with st.spinner("Processing entities..."):
                try:
                    # Limit text for performance
                    text_sample = text[:5000]
                    tokens = word_tokenize(text_sample)
                    pos_tags = pos_tag(tokens)
                    chunks = ne_chunk(pos_tags, binary=False)
                    
                    entities = []
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_text = ' '.join(c[0] for c in chunk)
                            entities.append({
                                'text': entity_text,
                                'type': chunk.label()
                            })
                    
                    if entities:
                        df = pd.DataFrame(entities)
                        st.dataframe(df)
                        
                        # Entity type distribution
                        entity_counts = df['type'].value_counts()
                        fig = px.bar(
                            x=entity_counts.index,
                            y=entity_counts.values,
                            title="Entity Distribution by Type",
                            labels={'x': 'Entity Type', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No named entities found")
                        
                except Exception as e:
                    st.error(f"Error processing entities: {str(e)}")

def show_intent_detection_page():
    """Display intent detection page"""
    st.markdown('<h2 class="section-title">Intent Detection</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Intent Detection</h3>
        <p>Detect common intents in text using pattern matching.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        
        if st.button("Detect Intent", type="primary"):
            with st.spinner("Analyzing intent..."):
                try:
                    intent_patterns = {
                        'Question': r'\?|who|what|when|where|why|how',
                        'Request': r'please|could|would|can you|will you',
                        'Opinion': r'think|believe|feel|seems|appears',
                        'Suggestion': r'should|could|might|recommend|suggest',
                        'Complaint': r'problem|issue|wrong|bad|terrible'
                    }
                    
                    detected_intents = []
                    sentences = sent_tokenize(text[:5000])
                    
                    for sent in sentences:
                        sent_lower = sent.lower()
                        for intent, pattern in intent_patterns.items():
                            if re.search(pattern, sent_lower):
                                detected_intents.append({
                                    'sentence': sent[:100] + '...' if len(sent) > 100 else sent,
                                    'intent': intent
                                })
                    
                    if detected_intents:
                        df = pd.DataFrame(detected_intents)
                        st.dataframe(df)
                        
                        intent_counts = Counter([d['intent'] for d in detected_intents])
                        fig = px.pie(
                            values=list(intent_counts.values()),
                            names=list(intent_counts.keys()),
                            title="Intent Distribution"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.info("No specific intents detected")
                    
                except Exception as e:
                    st.error(f"Error detecting intent: {str(e)}")

def show_ethics_page():
    """Display ethics and bias analysis page"""
    st.markdown('<h2 class="section-title">Ethics & Bias Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Ethics & Bias Analysis</h3>
        <p>Identify potential biases using word frequency and pattern analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.current_file_index
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        
        if st.button("Check for Potential Biases", type="primary"):
            with st.spinner("Analyzing for potential biases..."):
                try:
                    bias_indicators = {
                        'Gender': ['he', 'she', 'man', 'woman', 'male', 'female', 'boy', 'girl'],
                        'Age': ['young', 'old', 'elderly', 'youth', 'child', 'adult', 'teen'],
                        'Cultural': ['race', 'ethnicity', 'culture', 'tradition', 'minority', 'majority'],
                        'Socioeconomic': ['poor', 'rich', 'wealthy', 'poverty', 'affluent', 'disadvantaged']
                    }
                    
                    words = word_tokenize(text.lower())
                    bias_counts = {}
                    
                    for category, indicators in bias_indicators.items():
                        count = sum(1 for word in words if word in indicators)
                        bias_counts[category] = count
                    
                    st.subheader("Potential Bias Indicators")
                    
                    df = pd.DataFrame(list(bias_counts.items()), columns=['Category', 'Count'])
                    fig = px.bar(df, x='Category', y='Count', title='Bias Indicator Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    ### Recommendations:
                    - ✅ Review text for balanced representation
                    - ✅ Consider multiple perspectives
                    - ✅ Use inclusive language
                    - ✅ Be aware of cultural context
                    - ✅ Ensure fair representation of all groups
                    """)
                    
                except Exception as e:
                    st.error(f"Error analyzing biases: {str(e)}")

def show_history_page():
    """Display analysis history page"""
    st.markdown('<h2 class="section-title">Analysis History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("No files processed yet")
        return
    
    selected_file = st.selectbox(
        "Select a file to view history",
        st.session_state.file_names
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        file_id = st.session_state.file_ids[file_index]
        
        history = db_manager.get_analysis_history(file_id)
        
        if history:
            st.success(f"Found {len(history)} analysis records")
            
            for entry in history:
                with st.expander(f"{entry['analysis_type']} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    if isinstance(entry['results'], dict):
                        for key, value in entry['results'].items():
                            if isinstance(value, str) and len(value) > 500:
                                st.text_area(key, value[:500] + "...", height=150)
                            else:
                                st.write(f"**{key}:** {value}")
                    else:
                        st.write(entry['results'])
        else:
            st.info("No analysis history found for this file")

def show_settings_page():
    """Display settings page with error logs and system controls"""
    st.markdown('<h2 class="section-title">Settings & System Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Statistics")
        st.metric("Files Processed", len(st.session_state.processed_data))
        st.metric("Total Analyses", len(st.session_state.get('analysis_results', [])))
        
        if stability_modules_available:
            memory_manager = safe_session_state_get('memory_manager')
            if memory_manager:
                st.metric("Memory Cache Items", len(memory_manager.get_all()))
    
    with col2:
        st.subheader("🔧 System Controls")
        
        if st.button("Clear Session Data", type="secondary"):
            for key in list(st.session_state.keys()):
                if key not in ['memory_manager', 'health_status']:
                    del st.session_state[key]
            init_session_state()
            st.success("Session data cleared")
            st.rerun()
        
        if st.button("Reconnect Database", type="secondary"):
            db_manager._connect()
            st.success("Database reconnection attempted")
            st.rerun()
    
    # Error log
    st.subheader("📝 Error Log")
    error_log = st.session_state.get('error_log', [])
    
    if error_log:
        st.write(f"Recent errors ({len(error_log)} total)")
        
        # Show last 10 errors
        for error in error_log[-10:]:
            with st.expander(f"Error: {error.get('file', 'Unknown')} - {error.get('timestamp', 'Unknown time')}"):
                st.write(f"**Error:** {error.get('error', 'No details')}")
        
        if st.button("Clear Error Log", type="secondary"):
            st.session_state.error_log = []
            st.success("Error log cleared")
            st.rerun()
    else:
        st.success("No errors logged")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        st.write("**Configuration:**")
        for key, value in config.items():
            if key != 'theme_colors':
                st.write(f"- {key}: {value}")

def show_footer():
    """Display footer"""
    st.markdown("""
    <div style="text-align: center; margin-top: 2em; padding: 1em; background-color: #eaf2f8; border-radius: 10px; border-top: 2px solid #2874a6;">
        <p style="color: #2c3e50; font-weight: bold;">
            © 2024 NLP Tool for YPAR | Unified Version 3.0 | 
            Combining Stability & Advanced Features
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical application failure: {e}")
        logger.critical(traceback.format_exc())
        st.error("Critical error occurred. Please refresh the page.")
    finally:
        # Cleanup
        try:
            db_manager.close()
        except:
            pass