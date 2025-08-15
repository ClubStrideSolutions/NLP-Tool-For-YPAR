"""
Enhanced NLP Tool for YPAR with Complete Fallback Systems
Full-featured text analysis platform with AI and traditional NLP capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
import json
import re
import io
import base64
import random
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# NLTK imports with proper error handling
import nltk
from datetime import datetime
import functools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uuid
import yake
import logging

# Optional imports with fallback
try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import PyPDF2
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PdfReader = None

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    Document = None

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    chardet = None

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    markdown = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
st.set_page_config(
    page_title="NLP Tool for YPAR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS
THEME_CSS = """
<style>
    /* Professional Blue and Gold Theme */
    :root {
        --primary-color: #003262;
        --secondary-color: #FDB515;
        --text-primary: #333333;
        --text-secondary: #666666;
        --background: #FFFFFF;
        --card-background: #F8F9FA;
        --border-color: #E0E0E0;
    }
    
    /* Main container */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    h1 {
        border-bottom: 3px solid var(--secondary-color);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color) 0%, #004080 100%);
    }
    
    .css-1d391kg .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem;
    }
    
    /* Cards */
    .stExpander {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #004080 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #FFB800 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Success/Info/Warning messages */
    .stSuccess {
        background-color: #D4EDDA;
        border-left: 4px solid var(--secondary-color);
        color: var(--primary-color);
    }
    
    .stInfo {
        background-color: #D1ECF1;
        border-left: 4px solid var(--primary-color);
    }
    
    .stWarning {
        background-color: #FFF3CD;
        border-left: 4px solid var(--secondary-color);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    .stFileUpload {
        background: var(--card-background);
        border: 2px dashed var(--primary-color);
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Tables */
    .stDataFrame {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
</style>
"""

# ===================== NLTK INITIALIZATION =====================
def initialize_nltk():
    """Download required NLTK data packages with proper error handling"""
    required_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('tokenizers/punkt/english.pickle', 'punkt'),
    ]
    
    for path, package in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download {package}: {e}")
                # Continue anyway - we have fallbacks

# Initialize NLTK on import
initialize_nltk()

# Safe NLTK imports with fallbacks
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    NLTK_AVAILABLE = False
    stopwords = None
    word_tokenize = lambda x: x.split()
    sent_tokenize = lambda x: x.split('. ')
    ne_chunk = None
    pos_tag = None

# ===================== CONFIGURATION MANAGEMENT =====================
class Config:
    """Centralized configuration management"""
    
    @staticmethod
    def get_openai_api_key():
        """Get OpenAI API key from session state or environment"""
        # Check session state first
        if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
            return st.session_state.openai_api_key
        # Check environment variable
        return os.getenv('OPENAI_API_KEY')
    
    @staticmethod
    def get_mongodb_connection_string():
        """Get MongoDB connection string from session state or environment"""
        # Check session state first
        if 'mongodb_connection' in st.session_state and st.session_state.mongodb_connection:
            return st.session_state.mongodb_connection
        # Check environment variable
        return os.getenv('CONNECTION_STRING')
    
    @staticmethod
    def is_ai_enabled():
        """Check if AI features are enabled"""
        return bool(Config.get_openai_api_key()) and OPENAI_AVAILABLE
    
    @staticmethod
    def is_mongodb_enabled():
        """Check if MongoDB is enabled"""
        return bool(Config.get_mongodb_connection_string()) and PYMONGO_AVAILABLE

# ===================== DATABASE MANAGER =====================
class EnhancedDatabaseManager:
    """Database manager with automatic fallback to session state"""
    
    def __init__(self):
        # Always initialize attributes to avoid AttributeError
        self.client = None
        self.db = None
        self.connected = False
        self.cache = {}
        # Only try to connect if pymongo is available
        if PYMONGO_AVAILABLE:
            try:
                self._connect()
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.connected = False
                self.client = None
                self.db = None
    
    def _connect(self):
        """Establish database connection with proper error handling"""
        try:
            connection_string = Config.get_mongodb_connection_string()
            if connection_string and PYMONGO_AVAILABLE:
                self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                # Test connection
                self.client.server_info()
                self.db = self.client["nlp_tool"]
                self.connected = True
                logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.warning(f"MongoDB connection failed, using session state: {e}")
            self.connected = False
            self.client = None
            self.db = None
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any], 
                      filename: str = None, processing_time: float = 0) -> Optional[str]:
        """Store analysis with automatic fallback to session state"""
        # Always cache
        cache_key = f"{file_id}_{analysis_type}"
        self.cache[cache_key] = results
        
        # Try MongoDB first if connected
        if getattr(self, 'connected', False) and getattr(self, 'db', None):
            try:
                collection = self.db["analysis_results"]
                document = {
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "results": results,
                    "timestamp": datetime.now(),
                    "filename": filename,
                    "processing_time": processing_time
                }
                result = collection.insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"MongoDB storage failed, falling back to session state: {e}")
        
        # Fallback to session state (always works)
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        st.session_state.analysis_results.append({
            "file_id": file_id,
            "analysis_type": analysis_type,
            "results": results,
            "timestamp": datetime.now(),
            "filename": filename,
            "processing_time": processing_time
        })
        return file_id
    
    def get_from_cache(self, file_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get from cache first, then check storage"""
        cache_key = f"{file_id}_{analysis_type}"
        
        # Check memory cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check session state
        if 'analysis_results' in st.session_state:
            for result in st.session_state.analysis_results:
                if result['file_id'] == file_id and result['analysis_type'] == analysis_type:
                    self.cache[cache_key] = result['results']
                    return result['results']
        
        # Try MongoDB if connected
        if getattr(self, 'connected', False) and getattr(self, 'db', None):
            try:
                collection = self.db["analysis_results"]
                result = collection.find_one({
                    "file_id": file_id,
                    "analysis_type": analysis_type
                })
                if result:
                    self.cache[cache_key] = result['results']
                    return result['results']
            except Exception as e:
                logger.error(f"Error retrieving from MongoDB: {e}")
        
        return None
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Get all analyses from storage"""
        results = []
        
        # Get from session state
        if 'analysis_results' in st.session_state:
            results.extend(st.session_state.analysis_results)
        
        # Try MongoDB if connected
        if getattr(self, 'connected', False) and getattr(self, 'db', None):
            try:
                collection = self.db["analysis_results"]
                mongo_results = list(collection.find().sort("timestamp", -1).limit(100))
                # Convert MongoDB results
                for r in mongo_results:
                    r['_id'] = str(r['_id'])
                results.extend(mongo_results)
            except Exception as e:
                logger.error(f"Error retrieving from MongoDB: {e}")
        
        # Remove duplicates based on file_id and analysis_type
        seen = set()
        unique_results = []
        for r in results:
            key = (r.get('file_id'), r.get('analysis_type'))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results

# Initialize database manager
@st.cache_resource
def get_db_manager():
    return EnhancedDatabaseManager()

db_manager = get_db_manager()

# ===================== CONVERSATION MEMORY =====================
class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def add_interaction(self, user_input: str, assistant_response: str):
        """Add an interaction to history"""
        st.session_state.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now()
        })
        # Keep only recent history
        if len(st.session_state.conversation_history) > self.max_history:
            st.session_state.conversation_history = st.session_state.conversation_history[-self.max_history:]
    
    def get_context(self, n_recent: int = 3) -> str:
        """Get recent conversation context"""
        if not st.session_state.conversation_history:
            return ""
        
        recent = st.session_state.conversation_history[-n_recent:]
        context = []
        for interaction in recent:
            context.append(f"User: {interaction['user']}")
            context.append(f"Assistant: {interaction['assistant']}")
        
        return "\n".join(context)
    
    def clear(self):
        """Clear conversation history"""
        st.session_state.conversation_history = []

# ===================== TEXT ANALYZER =====================
class TextAnalyzer:
    """Unified text analysis with AI and traditional fallbacks"""
    
    def __init__(self):
        self.ai_enabled = Config.is_ai_enabled()
        self.memory = ConversationMemory()
        
        # Initialize OpenAI if available
        if self.ai_enabled and OPENAI_AVAILABLE:
            try:
                openai.api_key = Config.get_openai_api_key()
                self.client = openai.OpenAI(api_key=Config.get_openai_api_key())
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
                self.ai_enabled = False
                self.client = None
        else:
            self.client = None
    
    def _ai_analyze(self, text: str, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """AI-powered analysis using OpenAI"""
        if not self.ai_enabled or not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert text analyst."},
                    {"role": "user", "content": f"{prompt}\n\nText:\n{text[:3000]}"}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None
    
    def analyze_sentiment(self, text: str, file_id: str) -> str:
        """Analyze sentiment with AI or traditional methods"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "sentiment")
        if cached:
            return cached.get('sentiment', 'Unknown')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Analyze the sentiment of this text. Provide: 1) Overall sentiment (positive/negative/neutral), 2) Confidence score, 3) Brief emotional context."
            )
            if result:
                db_manager.store_analysis(file_id, "sentiment", {"sentiment": result})
                return result
        
        # Fallback to TextBlob
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            result = f"**Sentiment:** {sentiment}\n**Polarity:** {polarity:.2f}\n**Subjectivity:** {subjectivity:.2f}"
            
            # Add emotional context
            emotions = []
            if polarity > 0.5:
                emotions.append("very positive")
            elif polarity > 0.2:
                emotions.append("moderately positive")
            if subjectivity > 0.5:
                emotions.append("subjective")
            else:
                emotions.append("objective")
            
            if emotions:
                result += f"\n**Tone:** {', '.join(emotions)}"
            
            db_manager.store_analysis(file_id, "sentiment", {"sentiment": result})
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return "Sentiment analysis unavailable"
    
    def extract_themes(self, text: str, file_id: str) -> str:
        """Extract themes with AI or LDA"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "themes")
        if cached:
            return cached.get('themes', '')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract the top 5 main themes from this text. Format: **Theme Name**: Brief description"
            )
            if result:
                db_manager.store_analysis(file_id, "themes", {"themes": result})
                return result
        
        # Fallback to LDA
        try:
            # Prepare text
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            # Use TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(sentences[:100])  # Limit for performance
            
            # Apply LDA
            lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10)
            lda.fit(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract themes
            themes = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                themes.append(f"**Theme {topic_idx + 1}**: {', '.join(top_words[:3])}")
            
            result = "\n".join(themes)
            db_manager.store_analysis(file_id, "themes", {"themes": result})
            return result
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return "Theme extraction unavailable"
    
    def extract_keywords(self, text: str, file_id: str) -> str:
        """Extract keywords with AI or YAKE"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "keywords")
        if cached:
            return cached.get('keywords', '')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract the top 10 most important keywords or key phrases from this text. List them in order of importance."
            )
            if result:
                db_manager.store_analysis(file_id, "keywords", {"keywords": result})
                return result
        
        # Fallback to YAKE
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=2,  # max ngram size
                dedupLim=0.7,
                top=10
            )
            keywords = kw_extractor.extract_keywords(text)
            
            # Format results
            result = []
            for kw, score in keywords:
                # Lower score = more important in YAKE
                importance = "high" if score < 0.05 else "medium" if score < 0.1 else "normal"
                result.append(f"{kw} ({importance})")
            
            formatted = "\n".join(result)
            db_manager.store_analysis(file_id, "keywords", {"keywords": formatted})
            return formatted
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # Fallback to simple frequency-based extraction
            try:
                words = word_tokenize(text.lower()) if NLTK_AVAILABLE else text.lower().split()
                # Filter stopwords
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'])
                words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
                
                # Get frequency
                word_freq = Counter(words)
                top_words = word_freq.most_common(10)
                
                result = "\n".join([f"{word} (count: {count})" for word, count in top_words])
                db_manager.store_analysis(file_id, "keywords", {"keywords": result})
                return result
            except:
                return "Keyword extraction unavailable"
    
    def extract_entities(self, text: str, file_id: str) -> str:
        """Extract named entities"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "entities")
        if cached:
            return cached.get('entities', '')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract all named entities (people, organizations, locations, dates) from this text. Group by type."
            )
            if result:
                db_manager.store_analysis(file_id, "entities", {"entities": result})
                return result
        
        # Fallback to NLTK NER
        if NLTK_AVAILABLE and ne_chunk and pos_tag:
            try:
                sentences = sent_tokenize(text)[:10]  # Limit for performance
                entities = defaultdict(list)
                
                for sent in sentences:
                    words = word_tokenize(sent)
                    pos_tags = pos_tag(words)
                    chunks = ne_chunk(pos_tags, binary=False)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_name = ' '.join(c[0] for c in chunk)
                            entity_type = chunk.label()
                            entities[entity_type].append(entity_name)
                
                # Format results
                result = []
                for entity_type, names in entities.items():
                    unique_names = list(set(names))[:5]
                    if unique_names:
                        result.append(f"**{entity_type}**: {', '.join(unique_names)}")
                
                formatted = "\n".join(result) if result else "No entities found"
                db_manager.store_analysis(file_id, "entities", {"entities": formatted})
                return formatted
                
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
        
        # Simple fallback - look for capitalized words
        try:
            words = text.split()
            capitalized = [w for w in words if w[0].isupper() and len(w) > 2]
            unique_caps = list(set(capitalized))[:20]
            result = f"**Possible entities**: {', '.join(unique_caps)}"
            db_manager.store_analysis(file_id, "entities", {"entities": result})
            return result
        except:
            return "Entity extraction unavailable"
    
    def extract_quotes(self, text: str, file_id: str) -> str:
        """Extract meaningful quotes"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "quotes")
        if cached:
            return cached.get('quotes', '')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract 3-5 most impactful or meaningful quotes from this text. Include context if needed."
            )
            if result:
                db_manager.store_analysis(file_id, "quotes", {"quotes": result})
                return result
        
        # Fallback to pattern matching
        try:
            # Look for quoted text
            quotes = re.findall(r'"([^"]+)"', text)
            
            # If no quotes, extract impactful sentences
            if not quotes:
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
                # Filter for sentences with strong words
                strong_words = ['important', 'critical', 'essential', 'must', 'should', 'believe', 'think', 'feel', 'know']
                quotes = [s for s in sentences if any(w in s.lower() for w in strong_words)][:5]
            
            # Format results
            if quotes:
                result = "\n\n".join([f'"{q}"' if not q.startswith('"') else q for q in quotes[:5]])
            else:
                # Get first few sentences as sample
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
                result = "\n\n".join([f'"{s}"' for s in sentences[:3]])
            
            db_manager.store_analysis(file_id, "quotes", {"quotes": result})
            return result
            
        except Exception as e:
            logger.error(f"Quote extraction failed: {e}")
            return "Quote extraction unavailable"
    
    def generate_insights(self, text: str, file_id: str) -> str:
        """Generate research insights"""
        # Check cache
        cached = db_manager.get_from_cache(file_id, "insights")
        if cached:
            return cached.get('insights', '')
        
        # Try AI first
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                """Generate 3-5 key research insights from this text. Focus on:
                1. Main findings or arguments
                2. Patterns or trends
                3. Implications or recommendations
                Format each insight as a bullet point.""",
                max_tokens=600
            )
            if result:
                db_manager.store_analysis(file_id, "insights", {"insights": result})
                return result
        
        # Fallback to statistical insights
        try:
            # Basic text statistics
            words = word_tokenize(text) if NLTK_AVAILABLE else text.split()
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / max(sentence_count, 1)
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            
            # Identify patterns
            insights = []
            
            # Length insight
            if word_count > 1000:
                insights.append(f"• **Comprehensive text**: {word_count:,} words across {sentence_count} sentences")
            else:
                insights.append(f"• **Concise text**: {word_count} words in {sentence_count} sentences")
            
            # Complexity insight
            if avg_sentence_length > 20:
                insights.append(f"• **Complex writing style**: Average sentence length of {avg_sentence_length:.1f} words suggests detailed exposition")
            else:
                insights.append(f"• **Clear writing style**: Average sentence length of {avg_sentence_length:.1f} words suggests accessibility")
            
            # Diversity insight
            if lexical_diversity > 0.5:
                insights.append(f"• **Rich vocabulary**: {lexical_diversity:.1%} lexical diversity indicates varied language use")
            else:
                insights.append(f"• **Focused vocabulary**: {lexical_diversity:.1%} lexical diversity suggests consistent terminology")
            
            # Sentiment pattern
            try:
                blob = TextBlob(text)
                if blob.sentiment.polarity > 0.2:
                    insights.append("• **Positive tone**: Text exhibits generally optimistic or favorable language")
                elif blob.sentiment.polarity < -0.2:
                    insights.append("• **Critical tone**: Text contains skeptical or negative assessments")
                else:
                    insights.append("• **Balanced tone**: Text maintains neutral or objective perspective")
            except:
                pass
            
            result = "\n".join(insights)
            db_manager.store_analysis(file_id, "insights", {"insights": result})
            return result
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return "Insight generation unavailable"
    
    def answer_question(self, text: str, question: str, file_id: str) -> str:
        """Answer questions about the text"""
        # Try AI first
        if self.ai_enabled:
            # Add conversation context
            context = self.memory.get_context()
            
            prompt = f"""Based on the following text, answer this question: {question}

Previous context:
{context}

Text:
{text[:2000]}

Provide a clear, concise answer based only on the information in the text."""
            
            result = self._ai_analyze(text, prompt, max_tokens=300)
            if result:
                self.memory.add_interaction(question, result)
                return result
        
        # Fallback to keyword matching
        try:
            question_lower = question.lower()
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            # Extract question keywords
            question_words = set(word_tokenize(question_lower) if NLTK_AVAILABLE else question_lower.split())
            stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
            keywords = question_words - stop_words
            
            # Find relevant sentences
            relevant = []
            for sent in sentences:
                sent_lower = sent.lower()
                score = sum(1 for kw in keywords if kw in sent_lower)
                if score > 0:
                    relevant.append((score, sent))
            
            # Sort by relevance
            relevant.sort(reverse=True)
            
            if relevant:
                # Return top 3 most relevant sentences
                answer = "\n\n".join([sent for _, sent in relevant[:3]])
                self.memory.add_interaction(question, answer)
                return answer
            else:
                return "I couldn't find specific information about that in the text."
                
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return "Unable to process question at this time."

# ===================== FILE HANDLERS =====================
class FileHandler:
    """Handle various file formats"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (better for tables)
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                file.seek(0)
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
        
        return "PDF reading not available. Please install pdfplumber or PyPDF2."
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            return "DOCX reading not available. Please install python-docx."
        
        try:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return f"Error reading DOCX file: {str(e)}"
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            # Try to detect encoding
            if CHARDET_AVAILABLE:
                raw = file.read()
                result = chardet.detect(raw)
                encoding = result['encoding'] or 'utf-8'
                file.seek(0)
                text = file.read().decode(encoding, errors='ignore')
            else:
                # Default to UTF-8
                text = file.read().decode('utf-8', errors='ignore')
            
            return text
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
            # Try with different encoding
            try:
                file.seek(0)
                text = file.read().decode('latin-1', errors='ignore')
                return text
            except:
                return f"Error reading text file: {str(e)}"
    
    @staticmethod
    def extract_text_from_md(file) -> str:
        """Extract text from Markdown file"""
        try:
            text = file.read().decode('utf-8', errors='ignore')
            
            # Convert markdown to plain text if available
            if MARKDOWN_AVAILABLE:
                import markdown
                from bs4 import BeautifulSoup
                html = markdown.markdown(text)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
            
            return text
        except Exception as e:
            logger.error(f"Error reading Markdown: {e}")
            return f"Error reading Markdown file: {str(e)}"
    
    @staticmethod
    def process_file(uploaded_file) -> Tuple[str, str]:
        """Process uploaded file and return text content and file ID"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Generate file ID
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        file_id = hashlib.md5(file_content).hexdigest()
        
        # Extract text based on file type
        if file_extension == 'pdf':
            text = FileHandler.extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = FileHandler.extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = FileHandler.extract_text_from_txt(uploaded_file)
        elif file_extension == 'md':
            text = FileHandler.extract_text_from_md(uploaded_file)
        else:
            text = "Unsupported file format"
        
        return text, file_id

# ===================== VISUALIZATION TOOLS =====================
class Visualizer:
    """Create various visualizations"""
    
    @staticmethod
    def create_wordcloud(text: str) -> go.Figure:
        """Create word cloud visualization"""
        try:
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Blues',
                max_words=50
            ).generate(text)
            
            # Convert to plotly
            fig = go.Figure()
            fig.add_trace(go.Image(z=wordcloud.to_array()))
            fig.update_layout(
                title="Word Cloud",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=400
            )
            return fig
        except Exception as e:
            logger.error(f"Wordcloud generation failed: {e}")
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(text="Wordcloud generation failed", x=0.5, y=0.5)
            return fig
    
    @staticmethod
    def create_sentiment_chart(sentiments: List[float]) -> go.Figure:
        """Create sentiment trend chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sentiments,
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#003262', width=2),
            marker=dict(color='#FDB515', size=8)
        ))
        fig.update_layout(
            title="Sentiment Trend",
            xaxis_title="Segment",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            height=350
        )
        return fig
    
    @staticmethod
    def create_theme_network(themes: List[str], connections: List[Tuple[str, str]]) -> go.Figure:
        """Create theme network visualization"""
        try:
            # Create network graph
            G = nx.Graph()
            G.add_nodes_from(themes)
            G.add_edges_from(connections)
            
            # Get positions
            pos = nx.spring_layout(G)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=20,
                    color='#003262',
                    line=dict(color='#FDB515', width=2)
                ),
                hoverinfo='text'
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                title="Theme Network",
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=400
            )
            return fig
        except Exception as e:
            logger.error(f"Network visualization failed: {e}")
            fig = go.Figure()
            fig.add_annotation(text="Network visualization unavailable", x=0.5, y=0.5)
            return fig

# ===================== MAIN APPLICATION =====================
def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

def show_home():
    """Display home page"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    st.title("🔬 NLP Tool for Youth Participatory Action Research")
    st.markdown("### Empowering Youth Researchers with Advanced Text Analysis")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ai_status = "🟢 Enabled" if Config.is_ai_enabled() else "🔴 Disabled"
        st.metric("AI Analysis", ai_status)
        if not Config.is_ai_enabled():
            st.caption("Using traditional NLP")
    
    with col2:
        db_status = "🟢 Connected" if Config.is_mongodb_enabled() else "🟡 Session Storage"
        st.metric("Database", db_status)
        if not Config.is_mongodb_enabled():
            st.caption("Using local storage")
    
    with col3:
        file_count = len(st.session_state.uploaded_files)
        st.metric("Files Loaded", file_count)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Getting Started:**
        1. 📁 Upload your documents (PDF, DOCX, TXT, MD)
        2. 📊 Run comprehensive analysis
        3. 🎯 Extract themes and insights
        4. 💾 Export results
        
        **Available Analyses:**
        - Sentiment Analysis
        - Theme Extraction
        - Keyword Identification
        - Named Entity Recognition
        - Quote Extraction
        - Research Insights
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - ✨ Dual-mode analysis (AI + Traditional)
        - 🔄 Automatic fallback systems
        - 📈 Rich visualizations
        - 💬 Interactive Q&A
        - 📊 Batch processing
        - 🔒 Secure data handling
        
        **File Formats:**
        - PDF Documents
        - Word Documents (.docx)
        - Text Files (.txt)
        - Markdown Files (.md)
        """)
    
    # System health
    if PSUTIL_AVAILABLE:
        st.markdown("### 📊 System Health")
        col1, col2, col3 = st.columns(3)
        
        try:
            import psutil
            with col1:
                cpu_percent = psutil.cpu_percent(interval=1)
                st.progress(cpu_percent / 100)
                st.caption(f"CPU: {cpu_percent}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.progress(memory.percent / 100)
                st.caption(f"Memory: {memory.percent}%")
            
            with col3:
                st.info(f"Cache Size: {len(db_manager.cache)} items")
        except:
            pass

def show_upload():
    """Display file upload page"""
    st.title("📁 Upload Documents")
    
    st.markdown("""
    Upload your documents for analysis. Supported formats:
    - **PDF** - Research papers, reports
    - **DOCX** - Word documents
    - **TXT** - Plain text files
    - **MD** - Markdown files
    """)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            try:
                # Process file
                text, file_id = FileHandler.process_file(uploaded_file)
                
                # Store in session
                st.session_state.uploaded_files[file_id] = {
                    'name': uploaded_file.name,
                    'text': text,
                    'size': len(text),
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ {uploaded_file.name} processed successfully")
                
                # Show preview
                with st.expander(f"Preview: {uploaded_file.name}"):
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        status_text.text("Processing complete!")
        progress_bar.progress(1.0)
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### 📚 Uploaded Files")
        
        for file_id, file_info in st.session_state.uploaded_files.items():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(file_info['name'])
            with col2:
                st.text(f"{file_info['size']:,} chars")
            with col3:
                st.text(file_info['timestamp'].strftime("%H:%M"))
            with col4:
                if st.button("Remove", key=f"remove_{file_id}"):
                    del st.session_state.uploaded_files[file_id]
                    st.rerun()

def show_analysis():
    """Display text analysis page"""
    st.title("📊 Text Analysis")
    
    if not st.session_state.uploaded_files:
        st.warning("Please upload documents first")
        return
    
    # File selection
    file_options = {
        file_id: info['name'] 
        for file_id, info in st.session_state.uploaded_files.items()
    }
    
    selected_file_id = st.selectbox(
        "Select document to analyze",
        options=list(file_options.keys()),
        format_func=lambda x: file_options[x]
    )
    
    if selected_file_id:
        file_info = st.session_state.uploaded_files[selected_file_id]
        text = file_info['text']
        
        st.info(f"📄 Analyzing: {file_info['name']} ({len(text):,} characters)")
        
        # Initialize analyzer
        text_analyzer = TextAnalyzer()
        
        # Analysis options
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive analysis..."):
                    # Run all analyses
                    progress = st.progress(0)
                    
                    # 1. Sentiment Analysis
                    progress.progress(0.15)
                    sentiment = text_analyzer.analyze_sentiment(text, selected_file_id)
                    
                    # 2. Theme Extraction
                    progress.progress(0.30)
                    themes = text_analyzer.extract_themes(text, selected_file_id)
                    
                    # 3. Keyword Extraction
                    progress.progress(0.45)
                    keywords = text_analyzer.extract_keywords(text, selected_file_id)
                    
                    # 4. Entity Recognition
                    progress.progress(0.60)
                    entities = text_analyzer.extract_entities(text, selected_file_id)
                    
                    # 5. Quote Extraction
                    progress.progress(0.75)
                    quotes = text_analyzer.extract_quotes(text, selected_file_id)
                    
                    # 6. Research Insights
                    progress.progress(0.90)
                    insights = text_analyzer.generate_insights(text, selected_file_id)
                    
                    progress.progress(1.0)
                    
                    # Display results in columns
                    st.markdown("### 📋 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 😊 Sentiment")
                        st.markdown(sentiment)
                        
                        st.markdown("#### 🏷️ Entities")
                        st.markdown(entities)
                    
                    with col2:
                        st.markdown("#### 🎯 Themes")
                        st.markdown(themes)
                        
                        st.markdown("#### 💬 Key Quotes")
                        st.markdown(quotes)
                    
                    with col3:
                        st.markdown("#### 🔑 Keywords")
                        st.markdown(keywords)
                        
                        st.markdown("#### 💡 Insights")
                        st.markdown(insights)
                    
                    st.success("✅ Analysis complete!")
        
        with col2:
            # Individual analyses
            st.markdown("**Run Individual Analysis:**")
            
            if st.button("😊 Sentiment", use_container_width=True):
                with st.spinner("Analyzing sentiment..."):
                    result = text_analyzer.analyze_sentiment(text, selected_file_id)
                    st.markdown("### Sentiment Analysis")
                    st.markdown(result)
            
            if st.button("🎯 Themes", use_container_width=True):
                with st.spinner("Extracting themes..."):
                    result = text_analyzer.extract_themes(text, selected_file_id)
                    st.markdown("### Theme Extraction")
                    st.markdown(result)
            
            if st.button("🔑 Keywords", use_container_width=True):
                with st.spinner("Extracting keywords..."):
                    result = text_analyzer.extract_keywords(text, selected_file_id)
                    st.markdown("### Keyword Extraction")
                    st.markdown(result)
            
            if st.button("🏷️ Entities", use_container_width=True):
                with st.spinner("Recognizing entities..."):
                    result = text_analyzer.extract_entities(text, selected_file_id)
                    st.markdown("### Named Entities")
                    st.markdown(result)
            
            if st.button("💬 Quotes", use_container_width=True):
                with st.spinner("Extracting quotes..."):
                    result = text_analyzer.extract_quotes(text, selected_file_id)
                    st.markdown("### Key Quotes")
                    st.markdown(result)
            
            if st.button("💡 Insights", use_container_width=True):
                with st.spinner("Generating insights..."):
                    result = text_analyzer.generate_insights(text, selected_file_id)
                    st.markdown("### Research Insights")
                    st.markdown(result)
        
        # Interactive Q&A
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About This Document")
        
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Finding answer..."):
                answer = text_analyzer.answer_question(text, question, selected_file_id)
                st.markdown("**Answer:**")
                st.markdown(answer)

def show_visualizations():
    """Display visualizations page"""
    st.title("📈 Visualizations")
    
    if not st.session_state.uploaded_files:
        st.warning("Please upload and analyze documents first")
        return
    
    # File selection
    file_options = {
        file_id: info['name'] 
        for file_id, info in st.session_state.uploaded_files.items()
    }
    
    selected_file_id = st.selectbox(
        "Select document",
        options=list(file_options.keys()),
        format_func=lambda x: file_options[x]
    )
    
    if selected_file_id:
        file_info = st.session_state.uploaded_files[selected_file_id]
        text = file_info['text']
        
        st.info(f"📊 Visualizing: {file_info['name']}")
        
        visualizer = Visualizer()
        
        # Word Cloud
        st.markdown("### ☁️ Word Cloud")
        try:
            fig = visualizer.create_wordcloud(text)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create word cloud: {e}")
        
        # Sentiment Trend
        st.markdown("### 📈 Sentiment Trend")
        try:
            # Split text into segments
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            # Analyze sentiment for each segment
            sentiments = []
            for i in range(0, min(len(sentences), 20), 2):
                segment = ' '.join(sentences[i:i+2])
                try:
                    blob = TextBlob(segment)
                    sentiments.append(blob.sentiment.polarity)
                except:
                    sentiments.append(0)
            
            if sentiments:
                fig = visualizer.create_sentiment_chart(sentiments)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create sentiment chart: {e}")
        
        # Theme Network
        st.markdown("### 🕸️ Theme Network")
        try:
            # Extract themes for network
            text_analyzer = TextAnalyzer()
            themes_text = text_analyzer.extract_themes(text, selected_file_id)
            
            # Parse themes
            theme_lines = themes_text.split('\n')
            themes = []
            for line in theme_lines:
                if '**' in line:
                    theme = line.split('**')[1] if len(line.split('**')) > 1 else line
                    themes.append(theme.strip())
            
            if themes:
                # Create connections (simple example)
                connections = [(themes[i], themes[i+1]) for i in range(len(themes)-1)]
                
                fig = visualizer.create_theme_network(themes[:5], connections)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create theme network: {e}")

def show_history():
    """Display analysis history"""
    st.title("📜 Analysis History")
    
    # Get all analyses
    analyses = db_manager.get_all_analyses()
    
    if not analyses:
        st.info("No analysis history available yet")
        return
    
    # Create DataFrame
    df_data = []
    for analysis in analyses:
        df_data.append({
            'Timestamp': analysis.get('timestamp', 'N/A'),
            'File': analysis.get('filename', 'Unknown'),
            'Type': analysis.get('analysis_type', 'Unknown'),
            'File ID': analysis.get('file_id', 'N/A')[:8] + '...'
        })
    
    df = pd.DataFrame(df_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Export option
    if st.button("📥 Export History as CSV"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_history.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

def show_settings():
    """Display settings page"""
    st.title("⚙️ Settings")
    
    st.markdown("Configure your analysis environment")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    with st.form("api_settings"):
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Enter your OpenAI API key for AI-powered analysis"
        )
        
        # MongoDB Connection
        mongo_connection = st.text_input(
            "MongoDB Connection String",
            value="",
            type="password",
            help="Enter your MongoDB connection string for persistent storage"
        )
        
        if st.form_submit_button("Save Settings"):
            # Save to session state
            if openai_key:
                st.session_state.openai_api_key = openai_key
                st.success("✅ OpenAI API key saved")
            
            if mongo_connection:
                st.session_state.mongodb_connection = mongo_connection
                # Reinitialize database
                global db_manager
                db_manager = get_db_manager()
                st.success("✅ MongoDB connection saved")
            
            st.rerun()
    
    # Current Status
    st.markdown("### 📊 Current Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**OpenAI Status:**")
        if Config.is_ai_enabled():
            st.success("✅ Connected - AI features enabled")
        else:
            st.warning("⚠️ Not configured - Using traditional NLP")
    
    with col2:
        st.markdown("**MongoDB Status:**")
        if Config.is_mongodb_enabled():
            st.success("✅ Connected - Persistent storage enabled")
        else:
            st.info("ℹ️ Using session storage")
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("""
        **Environment Variables:**
        - `OPENAI_API_KEY` - OpenAI API key
        - `CONNECTION_STRING` - MongoDB connection string
        
        **Fallback Modes:**
        - Traditional NLP automatically activates when AI is unavailable
        - Session storage automatically activates when MongoDB is unavailable
        
        **Cache Management:**
        - Analysis results are cached in memory for performance
        - Cache persists during session
        """)
        
        if st.button("Clear Cache"):
            db_manager.cache.clear()
            st.success("Cache cleared")
    
    # System Info
    st.markdown("### 💻 System Information")
    
    info_data = {
        "Python Version": "3.8+",
        "Streamlit Version": st.__version__,
        "AI Available": "Yes" if OPENAI_AVAILABLE else "No",
        "MongoDB Available": "Yes" if PYMONGO_AVAILABLE else "No",
        "PDF Support": "Yes" if (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE) else "No",
        "DOCX Support": "Yes" if DOCX_AVAILABLE else "No",
        "NLTK Available": "Yes" if NLTK_AVAILABLE else "Limited"
    }
    
    for key, value in info_data.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text(key)
        with col2:
            st.text(value)

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Apply theme
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🔬 YPAR Tool")
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Home": "Home",
            "📁 Upload Data": "Upload",
            "📊 Text Analysis": "Analysis",
            "📈 Visualizations": "Visualizations",
            "📜 History": "History",
            "⚙️ Settings": "Settings"
        }
        
        for label, page in pages.items():
            if st.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Files", len(st.session_state.uploaded_files))
        st.metric("Analyses", len(st.session_state.get('analysis_results', [])))
        
        # Mode indicator
        st.markdown("---")
        mode = "AI Enhanced" if Config.is_ai_enabled() else "Traditional NLP"
        st.info(f"Mode: {mode}")
    
    # Main content area
    if st.session_state.page == "Home":
        show_home()
    elif st.session_state.page == "Upload":
        show_upload()
    elif st.session_state.page == "Analysis":
        show_analysis()
    elif st.session_state.page == "Visualizations":
        show_visualizations()
    elif st.session_state.page == "History":
        show_history()
    elif st.session_state.page == "Settings":
        show_settings()

if __name__ == "__main__":
    main()