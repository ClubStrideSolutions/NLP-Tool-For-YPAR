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
    from ui_components import UIComponents
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

# RAG system will be defined inline to avoid import issues
rag_available = True

# Stability modules not available, using fallback implementations
stability_available = False

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

# Apply enhanced UI theme - method not available, skipping
# if ui_available:
#     UIComponents.apply_berkeley_theme()

# Apply theme colors
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Content container */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Headers with gold color */
    h1, h2, h3 {
        color: #003262 !important;
        font-weight: 600;
    }
    
    /* Primary buttons with blue background */
    .stButton > button {
        background: linear-gradient(135deg, #003262 0%, #004d8a 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #004d8a 0%, #003262 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 50, 98, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003262 0%, #004d8a 100%);
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #003262 0%, #004d8a 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #FDB515;
    }
    
    [data-testid="metric-container"] label {
        color: #FDB515 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 5px;
        border-left: 4px solid #FDB515;
    }
    
    /* Text input fields */
    .stTextInput > div > div > input {
        border: 2px solid #003262;
        border-radius: 5px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FDB515;
        box-shadow: 0 0 0 2px rgba(253, 181, 21, 0.2);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(0, 50, 98, 0.05), rgba(253, 181, 21, 0.05));
        border: 2px dashed #003262;
        border-radius: 8px;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #FDB515;
        background: linear-gradient(135deg, rgba(253, 181, 21, 0.1), rgba(0, 50, 98, 0.1));
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize NLTK with proper error handling
def initialize_nltk():
    """Download required NLTK data packages"""
    required_packages = [
        'punkt',  # For sentence tokenization
        'punkt_tab',  # New punkt format
        'stopwords',  # For stopword removal
        'vader_lexicon',  # For sentiment analysis
        'maxent_ne_chunker',  # For named entity recognition
        'words',  # Word corpus
        'averaged_perceptron_tagger'  # For POS tagging
    ]
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                # Try alternative download method
                try:
                    nltk.download(package.replace('_tab', ''), quiet=True)
                except:
                    pass

# Run NLTK initialization
try:
    initialize_nltk()
except Exception as e:
    logger.warning(f"NLTK initialization warning: {e}")

# Configuration
class Config:
    """Application configuration"""
    
    # Theme Colors
    BERKELEY_BLUE = "#003262"
    CALIFORNIA_GOLD = "#FDB515"
    FOUNDERS_ROCK = "#3B7EA1"
    MEDALIST = "#C4820E"
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB - reduced for better performance
    # Only support file types we can reliably process and display
    SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'md']  # Text, PDF, Word, Markdown only
    BATCH_SIZE = 5
    CACHE_TTL = 3600
    
    @staticmethod
    def get_mongodb_connection_string():
        """Get MongoDB connection string from session state, secrets, or environment"""
        try:
            # First check session state (user-provided in settings)
            if 'mongodb_connection_string' in st.session_state:
                return st.session_state.get('mongodb_connection_string')
            # Then check secrets file
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
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any], 
                      filename: str = None, processing_time: float = 0) -> Optional[str]:
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
                    "timestamp": datetime.now(),
                    "filename": filename,
                    "processing_time": processing_time
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
            "timestamp": datetime.now(),
            "filename": filename,
            "processing_time": processing_time
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

# MongoDB-based RAG System Classes (moved here to fix initialization order)
class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add_exchange(self, query: str, response: str):
        """Add a query-response pair to history"""
        self.history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now()
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.history
    
    def clear(self):
        """Clear conversation history"""
        self.history = []

class PersonaManager:
    """Manage different analysis personas"""
    
    def __init__(self):
        self.personas = self._initialize_personas()
        self.active_persona = None
    
    def _initialize_personas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default personas"""
        return {
            "researcher": {
                "name": "Academic Researcher",
                "description": "Focused on rigorous analysis and evidence-based insights",
                "focus_areas": ["methodology", "validity", "reliability", "theoretical frameworks"],
                "analysis_style": "Systematic, thorough, citation-oriented",
                "output_format": "Academic paper style with references",
                "key_questions": [
                    "What are the theoretical implications?",
                    "How does this relate to existing literature?",
                    "What are the methodological considerations?"
                ]
            },
            "educator": {
                "name": "Youth Educator",
                "description": "Focused on learning outcomes and pedagogical applications",
                "focus_areas": ["learning objectives", "engagement", "accessibility", "scaffolding"],
                "analysis_style": "Clear, instructional, example-rich",
                "output_format": "Lesson plan format with activities",
                "key_questions": [
                    "How can this be taught effectively?",
                    "What are the learning outcomes?",
                    "How can we engage youth with this content?"
                ]
            },
            "youth_advocate": {
                "name": "Youth Advocate",
                "description": "Centered on youth voice, empowerment, and social justice",
                "focus_areas": ["youth voice", "empowerment", "equity", "action", "community"],
                "analysis_style": "Empowering, action-oriented, inclusive",
                "output_format": "Action plan with youth perspectives",
                "key_questions": [
                    "How does this empower youth?",
                    "What actions can be taken?",
                    "Whose voices are represented?"
                ]
            },
            "data_analyst": {
                "name": "Data Scientist",
                "description": "Focused on patterns, statistics, and quantitative insights",
                "focus_areas": ["patterns", "statistics", "trends", "correlations", "predictions"],
                "analysis_style": "Quantitative, precise, visual",
                "output_format": "Statistical report with visualizations",
                "key_questions": [
                    "What patterns emerge from the data?",
                    "What are the statistical significance?",
                    "How can we visualize these insights?"
                ]
            },
            "policy_maker": {
                "name": "Policy Advisor",
                "description": "Focused on policy implications and recommendations",
                "focus_areas": ["policy", "implementation", "stakeholders", "outcomes", "evaluation"],
                "analysis_style": "Strategic, pragmatic, evidence-based",
                "output_format": "Policy brief with recommendations",
                "key_questions": [
                    "What are the policy implications?",
                    "Who are the stakeholders?",
                    "How can this be implemented?"
                ]
            }
        }
    
    def set_active_persona(self, persona_key: str):
        """Set the active persona for analysis"""
        if persona_key in self.personas:
            self.active_persona = self.personas[persona_key]
            return True
        return False
    
    def get_active_persona(self) -> Optional[Dict[str, Any]]:
        """Get the currently active persona"""
        return self.active_persona
    
    def to_prompt_context(self) -> str:
        """Convert active persona to prompt context"""
        if not self.active_persona:
            return ""
        p = self.active_persona
        return f"""
        You are acting as a {p['name']}.
        Description: {p['description']}
        Focus Areas: {', '.join(p['focus_areas'])}
        Analysis Style: {p['analysis_style']}
        Output Format: {p['output_format']}
        Key Questions to Consider: {'; '.join(p['key_questions'])}
        """

class RAGSystem:
    """MongoDB-based RAG System for document retrieval and analysis"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or db_manager
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.documents = []
        self.document_vectors = None
        self.document_store = pd.DataFrame()
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add document to RAG system"""
        doc_id = str(uuid.uuid4())[:8]
        doc = {
            'id': doc_id,
            'text': text[:10000],  # Limit text length
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        
        # Store in MongoDB if available
        if self.db_manager and self.db_manager.connected:
            try:
                self.db_manager.db["rag_documents"].insert_one(doc)
            except Exception as e:
                logger.error(f"Error storing document in MongoDB: {e}")
        
        # Also store locally for quick access
        self.documents.append(doc)
        
        # Update vectors
        self._update_vectors()
        
        return doc_id
    
    def _update_vectors(self):
        """Update TF-IDF vectors for all documents"""
        if not self.documents:
            return
        
        texts = [doc['text'] for doc in self.documents]
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
            # Update document store dataframe
            self.document_store = pd.DataFrame(self.documents)
        except Exception as e:
            logger.error(f"Error updating vectors: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query documents using TF-IDF similarity"""
        if not self.documents or self.document_vectors is None:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top k documents
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx]['text'],
                        'metadata': self.documents[idx]['metadata'],
                        'score': float(similarities[idx]),
                        'id': self.documents[idx]['id']
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return []
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response based on query and retrieved context"""
        if not context:
            return "No relevant documents found for your query."
        
        # Combine context
        combined_context = "\n\n".join([f"Document {i+1}: {doc['text'][:500]}..." 
                                        for i, doc in enumerate(context[:3])])
        
        # Create response
        response = f"""Based on the analysis of {len(context)} relevant documents:
        
        **Key Findings:**
        {combined_context}
        
        **Summary:**
        The documents contain information relevant to your query about '{query[:100]}'.
        The top matching document has a relevance score of {context[0]['score']:.2f}.
        """
        
        return response

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
        
        # Initialize OpenAI analyzer if available
        try:
            from openai_analyzer import OpenAIAnalyzer
            self.ai_analyzer = OpenAIAnalyzer()
            self.use_ai = self.ai_analyzer.is_available()
            if self.use_ai:
                logger.info("OpenAI analyzer initialized - using AI for analysis")
                st.success("🤖 OpenAI connected - Using AI-powered analysis")
        except Exception as e:
            self.ai_analyzer = None
            self.use_ai = False
            logger.info(f"OpenAI not available, using traditional NLP: {e}")
        
        # Initialize stopwords with proper error handling
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback to basic stopwords if NLTK fails
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                              'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                              'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                              'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                              'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                              'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                              'own', 'same', 'so', 'than', 'too', 'very', 'just'}
    
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
                
                # Add to RAG knowledge base
                _self.rag_system.add_document(text, {"file_id": file_id, "persona": persona_name})
                
                # Generate analysis with persona context
                query = f"""
                {_self.persona_manager.to_prompt_context()}
                
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
        """Enhanced theme analysis with better error handling"""
        try:
            # Preprocess
            words = word_tokenize(text.lower())
            filtered = [w for w in words if w.isalnum() and w not in _self.stop_words and len(w) > 2]
            
            if len(filtered) < 10:
                return "⚠️ Insufficient text for analysis. Please provide a longer document."
            
            # Create documents for better LDA performance
            try:
                sentences = sent_tokenize(text)
            except:
                # Fallback to simple splitting
                sentences = text.split('. ')
            docs = [s for s in sentences if len(s.split()) > 5]
            
            # If not enough sentences, create chunks
            if len(docs) < 5:
                chunk_size = 50
                words_list = text.split()
                docs = [' '.join(words_list[i:i+chunk_size]) for i in range(0, len(words_list), chunk_size)]
                docs = [d for d in docs if len(d.split()) > 10]
            
            if len(docs) < 2:
                # Fallback to keyword extraction only
                return _self._extract_keywords_only(text, file_id)
            
            # TF-IDF with adjusted parameters
            try:
                vectorizer = TfidfVectorizer(
                    max_features=min(100, len(set(filtered))),
                    ngram_range=(1, 2),
                    min_df=1,  # Allow single occurrences for small documents
                    max_df=0.99,  # Very permissive max_df
                    stop_words='english'
                )
                
                doc_matrix = vectorizer.fit_transform(docs)
                
                # Check if we have enough terms
                if doc_matrix.shape[1] < 5:
                    return _self._extract_keywords_only(text, file_id)
                
            except ValueError as e:
                if "no terms remain" in str(e).lower() or "vocabulary" in str(e).lower():
                    return _self._extract_keywords_only(text, file_id)
                raise
            
            # LDA with appropriate number of topics
            n_topics = min(10, max(3, len(docs) // 3))
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                learning_method='online',
                random_state=42,
                max_iter=50,
                learning_offset=10.0
            )
            lda.fit(doc_matrix)
            
            # Extract themes
            feature_names = vectorizer.get_feature_names_out()
            themes_list = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-15:][::-1]
                top_words = [feature_names[i] for i in top_indices if i < len(feature_names)]
                themes_list.append({
                    'theme': f"Theme {topic_idx + 1}",
                    'words': top_words[:10],
                    'strength': float(topic[top_indices].sum())
                })
            
            # YAKE keywords as supplement
            try:
                kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=20)
                keywords = kw_extractor.extract_keywords(text[:10000])
            except:
                keywords = []
            
            # Format output
            output = "## 📊 Identified Themes\n\n"
            for theme in sorted(themes_list, key=lambda x: x['strength'], reverse=True)[:7]:
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
            return _self._extract_keywords_only(text, file_id)
    
    def extract_keywords(self, text: str, file_id: str) -> str:
        """Public method for keyword extraction"""
        return self._extract_keywords_only(text, file_id)
    
    def _extract_keywords_only(self, text: str, file_id: str) -> str:
        """Fallback to keyword extraction when theme modeling fails"""
        try:
            output = "## 🔑 Key Topics Analysis\n\n"
            output += "*Note: Document structure requires keyword-based analysis instead of full theme modeling.*\n\n"
            
            # Use YAKE for keyword extraction
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Max ngram size
                dedupLim=0.7,
                top=30
            )
            
            keywords = kw_extractor.extract_keywords(text[:20000])
            
            if keywords:
                # Group by similarity
                output += "### Top Keywords and Phrases\n\n"
                for i, (kw, score) in enumerate(keywords[:20], 1):
                    relevance = min(100, int((1/score) * 10))
                    output += f"{i}. **{kw}** - Relevance: {relevance}%\n"
            
            # Basic word frequency analysis
            words = word_tokenize(text.lower())
            filtered = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
            word_freq = Counter(filtered)
            
            output += "\n### Most Frequent Terms\n\n"
            for word, count in word_freq.most_common(15):
                output += f"- {word}: {count} occurrences\n"
            
            # Store and return
            db_manager.store_analysis(file_id, "keyword_analysis", {"keywords": output})
            return output
            
        except Exception as e:
            return f"⚠️ Analysis failed: {str(e)[:100]}. Please try with a different document."

# Initialize components
file_processor = EnhancedFileProcessor()
text_analyzer = EnhancedTextAnalyzer()

# Note: RAG classes (ConversationMemory, PersonaManager, RAGSystem) are defined above

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
        'rag_system': RAGSystem(db_manager) if rag_available else None,
        'persona_manager': PersonaManager() if rag_available else None,
        'conversation_memory': ConversationMemory() if rag_available else None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main Application
def main():
    """Enhanced main application"""
    # Initialize session state first
    init_session_state()
    
    # Header with theme colors
    if ui_available:
        UIComponents.render_modern_header()
    else:
        st.markdown("""
        <h1 style='text-align: center; color: #FDB515; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🔬 NLP Tool for YPAR
        </h1>
        <p style='text-align: center; color: #ffffff; font-size: 1.2em;'>
            Advanced Text Analysis Platform
        </p>
        """, unsafe_allow_html=True)
    
    # Enhanced navigation with better styling
    with st.sidebar:
        # Themed header
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #FDB515; margin: 0;">🔬 NLP YPAR Tool</h2>
            <p style="color: #ffffff; font-size: 14px; margin: 5px 0;">Research Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Clean Navigation Menu
        st.markdown("### 🧭 Navigation")
        
        # Simple and clean navigation
        nav_options = [
            "🏠 Home", 
            "📤 Upload Data", 
            "🔍 Text Analysis", 
            "📊 Visualizations",
            "🤖 RAG Analysis",
            "⚙️ Settings"
        ]
        
        selected = st.radio(
            "Select Page:",
            nav_options,
            label_visibility="collapsed",
            key="main_nav"
        )
        
        # Clean up emoji prefixes for routing
        selected = selected.split(' ', 1)[1] if ' ' in selected else selected
        
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
    elif selected == "Visualizations":
        show_visualizations()
    elif selected == "RAG Analysis":
        show_rag_analysis()
    elif selected == "Settings":
        show_settings()
    else:
        # Default to home if no match
        show_home()

def show_home():
    """Enhanced home page"""
    st.markdown("## Welcome to the Enhanced NLP Tool")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🚀 **Advanced Processing**\nSupport for 10+ file formats including PDF, Word, Excel, HTML, and more")
    
    with col2:
        st.info("🤖 **RAG System**\nRetrieval-Augmented Generation with personas and conversation memory")
    
    with col3:
        st.info("📊 **Rich Visualizations**\nInteractive charts, word clouds, and network graphs")
    
    # Stats
    # Display stats using metrics (render_stats_dashboard not available)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files Processed", len(st.session_state.processed_data))
    col2.metric("Analyses Run", len(st.session_state.get('analysis_results', [])))
    col3.metric("Active Personas", "3")
    col4.metric("Knowledge Base", "0")

def show_upload():
    """Enhanced upload page"""
    st.markdown("## 📤 Upload Your Data")
    
    # Use restricted file types for reliability
    supported_types = ['txt', 'pdf', 'docx', 'md']  # Only formats we can handle well
    
    # Display clear guidelines
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        ### ✅ Supported Formats
        - **Text files** (.txt) - Plain text documents
        - **PDF files** (.pdf) - Up to 10MB
        - **Word documents** (.docx) - Microsoft Word
        - **Markdown** (.md) - Formatted text
        """)
    
    with col2:
        st.warning(f"""
        ### ⚠️ File Requirements
        - Maximum size: **{Config.MAX_FILE_SIZE / (1024*1024):.0f}MB per file**
        - Text should be in English
        - Files should contain readable text (not images)
        - Avoid files with complex formatting
        """)
    
    uploaded_files = st.file_uploader(
        "Choose files (drag and drop or click to browse)",
        type=supported_types,
        accept_multiple_files=True,
        help="Select one or more text-based documents for analysis"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            # Validate files before processing
            valid_files = []
            for file in new_files:
                # Check file size
                file.seek(0, 2)  # Move to end of file
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                
                if file_size > Config.MAX_FILE_SIZE:
                    st.error(f"❌ {file.name}: File too large ({file_size / (1024*1024):.1f}MB). Maximum is {Config.MAX_FILE_SIZE / (1024*1024):.0f}MB")
                    continue
                
                # Check file extension
                file_ext = file.name.split('.')[-1].lower()
                if file_ext not in supported_types:
                    st.error(f"❌ {file.name}: Unsupported file type (.{file_ext})")
                    continue
                
                # Check for minimum file size (likely empty)
                if file_size < 10:
                    st.warning(f"⚠️ {file.name}: File appears to be empty")
                    continue
                
                valid_files.append(file)
            
            if valid_files:
                st.success(f"✅ {len(valid_files)} valid file(s) ready to process")
                
                if st.button(f"🚀 Process {len(valid_files)} File(s)", type="primary"):
                    progress = st.progress(0)
                    successful_uploads = 0
                    
                    for idx, file in enumerate(valid_files):
                        progress.progress((idx + 1) / len(valid_files))
                        
                        try:
                            result = file_processor.process_file(file)
                            if result:
                                content, file_id, unique_name, metadata = result
                                
                                # Additional validation - check if content is readable
                                if len(content.strip()) < 100:
                                    st.warning(f"⚠️ {file.name}: File has very little text content")
                                    continue
                                
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
                                successful_uploads += 1
                            else:
                                st.error(f"❌ {file.name}: Failed to extract text")
                        except Exception as e:
                            st.error(f"❌ {file.name}: Processing error - {str(e)[:100]}")
                            logger.error(f"File processing error for {file.name}: {e}")
                    
                    if successful_uploads > 0:
                        st.balloons()
                        st.info(f"Successfully processed {successful_uploads} out of {len(valid_files)} files")
                        st.rerun()  # Refresh to show the preview
                    else:
                        st.error("No files were successfully processed. Please check your files and try again.")
            else:
                st.warning("No valid files to process. Please check the file requirements above.")
        
        # Display processed files with preview
        if st.session_state.file_names:
            st.markdown("### 📁 Processed Files & Document Preview")
            
            # Create tabs for each file
            if len(st.session_state.file_names) > 1:
                tabs = st.tabs([f"📄 {name.split('_')[0]}" for name in st.session_state.file_names])
            else:
                tabs = [st.container()]
            
            for i, (tab, name, metadata, content) in enumerate(zip(
                tabs, 
                st.session_state.file_names, 
                st.session_state.file_metadata,
                st.session_state.processed_data
            )):
                with tab:
                    # File information header
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"### {name.split('_')[0]}")
                    with col2:
                        st.metric("Words", metadata.get('word_count', 0))
                    with col3:
                        st.metric("Characters", metadata.get('char_count', 0))
                    with col4:
                        try:
                            sentence_count = len(sent_tokenize(content[:5000]))
                        except:
                            # Fallback to simple sentence counting
                            sentence_count = content.count('.') + content.count('!') + content.count('?')
                        st.metric("Sentences", sentence_count)
                    
                    # Document preview section
                    st.markdown("---")
                    st.markdown("#### 📖 Document Preview")
                    
                    # Add controls for preview
                    preview_col1, preview_col2, preview_col3 = st.columns([2, 1, 1])
                    
                    with preview_col1:
                        preview_length = st.slider(
                            "Preview length (characters)", 
                            min_value=500, 
                            max_value=min(10000, len(content)), 
                            value=min(2000, len(content)),
                            key=f"preview_slider_{i}"
                        )
                    
                    with preview_col2:
                        show_full = st.checkbox("Show full document", key=f"show_full_{i}")
                    
                    with preview_col3:
                        highlight_keywords = st.checkbox("Highlight keywords", key=f"highlight_{i}")
                    
                    # Display content with formatting
                    display_content = content if show_full else content[:preview_length]
                    
                    # Format content with proper page breaks if detected
                    if "--- Page" in display_content or "Page " in display_content[:100]:
                        # Split by page markers
                        import re
                        pages = re.split(r'(?:---|–––)\s*Page\s*\d+\s*(?:---|–––)', display_content)
                        
                        # If no clear page breaks found, try other patterns
                        if len(pages) <= 1:
                            pages = re.split(r'\n\s*Page\s+\d+\s*\n', display_content)
                        
                        if len(pages) > 1:
                            # Display with page navigation
                            page_num = st.selectbox(
                                "Navigate pages:", 
                                range(1, len(pages) + 1),
                                format_func=lambda x: f"Page {x} of {len(pages)}",
                                key=f"page_nav_{i}"
                            )
                            current_page = pages[page_num - 1] if page_num <= len(pages) else pages[0]
                            
                            # Apply keyword highlighting if enabled
                            if highlight_keywords:
                                try:
                                    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=10)
                                    keywords = kw_extractor.extract_keywords(content[:5000])
                                    
                                    for keyword, _ in keywords[:10]:
                                        current_page = current_page.replace(
                                            keyword, 
                                            f'<mark style="background-color: #FDB515; padding: 2px; border-radius: 3px;">{keyword}</mark>'
                                        )
                                except:
                                    pass
                            
                            # Display current page
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #ffffff;
                                    border: 2px solid #003262;
                                    border-radius: 10px;
                                    padding: 30px;
                                    max-height: 600px;
                                    overflow-y: auto;
                                    font-family: 'Georgia', 'Times New Roman', serif;
                                    font-size: 16px;
                                    line-height: 1.8;
                                    color: #333;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                ">
                                    <div style="text-align: center; color: #003262; font-weight: bold; margin-bottom: 20px;">
                                        📖 Page {page_num} of {len(pages)}
                                    </div>
                                    <div style="white-space: pre-wrap; word-wrap: break-word;">
                                        {current_page.strip()}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            # No pages detected, show as single document
                            display_formatted = display_content
                            
                            # Apply keyword highlighting if enabled
                            if highlight_keywords:
                                try:
                                    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=10)
                                    keywords = kw_extractor.extract_keywords(content[:5000])
                                    
                                    for keyword, _ in keywords[:10]:
                                        display_formatted = display_formatted.replace(
                                            keyword, 
                                            f'<mark style="background-color: #FDB515; padding: 2px; border-radius: 3px;">{keyword}</mark>'
                                        )
                                except:
                                    pass
                            
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #ffffff;
                                    border: 2px solid #003262;
                                    border-radius: 10px;
                                    padding: 30px;
                                    max-height: 600px;
                                    overflow-y: auto;
                                    font-family: 'Georgia', 'Times New Roman', serif;
                                    font-size: 16px;
                                    line-height: 1.8;
                                    color: #333;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                ">
                                    <div style="white-space: pre-wrap; word-wrap: break-word;">
                                        {display_formatted}{'...' if not show_full and len(content) > preview_length else ''}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        # Standard display for non-paginated content
                        display_formatted = display_content
                        
                        # Apply keyword highlighting if enabled
                        if highlight_keywords:
                            try:
                                kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=10)
                                keywords = kw_extractor.extract_keywords(content[:5000])
                                
                                for keyword, _ in keywords[:10]:
                                    display_formatted = display_formatted.replace(
                                        keyword, 
                                        f'<mark style="background-color: #FDB515; padding: 2px; border-radius: 3px;">{keyword}</mark>'
                                    )
                            except:
                                pass
                        
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #ffffff;
                                border: 2px solid #003262;
                                border-radius: 10px;
                                padding: 30px;
                                max-height: 600px;
                                overflow-y: auto;
                                font-family: 'Georgia', 'Times New Roman', serif;
                                font-size: 16px;
                                line-height: 1.8;
                                color: #333;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <div style="white-space: pre-wrap; word-wrap: break-word;">
                                    {display_formatted}{'...' if not show_full and len(content) > preview_length else ''}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Quick analysis section
                    st.markdown("---")
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        if st.button(f"🔍 Quick Analysis", key=f"quick_analysis_{i}"):
                            with st.spinner("Analyzing..."):
                                # Sentiment analysis
                                try:
                                    blob = TextBlob(content[:5000])
                                    sentiment = blob.sentiment
                                    
                                    st.markdown("##### Sentiment Analysis")
                                    sentiment_score = sentiment.polarity
                                    if sentiment_score > 0.3:
                                        st.success(f"Positive ({sentiment_score:.2f})")
                                    elif sentiment_score < -0.3:
                                        st.error(f"Negative ({sentiment_score:.2f})")
                                    else:
                                        st.info(f"Neutral ({sentiment_score:.2f})")
                                    
                                    st.markdown(f"Subjectivity: {sentiment.subjectivity:.2f}")
                                except:
                                    st.warning("Sentiment analysis unavailable")
                    
                    with analysis_col2:
                        if st.button(f"📊 Extract Keywords", key=f"extract_kw_{i}"):
                            with st.spinner("Extracting keywords..."):
                                try:
                                    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=15)
                                    keywords = kw_extractor.extract_keywords(content[:10000])
                                    
                                    st.markdown("##### Top Keywords")
                                    for kw, score in keywords[:10]:
                                        st.markdown(f"• {kw}")
                                except:
                                    st.warning("Keyword extraction failed")
                    
                    with st.expander("📋 File Metadata"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Type**: {metadata.get('type', 'Unknown')}")
                            st.write(f"**Size**: {metadata.get('size', 0) / 1024:.1f}KB")
                        
                        with col2:
                            st.write(f"**Pages**: {metadata.get('pages', 'N/A')}")
                            st.write(f"**Tables**: {metadata.get('tables', 'N/A')}")
                        
                        with col3:
                            st.write(f"**Word Count**: {metadata.get('word_count', 0)}")
                            st.write(f"**Processed**: ✅")

def show_analysis():
    """Enhanced text analysis page with AI integration"""
    st.markdown("## 🔍 Text Analysis")
    
    # Check if OpenAI is available
    ai_status = "🤖 AI-Powered" if text_analyzer.use_ai else "📊 Traditional NLP"
    st.info(f"Analysis Mode: {ai_status}")
    
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
        metadata = st.session_state.file_metadata[file_index]
        
        # Create streamlined analysis interface
        st.markdown("### 📋 Analysis Options")
        
        if text_analyzer.use_ai:
            # AI-powered analysis tabs
            tab_names = ["🤖 Quick Analysis", "📊 Sentiment", "🎯 Themes", "📝 Summary", "💡 Insights"]
            tabs = st.tabs(tab_names)
        
        # AI Analysis Tab (if available)
        if text_analyzer.use_ai:
            with tabs[0]:
                st.subheader("🤖 Comprehensive AI Analysis")
                
                if st.button("Run Complete AI Analysis", type="primary"):
                    with st.spinner("Running AI analysis..."):
                        start_time = time.time()
                        
                        # Run multiple analyses
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment Analysis
                            sentiment_result = text_analyzer.ai_analyzer.analyze_sentiment(text)
                            if "error" not in sentiment_result:
                                st.markdown("### Sentiment Analysis")
                                st.metric("Sentiment", sentiment_result.get("sentiment", "N/A"))
                                st.metric("Score", f"{sentiment_result.get('score', 0):.2f}")
                                st.metric("Confidence", f"{sentiment_result.get('confidence', 0)}%")
                                st.write(f"**Explanation**: {sentiment_result.get('explanation', 'N/A')}")
                                
                                # Store in database
                                db_manager.store_analysis(file_id, "ai_sentiment", sentiment_result, 
                                                        filename=selected_file)
                        
                        with col2:
                            # Summary
                            summary_result = text_analyzer.ai_analyzer.summarize_text(text, 200)
                            if "error" not in summary_result:
                                st.markdown("### AI Summary")
                                st.write(summary_result.get("summary", "N/A"))
                                st.caption(f"Compression: {summary_result.get('compression_ratio', 'N/A')}")
                                
                                # Store in database
                                db_manager.store_analysis(file_id, "ai_summary", summary_result,
                                                        filename=selected_file)
                        
                        # Themes
                        themes_result = text_analyzer.ai_analyzer.extract_themes(text, 5)
                        if "error" not in themes_result and themes_result.get("themes"):
                            st.markdown("### Key Themes Identified")
                            for theme in themes_result["themes"][:5]:
                                with st.expander(f"📌 {theme.get('title', 'Theme')}"):
                                    st.write(f"**Description**: {theme.get('description', 'N/A')}")
                                    st.write(f"**Keywords**: {', '.join(theme.get('keywords', []))}")
                                    st.write(f"**Relevance**: {theme.get('relevance', 0)}%")
                            
                            db_manager.store_analysis(file_id, "ai_themes", themes_result,
                                                    filename=selected_file)
                        
                        processing_time = time.time() - start_time
                        st.success(f"✅ AI Analysis complete in {processing_time:.1f} seconds")
            
            # Sentiment Tab
            with tabs[1]:
                st.subheader("📊 AI Sentiment Analysis")
                if st.button("Analyze Sentiment", key="sentiment_btn"):
                    with st.spinner("Analyzing sentiment..."):
                        result = text_analyzer.ai_analyzer.analyze_sentiment(text)
                        if "error" not in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment", result.get("sentiment", "N/A"))
                            with col2:
                                st.metric("Score", f"{result.get('score', 0):.2f}")
                            with col3:
                                st.metric("Confidence", f"{result.get('confidence', 0)}%")
                            
                            st.write("### Emotional Themes")
                            for theme in result.get("themes", []):
                                st.write(f"• {theme}")
                            
                            st.write("### Analysis")
                            st.write(result.get("explanation", ""))
                            
                            db_manager.store_analysis(file_id, "ai_sentiment_detailed", result,
                                                    filename=selected_file)
                        else:
                            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
            # Themes Tab
            with tabs[2]:
                st.subheader("🎯 AI Theme Extraction")
                num_themes = st.slider("Number of themes", 3, 10, 5)
                if st.button("Extract Themes", key="themes_btn"):
                    with st.spinner("Extracting themes..."):
                        result = text_analyzer.ai_analyzer.extract_themes(text, num_themes)
                        if "error" not in result and result.get("themes"):
                            for i, theme in enumerate(result["themes"], 1):
                                with st.expander(f"Theme {i}: {theme.get('title', 'Untitled')}"):
                                    st.write(f"**Description**: {theme.get('description', 'N/A')}")
                                    st.write(f"**Related Concepts**: {', '.join(theme.get('keywords', []))}")
                                    st.write(f"**Relevance Score**: {theme.get('relevance', 0)}%")
                                    if theme.get('quotes'):
                                        st.write("**Supporting Quotes**:")
                                        for quote in theme.get('quotes', []):
                                            st.write(f"> {quote}")
                            
                            db_manager.store_analysis(file_id, "ai_themes_detailed", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to extract themes")
            
            # Keywords Tab
            with tabs[3]:
                st.subheader("🔑 AI Keyword Extraction")
                num_keywords = st.slider("Number of keywords", 10, 30, 20)
                if st.button("Extract Keywords", key="keywords_btn"):
                    with st.spinner("Extracting keywords..."):
                        result = text_analyzer.ai_analyzer.extract_keywords(text, num_keywords)
                        if "error" not in result and result.get("keywords"):
                            df_data = []
                            for kw in result["keywords"]:
                                df_data.append({
                                    "Keyword": kw.get("keyword", ""),
                                    "Importance": kw.get("importance", 0),
                                    "Category": kw.get("category", ""),
                                    "Context": kw.get("context", "")[:100]
                                })
                            
                            if df_data:
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True)
                            
                            db_manager.store_analysis(file_id, "ai_keywords", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to extract keywords")
            
            # Insights Tab
            with tabs[4]:
                st.subheader("📝 AI Text Summarization")
                summary_length = st.slider("Summary length (words)", 100, 500, 250)
                if st.button("Generate Summary", key="summary_btn"):
                    with st.spinner("Generating summary..."):
                        result = text_analyzer.ai_analyzer.summarize_text(text, summary_length)
                        if "error" not in result:
                            st.markdown("### Summary")
                            st.write(result.get("summary", ""))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Original Length", f"{result.get('original_length', 0):,} chars")
                            with col2:
                                st.metric("Compression", result.get('compression_ratio', 'N/A'))
                            
                            db_manager.store_analysis(file_id, "ai_summary_detailed", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to generate summary")
            
            # Additional analysis features can be added here
            # Quotes, Insights, Q&A tabs removed for simplicity
                num_quotes = st.slider("Number of quotes", 5, 20, 10)
                if st.button("Extract Quotes", key="quotes_btn"):
                    with st.spinner("Extracting quotes..."):
                        result = text_analyzer.ai_analyzer.extract_quotes(text, num_quotes)
                        if "error" not in result and result.get("quotes"):
                            for i, quote in enumerate(result["quotes"], 1):
                                sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
                                sentiment_icon = sentiment_color.get(quote.get("sentiment", "neutral"), "⚪")
                                
                                with st.expander(f"{sentiment_icon} Quote {i}"):
                                    st.write(f"**Quote**: \"{quote.get('quote', '')}\"")
                                    st.write(f"**Speaker**: {quote.get('speaker', 'Unknown')}")
                                    st.write(f"**Context**: {quote.get('context', 'N/A')}")
                                    st.write(f"**Significance**: {quote.get('significance', 'N/A')}")
                            
                            db_manager.store_analysis(file_id, "ai_quotes", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to extract quotes")
            
            # with tabs[6]: # Disabled - exceeds tab count
                st.subheader("💡 AI Research Insights")
                focus_areas = st.multiselect(
                    "Focus areas (optional)",
                    ["Methodology", "Findings", "Implications", "Recommendations", 
                     "Limitations", "Future Research", "Key Arguments"]
                )
                if st.button("Generate Insights", key="insights_btn"):
                    with st.spinner("Generating insights..."):
                        result = text_analyzer.ai_analyzer.generate_insights(text, focus_areas)
                        if "error" not in result:
                            insights = result.get("insights", {})
                            if isinstance(insights, dict):
                                for key, value in insights.items():
                                    st.markdown(f"### {key.replace('_', ' ').title()}")
                                    if isinstance(value, list):
                                        for item in value:
                                            st.write(f"• {item}")
                                    else:
                                        st.write(value)
                            else:
                                st.write(insights)
                            
                            db_manager.store_analysis(file_id, "ai_insights", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to generate insights")
            
            # with tabs[7]: # Disabled - exceeds tab count
                st.subheader("❓ Ask Questions About the Document")
                question = st.text_input("Enter your question:")
                if st.button("Get Answer", key="qa_btn") and question:
                    with st.spinner("Finding answer..."):
                        result = text_analyzer.ai_analyzer.answer_question(text, question)
                        if "error" not in result:
                            st.markdown("### Answer")
                            st.write(result.get("answer", ""))
                            
                            # Store Q&A in session
                            if 'qa_history' not in st.session_state:
                                st.session_state.qa_history = []
                            st.session_state.qa_history.append(result)
                            
                            db_manager.store_analysis(file_id, "ai_qa", result,
                                                    filename=selected_file)
                        else:
                            st.error("Failed to answer question")
                
                # Show Q&A history
                if 'qa_history' in st.session_state and st.session_state.qa_history:
                    st.markdown("### Previous Questions")
                    for qa in st.session_state.qa_history[-5:]:
                        with st.expander(qa.get("question", "Question")):
                            st.write(qa.get("answer", ""))
        
        # Traditional Analysis (fallback) - Run all at once
        else:
            st.subheader("📊 Traditional NLP Analysis")
            st.info("💡 Configure OpenAI API key in Settings for AI-powered analysis")
            
            if st.button("🔬 Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive analysis..."):
                    # Container for results
                    results_container = st.container()
                    
                    with results_container:
                        # Run all three analyses
                        col1, col2, col3 = st.columns(3)
                        
                        # 1. Basic Analysis
                        with col1:
                            st.markdown("### 📊 Text Statistics")
                            word_count = len(text.split())
                            char_count = len(text)
                            sentence_count = text.count('.') + text.count('!') + text.count('?')
                            
                            st.metric("Words", word_count)
                            st.metric("Characters", char_count)
                            st.metric("Sentences", sentence_count)
                        
                        # 2. Theme Analysis
                        with col2:
                            st.markdown("### 🎯 Key Themes")
                            themes = text_analyzer.analyze_themes(text, file_id)
                            if themes:
                                # Extract theme words from the result
                                import re
                                theme_words = re.findall(r'\*\*([^*]+)\*\*', themes)[:5]
                                for theme in theme_words:
                                    st.write(f"• {theme}")
                                
                                # Store themes
                                db_manager.store_analysis(file_id, "traditional_themes", 
                                                        {"themes": theme_words})
                        
                        # 3. Keyword Extraction
                        with col3:
                            st.markdown("### 🔑 Top Keywords")
                            keywords = text_analyzer.extract_keywords(text, file_id)
                            if keywords:
                                # Extract keywords from result
                                keyword_list = keywords.split('\n')[:5]
                                for kw in keyword_list:
                                    if kw.strip():
                                        st.write(f"• {kw.strip()}")
                                
                                # Store keywords
                                db_manager.store_analysis(file_id, "traditional_keywords", 
                                                        {"keywords": keyword_list})
                        
                        # Sentiment Analysis using TextBlob
                        st.divider()
                        sentiment_col1, sentiment_col2 = st.columns(2)
                        
                        with sentiment_col1:
                            st.markdown("### 😊 Sentiment Analysis")
                            from textblob import TextBlob
                            blob = TextBlob(text[:5000])  # Limit text for performance
                            polarity = blob.sentiment.polarity
                            subjectivity = blob.sentiment.subjectivity
                            
                            # Determine sentiment label
                            if polarity > 0.1:
                                sentiment_label = "Positive"
                                sentiment_color = "green"
                            elif polarity < -0.1:
                                sentiment_label = "Negative"
                                sentiment_color = "red"
                            else:
                                sentiment_label = "Neutral"
                                sentiment_color = "gray"
                            
                            st.metric("Sentiment", sentiment_label)
                            st.metric("Polarity", f"{polarity:.2f}")
                            st.metric("Subjectivity", f"{subjectivity:.2f}")
                            
                            # Store sentiment
                            db_manager.store_analysis(file_id, "traditional_sentiment", {
                                "sentiment": sentiment_label,
                                "polarity": polarity,
                                "subjectivity": subjectivity
                            })
                        
                        with sentiment_col2:
                            st.markdown("### 📈 Analysis Summary")
                            st.success(f"""
                            ✅ **Analysis Complete**
                            - Analyzed {word_count} words
                            - Extracted {len(theme_words) if 'theme_words' in locals() else 0} themes
                            - Found {len(keyword_list) if 'keyword_list' in locals() else 0} keywords
                            - Sentiment: {sentiment_label}
                            """)
                        
                        # Store complete analysis
                        complete_results = {
                            "word_count": word_count,
                            "themes": theme_words if 'theme_words' in locals() else [],
                            "keywords": keyword_list if 'keyword_list' in locals() else [],
                            "sentiment": sentiment_label,
                            "polarity": polarity,
                            "subjectivity": subjectivity
                        }
                        
                        db_manager.store_analysis(file_id, "traditional_complete", complete_results,
                                                filename=selected_file)

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
                try:
                    sentences = sent_tokenize(text)[:100]
                except:
                    # Fallback to simple splitting
                    sentences = text.split('. ')[:100]
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
                
                edge_x = []
                edge_y = []
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_colors.append(len(G[node]))
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    text=node_text,
                    mode='markers+text',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlOrRd',
                        size=10,
                        color=node_colors,
                        colorbar=dict(
                            thickness=15,
                            title=dict(text='Connections'),
                            xanchor='left'
                        )
                    )
                )
                
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
                    
                    # Check if OpenAI is available for enhanced response generation
                    text_analyzer = EnhancedTextAnalyzer()
                    
                    if text_analyzer.use_ai and text_analyzer.ai_analyzer:
                        # Use OpenAI to generate response with retrieved context
                        context_texts = []
                        for result in results[:5]:  # Use top 5 results
                            context_texts.append(result.get('text', ''))
                        
                        combined_context = "\n\n".join(context_texts)
                        
                        # Create enhanced prompt with persona and context
                        persona_prompts = {
                            "researcher": "You are a research analyst providing detailed academic insights.",
                            "educator": "You are an educator explaining concepts clearly for learning.",
                            "youth_advocate": "You are a youth advocate focusing on young people's perspectives.",
                            "data_analyst": "You are a data analyst providing statistical and analytical insights.",
                            "policy_maker": "You are a policy maker focusing on actionable recommendations."
                        }
                        
                        enhanced_prompt = f"""{persona_prompts.get(persona, "You are an expert analyst.")}
                        
Based on the following retrieved context from the documents, answer this question: {query}

Retrieved Context:
{combined_context[:8000]}  # Limit context to avoid token limits

Provide a comprehensive answer that:
1. Directly addresses the question
2. Synthesizes information from the context
3. Highlights key themes and patterns
4. Includes relevant quotes or examples from the context
5. Provides insights appropriate for the {persona} perspective
"""
                        
                        # Generate AI response
                        ai_response = text_analyzer.ai_analyzer.answer_question(combined_context, enhanced_prompt)
                        
                        if "error" not in ai_response:
                            response = ai_response.get("answer", "")
                            st.success("🤖 Using AI-enhanced RAG analysis")
                        else:
                            # Fallback to traditional RAG response
                            response = st.session_state.rag_system.generate_response(query, results)
                            st.info("Using traditional RAG analysis")
                    else:
                        # Use traditional RAG system
                        response = st.session_state.rag_system.generate_response(query, results)
                        st.info("💡 Configure OpenAI API key in Settings for AI-enhanced analysis")
                    
                    # Add to conversation memory
                    st.session_state.conversation_memory.add_exchange(query, response)
                    
                    # Display results
                    st.markdown("### 🎯 Analysis Results")
                    st.markdown(response)
                    
                    # Store RAG analysis in database
                    if db_manager and response:
                        # Create a unique file ID for this RAG session if not exists
                        if 'rag_session_id' not in st.session_state:
                            import hashlib
                            st.session_state.rag_session_id = hashlib.md5(f"rag_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
                        
                        rag_analysis_result = {
                            "query": query,
                            "response": response,
                            "persona": persona,
                            "analysis_depth": analysis_depth,
                            "retrieved_docs": [
                                {
                                    "text": r.get('text', '')[:500],
                                    "score": r.get('score', 0),
                                    "metadata": r.get('metadata', {})
                                } for r in results[:5]
                            ],
                            "ai_enhanced": text_analyzer.use_ai if 'text_analyzer' in locals() else False
                        }
                        
                        db_manager.store_analysis(
                            st.session_state.rag_session_id,
                            f"rag_analysis_{persona}",
                            rag_analysis_result,
                            filename=f"RAG Query: {query[:50]}..."
                        )
                        st.success("✅ Analysis saved to database")
                    
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

def show_analysis_history():
    """Analysis history and database view"""
    st.markdown("## 📜 Analysis History")
    
    # Import enhanced database manager if available
    try:
        from enhanced_db_manager import EnhancedMongoManager
        enhanced_db = EnhancedMongoManager()
        use_enhanced = enhanced_db.connected
    except:
        enhanced_db = None
        use_enhanced = False
    
    if use_enhanced:
        # Use enhanced MongoDB features
        st.success("✅ Connected to MongoDB - Full history available")
        
        # Get statistics
        stats = enhanced_db.get_analysis_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", stats["total_documents"])
        with col2:
            st.metric("Total Analyses", stats["total_analyses"])
        with col3:
            st.metric("Analysis Types", len(stats["analysis_types"]))
        with col4:
            st.metric("Recent Activity", len(stats["recent_activity"]))
        
        # Create tabs for different views
        tabs = st.tabs(["📊 Overview", "📄 Documents", "🔍 Search", "📈 Statistics", "💾 Export"])
        
        # Overview Tab
        with tabs[0]:
            st.subheader("Recent Analysis Activity")
            
            if stats["recent_activity"]:
                for activity in stats["recent_activity"]:
                    with st.expander(f"📝 {activity['document_name']} - {activity['analysis_type']}"):
                        st.write(f"**Time**: {activity['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Summary**: {activity.get('summary', 'N/A')}")
            else:
                st.info("No recent activity")
        
        # Documents Tab
        with tabs[1]:
            st.subheader("All Documents")
            
            documents = enhanced_db.get_all_documents()
            
            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Upload Date**: {doc['upload_timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Word Count**: {doc['metadata']['word_count']:,}")
                            st.write(f"**File Type**: {doc['metadata']['type']}")
                        
                        with col2:
                            st.write(f"**Analyses Performed**: {len(doc.get('analyses_performed', []))}")
                            if doc.get('last_analyzed'):
                                st.write(f"**Last Analyzed**: {doc['last_analyzed'].strftime('%Y-%m-%d %H:%M')}")
                        
                        if st.button(f"View Analyses", key=f"view_{doc['file_id']}"):
                            analyses = enhanced_db.get_document_analyses(doc['file_id'])
                            
                            if analyses:
                                st.markdown("### Analysis Results")
                                for analysis_type, results in analyses.items():
                                    st.markdown(f"**{analysis_type}**")
                                    for result in results:
                                        st.write(f"- {result['timestamp'].strftime('%Y-%m-%d %H:%M')}: {result.get('summary', 'Completed')}")
            else:
                st.info("No documents found")
        
        # Search Tab
        with tabs[2]:
            st.subheader("Search Analyses")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_query = st.text_input("Search keyword", placeholder="Enter search term...")
            
            with col2:
                analysis_type_filter = st.selectbox(
                    "Filter by type",
                    ["All"] + list(stats["analysis_types"].keys())
                )
            
            if st.button("🔍 Search"):
                filters = {}
                if analysis_type_filter != "All":
                    filters["analysis_type"] = analysis_type_filter
                
                results = enhanced_db.search_analyses(search_query, filters)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    for result in results[:20]:
                        with st.expander(f"{result['document_name']} - {result['analysis_type']}"):
                            st.write(f"**Time**: {result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Summary**: {result.get('summary', 'N/A')}")
                else:
                    st.info("No results found")
        
        # Statistics Tab
        with tabs[3]:
            st.subheader("Analysis Statistics")
            
            # Analysis type distribution
            if stats["analysis_types"]:
                import plotly.express as px
                
                df = pd.DataFrame(
                    list(stats["analysis_types"].items()),
                    columns=["Analysis Type", "Count"]
                )
                
                fig = px.pie(df, values="Count", names="Analysis Type", 
                           title="Analysis Type Distribution",
                           color_discrete_map={
                               "sentiment": "#FDB515",
                               "themes": "#003262",
                               "keywords": "#3B7EA1",
                               "entities": "#C4820E"
                           })
                st.plotly_chart(fig, use_container_width=True)
                
                # Timeline chart
                if stats["recent_activity"]:
                    timeline_data = []
                    for activity in stats["recent_activity"]:
                        timeline_data.append({
                            "Date": activity["timestamp"].date(),
                            "Analysis": activity["analysis_type"],
                            "Document": activity["document_name"][:30]
                        })
                    
                    if timeline_data:
                        df_timeline = pd.DataFrame(timeline_data)
                        
                        # Group by date and count
                        daily_counts = df_timeline.groupby("Date").size().reset_index(name="Count")
                        
                        fig2 = px.line(daily_counts, x="Date", y="Count",
                                     title="Analysis Activity Over Time",
                                     markers=True)
                        fig2.update_traces(line_color="#003262", marker_color="#FDB515")
                        st.plotly_chart(fig2, use_container_width=True)
        
        # Export Tab
        with tabs[4]:
            st.subheader("Export Data")
            
            export_option = st.radio(
                "Export format",
                ["JSON", "CSV", "Full Database Backup"]
            )
            
            if st.button("📥 Export"):
                if export_option == "JSON":
                    export_data = enhanced_db.export_analysis_history()
                    st.download_button(
                        "Download JSON",
                        json.dumps(export_data, default=str, indent=2),
                        "analysis_history.json",
                        "application/json"
                    )
                elif export_option == "CSV":
                    documents = enhanced_db.get_all_documents()
                    if documents:
                        df = pd.DataFrame(documents)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "documents_summary.csv",
                            "text/csv"
                        )
                else:
                    st.info("Run the database_backup.py script for full backup")
    
    else:
        # Fallback to session state
        st.warning("⚠️ MongoDB not connected - Showing session history only")
        
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            st.subheader("Session Analysis History")
            
            # Group by document
            by_document = {}
            for result in st.session_state.analysis_results:
                doc_id = result.get("document_id", result.get("file_id", "unknown"))
                if doc_id not in by_document:
                    by_document[doc_id] = []
                by_document[doc_id].append(result)
            
            for doc_id, analyses in by_document.items():
                doc_name = analyses[0].get("document_name", f"Document {doc_id[:8]}")
                
                with st.expander(f"📄 {doc_name} ({len(analyses)} analyses)"):
                    for analysis in analyses:
                        st.write(f"**Type**: {analysis.get('analysis_type', 'N/A')}")
                        st.write(f"**Time**: {analysis.get('timestamp', 'N/A')}")
                        
                        if 'results' in analysis:
                            st.json(analysis['results'])
                        
                        st.divider()
        else:
            st.info("No analysis history available in this session")
            st.write("Upload and analyze documents to see history here")

def show_settings():
    """Settings page"""
    st.markdown("## ⚙️ Settings")
    
    # Create tabs for different settings sections
    tabs = st.tabs(["🔑 API Configuration", "📊 System Info", "🔧 Actions"])
    
    # API Configuration Tab
    with tabs[0]:
        st.subheader("API & Database Configuration")
        
        # MongoDB Connection
        st.markdown("### 🍃 MongoDB Configuration")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Don't show existing connection string for security
            current_mongo_conn = st.session_state.get('mongodb_connection_string', '')
            has_mongo = bool(current_mongo_conn or Config.get_mongodb_connection_string())
            
            if has_mongo:
                st.info("✅ MongoDB is configured")
            
            mongo_conn = st.text_input(
                "MongoDB Connection String",
                value="",
                type="password",
                placeholder="mongodb://username:password@host:port/database",
                help="Enter your MongoDB connection string. Format: mongodb://[username:password@]host[:port][/database]"
            )
        
        with col2:
            if st.button("Test MongoDB", use_container_width=True):
                if mongo_conn:
                    try:
                        test_client = MongoClient(mongo_conn, serverSelectionTimeoutMS=5000)
                        test_client.server_info()
                        st.success("✅ Connected!")
                        test_client.close()
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)[:100]}")
                else:
                    st.warning("Please enter a connection string")
        
        # OpenAI Configuration
        st.markdown("### 🤖 OpenAI Configuration")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Don't show existing API key for security
            current_openai_key = st.session_state.get('openai_api_key', '')
            has_openai = bool(current_openai_key)
            
            if has_openai:
                st.info("✅ OpenAI is configured")
            
            openai_key = st.text_input(
                "OpenAI API Key",
                value="",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key for advanced AI features"
            )
        
        with col2:
            if st.button("Test OpenAI", use_container_width=True):
                if openai_key:
                    try:
                        import openai
                        from openai import OpenAI
                        
                        # Test with the new OpenAI client
                        client = OpenAI(api_key=openai_key)
                        # Try to list models as a test
                        models = client.models.list()
                        # If we get here, the key is valid
                        st.success("✅ Valid!")
                    except ImportError:
                        # Try older openai library format
                        try:
                            import requests
                            headers = {"Authorization": f"Bearer {openai_key}"}
                            response = requests.get("https://api.openai.com/v1/models", headers=headers)
                            if response.status_code == 200:
                                st.success("✅ Valid!")
                            else:
                                st.error(f"❌ Invalid key (HTTP {response.status_code})")
                        except:
                            st.info("📦 Install openai: pip install openai")
                    except Exception as e:
                        error_msg = str(e)
                        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                            st.error("❌ Invalid API key")
                        else:
                            st.error(f"❌ Error: {error_msg[:100]}")
                else:
                    st.warning("Please enter an API key")
        
        # Save Configuration
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                # Store in session state
                st.session_state['mongodb_connection_string'] = mongo_conn
                st.session_state['openai_api_key'] = openai_key
                
                # Reconnect MongoDB if connection string changed
                if mongo_conn != current_mongo_conn:
                    global db_manager
                    db_manager = EnhancedDatabaseManager()
                    db_manager.client = None
                    db_manager.connected = False
                    if mongo_conn:
                        try:
                            db_manager.client = MongoClient(mongo_conn, serverSelectionTimeoutMS=5000)
                            db_manager.client.server_info()
                            db_manager.db = db_manager.client["nlp_tool"]
                            db_manager.connected = True
                            st.success("✅ Configuration saved and database reconnected!")
                        except Exception as e:
                            st.error(f"Configuration saved but connection failed: {str(e)[:100]}")
                    else:
                        st.success("✅ Configuration saved!")
                else:
                    st.success("✅ Configuration saved!")
                
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True):
                if 'mongodb_connection_string' in st.session_state:
                    del st.session_state['mongodb_connection_string']
                if 'openai_api_key' in st.session_state:
                    del st.session_state['openai_api_key']
                st.success("Reset to default configuration")
                st.rerun()
        
        # Security Notice
        st.info("🔒 **Security Note**: API keys and connection strings are stored in your session and will be cleared when you close the browser. For production use, consider using environment variables or secrets management.")
    
    # System Info Tab
    with tabs[1]:
        st.subheader("📊 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Files Processed", len(st.session_state.processed_data))
            st.metric("Database Status", 'Connected' if db_manager.connected else 'Local Storage')
            
            # Show connection details if connected
            if db_manager.connected and db_manager.client:
                try:
                    db_info = db_manager.client.server_info()
                    st.caption(f"MongoDB v{db_info.get('version', 'Unknown')}")
                except:
                    pass
        
        with col2:
            st.metric("RAG System", 'Available' if rag_available else 'Not Available')
            st.metric("Enhanced UI", 'Available' if ui_available else 'Fallback')
            
            # Show OpenAI status
            if st.session_state.get('openai_api_key'):
                st.caption("OpenAI API: Configured ✅")
            else:
                st.caption("OpenAI API: Not configured")
    
    # Actions Tab
    with tabs[2]:
        st.subheader("🔧 System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['mongodb_connection_string', 'openai_api_key']:
                        del st.session_state[key]
                init_session_state()
                st.success("All data cleared (API keys preserved)")
                st.rerun()
            
            if st.button("🔄 Restart Application", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        with col2:
            if st.button("📥 Export Analysis Results", use_container_width=True):
                if st.session_state.get('analysis_results'):
                    df = pd.DataFrame(st.session_state.analysis_results)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "analysis_results.csv",
                        "text/csv"
                    )
                else:
                    st.info("No analysis results to export")
            
            if st.button("📊 Export System Logs", use_container_width=True):
                try:
                    with open('nlp_ypar.log', 'r') as f:
                        log_data = f.read()
                    st.download_button(
                        "Download Logs",
                        log_data,
                        "nlp_ypar_logs.txt",
                        "text/plain"
                    )
                except:
                    st.error("Could not read log file")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page or contact support")