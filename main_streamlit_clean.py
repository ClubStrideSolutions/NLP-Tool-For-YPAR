"""
NLP Tool for YPAR - Streamlit Cloud Optimized Version
This version has no MongoDB dependencies and uses only session state storage
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NO MONGODB IMPORTS AT ALL

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
        if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
            return st.session_state.openai_api_key
        return os.getenv('OPENAI_API_KEY')
    
    @staticmethod
    def is_ai_enabled():
        """Check if AI features are enabled"""
        return bool(Config.get_openai_api_key()) and OPENAI_AVAILABLE

# ===================== SESSION STATE DATABASE MANAGER =====================
class SessionStateDB:
    """Simple session state based storage - no external dependencies"""
    
    def __init__(self):
        self.cache = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any], 
                      filename: str = None, processing_time: float = 0) -> str:
        """Store analysis in session state"""
        # Cache in memory
        cache_key = f"{file_id}_{analysis_type}"
        self.cache[cache_key] = results
        st.session_state.analysis_cache[cache_key] = results
        
        # Store in session state list
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
        """Get from cache"""
        cache_key = f"{file_id}_{analysis_type}"
        
        # Check memory cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check session state cache
        if cache_key in st.session_state.analysis_cache:
            return st.session_state.analysis_cache[cache_key]
        
        # Check full results list
        for result in st.session_state.analysis_results:
            if result['file_id'] == file_id and result['analysis_type'] == analysis_type:
                self.cache[cache_key] = result['results']
                return result['results']
        
        return None
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Get all analyses from session state"""
        return st.session_state.analysis_results

# Initialize database manager
@st.cache_resource
def get_db_manager():
    return SessionStateDB()

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
        cached = db_manager.get_from_cache(file_id, "sentiment")
        if cached:
            return cached.get('sentiment', 'Unknown')
        
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Analyze the sentiment of this text. Provide: 1) Overall sentiment (positive/negative/neutral), 2) Confidence score, 3) Brief emotional context."
            )
            if result:
                db_manager.store_analysis(file_id, "sentiment", {"sentiment": result})
                return result
        
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
        cached = db_manager.get_from_cache(file_id, "themes")
        if cached:
            return cached.get('themes', '')
        
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract the top 5 main themes from this text. Format: **Theme Name**: Brief description"
            )
            if result:
                db_manager.store_analysis(file_id, "themes", {"themes": result})
                return result
        
        try:
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(sentences[:100])
            
            lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10)
            lda.fit(doc_term_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            
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
        cached = db_manager.get_from_cache(file_id, "keywords")
        if cached:
            return cached.get('keywords', '')
        
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract the top 10 most important keywords or key phrases from this text. List them in order of importance."
            )
            if result:
                db_manager.store_analysis(file_id, "keywords", {"keywords": result})
                return result
        
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=2,
                dedupLim=0.7,
                top=10
            )
            keywords = kw_extractor.extract_keywords(text)
            
            result = []
            for kw, score in keywords:
                importance = "high" if score < 0.05 else "medium" if score < 0.1 else "normal"
                result.append(f"{kw} ({importance})")
            
            formatted = "\n".join(result)
            db_manager.store_analysis(file_id, "keywords", {"keywords": formatted})
            return formatted
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            try:
                words = word_tokenize(text.lower()) if NLTK_AVAILABLE else text.lower().split()
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'])
                words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
                
                word_freq = Counter(words)
                top_words = word_freq.most_common(10)
                
                result = "\n".join([f"{word} (count: {count})" for word, count in top_words])
                db_manager.store_analysis(file_id, "keywords", {"keywords": result})
                return result
            except:
                return "Keyword extraction unavailable"
    
    def extract_entities(self, text: str, file_id: str) -> str:
        """Extract named entities"""
        cached = db_manager.get_from_cache(file_id, "entities")
        if cached:
            return cached.get('entities', '')
        
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract all named entities (people, organizations, locations, dates) from this text. Group by type."
            )
            if result:
                db_manager.store_analysis(file_id, "entities", {"entities": result})
                return result
        
        if NLTK_AVAILABLE and ne_chunk and pos_tag:
            try:
                sentences = sent_tokenize(text)[:10]
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
        cached = db_manager.get_from_cache(file_id, "quotes")
        if cached:
            return cached.get('quotes', '')
        
        if self.ai_enabled:
            result = self._ai_analyze(
                text,
                "Extract 3-5 most impactful or meaningful quotes from this text. Include context if needed."
            )
            if result:
                db_manager.store_analysis(file_id, "quotes", {"quotes": result})
                return result
        
        try:
            quotes = re.findall(r'"([^"]+)"', text)
            
            if not quotes:
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
                strong_words = ['important', 'critical', 'essential', 'must', 'should', 'believe', 'think', 'feel', 'know']
                quotes = [s for s in sentences if any(w in s.lower() for w in strong_words)][:5]
            
            if quotes:
                result = "\n\n".join([f'"{q}"' if not q.startswith('"') else q for q in quotes[:5]])
            else:
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
                result = "\n\n".join([f'"{s}"' for s in sentences[:3]])
            
            db_manager.store_analysis(file_id, "quotes", {"quotes": result})
            return result
            
        except Exception as e:
            logger.error(f"Quote extraction failed: {e}")
            return "Quote extraction unavailable"
    
    def generate_insights(self, text: str, file_id: str) -> str:
        """Generate research insights"""
        cached = db_manager.get_from_cache(file_id, "insights")
        if cached:
            return cached.get('insights', '')
        
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
        
        try:
            words = word_tokenize(text) if NLTK_AVAILABLE else text.split()
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / max(sentence_count, 1)
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            
            insights = []
            
            if word_count > 1000:
                insights.append(f"• **Comprehensive text**: {word_count:,} words across {sentence_count} sentences")
            else:
                insights.append(f"• **Concise text**: {word_count} words in {sentence_count} sentences")
            
            if avg_sentence_length > 20:
                insights.append(f"• **Complex writing style**: Average sentence length of {avg_sentence_length:.1f} words")
            else:
                insights.append(f"• **Clear writing style**: Average sentence length of {avg_sentence_length:.1f} words")
            
            if lexical_diversity > 0.5:
                insights.append(f"• **Rich vocabulary**: {lexical_diversity:.1%} lexical diversity")
            else:
                insights.append(f"• **Focused vocabulary**: {lexical_diversity:.1%} lexical diversity")
            
            try:
                blob = TextBlob(text)
                if blob.sentiment.polarity > 0.2:
                    insights.append("• **Positive tone**: Generally optimistic language")
                elif blob.sentiment.polarity < -0.2:
                    insights.append("• **Critical tone**: Skeptical or negative assessments")
                else:
                    insights.append("• **Balanced tone**: Neutral perspective")
            except:
                pass
            
            result = "\n".join(insights)
            db_manager.store_analysis(file_id, "insights", {"insights": result})
            return result
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return "Insight generation unavailable"
    
    def extract_topics_categories(self, text: str, file_id: str) -> Dict[str, Any]:
        """Extract topics, categories, and concepts with relationships"""
        cached = db_manager.get_from_cache(file_id, "topics_categories")
        if cached:
            return cached
        
        results = {
            'topics': [],
            'categories': [],
            'concepts': [],
            'relationships': []
        }
        
        if self.ai_enabled:
            # Use AI for sophisticated extraction
            prompt = """Analyze this text and extract:
            1. Main TOPICS (5-7 broad subject areas)
            2. CATEGORIES (classify into: Technical, Social, Political, Economic, Cultural, Environmental, Academic)
            3. Key CONCEPTS (important ideas, theories, or terms)
            
            Format each clearly labeled."""
            
            ai_result = self._ai_analyze(text, prompt, max_tokens=800)
            if ai_result:
                try:
                    lines = ai_result.strip().split('\n')
                    current_section = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if 'TOPIC' in line.upper():
                            current_section = 'topics'
                        elif 'CATEGOR' in line.upper():
                            current_section = 'categories'
                        elif 'CONCEPT' in line.upper():
                            current_section = 'concepts'
                        elif current_section and ':' in line:
                            content = line.split(':', 1)[-1].strip()
                            content = re.sub(r'^[-•*]\s*', '', content).strip()
                            if content and len(content) > 2:
                                results[current_section].append(content[:100])
                        elif current_section and line and not line[0].isdigit():
                            content = re.sub(r'^[-•*]\s*', '', line).strip()
                            if content and len(content) > 2:
                                results[current_section].append(content[:100])
                except Exception as e:
                    logger.error(f"AI parsing failed: {e}")
        
        # Traditional extraction using TF-IDF and clustering
        if not results['topics'] or len(results['topics']) < 3:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import LatentDirichletAllocation
                
                # Prepare text
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
                
                # Extract topics using LDA
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1
                )
                
                doc_term_matrix = vectorizer.fit_transform(sentences[:200])
                
                # LDA for topics
                lda = LatentDirichletAllocation(n_components=7, random_state=42, max_iter=20)
                lda.fit(doc_term_matrix)
                
                feature_names = vectorizer.get_feature_names_out()
                
                # Extract topics
                for topic_idx, topic in enumerate(lda.components_):
                    top_indices = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices[:5]]
                    topic_name = ', '.join(top_words[:3])
                    results['topics'].append(topic_name)
                    
                    # Categorize based on keywords
                    topic_lower = topic_name.lower()
                    if any(word in topic_lower for word in ['data', 'system', 'software', 'computer', 'technology']):
                        category = 'Technical'
                    elif any(word in topic_lower for word in ['social', 'people', 'community', 'society', 'human']):
                        category = 'Social'
                    elif any(word in topic_lower for word in ['policy', 'government', 'political', 'law']):
                        category = 'Political'
                    elif any(word in topic_lower for word in ['economic', 'money', 'cost', 'market', 'business']):
                        category = 'Economic'
                    elif any(word in topic_lower for word in ['culture', 'cultural', 'tradition', 'values']):
                        category = 'Cultural'
                    elif any(word in topic_lower for word in ['environment', 'climate', 'nature', 'ecological']):
                        category = 'Environmental'
                    elif any(word in topic_lower for word in ['research', 'study', 'analysis', 'academic']):
                        category = 'Academic'
                    else:
                        category = 'General'
                    
                    if category not in results['categories']:
                        results['categories'].append(category)
                    
                    # Extract key concepts
                    for word in top_words[:3]:
                        if word not in results['concepts'] and len(word) > 3:
                            results['concepts'].append(word)
                
                # Create relationships between topics
                for i, topic1 in enumerate(results['topics']):
                    for j, topic2 in enumerate(results['topics'][i+1:], i+1):
                        words1 = set(topic1.lower().split(', '))
                        words2 = set(topic2.lower().split(', '))
                        shared = words1 & words2
                        
                        if shared:
                            strength = 'strong' if len(shared) > 1 else 'medium'
                        else:
                            # Check for semantic similarity
                            strength = 'weak'
                        
                        results['relationships'].append({
                            'source': topic1.split(',')[0].strip()[:30],
                            'target': topic2.split(',')[0].strip()[:30],
                            'strength': strength
                        })
                        
                        if len(results['relationships']) >= 20:
                            break
                    if len(results['relationships']) >= 20:
                        break
                        
            except Exception as e:
                logger.error(f"Topic extraction failed: {e}")
                # Simple fallback
                words = word_tokenize(text.lower()) if NLTK_AVAILABLE else text.lower().split()
                word_freq = Counter([w for w in words if len(w) > 4 and w.isalpha()])
                top_words = word_freq.most_common(20)
                
                results['topics'] = [word for word, _ in top_words[:7]]
                results['categories'] = ['General', 'Academic']
                results['concepts'] = [word for word, _ in top_words[7:15]]
                
                # Simple relationships
                for i in range(min(5, len(results['topics'])-1)):
                    results['relationships'].append({
                        'source': results['topics'][i][:30],
                        'target': results['topics'][i+1][:30],
                        'strength': 'medium'
                    })
        
        # Ensure we have at least some relationships for visualization
        if not results['relationships'] and results['topics']:
            # Create a star pattern with first topic at center
            center = results['topics'][0] if results['topics'] else "Main Topic"
            for topic in results['topics'][1:6]:
                results['relationships'].append({
                    'source': center.split(',')[0].strip()[:30],
                    'target': topic.split(',')[0].strip()[:30],
                    'strength': 'medium'
                })
        
        # Store results
        db_manager.store_analysis(file_id, "topics_categories", results)
        return results
    
    def answer_question(self, text: str, question: str, file_id: str) -> str:
        """Answer questions about the text"""
        if self.ai_enabled:
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
        
        try:
            question_lower = question.lower()
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
            question_words = set(word_tokenize(question_lower) if NLTK_AVAILABLE else question_lower.split())
            stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
            keywords = question_words - stop_words
            
            relevant = []
            for sent in sentences:
                sent_lower = sent.lower()
                score = sum(1 for kw in keywords if kw in sent_lower)
                if score > 0:
                    relevant.append((score, sent))
            
            relevant.sort(reverse=True)
            
            if relevant:
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
            if CHARDET_AVAILABLE:
                raw = file.read()
                result = chardet.detect(raw)
                encoding = result['encoding'] or 'utf-8'
                file.seek(0)
                text = file.read().decode(encoding, errors='ignore')
            else:
                text = file.read().decode('utf-8', errors='ignore')
            
            return text
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
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
        
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        file_id = hashlib.md5(file_content).hexdigest()
        
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
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Blues',
                max_words=50
            ).generate(text)
            
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
    def create_topics_network(topics_data: Dict[str, Any]) -> go.Figure:
        """Create an interactive network plot of topics, categories, and concepts"""
        try:
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes for topics
            topics = topics_data.get('topics', [])
            categories = topics_data.get('categories', [])
            concepts = topics_data.get('concepts', [])
            relationships = topics_data.get('relationships', [])
            
            # Add all nodes with attributes
            for topic in topics[:10]:  # Limit for visualization
                G.add_node(topic[:30], node_type='topic', size=30)
            
            for category in categories[:5]:
                G.add_node(category, node_type='category', size=25)
            
            for concept in concepts[:15]:
                G.add_node(concept[:20], node_type='concept', size=15)
            
            # Add edges from relationships
            for rel in relationships:
                source = rel.get('source', '')[:30]
                target = rel.get('target', '')[:30]
                strength = rel.get('strength', 'medium')
                
                if source in G.nodes() and target in G.nodes():
                    weight = {'strong': 3, 'medium': 2, 'weak': 1}.get(strength, 1)
                    G.add_edge(source, target, weight=weight)
            
            # Add connections between categories and topics
            for topic in topics[:10]:
                topic_short = topic[:30]
                topic_lower = topic.lower()
                
                # Connect to appropriate category
                if 'technical' in categories or 'Technical' in categories:
                    if any(word in topic_lower for word in ['data', 'system', 'software']):
                        G.add_edge(topic_short, 'Technical', weight=2)
                
                if 'social' in categories or 'Social' in categories:
                    if any(word in topic_lower for word in ['social', 'people', 'community']):
                        G.add_edge(topic_short, 'Social', weight=2)
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = edge[2].get('weight', 1)
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=weight,
                        color='rgba(125,125,125,0.5)'
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create node traces by type
            node_traces = []
            
            # Topics (large blue nodes)
            topic_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'topic']
            if topic_nodes:
                x_topics = [pos[node][0] for node in topic_nodes]
                y_topics = [pos[node][1] for node in topic_nodes]
                
                topic_trace = go.Scatter(
                    x=x_topics,
                    y=y_topics,
                    mode='markers+text',
                    name='Topics',
                    text=topic_nodes,
                    textposition="top center",
                    textfont=dict(size=10, color='#003262'),
                    marker=dict(
                        size=25,
                        color='#003262',
                        line=dict(color='#FDB515', width=2),
                        symbol='circle'
                    ),
                    hovertemplate='<b>Topic:</b> %{text}<extra></extra>'
                )
                node_traces.append(topic_trace)
            
            # Categories (medium gold nodes)
            category_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'category']
            if category_nodes:
                x_categories = [pos[node][0] for node in category_nodes]
                y_categories = [pos[node][1] for node in category_nodes]
                
                category_trace = go.Scatter(
                    x=x_categories,
                    y=y_categories,
                    mode='markers+text',
                    name='Categories',
                    text=category_nodes,
                    textposition="bottom center",
                    textfont=dict(size=9, color='#FDB515'),
                    marker=dict(
                        size=20,
                        color='#FDB515',
                        line=dict(color='#003262', width=2),
                        symbol='square'
                    ),
                    hovertemplate='<b>Category:</b> %{text}<extra></extra>'
                )
                node_traces.append(category_trace)
            
            # Concepts (small gray nodes)
            concept_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'concept']
            if concept_nodes:
                x_concepts = [pos[node][0] for node in concept_nodes]
                y_concepts = [pos[node][1] for node in concept_nodes]
                
                concept_trace = go.Scatter(
                    x=x_concepts,
                    y=y_concepts,
                    mode='markers+text',
                    name='Concepts',
                    text=concept_nodes,
                    textposition="middle right",
                    textfont=dict(size=8, color='gray'),
                    marker=dict(
                        size=12,
                        color='lightgray',
                        line=dict(color='gray', width=1),
                        symbol='diamond'
                    ),
                    hovertemplate='<b>Concept:</b> %{text}<extra></extra>'
                )
                node_traces.append(concept_trace)
            
            # Create figure
            fig = go.Figure(data=edge_traces + node_traces)
            
            fig.update_layout(
                title="Topics, Categories & Concepts Network",
                showlegend=True,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Network visualization failed: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Network visualization unavailable: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    @staticmethod
    def create_theme_network(themes: List[str], connections: List[Tuple[str, str]]) -> go.Figure:
        """Create theme network visualization"""
        try:
            G = nx.Graph()
            G.add_nodes_from(themes)
            G.add_edges_from(connections)
            
            pos = nx.spring_layout(G)
            
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
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

def show_home():
    """Display home page"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    st.title("🔬 NLP Tool for Youth Participatory Action Research")
    st.markdown("### Empowering Youth Researchers with Advanced Text Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ai_status = "🟢 Enabled" if Config.is_ai_enabled() else "🔴 Disabled"
        st.metric("AI Analysis", ai_status)
        if not Config.is_ai_enabled():
            st.caption("Using traditional NLP")
    
    with col2:
        st.metric("Storage", "🟢 Session Active")
        st.caption("Using local storage")
    
    with col3:
        file_count = len(st.session_state.uploaded_files)
        st.metric("Files Loaded", file_count)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Getting Started:**
        1. 📁 Upload your documents
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
        - PDF, DOCX, TXT, MD
        """)

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
                text, file_id = FileHandler.process_file(uploaded_file)
                
                st.session_state.uploaded_files[file_id] = {
                    'name': uploaded_file.name,
                    'text': text,
                    'size': len(text),
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ {uploaded_file.name} processed successfully")
                
                with st.expander(f"Preview: {uploaded_file.name}"):
                    st.text(text[:500] + "..." if len(text) > 500 else text)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        status_text.text("Processing complete!")
        progress_bar.progress(1.0)
    
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
        
        text_analyzer = TextAnalyzer()
        
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive analysis..."):
                    progress = st.progress(0)
                    
                    progress.progress(0.15)
                    sentiment = text_analyzer.analyze_sentiment(text, selected_file_id)
                    
                    progress.progress(0.30)
                    themes = text_analyzer.extract_themes(text, selected_file_id)
                    
                    progress.progress(0.45)
                    keywords = text_analyzer.extract_keywords(text, selected_file_id)
                    
                    progress.progress(0.60)
                    entities = text_analyzer.extract_entities(text, selected_file_id)
                    
                    progress.progress(0.75)
                    quotes = text_analyzer.extract_quotes(text, selected_file_id)
                    
                    progress.progress(0.80)
                    insights = text_analyzer.generate_insights(text, selected_file_id)
                    
                    # 7. Topics, Categories & Concepts
                    progress.progress(0.95)
                    topics_data = text_analyzer.extract_topics_categories(text, selected_file_id)
                    
                    progress.progress(1.0)
                    
                    st.markdown("### 📋 Complete Analysis Results")
                    
                    # Sentiment Analysis - Full Width
                    st.markdown("---")
                    st.markdown("#### 😊 Sentiment Analysis")
                    with st.container():
                        st.markdown(sentiment)
                    
                    # Themes - Full Width
                    st.markdown("---")
                    st.markdown("#### 🎯 Main Themes")
                    with st.container():
                        st.markdown(themes)
                    
                    # Keywords - Full Width
                    st.markdown("---")
                    st.markdown("#### 🔑 Key Terms & Keywords")
                    with st.container():
                        st.markdown(keywords)
                    
                    # Named Entities - Full Width
                    st.markdown("---")
                    st.markdown("#### 🏷️ Named Entities")
                    with st.container():
                        st.markdown(entities)
                    
                    # Key Quotes - Full Width
                    st.markdown("---")
                    st.markdown("#### 💬 Important Quotes")
                    with st.container():
                        st.markdown(quotes)
                    
                    # Research Insights - Full Width
                    st.markdown("---")
                    st.markdown("#### 💡 Research Insights")
                    with st.container():
                        st.markdown(insights)
                    
                    # Topics, Categories & Concepts - Full Width
                    st.markdown("---")
                    st.markdown("#### 🎯 Topics, Categories & Concepts")
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Topics:**")
                            for topic in topics_data.get('topics', [])[:7]:
                                st.write(f"• {topic}")
                        
                        with col2:
                            st.markdown("**Categories:**")
                            for category in topics_data.get('categories', []):
                                st.write(f"• {category}")
                        
                        with col3:
                            st.markdown("**Key Concepts:**")
                            for concept in topics_data.get('concepts', [])[:10]:
                                st.write(f"• {concept}")
                        
                        # Show network visualization
                        st.markdown("**Topic Network Visualization:**")
                        visualizer = Visualizer()
                        network_fig = visualizer.create_topics_network(topics_data)
                        st.plotly_chart(network_fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.success("✅ Analysis complete!")
        
        with col2:
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
            
            if st.button("🎯 Topics & Concepts", use_container_width=True):
                with st.spinner("Extracting topics and concepts..."):
                    topics_data = text_analyzer.extract_topics_categories(text, selected_file_id)
                    st.markdown("### Topics, Categories & Concepts")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Topics:**")
                        for topic in topics_data.get('topics', [])[:7]:
                            st.write(f"• {topic}")
                        
                        st.markdown("**Categories:**")
                        for category in topics_data.get('categories', []):
                            st.write(f"• {category}")
                    
                    with col2:
                        st.markdown("**Key Concepts:**")
                        for concept in topics_data.get('concepts', [])[:10]:
                            st.write(f"• {concept}")
                    
                    # Show network
                    visualizer = Visualizer()
                    network_fig = visualizer.create_topics_network(topics_data)
                    st.plotly_chart(network_fig, use_container_width=True)
        
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
        
        st.markdown("### ☁️ Word Cloud")
        try:
            fig = visualizer.create_wordcloud(text)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create word cloud: {e}")
        
        st.markdown("### 📈 Sentiment Trend")
        try:
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('. ')
            
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
        
        # Topics & Concepts Network
        st.markdown("### 🕸️ Topics, Categories & Concepts Network")
        try:
            text_analyzer = TextAnalyzer()
            topics_data = text_analyzer.extract_topics_categories(text, selected_file_id)
            
            if topics_data and topics_data.get('topics'):
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Topics", len(topics_data.get('topics', [])))
                with col2:
                    st.metric("Categories", len(topics_data.get('categories', [])))
                with col3:
                    st.metric("Concepts", len(topics_data.get('concepts', [])))
                
                # Create and display network
                fig = visualizer.create_topics_network(topics_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show details in expander
                with st.expander("View Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Topics:**")
                        for topic in topics_data.get('topics', [])[:10]:
                            st.write(f"• {topic}")
                    
                    with col2:
                        st.markdown("**Categories:**")
                        for category in topics_data.get('categories', []):
                            st.write(f"• {category}")
                    
                    with col3:
                        st.markdown("**Concepts:**")
                        for concept in topics_data.get('concepts', [])[:15]:
                            st.write(f"• {concept}")
            else:
                st.info("Run Topics & Concepts analysis first from the Text Analysis page")
        except Exception as e:
            st.error(f"Could not create topics network: {e}")

def show_history():
    """Display analysis history"""
    st.title("📜 Analysis History")
    
    analyses = db_manager.get_all_analyses()
    
    if not analyses:
        st.info("No analysis history available yet")
        return
    
    df_data = []
    for analysis in analyses:
        df_data.append({
            'Timestamp': analysis.get('timestamp', 'N/A'),
            'File': analysis.get('filename', 'Unknown'),
            'Type': analysis.get('analysis_type', 'Unknown'),
            'File ID': analysis.get('file_id', 'N/A')[:8] + '...'
        })
    
    df = pd.DataFrame(df_data)
    
    st.dataframe(df, use_container_width=True)
    
    if st.button("📥 Export History as CSV"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_history.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

def show_settings():
    """Display settings page"""
    st.title("⚙️ Settings")
    
    st.markdown("Configure your analysis environment")
    
    st.markdown("### 🔑 API Configuration")
    
    with st.form("api_settings"):
        openai_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Enter your OpenAI API key for AI-powered analysis"
        )
        
        if st.form_submit_button("Save Settings"):
            if openai_key:
                st.session_state.openai_api_key = openai_key
                st.success("✅ OpenAI API key saved")
            
            st.rerun()
    
    st.markdown("### 📊 Current Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**OpenAI Status:**")
        if Config.is_ai_enabled():
            st.success("✅ Connected - AI features enabled")
        else:
            st.warning("⚠️ Not configured - Using traditional NLP")
    
    with col2:
        st.markdown("**Storage Status:**")
        st.success("✅ Session storage active")
    
    st.markdown("### 💻 System Information")
    
    info_data = {
        "Python Version": "3.8+",
        "Streamlit Version": st.__version__,
        "AI Available": "Yes" if OPENAI_AVAILABLE else "No",
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
    init_session_state()
    
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🔬 YPAR Tool")
        st.markdown("---")
        
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
        
        st.markdown("### 📊 Quick Stats")
        st.metric("Files", len(st.session_state.uploaded_files))
        st.metric("Analyses", len(st.session_state.get('analysis_results', [])))
        
        st.markdown("---")
        mode = "AI Enhanced" if Config.is_ai_enabled() else "Traditional NLP"
        st.info(f"Mode: {mode}")
    
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