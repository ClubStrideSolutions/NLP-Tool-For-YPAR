"""
NLP Tool for YPAR - Improved Version
Main application file with enhanced error handling, performance, and modularity
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

# Import custom modules
from config import Config
from utils import (
    sanitize_input, validate_file, generate_file_id,
    chunk_text, format_results, cache_key_generator,
    safe_json_parse, display_progress, get_word_statistics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data packages"""
    packages = ['punkt', 'stopwords', 'vader_lexicon', 'maxent_ne_chunker', 'words', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK package {package}: {e}")

download_nltk_data()

# Page configuration
config = Config.get_app_config()
st.set_page_config(
    page_title=config['page_title'],
    page_icon=config['page_icon'],
    layout=config['layout'],
    initial_sidebar_state=config['initial_sidebar_state']
)

# Custom CSS with theme colors
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
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# MongoDB connection with improved error handling
class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
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
        """Store analysis results"""
        if not self.connected:
            return self._store_local(file_id, analysis_type, results)
        
        try:
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
            logger.error(f"Error storing analysis: {e}")
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
        """Retrieve analysis history"""
        if not self.connected:
            return self._get_local_history(file_id)
        
        try:
            collection = self.db["analysis_results"]
            return list(collection.find({"file_id": file_id}).sort("timestamp", -1))
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return self._get_local_history(file_id)
    
    def _get_local_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Get history from session state"""
        if 'analysis_results' in st.session_state:
            return [r for r in st.session_state.analysis_results if r['file_id'] == file_id]
        return []

# Initialize database manager
@st.cache_resource
def get_db_manager():
    """Get cached database manager instance"""
    return DatabaseManager()

db_manager = get_db_manager()

# Enhanced text analysis functions
class TextAnalyzer:
    """Text analysis functionality"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
    
    @st.cache_data(ttl=config['cache_ttl'])
    def analyze_themes(_self, text: str, file_id: str) -> str:
        """Analyze themes using LDA topic modeling with improved accuracy"""
        try:
            text = sanitize_input(text)
            if not text:
                return "No text provided for analysis"
            
            # Enhanced preprocessing
            word_tokens = word_tokenize(text.lower())
            filtered_text = [
                w for w in word_tokens 
                if w.isalnum() and w not in _self.stop_words and len(w) > 2
            ]
            
            if len(filtered_text) < 10:
                return "Insufficient text for theme analysis"
            
            # Create document-term matrix with better parameters
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=0.01,
                max_df=0.95
            )
            
            doc_term_matrix = vectorizer.fit_transform([' '.join(filtered_text)])
            
            # Apply LDA with optimized parameters
            n_topics = min(10, len(filtered_text) // 20 + 1)
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
                top_words = [feature_names[i] for i in top_words_idx]
                theme_strength = topic[top_words_idx].sum()
                themes_list.append({
                    'theme': f"Theme {topic_idx + 1}",
                    'words': top_words[:8],
                    'strength': float(theme_strength)
                })
            
            # Use YAKE for keyword extraction
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=15,
                features=None
            )
            keywords = kw_extractor.extract_keywords(text)
            
            # Format results
            themes_text = "## Identified Themes\n\n"
            for theme in sorted(themes_list, key=lambda x: x['strength'], reverse=True):
                themes_text += f"### {theme['theme']} (Strength: {theme['strength']:.2f})\n"
                themes_text += f"**Key Terms**: {', '.join(theme['words'])}\n\n"
            
            themes_text += "\n## Key Concepts\n\n"
            for kw, score in keywords[:10]:
                themes_text += f"- **{kw}** (relevance: {1/score:.2f})\n"
            
            # Add statistics
            stats = get_word_statistics(text)
            themes_text += f"\n## Text Statistics\n"
            themes_text += f"- Words: {stats['word_count']}\n"
            themes_text += f"- Sentences: {stats['sentence_count']}\n"
            themes_text += f"- Vocabulary Richness: {stats['vocabulary_richness']:.2%}\n"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="theme_analysis",
                results={"themes": themes_text, "raw_themes": themes_list}
            )
            
            return themes_text
            
        except Exception as e:
            logger.error(f"Error analyzing themes: {e}")
            return f"Error analyzing themes: {str(e)}"
    
    @st.cache_data(ttl=config['cache_ttl'])
    def extract_quotes(_self, text: str, file_id: str) -> str:
        """Extract significant quotes with improved scoring"""
        try:
            text = sanitize_input(text)
            if not text:
                return "No text provided for analysis"
            
            sentences = sent_tokenize(text)
            if not sentences:
                return "No sentences found in text"
            
            scored_sentences = []
            
            for sent in sentences:
                score = 0
                sent_clean = sent.strip()
                
                # Skip very short or very long sentences
                word_count = len(sent_clean.split())
                if word_count < 5 or word_count > 100:
                    continue
                
                # Length score
                if 10 <= word_count <= 50:
                    score += 3
                
                # Quotation marks indicate direct quotes
                if '"' in sent or "'" in sent or '"' in sent:
                    score += 5
                
                # Sentiment analysis
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
                    'challenge', 'opportunity', 'experience', 'understand',
                    'realize', 'discover', 'learn', 'impact', 'change'
                ]
                
                sent_lower = sent.lower()
                for phrase in key_phrases:
                    if phrase in sent_lower:
                        score += 1
                
                # Question or exclamation
                if sent_clean.endswith('?'):
                    score += 2
                if sent_clean.endswith('!'):
                    score += 1
                
                if score > 0:
                    scored_sentences.append((sent_clean, score))
            
            # Sort and select top quotes
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_quotes = scored_sentences[:min(15, len(scored_sentences))]
            
            if not top_quotes:
                return "No significant quotes found"
            
            quotes_text = "## Significant Quotes\n\n"
            
            for i, (quote, score) in enumerate(top_quotes, 1):
                # Analyze sentiment
                try:
                    blob = TextBlob(quote)
                    sentiment = "positive" if blob.sentiment.polarity > 0.1 else \
                               "negative" if blob.sentiment.polarity < -0.1 else "neutral"
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                except:
                    sentiment = "neutral"
                    polarity = 0
                    subjectivity = 0
                
                quotes_text += f"### Quote {i}\n"
                quotes_text += f"> {quote}\n\n"
                quotes_text += f"**Sentiment**: {sentiment} (polarity: {polarity:.2f}, subjectivity: {subjectivity:.2f})\n"
                quotes_text += f"**Relevance Score**: {score}/20\n\n"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="quote_extraction",
                results={"quotes": quotes_text, "raw_quotes": top_quotes}
            )
            
            return quotes_text
            
        except Exception as e:
            logger.error(f"Error extracting quotes: {e}")
            return f"Error extracting quotes: {str(e)}"
    
    @st.cache_data(ttl=config['cache_ttl'])
    def generate_insights(_self, text: str, file_id: str) -> str:
        """Generate insights with enhanced analysis"""
        try:
            text = sanitize_input(text)
            if not text:
                return "No text provided for analysis"
            
            # Get basic statistics
            stats = get_word_statistics(text)
            sentences = sent_tokenize(text)
            
            # Word frequency analysis
            words = word_tokenize(text.lower())
            filtered_words = [w for w in words if w.isalnum() and len(w) > 3 and w not in _self.stop_words]
            word_freq = Counter(filtered_words)
            common_words = word_freq.most_common(15)
            
            # Sentiment analysis
            try:
                blob = TextBlob(text)
                overall_sentiment = blob.sentiment
                
                # Sentence-level sentiment
                sentiments = []
                for sent in sentences[:100]:  # Limit to first 100 sentences for performance
                    try:
                        sent_blob = TextBlob(sent)
                        sentiments.append(sent_blob.sentiment.polarity)
                    except:
                        continue
                
                sentiment_variance = np.var(sentiments) if sentiments else 0
                sentiment_trend = "stable"
                
                if len(sentiments) > 10:
                    first_half = np.mean(sentiments[:len(sentiments)//2])
                    second_half = np.mean(sentiments[len(sentiments)//2:])
                    if second_half - first_half > 0.2:
                        sentiment_trend = "increasingly positive"
                    elif first_half - second_half > 0.2:
                        sentiment_trend = "increasingly negative"
            except:
                overall_sentiment = None
                sentiment_variance = 0
                sentiment_trend = "unknown"
            
            # Build insights
            insights_text = "## Key Insights\n\n"
            
            # Text characteristics
            insights_text += "### Text Characteristics\n"
            insights_text += f"- **Total words**: {stats['word_count']:,}\n"
            insights_text += f"- **Total sentences**: {stats['sentence_count']:,}\n"
            insights_text += f"- **Average sentence length**: {stats['word_count']/max(stats['sentence_count'], 1):.1f} words\n"
            insights_text += f"- **Vocabulary richness**: {stats['vocabulary_richness']:.2%}\n"
            insights_text += f"- **Unique words**: {stats['unique_words']:,}\n\n"
            
            # Sentiment insights
            if overall_sentiment:
                insights_text += "### Sentiment Analysis\n"
                sentiment_label = 'Positive' if overall_sentiment.polarity > 0.1 else \
                                 'Negative' if overall_sentiment.polarity < -0.1 else 'Neutral'
                insights_text += f"- **Overall sentiment**: {sentiment_label}\n"
                insights_text += f"- **Sentiment strength**: {abs(overall_sentiment.polarity):.2f}\n"
                insights_text += f"- **Sentiment consistency**: {'High' if sentiment_variance < 0.1 else 'Variable'} (variance: {sentiment_variance:.3f})\n"
                insights_text += f"- **Sentiment trend**: {sentiment_trend}\n"
                insights_text += f"- **Subjectivity**: {overall_sentiment.subjectivity:.2f} ({'Objective' if overall_sentiment.subjectivity < 0.4 else 'Subjective'})\n\n"
            
            # Key terms
            insights_text += "### Most Frequent Terms\n"
            for word, count in common_words[:10]:
                frequency_pct = (count / len(filtered_words)) * 100 if filtered_words else 0
                insights_text += f"- **{word}**: {count} occurrences ({frequency_pct:.1f}%)\n"
            
            # Pattern observations
            insights_text += "\n### Patterns Observed\n"
            
            patterns = {
                'Contrasting viewpoints': ['however', 'but', 'although', 'despite', 'nevertheless'],
                'Causal relationships': ['therefore', 'thus', 'consequently', 'because', 'since'],
                'Questioning/exploratory': ['?', 'how', 'why', 'what if', 'could'],
                'Certainty expressions': ['definitely', 'certainly', 'surely', 'obviously', 'clearly'],
                'Uncertainty expressions': ['maybe', 'perhaps', 'possibly', 'might', 'could be']
            }
            
            text_lower = text.lower()
            found_patterns = []
            
            for pattern_name, indicators in patterns.items():
                if any(indicator in text_lower for indicator in indicators):
                    count = sum(text_lower.count(ind) for ind in indicators)
                    found_patterns.append(f"- **{pattern_name}** detected ({count} indicators)")
            
            if found_patterns:
                insights_text += "\n".join(found_patterns)
            else:
                insights_text += "- No significant rhetorical patterns detected"
            
            insights_text += "\n\n### Recommendations\n"
            
            # Generate recommendations based on analysis
            recommendations = []
            
            if stats['vocabulary_richness'] < 0.3:
                recommendations.append("- Consider expanding vocabulary diversity")
            
            if stats['word_count'] / max(stats['sentence_count'], 1) > 25:
                recommendations.append("- Consider breaking down complex sentences for clarity")
            
            if sentiment_variance > 0.3:
                recommendations.append("- Note the varying emotional tone throughout the text")
            
            if len(common_words) > 0 and common_words[0][1] / len(filtered_words) > 0.05:
                recommendations.append(f"- The term '{common_words[0][0]}' appears very frequently - ensure intentional repetition")
            
            if recommendations:
                insights_text += "\n".join(recommendations)
            else:
                insights_text += "- Text appears well-balanced in structure and style"
            
            # Store results
            db_manager.store_analysis(
                file_id=file_id,
                analysis_type="insight_generation",
                results={"insights": insights_text, "statistics": stats}
            )
            
            return insights_text
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"Error generating insights: {str(e)}"

# Initialize text analyzer
text_analyzer = TextAnalyzer()

# TF-IDF based embeddings
@st.cache_data(ttl=config['cache_ttl'])
def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate text embedding using TF-IDF"""
    try:
        text = sanitize_input(text)
        if not text:
            return None
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        embedding = vectorizer.fit_transform([text]).toarray()[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# File processing with improved error handling
@st.cache_data(ttl=config['cache_ttl'])
def process_file(file_content: bytes, file_name: str, file_type: str) -> Optional[Tuple[str, str, str]]:
    """Process uploaded file with validation"""
    try:
        content = ""
        
        if file_type == "text/plain":
            content = file_content.decode("utf-8", errors='ignore')
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(file_content))
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(BytesIO(file_content))
            content = df.to_string()
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Sanitize content
        content = sanitize_input(content)
        
        if not content or len(content.strip()) < 10:
            st.error("File appears to be empty or contains insufficient text")
            return None
        
        # Generate unique identifier
        file_id = generate_file_id(content, file_name)
        unique_file_name = f"{file_name}_{file_id[:8]}"
        
        # Get embedding
        embedding = get_text_embedding(content)
        if embedding is None:
            logger.warning("Could not generate embedding for file")
        
        return content, file_id, unique_file_name
        
    except Exception as e:
        st.error(f"Error processing file {file_name}: {str(e)}")
        logger.error(f"File processing error: {e}")
        return None

# Batch file processing with progress tracking
def batch_process_files(files: List[Any]) -> List[Dict[str, Any]]:
    """Process multiple files with progress tracking"""
    results = []
    progress = display_progress("Processing files", len(files))
    
    with ThreadPoolExecutor(max_workers=config['batch_size']) as executor:
        futures = {}
        
        for file in files:
            # Validate file first
            is_valid, msg = validate_file(file, config['max_file_size'])
            if not is_valid:
                st.warning(f"Skipping {file.name}: {msg}")
                continue
            
            # Read file content
            file_content = file.read()
            future = executor.submit(process_file, file_content, file.name, file.type)
            futures[future] = file.name
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
                    st.success(f"✅ Processed: {futures[future]}")
            except Exception as e:
                st.error(f"❌ Failed to process {futures[future]}: {str(e)}")
            progress(i)
    
    return results

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
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
        'analysis_results': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Main application
def main():
    """Main application logic"""
    
    # Title
    st.markdown('<h1 class="main-title">NLP Tool for YPAR</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
    
    pages = [
        "Home", "Data Upload", "Text Processing", "Topic Modeling",
        "Quote Extraction", "Insight Generation", "Visualization",
        "Named Entity Recognition", "Intent Detection", "Ethics & Bias",
        "Analysis History"
    ]
    
    page = st.sidebar.radio("Go to", pages, label_visibility="collapsed")
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_upload_page()
    elif page == "Text Processing":
        show_processing_page()
    elif page == "Topic Modeling":
        show_topic_modeling_page()
    elif page == "Quote Extraction":
        show_quote_extraction_page()
    elif page == "Insight Generation":
        show_insight_generation_page()
    elif page == "Visualization":
        show_visualization_page()
    elif page == "Named Entity Recognition":
        show_ner_page()
    elif page == "Intent Detection":
        show_intent_detection_page()
    elif page == "Ethics & Bias":
        show_ethics_page()
    elif page == "Analysis History":
        show_history_page()
    
    # Footer
    show_footer()

def show_home_page():
    """Display home page"""
    st.markdown('<h2 class="section-title">Welcome to the NLP Tool for YPAR</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Advanced Analysis</h3>
            <p>Powerful NLP tools for analyzing qualitative data with educational context and insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Quote Extraction</h3>
            <p>Identify and analyze representative quotes with context and significance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Visualization</h3>
            <p>Interactive visualizations to understand and communicate your findings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⚖️ Ethics & Bias</h3>
            <p>Comprehensive analysis of potential biases and ethical considerations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>About This Tool</h3>
        <p>This interactive tool helps you analyze and process text data for Youth Participatory Action Research (YPAR) projects.</p>
        <p>Features include:</p>
        <ul>
            <li>Advanced text processing and analysis</li>
            <li>Topic modeling with educational context</li>
            <li>Quote extraction and analysis</li>
            <li>Insight generation and pattern recognition</li>
            <li>Interactive visualizations</li>
            <li>Ethics and bias analysis</li>
        </ul>
        <p><strong>Version:</strong> 2.0 (Improved)</p>
        <p><strong>Status:</strong> {'✅ Database Connected' if db_manager.connected else '⚠️ Using Local Storage'}</p>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    """Display data upload page"""
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
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=config['supported_file_types'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    results = batch_process_files(new_files)
                    
                    for content, file_id, unique_name in results:
                        st.session_state.processed_data.append(content)
                        st.session_state.file_names.append(unique_name)
                        st.session_state.file_ids.append(file_id)
                        st.session_state.processed_files.add(new_files[0].name)
                    
                    if results:
                        st.success(f"✅ Successfully processed {len(results)} file(s)!")
                        st.balloons()
        else:
            st.info("All selected files have already been processed.")
        
        # Display processed files
        if st.session_state.file_names:
            st.markdown("""
            <div class="result-box">
                <h3>📋 Uploaded Files</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (name, content) in enumerate(zip(st.session_state.file_names, st.session_state.processed_data)):
                with st.expander(f"📄 {name.split('_')[0]}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        stats = get_word_statistics(content)
                        st.write(f"**Words:** {stats['word_count']:,}")
                        st.write(f"**Sentences:** {stats['sentence_count']:,}")
                    
                    with col2:
                        if st.button(f"View", key=f"view_{i}"):
                            st.text_area("Content", content[:1000] + "...", height=200)
                    
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
            <p>This section prepares your text for analysis through cleaning, standardization, and pattern identification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        text = st.session_state.processed_data[file_index]
        stats = get_word_statistics(text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", f"{stats['word_count']:,}")
        with col2:
            st.metric("Unique Words", f"{stats['unique_words']:,}")
        with col3:
            st.metric("Vocabulary Richness", f"{stats['vocabulary_richness']:.1%}")
        
        with st.expander("View Text Sample"):
            st.text_area("Text", text[:2000] + "...", height=200)
        
        if st.button("Generate Text Embedding", type="primary"):
            with st.spinner("Processing..."):
                embedding = get_text_embedding(text)
                if embedding:
                    st.success("✅ Text processed successfully!")
                    st.write(f"Embedding dimension: {len(embedding)}")

def show_topic_modeling_page():
    """Display topic modeling page"""
    st.markdown('<h2 class="section-title">Topic Modeling</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Topic Modeling</h3>
        <p>Topic modeling identifies key themes and patterns using LDA (Latent Dirichlet Allocation) and keyword extraction.</p>
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
                themes = text_analyzer.analyze_themes(text, file_id)
                if themes:
                    st.session_state.themes[selected_file] = themes
                    st.markdown(themes)

def show_quote_extraction_page():
    """Display quote extraction page"""
    st.markdown('<h2 class="section-title">Quote Extraction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Quote Extraction</h3>
        <p>This feature identifies significant quotes using sentence scoring and sentiment analysis.</p>
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
                quotes = text_analyzer.extract_quotes(text, file_id)
                if quotes:
                    st.session_state.quotes[selected_file] = quotes
                    st.markdown(quotes)

def show_insight_generation_page():
    """Display insight generation page"""
    st.markdown('<h2 class="section-title">Insight Generation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.warning("Please upload files first")
        return
    
    st.markdown("""
    <div class="info-box">
        <h3>About Insight Generation</h3>
        <p>Generate insights using statistical analysis, pattern recognition, and sentiment analysis.</p>
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
                insights = text_analyzer.generate_insights(text, file_id)
                if insights:
                    st.session_state.insights[selected_file] = insights
                    st.markdown(insights)

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
    
    if viz_option != "Theme Network":
        selected_file = st.selectbox(
            "Select a file to visualize",
            st.session_state.file_names + (["All Files"] if len(st.session_state.file_names) > 1 else []),
            index=st.session_state.current_file_index
        )
        
        if selected_file == "All Files":
            text_to_visualize = "\n".join(st.session_state.processed_data)
        else:
            file_index = st.session_state.file_names.index(selected_file)
            text_to_visualize = st.session_state.processed_data[file_index]
    
    if viz_option == "Word Cloud":
        st.subheader("Word Cloud Visualization")
        
        # Word cloud parameters
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
                        max_words=max_words,
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate(text_to_visualize)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
    
    elif viz_option == "Theme Network":
        st.subheader("Theme Network Analysis")
        
        if not st.session_state.themes:
            st.warning("Please analyze themes first")
            return
        
        # Build theme network
        st.info("Building theme network from analyzed files...")
        # Implementation would go here

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
                    # Tokenize and tag
                    tokens = word_tokenize(text[:5000])  # Limit for performance
                    pos_tags = pos_tag(tokens)
                    
                    # Chunk named entities
                    chunks = ne_chunk(pos_tags, binary=False)
                    
                    # Extract entities
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
                    # Define intent patterns
                    intent_patterns = {
                        'Question': r'\?|who|what|when|where|why|how',
                        'Request': r'please|could|would|can you|will you',
                        'Opinion': r'think|believe|feel|seems|appears',
                        'Suggestion': r'should|could|might|recommend|suggest',
                        'Complaint': r'problem|issue|wrong|bad|terrible'
                    }
                    
                    # Detect intents
                    detected_intents = []
                    sentences = sent_tokenize(text[:5000])  # Limit for performance
                    
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
                        
                        # Intent distribution
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
                    # Define bias indicators
                    bias_indicators = {
                        'Gender': ['he', 'she', 'man', 'woman', 'male', 'female', 'boy', 'girl'],
                        'Age': ['young', 'old', 'elderly', 'youth', 'child', 'adult', 'teen'],
                        'Cultural': ['race', 'ethnicity', 'culture', 'tradition', 'minority', 'majority'],
                        'Socioeconomic': ['poor', 'rich', 'wealthy', 'poverty', 'affluent', 'disadvantaged']
                    }
                    
                    # Count occurrences
                    words = word_tokenize(text.lower())
                    bias_counts = {}
                    
                    for category, indicators in bias_indicators.items():
                        count = sum(1 for word in words if word in indicators)
                        bias_counts[category] = count
                    
                    # Display results
                    st.subheader("Potential Bias Indicators")
                    
                    df = pd.DataFrame(list(bias_counts.items()), columns=['Category', 'Count'])
                    fig = px.bar(df, x='Category', y='Count', title='Bias Indicator Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
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

def show_footer():
    """Display footer"""
    st.markdown("""
    <div style="text-align: center; margin-top: 2em; padding: 1em; background-color: #eaf2f8; border-radius: 10px; border-top: 2px solid #2874a6;">
        <p style="color: #2c3e50; font-weight: bold;">© 2024 NLP Tool for YPAR | Open Source NLP | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()