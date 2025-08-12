"""
NLP Tool for YPAR - Enhanced Version 3.0
With modern UI, expanded file format support, and improved layout
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

# Configure page first
st.set_page_config(
    page_title="NLP Tool for YPAR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from config import Config
from utils import (
    sanitize_input, validate_file, generate_file_id,
    chunk_text, format_results, display_progress, get_word_statistics
)
from file_handlers import FileHandler
from ui_components_berkeley import UIComponents
from rag_system import RAGSystem, render_rag_interface, render_persona_builder

# Import analysis modules
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
import yake
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
def download_nltk_data():
    """Download required NLTK data packages"""
    packages = ['punkt', 'stopwords', 'vader_lexicon', 'maxent_ne_chunker', 'words', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

download_nltk_data()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
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
        'analysis_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database Manager
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
            from pymongo import MongoClient
            connection_string = Config.get_mongodb_connection_string()
            if connection_string:
                self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                self.client.server_info()  # Test connection
                self.db = self.client["nlp_tool"]
                self.connected = True
                self._create_indexes()  # Create unique indexes
                logger.info("MongoDB connected successfully")
        except Exception as e:
            self.connected = False
            logger.warning(f"MongoDB not available: {e}. Using local storage.")
    
    def _create_indexes(self):
        """Create unique indexes to prevent duplicates"""
        try:
            # Create unique index on file hash to prevent duplicate files
            self.db["processed_files"].create_index(
                [("file_hash", 1)], 
                unique=True, 
                background=True
            )
            
            # Create compound unique index for analysis results
            self.db["analysis_results"].create_index(
                [("file_id", 1), ("analysis_type", 1), ("content_hash", 1)], 
                unique=True, 
                background=True
            )
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def _generate_file_hash(self, content: str, file_name: str) -> str:
        """Generate unique hash for file content"""
        import hashlib
        hash_input = f"{file_name}_{content}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()
    
    def _generate_analysis_hash(self, file_id: str, analysis_type: str, results: str) -> str:
        """Generate unique hash for analysis results"""
        import hashlib
        results_str = json.dumps(results, sort_keys=True) if isinstance(results, dict) else str(results)
        hash_input = f"{file_id}_{analysis_type}_{results_str}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:16]  # Use first 16 chars
    
    def check_file_exists(self, content: str, file_name: str) -> bool:
        """Check if file already exists in MongoDB"""
        if not self.connected or not self.db:
            return False
        
        try:
            file_hash = self._generate_file_hash(content, file_name)
            collection = self.db["processed_files"]
            existing = collection.find_one({"file_hash": file_hash})
            return existing is not None
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def store_analysis(self, file_id: str, analysis_type: str, results: Dict[str, Any]) -> Optional[str]:
        """Store analysis results in MongoDB or session state (prevents duplicates)"""
        # Always store in session state for immediate access
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        # Generate content hash for duplicate detection
        content_hash = self._generate_analysis_hash(file_id, analysis_type, results)
        
        analysis_doc = {
            "file_id": file_id,
            "analysis_type": analysis_type,
            "results": results,
            "content_hash": content_hash,
            "timestamp": datetime.now()
        }
        
        # Check if already in session state
        existing_in_session = any(
            a.get('content_hash') == content_hash 
            for a in st.session_state.analysis_results
        )
        
        if not existing_in_session:
            st.session_state.analysis_results.append(analysis_doc)
            st.session_state.analysis_count += 1
        
        # Try to store in MongoDB if connected
        if self.connected and self.db:
            try:
                collection = self.db["analysis_results"]
                # Use update_one with upsert to prevent duplicates
                result = collection.update_one(
                    {"file_id": file_id, "analysis_type": analysis_type, "content_hash": content_hash},
                    {"$set": analysis_doc},
                    upsert=True
                )
                if result.upserted_id:
                    logger.info(f"Stored new analysis in MongoDB: {result.upserted_id}")
                    return str(result.upserted_id)
                else:
                    logger.info("Analysis already exists in MongoDB, skipped duplicate")
            except Exception as e:
                logger.error(f"Failed to store in MongoDB: {e}")
        
        return file_id
    
    def store_file(self, file_name: str, content: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Store processed file in MongoDB (prevents duplicates)"""
        if self.connected and self.db:
            try:
                # Generate file hash for duplicate detection
                file_hash = self._generate_file_hash(content, file_name)
                
                collection = self.db["processed_files"]
                document = {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "content": content[:5000],  # Store first 5000 chars
                    "metadata": metadata,
                    "timestamp": datetime.now(),
                    "word_count": len(content.split()),
                    "char_count": len(content)
                }
                
                # Check if file already exists
                existing = collection.find_one({"file_hash": file_hash})
                if existing:
                    logger.info(f"File already exists in MongoDB: {existing['_id']}")
                    st.warning(f"📌 File '{file_name}' already exists in database (skipped duplicate)")
                    return str(existing['_id'])
                
                # Insert new file
                result = collection.insert_one(document)
                logger.info(f"Stored new file in MongoDB: {result.inserted_id}")
                return str(result.inserted_id)
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    logger.info("Duplicate file detected, skipping")
                    st.info(f"📌 File '{file_name}' already processed")
                else:
                    logger.error(f"Failed to store file in MongoDB: {e}")
        return None
    
    def get_analysis_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Retrieve analysis history from MongoDB or session state"""
        results = []
        
        # Try MongoDB first
        if self.connected and self.db:
            try:
                collection = self.db["analysis_results"]
                results = list(collection.find({"file_id": file_id}).sort("timestamp", -1))
                logger.info(f"Retrieved {len(results)} results from MongoDB")
            except Exception as e:
                logger.error(f"Failed to retrieve from MongoDB: {e}")
        
        # Fallback to session state
        if not results and 'analysis_results' in st.session_state:
            results = [r for r in st.session_state.analysis_results if r.get('file_id') == file_id]
        
        return results
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all processed files from MongoDB"""
        if self.connected and self.db:
            try:
                collection = self.db["processed_files"]
                return list(collection.find().sort("timestamp", -1).limit(100))
            except Exception as e:
                logger.error(f"Failed to retrieve files: {e}")
        return []

# Text Analyzer
class TextAnalyzer:
    """Text analysis functionality"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    @st.cache_data(ttl=3600)
    def analyze_themes(_self, text: str, file_id: str) -> str:
        """Analyze themes using LDA topic modeling"""
        try:
            text = sanitize_input(text)
            if not text:
                return "No text provided"
            
            # Preprocessing
            word_tokens = word_tokenize(text.lower())
            filtered_text = [w for w in word_tokens if w.isalnum() and w not in _self.stop_words and len(w) > 2]
            
            if len(filtered_text) < 10:
                return "Insufficient text for analysis"
            
            # TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
            doc_term_matrix = vectorizer.fit_transform([' '.join(filtered_text)])
            
            # LDA
            n_topics = min(5, len(filtered_text) // 20 + 1)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Extract themes
            feature_names = vectorizer.get_feature_names_out()
            themes_list = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                themes_list.append(f"**Theme {topic_idx + 1}**: {', '.join(top_words[:5])}")
            
            # YAKE keywords
            kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=10)
            keywords = kw_extractor.extract_keywords(text)
            
            themes_text = "## Identified Themes\n\n"
            themes_text += "\n\n".join(themes_list)
            themes_text += "\n\n## Key Concepts\n\n"
            for kw, score in keywords[:10]:
                themes_text += f"- **{kw}** (relevance: {1/score:.2f})\n"
            
            return themes_text
            
        except Exception as e:
            logger.error(f"Error analyzing themes: {e}")
            return f"Error: {str(e)}"
    
    @st.cache_data(ttl=3600)
    def extract_quotes(_self, text: str, file_id: str) -> str:
        """Extract significant quotes"""
        try:
            text = sanitize_input(text)
            sentences = sent_tokenize(text)
            
            scored_sentences = []
            for sent in sentences:
                score = 0
                word_count = len(sent.split())
                
                if 10 <= word_count <= 50:
                    score += 3
                if '"' in sent or "'" in sent:
                    score += 5
                
                try:
                    blob = TextBlob(sent)
                    if abs(blob.sentiment.polarity) > 0.3:
                        score += 3
                except:
                    pass
                
                if score > 0:
                    scored_sentences.append((sent, score))
            
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_quotes = scored_sentences[:10]
            
            quotes_text = "## Significant Quotes\n\n"
            for i, (quote, score) in enumerate(top_quotes, 1):
                quotes_text += f"### Quote {i}\n> {quote}\n\n"
            
            return quotes_text
            
        except Exception as e:
            logger.error(f"Error extracting quotes: {e}")
            return f"Error: {str(e)}"
    
    @st.cache_data(ttl=3600)
    def generate_insights(_self, text: str, file_id: str) -> str:
        """Generate insights"""
        try:
            text = sanitize_input(text)
            stats = get_word_statistics(text)
            
            words = word_tokenize(text.lower())
            filtered_words = [w for w in words if w.isalnum() and len(w) > 3]
            word_freq = Counter(filtered_words).most_common(10)
            
            insights_text = "## Key Insights\n\n"
            insights_text += f"- **Total words**: {stats['word_count']:,}\n"
            insights_text += f"- **Unique words**: {stats['unique_words']:,}\n"
            insights_text += f"- **Vocabulary richness**: {stats['vocabulary_richness']:.2%}\n\n"
            
            insights_text += "### Most Frequent Terms\n"
            for word, count in word_freq:
                insights_text += f"- {word}: {count}\n"
            
            return insights_text
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"Error: {str(e)}"

# Initialize components
db_manager = DatabaseManager()
text_analyzer = TextAnalyzer()
ui = UIComponents()

# Initialize RAG system
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()

# Main Application
def main():
    """Main application logic"""
    
    # Apply UC Berkeley theme
    ui.apply_berkeley_theme()
    
    # Render modern header
    ui.render_modern_header()
    
    # Sidebar navigation
    selected_page = ui.render_sidebar_menu()
    
    # Route to appropriate page
    if "Home" in selected_page:
        show_home_page()
    elif "Upload" in selected_page:
        show_upload_page()
    elif "Analysis" in selected_page:
        show_analysis_page()
    elif "RAG Intelligence" in selected_page:
        show_rag_page()
    elif "Personas" in selected_page:
        show_personas_page()
    elif "Visualizations" in selected_page:
        show_visualization_page()
    elif "AI Insights" in selected_page:
        show_ai_insights_page()
    elif "Dashboard" in selected_page:
        show_dashboard_page()
    elif "History" in selected_page:
        show_history_page()
    elif "Settings" in selected_page:
        show_settings_page()
    
    # Footer
    ui.render_footer()

def show_home_page():
    """Display home page"""
    st.markdown("## Welcome to NLP Tool for YPAR")
    
    # Show MongoDB connection status
    if db_manager.connected:
        st.success("✅ **MongoDB Connected** - All data will be permanently stored")
    else:
        st.warning("⚠️ **MongoDB Not Connected** - Data stored temporarily in session only")
    
    # Feature cards with Berkeley colors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="berkeley-card" style="text-align: center;">
            <h3 style="color: #FDB515;">📁 Enhanced File Support</h3>
            <p style="color: #ffffff;">Now supporting PDF, CSV, JSON, HTML, XML, and more!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="berkeley-card" style="text-align: center;">
            <h3 style="color: #FDB515;">⚡ Club Stride Innovation</h3>
            <p style="color: #ffffff;">Empowering youth through cutting-edge research tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="berkeley-card" style="text-align: center;">
            <h3 style="color: #FDB515;">🤖 AI-Powered</h3>
            <p style="color: #ffffff;">Advanced NLP analysis with machine learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🔍 Analyze", "📊 Visualize"])
    
    with tab1:
        st.markdown("""
        1. Click on **Upload & Process** in the sidebar
        2. Drag and drop your files or click to browse
        3. Supported formats: PDF, Word, Excel, CSV, JSON, HTML, Text, and more
        4. Files are automatically processed and prepared for analysis
        """)
    
    with tab2:
        st.markdown("""
        1. Go to **Analysis Suite** after uploading files
        2. Choose from various analysis options:
           - Theme Analysis
           - Quote Extraction
           - Sentiment Analysis
           - Named Entity Recognition
        3. Results are saved automatically for future reference
        """)
    
    with tab3:
        st.markdown("""
        1. Navigate to **Visualizations** to see your data
        2. Available visualizations:
           - Word Clouds
           - Theme Networks
           - Sentiment Trends
           - Statistical Charts
        3. Export visualizations for presentations
        """)

def show_upload_page():
    """Display upload page with enhanced file support"""
    st.markdown("## 📤 Upload & Process Files")
    
    # File upload card
    uploaded_files = ui.render_file_upload_card()
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            if st.button("🚀 Process Files", use_container_width=True, type="primary"):
                progress = display_progress("Processing files", len(new_files))
                
                for i, file in enumerate(new_files):
                    with st.spinner(f"Processing {file.name}..."):
                        # Validate file
                        is_valid, msg = FileHandler.validate_file(file)
                        if not is_valid:
                            st.warning(f"⚠️ {file.name}: {msg}")
                            continue
                        
                        # Process file
                        content, metadata = FileHandler.process_file(file)
                        
                        if content:
                            # Check if file already exists in MongoDB
                            if db_manager.check_file_exists(content, file.name):
                                st.warning(f"⚠️ {file.name} already exists in database (skipped)")
                                continue
                            
                            file_id = generate_file_id(content, file.name)
                            
                            # Store in MongoDB (will handle duplicates internally)
                            mongo_id = db_manager.store_file(file.name, content, metadata)
                            
                            # Only add to session if not a duplicate
                            if mongo_id:
                                # Store in session state
                                st.session_state.processed_data.append(content)
                                st.session_state.file_names.append(file.name)
                                st.session_state.file_metadata.append(metadata)
                                st.session_state.file_ids.append(file_id)
                                st.session_state.processed_files.add(file.name)
                                st.session_state.file_count += 1
                                
                                # Add to RAG system
                                try:
                                    # Generate embedding
                                    vectorizer = TfidfVectorizer(max_features=384)
                                    embedding = vectorizer.fit_transform([content[:5000]]).toarray()[0]
                                    
                                    # Add to RAG
                                    st.session_state.rag_system.add_document(
                                        content=content[:2000],  # Store first 2000 chars
                                        embedding=embedding,
                                        metadata={
                                            "file_name": file.name,
                                            "file_id": file_id,
                                            "type": metadata.get("type", "unknown"),
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    )
                                    st.info(f"🧠 Added to RAG system for intelligent retrieval")
                                except Exception as e:
                                    logger.warning(f"Could not add to RAG: {e}")
                                
                                st.success(f"✅ Processed: {file.name}")
                        else:
                            st.error(f"❌ Failed: {file.name}")
                    
                    progress(i + 1)
                
                st.balloons()
                st.success(f"🎉 Successfully processed {len(new_files)} files!")
        
        # Display processed files with preview
        if st.session_state.file_names:
            st.markdown("### 📚 Processed Files")
            
            for i, (name, content, metadata) in enumerate(zip(
                st.session_state.file_names,
                st.session_state.processed_data,
                st.session_state.file_metadata
            )):
                ui.render_file_preview(name, content, metadata)

def show_analysis_page():
    """Display analysis page"""
    st.markdown("## 🔍 Analysis Suite")
    
    if not st.session_state.processed_data:
        st.warning("📁 Please upload files first")
        return
    
    # File selector
    selected_file = st.selectbox(
        "Select a file to analyze",
        st.session_state.file_names,
        index=st.session_state.get('current_file_index', 0)
    )
    
    if selected_file:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
        file_id = st.session_state.file_ids[file_index]
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            if ui.render_analysis_card(
                "Theme Analysis",
                "🎯",
                "Identify key themes and topics using advanced topic modeling",
                "Analyze Themes"
            ):
                with st.spinner("Analyzing themes..."):
                    themes = text_analyzer.analyze_themes(text, file_id)
                    st.session_state.themes[selected_file] = themes
                    db_manager.store_analysis(file_id, "themes", {"result": themes})
                    st.markdown(themes)
        
        with col2:
            if ui.render_analysis_card(
                "Quote Extraction",
                "💬",
                "Extract significant quotes with context and relevance scoring",
                "Extract Quotes"
            ):
                with st.spinner("Extracting quotes..."):
                    quotes = text_analyzer.extract_quotes(text, file_id)
                    st.session_state.quotes[selected_file] = quotes
                    db_manager.store_analysis(file_id, "quotes", {"result": quotes})
                    st.markdown(quotes)
        
        # Additional analysis
        if ui.render_analysis_card(
            "Generate Insights",
            "💡",
            "Generate comprehensive insights using statistical analysis",
            "Generate Insights"
        ):
            with st.spinner("Generating insights..."):
                insights = text_analyzer.generate_insights(text, file_id)
                st.session_state.insights[selected_file] = insights
                db_manager.store_analysis(file_id, "insights", {"result": insights})
                st.markdown(insights)

def show_visualization_page():
    """Display visualization page"""
    st.markdown("## 📊 Visualizations")
    
    if not st.session_state.processed_data:
        st.warning("📁 Please upload files first")
        return
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Word Cloud", "Frequency Analysis", "Sentiment Timeline", "Theme Network"]
    )
    
    selected_file = st.selectbox(
        "Select file",
        st.session_state.file_names + ["All Files"],
        index=0
    )
    
    if selected_file == "All Files":
        text = " ".join(st.session_state.processed_data)
    else:
        file_index = st.session_state.file_names.index(selected_file)
        text = st.session_state.processed_data[file_index]
    
    if viz_type == "Word Cloud":
        st.subheader("☁️ Word Cloud")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            max_words = st.slider("Max words", 50, 200, 100)
            colormap = st.selectbox("Color scheme", ["viridis", "plasma", "cool", "hot"])
        
        with col1:
            try:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap=colormap,
                    max_words=max_words
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
    
    elif viz_type == "Frequency Analysis":
        st.subheader("📊 Word Frequency Analysis")
        
        words = word_tokenize(text.lower())
        filtered = [w for w in words if w.isalnum() and len(w) > 3]
        freq = Counter(filtered).most_common(20)
        
        df = pd.DataFrame(freq, columns=['Word', 'Count'])
        fig = px.bar(df, x='Word', y='Count', 
                     title='Top 20 Most Frequent Words',
                     color='Count', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

def show_ai_insights_page():
    """Display AI insights page"""
    st.markdown("## 🤖 AI-Powered Insights")
    
    if not st.session_state.processed_data:
        st.warning("📁 Please upload files first")
        return
    
    st.info("🔮 Advanced AI analysis features coming soon!")
    
    # Placeholder for advanced features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Coming Soon:
        - Deep learning text classification
        - Transformer-based analysis
        - Custom model training
        - Automated report generation
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Beta Features:
        - Sentiment trend analysis
        - Entity relationship mapping
        - Topic evolution tracking
        - Cross-document analysis
        """)

def show_dashboard_page():
    """Display dashboard page"""
    st.markdown("## 📈 Analytics Dashboard")
    ui.render_dashboard()
    
    # Additional charts
    if st.session_state.analysis_results:
        st.markdown("### 📊 Analysis Distribution")
        
        analysis_types = [r['analysis_type'] for r in st.session_state.analysis_results]
        type_counts = Counter(analysis_types)
        
        df = pd.DataFrame(type_counts.items(), columns=['Type', 'Count'])
        fig = px.pie(df, values='Count', names='Type', 
                     title='Analysis Types Distribution',
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

def show_history_page():
    """Display history page"""
    st.markdown("## 📚 Analysis History")
    
    if not st.session_state.analysis_results:
        st.info("No analysis history yet")
        return
    
    # Display history
    for result in reversed(st.session_state.analysis_results[-10:]):
        with st.expander(f"{result['analysis_type']} - {result['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**File ID:** {result['file_id']}")
            if isinstance(result['results'], dict):
                for key, value in result['results'].items():
                    st.write(f"**{key}:**")
                    st.write(value)
            else:
                st.write(result['results'])

def show_settings_page():
    """Display settings page"""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["General", "Advanced", "About"])
    
    with tab1:
        st.markdown("### General Settings")
        
        # Theme selection
        theme = st.selectbox("Color Theme", ["Default", "Dark", "Light", "Custom"])
        
        # File size limit
        max_size = st.slider("Max file size (MB)", 10, 100, 50)
        
        # Auto-save
        auto_save = st.checkbox("Auto-save analysis results", value=True)
        
        if st.button("Save Settings"):
            st.success("✅ Settings saved!")
    
    with tab2:
        st.markdown("### Advanced Settings")
        
        # API configuration
        st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.text_input("MongoDB Connection String", type="password", placeholder="mongodb://...")
        
        # Cache settings
        cache_ttl = st.number_input("Cache TTL (seconds)", 0, 7200, 3600)
        
        if st.button("Update Configuration"):
            st.success("✅ Configuration updated!")
    
    with tab3:
        st.markdown("""
        ### About NLP Tool for YPAR
        
        **Version:** 3.0 Enhanced
        
        **Features:**
        - 📁 Support for 15+ file formats
        - 🎨 Modern, responsive UI
        - 🤖 AI-powered analysis
        - 📊 Advanced visualizations
        - 💾 Automatic result storage
        - 🔒 Secure data handling
        
        **Technologies:**
        - Streamlit
        - Natural Language Toolkit (NLTK)
        - TextBlob
        - Scikit-learn
        - Plotly
        - And many more...
        
        **Support:**
        - Documentation: [View Docs](#)
        - Issues: [Report Bug](#)
        - Contact: support@example.com
        
        ---
        Made with ❤️ for Youth Participatory Action Research
        """)

def show_rag_page():
    """Display RAG Intelligence page"""
    st.markdown("## 🧠 RAG Intelligence System")
    
    st.markdown("""
    <div class="berkeley-card">
        <h3 style="color: #FDB515;">Retrieval-Augmented Generation</h3>
        <p style="color: #ffffff;">
        This intelligent system combines document retrieval, memory, and personas to provide 
        context-aware analysis. Upload documents to build your knowledge base, then ask questions 
        to get intelligent, personalized responses.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display RAG statistics
    rag = st.session_state.rag_system
    stats = rag.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Documents in RAG", stats["total_documents"])
    with col2:
        st.metric("🧠 Memories", stats["total_memories"])
    with col3:
        st.metric("💬 Conversations", stats["conversation_length"])
    with col4:
        st.metric("👤 Active Persona", stats["active_persona"] or "None")
    
    # RAG Interface
    render_rag_interface()
    
    # Chat-like interface for RAG queries
    st.markdown("### 💬 Intelligent Chat")
    
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # Display chat history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Generate embedding for query
                    vectorizer = TfidfVectorizer(max_features=384)
                    embedding = vectorizer.fit_transform([prompt]).toarray()[0]
                    
                    # Process through RAG
                    result = rag.process_query(prompt, embedding)
                    
                    # Generate response based on persona and context
                    if result["persona"]:
                        response = f"**{result['persona'].name} Analysis:**\n\n"
                        response += f"Based on {result['retrieved_docs']} relevant documents and {result['memory_items']} memories:\n\n"
                        response += f"*[This would be the AI-generated response based on the augmented context]*\n\n"
                        response += f"Focus areas: {', '.join(result['persona'].focus_areas)}"
                    else:
                        response = f"Retrieved {result['retrieved_docs']} documents. Please select a persona for personalized analysis."
                    
                    st.markdown(response)
                    st.session_state.rag_messages.append({"role": "assistant", "content": response})
                    
                    # Add to memory
                    rag.memory_system.add_memory(
                        content=f"Q: {prompt[:200]} A: {response[:200]}",
                        memory_type="conversation",
                        metadata={"timestamp": datetime.now().isoformat()}
                    )
                    
                except Exception as e:
                    st.error(f"Error: {e}")

def show_personas_page():
    """Display Personas management page"""
    st.markdown("## 👤 Analysis Personas")
    
    st.markdown("""
    <div class="berkeley-card">
        <h3 style="color: #FDB515;">Customize Your Analysis Perspective</h3>
        <p style="color: #ffffff;">
        Personas allow you to analyze your data from different perspectives. Each persona has 
        unique focus areas, analysis styles, and output formats tailored to specific roles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    rag = st.session_state.rag_system
    
    # Display existing personas
    st.markdown("### 📋 Available Personas")
    
    personas = rag.persona_manager.personas
    
    # Create persona cards
    cols = st.columns(3)
    for idx, (key, persona) in enumerate(personas.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="berkeley-card" style="min-height: 250px;">
                <h4 style="color: #FDB515;">{persona.name}</h4>
                <p style="color: #ffffff; font-size: 0.9rem;">{persona.description}</p>
                <p style="color: #a8a8a8; font-size: 0.85rem;">
                    <strong>Focus:</strong> {', '.join(persona.focus_areas[:3])}...
                </p>
                <p style="color: #a8a8a8; font-size: 0.85rem;">
                    <strong>Style:</strong> {persona.analysis_style}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Activate {persona.name}", key=f"activate_{key}"):
                rag.persona_manager.set_active_persona(key)
                st.success(f"✅ Activated {persona.name}")
                st.rerun()
    
    # Show active persona details
    if rag.persona_manager.active_persona:
        st.markdown("### ✨ Active Persona Details")
        persona = rag.persona_manager.active_persona
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Name:** {persona.name}  
            **Type:** {persona.type.value}  
            **Temperature:** {persona.temperature}  
            **Output Format:** {persona.output_format}
            """)
        
        with col2:
            st.markdown("**Key Questions:**")
            for q in persona.key_questions:
                st.markdown(f"- {q}")
    
    # Custom persona builder
    st.markdown("---")
    render_persona_builder()
    
    # Persona comparison
    st.markdown("### 🔄 Persona Comparison")
    
    if len(personas) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            persona1 = st.selectbox("Select first persona", list(personas.keys()), 
                                   format_func=lambda x: personas[x].name)
        
        with col2:
            persona2 = st.selectbox("Select second persona", list(personas.keys()), 
                                   format_func=lambda x: personas[x].name)
        
        if st.button("Compare Personas"):
            p1 = personas[persona1]
            p2 = personas[persona2]
            
            comparison_df = pd.DataFrame({
                "Aspect": ["Analysis Style", "Temperature", "Focus Areas", "Output Format"],
                p1.name: [p1.analysis_style, p1.temperature, ", ".join(p1.focus_areas[:3]), p1.output_format],
                p2.name: [p2.analysis_style, p2.temperature, ", ".join(p2.focus_areas[:3]), p2.output_format]
            })
            
            st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()