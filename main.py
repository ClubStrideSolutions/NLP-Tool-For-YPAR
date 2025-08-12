import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
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
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from pymongo import MongoClient
from datetime import datetime
import bson
from bson import ObjectId
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any
import uuid
from transformers import pipeline
import yake

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Load environment variables
load_dotenv()

# Configure MongoDB connection
def get_mongodb_connection_string():
    """Get MongoDB connection string from environment or Streamlit secrets"""
    return st.secrets.get("mongodb_connection_string") or os.getenv("CONNECTION_STRING")

# Set page configuration with custom theme
st.set_page_config(
    page_title="NLP Tool for YPAR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with improved contrast
st.markdown("""
    <style>
    /* Main styling */
    .main-title {
        font-size: 2.8em;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1em;
        font-weight: bold;
    }
    .section-title {
        font-size: 2em;
        color: #154360;
        margin-top: 1em;
        margin-bottom: 0.5em;
        border-bottom: 3px solid #2874a6;
        padding-bottom: 0.3em;
    }
    .info-box {
        background-color: #eaf2f8;
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2874a6;
        color: #2c3e50;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        border: 1px solid #bdc3c7;
        color: #2c3e50;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background-color: #2874a6;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #154360;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 1.8em;
        color: #1a5276;
        margin-bottom: 1em;
        text-align: center;
    }
    .stProgress > div > div > div {
        background-color: #2874a6;
    }
    .stFileUploader {
        border: 2px dashed #2874a6;
        border-radius: 10px;
        padding: 2em;
        background-color: #f8f9fa;
    }
    .result-box {
        background-color: #eaf2f8;
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        border-left: 5px solid #27ae60;
        color: #2c3e50;
    }
    .viz-container {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1em 0;
        border: 1px solid #bdc3c7;
    }
    .stMarkdown {
        color: #2c3e50;
    }
    .stRadio > label {
        color: #2c3e50;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# MongoDB connection handling
def get_mongo_client():
    """Get a MongoDB client connection"""
    try:
        connection_string = get_mongodb_connection_string()
        if connection_string:
            return MongoClient(connection_string)
        return None
    except Exception as e:
        st.warning(f"MongoDB connection not available: {str(e)}. Using local storage.")
        return None

def get_db():
    """Get the database instance"""
    try:
        client = get_mongo_client()
        if client is not None:
            return client["nlp_tool"]
        return None
    except Exception as e:
        return None

def store_analysis_results(file_id: str, analysis_type: str, results: Dict[str, Any]) -> str:
    """Store analysis results in MongoDB or session state"""
    try:
        db = get_db()
        if db is not None:
            collection = db["analysis_results"]
            document = {
                "file_id": file_id,
                "analysis_type": analysis_type,
                "results": results,
                "timestamp": datetime.now()
            }
            result = collection.insert_one(document)
            if result.inserted_id:
                return str(result.inserted_id)
    except:
        pass
    
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

def get_analysis_history(file_id: str) -> List[Dict[str, Any]]:
    """Retrieve analysis history for a file"""
    try:
        db = get_db()
        if db is not None:
            collection = db["analysis_results"]
            return list(collection.find({"file_id": file_id}).sort("timestamp", -1))
    except:
        pass
    
    # Fallback to session state
    if 'analysis_results' in st.session_state:
        return [r for r in st.session_state.analysis_results if r['file_id'] == file_id]
    return []

# Text analysis functions using open-source alternatives
@st.cache_data(ttl=3600)
def analyze_themes(text: str, file_id: str) -> str:
    """Analyze themes using LDA topic modeling"""
    try:
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_text = [w for w in word_tokens if w.isalnum() and w not in stop_words]
        
        # Create document-term matrix
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform([' '.join(filtered_text)])
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract themes
        themes_list = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            themes_list.append(f"**Theme {topic_idx + 1}**: {', '.join(top_words[:5])}")
        
        # Use YAKE for keyword extraction
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.7, top=10)
        keywords = kw_extractor.extract_keywords(text)
        
        themes_text = "## Identified Themes\n\n"
        themes_text += "\n\n".join(themes_list)
        themes_text += "\n\n## Key Concepts\n\n"
        for kw, score in keywords:
            themes_text += f"- **{kw}** (relevance: {1/score:.2f})\n"
        
        # Store results
        store_analysis_results(
            file_id=file_id,
            analysis_type="theme_analysis",
            results={"themes": themes_text}
        )
        
        return themes_text
    except Exception as e:
        return f"Error analyzing themes: {str(e)}"

@st.cache_data(ttl=3600)
def extract_quotes(text: str, file_id: str) -> str:
    """Extract significant quotes using pattern matching and sentence scoring"""
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Score sentences based on various criteria
        scored_sentences = []
        for sent in sentences:
            score = 0
            # Length criteria
            if 10 < len(sent.split()) < 50:
                score += 2
            # Contains quotation marks
            if '"' in sent or "'" in sent:
                score += 3
            # Contains strong verbs or emotional words
            blob = TextBlob(sent)
            if abs(blob.sentiment.polarity) > 0.3:
                score += 2
            # Contains key phrases
            key_phrases = ['believe', 'think', 'feel', 'important', 'significant', 'challenge', 'opportunity']
            for phrase in key_phrases:
                if phrase.lower() in sent.lower():
                    score += 1
            
            scored_sentences.append((sent, score))
        
        # Sort by score and select top quotes
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_quotes = scored_sentences[:10]
        
        quotes_text = "## Significant Quotes\n\n"
        for i, (quote, score) in enumerate(top_quotes, 1):
            # Analyze sentiment
            blob = TextBlob(quote)
            sentiment = "positive" if blob.sentiment.polarity > 0 else "negative" if blob.sentiment.polarity < 0 else "neutral"
            
            quotes_text += f"### Quote {i}\n"
            quotes_text += f"> {quote}\n\n"
            quotes_text += f"**Sentiment**: {sentiment} (polarity: {blob.sentiment.polarity:.2f})\n"
            quotes_text += f"**Relevance Score**: {score}\n\n"
        
        # Store results
        store_analysis_results(
            file_id=file_id,
            analysis_type="quote_extraction",
            results={"quotes": quotes_text}
        )
        
        return quotes_text
    except Exception as e:
        return f"Error extracting quotes: {str(e)}"

@st.cache_data(ttl=3600)
def generate_insights(text: str, file_id: str) -> str:
    """Generate insights using statistical analysis and pattern recognition"""
    try:
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Word frequency analysis
        word_freq = Counter([w for w in words if w.isalnum() and len(w) > 3])
        common_words = word_freq.most_common(10)
        
        # Sentiment analysis
        blob = TextBlob(text)
        overall_sentiment = blob.sentiment
        
        # Sentence-level sentiment
        sentiments = [TextBlob(sent).sentiment.polarity for sent in sentences]
        sentiment_variance = np.var(sentiments) if sentiments else 0
        
        # Readability metrics
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
        
        insights_text = "## Key Insights\n\n"
        
        # Text characteristics
        insights_text += "### Text Characteristics\n"
        insights_text += f"- Total words: {len(words)}\n"
        insights_text += f"- Total sentences: {len(sentences)}\n"
        insights_text += f"- Average sentence length: {avg_sentence_length:.1f} words\n"
        insights_text += f"- Vocabulary richness: {len(set(words))/len(words):.2%}\n\n"
        
        # Sentiment insights
        insights_text += "### Sentiment Analysis\n"
        insights_text += f"- Overall sentiment: {'Positive' if overall_sentiment.polarity > 0 else 'Negative' if overall_sentiment.polarity < 0 else 'Neutral'}\n"
        insights_text += f"- Sentiment strength: {abs(overall_sentiment.polarity):.2f}\n"
        insights_text += f"- Sentiment consistency: {'High' if sentiment_variance < 0.1 else 'Variable'}\n"
        insights_text += f"- Subjectivity: {overall_sentiment.subjectivity:.2f}\n\n"
        
        # Key terms
        insights_text += "### Most Frequent Terms\n"
        for word, count in common_words:
            insights_text += f"- {word}: {count} occurrences\n"
        
        # Pattern observations
        insights_text += "\n### Patterns Observed\n"
        if any(word in text.lower() for word in ['however', 'but', 'although', 'despite']):
            insights_text += "- Text contains contrasting viewpoints\n"
        if any(word in text.lower() for word in ['therefore', 'thus', 'consequently', 'because']):
            insights_text += "- Text contains causal relationships\n"
        if text.count('?') > 2:
            insights_text += "- Text is questioning/exploratory in nature\n"
        
        # Store results
        store_analysis_results(
            file_id=file_id,
            analysis_type="insight_generation",
            results={"insights": insights_text}
        )
        
        return insights_text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# TF-IDF based embedding as replacement for OpenAI embeddings
@st.cache_data(ttl=3600)
def get_text_embedding(text: str) -> List[float]:
    """Generate text embedding using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(max_features=100)
        # Fit on a single document
        embedding = vectorizer.fit_transform([text]).toarray()[0]
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

# Main title with enhanced styling
st.markdown('<h1 class="main-title">NLP Tool for YPAR</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Upload", "Text Processing", "Topic Modeling", "Quote Extraction", 
     "Insight Generation", "Visualization", "Named Entity Recognition", 
     "Intent Detection", "Ethics & Bias", "Analysis History"],
    label_visibility="collapsed"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'themes' not in st.session_state:
    st.session_state.themes = {}
if 'sentiments' not in st.session_state:
    st.session_state.sentiments = {}
if 'quotes' not in st.session_state:
    st.session_state.quotes = {}
if 'insights' not in st.session_state:
    st.session_state.insights = {}
if 'file_ids' not in st.session_state:
    st.session_state.file_ids = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Process file function
@st.cache_data(ttl=3600)
def process_file(file):
    try:
        if file.type == "text/plain":
            content = file.read().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(file.read()))
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(BytesIO(file.read()))
            content = df.to_string()
        else:
            st.error(f"Unsupported file type: {file.type}")
            return None
        
        # Generate unique identifier
        unique_id = str(uuid.uuid4())[:8]
        unique_file_name = f"{file.name}_{unique_id}"
        
        # Get embedding
        embedding = get_text_embedding(content)
        if embedding is None:
            return None
        
        # Store locally or in MongoDB
        file_id = unique_id
        
        return content, file_id, unique_file_name
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)}")
        return None

# Batch process files
def batch_process_files(files: List[Any]) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in futures:
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    return results

# Progress tracking
def track_progress(operation: str, total: int):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update(completed: int):
        progress = completed / total
        progress_bar.progress(progress)
        status_text.text(f"{operation}: {completed}/{total} completed")
    
    return update

# Main content area
if page == "Home":
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
    </div>
    """, unsafe_allow_html=True)

elif page == "Data Upload":
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
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'docx', 'xlsx'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
        
        if new_files:
            with st.spinner("Processing files..."):
                update_progress = track_progress("Processing files", len(new_files))
                
                results = batch_process_files(new_files)
                
                new_contents = []
                new_names = []
                new_ids = []
                
                for i, result in enumerate(results):
                    if result:
                        content, file_id, unique_file_name = result
                        new_contents.append(content)
                        new_names.append(unique_file_name)
                        new_ids.append(file_id)
                        st.session_state.processed_files.add(new_files[i].name)
                    update_progress(i + 1)
                
                if new_contents:
                    st.session_state.processed_data.extend(new_contents)
                    st.session_state.file_names.extend(new_names)
                    st.session_state.file_ids.extend(new_ids)
                    st.success(f"✅ Successfully processed {len(new_contents)} new file(s)!")
        else:
            st.info("All selected files have already been processed.")
        
        if st.session_state.file_names:
            st.markdown("""
            <div class="result-box">
                <h3>📋 Uploaded Files</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (name, content) in enumerate(zip(st.session_state.file_names, st.session_state.processed_data)):
                original_name = name.split('_')[0]
                
                st.markdown(f"""
                <div class="feature-card" style="margin-bottom: 1em;">
                    <h4 style="margin: 0; color: #2c3e50;">{original_name}</h4>
                    <p style="margin: 0.5em 0; color: #7f8c8d; font-size: 0.9em;">
                        {len(content.split())} words
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"View Content", key=f"view_{i}"):
                        st.session_state.current_file_index = i
                        st.text_area(
                            f"Content of {original_name}",
                            content,
                            height=200,
                            key=f"content_{i}"
                        )
                with col2:
                    if st.button(f"Remove File", key=f"remove_{i}"):
                        try:
                            st.session_state.processed_data.pop(i)
                            st.session_state.file_names.pop(i)
                            st.session_state.file_ids.pop(i)
                            st.session_state.processed_files.remove(original_name)
                            st.success(f"File {original_name} removed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing file: {str(e)}")

elif page == "Text Processing":
    st.markdown('<h2 class="section-title">Text Processing</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
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
                <p>This section helps you prepare your text for analysis by:</p>
                <ul>
                    <li>Cleaning and standardizing the text</li>
                    <li>Identifying key patterns and structures</li>
                    <li>Preparing for deeper analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader(f"Processing: {selected_file}")
            st.text_area(
                "Text",
                st.session_state.processed_data[file_index],
                height=200
            )
            
            if st.button("Process Text"):
                with st.spinner("Processing text..."):
                    embedding = get_text_embedding(st.session_state.processed_data[file_index])
                    if embedding:
                        st.success("Text processed successfully!")
                        st.write("Embedding dimension:", len(embedding))

elif page == "Topic Modeling":
    st.markdown('<h2 class="section-title">Topic Modeling</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Topic Modeling</h3>
            <p>Topic modeling helps identify key themes and patterns in your text data using LDA (Latent Dirichlet Allocation).</p>
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
            
            st.subheader(f"Selected File: {selected_file}")
            st.text_area("File Content", text, height=200)
            
            if st.button("Analyze Themes"):
                with st.spinner("Analyzing themes..."):
                    try:
                        themes = analyze_themes(text, file_id)
                        if themes:
                            st.session_state.themes[selected_file] = themes
                            st.subheader("Identified Themes")
                            st.markdown(themes)
                    except Exception as e:
                        st.error(f"Error analyzing themes: {str(e)}")

elif page == "Quote Extraction":
    st.markdown('<h2 class="section-title">Quote Extraction</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Quote Extraction</h3>
            <p>This feature helps identify significant quotes using sentence scoring and sentiment analysis.</p>
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
            
            st.subheader(f"Selected File: {selected_file}")
            st.text_area("File Content", text, height=200)
            
            if st.button("Extract Quotes"):
                with st.spinner("Extracting quotes..."):
                    try:
                        quotes = extract_quotes(text, file_id)
                        if quotes:
                            st.session_state.quotes[selected_file] = quotes
                            st.subheader("Extracted Quotes")
                            st.markdown(quotes)
                    except Exception as e:
                        st.error(f"Error extracting quotes: {str(e)}")

elif page == "Insight Generation":
    st.markdown('<h2 class="section-title">Insight Generation</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Insight Generation</h3>
            <p>This feature uses statistical analysis and pattern recognition to generate insights.</p>
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
            
            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    insights = generate_insights(text, file_id)
                    if insights:
                        st.session_state.insights[selected_file] = insights
                        st.subheader("Generated Insights")
                        st.markdown(insights)

elif page == "Visualization":
    st.markdown('<h2 class="section-title">Data Visualization</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        viz_option = st.radio(
            "Select visualization type",
            ["Individual File", "Combined Analysis", "Theme Network"],
            horizontal=True
        )
        
        if viz_option == "Individual File":
            selected_file = st.selectbox(
                "Select a file to visualize",
                st.session_state.file_names,
                index=st.session_state.current_file_index
            )
            if selected_file:
                file_index = st.session_state.file_names.index(selected_file)
                text_to_visualize = st.session_state.processed_data[file_index]
        else:
            text_to_visualize = "\n".join(st.session_state.processed_data)
        
        if viz_option != "Theme Network":
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader("Word Cloud")
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(text_to_visualize)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Named Entity Recognition":
    st.markdown('<h2 class="section-title">Named Entity Recognition</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Named Entity Recognition</h3>
            <p>This feature uses NLTK to identify named entities in your text.</p>
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
            
            if st.button("Analyze Entities"):
                with st.spinner("Processing entities..."):
                    try:
                        # Tokenize and tag
                        tokens = word_tokenize(text)
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
                            st.info("No named entities found in the text.")
                            
                    except Exception as e:
                        st.error(f"Error processing entities: {str(e)}")

elif page == "Intent Detection":
    st.markdown('<h2 class="section-title">Intent Detection</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Intent Detection</h3>
            <p>This feature uses pattern matching to detect common intents in text.</p>
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
            
            if st.button("Detect Intent"):
                with st.spinner("Analyzing intent..."):
                    try:
                        # Define intent patterns
                        intent_patterns = {
                            'Question': r'\?|who|what|when|where|why|how',
                            'Request': r'please|could|would|can you|will you',
                            'Opinion': r'think|believe|feel|seems|appears',
                            'Suggestion': r'should|could|might|recommend|suggest',
                            'Complaint': r'problem|issue|wrong|bad|terrible|awful'
                        }
                        
                        # Detect intents
                        detected_intents = []
                        sentences = sent_tokenize(text)
                        
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
                            st.info("No specific intents detected.")
                        
                    except Exception as e:
                        st.error(f"Error detecting intent: {str(e)}")

elif page == "Ethics & Bias":
    st.markdown('<h2 class="section-title">Ethics & Bias Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Ethics & Bias Analysis</h3>
            <p>This feature helps identify potential biases using word frequency and pattern analysis.</p>
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
            
            if st.button("Check for Potential Biases"):
                with st.spinner("Analyzing for potential biases..."):
                    try:
                        # Define bias indicators
                        bias_indicators = {
                            'Gender': ['he', 'she', 'man', 'woman', 'male', 'female', 'boy', 'girl'],
                            'Age': ['young', 'old', 'elderly', 'youth', 'child', 'adult'],
                            'Cultural': ['race', 'ethnicity', 'culture', 'tradition', 'minority', 'majority']
                        }
                        
                        # Count occurrences
                        words = word_tokenize(text.lower())
                        bias_counts = {}
                        
                        for category, indicators in bias_indicators.items():
                            count = sum(1 for word in words if word in indicators)
                            bias_counts[category] = count
                        
                        st.subheader("Potential Bias Indicators")
                        
                        # Display findings
                        for category, count in bias_counts.items():
                            st.write(f"**{category}-related terms**: {count} occurrences")
                        
                        # Recommendations
                        st.markdown("""
                        ### Recommendations:
                        - Review the text for balanced representation
                        - Consider multiple perspectives
                        - Use inclusive language
                        - Be aware of cultural context
                        - Ensure fair representation of all groups
                        """)
                        
                    except Exception as e:
                        st.error(f"Error analyzing biases: {str(e)}")

elif page == "Analysis History":
    st.markdown('<h2 class="section-title">Analysis History</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        selected_file = st.selectbox(
            "Select a file to view history",
            st.session_state.file_names
        )
        
        if selected_file:
            file_index = st.session_state.file_names.index(selected_file)
            file_id = st.session_state.file_ids[file_index]
            
            history = get_analysis_history(file_id)
            if history:
                for entry in history:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>{entry['analysis_type']}</h3>
                        <p>Timestamp: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <div class="result-box">
                            {entry['results']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No analysis history found for this file.")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 2em; padding: 1em; background-color: #eaf2f8; border-radius: 10px; border-top: 2px solid #2874a6;">
        <p style="color: #2c3e50; font-weight: bold;">© 2024 NLP Tool for YPAR | Open Source NLP</p>
    </div>
""", unsafe_allow_html=True)