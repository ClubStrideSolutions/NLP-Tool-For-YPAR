import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import openai
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

# Load environment variables
load_dotenv()

# Configure API keys and connections
def get_openai_api_key():
    """Get OpenAI API key from environment or Streamlit secrets"""
    return st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

def get_mongodb_connection_string():
    """Get MongoDB connection string from environment or Streamlit secrets"""
    return st.secrets.get("mongodb_connection_string") or os.getenv("CONNECTION_STRING")

# Initialize OpenAI client with proper error handling
try:
    openai.api_key = get_openai_api_key()
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set it in your environment variables or Streamlit secrets.")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")

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
        color: #1a5276;  /* Darker blue for better contrast */
        text-align: center;
        margin-bottom: 1em;
        font-weight: bold;
    }
    .section-title {
        font-size: 2em;
        color: #154360;  /* Even darker blue */
        margin-top: 1em;
        margin-bottom: 0.5em;
        border-bottom: 3px solid #2874a6;
        padding-bottom: 0.3em;
    }
    .info-box {
        background-color: #eaf2f8;  /* Lighter background */
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2874a6;
        color: #2c3e50;  /* Dark text for better readability */
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 15px;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        border: 1px solid #bdc3c7;
        color: #2c3e50;  /* Dark text */
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    /* Button styling */
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
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 1.8em;
        color: #1a5276;
        margin-bottom: 1em;
        text-align: center;
    }
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #2874a6;
    }
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #2874a6;
        border-radius: 10px;
        padding: 2em;
        background-color: #f8f9fa;
    }
    /* Results styling */
    .result-box {
        background-color: #eaf2f8;
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        border-left: 5px solid #27ae60;
        color: #2c3e50;
    }
    /* Visualization container */
    .viz-container {
        background-color: #ffffff;
        padding: 1.5em;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1em 0;
        border: 1px solid #bdc3c7;
    }
    /* Text styling */
    .stMarkdown {
        color: #2c3e50;
    }
    /* Radio button styling */
    .stRadio > label {
        color: #2c3e50;
        font-weight: bold;
    }
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Update MongoDB connection handling
def get_mongo_client():
    """Get a MongoDB client connection"""
    try:
        connection_string = get_mongodb_connection_string()
        if not connection_string:
            st.error("MongoDB connection string not found. Please set it in your environment variables or Streamlit secrets.")
            return None
            
        return MongoClient(connection_string)
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {str(e)}")
        return None

def get_db():
    """Get the database instance"""
    try:
        client = get_mongo_client()
        if client is not None:
            return client["nlp_tool"]
        return None
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")
        return None

def store_analysis_results(file_id: str, analysis_type: str, results: Dict[str, Any]) -> str:
    """Store analysis results in MongoDB"""
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
            else:
                st.error("Failed to insert analysis results")
                return None
    except Exception as e:
        st.error(f"Error storing analysis results: {str(e)}")
        return None

def get_analysis_history(file_id: str) -> List[Dict[str, Any]]:
    """Retrieve analysis history for a file"""
    try:
        db = get_db()
        if db is not None:
            collection = db["analysis_results"]
            return list(collection.find({"file_id": file_id}).sort("timestamp", -1))
    except Exception as e:
        st.error(f"Error retrieving analysis history: {str(e)}")
        return []

def store_processed_data(file_name: str, content: str, embeddings: List[float], analysis_results: Dict[str, Any]) -> str:
    """Store processed data in MongoDB"""
    try:
        db = get_db()
        if db is not None:
            collection = db["processed_data"]
            
            document = {
                "file_name": file_name,
                "original_name": file_name.split('_')[0],  # Store original name without unique ID
                "content": content,
                "embeddings": embeddings,
                "analysis_results": analysis_results,
                "timestamp": datetime.now(),
                "metadata": {
                    "word_count": len(content.split()),
                    "processed": True
                }
            }
            
            result = collection.insert_one(document)
            if result.inserted_id:
                return str(result.inserted_id)
            else:
                st.error("Failed to insert processed data")
                return None
    except Exception as e:
        st.error(f"Error storing data: {str(e)}")
        return None

def get_processed_data(file_name: str) -> Dict[str, Any]:
    """Retrieve processed data from MongoDB"""
    try:
        db = get_db()
        if db is not None:
            collection = db["processed_data"]
            return collection.find_one({"file_name": file_name})
    except Exception as e:
        st.error(f"Error retrieving data: {str(e)}")
        return None

# Update the analyze_themes function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def analyze_themes(text: str, file_id: str) -> str:
    """Analyze themes in the text using GPT-4"""
    try:
        prompt = f"""Analyze the following text and identify the main themes and topics. 
        For each theme, provide:
        1. The theme name
        2. A brief description
        3. Key supporting points or evidence
        4. The significance of this theme
        
        Format the response in a clear, structured way with bullet points.
        
        Text to analyze:
        {text}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies and analyzes themes in text. Return a well-structured analysis with clear themes and supporting evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        themes = response.choices[0].message.content
        
        # Store the analysis results
        store_analysis_results(
            file_id=file_id,
            analysis_type="theme_analysis",
            results={"themes": themes}
        )
        
        return themes
    except Exception as e:
        st.error(f"Error analyzing themes: {str(e)}")
        return None

# Update the extract_quotes function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_quotes(text: str, file_id: str) -> str:
    """Extract significant quotes from the text using GPT-4"""
    try:
        prompt = f"""Analyze the following text and extract the most significant quotes. 
        For each quote, provide:
        1. The exact quote
        2. The context in which it appears
        3. Why it's significant
        4. Any relevant themes it relates to
        
        Format the response in a clear, structured way with bullet points.
        
        Text to analyze:
        {text}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts and analyzes significant quotes from text. Return a well-structured analysis with clear quotes and their context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        quotes = response.choices[0].message.content
        
        # Store the analysis results
        store_analysis_results(
            file_id=file_id,
            analysis_type="quote_extraction",
            results={"quotes": quotes}
        )
        
        return quotes
    except Exception as e:
        st.error(f"Error extracting quotes: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_insights(text: str, file_id: str) -> str:
    """Generate insights from the text using GPT-4"""
    try:
        prompt = f"""Analyze the following text and generate key insights. 
        For each insight, provide:
        1. The main finding or observation
        2. Supporting evidence from the text
        3. Implications or significance
        4. Potential applications or next steps
        
        Format the response in a clear, structured way with bullet points.
        
        Text to analyze:
        {text}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates meaningful insights from text. Return a well-structured analysis with clear insights and their implications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=2000   # Limit token usage
        )
        
        insights = response.choices[0].message.content
        
        # Store the analysis results
        store_analysis_results(
            file_id=file_id,
            analysis_type="insight_generation",
            results={"insights": insights}
        )
        
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None 
# Main title with enhanced styling
st.markdown('<h1 class="main-title">NLP Tool for YPAR</h1>', unsafe_allow_html=True)

# Sidebar for navigation with enhanced styling
st.sidebar.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Upload", "Text Processing", "Topic Modeling", "Quote Extraction", 
     "Insight Generation", "Visualization", "Named Entity Recognition", 
      "Intent Detection", "Ethics & Bias", "Analysis History"],
    label_visibility="collapsed"
)
# "Text Classification",

# Add a logo or icon in the sidebar
# st.sidebar.markdown("""
#     <div style="text-align: center; margin-bottom: 2em;">
#         <img src="Club-Stride-Logo.png" alt="Analysis Icon" style="width: 80px; height: 80px;">
#     </div>
# """, unsafe_allow_html=True)

# Initialize session state for multiple files
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
    st.session_state.processed_files = set()  # Track processed files by name

# Optimize OpenAI API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_openai_embedding(text: str) -> List[float]:
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # Using smaller model for speed
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
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

# Update the process_file function to use the new MongoDB functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        
        # Generate a unique identifier for the file
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        unique_file_name = f"{file.name}_{unique_id}"
        
        # Get embedding for the content
        embedding = get_openai_embedding(content)
        if embedding is None:
            st.error("Failed to generate embedding for the content")
            return None
        
        # Store in MongoDB
        file_id = store_processed_data(
            file_name=unique_file_name,
            content=content,
            embeddings=embedding,
            analysis_results={}
        )
        
        if file_id is None:
            st.error("Failed to store processed data")
            return None
        
        return content, file_id, unique_file_name
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)}")
        return None

# Optimize entity recognition
@st.cache_data(ttl=3600)  # Cache for 1 hour
def analyze_entities(text: str, categories: List[str]) -> List[Dict[str, Any]]:
    try:
        prompt = f"""Analyze the following text and identify named entities. 
        For each entity, provide:
        1. The entity text
        2. The category (from: {', '.join(categories)})
        3. The context in which it appears
        4. Its significance in the text
        
        Format the response as a JSON array of objects with these fields:
        - text: the entity text
        - category: the entity category
        - context: a brief context
        - significance: why it's important
        - start_char: starting character position
        - end_char: ending character position
        
        Text to analyze:
        {text}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise named entity recognition system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=2000   # Limit token usage
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error analyzing entities: {str(e)}")
        return []

# Add progress tracking
def track_progress(operation: str, total: int):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update(completed: int):
        progress = completed / total
        progress_bar.progress(progress)
        status_text.text(f"{operation}: {completed}/{total} completed")
    
    return update

# Main content area with enhanced UI
if page == "Home":
    st.markdown('<h2 class="section-title">Welcome to the NLP Tool for YPAR</h2>', unsafe_allow_html=True)
    
    # Feature cards in a grid
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
        <p>Tips for best results:</p>
        <ul>
            <li>Ensure text is clear and well-formatted</li>
            <li>Remove any sensitive information</li>
            <li>Consider cultural context in your data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'docx', 'xlsx'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        # Filter out already processed files
        new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
        
        if new_files:
            # Process all files in parallel
            with st.spinner("Processing files..."):
                update_progress = track_progress("Processing files", len(new_files))
                
                # Batch process files
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
                        st.session_state.processed_files.add(new_files[i].name)  # Add to processed files set
                    update_progress(i + 1)
                
                if new_contents:
                    if not isinstance(st.session_state.processed_data, list):
                        st.session_state.processed_data = []
                    
                    st.session_state.processed_data.extend(new_contents)
                    st.session_state.file_names.extend(new_names)
                    if 'file_ids' not in st.session_state:
                        st.session_state.file_ids = []
                    st.session_state.file_ids.extend(new_ids)
                    st.success(f"✅ Successfully processed {len(new_contents)} new file(s)!")
        else:
            st.info("All selected files have already been processed.")
        
        # Display file list with improved styling
        if st.session_state.file_names:
            st.markdown("""
            <div class="result-box">
                <h3>📋 Uploaded Files</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a table-like display for files
            for i, (name, content) in enumerate(zip(st.session_state.file_names, st.session_state.processed_data)):
                # Extract original name for display
                original_name = name.split('_')[0]
                
                # File card with improved styling
                st.markdown(f"""
                <div class="feature-card" style="margin-bottom: 1em;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0; color: #2c3e50;">{original_name}</h4>
                            <p style="margin: 0.5em 0; color: #7f8c8d; font-size: 0.9em;">
                                {len(content.split())} words
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for buttons
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
                            # Remove the file and its associated data
                            st.session_state.processed_data.pop(i)
                            st.session_state.file_names.pop(i)
                            st.session_state.file_ids.pop(i)
                            st.session_state.processed_files.remove(original_name)  # Remove from processed files set
                            
                            # Safely remove associated analysis results
                            if name in st.session_state.themes:
                                del st.session_state.themes[name]
                            if name in st.session_state.sentiments:
                                del st.session_state.sentiments[name]
                            if name in st.session_state.quotes:
                                del st.session_state.quotes[name]
                            if name in st.session_state.insights:
                                del st.session_state.insights[name]
                            
                            st.success(f"File {original_name} removed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing file: {str(e)}")

elif page == "Text Processing":
    st.markdown('<h2 class="section-title">Text Processing</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        # File selector
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
                    embedding = get_openai_embedding(st.session_state.processed_data[file_index])
                    if embedding:
                        st.success("Text processed successfully!")
                        st.write("Embedding dimension:", len(embedding))

elif page == "Topic Modeling":
    st.markdown('<h2 class="section-title">Topic Modeling</h2>', unsafe_allow_html=True)
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Topic Modeling</h3>
            <p>Topic modeling helps identify key themes and patterns in your text data.</p>
            <p>This analysis will:</p>
            <ul>
                <li>Identify main themes</li>
                <li>Provide educational context</li>
                <li>Show connections between themes</li>
                <li>Suggest implications for YPAR</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File selector
        selected_file = st.selectbox(
            "Select a file to analyze",
            st.session_state.file_names,
            index=st.session_state.current_file_index
        )
        
        if selected_file:
            file_index = st.session_state.file_names.index(selected_file)
            text = st.session_state.processed_data[file_index]
            file_id = st.session_state.file_ids[file_index]
            
            # Display the current file's content
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
                            
                            # Store the analysis results
                            store_analysis_results(
                                file_id=file_id,
                                analysis_type="theme_analysis",
                                results={"themes": themes}
                            )
                    except Exception as e:
                        st.error(f"Error analyzing themes: {str(e)}")

elif page == "Quote Extraction":
    st.markdown('<h2 class="section-title">Quote Extraction</h2>', unsafe_allow_html=True)
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Quote Extraction</h3>
            <p>This feature helps identify and analyze representative quotes from your data.</p>
            <p>For each quote, you'll get:</p>
            <ul>
                <li>The exact quote</li>
                <li>Analysis of its significance</li>
                <li>Connection to broader themes</li>
                <li>Suggestions for use in YPAR</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File selector
        selected_file = st.selectbox(
            "Select a file to analyze",
            st.session_state.file_names,
            index=st.session_state.current_file_index
        )
        
        if selected_file:
            file_index = st.session_state.file_names.index(selected_file)
            text = st.session_state.processed_data[file_index]
            file_id = st.session_state.file_ids[file_index]
            
            # Display the current file's content
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
                            
                            # Store the analysis results
                            store_analysis_results(
                                file_id=file_id,
                                analysis_type="quote_extraction",
                                results={"quotes": quotes}
                            )
                    except Exception as e:
                        st.error(f"Error extracting quotes: {str(e)}")

elif page == "Insight Generation":
    st.markdown('<h2 class="section-title">Insight Generation</h2>', unsafe_allow_html=True)
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Insight Generation</h3>
            <p>This feature helps identify key findings and patterns in your data.</p>
            <p>You'll get:</p>
            <ul>
                <li>Key findings and their significance</li>
                <li>Unexpected patterns or correlations</li>
                <li>Questions for further exploration</li>
                <li>Recommendations for next steps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File selector
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
        # Visualization options
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
        
        st.markdown("""
        <div class="info-box">
            <h3>📊 About Visualizations</h3>
            <p>Visualizations help you understand and communicate your findings.</p>
            <p>Available visualizations:</p>
            <ul>
                <li>Word clouds showing key terms</li>
                <li>Theme distribution charts</li>
                <li>Sentiment analysis graphs</li>
                <li>Theme network visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if viz_option != "Theme Network":
            # Word Cloud in a styled container
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader("Word Cloud")
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                contour_width=1,
                contour_color='#2874a6',
                max_words=100,
                min_font_size=10,
                max_font_size=200
            ).generate(text_to_visualize)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Theme Network Visualization
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.subheader("Theme Network")
            
            # Get themes from all files
            all_themes = []
            for name in st.session_state.file_names:
                if name in st.session_state.themes:
                    themes = st.session_state.themes[name]
                    if isinstance(themes, str):
                        # Parse themes from string if needed
                        theme_list = [t.strip() for t in themes.split('\n') if t.strip()]
                        all_themes.extend(theme_list)
                    elif isinstance(themes, list):
                        all_themes.extend(themes)
            
            if all_themes:
                # Get embeddings for themes
                theme_embeddings = []
                for theme in all_themes:
                    embedding = get_openai_embedding(theme)
                    if embedding:
                        theme_embeddings.append(embedding)
                
                if theme_embeddings:
                    # Calculate similarity matrix
                    similarity_matrix = cosine_similarity(theme_embeddings)
                    
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add nodes
                    for i, theme in enumerate(all_themes):
                        G.add_node(theme, size=10)
                    
                    # Add edges based on similarity
                    for i in range(len(all_themes)):
                        for j in range(i+1, len(all_themes)):
                            if similarity_matrix[i][j] > 0.7:  # Threshold for connection
                                G.add_edge(all_themes[i], all_themes[j], 
                                         weight=similarity_matrix[i][j])
                    
                    # Create plotly figure
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    
                    # Initialize lists for edges
                    edge_x = []
                    edge_y = []

                    # Add the edges to the graph
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    # Create the edge trace
                    edge_trace = go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )

                    # Initialize empty lists for node properties
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    node_color = []
                    node_hover_text = []

                    # Add nodes with enhanced information
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Calculate node properties
                        connections = len(G[node])
                        connected_nodes = list(G[node].keys())
                        
                        # Create hover text with detailed information
                        hover_text = f"Theme: {node}<br>"
                        hover_text += f"Connections: {connections}<br>"
                        hover_text += f"Related Themes: {', '.join(connected_nodes)}"
                        
                        node_text.append(node)
                        node_color.append(connections)
                        node_size.append(10 + (connections * 5))
                        node_hover_text.append(hover_text)

                    # Create the node trace with all collected data
                    node_trace = go.Scatter(
                        x=node_x,
                        y=node_y,
                        text=node_text,
                        mode='markers+text',
                        hoverinfo='text',
                        hovertext=node_hover_text,
                        textposition="bottom center",
                        textfont=dict(
                            size=10,
                            color='#2c3e50'
                        ),
                        marker=dict(
                            showscale=True,
                            colorscale='YlOrRd',
                            reversescale=False,
                            color=node_color,
                            size=node_size,
                            sizeref=2.*max(node_color) / (20.**2),
                            sizemode='area',
                            line=dict(width=1, color='#2c3e50'),
                            colorbar=dict(
                                thickness=15,
                                title=dict(
                                    text='Connections',
                                    side='right'
                                ),
                                xanchor='left'
                            )
                        )
                    )

                    # Update the figure creation
                    fig = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                    title=dict(
                                        text='Theme Network Analysis',
                                        font=dict(size=18, color='#2c3e50')
                                    ),
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    annotations=[
                                        dict(
                                            text="Themes are connected based on semantic similarity.<br>Node size and color indicate number of connections.",
                                            showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.005, y=-0.002,
                                            font=dict(size=12, color='#7f8c8d')
                                        )
                                    ],
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    plot_bgcolor='#f8f9fa',
                                    paper_bgcolor='#f8f9fa'
                                ))

                    # Add a legend-like annotation to explain the visualization
                    fig.add_annotation(
                        x=1.02,
                        y=1,
                        xref="paper",
                        yref="paper",
                        text="Node Properties:<br>" +
                             "- Size: Number of connections<br>" +
                             "- Color: Connection strength<br>" +
                             "- Hover: Detailed information",
                        showarrow=False,
                        font=dict(size=12, color='#2c3e50'),
                        align="left"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not generate embeddings for themes. Please try again.")
            else:
                st.warning("No themes found. Please analyze some text first.")
            
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Named Entity Recognition":
    st.markdown('<h2 class="section-title">Named Entity Recognition</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Named Entity Recognition</h3>
            <p>This feature uses advanced AI to identify and categorize named entities in your text, such as:</p>
            <ul>
                <li>People and Organizations</li>
                <li>Locations and Places</li>
                <li>Dates and Times</li>
                <li>Events and Activities</li>
                <li>Concepts and Ideas</li>
                <li>Custom Categories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File selector
        selected_file = st.selectbox(
            "Select a file to analyze",
            st.session_state.file_names,
            index=st.session_state.current_file_index
        )
        
        if selected_file:
            file_index = st.session_state.file_names.index(selected_file)
            text = st.session_state.processed_data[file_index]
            
            # Custom entity categories
            custom_categories = st.multiselect(
                "Select additional entity categories to detect",
                ["Emotions", "Actions", "Problems", "Solutions", "Resources", "Custom"],
                default=["Emotions", "Actions"]
            )
            
            if "Custom" in custom_categories:
                custom_category = st.text_input("Enter custom category name")
                if custom_category:
                    custom_categories.remove("Custom")
                    custom_categories.append(custom_category)
            
            if st.button("Analyze Entities"):
                with st.spinner("Processing entities..."):
                    try:
                        # Prepare the prompt for GPT-4
                        categories = [
                            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
                            "EVENT", "CONCEPT", "PRODUCT", "OTHER"
                        ] + custom_categories
                        
                        prompt = f"""Analyze the following text and identify named entities. 
                        For each entity, provide:
                        1. The entity text
                        2. The category (from: {', '.join(categories)})
                        3. The context in which it appears
                        4. Its significance in the text
                        
                        Format the response as a JSON array of objects with these fields:
                        - text: the entity text
                        - category: the entity category
                        - context: a brief context
                        - significance: why it's important
                        - start_char: starting character position
                        - end_char: ending character position
                        
                        Text to analyze:
                        {text}
                        """
                        
                        response = openai.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a precise named entity recognition system. Return only valid JSON."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        # Parse the response
                        entities = json.loads(response.choices[0].message.content)
                        
                        if entities:
                            # Store results in MongoDB
                            store_analysis_results(
                                st.session_state.file_ids[file_index],
                                "entity_recognition",
                                entities
                            )
                            
                            # Display results in a table
                            df = pd.DataFrame(entities)
                            st.dataframe(df)
                            
                            # Create interactive visualization
                            fig = go.Figure()
                            
                            # Add text with highlighted entities
                            text_with_highlights = text
                            for entity in sorted(entities, key=lambda x: x['start_char'], reverse=True):
                                text_with_highlights = (
                                    text_with_highlights[:entity['start_char']] +
                                    f"<span style='background-color: #ffeb3b;'>{entity['text']}</span>" +
                                    text_with_highlights[entity['end_char']:]
                                )
                            
                            # Add text to figure
                            fig.add_trace(go.Scatter(
                                x=[0],
                                y=[0],
                                text=[text_with_highlights],
                                mode="text",
                                textposition="middle center",
                                showlegend=False
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title="Text with Highlighted Entities",
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=500,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create entity distribution chart
                            category_counts = pd.DataFrame(entities)['category'].value_counts()
                            fig2 = px.bar(
                                x=category_counts.index,
                                y=category_counts.values,
                                title="Entity Distribution by Category",
                                labels={'x': 'Category', 'y': 'Count'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Create entity network
                            G = nx.Graph()
                            for entity in entities:
                                G.add_node(entity['text'], category=entity['category'])
                            
                            # Add edges between entities that appear close to each other
                            for i, entity1 in enumerate(entities):
                                for j, entity2 in enumerate(entities[i+1:], i+1):
                                    if abs(entity1['start_char'] - entity2['start_char']) < 100:
                                        G.add_edge(entity1['text'], entity2['text'])
                            
                            # Create network visualization
                            pos = nx.spring_layout(G)
                            
                            edge_trace = go.Scatter(
                                x=[],
                                y=[],
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines')
                            
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_trace['x'] += tuple([x0, x1, None])
                                edge_trace['y'] += tuple([y0, y1, None])
                            
                            node_trace = go.Scatter(
                                x=[],
                                y=[],
                                text=[],
                                mode='markers+text',
                                hoverinfo='text',
                                marker=dict(
                                    showscale=True,
                                    colorscale='YlGnBu',
                                    reversescale=True,
                                    color=[],
                                    size=10,
                                    colorbar=dict(
                                        thickness=15,
                                        title='Node Connections',
                                        xanchor='left',
                                        title_side='right'
                                    ),
                                    line_width=2))
                            
                            for node in G.nodes():
                                x, y = pos[node]
                                node_trace['x'] += tuple([x])
                                node_trace['y'] += tuple([y])
                                node_trace['text'] += tuple([node])
                                node_trace['marker']['color'] += tuple([len(G[node])])
                            
                            fig3 = go.Figure(data=[edge_trace, node_trace],
                                         layout=go.Layout(
                                            title='Entity Network',
                                            showlegend=False,
                                            hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=40),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                        )
                            
                            st.plotly_chart(fig3, use_container_width=True)
                            
                        else:
                            st.info("No named entities found in the text.")
                            
                    except Exception as e:
                        st.error(f"Error processing entities: {str(e)}")

elif page == "Text Classification":
    st.markdown('<h2 class="section-title">Text Classification</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Text Classification</h3>
            <p>This feature helps classify text into predefined categories based on its content.</p>
            <p>You can:</p>
            <ul>
                <li>Define custom categories</li>
                <li>Train a classifier on your data</li>
                <li>Classify new text</li>
                <li>View classification results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for classifier
        if 'classifier' not in st.session_state:
            st.session_state.classifier = None
        if 'categories' not in st.session_state:
            st.session_state.categories = []
        
        # Category management
        st.subheader("Category Management")
        new_category = st.text_input("Add new category")
        if st.button("Add Category") and new_category:
            if new_category not in st.session_state.categories:
                st.session_state.categories.append(new_category)
                st.success(f"Category '{new_category}' added!")
        
        if st.session_state.categories:
            st.write("Current categories:")
            for cat in st.session_state.categories:
                st.write(f"- {cat}")
            
            # Training data collection
            st.subheader("Training Data")
            selected_file = st.selectbox(
                "Select a file to classify",
                st.session_state.file_names
            )
            
            if selected_file:
                file_index = st.session_state.file_names.index(selected_file)
                text = st.session_state.processed_data[file_index]
                
                selected_category = st.selectbox(
                    "Select category for this text",
                    st.session_state.categories
                )
                
                if st.button("Add to Training Data"):
                    if 'training_data' not in st.session_state:
                        st.session_state.training_data = []
                        st.session_state.training_labels = []
                    
                    st.session_state.training_data.append(text)
                    st.session_state.training_labels.append(selected_category)
                    st.success("Text added to training data!")
            
            # Train classifier
            if 'training_data' in st.session_state and len(st.session_state.training_data) > 0:
                if st.button("Train Classifier"):
                    with st.spinner("Training classifier..."):
                        try:
                            # Create and train classifier
                            classifier = Pipeline([
                                ('tfidf', TfidfVectorizer()),
                                ('clf', MultinomialNB())
                            ])
                            classifier.fit(st.session_state.training_data, st.session_state.training_labels)
                            st.session_state.classifier = classifier
                            st.success("Classifier trained successfully!")
                        except Exception as e:
                            st.error(f"Error training classifier: {str(e)}")
            
            # Classify new text
            if st.session_state.classifier:
                st.subheader("Classify New Text")
                new_text = st.text_area("Enter text to classify")
                if new_text and st.button("Classify"):
                    try:
                        prediction = st.session_state.classifier.predict([new_text])[0]
                        st.success(f"Predicted category: {prediction}")
                    except Exception as e:
                        st.error(f"Error classifying text: {str(e)}")

elif page == "Intent Detection":
    st.markdown('<h2 class="section-title">Intent Detection</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Intent Detection</h3>
            <p>This feature helps identify the underlying intent or purpose in text, such as:</p>
            <ul>
                <li>Requests for information</li>
                <li>Expressions of opinion</li>
                <li>Questions</li>
                <li>Suggestions</li>
                <li>Complaints</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File selector
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
                        # Use OpenAI to detect intent
                        response = openai.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": """You are an intent detection system. 
                                Analyze the text and identify the main intent(s). 
                                Consider: requests, questions, opinions, suggestions, complaints.
                                Format the response as a clear list of intents with explanations."""},
                                {"role": "user", "content": f"Analyze this text for intent: {text}"}
                            ]
                        )
                        
                        st.subheader("Detected Intents")
                        st.markdown(response.choices[0].message.content)
                        
                        # Visualize intent distribution
                        intents = response.choices[0].message.content.split('\n')
                        intent_counts = {}
                        for intent in intents:
                            if intent.strip():
                                intent_type = intent.split(':')[0].strip()
                                intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
                        
                        if intent_counts:
                            fig = px.pie(
                                values=list(intent_counts.values()),
                                names=list(intent_counts.keys()),
                                title="Intent Distribution"
                            )
                            st.plotly_chart(fig)
                        
                    except Exception as e:
                        st.error(f"Error detecting intent: {str(e)}")

elif page == "Ethics & Bias":
    st.markdown('<h2 class="section-title">Ethics & Bias Analysis</h2>', unsafe_allow_html=True)
    if st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
            <h3>About Ethics & Bias Analysis</h3>
            <p>This feature helps identify potential biases and ethical considerations in your analysis.</p>
            <p>It examines:</p>
            <ul>
                <li>Cultural biases</li>
                <li>Language patterns</li>
                <li>Representation issues</li>
                <li>Ethical implications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Check for Potential Biases"):
            with st.spinner("Analyzing for potential biases..."):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": """You are a helpful assistant that identifies potential biases in text.
                            Provide:
                            1. Potential biases found
                            2. Cultural context considerations
                            3. Ethical implications
                            4. Recommendations for addressing biases
                            Format the response in a clear, educational manner."""},
                            {"role": "user", "content": f"Identify potential biases in this text: {st.session_state.processed_data}"}
                        ]
                    )
                    st.subheader("Potential Biases and Ethical Considerations")
                    st.markdown(response.choices[0].message.content)
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

# Footer with enhanced styling and better contrast
st.markdown("""
    <div style="text-align: center; margin-top: 2em; padding: 1em; background-color: #eaf2f8; border-radius: 10px; border-top: 2px solid #2874a6;">
        <p style="color: #2c3e50; font-weight: bold;">© 2024 NLP Tool for YPAR | Powered by OpenAI</p>
    </div>
""", unsafe_allow_html=True)

# Add back the missing analysis functions
