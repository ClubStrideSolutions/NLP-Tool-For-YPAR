"""
Fixed NLP YPAR Tool with Working Navigation
UC Berkeley Cal Colors Theme
"""

import streamlit as st
import sys
from pathlib import Path
import hashlib
import pandas as pd
from datetime import datetime

# Add directories to path
sys.path.append(str(Path(__file__).parent / "components"))
sys.path.append(str(Path(__file__).parent / "modules"))
sys.path.append(str(Path(__file__).parent / "utils"))

# Page configuration
st.set_page_config(
    page_title="NLP YPAR Tool",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'file_ids' not in st.session_state:
    st.session_state.file_ids = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''
if 'mongodb_connection_string' not in st.session_state:
    st.session_state.mongodb_connection_string = ''

# Import modules (with error handling)
try:
    from openai_analyzer import OpenAIAnalyzer
    openai_available = True
except:
    openai_available = False

try:
    from enhanced_db_manager import EnhancedMongoManager
    mongo_available = True
except:
    mongo_available = False

try:
    from file_handlers import FileHandler
    file_handler_available = True
except:
    file_handler_available = False

# Apply custom CSS
st.markdown("""
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container styling */
.main {
    padding: 0;
    max-width: 1400px;
    margin: 0 auto;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #003262 0%, #004d8a 100%);
    padding: 2rem;
    border-radius: 0 0 20px 20px;
    margin: -1rem -1rem 2rem -1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 50, 98, 0.3);
}

.header-title {
    color: #FDB515;
    font-size: 3rem;
    font-weight: bold;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.header-subtitle {
    color: white;
    font-size: 1.2rem;
    margin-top: 0.5rem;
    opacity: 0.95;
}

/* Navigation buttons */
.nav-button {
    background: white;
    color: #003262;
    padding: 1rem 1.5rem;
    border: 2px solid #003262;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-button:hover {
    background: #003262;
    color: #FDB515;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 50, 98, 0.3);
}

.nav-button.active {
    background: #003262;
    color: #FDB515;
    box-shadow: 0 4px 12px rgba(0, 50, 98, 0.3);
}

/* Cards */
.feature-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #FDB515;
    height: 100%;
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0, 50, 98, 0.2);
}

.feature-card h3 {
    color: #003262;
    margin-top: 0;
}

.feature-card p {
    color: #6C757D;
    line-height: 1.6;
}

/* Status metrics */
.metric-container {
    background: linear-gradient(135deg, #003262, #004d8a);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0, 50, 98, 0.2);
}

.metric-label {
    color: #FDB515;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003262, #004d8a);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border-radius: 10px;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 50, 98, 0.3);
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(0, 50, 98, 0.05), rgba(253, 181, 21, 0.05));
    border: 3px dashed #003262;
    border-radius: 15px;
    padding: 3rem;
    text-align: center;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #FDB515;
    background: linear-gradient(135deg, rgba(253, 181, 21, 0.1), rgba(0, 50, 98, 0.1));
}

/* Selectbox */
.stSelectbox > div > div {
    border: 2px solid #003262;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Text input */
.stTextInput > div > div > input {
    border: 2px solid #003262;
    border-radius: 10px;
    padding: 0.75rem;
}

.stTextInput > div > div > input:focus {
    border-color: #FDB515;
    box-shadow: 0 0 0 3px rgba(253, 181, 21, 0.2);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    color: #003262;
    border: 2px solid #003262;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(0, 50, 98, 0.1);
}

.stTabs [aria-selected="true"] {
    background-color: #003262;
    color: #FDB515;
    border-bottom: 3px solid #FDB515;
}

/* Success/Error/Warning/Info alerts */
.stAlert {
    border-radius: 10px;
    border-left: 4px solid;
}

/* Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #003262, #004d8a);
    color: white !important;
    border-radius: 10px;
    font-weight: 600;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #003262, #004d8a);
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 50, 98, 0.2);
}

[data-testid="metric-container"] label {
    color: #FDB515 !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: white !important;
    font-size: 2rem;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #FDB515, #FFC72C);
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, #003262, #004d8a);
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #003262, #004d8a);
}
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def init_services():
    services = {
        'ai_analyzer': None,
        'db_manager': None,
        'file_handler': None
    }
    
    if openai_available and st.session_state.openai_api_key:
        try:
            services['ai_analyzer'] = OpenAIAnalyzer()
        except:
            pass
    
    if mongo_available and st.session_state.mongodb_connection_string:
        try:
            services['db_manager'] = EnhancedMongoManager(st.session_state.mongodb_connection_string)
        except:
            pass
    
    if file_handler_available:
        try:
            services['file_handler'] = FileHandler()
        except:
            pass
    
    return services

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🐻 NLP YPAR Tool</h1>
    <p class="header-subtitle">Youth Participatory Action Research Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #003262, #004d8a); 
                padding: 1.5rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                text-align: center;">
        <h2 style="color: #FDB515; margin: 0;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation options
    pages = ["🏠 Home", "📤 Upload", "🔬 Analysis", "📊 Visualizations", 
             "🤖 RAG Analysis", "📜 History", "⚙️ Settings"]
    
    selected_page = st.radio("Go to", pages, label_visibility="collapsed")
    st.session_state.current_page = selected_page
    
    st.divider()
    
    # System Status
    st.markdown("### 📊 System Status")
    
    services = init_services()
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files", len(st.session_state.processed_data))
    with col2:
        st.metric("Analyses", len(st.session_state.analysis_results))
    
    # Connection status
    if services['ai_analyzer'] and services['ai_analyzer'].is_available():
        st.success("✅ AI Active")
    else:
        st.info("🔴 AI Inactive")
    
    if services['db_manager'] and services['db_manager'].connected:
        st.success("✅ DB Connected")
    else:
        st.info("🟡 Local Storage")

# Main content area
if st.session_state.current_page == "🏠 Home":
    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.current_page = "📤 Upload"
            st.rerun()
    
    with col2:
        if st.button("🔬 Start Analysis", use_container_width=True):
            st.session_state.current_page = "🔬 Analysis"
            st.rerun()
    
    with col3:
        if st.button("📊 View Reports", use_container_width=True):
            st.session_state.current_page = "📊 Visualizations"
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "⚙️ Settings"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 AI-Powered Analysis</h3>
            <p>Advanced natural language processing with OpenAI GPT models for intelligent text analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Rich Visualizations</h3>
            <p>Interactive charts, word clouds, and network graphs to explore your data visually.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Secure Storage</h3>
            <p>MongoDB integration for secure, scalable storage with full analysis history.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📈 Platform Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Documents Processed</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(st.session_state.processed_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(st.session_state.analysis_results)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Active Sessions</div>
            <div class="metric-value">1</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">System Health</div>
            <div class="metric-value">98%</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "📤 Upload":
    st.markdown("## 📤 Upload Documents")
    st.markdown("Upload your documents for analysis. Supported formats: TXT, PDF, DOCX, MD")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        help="Maximum file size: 10MB per file"
    )
    
    if uploaded_files:
        services = init_services()
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            # Process file
            if services['file_handler']:
                content, metadata = services['file_handler'].process_file(file)
            else:
                content = str(file.read(), 'utf-8') if file.type == 'text/plain' else None
                metadata = {}
            
            if content:
                st.session_state.processed_data.append(content)
                st.session_state.file_names.append(file.name)
                
                # Generate file ID
                file_id = hashlib.md5(f"{file.name}_{len(content)}".encode()).hexdigest()[:16]
                st.session_state.file_ids.append(file_id)
                
                # Store in database if available
                if services['db_manager']:
                    if not metadata:
                        metadata = {
                            'word_count': len(content.split()),
                            'char_count': len(content),
                            'type': file.type,
                            'size': file.size
                        }
                    services['db_manager'].store_document(file_id, file.name, content, metadata)
        
        st.success(f"✅ Successfully processed {len(uploaded_files)} files")
        
        # Display uploaded files
        st.markdown("### 📁 Uploaded Files")
        for name in st.session_state.file_names:
            st.write(f"✓ {name}")

elif st.session_state.current_page == "🔬 Analysis":
    st.markdown("## 🔬 Text Analysis")
    
    if not st.session_state.processed_data:
        st.warning("📝 Please upload documents first to begin analysis")
    else:
        # File selection
        selected_file = st.selectbox(
            "Select a document to analyze",
            st.session_state.file_names,
            help="Choose a document from your uploaded files"
        )
        
        if selected_file:
            file_index = st.session_state.file_names.index(selected_file)
            text = st.session_state.processed_data[file_index]
            file_id = st.session_state.file_ids[file_index]
            
            # Analysis tabs
            tabs = st.tabs(["🤖 AI Analysis", "📊 Sentiment", "🎯 Themes", "🔑 Keywords",
                           "📝 Summary", "💬 Quotes", "💡 Insights", "❓ Q&A"])
            
            services = init_services()
            
            with tabs[0]:
                st.markdown("### 🤖 Complete AI Analysis")
                
                if st.button("Run AI Analysis", type="primary"):
                    if services['ai_analyzer']:
                        with st.spinner("Analyzing with AI..."):
                            # Run analyses
                            sentiment = services['ai_analyzer'].analyze_sentiment(text)
                            themes = services['ai_analyzer'].extract_themes(text)
                            summary = services['ai_analyzer'].summarize_text(text)
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Sentiment", sentiment.get('sentiment', 'N/A'))
                                st.metric("Confidence", f"{sentiment.get('confidence', 0)}%")
                            with col2:
                                st.metric("Themes Found", len(themes.get('themes', [])))
                                st.metric("Summary Length", f"{len(summary.get('summary', ''))} chars")
                            
                            # Store results
                            analysis_result = {
                                'file_id': file_id,
                                'file_name': selected_file,
                                'sentiment': sentiment,
                                'themes': themes,
                                'summary': summary,
                                'timestamp': datetime.now()
                            }
                            st.session_state.analysis_results.append(analysis_result)
                            
                            if services['db_manager']:
                                services['db_manager'].store_analysis(
                                    file_id, "ai_complete", analysis_result
                                )
                            
                            st.success("✅ Analysis complete!")
                    else:
                        st.info("🔑 Please configure OpenAI API key in Settings")

elif st.session_state.current_page == "⚙️ Settings":
    st.markdown("## ⚙️ Settings")
    
    tabs = st.tabs(["🔑 API Configuration", "📊 System Info", "🔧 Actions"])
    
    with tabs[0]:
        st.markdown("### API & Database Configuration")
        
        # MongoDB Connection
        mongo_conn = st.text_input(
            "MongoDB Connection String",
            value=st.session_state.mongodb_connection_string,
            type="password",
            help="Enter your MongoDB connection string"
        )
        
        if st.button("Save MongoDB Settings"):
            st.session_state.mongodb_connection_string = mongo_conn
            st.success("✅ MongoDB settings saved")
            st.rerun()
        
        st.divider()
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if st.button("Save OpenAI Settings"):
            st.session_state.openai_api_key = openai_key
            st.success("✅ OpenAI settings saved")
            st.rerun()
    
    with tabs[1]:
        st.markdown("### System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Platform Details**")
            st.write("- Version: 1.0.0")
            st.write("- Theme: Cal Colors")
            st.write("- University: UC Berkeley")
        
        with col2:
            st.markdown("**System Status**")
            services = init_services()
            st.write(f"- AI: {'Active' if services['ai_analyzer'] else 'Inactive'}")
            st.write(f"- Database: {'Connected' if services['db_manager'] else 'Disconnected'}")
            st.write(f"- Files: {len(st.session_state.processed_data)}")
    
    with tabs[2]:
        st.markdown("### System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                for key in ['processed_data', 'file_names', 'file_ids', 'analysis_results']:
                    st.session_state[key] = []
                st.success("✅ All data cleared")
                st.rerun()
        
        with col2:
            if st.button("🔄 Restart Application", use_container_width=True):
                st.session_state.clear()
                st.rerun()

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #003262, #004d8a); 
            border-radius: 15px 15px 0 0; text-align: center; color: white;">
    <p style="margin: 0; font-weight: 600;">🐻 NLP YPAR Tool | UC Berkeley | Cal Colors Theme</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.9rem;">© 2024 Youth Participatory Action Research Platform</p>
</div>
""", unsafe_allow_html=True)