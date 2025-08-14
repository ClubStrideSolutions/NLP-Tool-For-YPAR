"""
Optimized NLP YPAR Tool with Enhanced Navigation
UC Berkeley Cal Colors Theme
"""

import streamlit as st
import sys
from pathlib import Path

# Add directories to path
sys.path.append(str(Path(__file__).parent / "components"))
sys.path.append(str(Path(__file__).parent / "modules"))
sys.path.append(str(Path(__file__).parent / "utils"))

# Page configuration
st.set_page_config(
    page_title="NLP YPAR Tool",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import components after path setup
from enhanced_navigation import EnhancedNavigation, render_enhanced_navigation
from openai_analyzer import OpenAIAnalyzer
from enhanced_db_manager import EnhancedMongoManager
from file_handlers import FileHandler
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_page': 'home',
        'processed_data': [],
        'file_names': [],
        'file_ids': [],
        'analysis_results': [],
        'openai_api_key': '',
        'mongodb_connection_string': '',
        'db_connected': False,
        'ai_analyzer': None,
        'db_manager': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def inject_global_styles():
    """Inject global Cal Colors CSS styles"""
    st.markdown("""
    <style>
    /* Global Cal Colors Theme */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #003262;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #004d8a;
    }
    
    /* Enhanced containers */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .content-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 50, 98, 0.1);
    }
    
    .content-card:hover {
        box-shadow: 0 6px 30px rgba(0, 50, 98, 0.15);
        transition: all 0.3s ease;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #003262;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Buttons enhancement */
    .stButton > button {
        background: linear-gradient(135deg, #003262, #004d8a);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 50, 98, 0.3);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #003262;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #FDB515;
        box-shadow: 0 0 0 3px rgba(253, 181, 21, 0.2);
    }
    
    /* File uploader enhancement */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(0, 50, 98, 0.05), rgba(253, 181, 21, 0.05));
        border: 3px dashed #003262;
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #FDB515;
        background: linear-gradient(135deg, rgba(253, 181, 21, 0.1), rgba(0, 50, 98, 0.1));
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #003262, #004d8a);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 50, 98, 0.2);
    }
    
    [data-testid="metric-container"] label {
        color: #FDB515 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #003262;
        border-radius: 10px;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #FDB515;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FDB515, #FFC72C);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #003262, #004d8a);
        color: white !important;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #004d8a, #003262);
    }
    
    /* Success/Error/Warning/Info boxes */
    .stAlert > div {
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_services():
    """Initialize AI and database services"""
    # Initialize database manager
    if not st.session_state.db_manager:
        if st.session_state.mongodb_connection_string:
            db_manager = EnhancedMongoManager(st.session_state.mongodb_connection_string)
            if db_manager.connected:
                st.session_state.db_manager = db_manager
                st.session_state.db_connected = True
                logger.info("Database connected successfully")
    
    # Initialize AI analyzer
    if not st.session_state.ai_analyzer:
        if st.session_state.openai_api_key:
            try:
                ai_analyzer = OpenAIAnalyzer()
                if ai_analyzer.is_available():
                    st.session_state.ai_analyzer = ai_analyzer
                    logger.info("OpenAI analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")

def render_home_page():
    """Render the home page"""
    st.markdown("""
    <div class="content-card">
        <h1 style="text-align: center; color: #003262; font-size: 3rem; margin-bottom: 0;">
            🐻 NLP YPAR Tool
        </h1>
        <p style="text-align: center; color: #6C757D; font-size: 1.2rem;">
            Youth Participatory Action Research Platform
        </p>
        <hr style="border: none; height: 3px; background: linear-gradient(90deg, #003262, #FDB515, #003262); margin: 2rem 0;">
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    nav = EnhancedNavigation()
    quick_actions = [
        {"label": "Upload Documents", "icon": "📤", "color": "#003262"},
        {"label": "Start Analysis", "icon": "🔬", "color": "#FDB515"},
        {"label": "View Reports", "icon": "📊", "color": "#3B7EA1"},
        {"label": "Settings", "icon": "⚙️", "color": "#004d8a"}
    ]
    st.markdown(nav.create_quick_actions(quick_actions), unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #003262;">🚀 AI-Powered Analysis</h3>
            <p>Advanced natural language processing with OpenAI GPT models for intelligent text analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #003262;">📊 Rich Visualizations</h3>
            <p>Interactive charts, word clouds, and network graphs to explore your data visually.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #003262;">🔒 Secure Storage</h3>
            <p>MongoDB integration for secure, scalable storage with full analysis history.</p>
        </div>
        """, unsafe_allow_html=True)

def render_upload_page():
    """Render the upload page"""
    st.markdown("""
    <div class="content-card">
        <h2 style="color: #003262;">📤 Upload Documents</h2>
        <p style="color: #6C757D;">Upload your documents for analysis. Supported formats: TXT, PDF, DOCX, MD</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        help="Maximum file size: 10MB per file"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        file_handler = FileHandler()
        
        for idx, file in enumerate(uploaded_files):
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            # Process file
            content = file_handler.extract_text(file)
            if content:
                st.session_state.processed_data.append(content)
                st.session_state.file_names.append(file.name)
                
                # Generate file ID
                import hashlib
                file_id = hashlib.md5(f"{file.name}_{len(content)}".encode()).hexdigest()[:16]
                st.session_state.file_ids.append(file_id)
                
                # Store in database if available
                if st.session_state.db_manager:
                    metadata = {
                        'word_count': len(content.split()),
                        'char_count': len(content),
                        'type': file.type,
                        'size': file.size
                    }
                    st.session_state.db_manager.store_document(file_id, file.name, content, metadata)
        
        st.success(f"✅ Successfully processed {len(uploaded_files)} files")

def render_analysis_page():
    """Render the analysis page with enhanced navigation"""
    render_enhanced_navigation("analysis")
    
    if not st.session_state.processed_data:
        st.warning("📝 Please upload documents first to begin analysis")
        return
    
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🤖 AI Analysis", "😊 Sentiment", "🎯 Themes", "🔑 Keywords",
            "📝 Summary", "💬 Quotes", "💡 Insights", "❓ Q&A"
        ])
        
        with tab1:
            st.markdown("""
            <div class="content-card">
                <h3>🤖 AI-Powered Analysis</h3>
                <p>Comprehensive analysis using OpenAI GPT models</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Complete AI Analysis", type="primary"):
                if st.session_state.ai_analyzer:
                    with st.spinner("Analyzing with AI..."):
                        # Run various AI analyses
                        sentiment = st.session_state.ai_analyzer.analyze_sentiment(text)
                        themes = st.session_state.ai_analyzer.extract_themes(text)
                        summary = st.session_state.ai_analyzer.summarize_text(text)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment.get('sentiment', 'N/A'))
                            st.metric("Confidence", f"{sentiment.get('confidence', 0)}%")
                        with col2:
                            st.metric("Themes Found", len(themes.get('themes', [])))
                            st.metric("Summary Length", f"{len(summary.get('summary', ''))} chars")
                        
                        # Store results
                        if st.session_state.db_manager:
                            st.session_state.db_manager.store_analysis(
                                file_id, "ai_complete", 
                                {'sentiment': sentiment, 'themes': themes, 'summary': summary}
                            )
                else:
                    st.info("🔑 Please configure OpenAI API key in Settings")

def main():
    """Main application"""
    init_session_state()
    inject_global_styles()
    initialize_services()
    
    # Enhanced navigation
    render_enhanced_navigation("main")
    
    # Page routing based on navigation
    page = st.session_state.current_page
    
    if page == 'home':
        render_home_page()
    elif page == 'upload':
        render_upload_page()
    elif page == 'analysis':
        render_analysis_page()
    # Add more pages as needed
    
    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #003262, #004d8a); 
                border-radius: 15px; text-align: center; color: white;">
        <p style="margin: 0;">🐻 NLP YPAR Tool | UC Berkeley | Cal Colors Theme</p>
        <p style="margin: 0; opacity: 0.8; font-size: 0.9rem;">© 2024 Youth Participatory Action Research Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()