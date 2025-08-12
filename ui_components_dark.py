"""
Enhanced UI components with improved readability and dark theme
"""
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import pandas as pd

class UIComponents:
    """Modern UI components with better contrast"""
    
    @staticmethod
    def apply_dark_theme():
        """Apply dark theme with better readability"""
        st.markdown("""
        <style>
        /* Global styles for better readability */
        .stApp {
            background-color: #1a1a2e;
            color: #eee;
        }
        
        /* Main content area */
        .main .block-container {
            background-color: #16213e;
            padding: 2rem;
            border-radius: 10px;
            color: #e8e8e8;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #0f3460 !important;
        }
        
        .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
            color: #e8e8e8 !important;
        }
        
        /* Headers and text */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        p, span, div {
            color: #e8e8e8 !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #2a2a3e;
            color: #ffffff;
            border: 1px solid #4a4a6a;
        }
        
        .stSelectbox > div > div > div {
            background-color: #2a2a3e;
            color: #ffffff;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #e94560;
            color: white;
            border: none;
            font-weight: 600;
        }
        
        .stButton > button:hover {
            background-color: #c13651;
            box-shadow: 0 5px 15px rgba(233, 69, 96, 0.3);
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: #2a2a3e;
            border: 2px dashed #e94560;
            border-radius: 10px;
            padding: 2rem;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #2a2a3e;
            color: #ffffff;
            border-radius: 5px;
        }
        
        .streamlit-expanderContent {
            background-color: #1e1e2e;
            border: 1px solid #3a3a5a;
        }
        
        /* Metrics */
        [data-testid="metric-container"] {
            background-color: #2a2a3e;
            border: 1px solid #4a4a6a;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        
        [data-testid="metric-container"] label {
            color: #a8a8c8 !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            color: #ffffff !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2a2a3e;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #a8a8c8;
            background-color: transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e94560;
            color: white;
        }
        
        /* Tables and dataframes */
        .dataframe {
            background-color: #2a2a3e !important;
            color: #ffffff !important;
        }
        
        .dataframe th {
            background-color: #1e1e2e !important;
            color: #ffffff !important;
        }
        
        .dataframe td {
            background-color: #2a2a3e !important;
            color: #e8e8e8 !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background-color: #e94560;
        }
        
        /* Info, warning, error boxes */
        .stAlert {
            background-color: #2a2a3e;
            color: #ffffff;
            border-left: 4px solid #e94560;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #1e1e2e;
            border: 1px solid #3a3a5a;
        }
        
        /* Custom classes */
        .dark-card {
            background-color: #2a2a3e;
            border: 1px solid #4a4a6a;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .gradient-header {
            background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .feature-card-dark {
            background-color: #2a2a3e;
            border: 1px solid #4a4a6a;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .feature-card-dark:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(233, 69, 96, 0.2);
            border-color: #e94560;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_modern_header():
        """Render modern header with better contrast"""
        st.markdown("""
        <div class="gradient-header">
            <h1 style="color: white; font-size: 3rem; font-weight: 700; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                🔬 NLP Tool for YPAR
            </h1>
            <p style="color: rgba(255,255,255,0.95); font-size: 1.2rem; text-align: center; margin-top: 0.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                Advanced Natural Language Processing for Youth Participatory Action Research
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_menu() -> str:
        """Render modern sidebar navigation menu with dark theme"""
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e94560 0%, #0f3460 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                <h2 style="color: white; font-size: 1.5rem; font-weight: 600; margin: 0;">Navigation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            selected = option_menu(
                menu_title=None,
                options=[
                    "🏠 Home",
                    "📤 Upload & Process",
                    "🔍 Analysis Suite",
                    "📊 Visualizations",
                    "🤖 AI Insights",
                    "📈 Dashboard",
                    "📚 History",
                    "⚙️ Settings"
                ],
                icons=[
                    "house-fill", "cloud-upload-fill", "search", 
                    "bar-chart-fill", "robot", "speedometer2",
                    "clock-history", "gear-fill"
                ],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#e94560", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "padding": "10px",
                        "color": "#e8e8e8",
                        "--hover-color": "#2a2a3e",
                        "border-radius": "10px",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, #e94560 0%, #c13651 100%)",
                        "color": "white",
                        "font-weight": "600",
                    },
                }
            )
            
            # Add quick stats with dark theme
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", st.session_state.get('file_count', 0))
            with col2:
                st.metric("Analyses", st.session_state.get('analysis_count', 0))
            
            return selected
    
    @staticmethod
    def render_file_upload_card():
        """Render file upload interface with dark theme"""
        st.markdown("""
        <div class="dark-card" style="border: 2px dashed #e94560; background-color: #1e1e2e;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: #e94560; margin-bottom: 1rem;">📁</div>
                <h3 style="color: #ffffff; margin: 0;">Drop your files here</h3>
                <p style="color: #a8a8c8; margin-top: 0.5rem;">or click to browse</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        from file_handlers import FileHandler
        uploaded_files = st.file_uploader(
            "Choose files",
            type=FileHandler.get_supported_extensions(),
            accept_multiple_files=True,
            key="file_uploader_dark",
            label_visibility="collapsed"
        )
        
        # Show supported formats with dark theme
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📄 PDF</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📝 Word</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📊 Excel</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📑 CSV</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">🔤 Text</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">🌐 HTML</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📋 JSON</span>
            </div>
            <div class="feature-card-dark" style="text-align: center; padding: 0.5rem;">
                <span style="color: #e94560; font-weight: 600;">📰 Markdown</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return uploaded_files
    
    @staticmethod
    def render_analysis_card(title: str, icon: str, description: str, button_text: str = "Analyze") -> bool:
        """Render analysis option card with dark theme"""
        st.markdown(f"""
        <div class="feature-card-dark">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <h3 style="color: #ffffff; font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0;">
                {title}
            </h3>
            <p style="color: #a8a8c8; font-size: 0.95rem; line-height: 1.5;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.button(button_text, key=f"btn_{title}", use_container_width=True, type="primary")
    
    @staticmethod
    def render_dashboard():
        """Render dashboard with dark theme"""
        # Calculate metrics
        file_count = len(st.session_state.get('file_names', []))
        analysis_count = len(st.session_state.get('analysis_results', []))
        theme_count = len(st.session_state.get('themes', {}))
        insight_count = len(st.session_state.get('insights', {}))
        
        # Render metric cards with dark theme
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="dark-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #e94560;">
                    {file_count}
                </div>
                <div style="color: #a8a8c8; font-size: 1rem; font-weight: 500;">
                    Files Processed
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dark-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #e94560;">
                    {analysis_count}
                </div>
                <div style="color: #a8a8c8; font-size: 1rem; font-weight: 500;">
                    Analyses Run
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="dark-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #e94560;">
                    {theme_count}
                </div>
                <div style="color: #a8a8c8; font-size: 1rem; font-weight: 500;">
                    Themes Found
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="dark-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💡</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #e94560;">
                    {insight_count}
                </div>
                <div style="color: #a8a8c8; font-size: 1rem; font-weight: 500;">
                    Insights Generated
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity with dark theme
        if st.session_state.get('analysis_results'):
            st.markdown("### 📊 Recent Activity")
            
            recent = st.session_state.analysis_results[-5:]
            for item in reversed(recent):
                st.markdown(f"""
                <div class="dark-card" style="padding: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #e94560;">📄 {item.get('file_id', 'Unknown')[:8]}...</span>
                        <span style="color: #a8a8c8;">🔍 {item.get('analysis_type', 'Unknown')}</span>
                        <span style="color: #888;">🕐 {item.get('timestamp', 'Unknown')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_footer():
        """Render footer with dark theme"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e94560 0%, #0f3460 100%); 
                    color: white; padding: 2rem; border-radius: 15px; 
                    margin-top: 3rem; text-align: center; 
                    box-shadow: 0 -5px 20px rgba(0,0,0,0.3);">
            <p style="margin: 0.5rem 0; font-size: 1rem; color: white;">
                © 2024 NLP Tool for YPAR | Version 3.0
            </p>
            <p style="margin: 0.5rem 0; font-size: 1rem; color: white;">
                🚀 Powered by Advanced AI & Open Source NLP
            </p>
            <div style="margin-top: 1rem;">
                <a href="#" style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                    Documentation
                </a>
                <a href="#" style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                    Support
                </a>
                <a href="#" style="color: white; text-decoration: none; margin: 0 1rem; font-weight: 500;">
                    About
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)