"""
UC Berkeley themed UI components
Colors: Berkeley Blue (#003262) and California Gold (#FDB515)
"""
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import pandas as pd

class UIComponents:
    """UC Berkeley themed UI components"""
    
    # UC Berkeley Colors
    BERKELEY_BLUE = "#003262"
    CALIFORNIA_GOLD = "#FDB515"
    FOUNDERS_ROCK = "#3B7EA1"  # Secondary blue
    MEDALIST = "#C4820E"  # Secondary gold
    SATHER_GATE = "#B9D3B6"  # Light accent
    BAY_FOG = "#D2C295"  # Neutral
    LAWRENCE = "#00B0DA"  # Light blue accent
    
    @staticmethod
    def apply_berkeley_theme():
        """Apply UC Berkeley themed styling"""
        st.markdown(f"""
        <style>
        /* Global Berkeley Theme */
        .stApp {{
            background: linear-gradient(180deg, #001a33 0%, {UIComponents.BERKELEY_BLUE} 100%);
            color: #ffffff;
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: rgba(0, 50, 98, 0.95);
            padding: 2rem;
            border-radius: 10px;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        /* Sidebar - Berkeley Blue */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {UIComponents.BERKELEY_BLUE} 0%, #001a33 100%) !important;
            border-right: 3px solid {UIComponents.CALIFORNIA_GOLD};
        }}
        
        .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {{
            color: #ffffff !important;
        }}
        
        /* Headers with California Gold accent */
        h1, h2, h3 {{
            color: {UIComponents.CALIFORNIA_GOLD} !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        h4, h5, h6 {{
            color: #ffffff !important;
        }}
        
        /* Text */
        p, span, div, label {{
            color: #ffffff !important;
        }}
        
        /* Input fields with Berkeley styling */
        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            border: 2px solid {UIComponents.FOUNDERS_ROCK};
            border-radius: 5px;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {UIComponents.CALIFORNIA_GOLD};
            box-shadow: 0 0 0 2px rgba(253, 181, 21, 0.2);
        }}
        
        .stSelectbox > div > div > div {{
            background-color: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            border: 2px solid {UIComponents.FOUNDERS_ROCK};
        }}
        
        /* Buttons - California Gold */
        .stButton > button {{
            background: linear-gradient(135deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
            color: {UIComponents.BERKELEY_BLUE};
            border: none;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(253, 181, 21, 0.3);
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {UIComponents.MEDALIST} 0%, {UIComponents.CALIFORNIA_GOLD} 100%);
            box-shadow: 0 6px 12px rgba(253, 181, 21, 0.5);
            transform: translateY(-2px);
        }}
        
        /* File uploader */
        .stFileUploader {{
            background-color: rgba(0, 50, 98, 0.5);
            border: 3px dashed {UIComponents.CALIFORNIA_GOLD};
            border-radius: 10px;
            padding: 2rem;
        }}
        
        .stFileUploader:hover {{
            background-color: rgba(253, 181, 21, 0.05);
            border-color: {UIComponents.MEDALIST};
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background: linear-gradient(90deg, {UIComponents.BERKELEY_BLUE} 0%, {UIComponents.FOUNDERS_ROCK} 100%);
            color: {UIComponents.CALIFORNIA_GOLD};
            border-radius: 5px;
            border: 1px solid {UIComponents.CALIFORNIA_GOLD};
        }}
        
        .streamlit-expanderContent {{
            background-color: rgba(0, 50, 98, 0.3);
            border: 1px solid {UIComponents.FOUNDERS_ROCK};
            border-top: 3px solid {UIComponents.CALIFORNIA_GOLD};
        }}
        
        /* Metrics - Berkeley styled */
        [data-testid="metric-container"] {{
            background: linear-gradient(135deg, rgba(0, 50, 98, 0.9) 0%, rgba(59, 126, 161, 0.9) 100%);
            border: 2px solid {UIComponents.CALIFORNIA_GOLD};
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        [data-testid="metric-container"] label {{
            color: {UIComponents.CALIFORNIA_GOLD} !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }}
        
        [data-testid="metric-container"] [data-testid="metric-value"] {{
            color: #ffffff !important;
            font-weight: 700;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: rgba(0, 50, 98, 0.5);
            border-radius: 10px;
            border: 1px solid {UIComponents.FOUNDERS_ROCK};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: #ffffff;
            background-color: transparent;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
            color: {UIComponents.BERKELEY_BLUE};
            font-weight: 700;
        }}
        
        /* Tables - Berkeley themed */
        .dataframe {{
            background-color: rgba(0, 50, 98, 0.5) !important;
            color: #ffffff !important;
            border: 1px solid {UIComponents.CALIFORNIA_GOLD};
        }}
        
        .dataframe th {{
            background: linear-gradient(90deg, {UIComponents.BERKELEY_BLUE} 0%, {UIComponents.FOUNDERS_ROCK} 100%) !important;
            color: {UIComponents.CALIFORNIA_GOLD} !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .dataframe td {{
            background-color: rgba(0, 50, 98, 0.3) !important;
            color: #ffffff !important;
            border-color: {UIComponents.FOUNDERS_ROCK} !important;
        }}
        
        /* Progress bars - Gold */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
        }}
        
        /* Info boxes */
        .stAlert {{
            background: linear-gradient(135deg, rgba(0, 50, 98, 0.9) 0%, rgba(59, 126, 161, 0.9) 100%);
            color: #ffffff;
            border-left: 4px solid {UIComponents.CALIFORNIA_GOLD};
            border-radius: 5px;
        }}
        
        /* Custom Berkeley classes */
        .berkeley-card {{
            background: linear-gradient(135deg, rgba(0, 50, 98, 0.95) 0%, rgba(59, 126, 161, 0.95) 100%);
            border: 2px solid {UIComponents.CALIFORNIA_GOLD};
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }}
        
        .berkeley-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
        }}
        
        .berkeley-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(253, 181, 21, 0.2);
            border-color: {UIComponents.MEDALIST};
        }}
        
        .gold-accent {{
            color: {UIComponents.CALIFORNIA_GOLD};
            font-weight: 700;
        }}
        
        .berkeley-header {{
            background: linear-gradient(135deg, {UIComponents.BERKELEY_BLUE} 0%, {UIComponents.FOUNDERS_ROCK} 100%);
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            border: 3px solid {UIComponents.CALIFORNIA_GOLD};
            position: relative;
            overflow: hidden;
        }}
        
        .berkeley-header::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
        }}
        
        /* Berkeley styled scrollbar */
        ::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {UIComponents.BERKELEY_BLUE};
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%);
            border-radius: 10px;
            border: 2px solid {UIComponents.BERKELEY_BLUE};
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {UIComponents.MEDALIST};
        }}
        
        /* Links */
        a {{
            color: {UIComponents.CALIFORNIA_GOLD} !important;
            text-decoration: none;
            font-weight: 600;
        }}
        
        a:hover {{
            color: {UIComponents.MEDALIST} !important;
            text-decoration: underline;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_modern_header():
        """Render header with Club Stride branding"""
        st.markdown(f"""
        <div class="berkeley-header">
            <h1 style="color: {UIComponents.CALIFORNIA_GOLD}; font-size: 3rem; font-weight: 700; margin: 0; text-align: center; text-shadow: 3px 3px 6px rgba(0,0,0,0.5);">
                🚀 NLP Tool for YPAR
            </h1>
            <p style="color: #ffffff; font-size: 1.3rem; text-align: center; margin-top: 0.8rem; font-weight: 300; letter-spacing: 2px;">
                CLUB STRIDE
            </p>
            <p style="color: rgba(255,255,255,0.9); font-size: 1rem; text-align: center; margin-top: 0.5rem;">
                Advanced Natural Language Processing for Youth Participatory Action Research
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_menu() -> str:
        """Render sidebar with Berkeley theme"""
        with st.sidebar:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <h2 style="color: {UIComponents.BERKELEY_BLUE}; font-size: 1.5rem; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 2px;">
                    Navigation
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            selected = option_menu(
                menu_title=None,
                options=[
                    "🏠 Home",
                    "📤 Upload & Process",
                    "🔍 Analysis Suite",
                    "🧠 RAG Intelligence",
                    "👤 Personas",
                    "📊 Visualizations",
                    "🤖 AI Insights",
                    "📈 Dashboard",
                    "📚 History",
                    "⚙️ Settings"
                ],
                icons=[
                    "house-fill", "cloud-upload-fill", "search",
                    "brain", "person-badge", "bar-chart-fill", 
                    "robot", "speedometer2", "clock-history", "gear-fill"
                ],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": UIComponents.CALIFORNIA_GOLD, "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "5px",
                        "padding": "10px",
                        "color": "#ffffff",
                        "background-color": "rgba(255,255,255,0.05)",
                        "--hover-color": "rgba(253, 181, 21, 0.1)",
                        "border-radius": "10px",
                        "border": "1px solid transparent",
                    },
                    "nav-link-selected": {
                        "background": f"linear-gradient(135deg, {UIComponents.CALIFORNIA_GOLD} 0%, {UIComponents.MEDALIST} 100%)",
                        "color": UIComponents.BERKELEY_BLUE,
                        "font-weight": "700",
                        "border": f"2px solid {UIComponents.BERKELEY_BLUE}",
                    },
                }
            )
            
            # Quick stats with Berkeley styling
            st.markdown("---")
            st.markdown(f"### <span class='gold-accent'>📊 Quick Stats</span>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", st.session_state.get('file_count', 0))
            with col2:
                st.metric("Analyses", st.session_state.get('analysis_count', 0))
            
            # Add Club Stride branding
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <p style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700; margin: 0;">CLUB STRIDE</p>
                <p style="color: #ffffff; font-size: 0.8rem; margin-top: 0.5rem;">Empowering Youth Through Research</p>
            </div>
            """, unsafe_allow_html=True)
            
            return selected
    
    @staticmethod
    def render_file_upload_card():
        """Render file upload with Berkeley theme"""
        st.markdown(f"""
        <div class="berkeley-card" style="border: 3px dashed {UIComponents.CALIFORNIA_GOLD}; background: rgba(0, 50, 98, 0.3);">
            <div style="text-align: center;">
                <div style="font-size: 3rem; color: {UIComponents.CALIFORNIA_GOLD}; margin-bottom: 1rem;">📁</div>
                <h3 style="color: {UIComponents.CALIFORNIA_GOLD}; margin: 0; font-weight: 700;">Drop your files here</h3>
                <p style="color: #ffffff; margin-top: 0.5rem; opacity: 0.9;">or click to browse</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        from file_handlers import FileHandler
        uploaded_files = st.file_uploader(
            "Choose files",
            type=FileHandler.get_supported_extensions(),
            accept_multiple_files=True,
            key="file_uploader_berkeley",
            label_visibility="collapsed"
        )
        
        # Supported formats grid
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📄 PDF</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📝 Word</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📊 Excel</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📑 CSV</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">🔤 Text</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">🌐 HTML</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📋 JSON</span>
            </div>
            <div class="berkeley-card" style="text-align: center; padding: 0.8rem;">
                <span style="color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700;">📰 Markdown</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return uploaded_files
    
    @staticmethod
    def render_analysis_card(title: str, icon: str, description: str, button_text: str = "Analyze") -> bool:
        """Render analysis card with Berkeley theme"""
        st.markdown(f"""
        <div class="berkeley-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <h3 style="color: {UIComponents.CALIFORNIA_GOLD}; font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0; text-transform: uppercase; letter-spacing: 1px;">
                {title}
            </h3>
            <p style="color: #ffffff; font-size: 0.95rem; line-height: 1.5; opacity: 0.9;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.button(button_text, key=f"btn_{title}", use_container_width=True, type="primary")
    
    @staticmethod
    def render_dashboard():
        """Render dashboard with Berkeley theme"""
        file_count = len(st.session_state.get('file_names', []))
        analysis_count = len(st.session_state.get('analysis_results', []))
        theme_count = len(st.session_state.get('themes', {}))
        insight_count = len(st.session_state.get('insights', {}))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="berkeley-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {UIComponents.CALIFORNIA_GOLD};">
                    {file_count}
                </div>
                <div style="color: #ffffff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Files Processed
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="berkeley-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {UIComponents.CALIFORNIA_GOLD};">
                    {analysis_count}
                </div>
                <div style="color: #ffffff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Analyses Run
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="berkeley-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {UIComponents.CALIFORNIA_GOLD};">
                    {theme_count}
                </div>
                <div style="color: #ffffff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Themes Found
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="berkeley-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💡</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {UIComponents.CALIFORNIA_GOLD};">
                    {insight_count}
                </div>
                <div style="color: #ffffff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    Insights Generated
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_footer():
        """Render footer with Club Stride branding"""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {UIComponents.BERKELEY_BLUE} 0%, {UIComponents.FOUNDERS_ROCK} 100%); 
                    color: white; padding: 2.5rem; border-radius: 15px; 
                    margin-top: 3rem; text-align: center; 
                    box-shadow: 0 -5px 20px rgba(0,0,0,0.3);
                    border: 3px solid {UIComponents.CALIFORNIA_GOLD};">
            <p style="margin: 0.5rem 0; font-size: 1.2rem; color: {UIComponents.CALIFORNIA_GOLD}; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">
                CLUB STRIDE
            </p>
            <p style="margin: 0.5rem 0; font-size: 1rem; color: white;">
                © 2024 NLP Tool for YPAR | Version 3.0
            </p>
            <p style="margin: 0.5rem 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);">
                🚀 Empowering Youth Through Research & Innovation
            </p>
            <div style="margin-top: 1.5rem;">
                <a href="#" style="color: {UIComponents.CALIFORNIA_GOLD}; text-decoration: none; margin: 0 1rem; font-weight: 600;">
                    Documentation
                </a>
                <a href="#" style="color: {UIComponents.CALIFORNIA_GOLD}; text-decoration: none; margin: 0 1rem; font-weight: 600;">
                    Research
                </a>
                <a href="#" style="color: {UIComponents.CALIFORNIA_GOLD}; text-decoration: none; margin: 0 1rem; font-weight: 600;">
                    Support
                </a>
                <a href="#" style="color: {UIComponents.CALIFORNIA_GOLD}; text-decoration: none; margin: 0 1rem; font-weight: 600;">
                    About Club Stride
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)