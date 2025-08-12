"""
Enhanced UI components and layouts for NLP Tool
"""
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import pandas as pd

class UIComponents:
    """Modern UI components for the application"""
    
    @staticmethod
    def render_modern_header():
        """Render modern header with gradient"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .main-title {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .main-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        </style>
        <div class="main-header">
            <h1 class="main-title">🔬 NLP Tool for YPAR</h1>
            <p class="main-subtitle">Advanced Natural Language Processing for Youth Participatory Action Research</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_menu() -> str:
        """Render modern sidebar navigation menu"""
        with st.sidebar:
            st.markdown("""
            <style>
            .sidebar-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                text-align: center;
            }
            .sidebar-title {
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
            }
            </style>
            <div class="sidebar-header">
                <h2 class="sidebar-title">Navigation</h2>
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
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#667eea", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "padding": "10px",
                        "--hover-color": "#f0f2f6",
                        "border-radius": "10px",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "color": "white",
                        "font-weight": "600",
                    },
                }
            )
            
            # Add quick stats
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
        """Render modern file upload interface"""
        st.markdown("""
        <style>
        .upload-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 2px dashed #667eea;
            transition: all 0.3s ease;
        }
        .upload-card:hover {
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-color: #764ba2;
        }
        .upload-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .upload-icon {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 1rem;
        }
        .format-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        .format-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            text-align: center;
            font-weight: 600;
            font-size: 0.9rem;
            box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="upload-header">
                <div class="upload-icon">📁</div>
                <h3 style="color: #333; margin: 0;">Drop your files here</h3>
                <p style="color: #666; margin-top: 0.5rem;">or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
            
            # File uploader
            from file_handlers import FileHandler
            uploaded_files = st.file_uploader(
                "Choose files",
                type=FileHandler.get_supported_extensions(),
                accept_multiple_files=True,
                key="file_uploader_modern",
                label_visibility="collapsed"
            )
            
            # Show supported formats
            st.markdown("""
            <div class="format-grid">
                <div class="format-badge">📄 PDF</div>
                <div class="format-badge">📝 Word</div>
                <div class="format-badge">📊 Excel</div>
                <div class="format-badge">📑 CSV</div>
                <div class="format-badge">🔤 Text</div>
                <div class="format-badge">🌐 HTML</div>
                <div class="format-badge">📋 JSON</div>
                <div class="format-badge">📰 Markdown</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return uploaded_files
    
    @staticmethod
    def render_file_preview(file_name: str, content: str, metadata: Dict[str, Any]):
        """Render file preview with metadata"""
        with st.expander(f"📄 {file_name}", expanded=False):
            # Metadata tabs
            tab1, tab2, tab3 = st.tabs(["📖 Preview", "📊 Metadata", "📈 Statistics"])
            
            with tab1:
                # Show preview
                from file_handlers import FileHandler
                preview = FileHandler.get_file_preview(content, max_length=1000)
                st.text_area("Content Preview", preview, height=300, disabled=True)
            
            with tab2:
                # Show metadata
                if metadata:
                    for key, value in metadata.items():
                        if key != "error":
                            if isinstance(value, list) and len(value) > 5:
                                st.write(f"**{key}:** {len(value)} items")
                            else:
                                st.write(f"**{key}:** {value}")
            
            with tab3:
                # Show statistics
                from utils import get_word_statistics
                stats = get_word_statistics(content)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", f"{stats['word_count']:,}")
                with col2:
                    st.metric("Sentences", f"{stats['sentence_count']:,}")
                with col3:
                    st.metric("Unique Words", f"{stats['unique_words']:,}")
                
                # Vocabulary richness gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stats['vocabulary_richness'] * 100,
                    title={'text': "Vocabulary Richness"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 25], 'color': "#ffebee"},
                            {'range': [25, 50], 'color': "#fff3e0"},
                            {'range': [50, 75], 'color': "#e8f5e9"},
                            {'range': [75, 100], 'color': "#e1f5fe"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_analysis_card(title: str, icon: str, description: str, button_text: str = "Analyze") -> bool:
        """Render analysis option card"""
        st.markdown(f"""
        <style>
        .analysis-card {{
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }}
        .analysis-card:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        .analysis-icon {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .analysis-title {{
            color: #333;
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }}
        .analysis-desc {{
            color: #666;
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown(f"""
            <div class="analysis-card">
                <div class="analysis-icon">{icon}</div>
                <h3 class="analysis-title">{title}</h3>
                <p class="analysis-desc">{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            return st.button(button_text, key=f"btn_{title}", use_container_width=True, type="primary")
    
    @staticmethod
    def render_dashboard():
        """Render main dashboard with metrics"""
        st.markdown("""
        <style>
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #666;
            font-size: 1rem;
            font-weight: 500;
        }
        .metric-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        file_count = len(st.session_state.get('file_names', []))
        analysis_count = len(st.session_state.get('analysis_results', []))
        theme_count = len(st.session_state.get('themes', {}))
        insight_count = len(st.session_state.get('insights', {}))
        
        # Render metric cards
        st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📁</div>
                <div class="metric-value">{file_count}</div>
                <div class="metric-label">Files Processed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔍</div>
                <div class="metric-value">{analysis_count}</div>
                <div class="metric-label">Analyses Run</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{theme_count}</div>
                <div class="metric-label">Themes Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💡</div>
                <div class="metric-value">{insight_count}</div>
                <div class="metric-label">Insights Generated</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        if st.session_state.get('analysis_results'):
            st.markdown("### 📊 Recent Activity")
            
            recent = st.session_state.analysis_results[-5:]
            for item in reversed(recent):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"📄 {item.get('file_id', 'Unknown')[:8]}...")
                    with col2:
                        st.write(f"🔍 {item.get('analysis_type', 'Unknown')}")
                    with col3:
                        st.write(f"🕐 {item.get('timestamp', 'Unknown')}")
    
    @staticmethod
    def render_progress_ring(value: float, label: str):
        """Render circular progress indicator"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': label},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#f0f0f0'},
                    {'range': [50, 100], 'color': '#e8eaf6'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#333", 'family': "Arial"}
        )
        
        return fig
    
    @staticmethod
    def render_footer():
        """Render modern footer"""
        st.markdown("""
        <style>
        .footer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 3rem;
            text-align: center;
            box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
        }
        .footer-text {
            margin: 0.5rem 0;
            font-size: 1rem;
        }
        .footer-links {
            margin-top: 1rem;
        }
        .footer-links a {
            color: white;
            text-decoration: none;
            margin: 0 1rem;
            font-weight: 500;
            transition: opacity 0.3s;
        }
        .footer-links a:hover {
            opacity: 0.8;
        }
        </style>
        <div class="footer">
            <p class="footer-text">© 2024 NLP Tool for YPAR | Version 3.0</p>
            <p class="footer-text">🚀 Powered by Advanced AI & Open Source NLP</p>
            <div class="footer-links">
                <a href="#">Documentation</a>
                <a href="#">Support</a>
                <a href="#">About</a>
                <a href="#">Privacy</a>
            </div>
        </div>
        """, unsafe_allow_html=True)