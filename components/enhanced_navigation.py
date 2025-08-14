"""
Enhanced Navigation Component with Heavy HTML/CSS Styling
Cal Colors Theme with Professional UI
"""

import streamlit as st
from typing import List, Dict, Optional, Tuple

class EnhancedNavigation:
    """Enhanced navigation with heavy HTML styling"""
    
    @staticmethod
    def inject_navigation_styles():
        """Inject comprehensive navigation CSS"""
        st.markdown("""
        <style>
        /* Enhanced Navigation Styles */
        .nav-wrapper {
            background: linear-gradient(135deg, #003262 0%, #004d8a 100%);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 50, 98, 0.3);
            margin-bottom: 2rem;
        }
        
        .nav-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem;
            background: rgba(253, 181, 21, 0.1);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .nav-title {
            color: #FDB515;
            font-size: 1.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            margin: 0;
        }
        
        .nav-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 0.9rem;
            margin: 0;
        }
        
        .nav-items {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 0.5rem;
        }
        
        .nav-item {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 0.75rem 1.25rem;
            border-radius: 10px;
            text-decoration: none;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 500;
        }
        
        .nav-item:hover {
            background: rgba(253, 181, 21, 0.2);
            border-color: #FDB515;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(253, 181, 21, 0.3);
        }
        
        .nav-item.active {
            background: #FDB515;
            color: #003262;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(253, 181, 21, 0.5);
        }
        
        .nav-icon {
            font-size: 1.2rem;
        }
        
        .nav-badge {
            background: #DC3545;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        
        /* Sidebar Navigation Styles */
        .sidebar-nav {
            background: linear-gradient(180deg, #003262 0%, #004d8a 100%);
            padding: 1rem;
            border-radius: 15px;
            height: 100%;
        }
        
        .sidebar-nav-item {
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            color: white;
            text-decoration: none;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
        }
        
        .sidebar-nav-item:hover {
            background: rgba(253, 181, 21, 0.15);
            border-left-color: #FDB515;
            padding-left: 1.5rem;
        }
        
        .sidebar-nav-item.active {
            background: linear-gradient(90deg, rgba(253, 181, 21, 0.3), rgba(253, 181, 21, 0.1));
            border-left-color: #FDB515;
            font-weight: bold;
        }
        
        .sidebar-nav-icon {
            font-size: 1.5rem;
            margin-right: 1rem;
            color: #FDB515;
        }
        
        .sidebar-nav-text {
            flex: 1;
        }
        
        .sidebar-nav-indicator {
            width: 8px;
            height: 8px;
            background: #28A745;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        /* Tab Navigation Styles */
        .tab-nav-wrapper {
            background: white;
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .tab-nav-header {
            border-bottom: 3px solid #003262;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .tab-nav-items {
            display: flex;
            gap: 0.25rem;
            overflow-x: auto;
            padding-bottom: 0.5rem;
        }
        
        .tab-nav-item {
            padding: 0.75rem 1.5rem;
            background: #F8F9FA;
            border: 2px solid transparent;
            border-radius: 10px 10px 0 0;
            color: #003262;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
            position: relative;
        }
        
        .tab-nav-item:hover {
            background: rgba(0, 50, 98, 0.1);
            border-color: #003262;
        }
        
        .tab-nav-item.active {
            background: #003262;
            color: #FDB515;
            border-color: #003262;
        }
        
        .tab-nav-item.active::after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            right: 0;
            height: 3px;
            background: #FDB515;
        }
        
        /* Breadcrumb Navigation */
        .breadcrumb-nav {
            display: flex;
            align-items: center;
            padding: 1rem;
            background: rgba(0, 50, 98, 0.05);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .breadcrumb-item {
            color: #003262;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        
        .breadcrumb-item:hover {
            color: #FDB515;
        }
        
        .breadcrumb-separator {
            margin: 0 0.75rem;
            color: #6C757D;
        }
        
        .breadcrumb-current {
            color: #FDB515;
            font-weight: bold;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .nav-items {
                flex-direction: column;
            }
            
            .nav-item {
                width: 100%;
                justify-content: center;
            }
            
            .tab-nav-items {
                flex-direction: column;
            }
            
            .tab-nav-item {
                width: 100%;
                border-radius: 10px;
                margin-bottom: 0.25rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_main_navigation(items: List[Dict[str, str]], active: str = None) -> str:
        """
        Create main navigation bar
        items: List of dicts with 'label', 'icon', 'page' keys
        """
        nav_html = """
        <div class="nav-wrapper">
            <div class="nav-header">
                <div>
                    <h2 class="nav-title">🐻 NLP YPAR Tool</h2>
                    <p class="nav-subtitle">Youth Participatory Action Research Platform</p>
                </div>
            </div>
            <div class="nav-items">
        """
        
        for item in items:
            active_class = "active" if item.get('page') == active else ""
            badge = f'<span class="nav-badge">{item.get("badge", "")}</span>' if item.get("badge") else ""
            nav_html += f"""
                <div class="nav-item {active_class}" onclick="handleNavClick('{item['page']}')">
                    <span class="nav-icon">{item['icon']}</span>
                    <span>{item['label']}</span>
                    {badge}
                </div>
            """
        
        nav_html += """
            </div>
        </div>
        <script>
            function handleNavClick(page) {
                // This will be handled by Streamlit's session state
                console.log('Navigate to:', page);
            }
        </script>
        """
        
        return nav_html
    
    @staticmethod
    def create_sidebar_navigation(items: List[Dict[str, str]], active: str = None) -> str:
        """
        Create sidebar navigation
        items: List of dicts with 'label', 'icon', 'page', 'status' keys
        """
        nav_html = '<div class="sidebar-nav">'
        
        for item in items:
            active_class = "active" if item.get('page') == active else ""
            status_indicator = '<div class="sidebar-nav-indicator"></div>' if item.get('status') == 'online' else ''
            
            nav_html += f"""
                <div class="sidebar-nav-item {active_class}">
                    <span class="sidebar-nav-icon">{item['icon']}</span>
                    <span class="sidebar-nav-text">{item['label']}</span>
                    {status_indicator}
                </div>
            """
        
        nav_html += '</div>'
        return nav_html
    
    @staticmethod
    def create_tab_navigation(tabs: List[Tuple[str, str]], active_tab: int = 0) -> str:
        """
        Create tab navigation for analysis sections
        tabs: List of tuples (icon, label)
        """
        nav_html = """
        <div class="tab-nav-wrapper">
            <div class="tab-nav-header">
                <h3 style="margin: 0; color: #003262;">Analysis Options</h3>
            </div>
            <div class="tab-nav-items">
        """
        
        for idx, (icon, label) in enumerate(tabs):
            active_class = "active" if idx == active_tab else ""
            nav_html += f"""
                <div class="tab-nav-item {active_class}" data-tab="{idx}">
                    {icon} {label}
                </div>
            """
        
        nav_html += """
            </div>
        </div>
        """
        
        return nav_html
    
    @staticmethod
    def create_breadcrumb(path: List[str]) -> str:
        """
        Create breadcrumb navigation
        path: List of page names in order
        """
        nav_html = '<div class="breadcrumb-nav">'
        
        for idx, item in enumerate(path):
            if idx < len(path) - 1:
                nav_html += f"""
                    <a href="#" class="breadcrumb-item">{item}</a>
                    <span class="breadcrumb-separator">›</span>
                """
            else:
                nav_html += f'<span class="breadcrumb-current">{item}</span>'
        
        nav_html += '</div>'
        return nav_html
    
    @staticmethod
    def create_quick_actions(actions: List[Dict[str, str]]) -> str:
        """
        Create quick action buttons
        actions: List of dicts with 'label', 'icon', 'color' keys
        """
        html = """
        <div style="display: flex; gap: 1rem; padding: 1rem; background: white; 
                    border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        """
        
        for action in actions:
            color = action.get('color', '#003262')
            html += f"""
                <button style="flex: 1; padding: 1rem; background: {color}; color: white;
                              border: none; border-radius: 10px; font-weight: bold;
                              cursor: pointer; transition: all 0.3s ease;
                              display: flex; align-items: center; justify-content: center; gap: 0.5rem;"
                        onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)';"
                        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <span style="font-size: 1.5rem;">{action['icon']}</span>
                    <span>{action['label']}</span>
                </button>
            """
        
        html += '</div>'
        return html
    
    @staticmethod
    def create_analysis_navigation() -> str:
        """Create enhanced analysis navigation tabs"""
        return """
        <div class="tab-nav-wrapper">
            <div style="background: linear-gradient(135deg, #003262, #004d8a); 
                        padding: 1.5rem; border-radius: 10px 10px 0 0; margin: -1rem -1rem 1rem -1rem;">
                <h2 style="color: #FDB515; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🔬 Advanced Text Analysis Suite
                </h2>
                <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">
                    AI-Powered Natural Language Processing
                </p>
            </div>
            <div class="tab-nav-items">
                <div class="tab-nav-item active" data-tab="0">
                    <span style="font-size: 1.2rem;">🤖</span> AI Analysis
                </div>
                <div class="tab-nav-item" data-tab="1">
                    <span style="font-size: 1.2rem;">😊</span> Sentiment
                </div>
                <div class="tab-nav-item" data-tab="2">
                    <span style="font-size: 1.2rem;">🎯</span> Themes
                </div>
                <div class="tab-nav-item" data-tab="3">
                    <span style="font-size: 1.2rem;">🔑</span> Keywords
                </div>
                <div class="tab-nav-item" data-tab="4">
                    <span style="font-size: 1.2rem;">📝</span> Summary
                </div>
                <div class="tab-nav-item" data-tab="5">
                    <span style="font-size: 1.2rem;">💬</span> Quotes
                </div>
                <div class="tab-nav-item" data-tab="6">
                    <span style="font-size: 1.2rem;">💡</span> Insights
                </div>
                <div class="tab-nav-item" data-tab="7">
                    <span style="font-size: 1.2rem;">❓</span> Q&A
                </div>
            </div>
        </div>
        """
    
    @staticmethod
    def create_status_bar(stats: Dict[str, any]) -> str:
        """Create a status bar with system stats"""
        return f"""
        <div style="background: linear-gradient(90deg, #003262, #004d8a); 
                    padding: 1rem; border-radius: 10px; color: white;
                    display: flex; justify-content: space-around; align-items: center;
                    box-shadow: 0 4px 12px rgba(0, 50, 98, 0.3); margin-bottom: 2rem;">
            <div style="text-align: center;">
                <div style="color: #FDB515; font-size: 0.9rem; font-weight: 600;">Files Processed</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{stats.get('files', 0)}</div>
            </div>
            <div style="width: 2px; height: 40px; background: rgba(253, 181, 21, 0.3);"></div>
            <div style="text-align: center;">
                <div style="color: #FDB515; font-size: 0.9rem; font-weight: 600;">Analyses Run</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{stats.get('analyses', 0)}</div>
            </div>
            <div style="width: 2px; height: 40px; background: rgba(253, 181, 21, 0.3);"></div>
            <div style="text-align: center;">
                <div style="color: #FDB515; font-size: 0.9rem; font-weight: 600;">AI Status</div>
                <div style="font-size: 1rem; font-weight: bold;">
                    {'🟢 Active' if stats.get('ai_active') else '🔴 Inactive'}
                </div>
            </div>
            <div style="width: 2px; height: 40px; background: rgba(253, 181, 21, 0.3);"></div>
            <div style="text-align: center;">
                <div style="color: #FDB515; font-size: 0.9rem; font-weight: 600;">Database</div>
                <div style="font-size: 1rem; font-weight: bold;">
                    {'🟢 Connected' if stats.get('db_connected') else '🟡 Local'}
                </div>
            </div>
        </div>
        """

def render_enhanced_navigation(page_type: str = "main"):
    """Render the appropriate navigation based on page type"""
    nav = EnhancedNavigation()
    nav.inject_navigation_styles()
    
    if page_type == "main":
        # Main navigation items
        main_nav_items = [
            {"label": "Home", "icon": "🏠", "page": "home"},
            {"label": "Upload", "icon": "📤", "page": "upload", "badge": "New"},
            {"label": "Analysis", "icon": "🔬", "page": "analysis"},
            {"label": "Visualize", "icon": "📊", "page": "visualize"},
            {"label": "RAG", "icon": "🤖", "page": "rag"},
            {"label": "History", "icon": "📜", "page": "history"},
            {"label": "Settings", "icon": "⚙️", "page": "settings"}
        ]
        
        # Get current page from session state
        current_page = st.session_state.get('current_page', 'home')
        
        # Render navigation
        st.markdown(nav.create_main_navigation(main_nav_items, current_page), unsafe_allow_html=True)
        
        # Status bar
        stats = {
            'files': len(st.session_state.get('processed_data', [])),
            'analyses': len(st.session_state.get('analysis_results', [])),
            'ai_active': bool(st.session_state.get('openai_api_key')),
            'db_connected': st.session_state.get('db_connected', False)
        }
        st.markdown(nav.create_status_bar(stats), unsafe_allow_html=True)
    
    elif page_type == "analysis":
        # Analysis page navigation
        st.markdown(nav.create_analysis_navigation(), unsafe_allow_html=True)
    
    elif page_type == "sidebar":
        # Sidebar navigation
        sidebar_items = [
            {"label": "Dashboard", "icon": "📊", "page": "dashboard", "status": "online"},
            {"label": "Documents", "icon": "📄", "page": "documents"},
            {"label": "Analytics", "icon": "📈", "page": "analytics"},
            {"label": "Reports", "icon": "📑", "page": "reports"},
            {"label": "Settings", "icon": "⚙️", "page": "settings"}
        ]
        st.markdown(nav.create_sidebar_navigation(sidebar_items), unsafe_allow_html=True)