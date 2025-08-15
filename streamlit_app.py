"""
Streamlit Cloud Optimized Entry Point
Lightweight launcher that avoids heavy imports
"""

import streamlit as st
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NLP Tool for YPAR",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application with lazy loading"""
    
    # Title
    st.title("🔬 NLP Tool for YPAR")
    st.markdown("Youth Participatory Action Research Platform")
    
    # Check if we can import the main app
    try:
        # Try to import main application
        st.info("Loading application modules...")
        
        # Import only what we need, when we need it
        import main_enhanced_cal
        
        st.success("Application loaded successfully!")
        st.markdown("---")
        
        # Run the main app
        st.write("Application is ready. Please use the navigation menu.")
        
    except ImportError as e:
        st.error(f"Import Error: {str(e)}")
        st.warning("The application couldn't load some dependencies.")
        
        # Show fallback interface
        st.markdown("---")
        st.subheader("🔧 Troubleshooting")
        
        with st.expander("View Error Details"):
            st.code(str(e))
        
        st.markdown("""
        ### Possible Solutions:
        1. Check that all dependencies are installed
        2. Some heavy dependencies might timeout on Streamlit Cloud
        3. Try running locally with: `streamlit run main_enhanced_cal.py`
        
        ### Minimal Dependencies for Fallback Mode:
        - streamlit
        - pandas
        - numpy
        - textblob
        - yake
        - scikit-learn
        """)
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Failed to start application: {e}")

if __name__ == "__main__":
    main()