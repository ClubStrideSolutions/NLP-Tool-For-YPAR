"""
Streamlit Cloud entry point for NLP YPAR Tool
"""

import os

# Disable MongoDB on Streamlit Cloud to prevent NotImplementedError
os.environ['DISABLE_MONGODB'] = 'true'

# Import the fixed main module
import main_enhanced_cal_fixed

# Always run the main function (Streamlit Cloud doesn't use __main__)
main_enhanced_cal_fixed.main()