"""
Streamlit Cloud Safe Entry Point - No MongoDB
"""

import os

# Disable MongoDB completely for Streamlit Cloud
os.environ['DISABLE_MONGODB'] = 'true'

# Import the fixed main module
import main_enhanced_cal_fixed

# Always run the main function
main_enhanced_cal_fixed.main()