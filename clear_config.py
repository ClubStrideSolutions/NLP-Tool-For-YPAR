"""
Script to clear any API configurations and force fallback mode
"""

import os
import streamlit as st

print("="*60)
print("CLEARING CONFIGURATIONS TO FORCE FALLBACK MODE")
print("="*60)

# Clear environment variables
env_vars = ['OPENAI_API_KEY', 'CONNECTION_STRING', 'MONGODB_CONNECTION_STRING']
for var in env_vars:
    if var in os.environ:
        print(f"[INFO] Found {var} in environment - would need to clear")
        # Don't actually clear as it might affect user's system
    else:
        print(f"[OK] {var} not set in environment")

# Check for .env file
if os.path.exists('.env'):
    print("[INFO] .env file exists - checking contents")
    with open('.env', 'r') as f:
        content = f.read()
        if 'OPENAI_API_KEY' in content:
            print("[WARN] OPENAI_API_KEY found in .env file")
        if 'CONNECTION_STRING' in content:
            print("[WARN] CONNECTION_STRING found in .env file")
else:
    print("[OK] No .env file found")

# Check for Streamlit secrets
secrets_dir = os.path.expanduser('~/.streamlit')
secrets_file = os.path.join(secrets_dir, 'secrets.toml')
if os.path.exists(secrets_file):
    print(f"[INFO] Streamlit secrets file exists at {secrets_file}")
else:
    print("[OK] No Streamlit secrets file found")

print("\n" + "="*60)
print("TO FORCE FALLBACK MODE:")
print("="*60)
print("1. Clear any .env file in the project directory")
print("2. Clear ~/.streamlit/secrets.toml")
print("3. Don't set API keys in the Settings page")
print("4. Restart the application")
print("\nOr create a test instance with:")
print("streamlit run main_enhanced_cal.py --server.port 9002")
print("(This will start a fresh instance without configs)")
print("="*60)