"""
Test script to verify fallback usage in live application
"""

import requests
import json

def test_live_app():
    """Test the live application at port 9001"""
    base_url = "http://localhost:9001"
    
    print("="*60)
    print("TESTING LIVE APPLICATION FALLBACK USAGE")
    print("="*60)
    
    try:
        # Test if app is running
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print(f"[OK] Application running at {base_url}")
            
            # Check for indicators in the response
            content = response.text.lower()
            
            # Check for fallback indicators
            if "traditional nlp" in content or "traditional analysis" in content:
                print("[INFO] Application shows Traditional NLP mode")
            elif "ai-powered" in content or "openai" in content:
                print("[INFO] Application shows AI-powered mode")
            
            if "local storage" in content:
                print("[INFO] Application using local storage")
            elif "db connected" in content or "mongodb" in content:
                print("[INFO] Application shows database connected")
            
            # Look for specific UI elements
            if "run complete analysis" in content:
                print("[OK] Traditional analysis button found - FALLBACK ACTIVE")
            
            if "configure openai api key" in content:
                print("[OK] OpenAI configuration prompt found - FALLBACK ACTIVE")
                
        else:
            print(f"[WARN] Application returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to {base_url}")
        print("Make sure the application is running")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
    
    print("\n" + "="*60)
    print("HOW TO VERIFY FALLBACKS MANUALLY:")
    print("="*60)
    print("1. Open http://localhost:9001 in your browser")
    print("2. Navigate to Settings page")
    print("3. Check that OpenAI API Key field is empty")
    print("4. Check that MongoDB Connection field is empty")
    print("5. Go to Text Analysis page")
    print("6. You should see:")
    print("   - 'Traditional NLP' indicator")
    print("   - 'Run Complete Analysis' button")
    print("   - No AI-specific tabs")
    print("7. Run an analysis - it should use TextBlob, YAKE, etc.")
    print("="*60)

if __name__ == "__main__":
    test_live_app()