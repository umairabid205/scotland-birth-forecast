#!/usr/bin/env python3
"""
Quick launch script for the Scotland Birth Forecast UI
Run this after the model training is complete
"""

import subprocess
import sys
import os

def check_model_files():
    """Check if all required model files exist"""
    required_files = [
        "models/trained_lstm_model.pth",
        "models/feature_scaler.pkl", 
        "models/nhs_board_mapping.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing model files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Run 'python save_model_for_ui.py' first to train and save the model.")
        return False
    
    print("✅ All model files found!")
    return True

def launch_streamlit():
    """Launch the Streamlit app"""
    print("🚀 Launching Scotland Birth Forecast UI...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Since you're already in the conda environment, just run streamlit directly
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 UI server stopped by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        print("💡 Make sure you're in the conda environment and streamlit is installed:")
        print("   conda activate /Users/umair/Downloads/projects/project_1/venv")
        print("   pip install streamlit")

def main():
    print("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast - Quick Launch")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Check model files
    if not check_model_files():
        return
    
    # Launch the UI
    launch_streamlit()

if __name__ == "__main__":
    main()
