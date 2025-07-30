#!/usr/bin/env python3
"""
Setup script for Scotland Birth Forecast UI
This script prepares the trained model and launches the Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages for the UI"""
    print("📦 Installing UI requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ui.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def prepare_model():
    """Train and save the model for UI use"""
    print("🤖 Preparing trained model for UI...")
    
    # Check if model files already exist
    model_files = [
        "models/trained_lstm_model.pth",
        "models/feature_scaler.pkl", 
        "models/nhs_board_mapping.pkl"
    ]
    
    if all(os.path.exists(f) for f in model_files):
        print("✅ Model files already exist!")
        return True
    
    try:
        # Run the model training and saving script
        subprocess.check_call([sys.executable, "save_model_for_ui.py"])
        print("✅ Model prepared successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error preparing model: {e}")
        return False

def launch_ui():
    """Launch the Streamlit UI"""
    print("🚀 Launching Scotland Birth Forecast UI...")
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 UI closed by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

def main():
    """Main setup and launch process"""
    print("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast - UI Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 2: Prepare model
    if not prepare_model():
        print("❌ Setup failed at model preparation")
        return
    
    # Step 3: Launch UI
    print("\n" + "=" * 50)
    print("🎉 Setup complete! Launching UI...")
    print("=" * 50)
    launch_ui()

if __name__ == "__main__":
    main()
