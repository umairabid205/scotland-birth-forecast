#!/usr/bin/env python3
"""
Simple UI launch script - Creates placeholder model files if they don't exist
"""

import os
import sys
import subprocess
import torch
import pickle
import numpy as np

def create_placeholder_model():
    """Create placeholder model files for demonstration"""
    print("🔧 Creating placeholder model files for demo...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    try:
        # Create a simple placeholder model (for demo purposes)
        from src.components.model import StackedLSTM
        placeholder_model = StackedLSTM(input_size=11, hidden_size=128, num_layers=2)
        torch.save(placeholder_model.state_dict(), "models/trained_lstm_model.pth")
        print("✅ Placeholder model created")
        
        # Create placeholder scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit on dummy data
        dummy_data = np.random.randn(100, 11)
        scaler.fit(dummy_data)
        with open("models/feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Placeholder scaler created")
        
        # Create NHS Board mapping
        nhs_board_mapping = {
            'Ayrshire and Arran': 0, 'Borders': 1, 'Dumfries and Galloway': 2,
            'Fife': 3, 'Forth Valley': 4, 'Grampian': 5, 'Greater Glasgow and Clyde': 6,
            'Highland': 7, 'Lanarkshire': 8, 'Lothian': 9, 'Orkney': 10,
            'Shetland': 11, 'Tayside': 12, 'Western Isles': 13
        }
        with open("models/nhs_board_mapping.pkl", 'wb') as f:
            pickle.dump(nhs_board_mapping, f)
        print("✅ NHS Board mapping created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating placeholder files: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    required_files = [
        "models/trained_lstm_model.pth",
        "models/feature_scaler.pkl", 
        "models/nhs_board_mapping.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    return len(missing_files) == 0, missing_files

def main():
    print("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast - Quick Start")
    print("=" * 60)
    
    # Check if model files exist
    files_exist, missing = check_model_files()
    
    if not files_exist:
        print("📋 Model files not found. Creating placeholder files for demo...")
        if not create_placeholder_model():
            print("❌ Failed to create placeholder files")
            return
    else:
        print("✅ Model files found!")
    
    print("\n🚀 Launching Streamlit UI...")
    print("📱 The app will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 UI server stopped by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

if __name__ == "__main__":
    main()
