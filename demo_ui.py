#!/usr/bin/env python3
"""
Demo script for the Scotland Birth Forecast UI
This script demonstrates the UI functionality without requiring the full Streamlit interface
"""

import pandas as pd
import numpy as np
import torch
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_prediction():
    """Demonstrate a prediction using the same logic as the UI"""
    
    print("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast - Demo")
    print("=" * 50)
    
    # Demo parameters
    demo_year = 2024
    demo_month = 6  # June
    demo_nhs_board = "Greater Glasgow and Clyde"
    
    print(f"📊 Demo Parameters:")
    print(f"   Year: {demo_year}")
    print(f"   Month: {demo_month} (June)")
    print(f"   NHS Board: {demo_nhs_board}")
    print()
    
    # NHS Board mapping (same as in the UI)
    nhs_board_mapping = {
        'Ayrshire and Arran': 0, 'Borders': 1, 'Dumfries and Galloway': 2,
        'Fife': 3, 'Forth Valley': 4, 'Grampian': 5, 'Greater Glasgow and Clyde': 6,
        'Highland': 7, 'Lanarkshire': 8, 'Lothian': 9, 'Orkney': 10,
        'Shetland': 11, 'Tayside': 12, 'Western Isles': 13
    }
    
    # Preprocess input (same logic as UI)
    month_num = demo_month
    nhs_board_code = nhs_board_mapping.get(demo_nhs_board, 0)
    
    # Cyclical encoding for month
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)
    
    # Normalize year (based on training range 1998-2022)
    year_norm = (demo_year - 1998) / (2022 - 1998)
    
    # Create lag features (using historical averages as placeholders)
    avg_births = 6.2  # Log-transformed average from training data
    birth_lag_1 = avg_births
    birth_lag_2 = avg_births
    birth_lag_3 = avg_births
    birth_rolling_avg = avg_births
    birth_trend = 0.0
    
    # Create feature vector
    features = np.array([[
        demo_year, month_num, nhs_board_code, month_sin, month_cos, year_norm,
        birth_lag_1, birth_lag_2, birth_lag_3, birth_rolling_avg, birth_trend
    ]], dtype=np.float32)
    
    print(f"🔧 Preprocessed Features:")
    feature_names = [
        'Year', 'Month_num', 'NHS_Board_code', 'Month_sin', 'Month_cos', 'Year_norm',
        'Birth_lag_1', 'Birth_lag_2', 'Birth_lag_3', 'Birth_rolling_avg', 'Birth_trend'
    ]
    
    for i, (name, value) in enumerate(zip(feature_names, features[0])):
        print(f"   {name}: {value:.4f}")
    print()
    
    # Simulate prediction (since model might still be training)
    # In the real UI, this would use the trained model
    log_prediction = 6.1822  # Example log-scaled prediction
    prediction = np.expm1(log_prediction)  # Convert back from log scale
    prediction = max(0, int(round(prediction)))  # Ensure non-negative integer
    
    print(f"🔮 Prediction Result:")
    print(f"   Log-scale prediction: {log_prediction:.4f}")
    print(f"   Final prediction: {prediction:,} births")
    print()
    
    print(f"📋 Summary:")
    print(f"   Expected birth registrations in {demo_nhs_board}")
    print(f"   for June {demo_year}: {prediction:,} births")
    print()
    
    print("🚀 UI Features:")
    print("   ✅ Interactive web interface")
    print("   ✅ Real-time predictions")
    print("   ✅ All 14 NHS Board areas supported")
    print("   ✅ Monthly resolution forecasting")
    print("   ✅ Professional styling and user experience")
    print()
    
    print("📱 To launch the full UI:")
    print("   1. Wait for model training to complete")
    print("   2. Run: streamlit run app.py")
    print("   3. Open browser to http://localhost:8501")
    print()
    
    return prediction

if __name__ == "__main__":
    demo_prediction()
