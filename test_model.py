"""
Simple test script for the NHS Birth Prediction Model
"""
import sys
import os
import pandas as pd
import pickle

# Add project root to path
project_root = '/Users/umair/Downloads/projects/project_1'
sys.path.append(project_root)

# Test the model directly without the wrapper
def test_simple_prediction():
    print("Testing NHS Birth Prediction Model")
    print("=" * 40)
    
    try:
        # Load the model directly
        print("Loading Linear Regression model...")
        with open('/Users/umair/Downloads/projects/project_1/models/linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load feature names
        with open('/Users/umair/Downloads/projects/project_1/models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        print(f"Feature names: {feature_names}")
        print(f"Number of features: {len(feature_names)}")
        
        # Create a simple prediction
        import numpy as np
        
        # Example: June 2025, NHS Board 1
        year = 2025
        month = 6
        nhs_board_code = 1
        
        # Create features in the expected order
        features = {
            'Year': year,
            'Month_num': month,
            'NHS_Board_area_code': nhs_board_code,
            'Month_sin': np.sin(2 * np.pi * month / 12),
            'Month_cos': np.cos(2 * np.pi * month / 12),
            'Year_norm': (year - 2015) / 10.0,
            'Births registered_lag_1': 50.0,  # Default
            'Births registered_lag_2': 50.0,  # Default
            'Births registered_rolling_3m': 50.0  # Default
        }
        
        # Create feature array
        X = np.array([features[col] for col in feature_names]).reshape(1, -1)
        
        print(f"Input features: {X}")
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        print(f"\nPrediction Results:")
        print(f"Date: {month}/{year}")
        print(f"NHS Board Code: {nhs_board_code}")
        print(f"Predicted Births: {prediction:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_prediction()
    if success:
        print("\n✅ Model test successful!")
    else:
        print("\n❌ Model test failed!")
