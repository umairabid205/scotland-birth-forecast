#!/usr/bin/env python3
"""
Debug script to check model loading and prediction pipeline
"""

import sys
import os
sys.path.append('src')

from components.production_predictor import NHSBirthPredictor
import numpy as np

def debug_model_system():
    """Debug the model loading and prediction process"""
    
    print("Debugging NHS Birth Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = NHSBirthPredictor()
    
    # Check model info
    print("\n1. Model Information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test feature creation
    print("\n2. Feature Creation Test:")
    try:
        features = predictor._create_features(2024, 3, 6)  # March 2024, Greater Glasgow
        print(f"   Features shape: {features.shape}")
        print(f"   Features: {features.flatten()}")
        print(f"   Feature names: {predictor.feature_names}")
    except Exception as e:
        print(f"   Error creating features: {e}")
    
    # Test model prediction
    print("\n3. Model Prediction Test:")
    try:
        if predictor.model is not None:
            # Test with raw features
            if predictor.scaler is not None:
                scaled_features = predictor.scaler.transform(features)
                print(f"   Scaled features: {scaled_features.flatten()}")
            else:
                scaled_features = features
                print("   No scaler applied")
            
            prediction = predictor.model.predict(scaled_features)
            print(f"   Raw prediction: {prediction}")
        else:
            print("   No model loaded!")
    except Exception as e:
        print(f"   Error in model prediction: {e}")
    
    # Test full prediction pipeline
    print("\n4. Full Pipeline Test:")
    try:
        result = predictor.predict(2024, 3, 6)
        print(f"   Full pipeline result: {result}")
    except Exception as e:
        print(f"   Error in full pipeline: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_model_system()
