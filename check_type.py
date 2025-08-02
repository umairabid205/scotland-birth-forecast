#!/usr/bin/env python3
"""
Check what type the prediction is returning
"""

import sys
import os
sys.path.append('src')

from components.production_predictor import NHSBirthPredictor

def check_prediction_type():
    """Check what type the prediction returns"""
    
    predictor = NHSBirthPredictor()
    result = predictor.predict(2024, 3, 6)
    
    prediction = result['prediction']
    print(f"Prediction value: {prediction}")
    print(f"Prediction type: {type(prediction)}")
    print(f"Is instance of int: {isinstance(prediction, int)}")
    print(f"Is instance of float: {isinstance(prediction, float)}")
    print(f"Is instance of (int, float): {isinstance(prediction, (int, float))}")
    
    # Try different conversion methods
    print(f"float(prediction): {float(prediction)}")
    print(f"type(float(prediction)): {type(float(prediction))}")

if __name__ == "__main__":
    check_prediction_type()
