#!/usr/bin/env python3
"""
Test script to verify that the prediction system produces different outputs for different inputs
"""

import sys
import os
sys.path.append('src')

from components.production_predictor import predict_births

def test_prediction_variety():
    """Test that different inputs produce different predictions"""
    
    print("Testing NHS Birth Prediction System")
    print("=" * 50)
    
    # Test different months for the same year and NHS board
    print("\n1. Testing different months (2024, NHS Board 0 - Ayrshire and Arran):")
    for month in [1, 3, 6, 9, 12]:
        try:
            prediction = predict_births(2024, month, 0)
            print(f"   Month {month:2d}: {prediction:.1f} births")
        except Exception as e:
            print(f"   Month {month:2d}: Error - {e}")
    
    # Test different NHS boards for the same month/year
    print("\n2. Testing different NHS boards (2024, March):")
    nhs_boards = {
        0: "Ayrshire and Arran",
        1: "Borders", 
        2: "Dumfries and Galloway",
        5: "Grampian",
        6: "Greater Glasgow and Clyde",
        9: "Lothian"
    }
    
    for code, name in nhs_boards.items():
        try:
            prediction = predict_births(2024, 3, code)
            print(f"   {name:25s}: {prediction:.1f} births")
        except Exception as e:
            print(f"   {name:25s}: Error - {e}")
    
    # Test different years
    print("\n3. Testing different years (March, NHS Board 6 - Greater Glasgow and Clyde):")
    for year in [2020, 2022, 2024, 2026, 2028]:
        try:
            prediction = predict_births(year, 3, 6)
            print(f"   Year {year}: {prediction:.1f} births")
        except Exception as e:
            print(f"   Year {year}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed! Check if predictions vary reasonably.")

if __name__ == "__main__":
    test_prediction_variety()
