#!/usr/bin/env python3
"""
Quick Prediction Test - Compare Local vs Deployment Logic
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prediction_paths():
    """Test different prediction code paths"""
    print("=" * 60)
    print("PREDICTION PATH ANALYSIS")
    print("=" * 60)
    
    # Test parameters that match the problematic case
    test_nhs_board = "Greater Glasgow and Clyde"
    test_year = 2025
    test_month = 7
    
    print(f"Testing: {test_nhs_board}, {test_month}/{test_year}")
    print()
    
    # Test 1: Fallback multiplier logic
    print("1. FALLBACK MULTIPLIER TEST:")
    board_multipliers = {
        'Greater Glasgow and Clyde': 450, 'Lothian': 400, 'Lanarkshire': 300,
        'Grampian': 250, 'Tayside': 200, 'Ayrshire and Arran': 180,
        'Highland': 150, 'Fife': 170, 'Forth Valley': 130,
        'Dumfries and Galloway': 80, 'Borders': 60, 'Western Isles': 15,
        'Orkney': 12, 'Shetland': 14
    }
    fallback_prediction = board_multipliers.get(test_nhs_board, 100)
    print(f"   Fallback prediction: {fallback_prediction}")
    
    # Test 2: Dummy model logic
    print("\n2. DUMMY MODEL TEST:")
    dummy_log = np.log1p(100.0)
    dummy_prediction = int(np.expm1(dummy_log))
    print(f"   Dummy log value: {dummy_log}")
    print(f"   Dummy prediction: {dummy_prediction}")
    
    # Test 3: Check imports
    print("\n3. IMPORT TEST:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        TORCH_AVAILABLE = True
    except ImportError:
        print("   ❌ PyTorch: Not available")
        TORCH_AVAILABLE = False
    
    try:
        from src.components.model import StackedLSTM
        print(f"   ✅ StackedLSTM: {StackedLSTM}")
        CUSTOM_MODULES_AVAILABLE = True
    except ImportError as e:
        print(f"   ❌ StackedLSTM: {e}")
        CUSTOM_MODULES_AVAILABLE = False
    
    # Test 4: Model file check
    print("\n4. MODEL FILES TEST:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    model_path = os.path.join(models_dir, "trained_lstm_model.pth")
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"   ✅ Model file: {size} bytes")
    else:
        print("   ❌ Model file: Not found")
    
    # Test 5: Simulate the actual prediction logic
    print("\n5. PREDICTION LOGIC SIMULATION:")
    print(f"   TORCH_AVAILABLE: {TORCH_AVAILABLE}")
    print(f"   CUSTOM_MODULES_AVAILABLE: {CUSTOM_MODULES_AVAILABLE}")
    
    if not TORCH_AVAILABLE:
        result = fallback_prediction
        print(f"   → Path: PyTorch fallback → {result}")
    elif not CUSTOM_MODULES_AVAILABLE:
        result = dummy_prediction  
        print(f"   → Path: Dummy model → {result}")
    else:
        print(f"   → Path: Should use real model → ~505")
    
    # Test 6: Check for the mysterious 93
    print("\n6. MYSTERY 93 ANALYSIS:")
    print("   Where could 93 come from?")
    
    # Could it be a scaling issue?
    test_values = [93, 100, 450, 505]
    for val in test_values:
        ratio_to_505 = val / 505.0
        ratio_to_450 = val / 450.0 if val != 450 else 1.0
        print(f"   {val}: ratio to 505 = {ratio_to_505:.3f}, ratio to 450 = {ratio_to_450:.3f}")
    
    # Could it be a feature preprocessing issue?
    print("\n   93/505 ratio = 0.184 (18.4%)")
    print("   This suggests a systematic scaling/preprocessing difference")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("1. Check if deployment is using different scaler values")
    print("2. Verify model weights are loading correctly")
    print("3. Compare feature preprocessing between environments")
    print("4. Check for silent errors in model loading")
    print("5. Test with identical input features to isolate the issue")
    print("=" * 60)

if __name__ == "__main__":
    test_prediction_paths()
