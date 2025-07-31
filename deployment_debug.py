#!/usr/bin/env python3
"""
Deployment Debugging Script - Copy this to your deployed app
This script will log the exact differences causing the 93 vs 505 issue
"""

import sys
import os
import pickle
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def detailed_debug():
    """Detailed debugging to find the 93 vs 505 difference"""
    print("=" * 80)
    print("COMPREHENSIVE DEPLOYMENT DEBUG ANALYSIS")
    print("=" * 80)
    
    # Test parameters
    year, month, nhs_board = 2025, 7, "Greater Glasgow and Clyde"
    print(f"Testing: {nhs_board}, {month}/{year}")
    
    # Check imports
    print("\n1. IMPORT STATUS:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        TORCH_AVAILABLE = True
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        TORCH_AVAILABLE = False
    
    try:
        from src.components.model import StackedLSTM
        print(f"   ✅ StackedLSTM: Available")
        CUSTOM_MODULES_AVAILABLE = True
    except ImportError as e:
        print(f"   ❌ StackedLSTM: {e}")
        CUSTOM_MODULES_AVAILABLE = False
    
    # Check model files
    print("\n2. MODEL FILES:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    model_files = {
        'trained_lstm_model.pth': None,
        'feature_scaler.pkl': None,
        'nhs_board_mapping.pkl': None
    }
    
    for filename in model_files.keys():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            model_files[filename] = size
            print(f"   ✅ {filename}: {size} bytes")
        else:
            print(f"   ❌ {filename}: Not found")
    
    # Load and inspect scaler
    print("\n3. SCALER ANALYSIS:")
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"   ✅ Scaler type: {type(scaler)}")
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                print(f"   ✅ Scaler mean (first 5): {scaler.mean_[:5]}")
                print(f"   ✅ Scaler mean (last 5): {scaler.mean_[-5:]}")
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                print(f"   ✅ Scaler scale (first 5): {scaler.scale_[:5]}")
                print(f"   ✅ Scaler scale (last 5): {scaler.scale_[-5:]}")
        except Exception as e:
            print(f"   ❌ Scaler loading error: {e}")
            scaler = None
    else:
        print("   ❌ Scaler file not found")
        scaler = None
    
    # Load NHS Board mapping
    print("\n4. NHS BOARD MAPPING:")
    mapping_path = os.path.join(models_dir, "nhs_board_mapping.pkl")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                nhs_mapping = pickle.load(f)
            print(f"   ✅ Mapping loaded: {len(nhs_mapping)} boards")
            board_code = nhs_mapping.get(nhs_board, -1)
            print(f"   ✅ {nhs_board} → {board_code}")
        except Exception as e:
            print(f"   ❌ Mapping loading error: {e}")
            nhs_mapping = {}
            board_code = 0
    else:
        print("   ❌ Mapping file not found")
        nhs_mapping = {}
        board_code = 0
    
    # Test feature preprocessing
    print("\n5. FEATURE PREPROCESSING:")
    month_num = month
    nhs_board_code = board_code
    
    # Cyclical encoding
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)
    
    # Year normalization
    year_norm = (year - 1998) / (2022 - 1998)
    
    # Lag features
    avg_births = 6.2
    birth_lag_1 = avg_births
    birth_lag_2 = avg_births
    birth_lag_3 = avg_births
    birth_rolling_avg = avg_births
    birth_trend = 0.0
    
    print(f"   Year: {year} → normalized: {year_norm:.4f}")
    print(f"   Month: {month} → sin: {month_sin:.4f}, cos: {month_cos:.4f}")
    print(f"   NHS Board: {nhs_board} → code: {nhs_board_code}")
    print(f"   Lag features: {birth_lag_1}, {birth_lag_2}, {birth_lag_3}")
    
    # Create raw feature vector
    raw_features = np.array([[
        year, month_num, nhs_board_code, month_sin, month_cos, year_norm,
        birth_lag_1, birth_lag_2, birth_lag_3, birth_rolling_avg, birth_trend
    ]], dtype=np.float32)
    
    print(f"   Raw features: {raw_features[0]}")
    
    # Apply scaling
    if scaler is not None:
        try:
            scaled_features = scaler.transform(raw_features)
            print(f"   Scaled features: {scaled_features[0]}")
        except Exception as e:
            print(f"   ❌ Scaling error: {e}")
            scaled_features = raw_features
    else:
        print("   ⚠️  No scaler - using raw features")
        scaled_features = raw_features
    
    # Test model loading and prediction
    print("\n6. MODEL TESTING:")
    if TORCH_AVAILABLE and CUSTOM_MODULES_AVAILABLE:
        try:
            model = StackedLSTM(input_size=11, hidden_size=128, num_layers=2)
            print(f"   ✅ Model initialized: {type(model)}")
            
            model_path = os.path.join(models_dir, "trained_lstm_model.pth")
            if os.path.exists(model_path):
                import torch
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                print("   ✅ Model weights loaded")
                
                # Test prediction
                features_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1)
                print(f"   Features tensor shape: {features_tensor.shape}")
                
                with torch.no_grad():
                    log_prediction = model(features_tensor).item()
                    prediction = np.expm1(log_prediction)
                    final_prediction = max(0, int(round(prediction)))
                
                print(f"   Raw model output (log): {log_prediction}")
                print(f"   After expm1: {prediction}")
                print(f"   Final prediction: {final_prediction}")
                
            else:
                print("   ❌ Model file not found for loading")
        except Exception as e:
            print(f"   ❌ Model error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ❌ Cannot test model - missing dependencies")
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY:")
    print("If you see different values here compared to local,")
    print("that's where the 93 vs 505 difference is coming from!")
    print("=" * 80)

if __name__ == "__main__":
    detailed_debug()
