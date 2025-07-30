#!/usr/bin/env python3
"""
Test the fixed scaler to ensure no warnings
"""

import warnings
import pickle
import numpy as np

def test_scaler():
    print("🧪 Testing Fixed Scaler")
    print("=" * 30)
    
    # Capture any warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Load the scaler
            with open("models/feature_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            
            print("✅ Scaler loaded successfully")
            
            # Test transformation
            test_data = np.array([[
                2025, 7, 6, 0.707, 0.707, 1.125, 6.2, 6.1, 6.3, 6.15, 0.1
            ]], dtype=np.float32)
            
            scaled_data = scaler.transform(test_data)
            print(f"✅ Transformation successful - shape: {scaled_data.shape}")
            
            # Check for warnings
            if len(w) == 0:
                print("🎉 NO WARNINGS! Scaler issue completely fixed!")
                return True
            else:
                print(f"⚠️ {len(w)} warnings detected:")
                for warning in w:
                    print(f"   - {warning.message}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing scaler: {e}")
            return False

if __name__ == "__main__":
    test_scaler()
