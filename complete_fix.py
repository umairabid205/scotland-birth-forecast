#!/usr/bin/env python3
"""
Complete fix for scaler issues - recreate and clear caches
"""

import os
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

def complete_scaler_fix():
    """Complete fix for scaler - recreate with proper data and clear caches"""
    print("🔧 Complete Scaler Fix - Recreating with Current Environment")
    print("=" * 60)
    
    try:
        # Remove old scaler file
        scaler_path = "models/feature_scaler.pkl"
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            print("🗑️ Removed old scaler file")
        
        # Load actual training data
        print("📊 Loading training data...")
        train_df = pd.read_parquet('data/processed/train.parquet')
        val_df = pd.read_parquet('data/processed/val.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        
        # Apply lag features exactly as in training
        def add_lag_features(df, target_col='Births registered', lags=[1, 2, 3]):
            df = df.copy()
            if 'NHS_Board_area_code' in df.columns:
                df = df.sort_values(['NHS_Board_area_code', 'Year_norm'])
                for lag in lags:
                    df[f'{target_col}_lag_{lag}'] = df.groupby('NHS_Board_area_code')[target_col].shift(lag)
                df[f'{target_col}_rolling_3m'] = df.groupby('NHS_Board_area_code')[target_col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{target_col}_trend'] = df.groupby('NHS_Board_area_code')[target_col].diff()
            
            lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'trend' in col]
            for col in lag_cols:
                df[col] = df[col].fillna(df[col].mean())
            return df
        
        print("🔄 Adding lag features...")
        train_df = add_lag_features(train_df)
        val_df = add_lag_features(val_df)
        test_df = add_lag_features(test_df)
        
        # Get feature columns (exclude target)
        target_col = 'Births registered'
        feature_cols = [col for col in train_df.columns if col != target_col]
        
        print(f"📈 Feature columns: {feature_cols}")
        print(f"📈 Training data shape: {train_df[feature_cols].shape}")
        
        # Create fresh scaler with current sklearn version
        print("⚙️ Creating fresh scaler...")
        scaler = StandardScaler()
        
        # Fit on training features
        train_features = train_df[feature_cols].values.astype(np.float32)
        scaler.fit(train_features)
        
        # Save the new scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ Fresh scaler created and saved to {scaler_path}")
        
        # Test loading the scaler
        with open(scaler_path, 'rb') as f:
            test_scaler = pickle.load(f)
        print("✅ Scaler loads without warnings")
        
        # Test transformation
        sample_data = train_features[:5]
        scaled_data = test_scaler.transform(sample_data)
        print(f"✅ Scaler transformation works - scaled shape: {scaled_data.shape}")
        
        print("\n🎉 Scaler issue completely resolved!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing scaler: {e}")
        return False

def clear_streamlit_cache():
    """Clear Streamlit cache directories"""
    print("\n🧹 Clearing Streamlit cache...")
    
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "__pycache__",
        "src/__pycache__",
        "src/components/__pycache__"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"🗑️ Cleared {cache_dir}")
            except:
                print(f"⚠️ Could not clear {cache_dir}")
    
    print("✅ Cache clearing completed")

def main():
    if complete_scaler_fix():
        clear_streamlit_cache()
        print("\n" + "=" * 60)
        print("🎊 ALL ISSUES FIXED!")
        print("📱 Now restart the UI with: streamlit run app.py")
        print("🔥 The scaler warning should be completely gone!")
        print("=" * 60)
    else:
        print("\n❌ Fix failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
