#!/usr/bin/env python3
"""
Fix scaler version compatibility issues
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def recreate_scaler_with_current_version():
    """Recreate the scaler using actual training data to avoid version conflicts"""
    print("🔧 Recreating scaler with current sklearn version...")
    
    try:
        # Load the actual training data to fit the scaler properly
        train_df = pd.read_parquet('data/processed/train.parquet')
        val_df = pd.read_parquet('data/processed/val.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        
        # Apply the same lag features processing as in the training
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
        
        # Add lag features
        train_df = add_lag_features(train_df)
        
        # Extract feature columns (exclude target)
        target_col = 'Births registered'
        feature_cols = [col for col in train_df.columns if col != target_col]
        
        # Create and fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])
        
        # Save the new scaler
        with open("models/feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✅ Scaler recreated with current sklearn version")
        print(f"   Feature columns: {feature_cols}")
        print(f"   Training shape: {train_df[feature_cols].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recreating scaler: {e}")
        return False

def main():
    print("🔧 Fixing Sklearn Version Compatibility")
    print("=" * 50)
    
    if recreate_scaler_with_current_version():
        print("\n🎉 Scaler version issue fixed!")
        print("📱 The UI should now work without version warnings.")
    else:
        print("\n❌ Failed to fix scaler version issue.")

if __name__ == "__main__":
    main()
