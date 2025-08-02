import pandas as pd
import numpy as np
import random
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.components.xgb_model import XGBoostForecaster
from src.logger import logging
from src.exception import CustomException

logging.info("Starting XGBoost birth forecast pipeline...")

# Load data
try:
    logging.info("Loading training, validation, and test data...")
    train_df = pd.read_parquet('/Users/umair/Downloads/projects/project_1/data/processed/train.parquet')
    val_df = pd.read_parquet('/Users/umair/Downloads/projects/project_1/data/processed/val.parquet')
    test_df = pd.read_parquet('/Users/umair/Downloads/projects/project_1/data/processed/test.parquet')
    logging.info(f"Data loaded successfully - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
except Exception as e:
    logging.error("Error loading data files")
    raise CustomException(e, sys)

# Add lag features for better temporal modeling
def add_lag_features(df, target_col='Births registered', lags=[1, 2, 3, 6, 12]):
    """Add lag features and rolling statistics with proper handling"""
    
    try:
        logging.info(f"Adding lag features for target column: {target_col}")
        df = df.copy()
        
        # Sort by area and time to ensure proper lag calculation
        if 'NHS_Board_area_code' in df.columns:
            df = df.sort_values(['NHS_Board_area_code', 'Year_norm'])
            logging.info("Data sorted by NHS_Board_area_code and Year_norm")
            
            # Add lag features within each area (prevents leakage across areas)
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby('NHS_Board_area_code')[target_col].shift(lag)
                logging.info(f"Added lag feature: {target_col}_lag_{lag}")
            
            # Add rolling statistics within each area
            for window in [3, 6, 12]:
                df[f'{target_col}_rolling_{window}m'] = df.groupby('NHS_Board_area_code')[target_col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                logging.info(f"Added rolling average feature: {target_col}_rolling_{window}m")
            
            # Add trend features within each area
            df[f'{target_col}_trend'] = df.groupby('NHS_Board_area_code')[target_col].diff()
            df[f'{target_col}_trend_3m'] = df.groupby('NHS_Board_area_code')[target_col].diff(3)
            logging.info(f"Added trend features")
            
            # Add seasonal features
            df['Month_sin_annual'] = np.sin(2 * np.pi * df['Month_num'] / 12)
            df['Month_cos_annual'] = np.cos(2 * np.pi * df['Month_num'] / 12)
            logging.info(f"Added seasonal features")
        
        # Fill NaN values with mean of the column to avoid data loss
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'trend' in col]
        for col in lag_cols:
            nan_count = df[col].isna().sum()
            df[col] = df[col].fillna(df[col].mean())
            logging.info(f"Filled {nan_count} NaN values in {col} with mean")
        
        logging.info("Lag features added successfully")
        return df
    except Exception as e:
        logging.error("Error adding lag features")
        raise CustomException(e, sys)

# Apply lag features to all datasets
try:
    logging.info("Applying lag features to all datasets...")
    print("Adding lag features for XGBoost...")
    train_df = add_lag_features(train_df)
    val_df = add_lag_features(val_df)
    test_df = add_lag_features(test_df)
    logging.info("Lag features applied to all datasets successfully")
except Exception as e:
    logging.error("Error applying lag features to datasets")
    raise CustomException(e, sys)

# Normalize features to improve training stability
def normalize_features(train_df, val_df, test_df, target_col='Births registered'):
    """Normalize features except target column"""
    try:
        logging.info("Starting feature normalization...")
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != target_col]
        logging.info(f"Feature columns to normalize: {len(feature_cols)} features")
        
        # For XGBoost, we can choose to normalize or not
        # Often XGBoost works well without normalization, but we'll normalize for consistency
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_df[feature_cols])
        val_features_scaled = scaler.transform(val_df[feature_cols])
        test_features_scaled = scaler.transform(test_df[feature_cols])
        logging.info("Features scaled using StandardScaler")
        
        # Create new dataframes with scaled features
        train_scaled = pd.DataFrame(train_features_scaled, columns=feature_cols, index=train_df.index)
        train_scaled[target_col] = train_df[target_col]
        
        val_scaled = pd.DataFrame(val_features_scaled, columns=feature_cols, index=val_df.index)
        val_scaled[target_col] = val_df[target_col]
        
        test_scaled = pd.DataFrame(test_features_scaled, columns=feature_cols, index=test_df.index)
        test_scaled[target_col] = test_df[target_col]
        
        # Save the scaler for later use
        try:
            logging.info("Saving feature scaler...")
            with open('/Users/umair/Downloads/projects/project_1/models/feature_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            logging.info("Feature scaler saved successfully")
        except Exception as e:
            logging.error("Error saving feature scaler")
            raise CustomException(e, sys)
        
        logging.info("Normalized dataframes created successfully")
        return train_scaled, val_scaled, test_scaled, scaler, feature_cols
    except Exception as e:
        logging.error("Error during feature normalization")
        raise CustomException(e, sys)

try:
    logging.info("Normalizing features...")
    print("Normalizing features...")
    train_df, val_df, test_df, scaler, feature_cols = normalize_features(train_df, val_df, test_df)
    logging.info("Feature normalization completed successfully")
    
    # Save NHS Board mapping after normalization
    try:
        logging.info("Saving NHS Board mapping...")
        # Create mapping from the original data
        original_train = pd.read_parquet('/Users/umair/Downloads/projects/project_1/data/processed/train.parquet')
        nhs_board_areas = original_train['NHS Board area'].unique() if 'NHS Board area' in original_train.columns else []
        nhs_board_mapping = dict(enumerate(nhs_board_areas))
        with open('/Users/umair/Downloads/projects/project_1/models/nhs_board_mapping.pkl', 'wb') as f:
            pickle.dump(nhs_board_mapping, f)
        logging.info("NHS Board mapping saved successfully")
    except Exception as e:
        logging.error("Error saving NHS Board mapping")
        raise CustomException(e, sys)
    
except Exception as e:
    logging.error("Error during feature normalization process")
    raise CustomException(e, sys)

# Prepare data for XGBoost (no need for tensor conversion)
try:
    logging.info("Preparing data for XGBoost...")
    
    # Remove any remaining NaN values
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()
    
    # Separate features and targets
    target_col = 'Births registered'
    
    X_train = train_df.drop(columns=[target_col]).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    
    X_val = val_df.drop(columns=[target_col]).values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)
    
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)
    
    logging.info(f"Data prepared - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Feature dimensions: {X_train.shape[1]} features")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
except Exception as e:
    logging.error("Error preparing data for XGBoost")
    raise CustomException(e, sys)

# Initialize and train XGBoost model
try:
    logging.info("Initializing XGBoost model...")
    
    # Create XGBoost model with optimized parameters
    model = XGBoostForecaster(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    logging.info("Starting XGBoost model training...")
    print("Training XGBoost model...")
    
    # Train with validation set for early stopping
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    
    logging.info("XGBoost model training completed successfully")
    
except Exception as e:
    logging.error("Error during XGBoost model initialization or training")
    raise CustomException(e, sys)

# Evaluate the model
try:
    logging.info("Starting model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = model.evaluate(X_test, y_test)
    
    logging.info(f"Model evaluation completed - Metrics: {metrics}")
    print('XGBoost Test metrics:', metrics)
    
    # Display feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(importance.head(10).items()):
        print(f"{i+1:2d}. {feature}: {score:.4f}")
    
except Exception as e:
    logging.error("Error during model evaluation")
    raise CustomException(e, sys)

# Compare with baseline
try:
    logging.info("Preparing data for baseline model comparison...")
    
    # Train baseline model (Linear Regression)
    logging.info("Training Linear Regression baseline...")
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    logging.info("Baseline model training completed")
    
    # Compare baseline vs XGBoost
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    logging.info(f"Baseline vs XGBoost comparison - Baseline MAE: {baseline_mae:.4f}, XGBoost MAE: {metrics['MAE']:.4f}")
    logging.info(f"Baseline vs XGBoost comparison - Baseline RMSE: {baseline_rmse:.4f}, XGBoost RMSE: {metrics['RMSE']:.4f}")
    
    print(f"\nBaseline (Linear Regression) vs XGBoost:")
    print(f"Baseline MAE: {baseline_mae:.4f} | XGBoost MAE: {metrics['MAE']:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f} | XGBoost RMSE: {metrics['RMSE']:.4f}")
    
    # Performance improvement
    mae_improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
    rmse_improvement = ((baseline_rmse - metrics['RMSE']) / baseline_rmse) * 100
    
    print(f"\nPerformance Improvement over Baseline:")
    print(f"MAE improvement: {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    
    logging.info("Pipeline execution completed successfully")
    
except Exception as e:
    logging.error("Error during baseline model comparison")
    raise CustomException(e, sys)

# Save the trained XGBoost model
try:
    logging.info("Saving trained XGBoost model...")
    model.save_model('/Users/umair/Downloads/projects/project_1/models/trained_xgb_model.pkl')
    logging.info("XGBoost model saved successfully")
    
    # Also save as a simple pickle for compatibility
    with open('/Users/umair/Downloads/projects/project_1/models/xgb_model_simple.pkl', 'wb') as f:
        pickle.dump(model.model, f)
    logging.info("Simple XGBoost model saved for compatibility")
    
except Exception as e:
    logging.error("Error saving XGBoost model")
    raise CustomException(e, sys)

print("\n" + "="*50)
print("XGBoost Birth Forecast Pipeline Completed!")
print("="*50)
