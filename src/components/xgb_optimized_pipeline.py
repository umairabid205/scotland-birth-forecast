import pandas as pd
import numpy as np
import random
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.logger import logging
from src.exception import CustomException

logging.info("Starting Optimized XGBoost birth forecast pipeline...")

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

# Simplified lag features for better performance
def add_optimized_lag_features(df, target_col='Births registered', lags=[1, 2, 3]):
    """Add optimized lag features with minimal overfitting"""
    
    try:
        logging.info(f"Adding optimized lag features for target column: {target_col}")
        df = df.copy()
        
        # Sort by area and time to ensure proper lag calculation
        if 'NHS_Board_area_code' in df.columns:
            df = df.sort_values(['NHS_Board_area_code', 'Year_norm'])
            logging.info("Data sorted by NHS_Board_area_code and Year_norm")
            
            # Add only essential lag features within each area
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby('NHS_Board_area_code')[target_col].shift(lag)
                logging.info(f"Added lag feature: {target_col}_lag_{lag}")
            
            # Add one rolling average (most important from previous run)
            df[f'{target_col}_rolling_3m'] = df.groupby('NHS_Board_area_code')[target_col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            logging.info(f"Added rolling average feature: {target_col}_rolling_3m")
            
            # Add simple trend feature
            df[f'{target_col}_trend'] = df.groupby('NHS_Board_area_code')[target_col].diff()
            logging.info(f"Added trend feature")
        
        # Fill NaN values with forward fill then backward fill to preserve temporal patterns
        lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'trend' in col]
        for col in lag_cols:
            nan_count = df[col].isna().sum()
            # Use forward fill and backward fill to preserve temporal structure better
            df[col] = df.groupby('NHS_Board_area_code')[col].ffill().bfill()
            remaining_nan = df[col].isna().sum()
            if remaining_nan > 0:
                df[col] = df[col].fillna(df[col].mean())
            logging.info(f"Filled {nan_count} NaN values in {col}")
        
        logging.info("Optimized lag features added successfully")
        return df
    except Exception as e:
        logging.error("Error adding optimized lag features")
        raise CustomException(e, sys)

# Apply lag features to all datasets
try:
    logging.info("Applying optimized lag features to all datasets...")
    print("Adding optimized lag features for XGBoost...")
    train_df = add_optimized_lag_features(train_df)
    val_df = add_optimized_lag_features(val_df)
    test_df = add_optimized_lag_features(test_df)
    logging.info("Optimized lag features applied to all datasets successfully")
except Exception as e:
    logging.error("Error applying lag features to datasets")
    raise CustomException(e, sys)

# For XGBoost, we often don't need normalization, let's try without it
def prepare_data_no_scaling(train_df, val_df, test_df, target_col='Births registered'):
    """Prepare data without scaling (XGBoost can handle different scales)"""
    try:
        logging.info("Preparing data without scaling for XGBoost...")
        
        # Combine train and validation for cross-validation
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Remove any remaining NaN values
        combined_df = combined_df.dropna()
        test_df = test_df.dropna()
        
        # Separate features and targets
        feature_cols = [col for col in combined_df.columns if col != target_col]
        
        X_combined = combined_df[feature_cols].values.astype(np.float32)
        y_combined = combined_df[target_col].values.astype(np.float32)
        
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)
        
        # Save feature names for interpretability
        with open('/Users/umair/Downloads/projects/project_1/models/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        logging.info(f"Data prepared - X_combined: {X_combined.shape}, X_test: {X_test.shape}")
        return X_combined, y_combined, X_test, y_test, feature_cols
        
    except Exception as e:
        logging.error("Error preparing data")
        raise CustomException(e, sys)

try:
    logging.info("Preparing data...")
    print("Preparing data without scaling...")
    X_combined, y_combined, X_test, y_test, feature_cols = prepare_data_no_scaling(train_df, val_df, test_df)
    logging.info("Data preparation completed successfully")
    
    print(f"Feature dimensions: {X_combined.shape[1]} features")
    print(f"Training+Validation samples: {X_combined.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
except Exception as e:
    logging.error("Error during data preparation")
    raise CustomException(e, sys)

# Hyperparameter tuning for XGBoost
try:
    logging.info("Starting hyperparameter tuning for XGBoost...")
    print("Performing hyperparameter tuning...")
    
    # Define a smaller parameter grid for faster tuning
    param_grid = {
        'n_estimators': [300, 500, 800],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Use TimeSeriesSplit for time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create base model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        verbosity=0
    )
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_combined, y_combined)
    
    logging.info("Hyperparameter tuning completed")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best CV score (MAE): {-grid_search.best_score_:.4f}")
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.4f}")
    
    # Use the best model
    best_model = grid_search.best_estimator_
    
except Exception as e:
    logging.error("Error during hyperparameter tuning")
    raise CustomException(e, sys)

# Evaluate the optimized model
try:
    logging.info("Evaluating optimized XGBoost model...")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # SMAPE calculation
    def smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        return 100 * np.mean(diff)
    
    smape_score = smape(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SMAPE': smape_score
    }
    
    logging.info(f"Optimized model evaluation completed - Metrics: {metrics}")
    print('Optimized XGBoost Test metrics:', metrics)
    
    # Display feature importance
    importance = best_model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(feature_importance.head(10).items()):
        print(f"{i+1:2d}. {feature}: {score:.4f}")
    
except Exception as e:
    logging.error("Error during model evaluation")
    raise CustomException(e, sys)

# Compare with baseline
try:
    logging.info("Comparing with baseline...")
    
    # Train baseline model (Linear Regression)
    baseline = LinearRegression()
    baseline.fit(X_combined, y_combined)
    baseline_pred = baseline.predict(X_test)
    
    # Calculate baseline metrics
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    logging.info(f"Baseline vs Optimized XGBoost - Baseline MAE: {baseline_mae:.4f}, XGBoost MAE: {metrics['MAE']:.4f}")
    logging.info(f"Baseline vs Optimized XGBoost - Baseline RMSE: {baseline_rmse:.4f}, XGBoost RMSE: {metrics['RMSE']:.4f}")
    
    print(f"\nBaseline (Linear Regression) vs Optimized XGBoost:")
    print(f"Baseline MAE: {baseline_mae:.4f} | XGBoost MAE: {metrics['MAE']:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f} | XGBoost RMSE: {metrics['RMSE']:.4f}")
    
    # Performance comparison
    if metrics['MAE'] < baseline_mae:
        mae_improvement = ((baseline_mae - metrics['MAE']) / baseline_mae) * 100
        print(f"✅ MAE improvement: {mae_improvement:.2f}%")
    else:
        mae_degradation = ((metrics['MAE'] - baseline_mae) / baseline_mae) * 100
        print(f"❌ MAE degradation: {mae_degradation:.2f}%")
    
    if metrics['RMSE'] < baseline_rmse:
        rmse_improvement = ((baseline_rmse - metrics['RMSE']) / baseline_rmse) * 100
        print(f"✅ RMSE improvement: {rmse_improvement:.2f}%")
    else:
        rmse_degradation = ((metrics['RMSE'] - baseline_rmse) / baseline_rmse) * 100
        print(f"❌ RMSE degradation: {rmse_degradation:.2f}%")
    
    logging.info("Pipeline execution completed successfully")
    
except Exception as e:
    logging.error("Error during baseline comparison")
    raise CustomException(e, sys)

# Save the optimized model
try:
    logging.info("Saving optimized XGBoost model...")
    
    # Save the best model
    with open('/Users/umair/Downloads/projects/project_1/models/optimized_xgb_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save model with metadata
    model_data = {
        'model': best_model,
        'feature_names': feature_cols,
        'best_params': grid_search.best_params_,
        'cv_score': -grid_search.best_score_,
        'test_metrics': metrics
    }
    
    with open('/Users/umair/Downloads/projects/project_1/models/xgb_model_full.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logging.info("Optimized XGBoost model saved successfully")
    
except Exception as e:
    logging.error("Error saving optimized model")
    raise CustomException(e, sys)

print("\n" + "="*60)
print("Optimized XGBoost Birth Forecast Pipeline Completed!")
print("="*60)
print(f"Final Test MAE: {metrics['MAE']:.4f}")
print(f"Final Test RMSE: {metrics['RMSE']:.4f}")
print(f"Final Test SMAPE: {metrics['SMAPE']:.2f}%")
