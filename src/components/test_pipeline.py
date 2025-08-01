import torch
import pandas as pd
import numpy as np
import random
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.components.model import StackedLSTM
from src.components.training_utils import train_model
from src.components.evaluation_utils import evaluate_forecast, forecast_horizon_analysis, diebold_mariano_test
from src.components.interpretability_utils import visualize_attention_weights, integrated_gradients, saliency_map
from src.logger import logging
from src.exception import CustomException

logging.info("Starting birth forecast pipeline...")







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
def add_lag_features(df, target_col='Births registered', lags=[1, 2, 3]):
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
            df[f'{target_col}_rolling_3m'] = df.groupby('NHS_Board_area_code')[target_col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
            logging.info(f"Added rolling average feature: {target_col}_rolling_3m")
            
            # Add trend feature within each area
            df[f'{target_col}_trend'] = df.groupby('NHS_Board_area_code')[target_col].diff()
            logging.info(f"Added trend feature: {target_col}_trend")
        
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
    print("Adding lag features...")
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
        logging.info(f"Feature columns to normalize: {feature_cols}")
        
        # Fit scaler on training data only
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
        
        logging.info("Normalized dataframes created successfully")
        return train_scaled, val_scaled, test_scaled
    except Exception as e:
        logging.error("Error during feature normalization")
        raise CustomException(e, sys)





try:
    logging.info("Normalizing features...")
    print("Normalizing features...")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df)
    logging.info("Feature normalization completed successfully")
except Exception as e:
    logging.error("Error during feature normalization process")
    raise CustomException(e, sys)






# Assume last column is target, rest are features
def df_to_tensor(df):
    try:
        logging.info("Converting DataFrame to tensors...")
        df = df.dropna()  # Remove rows with NaN
        logging.info(f"DataFrame shape after dropping NaN: {df.shape}")
        
        x_np = df.iloc[:, :-1].astype(np.float32).values  # Ensure numeric type
        y_np = df.iloc[:, -1].astype(np.float32).values
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # (batch, 1, features)
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)
        
        logging.info(f"Tensors created - X shape: {x.shape}, Y shape: {y.shape}")
        return x, y
    except Exception as e:
        logging.error("Error converting DataFrame to tensors")
        raise CustomException(e, sys)
    





try:
    logging.info("Converting all datasets to tensors...")
    x_train, y_train = df_to_tensor(train_df)
    x_val, y_val = df_to_tensor(val_df)
    x_test, y_test = df_to_tensor(test_df)
    
    logging.info(f"Feature dimensions: {x_train.shape[2]} features")
    logging.info(f"Training samples: {x_train.shape[0]}")
    logging.info(f"Validation samples: {x_val.shape[0]}")
    logging.info(f"Test samples: {x_test.shape[0]}")
    
    print(f"Feature dimensions: {x_train.shape[2]} features")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
except Exception as e:
    logging.error("Error during tensor conversion process")
    raise CustomException(e, sys)






# Create DataLoaders
try:
    logging.info("Creating DataLoaders...")
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)
    logging.info("DataLoaders created successfully")
except Exception as e:
    logging.error("Error creating DataLoaders")
    raise CustomException(e, sys)

# Initialize and train an optimized model with lag features
try:
    
    logging.info("Initializing LSTM model...")
    # Using 2 layers, 128 units, higher learning rate for better convergence
    model = StackedLSTM(input_size=x_train.shape[2], hidden_size=128, num_layers=2)
    logging.info(f"Model initialized - Input size: {x_train.shape[2]}, Hidden size: 128, Layers: 2")
    
    # Train model with optimized parameters
    logging.info("Starting model training...")
    model = train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu')
    logging.info("Model training completed successfully")



except Exception as e:
    logging.error("Error during model initialization or training")
    raise CustomException(e, sys)







# Evaluate
try:


    logging.info("Starting model evaluation...")
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x, _ in test_loader:
            y_pred.append(model(x).cpu().numpy())
    y_pred = torch.tensor(np.concatenate(y_pred)).squeeze().numpy()
    
    metrics = evaluate_forecast(y_test.squeeze().numpy(), y_pred)
    logging.info(f"Model evaluation completed - Metrics: {metrics}")
    print('Test metrics:', metrics)


except Exception as e:
    logging.error("Error during model evaluation")
    raise CustomException(e, sys)







# Interpretability example (for a random test sample)
try:


    logging.info("Starting interpretability analysis...")
    sample_idx = random.randint(0, x_test.shape[0] - 1)
    sample_x = x_test[sample_idx:sample_idx+1]  # (1, 1, features)
    sample_true = y_test[sample_idx].item()
    sample_pred = model(sample_x).item()
    
    logging.info(f"Sample analysis - Index: {sample_idx}, True: {sample_true:.4f}, Predicted: {sample_pred:.4f}")
    print(f"Sample index: {sample_idx}")
    print(f"True value: {sample_true:.4f}, Predicted value: {sample_pred:.4f}")
    
    # If using a transformer model, visualize attention (not used for LSTM)
    if hasattr(model, "transformer_encoder"):
        logging.info("Visualizing attention weights...")
        visualize_attention_weights(model, sample_x)
    
    # Compute attributions and saliency
    logging.info("Computing integrated gradients and saliency maps...")
    attributions = integrated_gradients(model, sample_x)
    saliency = saliency_map(model, sample_x)
    
    logging.info(f"Interpretability analysis completed - Attributions shape: {attributions.shape}, Saliency shape: {saliency.shape}")
    print('Attributions shape:', attributions.shape)
    print('Saliency shape:', saliency.shape)



except Exception as e:
    logging.error("Error during interpretability analysis")
    raise CustomException(e, sys)







# Prepare data for baseline
try:


    logging.info("Preparing data for baseline model comparison...")
    X_train_flat = x_train.squeeze(1).numpy()  # Remove sequence dimension
    X_test_flat = x_test.squeeze(1).numpy()
    y_train_flat = y_train.squeeze().numpy()
    y_test_flat = y_test.squeeze().numpy()
    
    logging.info(f"Baseline data prepared - Train shape: {X_train_flat.shape}, Test shape: {X_test_flat.shape}")
    
    # Train baseline model
    logging.info("Training Linear Regression baseline...")
    baseline = LinearRegression()
    baseline.fit(X_train_flat, y_train_flat)
    baseline_pred = baseline.predict(X_test_flat)
    logging.info("Baseline model training completed")
    
    # Compare baseline vs LSTM
    baseline_mae = mean_absolute_error(y_test_flat, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test_flat, baseline_pred))
    
    logging.info(f"Baseline vs LSTM comparison - Baseline MAE: {baseline_mae:.4f}, LSTM MAE: {metrics['MAE']:.4f}")
    logging.info(f"Baseline vs LSTM comparison - Baseline RMSE: {baseline_rmse:.4f}, LSTM RMSE: {metrics['RMSE']:.4f}")
    
    print(f"\nBaseline (Linear Regression) vs LSTM:")
    print(f"Baseline MAE: {baseline_mae:.4f} | LSTM MAE: {metrics['MAE']:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f} | LSTM RMSE: {metrics['RMSE']:.4f}")
    

    
    logging.info("Pipeline execution completed successfully")
except Exception as e:
    logging.error("Error during baseline model comparison")
    raise CustomException(e, sys)

# All files are imported and communicate with each other in this pipeline.
