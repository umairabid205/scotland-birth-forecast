import torch
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from src.components.model import StackedLSTM
from src.components.training_utils import train_model
from src.components.evaluation_utils import evaluate_forecast
from src.logger import logging
from src.exception import CustomException
import sys

def save_model_components():
    """Save trained model and preprocessing components for the UI"""
    try:
        logging.info("Starting model training and saving process...")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Load and prepare data (same as in test_pipeline.py)
        train_df = pd.read_parquet('data/processed/train.parquet')
        val_df = pd.read_parquet('data/processed/val.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        
        # Add lag features function (copy from test_pipeline.py)
        def add_lag_features(df, target_col='Births registered', lags=[1, 2, 3]):
            """Add lag features and rolling statistics with proper handling"""
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
        
        # Apply lag features
        train_df = add_lag_features(train_df)
        val_df = add_lag_features(val_df)
        test_df = add_lag_features(test_df)
        
        # Normalize features function
        def normalize_features(train_df, val_df, test_df, target_col='Births registered'):
            """Normalize features except target column"""
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_df[feature_cols])
            val_features_scaled = scaler.transform(val_df[feature_cols])
            test_features_scaled = scaler.transform(test_df[feature_cols])
            
            train_scaled = pd.DataFrame(train_features_scaled, columns=feature_cols, index=train_df.index)
            train_scaled[target_col] = train_df[target_col]
            
            val_scaled = pd.DataFrame(val_features_scaled, columns=feature_cols, index=val_df.index)
            val_scaled[target_col] = val_df[target_col]
            
            test_scaled = pd.DataFrame(test_features_scaled, columns=feature_cols, index=test_df.index)
            test_scaled[target_col] = test_df[target_col]
            
            return train_scaled, val_scaled, test_scaled, scaler
        
        # Normalize features and get scaler
        train_df, val_df, test_df, scaler = normalize_features(train_df, val_df, test_df)
        
        # Convert to tensors
        def df_to_tensor(df):
            df = df.dropna()
            x_np = df.iloc[:, :-1].astype(np.float32).values
            y_np = df.iloc[:, -1].astype(np.float32).values
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
            y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)
            return x, y
        
        x_train, y_train = df_to_tensor(train_df)
        x_val, y_val = df_to_tensor(val_df)
        x_test, y_test = df_to_tensor(test_df)
        
        # Create DataLoaders
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)
        
        # Train model
        model = StackedLSTM(input_size=x_train.shape[2], hidden_size=128, num_layers=2)
        trained_model = train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu')
        
        # Save model state dict
        torch.save(trained_model.state_dict(), "models/trained_lstm_model.pth")
        logging.info("Model saved to models/trained_lstm_model.pth")
        
        # Save scaler
        with open("models/feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        logging.info("Scaler saved to models/feature_scaler.pkl")
        
        # Create NHS Board mapping
        nhs_board_mapping = {
            'Ayrshire and Arran': 0, 'Borders': 1, 'Dumfries and Galloway': 2,
            'Fife': 3, 'Forth Valley': 4, 'Grampian': 5, 'Greater Glasgow and Clyde': 6,
            'Highland': 7, 'Lanarkshire': 8, 'Lothian': 9, 'Orkney': 10,
            'Shetland': 11, 'Tayside': 12, 'Western Isles': 13
        }
        
        # Save NHS Board mapping
        with open("models/nhs_board_mapping.pkl", 'wb') as f:
            pickle.dump(nhs_board_mapping, f)
        logging.info("NHS Board mapping saved to models/nhs_board_mapping.pkl")
        
        # Test the saved model
        trained_model.eval()
        y_pred = []
        with torch.no_grad():
            for x, _ in test_loader:
                y_pred.append(trained_model(x).cpu().numpy())
        y_pred = torch.tensor(np.concatenate(y_pred)).squeeze().numpy()
        metrics = evaluate_forecast(y_test.squeeze().numpy(), y_pred)
        
        logging.info(f"Model validation metrics: {metrics}")
        print(f"Model saved successfully! Test metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error saving model components: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    save_model_components()
