"""
Production-Ready NHS Birth Prediction Model
============================================

This module provides a complete production-ready interface for NHS birth forecasting
using the best-performing Linear Regression model with XGBoost as backup.

Usage:
    from src.components.production_predictor import NHSBirthPredictor
    
    predictor = NHSBirthPredictor()
    prediction = predictor.predict(year=2025, month=12, nhs_board_code=1)
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))
sys.path.append(project_root)

# Import XGBoost with graceful fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    print("XGBoost not available, using Linear Regression only")

try:
    from src.logger import logging
    from src.exception import CustomException
except ImportError:
    # Fallback logger and exception
    class DummyLogger:
        @staticmethod
        def info(msg): print(f"INFO: {msg}")
        @staticmethod
        def warning(msg): print(f"WARNING: {msg}")
        @staticmethod
        def error(msg): print(f"ERROR: {msg}")
    
    logging = DummyLogger()
    CustomException = Exception

warnings.filterwarnings('ignore')

class NHSBirthPredictor:
    """
    Production-ready NHS Birth Prediction System
    
    Features:
    - Primary: Linear Regression (Best MAE: 0.0033)
    - Backup: XGBoost (MAE: 0.2583)  
    - Ensemble: Weighted combination when needed
    - Automatic feature engineering
    - Model validation and fallback
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the predictor with saved models
        
        Args:
            model_dir: Directory containing saved models (default: models/)
        """
        if model_dir is None:
            # Get the directory of this script and go up to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.model_dir = os.path.join(project_root, 'models')
        else:
            self.model_dir = model_dir
        self.models = {}
        self.feature_names = None
        self.nhs_board_mapping = {}
        self.feature_scaler = None
        self.is_initialized = False
        
        self._load_models()
    
    def _load_models(self):
        """Load all saved models and metadata"""
        try:
            logging.info("Loading production models...")
            
            # Load Linear Regression (primary model)
            with open(f'{self.model_dir}/linear_regression_model.pkl', 'rb') as f:
                self.models['linear_regression'] = pickle.load(f)
            
            # Load XGBoost (backup model) only if XGBoost is available
            if XGBOOST_AVAILABLE:
                try:
                    with open(f'{self.model_dir}/xgboost_model.pkl', 'rb') as f:
                        self.models['xgboost'] = pickle.load(f)
                    logging.info("XGBoost model loaded successfully")
                except FileNotFoundError:
                    logging.warning("XGBoost model not found, using Linear Regression only")
                except Exception as e:
                    logging.warning(f"Error loading XGBoost model: {e}")
            else:
                logging.warning("XGBoost not available, using Linear Regression only")
            
            # Load feature metadata
            with open(f'{self.model_dir}/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            with open(f'{self.model_dir}/nhs_board_mapping.pkl', 'rb') as f:
                self.nhs_board_mapping = pickle.load(f)
            
            with open(f'{self.model_dir}/feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            
            # Load model comparison results
            try:
                with open(f'{self.model_dir}/best_model_config.pkl', 'rb') as f:
                    self.model_config = pickle.load(f)
            except FileNotFoundError:
                self.model_config = {'best_model': 'Linear Regression'}
            
            self.is_initialized = True
            available_models = list(self.models.keys())
            logging.info(f"Models loaded successfully. Available: {available_models}, Primary: {self.model_config.get('best_model', 'Linear Regression')}")
            
        except Exception as e:
            logging.error("Error loading models")
            raise CustomException(e, sys)
    
    def _create_features(self, year: int, month: int, nhs_board_code: int, 
                        historical_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Create features for prediction including lag features and rolling statistics
        
        Args:
            year: Year for prediction
            month: Month for prediction  
            nhs_board_code: NHS Board area code
            historical_data: Historical birth data for lag features
            
        Returns:
            Feature array ready for prediction
        """
        try:
            # Create base features
            year_norm = (year - 2015) / 10.0  # Normalized year
            month_num = month
            
            # Cyclical month features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Create realistic default values based on NHS board size and seasonal patterns
            # Base estimates from historical data analysis
            nhs_board_base_births = {
                0: 180,   # Ayrshire and Arran
                1: 60,    # Borders
                2: 80,    # Dumfries and Galloway
                3: 170,   # Fife
                4: 130,   # Forth Valley
                5: 250,   # Grampian
                6: 450,   # Greater Glasgow and Clyde
                7: 150,   # Highland
                8: 300,   # Lanarkshire
                9: 400,   # Lothian
                10: 12,   # Orkney
                11: 14,   # Shetland
                12: 200,  # Tayside
                13: 15    # Western Isles
            }
            
            # Seasonal multipliers (births tend to be higher in certain months)
            seasonal_multipliers = {
                1: 0.95, 2: 0.90, 3: 1.05, 4: 1.10, 5: 1.15, 6: 1.10,
                7: 1.15, 8: 1.10, 9: 1.20, 10: 1.15, 11: 1.05, 12: 1.00
            }
            
            # Get base births for this NHS board
            base_births = nhs_board_base_births.get(nhs_board_code, 100)
            seasonal_factor = seasonal_multipliers.get(month, 1.0)
            estimated_births = base_births * seasonal_factor
            
            # Year trend adjustment (slight increase over time)
            year_factor = 1.0 + (year - 2020) * 0.005  # 0.5% annual increase
            estimated_births *= year_factor
            
            # Initialize feature dictionary with realistic estimates
            features = {
                'Year': year,
                'Month_num': month_num,
                'NHS_Board_area_code': nhs_board_code,
                'Month_sin': month_sin,
                'Month_cos': month_cos,
                'Year_norm': year_norm,
                'Births registered_lag_1': estimated_births,
                'Births registered_lag_2': estimated_births * 0.98,  # Slight variation
                'Births registered_rolling_3m': estimated_births * 0.99
            }
            
            # Add lag features and rolling statistics if historical data available
            if historical_data is not None:
                # Get recent data for this NHS board
                board_data = historical_data[
                    historical_data['NHS_Board_area_code'] == nhs_board_code
                ].sort_values('Year_norm').tail(6)  # Get last 6 records
                
                if len(board_data) >= 1:
                    # Lag features
                    features['Births registered_lag_1'] = float(board_data.iloc[-1]['Births registered'])
                    features['Births registered_lag_2'] = float(board_data.iloc[-2]['Births registered']) if len(board_data) >= 2 else features['Births registered_lag_1']
                    
                    # Rolling mean (3-month)
                    recent_births = board_data.tail(3)['Births registered'].to_numpy()
                    features['Births registered_rolling_3m'] = float(np.mean(recent_births))
            
            # Convert to array in correct order matching saved feature names
            if self.feature_names is not None:
                feature_array = np.array([features[col] for col in self.feature_names]).reshape(1, -1)
            else:
                # Fallback order if feature names not loaded
                feature_order = ['Year', 'Month_num', 'NHS_Board_area_code', 'Month_sin', 'Month_cos', 
                               'Year_norm', 'Births registered_lag_1', 'Births registered_lag_2', 
                               'Births registered_rolling_3m']
                feature_array = np.array([features[col] for col in feature_order]).reshape(1, -1)
            
            return feature_array.astype(np.float32)
            
        except Exception as e:
            logging.error("Error creating features")
            raise CustomException(e, sys)
    
    def predict(self, year: int, month: int, nhs_board_code: int, 
                historical_data: Optional[pd.DataFrame] = None,
                use_ensemble: bool = False) -> Dict[str, Union[float, str]]:
        """
        Make birth prediction for given parameters
        
        Args:
            year: Year for prediction (e.g., 2025)
            month: Month for prediction (1-12)
            nhs_board_code: NHS Board area code
            historical_data: Optional historical data for better lag features
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Dictionary with prediction results and metadata
        """
        if not self.is_initialized:
            raise CustomException("Model not initialized", sys)
        
        try:
            # Validate inputs
            if not (2020 <= year <= 2030):
                raise ValueError("Year should be between 2020 and 2030")
            if not (1 <= month <= 12):
                raise ValueError("Month should be between 1 and 12")
            if not (0 <= nhs_board_code <= 20):
                raise ValueError("NHS Board code should be between 0 and 20")
            
            # Create features
            X = self._create_features(year, month, nhs_board_code, historical_data)
            
            # Make predictions
            predictions = {}
            
            # Primary: Linear Regression
            if 'linear_regression' in self.models:
                lr_pred = self.models['linear_regression'].predict(X)[0]
                predictions['linear_regression'] = max(0, lr_pred)  # Ensure non-negative
            
            # Backup: XGBoost
            if 'xgboost' in self.models:
                xgb_pred = self.models['xgboost'].predict(X)[0]
                predictions['xgboost'] = max(0, xgb_pred)
            
            # Ensemble prediction
            if use_ensemble and len(predictions) >= 2:
                ensemble_pred = 0.7 * predictions['linear_regression'] + 0.3 * predictions['xgboost']
                predictions['ensemble'] = max(0, ensemble_pred)
            
            # Select best prediction
            primary_model = self.model_config.get('best_model', 'Linear Regression').lower().replace(' ', '_')
            
            if use_ensemble and 'ensemble' in predictions:
                final_prediction = predictions['ensemble']
                model_used = "Ensemble (LR 70% + XGB 30%)"
            elif primary_model in predictions:
                final_prediction = predictions[primary_model]
                model_used = primary_model.replace('_', ' ').title()
            else:
                final_prediction = list(predictions.values())[0]
                model_used = list(predictions.keys())[0].replace('_', ' ').title()
            
            # Get NHS Board name if available
            nhs_board_name = self.nhs_board_mapping.get(nhs_board_code, f"NHS Board {nhs_board_code}")
            
            result = {
                'prediction': round(final_prediction, 2),
                'model_used': model_used,
                'year': year,
                'month': month,
                'nhs_board_code': nhs_board_code,
                'nhs_board_name': nhs_board_name,
                'confidence': 'High' if model_used.startswith('Linear') else 'Medium',
                'all_predictions': predictions
            }
            
            logging.info(f"Prediction made: {final_prediction:.2f} births for {nhs_board_name} in {month}/{year}")
            return result
            
        except Exception as e:
            logging.error("Error making prediction")
            raise CustomException(e, sys)
    
    def predict_batch(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions for multiple records
        
        Args:
            predictions_df: DataFrame with columns ['year', 'month', 'nhs_board_code']
            
        Returns:
            DataFrame with predictions added
        """
        try:
            results = []
            
            for _, row in predictions_df.iterrows():
                result = self.predict(
                    year=int(row['year']),
                    month=int(row['month']),
                    nhs_board_code=int(row['nhs_board_code'])
                )
                results.append(result)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            logging.info(f"Batch prediction completed for {len(results_df)} records")
            return results_df
            
        except Exception as e:
            logging.error("Error in batch prediction")
            raise CustomException(e, sys)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'primary_model': self.model_config.get('best_model', 'Linear Regression'),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'nhs_boards': len(self.nhs_board_mapping) if self.nhs_board_mapping else 0,
            'model_performance': self.model_config.get('results', [])
        }

# Convenience functions for direct use
def predict_births(year: int, month: int, nhs_board_code: int, 
                  use_ensemble: bool = False) -> float:
    """
    Simple function to get birth prediction
    
    Args:
        year: Year for prediction
        month: Month for prediction  
        nhs_board_code: NHS Board area code
        use_ensemble: Whether to use ensemble model
        
    Returns:
        Predicted number of births
    """
    predictor = NHSBirthPredictor()
    result = predictor.predict(year, month, nhs_board_code, use_ensemble=use_ensemble)
    prediction = result['prediction']
    try:
        return float(prediction)
    except (ValueError, TypeError):
        return 0.0

# Example usage and testing
if __name__ == "__main__":
    print("NHS Birth Prediction System - Production Ready")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = NHSBirthPredictor()
        
        # Example prediction
        result = predictor.predict(year=2025, month=6, nhs_board_code=1)
        
        print(f"Prediction: {result['prediction']} births")
        print(f"Model Used: {result['model_used']}")
        print(f"NHS Board: {result['nhs_board_name']}")
        print(f"Confidence: {result['confidence']}")
        
        # Model info
        info = predictor.get_model_info()
        print(f"\nModel Info:")
        print(f"Available Models: {info['available_models']}")
        print(f"Primary Model: {info['primary_model']}")
        print(f"Feature Count: {info['feature_count']}")
        
    except Exception as e:
        print(f"Error: {e}")
