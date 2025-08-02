import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostForecaster:
    """
    XGBoost model for time series forecasting.
    
    This model is designed for Scotland birth forecasting with:
    1. Optimized hyperparameters for time series data
    2. Feature importance analysis
    3. Cross-validation for robust performance
    4. Early stopping to prevent overfitting
    """
    
    def __init__(self, 
                 n_estimators=1000,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 random_state=42):
        """
        Initialize XGBoost model with optimized parameters for time series.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate (eta)
            subsample (float): Subsample ratio of training instances
            colsample_bytree (float): Subsample ratio of features
            random_state (int): Random seed for reproducibility
        """
        logger.info("Initializing XGBoost Forecaster with parameters:")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - max_depth: {max_depth}")
        logger.info(f"  - learning_rate: {learning_rate}")
        logger.info(f"  - subsample: {subsample}")
        logger.info(f"  - colsample_bytree: {colsample_bytree}")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Train the XGBoost model with optional validation set.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            X_val (array-like, optional): Validation features
            y_val (array-like, optional): Validation targets
            feature_names (list, optional): Names of features
        """
        logger.info("Starting XGBoost model training...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        self.feature_names = feature_names
        
        # Prepare evaluation set for early stopping
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            logger.info(f"Validation data shape: {X_val.shape}")
        else:
            eval_set = [(X_train, y_train)]
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_fitted = True
        logger.info("XGBoost model training completed successfully")
        
        # Log feature importance
        if self.feature_names:
            importance = self.get_feature_importance()
            logger.info("Top 5 most important features:")
            for i, (feature, score) in enumerate(importance.head().items()):
                logger.info(f"  {i+1}. {feature}: {score:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (array-like): Features for prediction
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making predictions for {X.shape[0]} samples")
        predictions = self.model.predict(X)
        logger.info("Predictions completed")
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            pandas.Series: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        else:
            return pd.Series(importance).sort_values(ascending=False)
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training targets
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Use TimeSeriesSplit for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                random_state=42,
                verbose=False
            ),
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return grid_search.best_params_
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        logger.info(f"Saving XGBoost model to {filepath}")
        
        # Save using pickle for compatibility
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            XGBoostForecaster: Loaded model instance
        """
        logger.info(f"Loading XGBoost model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info("Model loaded successfully")
        return instance
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info("Evaluating model performance...")
        
        y_pred = self.predict(X_test)
        
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
        
        logger.info("Model evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics

# Export the model class
__all__ = ["XGBoostForecaster"]
