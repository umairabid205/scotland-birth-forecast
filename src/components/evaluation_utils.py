import numpy as np  # For numerical operations
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For MAE and RMSE
import torch  # For PyTorch operations

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # Convert to numpy arrays
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0  # Average abs values
    diff = np.abs(y_true - y_pred) / denominator  # Relative error
    diff[denominator == 0] = 0.0  # Avoid division by zero
    return 100 * np.mean(diff)  # Return as percentage

# Evaluation metrics: MAE, RMSE, SMAPE
def evaluate_forecast(y_true, y_pred):
    """ Evaluate forecast performance using MAE, RMSE, and SMAPE."""
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    smape_val = smape(y_true, y_pred)  # Symmetric Mean Absolute Percentage Error
    return {'MAE': mae, 'RMSE': rmse, 'SMAPE': smape_val}  # Return as dict

# Forecast horizon analysis: multi-step ahead
def forecast_horizon_analysis(model, data_loader, horizons=[1, 3, 6, 12], device='cpu'):
    """ Analyze forecast performance across multiple horizons."""
    model.eval()  # Set model to evaluation mode
    results = {}  # Store results for each horizon
    with torch.no_grad():  # No gradient computation
        for horizon in horizons:  # Loop over forecast horizons
            y_trues, y_preds = [], []  # Store true and predicted values
            for x, y in data_loader:  # Loop over batches
                x = x.to(device)  # Move input to device
                # Assume y is (batch, horizon) and model predicts (batch, horizon)
                y_pred = model(x)  # Model prediction
                y_trues.append(y[:, :horizon].cpu().numpy())  # True values for horizon
                y_preds.append(y_pred[:, :horizon].cpu().numpy())  # Predicted values for horizon
            y_trues = np.concatenate(y_trues)  # Concatenate all batches
            y_preds = np.concatenate(y_preds)
            metrics = evaluate_forecast(y_trues, y_preds)  # Compute metrics
            results[horizon] = metrics  # Store metrics
    return results  # Return all results

# Diebold-Mariano test for forecast comparison
def diebold_mariano_test(e1, e2, h=1):
    """e1, e2: forecast errors from two models (arrays), h: forecast horizon"""
    from statsmodels.stats.diagnostic import acorr_ljungbox  # For autocorrelation (not used here)
    d = e1 - e2  # Difference in errors
    mean_d = np.mean(d)  # Mean difference
    n = len(d)  # Number of samples
    var_d = np.var(d, ddof=1)  # Variance of difference
    dm_stat = mean_d / np.sqrt(var_d / n)  # DM test statistic
    # For large n, DM ~ N(0,1)
    from scipy.stats import norm  # Normal distribution
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))  # Two-sided p-value
    return dm_stat, p_value  # Return statistic and p-value

# Example usage (not run):
# y_true, y_pred = ...  # True and predicted values
# print(evaluate_forecast(y_true, y_pred))  # Print metrics
# e1, e2 = model1_errors, model2_errors  # Errors from two models
# print(diebold_mariano_test(e1, e2))  # Print DM test result
