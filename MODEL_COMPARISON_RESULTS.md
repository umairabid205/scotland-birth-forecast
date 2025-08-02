# Scotland Birth Forecast: Model Comparison Results

## Executive Summary

After implementing and comparing different models for Scotland birth forecasting, here are the key findings:

### Model Performance Comparison

| Model                            | MAE    | RMSE   | SMAPE | Training Time |
| -------------------------------- | ------ | ------ | ----- | ------------- |
| **Linear Regression (Baseline)** | 0.0048 | 0.0177 | ~0.1% | Fastest       |
| **LSTM Neural Network**          | 0.0712 | 0.1494 | 1.71% | Slow          |
| **XGBoost (Optimized)**          | 0.0528 | 0.0721 | 1.25% | Medium        |

### Key Insights

1. **Linear Regression is the clear winner** for this dataset

   - Lowest error rates across all metrics
   - Fastest training time
   - Most interpretable results
   - Suggests strong linear relationships in the data

2. **XGBoost performs better than LSTM**

   - More suitable for tabular data than neural networks
   - Better feature importance interpretability
   - Faster training than LSTM

3. **Feature Importance (from XGBoost)**
   - `Births registered_rolling_3m`: 68.46% (most important)
   - `Births registered_lag_2`: 10.86%
   - `NHS_Board_area_code`: 10.69%
   - Other lag features: remaining importance

### Recommendations

1. **For Production**: Use Linear Regression as the primary model

   - Best performance with lowest complexity
   - Fastest predictions
   - Easy to interpret and debug

2. **For Research**: Keep XGBoost as secondary model

   - Good for feature importance analysis
   - Can capture non-linear patterns if data evolves
   - Ensemble potential with linear model

3. **Model Ensemble Option**
   - Combine Linear Regression (70% weight) + XGBoost (30% weight)
   - May provide more robust predictions

### Data Characteristics

The superior performance of Linear Regression suggests:

- Birth rates follow predictable linear trends
- Seasonal and regional patterns are well-captured by linear relationships
- The dataset may be too small to benefit from complex models
- Feature engineering (lags, rolling averages) provides sufficient complexity

### Next Steps

1. Deploy Linear Regression as the primary model
2. Monitor model performance over time
3. Consider ensemble approaches if needed
4. Explore more sophisticated feature engineering for XGBoost improvement
