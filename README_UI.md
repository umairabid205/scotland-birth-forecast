# 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast UI

A web-based user interface for predicting birth registrations in Scottish NHS Board areas using a trained LSTM neural network.

## 🚀 Quick Start

1. **Setup and Launch** (Automated):

   ```bash
   python setup_ui.py
   ```

   This will:

   - Install required dependencies
   - Train and save the LSTM model
   - Launch the Streamlit web interface

2. **Manual Setup**:

   ```bash
   # Install dependencies
   pip install -r requirements_ui.txt

   # Prepare the trained model
   python save_model_for_ui.py

   # Launch the UI
   streamlit run app.py
   ```

## 📱 Using the Interface

### Input Parameters:

- **NHS Board Area**: Select from 14 Scottish health board regions
- **Year**: Choose prediction year (2023 onwards)
- **Month**: Select month of the year

### Output:

- **Birth Prediction**: Number of expected birth registrations
- **Confidence Information**: Model performance metrics
- **Visual Feedback**: Clear display of prediction results

## 🎯 Features

### Core Functionality:

- ✅ Real-time birth predictions
- ✅ Interactive web interface
- ✅ Support for all 14 NHS Board areas
- ✅ Temporal predictions (monthly resolution)
- ✅ Comprehensive logging and error handling

### Model Information:

- **Architecture**: 2-layer LSTM with 128 hidden units
- **Training Data**: Historical births (1998-2022)
- **Features**: 11 engineered features including lag variables
- **Accuracy**: ~98% on test data (1.25% SMAPE)

### User Experience:

- 🎨 Clean, professional interface
- 📱 Responsive design
- ⚡ Fast predictions (<1 second)
- 📊 Clear parameter visualization
- ℹ️ Helpful information and guidance

## 🏗️ Project Structure

```
scotland-birth-forecast/
├── app.py                 # Main Streamlit application
├── setup_ui.py           # Automated setup script
├── save_model_for_ui.py   # Model training and saving
├── requirements_ui.txt    # UI dependencies
├── models/                # Saved model components
│   ├── trained_lstm_model.pth
│   ├── feature_scaler.pkl
│   └── nhs_board_mapping.pkl
├── src/                   # Source code modules
│   ├── components/        # Model architecture and utilities
│   ├── logger.py          # Custom logging system
│   └── exception.py       # Custom exception handling
└── data/                  # Training data
    └── processed/         # Preprocessed parquet files
```

## 🔧 Technical Details

### Model Pipeline:

1. **Data Loading**: Historical birth registration data
2. **Feature Engineering**: Lag features, cyclical encoding, normalization
3. **Model Training**: LSTM with early stopping and validation
4. **Preprocessing**: StandardScaler for feature normalization
5. **Prediction**: Log-scale prediction with inverse transformation

### UI Components:

- **Streamlit Framework**: Modern web interface
- **Custom CSS**: Professional styling and theming
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed operation tracking

## 📊 NHS Board Areas Supported

- Ayrshire and Arran
- Borders
- Dumfries and Galloway
- Fife
- Forth Valley
- Grampian
- Greater Glasgow and Clyde
- Highland
- Lanarkshire
- Lothian
- Orkney
- Shetland
- Tayside
- Western Isles

## ⚠️ Important Notes

### Production Considerations:

- **Data Currency**: Model uses historical averages for lag features
- **Model Updates**: Retrain periodically with new data
- **Validation**: Cross-check predictions with domain experts
- **Scale**: Designed for demonstration; enhance for production use

### Limitations:

- Requires recent historical data for optimal lag feature performance
- Predictions are statistical estimates, not guarantees
- Model trained on 1998-2022 data patterns

## 🛠️ Development

### Adding New Features:

1. Extend the `BirthPredictor` class in `app.py`
2. Update model architecture in `src/components/model.py`
3. Enhance preprocessing in `save_model_for_ui.py`

### Customization:

- Modify CSS styling in the `st.markdown()` sections
- Add new input parameters to the sidebar
- Enhance visualization with additional charts

## 📝 Logging and Monitoring

- **Log Location**: Automatically timestamped in `logs/` directory
- **Coverage**: All major operations and predictions logged
- **Error Tracking**: Detailed exception information captured
- **Performance**: Model prediction times and accuracy tracked

## 🤝 Support

For issues or questions:

1. Check the console output for error messages
2. Review log files in the `logs/` directory
3. Ensure all model files are properly generated
4. Verify input parameters are within expected ranges

---

**Built with ❤️ using PyTorch, Streamlit, and Scikit-learn**
