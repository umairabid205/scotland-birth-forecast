# Streamlit Cloud Deployment Guide

## Files Required for Deployment

### Core Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

### Model Files (in `models/` directory)

- `trained_lstm_model.pth` - Trained PyTorch LSTM model
- `feature_scaler.pkl` - Feature scaling transformer
- `nhs_board_mapping.pkl` - NHS Board area mappings

### Source Code (in `src/` directory)

- `src/components/model.py` - StackedLSTM model definition
- `src/logger.py` - Logging configuration
- `src/exception.py` - Custom exception handling

## Deployment Steps

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub:**

   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select this repository and the `main` branch
   - Set the main file path to `app.py`
   - Click "Deploy"

### Option 2: Local Testing

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally:**
   ```bash
   streamlit run app.py
   ```

## Troubleshooting Common Issues

### 1. Module Import Errors

- Ensure all files in `src/` directory are included
- Check that `requirements.txt` includes all dependencies

### 2. Model Loading Issues

- Verify all model files are in the `models/` directory
- The app will create default components if files are missing

### 3. PyTorch Compatibility

- Using CPU-only PyTorch for better deployment compatibility
- Model loading includes fallback for older PyTorch versions

### 4. File Path Issues

- App uses relative paths from the script location
- Works both locally and in deployed environments

## Environment Variables (Optional)

No environment variables are required for basic functionality.

## Resource Requirements

- **Memory:** ~500MB (for PyTorch model)
- **CPU:** 1 core sufficient
- **Storage:** ~50MB for all files

## Performance Notes

- Model loading is cached using `@st.cache_resource`
- Predictions are fast (~100ms per request)
- Suitable for moderate traffic loads
