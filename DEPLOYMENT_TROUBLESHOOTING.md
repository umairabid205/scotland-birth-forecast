# Deployment Troubleshooting Guide

## Problem: Different Predictions Between Local and Deployment

### Quick Fix Checklist:

1. **Check the Debug Panel** (Most Important!)

   - Run the app and click "🚀 Generate Prediction"
   - Expand the "🔍 Debug Information" panel
   - Look for these key indicators:
     - `PyTorch Available: True/False`
     - `Custom Modules Available: True/False`
     - `Model Type: Real LSTM Model / Fallback/Dummy Model`

2. **Common Issues and Solutions:**

   **Issue: PyTorch Available = False**

   - Solution: Install PyTorch in deployment environment
   - Command: `pip install torch==2.7.1`
   - Or use: `requirements_deployment.txt`

   **Issue: Custom Modules Available = False**

   - Solution: Ensure `src/` directory is deployed
   - Check that `src/components/model.py` exists
   - Verify Python path includes project root

   **Issue: Model Type = Fallback/Dummy Model**

   - Solution: Check model files are present and readable
   - Verify file permissions in deployment
   - Ensure models/ directory contains all 3 files

3. **Deployment Environment Setup:**

   ```bash
   # Use exact version requirements
   pip install -r requirements_deployment.txt

   # Verify PyTorch installation
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"

   # Check custom modules
   python -c "from src.components.model import StackedLSTM; print('Custom modules OK')"

   # Verify model files
   ls -la models/
   ```

4. **Expected vs Problematic Outputs:**

   **✅ CORRECT (Local Environment):**

   - PyTorch Available: True
   - Custom Modules Available: True
   - Model Type: Real LSTM Model
   - Prediction: ~93 births (realistic for Greater Glasgow)

   **❌ PROBLEMATIC (Deployment Environment):**

   - PyTorch Available: False OR
   - Custom Modules Available: False OR
   - Model Type: Fallback/Dummy Model
   - Prediction: ~505 births (fallback/dummy result)

5. **File Structure Verification:**

   ```
   project/
   ├── app.py
   ├── requirements_deployment.txt
   ├── src/
   │   ├── components/
   │   │   └── model.py
   │   ├── logger.py
   │   └── exception.py
   └── models/
       ├── trained_lstm_model.pth (821KB)
       ├── feature_scaler.pkl (1KB)
       └── nhs_board_mapping.pkl (241 bytes)
   ```

6. **If Problems Persist:**
   - Check deployment logs for import errors
   - Verify Python version compatibility (3.8-3.12)
   - Ensure all dependencies in requirements_deployment.txt
   - Test with: `python check_deployment_differences.py`

## Key Insight:

The 93 vs 505 difference indicates your deployment is using the fallback prediction logic instead of the real LSTM model. This happens when PyTorch or custom modules fail to load properly.

## Solution Priority:

1. Fix PyTorch installation in deployment ⭐⭐⭐
2. Ensure src/ directory is deployed ⭐⭐⭐
3. Verify model files are present ⭐⭐
4. Use exact dependency versions ⭐
