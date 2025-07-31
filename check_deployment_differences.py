#!/usr/bin/env python3
"""
Deployment Difference Checker
Helps identify differences between local and deployment environments
"""

import sys
import os
import platform
import subprocess
import importlib.util

def check_environment():
    """Check the current environment details"""
    print("=" * 60)
    print("ENVIRONMENT ANALYSIS")
    print("=" * 60)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path[:3]}...")  # Show first 3 paths
    
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    # Check critical dependencies
    dependencies = {
        'torch': 'PyTorch',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'pandas': 'Pandas', 
        'sklearn': 'Scikit-learn',
        'pickle': 'Pickle (built-in)'
    }
    
    for module, name in dependencies.items():
        try:
            if module == 'pickle':
                import pickle
                spec = importlib.util.find_spec('pickle')
                version = "Built-in"
            else:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'Unknown')
            
            print(f"✅ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: Not found - {e}")
    
    print("\n" + "=" * 60)
    print("PYTORCH SPECIFIC CHECKS")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ CPU Device: {torch.device('cpu')}")
        
        # Test tensor operations
        test_tensor = torch.zeros(1, 1, 11)
        print(f"✅ Tensor Creation: {test_tensor.shape}")
        
        # Test model loading capability
        try:
            from src.components.model import StackedLSTM
            print("✅ Custom StackedLSTM: Available")
        except ImportError as e:
            print(f"❌ Custom StackedLSTM: {e}")
            
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    print("\n" + "=" * 60)
    print("FILE SYSTEM CHECK")
    print("=" * 60)
    
    # Check model files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    print(f"Current directory: {current_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {os.path.exists(models_dir)}")
    
    if os.path.exists(models_dir):
        model_files = [
            'trained_lstm_model.pth',
            'feature_scaler.pkl', 
            'nhs_board_mapping.pkl'
        ]
        
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            exists = os.path.exists(file_path)
            if exists:
                size = os.path.getsize(file_path)
                print(f"✅ {file}: {size} bytes")
            else:
                print(f"❌ {file}: Not found")
    
    print("\n" + "=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)
    
    # Test prediction consistency
    try:
        # Import the app components
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if we can import without Streamlit
        print("Testing prediction components...")
        
        # Simulate the same prediction that's causing issues
        test_params = {
            'nhs_board': 'Greater Glasgow and Clyde',
            'year': 2025,
            'month': 7  # July
        }
        
        print(f"Test parameters: {test_params}")
        
        # This would require significant refactoring to test without Streamlit
        print("⚠️  Full prediction test requires Streamlit environment")
        print("💡 Run the app and check the debug information panel")
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("To fix deployment differences:")
    print("1. Ensure PyTorch is properly installed in deployment")
    print("2. Verify all model files are present and readable")
    print("3. Check that custom modules (src/components/model.py) are available")
    print("4. Compare dependency versions between local and deployment")
    print("5. Use the debug information panel in the app to identify issues")
    print("6. Consider using requirements.txt with exact versions")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_environment()
