"""
Deployment verification script for Streamlit Cloud
This script checks if all required files and dependencies are available
"""
import os
import sys
import warnings

def check_deployment_readiness():
    """Check if the app is ready for deployment"""
    issues = []
    
    # Check if models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        issues.append("❌ Models directory not found")
    else:
        print("✅ Models directory found")
        
        # Check for required model files
        required_files = [
            "trained_lstm_model.pth",
            "feature_scaler.pkl", 
            "nhs_board_mapping.pkl"
        ]
        
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} found")
            else:
                issues.append(f"⚠️ {file} missing (will use defaults)")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} is compatible")
    else:
        issues.append(f"❌ Python {python_version.major}.{python_version.minor} may not be compatible")
    
    # Check for src directory
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    if os.path.exists(src_dir):
        print("✅ Source directory found")
    else:
        issues.append("❌ Source directory not found")
    
    # Summary
    if not issues:
        print("\n🎉 Deployment ready!")
    else:
        print("\n⚠️ Deployment issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    check_deployment_readiness()
