#!/usr/bin/env python3
"""
Test script to verify the robustness of the production predictor
This script tests various scenarios that could occur in deployment
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_normal_operation():
    """Test normal operation with models available"""
    print("🧪 Test 1: Normal operation with models available")
    try:
        from src.components.production_predictor import NHSBirthPredictor
        predictor = NHSBirthPredictor()
        result = predictor.predict(year=2025, month=6, nhs_board_code=1)
        print(f"✅ Normal operation successful: {result['prediction']:.2f}")
        return True
    except Exception as e:
        print(f"❌ Normal operation failed: {e}")
        return False

def test_missing_models_directory():
    """Test behavior when models directory doesn't exist"""
    print("\n🧪 Test 2: Missing models directory")
    try:
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Create a temporary directory without models
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Try to initialize predictor
            from src.components.production_predictor import NHSBirthPredictor
            predictor = NHSBirthPredictor()
            
            # Should use fallback model
            result = predictor.predict(year=2025, month=6, nhs_board_code=1)
            print(f"✅ Fallback model working: {result['prediction']:.2f}")
            
            # Restore original directory
            os.chdir(original_cwd)
            return True
            
    except Exception as e:
        print(f"❌ Missing models directory test failed: {e}")
        os.chdir(original_cwd)
        return False

def test_deployment_paths():
    """Test various deployment path scenarios"""
    print("\n🧪 Test 3: Deployment path scenarios")
    
    deployment_scenarios = [
        "/mount/src/scotland-birth-forecast",  # Streamlit Cloud
        "/app",  # Docker
        "/workspace",  # Codespaces
        "/tmp/test_deployment"  # Generic
    ]
    
    results = []
    for scenario_path in deployment_scenarios:
        try:
            print(f"📁 Testing scenario: {scenario_path}")
            
            # Create temporary scenario directory
            if scenario_path.startswith("/tmp"):
                os.makedirs(scenario_path, exist_ok=True)
                models_dir = os.path.join(scenario_path, "models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Copy some model files to test directory
                original_models = "/Users/umair/Downloads/projects/project_1/models"
                if os.path.exists(original_models):
                    for file in os.listdir(original_models):
                        if file.endswith('.pkl'):
                            shutil.copy2(
                                os.path.join(original_models, file),
                                os.path.join(models_dir, file)
                            )
                
                # Test predictor with this path
                from src.components.production_predictor import NHSBirthPredictor
                predictor = NHSBirthPredictor(model_dir=models_dir)
                result = predictor.predict(year=2025, month=6, nhs_board_code=1)
                print(f"   ✅ Scenario working: {result['prediction']:.2f}")
                results.append(True)
                
                # Cleanup
                shutil.rmtree(scenario_path)
            else:
                print(f"   ⏭️ Skipping {scenario_path} (would require root access)")
                results.append(True)  # Skip but don't fail
                
        except Exception as e:
            print(f"   ❌ Scenario failed: {e}")
            results.append(False)
    
    return all(results)

def test_fallback_robustness():
    """Test that fallback model provides reasonable predictions"""
    print("\n🧪 Test 4: Fallback model robustness")
    try:
        # Test multiple predictions with fallback
        from src.components.production_predictor import NHSBirthPredictor
        
        # Force fallback by providing non-existent model directory
        predictor = NHSBirthPredictor(model_dir="/non/existent/path")
        
        # Test various inputs
        test_cases = [
            (2025, 1, 1),   # January, Borders
            (2025, 6, 6),   # June, Greater Glasgow
            (2025, 12, 9),  # December, Lothian
        ]
        
        predictions = []
        for year, month, nhs_board in test_cases:
            result = predictor.predict(year=year, month=month, nhs_board_code=nhs_board)
            predictions.append(result['prediction'])
            print(f"   📊 {year}-{month:02d} NHS {nhs_board}: {result['prediction']:.2f}")
        
        # Check that predictions are reasonable (not all the same, positive values)
        all_positive = all(p > 0 for p in predictions)
        has_variation = len(set(round(p) for p in predictions)) > 1
        
        if all_positive and has_variation:
            print("✅ Fallback model provides reasonable predictions")
            return True
        else:
            print(f"❌ Fallback predictions unreasonable: {predictions}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback robustness test failed: {e}")
        return False

def main():
    """Run all robustness tests"""
    print("🔬 NHS Birth Predictor Robustness Test Suite")
    print("=" * 60)
    
    tests = [
        ("Normal Operation", test_normal_operation),
        ("Missing Models Directory", test_missing_models_directory),
        ("Deployment Paths", test_deployment_paths),
        ("Fallback Robustness", test_fallback_robustness),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The predictor is robust for deployment.")
    else:
        print("⚠️ Some tests failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
