import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sys
import os
import warnings
from typing import Any, Union

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import PyTorch with error handling for deployment
try:
    import torch
    TORCH_AVAILABLE = True
    torch_module = torch
except ImportError:
    TORCH_AVAILABLE = False
    torch_module = None

# Import custom components with error handling
try:
    # Add current directory to Python path for deployment compatibility
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from src.components.model import StackedLSTM  # type: ignore
    from src.logger import logging
    from src.exception import CustomException
    CUSTOM_MODULES_AVAILABLE = True

except (ImportError, KeyError, ModuleNotFoundError) as e:
    CUSTOM_MODULES_AVAILABLE = False
    print(f"⚠️ Custom modules import failed: {e}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🔍 Python path: {sys.path[:3]}")  # Show first 3 entries
    print(f"📂 Directory contents: {os.listdir('.')}")
    if os.path.exists('src'):
        print(f"✅ src/ directory exists: {os.listdir('src')}")
    else:
        print("❌ src/ directory not found")
    
    # Create dummy logger and exception for fallback
    class DummyLogger:
        @staticmethod
        def info(msg): print(f"INFO: {msg}")
        @staticmethod
        def warning(msg): print(f"WARNING: {msg}")
        @staticmethod
        def error(msg): print(f"ERROR: {msg}")

    logging = DummyLogger()
    CustomException = Exception



    # Dummy StackedLSTM fallback
    class StackedLSTM:
        """ Dummy StackedLSTM class for fallback in case of import failure. """
        def __init__(self, *args, **kwargs):
            self.is_dummy = True  # Mark as dummy model
            pass
        def load_state_dict(self, *args, **kwargs):
            pass
        def eval(self):
            pass
        def __call__(self, x):
            # Return a dummy result
            class DummyResult:
                def item(self):
                    return np.log1p(100.0)  # This gives ~100 births after expm1
            return DummyResult()


# Configure Streamlit page
st.set_page_config(
    page_title="Scotland Birth Forecast",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F18F01;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    .info-box {
        background-color: #E8F4FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)



class BirthPredictor:
    """Class to handle birth prediction using a trained model and preprocessing components."""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.nhs_board_mapping = {}
        self.model_loaded = False
        
    def load_model_and_scaler(self):
        """Load the trained model and preprocessing components"""
        try:
            # Debug logging for deployment
            logging.info(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")
            logging.info(f"CUSTOM_MODULES_AVAILABLE: {CUSTOM_MODULES_AVAILABLE}")
            
            if not TORCH_AVAILABLE:
                st.error("❌ PyTorch not available. This will affect prediction accuracy.")
                st.warning("⚠️ Running with fallback prediction mode - results may be inaccurate.")
                
            if not CUSTOM_MODULES_AVAILABLE:
                st.error("❌ Custom modules not available. This will affect prediction accuracy.")
                st.warning("⚠️ Running with dummy model - results may be inaccurate.")
                
            # Force fail if critical dependencies missing in production
            if not TORCH_AVAILABLE or not CUSTOM_MODULES_AVAILABLE:
                st.error("🚫 Critical dependencies missing. Please check deployment configuration.")
                if st.button("🔄 Retry Loading"):
                    st.rerun()
                return False
                
            logging.info("Loading trained model and preprocessing components...")
            
            # Get the current directory and construct absolute paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "models")
            
            logging.info(f"Current directory: {current_dir}")
            logging.info(f"Models directory: {models_dir}")
            
            # Ensure models directory exists
            if not os.path.exists(models_dir):
                st.error("⚠️ Models directory not found. Please ensure model files are uploaded.")
                return False
            
            # Load the trained model
            model_path = os.path.join(models_dir, "trained_lstm_model.pth")
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
            mapping_path = os.path.join(models_dir, "nhs_board_mapping.pkl")

            # Check file existence with detailed logging
            logging.info(f"Model path exists: {os.path.exists(model_path)}")
            logging.info(f"Scaler path exists: {os.path.exists(scaler_path)}")
            logging.info(f"Mapping path exists: {os.path.exists(mapping_path)}")
            
            if os.path.exists(model_path):
                # Initialize model architecture
                self.model = StackedLSTM(input_size=11, hidden_size=128, num_layers=2)
                
                # Load model with CPU mapping for deployment compatibility
                model_load_success = False
                try:
                    if TORCH_AVAILABLE and torch_module:
                        # Try modern PyTorch first
                        try:
                            state_dict = torch_module.load(model_path, map_location='cpu', weights_only=True)
                            logging.info("Loaded model with weights_only=True")
                        except TypeError:
                            # Fallback for older PyTorch versions
                            state_dict = torch_module.load(model_path, map_location='cpu')
                            logging.info("Loaded model with legacy method")
                        
                        self.model.load_state_dict(state_dict)
                        self.model.eval()
                        
                        # Verify model is actually loaded
                        test_input = torch_module.zeros(1, 1, 11)
                        with torch_module.no_grad():
                            test_output = self.model(test_input)
                            logging.info(f"Model test output: {test_output.item()}")
                            
                        model_load_success = True
                    else:
                        raise Exception("PyTorch not available")
                        
                except Exception as e:
                    logging.error(f"Failed to load model: {str(e)}")
                    st.error(f"❌ Failed to load model: {str(e)}")
                    return False
                
                if model_load_success:
                    logging.info("✅ Model loaded successfully")
                else:
                    logging.error("❌ Model loading failed")
                    return False
            else:
                st.error(f"⚠️ Model file not found at: {model_path}")
                st.info("Please ensure the trained model file 'trained_lstm_model.pth' is in the models/ directory")
                return False
                
            # Load scaler with comprehensive error handling
            if os.path.exists(scaler_path):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(scaler_path, 'rb') as f: # Load the scaler 
                            self.scaler = pickle.load(f)
                    logging.info("✅ Scaler loaded successfully")
                    
                    # DEBUG: Log scaler parameters immediately after loading
                    if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                        logging.info(f"🔍 Loaded scaler mean: {self.scaler.mean_}")
                        logging.info(f"🔍 Loaded scaler scale: {self.scaler.scale_}")
                    
                except Exception as e:
                    logging.warning(f"❌ Error loading scaler: {e}. Creating default scaler.")
                    # Create a default scaler if loading fails
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                    # Fit with dummy data based on expected feature ranges
                    dummy_data = np.array([[2010, 6, 7, 0, 1, 0.5, 6, 6, 6, 6, 0]] * 100)
                    self.scaler.fit(dummy_data)
            else:
                logging.warning(f"Scaler file not found at: {scaler_path}. Creating default scaler.")
                # Create default scaler
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                dummy_data = np.array([[2010, 6, 7, 0, 1, 0.5, 6, 6, 6, 6, 0]] * 100)
                self.scaler.fit(dummy_data)
                
            # Load NHS Board mapping
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'rb') as f:
                        self.nhs_board_mapping = pickle.load(f)
                    logging.info("NHS Board mapping loaded successfully")
                except Exception as e:
                    logging.warning(f"Error loading NHS mapping: {e}. Using default mapping.")
                    self.nhs_board_mapping = self._get_default_mapping()
            else:
                logging.warning(f"NHS Board mapping file not found. Using default mapping.")
                self.nhs_board_mapping = self._get_default_mapping()
                
            self.model_loaded = True
            return True
            
        except Exception as e:
            logging.error(f"Error loading model components: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def _get_default_mapping(self):
        """Get default NHS Board mapping"""
        return {
            'Ayrshire and Arran': 0, 'Borders': 1, 'Dumfries and Galloway': 2,
            'Fife': 3, 'Forth Valley': 4, 'Grampian': 5, 'Greater Glasgow and Clyde': 6,
            'Highland': 7, 'Lanarkshire': 8, 'Lothian': 9, 'Orkney': 10,
            'Shetland': 11, 'Tayside': 12, 'Western Isles': 13
        }
    
    def preprocess_input(self, year, month, nhs_board):
        """Preprocess user input into model-ready format"""
        try:
            if not TORCH_AVAILABLE:
                raise Exception("PyTorch not available for tensor operations")
                
            # Create base features
            month_num = month
            nhs_board_code = self.nhs_board_mapping.get(nhs_board, 0)
            
            # DEBUG: Log input values
            logging.info(f"🔍 PREPROCESSING DEBUG:")
            logging.info(f"   Input: year={year}, month={month}, nhs_board={nhs_board}")
            logging.info(f"   NHS board code: {nhs_board_code}")
            
            # Cyclical encoding for month
            month_sin = np.sin(2 * np.pi * month_num / 12)
            month_cos = np.cos(2 * np.pi * month_num / 12)
            
            # Normalize year (based on training range 1998-2022)
            year_norm = (year - 1998) / (2022 - 1998)
            
            # Create lag features (using historical averages as placeholders)
            # In a real scenario, you'd need recent historical data
            avg_births = 6.2  # Log-transformed average from training data
            birth_lag_1 = avg_births
            birth_lag_2 = avg_births
            birth_lag_3 = avg_births
            birth_rolling_avg = avg_births
            birth_trend = 0.0
            
            logging.info(f"   Cyclical: sin={month_sin:.6f}, cos={month_cos:.6f}")
            logging.info(f"   Year norm: {year_norm:.6f}")
            logging.info(f"   Lag features: {birth_lag_1}")
            
            # Create feature vector
            features = np.array([[
                year, month_num, nhs_board_code, month_sin, month_cos, year_norm,
                birth_lag_1, birth_lag_2, birth_lag_3, birth_rolling_avg, birth_trend
            ]], dtype=np.float32)
            
            logging.info(f"   Raw features: {features[0]}")
            
            # Apply scaling if scaler is available
            if self.scaler is not None:
                logging.info(f"   Scaler type: {type(self.scaler)}")
                
                # DEBUG: Log scaler parameters
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    logging.info(f"   Scaler mean (sample): {self.scaler.mean_[:5]}")
                if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                    logging.info(f"   Scaler scale (sample): {self.scaler.scale_[:5]}")
                
                features = self.scaler.transform(features)
                logging.info(f"   Scaled features: {features[0]}")
            else:
                logging.warning("   No scaler available!")
            
            # Convert to tensor format expected by model
            if TORCH_AVAILABLE and torch_module:
                features_tensor = torch_module.tensor(features, dtype=torch_module.float32).unsqueeze(1)  # (1, 1, 11)
                logging.info(f"   Tensor shape: {features_tensor.shape}")
            else:
                # Create a dummy tensor-like object
                class DummyTensor:
                    def __init__(self, data):
                        self.data = data
                        self.shape = (1, 1, 11)
                    def unsqueeze(self, dim):
                        return self
                features_tensor = DummyTensor(features)
                logging.info(f"   Dummy tensor shape: {features_tensor.shape}")
            
            return features_tensor
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise CustomException(e, sys)
    
    def predict(self, year, month, nhs_board):
        """Make prediction for given inputs"""
        try:
            # Debug logging
            logging.info(f"Prediction request: {nhs_board}, {month}/{year}")
            logging.info(f"Model loaded: {self.model_loaded}")
            logging.info(f"PyTorch available: {TORCH_AVAILABLE}")
            logging.info(f"Using real model: {self.model is not None and not hasattr(self.model, 'is_dummy')}")
            
            if not self.model_loaded:
                raise Exception("Model not loaded. Please check dependencies.")
            
            if not TORCH_AVAILABLE:
                st.warning("⚠️ PyTorch not available - using fallback prediction mode")
                # Return a more realistic fallback based on NHS board size
                board_multipliers = {
                    'Greater Glasgow and Clyde': 450, 'Lothian': 400, 'Lanarkshire': 300,
                    'Grampian': 250, 'Tayside': 200, 'Ayrshire and Arran': 180,
                    'Highland': 150, 'Fife': 170, 'Forth Valley': 130,
                    'Dumfries and Galloway': 80, 'Borders': 60, 'Western Isles': 15,
                    'Orkney': 12, 'Shetland': 14
                }
                fallback_prediction = board_multipliers.get(nhs_board, 100)
                logging.warning(f"Using fallback prediction: {fallback_prediction}")
                return fallback_prediction
            
            # Preprocess input
            features = self.preprocess_input(year, month, nhs_board)
            
            # Make prediction
            if self.model is None:
                raise Exception("Model is not loaded. Please check the model file and loading process.")
            
            # Debug feature values
            if hasattr(features, 'data'):
                logging.info(f"Features shape: {features.shape}")
                if hasattr(features, 'numpy'):
                    feature_values = features.data if hasattr(features, 'data') else features
                    logging.info(f"Feature values: {feature_values}")
            
            if TORCH_AVAILABLE and torch_module:
                with torch_module.no_grad():
                    log_prediction = self.model(features).item()
                    logging.info(f"Raw model output (log scale): {log_prediction}")
            else:
                # This should not happen due to earlier check
                log_prediction = self.model(features).item()
                logging.info(f"Fallback model output: {log_prediction}")
            
            # Convert back from log scale
            prediction = np.expm1(log_prediction)  # Inverse of log1p
            logging.info(f"Prediction after exp transformation: {prediction}")
            
            final_prediction = max(0, int(round(prediction)))
            logging.info(f"Final prediction: {final_prediction} births for {nhs_board} in {month}/{year}")
            
            return final_prediction
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            # Provide debug info to user
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Debug info: Check that all model files are properly loaded")
            raise CustomException(e, sys)

# Initialize predictor
@st.cache_resource
def get_predictor():
    predictor = BirthPredictor()
    predictor.load_model_and_scaler()
    return predictor

def main():
    # Check deployment environment
    is_deployed = not os.path.exists('/Users') and not os.path.exists('/home/umair')
    
    if is_deployed:
        st.info("🚀 Running in deployed environment")
    
    # Main header
    st.markdown('<h1 class="main-header">🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland Birth Forecast System</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">📊 Input Parameters</h2>', unsafe_allow_html=True)
    
    # NHS Board selection
    nhs_boards = [
        'Ayrshire and Arran', 'Borders', 'Dumfries and Galloway', 'Fife',
        'Forth Valley', 'Grampian', 'Greater Glasgow and Clyde', 'Highland',
        'Lanarkshire', 'Lothian', 'Orkney', 'Shetland', 'Tayside', 'Western Isles'
    ]
    
    selected_nhs_board = st.sidebar.selectbox(
        "🏥 Select NHS Board Area:",
        nhs_boards,
        index=6  # Default to Greater Glasgow and Clyde
    )
    
    # Year selection
    current_year = datetime.now().year
    selected_year = st.sidebar.slider(
        "📅 Select Year:",
        min_value=2023,
        max_value=current_year + 5,
        value=current_year,
        step=1
    )
    
    # Month selection
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    selected_month_name = st.sidebar.selectbox(
        "🗓️ Select Month:",
        months,
        index=datetime.now().month - 1
    )
    selected_month = months.index(selected_month_name) + 1
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🔮 Birth Prediction", "📊 Dashboard Analytics"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔮 Birth Prediction</h2>', unsafe_allow_html=True)
        
        # Display selected parameters
        st.markdown(f"""
        <div class="info-box">
            <h4>Selected Parameters:</h4>
            <ul>
                <li><strong>NHS Board:</strong> {selected_nhs_board}</li>
                <li><strong>Year:</strong> {selected_year}</li>
                <li><strong>Month:</strong> {selected_month_name}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🚀 Generate Prediction", type="primary"):
            try:
                # Load predictor
                predictor = get_predictor()
                
                # Show debug information
                with st.expander("🔍 Debug Information", expanded=False):
                    st.write(f"**PyTorch Available:** {TORCH_AVAILABLE}")
                    st.write(f"**Custom Modules Available:** {CUSTOM_MODULES_AVAILABLE}")
                    st.write(f"**Model Loaded:** {predictor.model_loaded}")
                    st.write(f"**Environment:** {'Deployed' if is_deployed else 'Local'}")
                    st.write(f"**Python Version:** {sys.version}")
                    
                    if predictor.model is not None:
                        model_type = "Real LSTM Model" if not hasattr(predictor.model, 'is_dummy') else "Fallback/Dummy Model"
                        st.write(f"**Model Type:** {model_type}")
                    
                    # Add scaler debug info
                    if predictor.scaler is not None:
                        if hasattr(predictor.scaler, 'mean_') and predictor.scaler.mean_ is not None:
                            st.write(f"**Scaler Mean (first 3):** {predictor.scaler.mean_[:3]}")
                        if hasattr(predictor.scaler, 'scale_') and predictor.scaler.scale_ is not None:
                            st.write(f"**Scaler Scale (first 3):** {predictor.scaler.scale_[:3]}")
                    else:
                        st.write("**Scaler:** Not loaded")
                    
                    # Add NHS board mapping debug
                    nhs_code = predictor.nhs_board_mapping.get(selected_nhs_board, -1)
                    st.write(f"**NHS Board Code:** {selected_nhs_board} → {nhs_code}")
                
                if predictor.model_loaded:
                    with st.spinner('🔄 Generating prediction...'):
                        # Make prediction
                        prediction = predictor.predict(selected_year, selected_month, selected_nhs_board)
                        
                        # Display result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Birth Registrations</h3>
                            <div class="prediction-value">{prediction:,}</div>
                            <p>births expected in {selected_month_name} {selected_year}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional insights with debugging
                        if TORCH_AVAILABLE and CUSTOM_MODULES_AVAILABLE:
                            st.success(f"✅ Prediction generated successfully for {selected_nhs_board}")
                        else:
                            st.warning(f"⚠️ Prediction generated using fallback mode for {selected_nhs_board}")
                            st.info("💡 For accurate predictions, ensure PyTorch and custom modules are properly installed")
                        
                        # Log the prediction
                        logging.info(f"UI Prediction: {prediction} births for {selected_nhs_board} in {selected_month_name} {selected_year}")
                        
                else:
                    st.error("❌ Model not loaded. Please check if the trained model files exist.")
                    st.info("💡 Try refreshing the page or check the deployment logs for errors.")
                    
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")
                logging.error(f"UI Error: {str(e)}")
                
                # Show detailed error info
                with st.expander("🔧 Error Details", expanded=True):
                    st.code(str(e))
                    st.write("**Possible solutions:**")
                    st.write("1. Check that all model files are present")
                    st.write("2. Verify PyTorch installation")
                    st.write("3. Ensure proper file permissions")
                    st.write("4. Check deployment logs for missing dependencies")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Scotland Birth Rate Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Interactive Tableau Dashboard</h4>
            <p>Explore historical birth data trends across Scottish NHS Board areas with interactive visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Embed Tableau dashboard
        tableau_url = "https://public.tableau.com/views/ScotlandsBirthrate/Dashboard1"
        
        # Create iframe for Tableau dashboard
        tableau_embed_code = f"""
        <div style="margin: 20px 0;">
            <iframe 
                src="{tableau_url}?:embed=yes&:display_count=yes&:origin=viz_share_link&:showVizHome=no" 
                width="100%" 
                height="800" 
                style="border: none; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            </iframe>
        </div>
        """
        
        st.markdown(tableau_embed_code, unsafe_allow_html=True)
        
        # Additional information about the dashboard
        st.markdown("""
        ### 📈 Dashboard Features:
        - **Historical Trends**: View birth registration patterns over time
        - **Regional Analysis**: Compare birth rates across NHS Board areas
        - **Interactive Filters**: Explore data by year, month, and region
        - **Visual Insights**: Charts and graphs for better understanding
        
        ### 🔗 External Access:
        You can also view this dashboard directly on Tableau Public: 
        [Scotland's Birth Rate Dashboard](https://public.tableau.com/app/profile/umairabid205/viz/ScotlandsBirthrate/Dashboard1)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Scotland Birth Forecast System | Built with Streamlit & PyTorch | 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
