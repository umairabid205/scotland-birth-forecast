import torch
import torch.nn as nn
import logging

# Set up logging for model operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# LSTM model for time series forecasting
class StackedLSTM(nn.Module):
    """
    Stacked LSTM model for time series forecasting.
    
    This model is designed to work with:
    1. MSE Loss + First Differences auxiliary loss for short-term fluctuations
    2. Adam optimizer with learning rate scheduling and early stopping
    3. Hyperparameter tuning for layers, hidden units, learning rate, and sequence length
    
    Architecture:
    - Multi-layer LSTM with configurable depth and hidden size
    - Dropout regularization between LSTM layers
    - Linear output layer for regression
    - Supports batch processing with batch_first=True
    """


    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        """
        Initialize the Stacked LSTM model.
        
        Args:
            input_size (int): Number of input features (for Scotland birth data)
            hidden_size (int): Number of hidden units in each LSTM layer (controls model capacity)
            num_layers (int): Number of stacked LSTM layers (depth of the network)
            dropout (float): Dropout probability between LSTM layers (0.0-1.0, for regularization)
        """
        # Log model initialization parameters
        logger.info(f"Initializing StackedLSTM model with parameters:")
        logger.info(f"  - input_size: {input_size} (number of features per timestep)")
        logger.info(f"  - hidden_size: {hidden_size} (LSTM hidden dimension)")
        logger.info(f"  - num_layers: {num_layers} (number of stacked LSTM layers)")
        logger.info(f"  - dropout: {dropout} (regularization strength)")
        
        # Call parent class constructor to initialize nn.Module
        super(StackedLSTM, self).__init__()
        
        # Store model hyperparameters as instance variables
        self.hidden_size = hidden_size  # Number of features in the hidden state
        self.num_layers = num_layers    # Number of recurrent layers
        
        # Create the main LSTM layer stack
        # input_size: number of expected features in the input x
        # hidden_size: number of features in the hidden state h
        # num_layers: number of recurrent layers (default: 1)
        # dropout: if non-zero, introduces dropout on outputs of each LSTM layer except the last
        # batch_first: if True, input and output tensors are (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,      # Input feature dimension
            hidden_size=hidden_size,    # Hidden state dimension
            num_layers=num_layers,      # Number of LSTM layers to stack
            dropout=dropout,            # Dropout between layers (only if num_layers > 1)
            batch_first=True           # Input format: (batch_size, sequence_length, input_size)
        )
        
        # Create the final linear layer for regression output
        # Maps from hidden_size to 1 output (birth count prediction)
        self.fc = nn.Linear(hidden_size, 1)
        
        # Log successful model creation
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"StackedLSTM model created successfully with {total_params:,} total parameters")



    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
                             batch_size: number of samples in the batch
                             sequence_length: number of time steps (usually 1 for this project)
                             input_size: number of features per time step
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
                         Each value represents predicted log-transformed birth count
        """
        # Log input tensor information
        logger.debug(f"Forward pass input shape: {x.shape}")
        logger.debug(f"Input tensor stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        # Pass input through LSTM layers
        # lstm_out: tensor of shape (batch_size, sequence_length, hidden_size)
        #          contains output features (h_t) from the last layer of the LSTM, for each t
        # hidden_states: tuple of (h_n, c_n) where:
        #               h_n: tensor of shape (num_layers, batch_size, hidden_size) - final hidden state
        #               c_n: tensor of shape (num_layers, batch_size, hidden_size) - final cell state
        lstm_out, hidden_states = self.lstm(x)
        
        # Log LSTM output information
        logger.debug(f"LSTM output shape: {lstm_out.shape}")
        logger.debug(f"LSTM output stats - min: {lstm_out.min().item():.4f}, max: {lstm_out.max().item():.4f}")
        
        # Extract the output from the last time step
        # lstm_out[:, -1, :] selects:
        # - : (all samples in batch)
        # - -1: (last time step in sequence)
        # - : (all hidden features)
        # This gives us shape (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        logger.debug(f"Last time step output shape: {last_output.shape}")
        
        # Pass through final linear layer to get prediction
        # Maps from hidden_size dimensions to 1 output value
        y = self.fc(last_output)
        
        # Log final output information
        logger.debug(f"Final output shape: {y.shape}")
        logger.debug(f"Final output stats - min: {y.min().item():.4f}, max: {y.max().item():.4f}")
        
        # Return the prediction (should be log-transformed birth count)
        return y

# Export the model class for use in other modules
__all__ = ["StackedLSTM"]
