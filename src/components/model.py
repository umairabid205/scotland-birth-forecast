import torch
import torch.nn as nn

# LSTM model for time series forecasting
class StackedLSTM(nn.Module):
    """Stacked LSTM model for time series forecasting."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        """
        Initialize the Stacked LSTM model.
        """
        
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y = self.fc(lstm_out[:, -1, :])
        return y

__all__ = ["StackedLSTM"]
