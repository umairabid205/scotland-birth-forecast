

import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network module


# Stacked LSTM model for time series forecasting
class StackedLSTM(nn.Module):
    """
    Stacked LSTM model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(StackedLSTM, self).__init__()
        # LSTM layer (optionally bidirectional)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction = 2 if bidirectional else 1  # 2 for bidirectional, 1 otherwise
        # Output layer
        self.fc = nn.Linear(hidden_size * direction, 1)

    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)  # Final output
        return out
