import torch  # PyTorch main package
import torch.nn as nn  # Neural network module
import numpy as np  # For averaging losses
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler


# Mean Squared Error loss with optional auxiliary loss on first differences
def mse_with_diff_loss(y_pred, y_true, aux_weight=0.0):
    mse = nn.functional.mse_loss(y_pred, y_true)  # Standard MSE loss
    if aux_weight > 0:
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]  # First differences of predictions
        diff_true = y_true[:, 1:] - y_true[:, :-1]  # First differences of targets
        diff_loss = nn.functional.mse_loss(diff_pred, diff_true)  # MSE on differences
        return mse + aux_weight * diff_loss  # Weighted sum
    return mse  # Just MSE if no auxiliary loss



# Training loop with early stopping and learning rate scheduling
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, aux_weight=0.0, patience=10, device='cpu'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # LR scheduler
    best_val_loss = float('inf')  # Track best validation loss
    best_model = None  # Store best model weights
    patience_counter = 0  # Early stopping counter
    
    for epoch in range(epochs):  # Loop over epochs
        model.train()  # Set model to training mode
        train_losses = []  # Store training losses
        
        for x_batch, y_batch in train_loader:  # Loop over training batches
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move to device
            optimizer.zero_grad()  # Zero gradients
            y_pred = model(x_batch)  # Forward pass
            loss = mse_with_diff_loss(y_pred, y_batch, aux_weight)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_losses.append(loss.item())  # Store loss
        model.eval()  # Set model to evaluation mode
        val_losses = []  # Store validation losses
        with torch.no_grad():  # No gradient computation
            
            for x_val, y_val in val_loader:  # Loop over validation batches
                x_val, y_val = x_val.to(device), y_val.to(device)  # Move to device
                y_pred = model(x_val)  # Forward pass
                val_loss = mse_with_diff_loss(y_pred, y_val, aux_weight)  # Compute loss
                val_losses.append(val_loss.item())  # Store loss
        avg_train_loss = np.mean(train_losses)  # Average training loss
        avg_val_loss = np.mean(val_losses)  # Average validation loss
        scheduler.step(avg_val_loss)  # Step LR scheduler
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")  # Print progress
        
        if avg_val_loss < best_val_loss:  # If validation improves
            best_val_loss = avg_val_loss  # Update best loss
            best_model = model.state_dict()  # Save model weights
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1  # Increment patience
            if patience_counter >= patience:  # Early stopping
                print("Early stopping triggered.")
                break
    
    if best_model is not None:
        model.load_state_dict(best_model)  # Restore best model
    return model  # Return trained model

# Example usage (not run):
# from temporal_models import TemporalConvNet, TimeSeriesTransformer  # Import models
# model = TemporalConvNet(num_inputs=4, num_channels=[64, 64, 64])  # Create model
# trained_model = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, aux_weight=0.1, patience=10, device='cuda')  # Train model
