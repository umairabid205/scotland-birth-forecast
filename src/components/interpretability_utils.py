import torch
import numpy as np
import matplotlib.pyplot as plt




# Attention visualization for Transformer

def visualize_attention_weights(model, input_seq, layer_idx=0, head_idx=0, device='cpu'):
    """
    Visualize attention weights for a given input sequence and transformer model.
    Args:
        model: TimeSeriesTransformer instance
        input_seq: torch.Tensor of shape (1, seq_len, features)
        layer_idx: which encoder layer to visualize
        head_idx: which attention head to visualize
        device: 'cpu' or 'cuda'
    """
    model.eval()
    input_seq = input_seq.to(device)
    # Forward pass to get attention weights
    with torch.no_grad():
        # Register a hook to extract attention weights
        attn_weights = {}
        def hook(module, input, output):
            attn_weights['weights'] = module.self_attn.attn_output_weights.detach().cpu().numpy()
        handle = model.transformer_encoder.layers[layer_idx].register_forward_hook(hook)
        _ = model(input_seq)
        handle.remove()
        # attn_weights['weights'] shape: (batch, num_heads, seq_len, seq_len)
        weights = attn_weights['weights'][0, head_idx]  # (seq_len, seq_len)
        plt.figure(figsize=(8, 6))
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Input Time Step')
        plt.ylabel('Output Time Step')
        plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
        plt.show()




# Integrated Gradients for LSTM/GRU

def integrated_gradients(model, input_seq, baseline=None, target_idx=0, steps=50, device='cpu'):
    """
    Compute Integrated Gradients for a sequence model (LSTM/GRU).
    Args:
        model: PyTorch model
        input_seq: torch.Tensor of shape (1, seq_len, features)
        baseline: torch.Tensor, same shape as input_seq (default: zeros)
        target_idx: index of output to attribute (default: 0)
        steps: number of interpolation steps
        device: 'cpu' or 'cuda'
    Returns:
        attributions: np.ndarray of shape (seq_len, features)
    """
    model.eval()
    input_seq = input_seq.to(device)
    if baseline is None:
        baseline = torch.zeros_like(input_seq).to(device)
    scaled_inputs = [baseline + (float(i) / steps) * (input_seq - baseline) for i in range(steps + 1)]
    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad = True
        output = model(scaled)
        target = output[0, target_idx]
        target.backward(retain_graph=True)
        grads.append(scaled.grad.detach().cpu().numpy())
        model.zero_grad()
    avg_grads = np.mean(grads, axis=0)
    attributions = (input_seq.cpu().numpy() - baseline.cpu().numpy()) * avg_grads
    return attributions



# Saliency map for LSTM/GRU

def saliency_map(model, input_seq, target_idx=0, device='cpu'):
    """
    Compute saliency map for a sequence model (LSTM/GRU).
    Args:
        model: PyTorch model
        input_seq: torch.Tensor of shape (1, seq_len, features)
        target_idx: index of output to attribute (default: 0)
        device: 'cpu' or 'cuda'
    Returns:
        saliency: np.ndarray of shape (seq_len, features)
    """
    model.eval()
    input_seq = input_seq.to(device)
    input_seq.requires_grad = True
    output = model(input_seq)
    target = output[0, target_idx]
    target.backward()
    saliency = input_seq.grad.detach().cpu().numpy()
    return saliency

# Example usage (not run):
# visualize_attention_weights(model, input_seq)
# attributions = integrated_gradients(model, input_seq)
# saliency = saliency_map(model, input_seq)
