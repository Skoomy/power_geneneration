"""Temporal Fusion Transformer model for quantile regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + query)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.1, use_time_distributed: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_time_distributed = use_time_distributed
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu1 = nn.ELU()
        
        # Layer 2 
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.elu2 = nn.ELU()
        
        # Gating mechanism
        self.gate = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # Normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        skip = self.skip(x)
        
        # Main pathway
        hidden = self.fc1(x)
        hidden = self.elu1(hidden)
        hidden = self.dropout(hidden)
        
        hidden = self.fc2(hidden)
        hidden = self.elu2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating
        gate = self.sigmoid(self.gate(x))
        
        # Gated addition
        output = gate * hidden + (1 - gate) * skip
        
        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for choosing relevant features."""
    
    def __init__(self, input_dim: int, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        
        # Feature weights
        self.feature_weights = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, 1, dropout) 
            for _ in range(num_features)
        ])
        
        # Softmax for weights
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, context: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        # Calculate weights for each feature
        weights = []
        for i, weight_fn in enumerate(self.feature_weights):
            weight = weight_fn(context)
            weights.append(weight)
        
        weights = torch.cat(weights, dim=-1)
        weights = self.softmax(weights)
        
        # Weighted combination of features
        output = torch.zeros_like(features[0])
        for i, feature in enumerate(features):
            output += weights[..., i:i+1] * feature
            
        return output, weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for probabilistic time series forecasting."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 num_quantiles: int = 3,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 dropout: float = 0.1,
                 forecast_horizon: int = 24):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.quantiles = quantiles
        self.forecast_horizon = forecast_horizon
        
        # Input processing
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Static context enrichment
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Temporal self-attention
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Position-wise feed-forward
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.feed_forward_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Quantile output layers
        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_quantiles)
        ])
        
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
        
    def forward(self, x: torch.Tensor, future_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Input embedding
        embedded = self.input_embedding(x)
        
        # LSTM encoding for temporal patterns
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Static enrichment
        enriched = self.static_enrichment(lstm_out)
        
        # Self-attention blocks
        attention_out = enriched
        attention_mask = self.create_attention_mask(seq_len, device)
        
        for i in range(len(self.attention_layers)):
            # Multi-head attention
            attention_out = self.attention_layers[i](
                attention_out, attention_out, attention_out, attention_mask
            )
            
            # Feed-forward
            ff_out = self.feed_forward[i](attention_out)
            attention_out = self.feed_forward_norm[i](ff_out + attention_out)
        
        # For forecasting, we use the last hidden state to predict future values
        if future_features is not None:
            # Use decoder for future predictions
            future_embedded = self.input_embedding(future_features)
            future_out, _ = self.lstm(future_embedded, (hidden, cell))
            forecast_input = future_out
        else:
            # Use last state repeated for forecast horizon
            last_state = attention_out[:, -1:, :]
            forecast_input = last_state.repeat(1, self.forecast_horizon, 1)
        
        # Generate quantile predictions
        quantile_outputs = []
        for i, quantile_proj in enumerate(self.quantile_projections):
            quantile_pred = quantile_proj(forecast_input)
            quantile_outputs.append(quantile_pred)
        
        # Stack quantile predictions
        predictions = torch.cat(quantile_outputs, dim=-1)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_out,
            'quantiles': self.quantiles
        }


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Expand targets to match quantile dimensions
        targets = targets.unsqueeze(-1).expand_as(predictions)
        
        # Calculate quantile loss
        errors = targets - predictions
        quantiles = self.quantiles.to(predictions.device).view(1, 1, -1)
        
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        
        return loss.mean()