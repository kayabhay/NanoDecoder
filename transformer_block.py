# transformer_block.py
# -----------------------------------------------------------------------------
# Transformer Building Blocks (Decoder-Only Architecture)
# -----------------------------------------------------------------------------
# This file contains the standard components of a Transformer Decoder layer:
# 1. SelfAttentionHead: Single-head causal self-attention with masking
# 2. MultiHeadAttention: Multiple attention heads running in parallel
# 3. FeedForward: Position-wise non-linear feature transformation
# 4. Block: Complete Transformer layer with LayerNorm & Residual Connections
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Block 1: Single Self-Attention Head
# -----------------------------------------------------------------------------
class SelfAttentionHead(nn.Module):
    """
    Computes scaled dot-product attention for a single head.
    Uses causal masking so tokens can only look at past and current tokens,
    preventing them from looking ahead into future tokens during training/generation.
    """
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        # Linear projections to project input tokens into Query, Key, and Value spaces
        self.key = nn.Linear(embedding_dim, head_size, bias=False)    # "What information do I contain?"
        self.query = nn.Linear(embedding_dim, head_size, bias=False)  # "What information am I looking for?"
        self.value = nn.Linear(embedding_dim, head_size, bias=False)  # "What content do I pass along?"

        # Lower-triangular matrix buffer used for causal masking (1s on and below diagonal, 0s above)
        # Registering as a buffer means it moves with the model to GPU/CPU but is not a trainable parameter.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: (Batch size B, Sequence length T, Embedding dimension C)
        B, T, C = x.shape

        # 1. Project inputs into Key and Query representations
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 2. Compute attention scores ("affinity" between every pair of tokens)
        # Scaled dot-product: Q @ K^T / sqrt(head_size or embedding_dim)
        # Scaling by C**0.5 stabilizes gradients so softmax does not saturate.
        wei = q @ k.transpose(-2, -1) / (C ** 0.5) # Shape: (B, T, T)

        # 3. Apply causal mask: replace upper-triangle (future positions) with -infinity
        # Softmax of -infinity becomes 0, so tokens cannot attend to future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # 4. Convert affinity scores into a probability distribution along the last dimension
        wei = F.softmax(wei, dim=-1) # Shape: (B, T, T)

        # 5. Multiply attention probabilities by Values to get weighted summary of context
        v = self.value(x) # (B, T, head_size)
        out = wei @ v    # (B, T, head_size)
        return out


# -----------------------------------------------------------------------------
# Block 2: Multi-Head Attention
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Runs multiple Self-Attention heads in parallel.
    Each head allows the model to attend to information from different representation
    subspaces at different positions simultaneously.
    """
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        # Calculate the dimension size for each individual head
        head_size = embedding_dim // num_heads

        # Create a list of independent SelfAttentionHead instances
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, block_size, head_size) 
            for _ in range(num_heads)
        ])

        # Final projection layer to combine all head outputs back into embedding_dim
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        # 1. Run input x through each self-attention head in parallel
        # 2. Concatenate head outputs along the feature dimension (dim=-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 3. Project concatenated features back to embedding_dim
        return self.proj(out)


# -----------------------------------------------------------------------------
# Block 3: Feed-Forward Network
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) applied to each token independently.
    Expands feature space by 4x with non-linearity (ReLU) before contracting back.
    Allows the network to perform computation on the token features after attention mixing.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expand dimension (e.g. 32 -> 128)
            nn.ReLU(),                     # Non-linear activation function
            nn.Linear(4 * n_embd, n_embd)  # Project back to embedding dimension
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Block 4: Complete Transformer Decoder Block
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """
    Combines Multi-Head Self-Attention and Feed-Forward components with:
    - Pre-Layer Normalization (LayerNorm before each sub-layer)
    - Residual (Skip) Connections (adding input x to output of sub-layer)
    """
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        # Sub-layer 1: Multi-Head Self-Attention
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        
        # Sub-layer 2: Feed-Forward Network
        self.ffwd = FeedForward(embedding_dim)
        
        # Layer Normalization layers for feature standardization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Pre-LN Transformer Architecture:
        # 1. Apply LayerNorm, run through Attention, and add residual connection
        x = x + self.sa(self.ln1(x))

        # 2. Apply LayerNorm, run through Feed-Forward, and add residual connection
        x = x + self.ffwd(self.ln2(x))
        return x