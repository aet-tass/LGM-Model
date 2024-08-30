import os
import warnings

from torch import Tensor
from torch import nn

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None  # Check if xFormers is enabled via environment variable
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind  # Import xFormers if enabled

        XFORMERS_AVAILABLE = True  # Flag indicating xFormers is available
        warnings.warn("xFormers is available (Attention)")  # Warning message about xFormers availability
    else:
        warnings.warn("xFormers is disabled (Attention)")  # Warning if xFormers is disabled
        raise ImportError  # Trigger ImportError to handle absence of xFormers
except ImportError:
    XFORMERS_AVAILABLE = False  # Flag indicating xFormers is not available
    warnings.warn("xFormers is not available (Attention)")  # Warning message about the unavailability of xFormers


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads  # Number of attention heads
        head_dim = dim // num_heads  # Dimension per head
        self.scale = head_dim**-0.5  # Scale factor for query to stabilize gradients

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Linear layer to compute queries, keys, and values
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout for attention probabilities
        self.proj = nn.Linear(dim, dim, bias=proj_bias)  # Linear layer to project output
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout for the final projection

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape  # Batch size, sequence length, and embedding dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]  # Separate queries, keys, and values and scale queries

        attn = q @ k.transpose(-2, -1)  # Compute attention scores by dot product of queries and keys
        attn = attn.softmax(dim=-1)  # Apply softmax to normalize attention scores
        attn = self.attn_drop(attn)  # Apply dropout to attention scores

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Compute the weighted sum of values based on attention
        x = self.proj(x)  # Apply the final linear projection
        x = self.proj_drop(x)  # Apply dropout to the final output
        return x  # Return the output


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")  # Raise error if xFormers is required but not available
            return super().forward(x)  # Fallback to regular attention if xFormers is unavailable

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)  # Unbind queries, keys, and values along the appropriate dimension

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)  # Use memory-efficient attention mechanism
        x = x.reshape([B, N, C])  # Reshape the output to the expected format

        x = self.proj(x)  # Apply the final linear projection
        x = self.proj_drop(x)  # Apply dropout to the final output
        return x  # Return the output


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim  # Output dimension
        self.num_heads = num_heads  # Number of attention heads
        head_dim = dim // num_heads  # Dimension per head
        self.scale = head_dim**-0.5  # Scale factor for query to stabilize gradients

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)  # Linear layer to project queries
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)  # Linear layer to project keys
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)  # Linear layer to project values
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout for attention probabilities
        self.proj = nn.Linear(dim, dim, bias=proj_bias)  # Linear layer to project output
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout for the final projection

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, N, _ = q.shape  # Batch size, sequence length of queries
        M = k.shape[1]  # Sequence length of keys/values
        
        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # Project and reshape queries
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # Project and reshape keys
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # Project and reshape values

        attn = q @ k.transpose(-2, -1)  # Compute attention scores by dot product of queries and keys

        attn = attn.softmax(dim=-1)  # Apply softmax to normalize attention scores
        attn = self.attn_drop(attn)  # Apply dropout to attention scores

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # Compute the weighted sum of values based on attention
        x = self.proj(x)  # Apply the final linear projection
        x = self.proj_drop(x)  # Apply dropout to the final output
        return x  # Return the output


class MemEffCrossAttention(CrossAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")  # Raise error if xFormers is required but not available
            return super().forward(q, k, v)  # Fallback to regular cross-attention if xFormers is unavailable

        B, N, _ = q.shape  # Batch size, sequence length of queries
        M = k.shape[1]  # Sequence length of keys/values

        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads)  # Project and reshape queries
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads)  # Project and reshape keys
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads)  # Project and reshape values

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)  # Use memory-efficient attention mechanism
        x = x.reshape(B, N, -1)  # Reshape the output to the expected format

        x = self.proj(x)  # Apply the final linear projection
        x = self.proj_drop(x)  # Apply dropout to the final output
        return x  # Return the output
