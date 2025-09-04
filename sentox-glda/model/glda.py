import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LocalDifferentialAttention(nn.Module):
    """
    Local Differential Attention (LDA) module
    Captures short-range semantic variations using first and second-order differences
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Differential feature projections
        self.first_order_proj = nn.Linear(hidden_size, hidden_size)
        self.second_order_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def compute_differential_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute first and second-order differential features
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        Returns:
            first_order: First-order differences [batch_size, seq_len, hidden_size]
            second_order: Second-order differences [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # Initialize output tensors
        first_order = torch.zeros_like(x)
        second_order = torch.zeros_like(x)

        # Pad sequences for differential computation
        x_padded = F.pad(x, (0, 0, 1, 1), mode='replicate')  # [batch_size, seq_len+2, hidden_size]

        # First-order differences (gradient-like features)
        # Central difference for interior points
        if seq_len > 2:
            first_order[:, 1:-1] = (x_padded[:, 3:-1] - x_padded[:, 1:-3]) / 2.0

        # Forward difference at start
        first_order[:, 0] = x_padded[:, 1] - x_padded[:, 0]

        # Backward difference at end
        if seq_len > 1:
            first_order[:, -1] = x_padded[:, -1] - x_padded[:, -2]

        # Second-order differences (curvature-like features)
        # Central second difference for interior points
        if seq_len > 2:
            second_order[:, 1:-1] = x_padded[:, 3:-1] - 2 * x_padded[:, 2:-2] + x_padded[:, 1:-3]

        # Handle boundary conditions for second-order
        if seq_len > 1:
            second_order[:, 0] = x_padded[:, 2] - 2 * x_padded[:, 1] + x_padded[:, 0]
            second_order[:, -1] = x_padded[:, -1] - 2 * x_padded[:, -2] + x_padded[:, -3]

        return first_order, second_order

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Local Differential Attention
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            output: Attended features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        residual = x

        # Compute differential features
        first_order, second_order = self.compute_differential_features(x)

        # Project differential features
        first_order_features = self.first_order_proj(first_order)
        second_order_features = self.second_order_proj(second_order)

        # Combine original features with differential features
        enhanced_x = x + 0.1 * first_order_features + 0.05 * second_order_features

        # Multi-head attention computation
        Q = self.query_proj(enhanced_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(enhanced_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(enhanced_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, -1)
            scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Output projection
        output = self.out_proj(attn_output)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output


class GlobalDifferentialAttention(nn.Module):
    """
    Global Differential Attention (GDA) module
    Models long-range dependencies through downsampled sequence representation
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, downsample_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.downsample_ratio = downsample_ratio

        # Downsampling projection
        self.downsample_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=downsample_ratio,
                                         stride=downsample_ratio, padding=0)

        # Global attention projections
        self.global_query_proj = nn.Linear(hidden_size, hidden_size)
        self.global_key_proj = nn.Linear(hidden_size, hidden_size)
        self.global_value_proj = nn.Linear(hidden_size, hidden_size)

        # Local-global interaction projections
        self.local_query_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Upsampling for global features
        self.upsample = nn.Linear(hidden_size, hidden_size)

    def downsample_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample sequence to capture global context
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        Returns:
            downsampled: Downsampled tensor [batch_size, downsampled_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # Transpose for conv1d: [batch_size, hidden_size, seq_len]
        x_transposed = x.transpose(1, 2)

        # Apply downsampling convolution
        downsampled = self.downsample_conv(x_transposed)

        # Transpose back: [batch_size, downsampled_len, hidden_size]
        downsampled = downsampled.transpose(1, 2)

        return downsampled

    def upsample_and_align(self, global_features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Upsample global features to match original sequence length
        Args:
            global_features: Global features [batch_size, downsampled_len, hidden_size]
            seq_len: Target sequence length
        Returns:
            upsampled: Upsampled features [batch_size, seq_len, hidden_size]
        """
        batch_size, global_len, hidden_size = global_features.shape

        # Interpolate to match sequence length
        global_features_transposed = global_features.transpose(1, 2)  # [batch_size, hidden_size, global_len]
        upsampled = F.interpolate(global_features_transposed, size=seq_len, mode='linear', align_corners=False)
        upsampled = upsampled.transpose(1, 2)  # [batch_size, seq_len, hidden_size]

        # Apply projection
        upsampled = self.upsample(upsampled)

        return upsampled

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Global Differential Attention
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            output: Attended features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        residual = x

        # Downsample for global context
        global_x = self.downsample_sequence(x)
        global_len = global_x.shape[1]

        # Global self-attention on downsampled features
        global_Q = self.global_query_proj(global_x).view(batch_size, global_len, self.num_heads,
                                                         self.head_dim).transpose(1, 2)
        global_K = self.global_key_proj(global_x).view(batch_size, global_len, self.num_heads, self.head_dim).transpose(
            1, 2)
        global_V = self.global_value_proj(global_x).view(batch_size, global_len, self.num_heads,
                                                         self.head_dim).transpose(1, 2)

        # Global attention computation
        global_scores = torch.matmul(global_Q, global_K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        global_attn_weights = F.softmax(global_scores, dim=-1)
        global_attn_weights = self.dropout(global_attn_weights)

        global_attn_output = torch.matmul(global_attn_weights, global_V)
        global_attn_output = global_attn_output.transpose(1, 2).contiguous().view(batch_size, global_len, hidden_size)

        # Upsample global features to match original sequence length
        upsampled_global = self.upsample_and_align(global_attn_output, seq_len)

        # Local-global cross attention
        local_Q = self.local_query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        global_K_upsampled = upsampled_global.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        global_V_upsampled = upsampled_global.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention scores
        cross_scores = torch.matmul(local_Q, global_K_upsampled.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, -1)
            cross_scores.masked_fill_(mask == 0, -1e9)

        cross_attn_weights = F.softmax(cross_scores, dim=-1)
        cross_attn_weights = self.dropout(cross_attn_weights)

        cross_attn_output = torch.matmul(cross_attn_weights, global_V_upsampled)
        cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Final output projection
        output = self.out_proj(cross_attn_output)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output


class GLDA(nn.Module):
    """
    Global-Local-Differential Attention module combining LDA and GDA
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, downsample_ratio: int = 4,
                 dropout: float = 0.1, fusion_method: str = 'concat'):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method

        # Local and Global attention modules
        self.local_attention = LocalDifferentialAttention(hidden_size, num_heads, dropout)
        self.global_attention = GlobalDifferentialAttention(hidden_size, num_heads, downsample_ratio, dropout)

        # Fusion layer
        if fusion_method == 'concat':
            self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size)
        elif fusion_method == 'add':
            self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        elif fusion_method == 'gate':
            self.gate = nn.Linear(2 * hidden_size, 1)
            self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of GLDA
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            output: Final attended features [batch_size, seq_len, hidden_size]
        """
        # Apply local and global attention
        local_output = self.local_attention(x, attention_mask)
        global_output = self.global_attention(x, attention_mask)

        # Fusion
        if self.fusion_method == 'concat':
            # Concatenation fusion
            combined = torch.cat([local_output, global_output], dim=-1)
            output = self.fusion_proj(combined)
        elif self.fusion_method == 'add':
            # Addition fusion
            combined = local_output + global_output
            output = self.fusion_proj(combined)
        elif self.fusion_method == 'gate':
            # Gated fusion
            gate_input = torch.cat([local_output, global_output], dim=-1)
            gate_weights = torch.sigmoid(self.gate(gate_input))
            combined = gate_weights * local_output + (1 - gate_weights) * global_output
            output = self.fusion_proj(combined)

        # Final processing
        output = self.dropout(output)
        output = self.final_layer_norm(output)

        return output


def main():
    """Test GLDA module with sample data"""
    # Test parameters
    batch_size = 2
    seq_len = 128
    hidden_size = 768
    num_heads = 12

    # Create sample data
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    # Mask some tokens for testing
    attention_mask[0, 100:] = 0
    attention_mask[1, 110:] = 0

    print(f"Input shape: {x.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Test Local Differential Attention
    print("\n=== Testing Local Differential Attention ===")
    lda = LocalDifferentialAttention(hidden_size, num_heads)
    lda_output = lda(x, attention_mask)
    print(f"LDA output shape: {lda_output.shape}")
    print(f"LDA output mean: {lda_output.mean().item():.6f}")
    print(f"LDA output std: {lda_output.std().item():.6f}")

    # Test Global Differential Attention
    print("\n=== Testing Global Differential Attention ===")
    gda = GlobalDifferentialAttention(hidden_size, num_heads, downsample_ratio=4)
    gda_output = gda(x, attention_mask)
    print(f"GDA output shape: {gda_output.shape}")
    print(f"GDA output mean: {gda_output.mean().item():.6f}")
    print(f"GDA output std: {gda_output.std().item():.6f}")

    # Test full GLDA with different fusion methods
    fusion_methods = ['concat', 'add', 'gate']

    for fusion_method in fusion_methods:
        print(f"\n=== Testing GLDA with {fusion_method} fusion ===")
        glda = GLDA(hidden_size, num_heads, fusion_method=fusion_method)
        glda_output = glda(x, attention_mask)
        print(f"GLDA output shape: {glda_output.shape}")
        print(f"GLDA output mean: {glda_output.mean().item():.6f}")
        print(f"GLDA output std: {glda_output.std().item():.6f}")

        # Test gradient flow
        loss = glda_output.mean()
        loss.backward()
        print(f"Gradients computed successfully for {fusion_method} fusion")
        glda.zero_grad()

    print("\nGLDA module testing completed successfully!")


if __name__ == "__main__":
    main()
