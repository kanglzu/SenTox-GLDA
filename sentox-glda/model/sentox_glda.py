import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from glda import GLDA
import math
from typing import Dict, Tuple, Optional


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion module for dynamically balancing sentiment and toxicity features
    Based on the gating mechanism described in the paper
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Gating network
        self.gate_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.gate_activation = nn.Tanh()
        self.gate_output = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, sentiment_features: torch.Tensor, toxicity_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion of sentiment and toxicity features
        Args:
            sentiment_features: [batch_size, seq_len, hidden_size]
            toxicity_features: [batch_size, seq_len, hidden_size]
        Returns:
            fused_features: [batch_size, seq_len, hidden_size]
        """
        # Concatenate features for gating
        concat_features = torch.cat([sentiment_features, toxicity_features], dim=-1)

        # Compute gating weights: g = σ(W_g[h_s; h_t] + b_g)
        gate_hidden = self.gate_activation(self.gate_projection(concat_features))
        gate_weights = torch.sigmoid(self.gate_output(gate_hidden))  # [batch_size, seq_len, 1]

        # Adaptive fusion: h_f = g · h_s + (1 - g) · h_t
        fused_features = gate_weights * sentiment_features + (1 - gate_weights) * toxicity_features

        # Apply dropout and layer normalization
        fused_features = self.dropout(fused_features)
        fused_features = self.layer_norm(fused_features)

        return fused_features


class KANClassifier(nn.Module):
    """
    Kolmogorov-Arnold Networks (KAN) classifier using B-spline basis functions
    Replaces traditional MLPs as described in the paper
    """

    def __init__(self, input_size: int, num_classes: int, num_splines: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_splines = num_splines

        # B-spline parameters
        self.spline_weights = nn.Parameter(torch.randn(input_size, num_splines))
        self.spline_bias = nn.Parameter(torch.zeros(input_size))

        # Output projection
        self.output_projection = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize KAN parameters"""
        nn.init.normal_(self.spline_weights, std=0.1)
        nn.init.zeros_(self.spline_bias)

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions
        Args:
            x: Input tensor [batch_size, input_size]
        Returns:
            basis: B-spline basis values [batch_size, input_size, num_splines]
        """
        # Normalize input to [0, 1] range
        x_norm = torch.sigmoid(x)

        # Create knot vector for B-spline
        knots = torch.linspace(0, 1, self.num_splines + 2, device=x.device)

        # Compute B-spline basis functions (simplified cubic B-splines)
        basis_values = []
        for i in range(self.num_splines):
            t0, t1, t2, t3 = knots[i:i + 4]

            # Cubic B-spline basis function
            mask1 = (x_norm >= t0) & (x_norm < t1)
            mask2 = (x_norm >= t1) & (x_norm < t2)
            mask3 = (x_norm >= t2) & (x_norm < t3)

            basis = torch.zeros_like(x_norm)

            # Piecewise cubic polynomial
            dt1, dt2, dt3 = t1 - t0, t2 - t1, t3 - t2

            if dt1 > 0:
                basis += mask1 * ((x_norm - t0) / dt1) ** 3
            if dt2 > 0:
                basis += mask2 * (1 - 3 * ((x_norm - t1) / dt2) ** 2 + 3 * ((x_norm - t1) / dt2) ** 3)
            if dt3 > 0:
                basis += mask3 * (1 - (x_norm - t2) / dt3) ** 3

            basis_values.append(basis)

        return torch.stack(basis_values, dim=-1)  # [batch_size, input_size, num_splines]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of KAN classifier
        Args:
            x: Input features [batch_size, seq_len, input_size] or [batch_size, input_size]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Handle sequence input by pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # Global average pooling: [batch_size, input_size]

        # Compute B-spline basis functions
        basis = self.b_spline_basis(x)  # [batch_size, input_size, num_splines]

        # Apply spline weights
        kan_output = torch.sum(basis * self.spline_weights.unsqueeze(0), dim=-1)  # [batch_size, input_size]
        kan_output = kan_output + self.spline_bias

        # Apply dropout
        kan_output = self.dropout(kan_output)

        # Final classification
        logits = self.output_projection(kan_output)

        return logits


class SenToxGLDA(nn.Module):
    """
    Complete SenTox-GLDA model for Chinese online abuse detection
    Combines dual encoders, adaptive fusion, GLDA attention, and KAN classifier
    """

    def __init__(self,
                 sentiment_model_path: str,
                 toxicity_model_path: str,
                 num_classes: int = 2,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 glda_downsample_ratio: int = 4,
                 dropout: float = 0.1,
                 freeze_encoders: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Dual encoders
        self.sentiment_encoder = BertModel.from_pretrained(sentiment_model_path)
        self.toxicity_encoder = RobertaModel.from_pretrained(toxicity_model_path)

        # Freeze encoder parameters if specified
        if freeze_encoders:
            for param in self.sentiment_encoder.parameters():
                param.requires_grad = False
            for param in self.toxicity_encoder.parameters():
                param.requires_grad = False

        # Adaptive fusion module
        self.adaptive_fusion = AdaptiveFusionModule(hidden_size, dropout)

        # GLDA attention mechanism
        self.glda_attention = GLDA(
            hidden_size=hidden_size,
            num_heads=num_heads,
            downsample_ratio=glda_downsample_ratio,
            dropout=dropout,
            fusion_method='gate'
        )

        # KAN classifier
        self.kan_classifier = KANClassifier(
            input_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

        # Additional layers
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SenTox-GLDA
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_features: Whether to return intermediate features
        Returns:
            Dictionary containing logits and optionally features
        """
        # Dual encoder outputs
        sentiment_outputs = self.sentiment_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        toxicity_outputs = self.toxicity_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Extract hidden states
        sentiment_features = sentiment_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        toxicity_features = toxicity_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Adaptive fusion
        fused_features = self.adaptive_fusion(sentiment_features, toxicity_features)

        # GLDA attention
        attended_features = self.glda_attention(fused_features, attention_mask)

        # Apply dropout
        attended_features = self.dropout(attended_features)

        # KAN classification
        logits = self.kan_classifier(attended_features)

        # Prepare output
        output = {'logits': logits}

        if return_features:
            output.update({
                'sentiment_features': sentiment_features,
                'toxicity_features': toxicity_features,
                'fused_features': fused_features,
                'attended_features': attended_features
            })

        return output

    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights for analysis"""
        with torch.no_grad():
            # Get features up to GLDA
            sentiment_outputs = self.sentiment_encoder(input_ids=input_ids, attention_mask=attention_mask)
            toxicity_outputs = self.toxicity_encoder(input_ids=input_ids, attention_mask=attention_mask)

            sentiment_features = sentiment_outputs.last_hidden_state
            toxicity_features = toxicity_outputs.last_hidden_state

            fused_features = self.adaptive_fusion(sentiment_features, toxicity_features)

            # Extract attention weights from GLDA (would need modification to GLDA class to return weights)
            attended_features = self.glda_attention(fused_features, attention_mask)

        return {
            'fused_features': fused_features,
            'attended_features': attended_features
        }


def main():
    """Test SenTox-GLDA model with virtual samples"""
    print("Testing SenTox-GLDA Model")
    print("=" * 50)

    # Model configuration
    config = {
        'sentiment_model_path': 'bert-base-chinese',  # Using base model for testing
        'toxicity_model_path': 'hfl/chinese-roberta-wwm-ext',  # Using base model for testing
        'num_classes': 2,
        'hidden_size': 768,
        'num_heads': 12,
        'glda_downsample_ratio': 4,
        'dropout': 0.1,
        'freeze_encoders': False
    }

    # Create model
    print("Initializing SenTox-GLDA model...")
    model = SenToxGLDA(**config)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create virtual test data
    batch_size = 2
    seq_len = 128

    # Random token IDs (simulating tokenized Chinese text)
    input_ids = torch.randint(100, 21000, (batch_size, seq_len))  # Chinese vocab range
    attention_mask = torch.ones(batch_size, seq_len)

    # Mask some tokens to simulate real scenarios
    attention_mask[0, 100:] = 0  # First sample has 100 tokens
    attention_mask[1, 110:] = 0  # Second sample has 110 tokens

    print(f"\nInput shapes:")
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()

    with torch.no_grad():
        # Basic forward pass
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']

        print(f"Output logits shape: {logits.shape}")
        print(f"Logits mean: {logits.mean().item():.6f}")
        print(f"Logits std: {logits.std().item():.6f}")

        # Test with return_features=True
        outputs_with_features = model(input_ids, attention_mask, return_features=True)

        print(f"\nIntermediate features shapes:")
        for key, tensor in outputs_with_features.items():
            if key != 'logits':
                print(f"{key}: {tensor.shape}")

    # Test gradient flow
    print(f"\nTesting gradient flow...")
    model.train()

    # Create dummy labels
    labels = torch.randint(0, config['num_classes'], (batch_size,))

    # Forward pass
    outputs = model(input_ids, attention_mask)
    logits = outputs['logits']

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'kan_classifier' in name or 'adaptive_fusion' in name or 'glda_attention' in name:
                print(f"{name}: grad_norm = {grad_norm:.6f}")

    print(f"Average gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")

    # Test different input lengths
    print(f"\nTesting different sequence lengths...")
    test_lengths = [64, 256, 512]

    for test_len in test_lengths:
        if test_len <= 512:  # Avoid memory issues
            test_input_ids = torch.randint(100, 21000, (1, test_len))
            test_attention_mask = torch.ones(1, test_len)

            try:
                with torch.no_grad():
                    test_outputs = model(test_input_ids, test_attention_mask)
                    print(f"Sequence length {test_len}: SUCCESS - Output shape {test_outputs['logits'].shape}")
            except Exception as e:
                print(f"Sequence length {test_len}: FAILED - {e}")

    # Test prediction probabilities
    print(f"\nTesting prediction probabilities...")
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids[:1], attention_mask[:1])  # Single sample
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)

        print(f"Sample prediction probabilities:")
        for i, prob in enumerate(probs[0]):
            print(f"  Class {i}: {prob.item():.4f}")

    # Performance timing
    print(f"\nPerformance timing...")
    import time

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids, attention_mask)

        # Timing
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(100):
            with torch.no_grad():
                _ = model(input_ids, attention_mask)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        print(f"Average inference time: {avg_time * 1000:.2f} ms (GPU)")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    else:
        # CPU timing
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids, attention_mask)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Average inference time: {avg_time * 1000:.2f} ms (CPU)")

    print(f"\nSenTox-GLDA model testing completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
