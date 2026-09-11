import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    """
    Pure from-scratch implementation of Scaled Dot-Product Attention.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Three distinct linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, d_model)
        B, N, D = x.shape

        # Step 1: Project inputs to Q, K, V
        Q = self.w_q(x)  # (B, N, D)
        K = self.w_k(x)  # (B, N, D)
        V = self.w_v(x)  # (B, N, D)

        # Step 2: Calculate raw similarity scores (Q * K^T)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (B, N, N)

        # Step 3: Scale by sqrt(d_k)
        scale = D ** 0.5
        scaled_scores = scores / scale

        # Step 4: Apply Softmax along the last dimension (rows)
        attn_weights = F.softmax(scaled_scores, dim=-1)  # Shape: (B, N, N)

        # Step 5: Weighted aggregation of Values
        out = torch.matmul(attn_weights, V)  # Shape: (B, N, D)

        return out, attn_weights

if __name__ == "__main__":
    # Quick sanity check
    B, N, D = 2, 4, 8  # 2 batches, sequence of 4 tokens, feature dimension 8
    x = torch.randn(B, N, D)

    layer = SingleHeadAttention(d_model=D)
    out, weights = layer(x)

    print("Input shape:       ", x.shape)
    print("Attention matrix:  ", weights.shape)
    print("Output shape:      ", out.shape)
    print("\nAttention weights for Token 0 in Batch 0 (sums to 1.0):")
    print(weights[0, 0].detach().numpy())
    print("Row sum:", weights[0, 0].sum().item())

