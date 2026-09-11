import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """
    Linear Attention using Kernel Feature Map: phi(x) = elu(x) + 1
    Changes multiplication order from (Q * K^T) * V to Q * (K^T * V).
    Time & Memory Complexity: O(N * d^2) - Linear in sequence length N!
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Standard linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kernel Feature Map: ensures strictly positive similarity scores.
        elu(x) + 1 is > 0 everywhere, smooth, and avoids negative weights.
        """
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, d_model)
        B, N, D = x.shape

        # Step 1: Project inputs
        Q = self.w_q(x)  # (B, N, D)
        K = self.w_k(x)  # (B, N, D)
        V = self.w_v(x)  # (B, N, D)

        # Step 2: Apply non-negative feature map phi()
        Q_prime = self.phi(Q)  # (B, N, D)
        K_prime = self.phi(K)  # (B, N, D)

        # Step 3: THE MAGIC - Contract K^T and V FIRST!
        # (B, D, N) @ (B, N, D) -> (B, D, D)
        # Notice the sequence dimension N is contracted here!
        KV_context = torch.matmul(K_prime.transpose(-2, -1), V)  # Shape: (B, D, D)

        # Step 4: Multiply Q_prime by the (D x D) context matrix
        # (B, N, D) @ (B, D, D) -> (B, N, D)
        numerator = torch.matmul(Q_prime, KV_context)  # Shape: (B, N, D)

        # Step 5: Normalization denominator: Q_prime * sum(K_prime)
        # Sum of all key features across the sequence: (B, 1, D)
        K_sum = K_prime.sum(dim=1, keepdim=True)
        # Dot product of each query with the total key sum: (B, N, 1)
        denominator = (Q_prime * K_sum).sum(dim=-1, keepdim=True) + self.eps

        # Final normalized output
        out = numerator / denominator  # Shape: (B, N, D)

        return out, KV_context

if __name__ == "__main__":
    print("=" * 65)
    print(" STEP 3: KERNEL LINEAR ATTENTION DEMONSTRATION")
    print("=" * 65)

    B, N, D = 2, 4096, 64  # Long sequence: N = 4096 tokens!
    x = torch.randn(B, N, D)

    layer = LinearAttention(d_model=D)
    out, kv_context = layer(x)

    print(f"Input shape:            {x.shape}")
    print(f"Intermediate Context:   {kv_context.shape}  <-- Stored as a compact (D x D) matrix!")
    print(f"Final Output shape:     {out.shape}")
    print("\nNote: At NO point did we ever create a (4096 x 4096) matrix in memory!")

