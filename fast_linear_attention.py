import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLinearAttention(nn.Module):
    """
    Causal Linear Attention with positive feature map phi(x) = elu(x) + 1.
    - Training mode: O(N) parallel prefix sum via torch.cumsum.
    - Inference mode: O(1) constant-memory recurrent state update.
    """
    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # Non-negative kernel representation ensuring strictly positive scores
        return F.elu(x) + 1.0

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel execution during training: O(N * d^2) instead of O(N^2 * d)
        Shape: (B, N, D)
        """
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # Project and reshape: (B, H, N, D)
        q = self.feature_map(self.q_proj(x).view(B, N, H, D).transpose(1, 2))
        k = self.feature_map(self.k_proj(x).view(B, N, H, D).transpose(1, 2))
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # 1. Compute outer products per token: K^T * V -> (B, H, N, D, D)
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)

        # 2. Causal accumulation (Prefix Sum over sequence dimension N)
        kv_state = torch.cumsum(kv, dim=2)      # (B, H, N, D, D)
        k_state = torch.cumsum(k, dim=2)        # (B, H, N, D) for normalization

        # 3. Multiply Q by the accumulated state: Q * S_t -> (B, H, N, D)
        num = torch.einsum('bhnd,bhnde->bhne', q, kv_state)
        denom = torch.einsum('bhnd,bhnd->bhn', q, k_state).unsqueeze(-1) + self.eps

        out = num / denom
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)

    def step_recurrent(self, x_t: torch.Tensor, prev_state=None):
        """
        Streaming/On-Robot inference: O(1) constant time & memory per control step!
        x_t: (B, 1, D)
        prev_state: tuple (kv_state, k_state) where kv_state is (B, H, D, D)
        """
        B, _, _ = x_t.shape
        H, D = self.n_heads, self.head_dim

        q_t = self.feature_map(self.q_proj(x_t).view(B, 1, H, D).transpose(1, 2))
        k_t = self.feature_map(self.k_proj(x_t).view(B, 1, H, D).transpose(1, 2))
        v_t = self.v_proj(x_t).view(B, 1, H, D).transpose(1, 2)

        if prev_state is None:
            kv_state = torch.zeros(B, H, D, D, device=x_t.device, dtype=x_t.dtype)
            k_state = torch.zeros(B, H, 1, D, device=x_t.device, dtype=x_t.dtype)
        else:
            kv_state, k_state = prev_state

        # Update O(d^2) state in-place with the current token (No growing KV-cache!)
        kv_state = kv_state + torch.einsum('bh1d,bh1e->bhde', k_t, v_t)
        k_state = k_state + k_t

        num = torch.einsum('bh1d,bhde->bh1e', q_t, kv_state)
        denom = torch.einsum('bh1d,bh1d->bh1', q_t, k_state).unsqueeze(-1) + self.eps

        out = num / denom
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return self.out_proj(out), (kv_state, k_state)