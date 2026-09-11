import torch
import torch.nn.functional as F

def verify_associativity_failure():
    print("=" * 65)
    print(" 1. EMPIRICAL PROOF: softmax(QK^T)V != Q(K^TV)")
    print("=" * 65)

    torch.manual_seed(42)
    N, D = 4, 3  # 4 tokens, feature dimension 3

    Q = torch.randn(N, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    # 1. Standard Attention: softmax(QK^T) * V
    scores = torch.matmul(Q, K.T) / (D ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    standard_result = torch.matmul(attn_weights, V)

    # 2. What happens if someone tries to do Q * (K^T * V) directly?
    naive_reordered = torch.matmul(Q, torch.matmul(K.T, V))

    print("\n[Standard Attention: softmax(QK^T) * V]:")
    print(standard_result.round(decimals=3))

    print("\n[Naive Reordered: Q * (K^T * V)]:")
    print(naive_reordered.round(decimals=3))

    difference = (standard_result - naive_reordered).abs().max().item()
    print(f"\nMaximum absolute discrepancy: {difference:.4f}")
    print("CONCLUSION: You CANNOT simply change multiplication order with Softmax!")

def print_complexity_table():
    print("\n" + "=" * 65)
    print(" 2. FLOPs & MEMORY COMPARISON: (QK^T)V vs Q(K^TV)")
    print("=" * 65)
    print(f"{'Seq Length (N)':<16} | {'(QK^T)V FLOPs (O(N^2*d))':<24} | {'Q(K^TV) FLOPs (O(N*d^2))':<24}")
    print("-" * 68)

    d = 64  # Head dimension
    for N in [512, 2048, 8192, 16384, 32768]:
        # (QK^T)*V requires 2 * N^2 * d + 2 * N^2 * d = 4 * N^2 * d FLOPs
        flops_standard = 4 * (N ** 2) * d
        # Q*(K^T*V) requires 2 * N * d^2 + 2 * N * d^2 = 4 * N * d^2 FLOPs
        flops_linear = 4 * N * (d ** 2)

        ratio = flops_standard / flops_linear
        print(f"N = {N:<12} | {flops_standard / 1e9:14.3f} GFLOPs    | {flops_linear / 1e9:14.3f} GFLOPs    | {ratio:6.1f}x less!")

if __name__ == "__main__":
    verify_associativity_failure()
    print_complexity_table()

