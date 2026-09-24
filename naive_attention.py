"""
Computes standard scaled dot-product attention:
    S = (Q @ K^T) / sqrt(d)
    P = softmax(S)
    O = P @ V
"""

import math
import random
import time


def random_matrix(rows, cols, seed=42):
    """Generate a 2D list of floats with Gaussian random values."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def softmax(S):
    """Numerically stable row-wise softmax."""
    P = []
    for row in S:
        m = max(row)
        exp_row = [math.exp(x - m) for x in row]
        row_sum = sum(exp_row)
        P.append([x / row_sum for x in exp_row])
    return P


def naive_attention(Q, K, V):
    """
    Sequential naive attention without external libraries or GPU.
    Q: N x d
    K: N x d
    V: N x d
    Returns: O (N x d)
    """
    N = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    # 1. S = (Q @ K^T) * scale
    S = []
    for i in range(N):
        row = []
        for j in range(N):
            dot = sum(Q[i][k] * K[j][k] for k in range(d))
            row.append(dot * scale)
        S.append(row)

    # 2. P = softmax(S)
    P = softmax(S)

    # 3. O = P @ V
    O = []
    for i in range(N):
        row = []
        for j in range(d):
            val = sum(P[i][k] * V[k][j] for k in range(N))
            row.append(val)
        O.append(row)

    return O


if __name__ == "__main__":
    # Test different sequence lengths (N) and head dimensions (d)
    sequence_lengths = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 20480, 24576]
    head_dimensions = [64]


    print(f"{'Seq Len (N)':<14} | {'Head Dim (d)':<14} | {'Output Shape':<14} | {'Time (ms)':<10}")
    print("-" * 65)

    for d in head_dimensions:
        for N in sequence_lengths:
            # Generate random Q, K, V matrices
            Q = random_matrix(N, d, seed=1)
            K = random_matrix(N, d, seed=2)
            V = random_matrix(N, d, seed=3)

            # Time the attention computation
            start_time = time.perf_counter()
            O = naive_attention(Q, K, V)
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000.0

            out_shape_str = f"{len(O)} x {len(O[0])}"
            print(f"{N:<14} | {d:<14} | {out_shape_str:<14} | {elapsed_ms:<10.2f}")

