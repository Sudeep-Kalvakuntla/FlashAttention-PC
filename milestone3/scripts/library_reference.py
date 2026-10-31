"""Optional library comparison. Requires PyTorch; flash-attn backend also needs flash-attn.
Different precision from our CUDA code: report these rows separately.
"""
import argparse
import csv
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend', choices=['math', 'flash-attn'], default='math')
p.add_argument('--workload', default='workloads/test.csv')
p.add_argument('--iters', type=int, default=50)
a = p.parse_args()
if a.iters < 1:
    p.error('iters must be positive')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required')
if a.backend == 'flash-attn':
    from flash_attn import flash_attn_func

torch.manual_seed(42)
results = []
with open(a.workload, newline='') as f:
    workload = list(csv.DictReader(f))
for row in workload:
    b, h, m, n, d, causal = [int(row[k]) for k in ['b', 'h', 'm', 'n', 'd', 'causal']]
    dtype = torch.float16 if a.backend == 'flash-attn' else torch.float32
    # BHMD for PyTorch, BMHD for official FlashAttention.
    q = torch.rand(b, h, m, d, device='cuda', dtype=dtype) * 2 - 1
    k = torch.rand(b, h, n, d, device='cuda', dtype=dtype) * 2 - 1
    v = torch.rand_like(k) * 2 - 1
    mask = None
    if causal:
        mask = torch.arange(n, device='cuda')[None, :] <= n-m+torch.arange(m, device='cuda')[:, None]
    # Explicit bottom-right mask. SDPA is_causal=True has different rectangular semantics.
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=mask)
    if a.backend == 'flash-attn':
        qf, kf, vf = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        def run():
            return flash_attn_func(qf, kf, vf, causal=bool(causal)).transpose(1, 2)
    else:
        def run():
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    with torch.inference_mode():
        output = run()
        torch.testing.assert_close(output.double(), reference, atol=2e-3 if dtype == torch.float16 else 2e-4,
                                   rtol=2e-3 if dtype == torch.float16 else 2e-4)
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(a.iters):
            start.record()
            run()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end)*1000)
    results.append(dict(row, backend=a.backend, dtype=str(dtype), gpu_median_us=statistics.median(times)))
Path('results').mkdir(exist_ok=True)
with open(f'results/library_{a.backend}.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
print('Saved library results. These use different random inputs and a Python launch harness.')
