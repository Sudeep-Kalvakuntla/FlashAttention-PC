# Connection to your Milestone 2 code

`m2_original.cu` is an unchanged copy of your supplied `cuda_attention.cu`.
It is retained for reference and is not built by the M3 Makefile.

The M3 naive path preserves FP32 arithmetic, three separate kernels, 16x16
score/output blocks, 256-thread softmax blocks, one thread per softmax row,
in-place stable softmax, and the host-computed score scale. M3 adds batches,
heads, rectangular shapes and bottom-right causal masking.

**Your original M2 is noncausal.** To reproduce its attention semantics, use
B=H=1, M=N, causal=0. Example commands to run on the target laptop:

```sh
./attention --mode naive --b 1 --h 1 --m 512 --n 512 --d 64 --causal 0 --check
./attention --mode fixed --b 1 --h 1 --m 512 --n 512 --d 64 --causal 0 --config 4x32 --check
```

The tile above is illustrative, not a measured winner. Use the tuning workflow
for the best fixed configuration on the chosen workload.

| Item | Supplied M2 program | M3 harness |
|---|---|---|
| Inputs | Gaussian, separate seeds 1/2/3 | Uniform [-1,1], seed 42 |
| Warmup | Two N=64 runs for the whole program | Ten per shape/mode by default |
| Timing | One invocation per shape | Fifty per round, three rounds by default |
| Buffers | Allocated per call | Reused within each measured run |
| Build | Recommended -O2 | -O3, sm_89 for all modes |
| Correctness in source | No reference check in main | Double CPU reference |
| Attention | Square, one head, noncausal | Batched rectangular, optional causal |

The M2 report may describe correctness tests performed elsewhere; the table only
characterizes the supplied CUDA source. Both programs time GPU computation rather
than transfers. M3 also records host latency including selection and synchronization.

Re-run the adapted naive baseline alongside the fused modes. Do not compare old
M2 timings directly with new causal workload timings or claim an exact binary
reproduction. M3 gives all three modes identical inputs and build settings.

No code was compiled or executed during this update.
