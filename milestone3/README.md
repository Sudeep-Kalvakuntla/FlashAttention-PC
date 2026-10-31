# Milestone 3: workload-adaptive attention on RTX 4060

**Status: compiled and correctness-tested (322 cases) on an RTX 4060 Laptop GPU and an
RTX 5000 Ada.** Re-run the correctness suite after any change before using timings.

Question: does selecting a tile configuration for each attention shape reduce latency
compared with the best overall fixed configuration on the same GPU?

## What is included

- `report/report.tex` (and `report.pdf`): the Milestone 3 report.
- `results_final/`: the results used in the report, for both GPUs.

- `src/attention.cu`: sequential CPU reference, adapted M2 three-kernel CUDA
  baseline, tiled fused attention, lookup-table selector, timing and correctness checks.
- `scripts/experiment.py`: correctness suite, offline tuning, held-out benchmarking,
  and CSV summaries. Only Python's standard library is required.
- `scripts/library_reference.py`: optional PyTorch / official FlashAttention comparison.
- `workloads/train.csv` and `test.csv`: synthetic chatbot-style chunked-prefill shapes.
- `Makefile`: native CUDA build for compute capability 8.9.
- `reference/m2_original.cu`: your supplied M2 source, preserved unchanged.
- `reference/M2_COMPARISON.md`: implementation and measurement differences.

The baseline has been checked against your supplied M2 source and follows its design:
16x16 score/output blocks, one thread per softmax row, one materialized score buffer.
It adds batches, heads, rectangular shapes and causal masking. Rebenchmark it on the
same laptop; do not directly merge these measurements with old report timings.

## Scope and implementation

Inputs are contiguous FP32 arrays: Q `[B,H,M,D]`, K/V `[B,H,N,D]`.
Output is `[B,H,M,D]`. D can be 32, 64, or 128. All heads have the same lengths.
M is incoming query chunk length; N includes both cached history and the new chunk.
Causal mode requires M <= N and permits key j when `j <= N-M+i` for query i.
This is bottom-right aligned causal attention, including ordinary causal attention
when M=N. Noncausal mode also allows M>N.

Each warp computes one query row. A block has 4 or 8 warps and shares 16, 32,
or 64 K/V rows. There are six compiled tile configurations, written as `rows x keys`.
Q and output accumulators live in registers, K/V tiles in shared memory.
Warp shuffles reduce dot products and softmax statistics. Online softmax rescales
previous accumulators when a later tile has a larger maximum. The fused path never
allocates a full M-by-N score/probability matrix.

This is a small educational implementation of FlashAttention's tiled, fused,
online-softmax algorithm. It does not reproduce the performance engineering of
FlashAttention-2. It has no Tensor Cores, backward pass, dropout, paged KV cache,
GQA, ragged batches, or actual chatbot server. FP32 is intentional to match M2.

### Split-KV (FlashDecoding-style)

Plain FlashAttention gives each block all N keys, so the grid has only
`ceil(M/rows) * B * H` blocks. For decoding (M=1) or short chunks that is a handful
of blocks on a GPU with dozens of SMs, and the kernel is limited by one block walking
through every key. Split-KV adds a third grid dimension: the keys are divided into
S contiguous ranges, each handled by its own block. Each block writes its unnormalized
output and its softmax max/sum; a small `merge` kernel combines the S partials with
the same rescaling rule used between tiles:

    m = max_i m_i,   l = sum_i l_i exp(m_i - m),   O = sum_i o_i exp(m_i - m) / l

Configurations with splits are written `rows x keys s splits`, e.g. `8x16s4`;
S can be 1, 2, 4, 8, 16, or 32. S=1 is plain FlashAttention and needs no merge.
The scratch buffer is `S*B*H*M*(D+2)` floats, far smaller than naive's M*N scores.
Splitting helps when the grid is small and hurts once the grid already fills the GPU
(extra merge pass and partial writes), so the best S depends on shape and GPU.

The selector rounds M/N up to powers of two and looks up
`(B,H,M_bucket,N_bucket,D,causal)`. The table is loaded once. Each attention call
selects a compiled kernel (tile and split count); nothing changes inside it. Missing buckets
fall back to the best fixed configuration. A linear table scan is deliberately simple.
No future shape prediction, learning model, or online benchmarking is involved.

## Build on your friend's RTX 4060

Install the NVIDIA driver, CUDA Toolkit with nvcc (CUDA 12.x is a reasonable starting
point), Python 3.9+, and a compatible C++ compiler. The ordinary implementation does
not require PyTorch, NumPy, cuBLAS, or the flash-attn package.
Copy this whole folder to the laptop and open a terminal inside it.

Linux / WSL2 with working NVIDIA CUDA support:

```sh
nvidia-smi
nvcc --version
make
```

Windows: use an **x64 Native Tools Command Prompt for Visual Studio** with CUDA on PATH:

```bat
nvidia-smi
nvcc --version
nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo src/attention.cu -o attention.exe
```

For Windows, append `--exe ./attention.exe` to the Python commands below.
If Python is named `python3` on your system, use that instead of `python`.
The executable uses GPU device 0. Set CUDA_VISIBLE_DEVICES if needed.

## Run in this order

### 1. Correctness

```sh
python scripts/experiment.py check
```

This checks every tile configuration and the naive baseline against sequential
CPU double-precision attention. Cases include nonmultiples of tile sizes, batch/head
indexing, square/rectangular and causal/noncausal attention, one-token queries,
uniform logits and large logits. Dynamic dispatch has both table-hit and fallback
checks. Any failure stops the script with a nonzero exit code. The script temporarily
writes a tiny selector table and restores any existing one afterward.

Tolerance per output: `abs(error) <= 2e-4 + 2e-4*abs(reference)`; nonfinite outputs fail.
Read `results/run.log` for max absolute errors. `correctness.csv` is only produced
on successful completion. A partial suite must not be reported as passing.

Optional CUDA memory/synchronization checks on the target machine:

```sh
compute-sanitizer --tool memcheck ./attention --m 33 --n 65 --d 128 --config 8x64 --iters 1 --warmup 0 --check
compute-sanitizer --tool racecheck ./attention --m 33 --n 65 --d 128 --config 8x64 --iters 1 --warmup 0 --check
```

### 2. Offline tuning

```sh
python scripts/experiment.py tune
```

Each configuration gets identical inputs and warmup/measurement counts. Three rounds
randomize shape/configuration order to reduce order bias. The tuner takes the median
GPU latency across rounds. It chooses the global fixed configuration minimizing
geometric mean normalized latency over training shapes, then applies the same objective
inside each bucket. A bucket must be at least 3% faster than fixed to choose another
configuration. This margin is a simple noise guard, not a statistical significance test.

Outputs: `tuning.csv`, `tuning.json` (including fixed configuration), `selector.csv`.
No prefilled winning configurations are supplied: only the target GPU can determine them.
The training set is small and mostly has one shape per bucket. Expand it to multiple
shapes per bucket for a stronger study; keep the final test set untouched.

### 3. Held-out benchmark

```sh
python scripts/experiment.py bench
```

This loads the saved global fixed configuration and selector table. It compares
naive / flash (best configuration without splits) / fixed (best overall, possibly split) /
dynamic on identical shapes and inputs, randomizing order across
three rounds. Test shapes are rejected if they exactly overlap tuning shapes.
Most tests fall into trained buckets; the B=2, N=3072 shape exercises an unseen bucket (fallback).
The small-M shapes (M=1..16) at the end of both files target decoding, where split-KV matters.

Outputs: `benchmark.csv` (raw per-round metrics), `summary.csv` (per-shape comparisons).
Speedup > 1 means faster. The script prints geometric mean GPU and host speedups,
and the number of GPU regressions exceeding 3%. Do not tune against these test results;
if you use them for development, create a fresh final test set.

For longer runs use `--iters 200 --rounds 5` for both tuning and benchmarking.
Stop other GPU workloads, plug the laptop into power, keep the same power mode and
cooling conditions, and record the exact GPU name/VRAM/driver/CUDA version. Laptop and
desktop RTX 4060 power limits differ, so report which device was used.

### 4. Optional established-library reference

Install a CUDA-compatible PyTorch separately. For official FlashAttention, follow
its repository installation instructions (Linux/WSL is generally simpler than Windows).
Then run:

```sh
python scripts/library_reference.py --backend math
python scripts/library_reference.py --backend flash-attn
```

The math baseline is PyTorch FP32 SDPA with an explicit mask. The official FlashAttention
baseline uses FP16 and its bottom-right causal behavior. Both independently check outputs
against double-precision PyTorch math attention. Neither is used to train our selector.
Keep FP16 official-FlashAttention results in a separate table: they combine precision,
Tensor Core use and library engineering. They are context, not an isolated selector
comparison. The optional script has a Python timing harness and different random inputs.
Report this limitation, especially for short kernels. It is not expected that our FP32
educational kernel will outperform the official implementation.

## Understanding the timing columns

- `gpu_median_us`, `gpu_p95_us`: CUDA event interval around one invocation. The naive
  interval contains all three kernels and intervening launch gaps. It excludes allocation,
  transfers, table loading, correctness and selector lookup. Event timing can include GPU
  idle time caused by host launch gaps; it is not summed profiler kernel duration.
- `selector_median_us`: host clock interval around choosing a configuration. Fixed/naive
  also measure the trivial assignment, which exposes the clock measurement floor.
  Submicrosecond values are noisy; do not interpret them as precise isolated costs.
- `host_median_us`, `host_p95_us`: wall time from before selection through event recording,
  launch, and event synchronization. Includes selector and instrumentation overhead.
  Compare this too: faster kernels may lose their advantage after dispatch overhead.
- `scratch_bytes`: extra global score-buffer allocation; zero for fused modes. This is
  not total GPU memory consumption or measured peak VRAM.

The script runs a new process per shape/mode and reuses buffers inside each run. These
are isolated, warm-buffer microbenchmarks, not an interleaved serving trace. They do not
measure tokenizer/model execution, cache appends, request queues, network latency or
chatbot tokens/second. A table selection occurs every measured dynamic invocation, but
there is no within-process sequence of changing requests in this version.

## What to report for M3

Compare adapted M2 -> fixed fused -> dynamic fused. Show per-shape latency and
regressions as well as aggregate speedup. Include B/H/M/N/D, causal mode, FP32 precision,
chosen tiles, repetitions, tolerances, hardware/software versions, and tuning cost
(number of trials and wall time, measured externally).

For scaling, use the included query-length and batch sweeps; add N=512/1024/2048/4096
at fixed B/H/M/D if needed. Do not label sequence-length scaling as multi-GPU scaling.
The fixed and dynamic variants use identical arithmetic and candidate kernels.
If all workloads prefer the same tile, or gains disappear in host timing, report that
result honestly. A selector is not automatically an improvement.

A successful result establishes a benefit over our own fixed FP32 kernel for the tested
workload on this GPU. It does not establish a new FlashAttention algorithm or superiority
over existing autotuners. Instructor acceptance of this incremental contribution remains
a separate question.

## References and prior art

- Dao et al., FlashAttention (2022): https://arxiv.org/abs/2205.14135
- Dao, FlashAttention-2 (2023): https://arxiv.org/abs/2307.08691
- Official implementation: https://github.com/Dao-AILab/flash-attention
- Triton fused attention tutorial (existing tile/warp autotuning):
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- NVIDIA Ada tuning guide: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- vLLM chunked-prefill motivation:
  https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill

These techniques are established. Our project investigates a small, measured
configuration-selection policy for our kernel and workload on the available GPU.
