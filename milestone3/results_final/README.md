# Final results used in report/report.tex

- `rtx4060_laptop/` – full pipeline (check, tune, bench) with split-KV, RTX 4060 Laptop GPU.
- `rtx5000_ada/` – same pipeline on the RTX 5000 Ada lab workstation.
  `split_sweep.txt` is the data for Figure 1 (right).
- `tile_only_rtx4060/` – the earlier experiment with only the six tile sizes (no split-KV),
  10 warm-up / 50 timed runs. Its timings are noisy (see report, Lesson 1).

Commands used for the split-KV runs:

    python scripts/experiment.py check
    python scripts/experiment.py tune  --warmup 500 --iters 100
    python scripts/experiment.py bench --warmup 500 --iters 100

Split sweeps (Figure 1): `--mode fixed --config 8x16sS --warmup 1000 --iters 200` for S = 1..32.
