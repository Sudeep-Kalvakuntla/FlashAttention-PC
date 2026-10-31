# Laptop run checklist

- [ ] Copy the whole milestone3 folder; keep its directory structure.
- [ ] Record GPU name, driver, CUDA Toolkit, OS, and laptop power mode.
- [ ] Build using README commands.
- [ ] Run correctness; inspect failures before tuning.
- [ ] Optionally run compute-sanitizer.
- [ ] Run tuning; keep tuning.csv, tuning.json and selector.csv together.
- [ ] Run held-out benchmarks without editing the table afterward.
- [ ] Check both GPU latency and host latency; include regressions.
- [ ] Optionally run the library baselines; distinguish FP16 from FP32.
- [ ] Send back the results folder plus hardware/software details and any errors.

Do not submit fabricated timings or claim the implementation has passed correctness
until the laptop run actually passes. No IEEE paper draft is included.
