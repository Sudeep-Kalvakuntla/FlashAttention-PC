"""Runs the Milestone 3 experiments. Run it from the milestone3 folder.

    python scripts/experiment.py check   # correctness tests
    python scripts/experiment.py tune    # time all configs on the training shapes
    python scripts/experiment.py bench   # compare naive / flash / fixed / dynamic on test shapes

Only the Python standard library is needed. Results go into the results/ folder.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

# Tile configurations: "<rows per block>x<keys per tile>".
TILES = ['4x16', '4x32', '4x64', '8x16', '8x32', '8x64']
# Split-KV configurations: "8x16s4" means 8x16 tiles with the keys split across 4 blocks.
SPLITS = ['8x16s2', '8x16s4', '8x16s8', '8x16s16', '8x16s32', '8x32s2', '8x32s4', '8x32s8']
CONFIGS = TILES + SPLITS

SHAPE_FIELDS = ['b', 'h', 'm', 'n', 'd', 'causal']
RESULTS = Path('results')


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def load_shapes(path):
    """Read a workload CSV into a list of dicts like {'b': 1, 'h': 4, ...}."""
    shapes = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            shapes.append({field: int(row[field]) for field in SHAPE_FIELDS})
    if not shapes:
        raise ValueError('Empty workload')
    keys = [shape_key(s) for s in shapes]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate workload shapes')
    return shapes


def shape_key(shape):
    """(b, h, m, n, d, causal) as a tuple, so shapes can be compared and used as dict keys."""
    return tuple(int(shape[field]) for field in SHAPE_FIELDS)


def next_power_of_two(x):
    """Same as bucket() in attention.cu, e.g. 400 -> 512."""
    p = 1
    while p < x:
        p *= 2
    return p


def bucket_key(shape):
    """Shapes with the same bucket key share one entry in the selector table."""
    return (int(shape['b']), int(shape['h']),
            next_power_of_two(int(shape['m'])), next_power_of_two(int(shape['n'])),
            int(shape['d']), int(shape['causal']))


def config_name(row):
    """Turn rows/keys/splits columns back into a name like '8x16' or '8x16s4'."""
    name = f"{row['rows']}x{row['keys']}"
    if int(row['splits']) > 1:
        name += f"s{row['splits']}"
    return name


def split_config_name(name):
    """'8x16s4' -> (8, 16, 4) and '8x16' -> (8, 16, 1)."""
    splits = 1
    if 's' in name:
        name, splits = name.split('s')
    rows, keys = name.split('x')
    return int(rows), int(keys), int(splits)


def geomean(values):
    """Geometric mean. Better than the normal mean for averaging ratios like speedups."""
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_attention(args, shape, mode, config='4x32', check=False, scale=1, seed=42):
    """Run the attention program once and return its CSV output row as a dict.

    Everything the program prints is also appended to results/run.log.
    """
    command = [str(Path(args.exe).resolve()),
               '--mode', mode,
               '--config', config,
               '--warmup', str(args.warmup),
               '--iters', str(args.iters),
               '--input-scale', str(scale),
               '--seed', str(seed)]
    for field, value in shape.items():
        command += ['--' + field, str(value)]
    if mode == 'dynamic':
        command += ['--table', str(RESULTS / 'selector.csv')]
    if check:
        command += ['--check']

    result = subprocess.run(command, capture_output=True, text=True)
    with open(RESULTS / 'run.log', 'a') as log:
        log.write(' '.join(command) + '\n' + result.stderr + result.stdout + '\n')
    if result.returncode != 0:
        raise RuntimeError(result.stderr + '\nCommand: ' + ' '.join(command))

    # stdout is a header line plus one data line.
    return next(csv.DictReader(io.StringIO(result.stdout)))


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def correctness(args):
    # Delete the old file first, so a failed run can't leave an old "pass" behind.
    (RESULTS / 'correctness.csv').unlink(missing_ok=True)

    # Small shapes, chosen to hit edge cases: sizes that are not multiples of the
    # tile size, several heads/batches, one query, M > N (non-causal), and so on.
    cases = [
        dict(b=1, h=1, m=1, n=1, d=32, causal=1),
        dict(b=1, h=2, m=7, n=19, d=64, causal=1),
        dict(b=2, h=2, m=33, n=65, d=128, causal=1),
        dict(b=1, h=1, m=65, n=65, d=64, causal=1),
        dict(b=1, h=2, m=17, n=63, d=32, causal=0),
        dict(b=1, h=1, m=1, n=257, d=128, causal=1),
        dict(b=1, h=1, m=19, n=7, d=64, causal=0),
    ]
    args.warmup = 0
    args.iters = 1
    rows = []

    # Input scale 0 makes all scores equal, 1 is normal, 8 gives very large scores.
    for case in cases:
        for scale in [0, 1, 8]:
            row = run_attention(args, case, 'naive', check=True, scale=scale)
            row.update(input_scale=scale, status='PASS')
            rows.append(row)
            for config in CONFIGS:
                row = run_attention(args, case, 'fixed', config, check=True, scale=scale)
                row.update(input_scale=scale, status='PASS')
                rows.append(row)

    # Test dynamic mode with a tiny made-up table: one case finds its bucket in
    # the table, the others fall back. The real selector.csv is restored afterwards.
    table_path = RESULTS / 'selector.csv'
    saved_table = table_path.read_bytes() if table_path.exists() else None
    try:
        table_path.write_text('b,h,m_bucket,n_bucket,d,causal,rows,keys,splits\n'
                              '1,2,8,32,64,1,8,16,4\n'
                              '1,1,1,512,128,1,8,32,8\n')
        for case in cases:
            row = run_attention(args, case, 'dynamic', check=True)
            row.update(input_scale=1, status='PASS')
            rows.append(row)
    finally:
        if saved_table is None:
            table_path.unlink(missing_ok=True)
        else:
            table_path.write_bytes(saved_table)

    # run_attention raises an error on any failure, so reaching here means all passed.
    write_csv(RESULTS / 'correctness.csv', rows)
    print(f'{len(rows)} correctness cases passed. See results/run.log for errors.')


# ---------------------------------------------------------------------------
# 2. Tuning
# ---------------------------------------------------------------------------

def tune(args):
    train = load_shapes(args.train)

    # Time every config on every training shape. Each round runs the jobs in a
    # different random order, so slow-downs over time don't always hit the same config.
    rng = random.Random(2026)
    rows = []
    for round_number in range(args.rounds):
        jobs = []
        for shape in train:
            for config in CONFIGS:
                jobs.append((shape, config))
        rng.shuffle(jobs)

        for i, (shape, config) in enumerate(jobs):
            print(f'Tuning round {round_number+1}/{args.rounds}: {i+1}/{len(jobs)}', flush=True)
            row = run_attention(args, shape, 'fixed', config)
            row['round'] = round_number
            rows.append(row)
        write_csv(RESULTS / 'tuning.csv', rows)

    # time[(shape, config)] = median GPU time over the rounds.
    time = {}
    for shape in train:
        for config in CONFIGS:
            samples = [float(r['gpu_median_us']) for r in rows
                       if shape_key(r) == shape_key(shape) and config_name(r) == config]
            time[shape_key(shape), config] = statistics.median(samples)

    # Fastest time for each shape over all configs.
    best_time = {}
    for shape in train:
        best_time[shape_key(shape)] = min(time[shape_key(shape), c] for c in CONFIGS)

    def score(config, shapes):
        """How much slower `config` is than the best config, averaged (geomean) over `shapes`.
        1.0 means it was the fastest on every shape; 1.2 means 20% slower on average."""
        return geomean(time[shape_key(s), config] / best_time[shape_key(s)] for s in shapes)

    # The single best config for all shapes, and the best one without split-KV
    # (that one is our "plain FlashAttention" baseline).
    fixed = min(CONFIGS, key=lambda c: score(c, train))
    unsplit = min(TILES, key=lambda c: score(c, train))

    # One selector table entry per bucket.
    entries = []
    for key in sorted({bucket_key(s) for s in train}):
        shapes_in_bucket = [s for s in train if bucket_key(s) == key]
        best = min(CONFIGS, key=lambda c: score(c, shapes_in_bucket))
        # Small differences are probably just noise, so only switch away from the
        # fixed config if the other one is at least 3% faster.
        if score(best, shapes_in_bucket) > 0.97 * score(fixed, shapes_in_bucket):
            best = fixed
        rows_per_block, keys_per_tile, splits = split_config_name(best)
        entry = dict(zip(['b', 'h', 'm_bucket', 'n_bucket', 'd', 'causal'], key))
        entry.update(rows=rows_per_block, keys=keys_per_tile, splits=splits)
        entries.append(entry)

    write_csv(RESULTS / 'selector.csv', entries)
    with open(RESULTS / 'tuning.json', 'w') as f:
        json.dump({'fixed_config': fixed,
                   'unsplit_config': unsplit,
                   'training_shapes': train,
                   'objective': 'geometric mean GPU latency normalized by per-shape best',
                   'rounds': args.rounds,
                   'iters': args.iters,
                   'configs': CONFIGS}, f, indent=2)

    print('Best overall fixed config:', fixed)
    print('Best fixed config without split-KV:', unsplit)
    print('Selected configurations:', sorted({config_name(e) for e in entries}))
    print('If all buckets choose one config, there is no evidence for dynamic selection yet.')


# ---------------------------------------------------------------------------
# 3. Benchmark on held-out test shapes
# ---------------------------------------------------------------------------

def benchmark(args):
    with open(RESULTS / 'tuning.json') as f:
        tuning = json.load(f)
    test = load_shapes(args.test)

    train_keys = {shape_key(s) for s in tuning['training_shapes']}
    if any(shape_key(s) in train_keys for s in test):
        raise ValueError('Test shapes overlap training shapes. Keep exact shapes held out.')

    fixed = tuning['fixed_config']
    # Older tuning.json files have no 'unsplit_config'; use the fixed config then.
    unsplit = tuning.get('unsplit_config', fixed)

    # The four things we compare: name -> (program mode, config).
    variants = {
        'naive': ('naive', fixed),      # Milestone 2 kernels (config is ignored)
        'flash': ('fixed', unsplit),    # plain FlashAttention, best config without splits
        'fixed': ('fixed', fixed),      # best single config, may use split-KV
        'dynamic': ('dynamic', fixed),  # selector table, falls back to `fixed`
    }

    rng = random.Random(41)
    rows = []
    for round_number in range(args.rounds):
        jobs = []
        for shape in test:
            for variant in variants:
                jobs.append((shape, variant))
        rng.shuffle(jobs)

        for i, (shape, variant) in enumerate(jobs):
            print(f'Benchmark round {round_number+1}/{args.rounds}: {i+1}/{len(jobs)}', flush=True)
            mode, config = variants[variant]
            row = run_attention(args, shape, mode, config)
            row['variant'] = variant
            row['config'] = 'naive' if variant == 'naive' else config_name(row)
            row['round'] = round_number
            rows.append(row)
        write_csv(RESULTS / 'benchmark.csv', rows)

    # One summary line per test shape: median over rounds for each variant.
    summary = []
    for shape in test:
        result = dict(shape)
        for variant in variants:
            matches = [r for r in rows
                       if shape_key(r) == shape_key(shape) and r['variant'] == variant]
            result[variant + '_config'] = matches[0]['config']
            for metric in ['gpu_median_us', 'host_median_us', 'selector_median_us']:
                result[variant + '_' + metric] = statistics.median(float(r[metric]) for r in matches)

        # Speedup > 1 means the second variant is faster.
        result['dynamic_gpu_speedup'] = result['fixed_gpu_median_us'] / result['dynamic_gpu_median_us']
        result['dynamic_host_speedup'] = result['fixed_host_median_us'] / result['dynamic_host_median_us']
        result['flash_vs_naive_speedup'] = result['naive_gpu_median_us'] / result['flash_gpu_median_us']
        result['dynamic_vs_flash_speedup'] = result['flash_gpu_median_us'] / result['dynamic_gpu_median_us']
        summary.append(result)
    write_csv(RESULTS / 'summary.csv', summary)

    print('Plain FlashAttention vs naive geometric mean GPU speedup:',
          geomean(r['flash_vs_naive_speedup'] for r in summary))
    print('Dynamic (with split-KV) vs plain FlashAttention geometric mean GPU speedup:',
          geomean(r['dynamic_vs_flash_speedup'] for r in summary))
    print('Dynamic vs best fixed geometric mean GPU speedup:',
          geomean(r['dynamic_gpu_speedup'] for r in summary))
    print('Dynamic vs best fixed geometric mean host speedup:',
          geomean(r['dynamic_host_speedup'] for r in summary))
    regressions = sum(1 for r in summary if r['dynamic_gpu_speedup'] < 1 / 1.03)
    print('GPU regressions vs best fixed > 3%:', regressions)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['check', 'tune', 'bench'])
    parser.add_argument('--exe', default='./attention', help='path to the compiled program')
    parser.add_argument('--train', default='workloads/train.csv')
    parser.add_argument('--test', default='workloads/test.csv')
    parser.add_argument('--warmup', type=int, default=10, help='untimed runs before timing')
    parser.add_argument('--iters', type=int, default=50, help='timed runs per measurement')
    parser.add_argument('--rounds', type=int, default=3, help='repeat everything this many times')
    args = parser.parse_args()
    if args.rounds < 1 or args.iters < 1 or args.warmup < 0:
        parser.error('rounds and iters must be positive; warmup must be nonnegative')

    RESULTS.mkdir(exist_ok=True)
    if args.action == 'check':
        correctness(args)
    elif args.action == 'tune':
        tune(args)
    else:
        benchmark(args)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        sys.exit(str(error))
