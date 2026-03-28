import pandas as pd, sys, math

benchmarks = [
    ('matmul',    'data/matmul.csv'),
    ('sort',      'data/sort.csv'),
    ('bfs',       'data/bfs.csv'),
    ('hashtable', 'data/hashtable.csv'),
]

H = 64
BITS = 16
H0 = 64 * 8 * 2

print(f"{'Benchmark':<12} {'Vocab':>7} {'Vanilla params':>15} {'Binary params':>14} {'n/logn':>8} {'Total compression':>18}")
print("-" * 80)

for name, csv in benchmarks:
    df = pd.read_csv(csv)
    n = df['delta_id'].nunique()
    vanilla = H0 + H * n
    binary  = H0 + H * BITS
    ratio   = vanilla / binary
    theory  = n / math.log2(n) if n > 1 else 1
    print(f"{name:<12} {n:>7} {vanilla:>15,} {binary:>14,} {theory:>8.1f} {ratio:>18.1f}x")
