import pandas as pd
import numpy as np
from collections import Counter
import sys, os

def parse_trace(filepath, warmup=100_000, use=200_000, min_freq=10):
    pcs, addrs = [], []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i < warmup: continue
            if i >= warmup + use: break
            parts = line.strip().split()
            if len(parts) != 2: continue
            pcs.append(int(parts[0], 16))
            addrs.append(int(parts[1], 16))

    pcs = np.array(pcs)
    addrs = np.array(addrs)
    deltas = np.diff(addrs)
    pcs = pcs[1:]

    freq = Counter(deltas)
    valid = {d for d, c in freq.items() if c >= min_freq}
    mask = np.array([d in valid for d in deltas])
    deltas = deltas[mask]
    pcs = pcs[mask]

    vocab = sorted(valid)
    d2i = {d: i for i, d in enumerate(vocab)}
    delta_ids = np.array([d2i[d] for d in deltas])

    df = pd.DataFrame({'pc': pcs, 'delta_id': delta_ids, 'raw_delta': deltas})
    return df, vocab

if __name__ == "__main__":
    trace = sys.argv[1]
    name = os.path.basename(trace).replace('.out','')
    df, vocab = parse_trace(trace)
    os.makedirs('data', exist_ok=True)
    df.to_csv(f"data/{name}.csv", index=False)
    print(f"{name}: {len(df)} samples, vocab size {len(vocab)}")
    print(df.head())
    print("delta_id range:", df.delta_id.min(), "to", df.delta_id.max())
