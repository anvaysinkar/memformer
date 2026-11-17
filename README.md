# MemFormer: a compact Transformer memory-access predictor

MemFormer predicts the next memory-address delta from the last 8 deltas and the program counters (PCs) that issued them. The output is a 16-bit binary code for the delta's class id instead of an n-way softmax, so the output layer stays the same size however many distinct deltas a workload has. A PC-aware Transformer with a binary output is a step towards ML prefetchers small enough to be practical.

Team: Viyan Shetty, Anvay Sinkar, Aaryan Shinde. Originally developed in [ViyanShetty/memformer](https://github.com/ViyanShetty/memformer).

## Results

Test accuracy is exact-match on the next delta class, scored on the last 20% of each trace (a chronological split). The two baselines are computed on the same test windows: "most frequent" always predicts the most common delta from the training split, and "repeat last" predicts the previous delta again.

| Trace | Access pattern | Delta classes | MemFormer | Most frequent | Repeat last | ONNX latency (CPU) |
|---|---|---|---|---|---|---|
| matmul | regular, strided | 9 | **100.00%** | 24.99% | 0.00% | 93.6 µs |
| sort | mixed | 84 | **95.79%** | 4.35% | 0.00% | 94.0 µs |
| bfs | graph traversal | 125 | **94.68%** | 0.00% | 7.50% | 98.9 µs |
| hashtable | irregular | 10,826 | **72.85%** | 45.27% | 10.80% | 96.4 µs |
| ebpf_live | live kernel page faults | 1,524 | 75.24% | **76.08%** | 66.46% | ~95 µs |

Reading the table:

- **Pin traces:** the model clearly learns the structure. On hashtable it gets 72.9% against 45.3% for the best baseline.
- **ebpf_live:** the model does **not** beat always predicting the most common delta. The live page-fault stream is dominated by one delta, and these 43k samples are not enough for the model to learn more than that. Treat this row as a working pipeline (kernel → trace → model), not as a predictive result.
- **Latency:** one prediction at batch size 1 with ONNX Runtime on a laptop CPU, about 10,000 predictions per second.

## Model and size

- **Input:** 8 delta ids, plus the low 16 bits of each PC, embedded into 32 dimensions each.
- **Encoder:** 2 Transformer layers, 4 heads, d_model 64.
- **Output:** 16 logits, one bit each of the delta class id, trained with BCE.

The binary output shrinks the **output layer** from `64·n` to `64·16` weights. `results/compression_analysis.py` reports this as 338.8× for hashtable, using a simplified count of the core weights.

The whole model is a different story. About 2.1 M of its 2.2–2.5 M parameters sit in the PC embedding table (65,536 × 32), so the end-to-end parameter saving is 1.28× on hashtable and about 1× on the other traces. Each model is about 8.8–10.2 MB in FP32. The `.onnx` files are small only because ONNX stores the weights separately in `.onnx.data`.

Making the PC path smaller, by hashing PCs into fewer buckets or using a smaller embedding, is the obvious next step towards a hardware-sized model. INT8 dynamic quantization (`model/quantize.py`) only covers the Linear layers, so it barely changes the file size.

## Data

- **Pin traces** (`pintool/memtrace.cpp`): PC and effective address of every memory operand, for the first 2,000,000 memory operands of each benchmark run.
- **Preprocessing** (`data/parse_trace.py`): uses the first 200,000 records of each trace, takes address deltas, and drops deltas that occur fewer than 2 times. The CSVs in `data/` have 145k–191k rows.
- **ebpf_live** (`ebpf/collector.py`): 50,000 events from a kprobe on `handle_mm_fault`, giving the faulting address and kernel PC, across all processes on Linux 6.8.

Benchmarks:
- matmul: 1024×1024 matrix multiply
- sort: qsort on 1M integers
- bfs: BFS on a 50k-node random graph
- hashtable: open addressing, 80k entries

## eBPF prefetch daemon (prototype)

`ebpf/prefetch_daemon.py` runs the ebpf_live model on page faults as they happen and calls `madvise(MADV_WILLNEED)` on the predicted next page. Its "prefetch rate" is the share of faults for which a hint was issued, **not** a hit rate.

Known limitation: `madvise` acts on the calling process's own address space, but the faults come from other processes. So the hints don't reach the faulting process, and their return values are not checked. A working version needs `process_madvise` with a pidfd for each faulting PID, plus a measurement of fault count or runtime with and without the daemon.

## Layout

```
pintool/   memtrace.cpp        Intel Pin tool: PC + address trace
data/      parse_trace.py      trace → delta classes CSV;  verify.py;  *.csv datasets
model/     memformer.py        model
           dataset.py          sliding window, 80/20 chronological split
           train.py            training (5 epochs, Adam, BCE on 16 bits)
           export_onnx.py      ONNX export;   quantize.py   INT8 (Linear only)
           measure_latency.py  ONNX Runtime latency
           *_memformer.pt / .onnx / .onnx.data   trained models
results/   compression_analysis.py   output-layer compression
ebpf/      collector.py        live page-fault trace (needs root + BCC)
           prefetch_daemon.py  live inference + madvise hints (prototype)
```

## Quick start

```bash
git clone https://github.com/anvaysinkar/memformer
cd memformer
python3 -m venv venv && source venv/bin/activate
pip install torch numpy pandas tqdm scikit-learn onnx onnxruntime onnxscript

# collect a trace (Intel Pin on Linux)
~/pin/pin -t pintool/memtrace.so -o traces/myapp.out -- /your/program
python data/parse_trace.py traces/myapp.out          # -> data/myapp.csv

python model/train.py data/myapp.csv                 # -> model/myapp_memformer.pt
python model/export_onnx.py data/myapp.csv model/myapp_memformer.pt
python model/measure_latency.py model/myapp_memformer.onnx
```

## Reference

Srivastava et al., "Predicting Memory Accesses: The Road to Compact ML-driven Prefetcher", MEMSYS 2019. This is the origin of the binary output encoding. MemFormer uses a Transformer instead of an LSTM and feeds in the PC alongside the deltas.
