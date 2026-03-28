import onnxruntime as ort
import numpy as np, time, sys, os

path = sys.argv[1]
sess = ort.InferenceSession(path)

dummy = {
    'delta_seq': np.zeros((1, 8), dtype=np.int64),
    'pc_seq':    np.zeros((1, 8), dtype=np.int64)
}

for _ in range(200):
    sess.run(None, dummy)

N = 5000
t0 = time.perf_counter()
for _ in range(N):
    sess.run(None, dummy)
t1 = time.perf_counter()

avg_us = (t1 - t0) / N * 1e6
print(f"Model: {os.path.basename(path)}")
print(f"Avg inference latency: {avg_us:.2f} microseconds")
print(f"Throughput: {1e6/avg_us:.0f} predictions/second")
