#!/usr/bin/env python3
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'model'))
os.chdir(ROOT)

from bcc import BPF
import ctypes, numpy as np, torch
from collections import deque
from memformer import MemFormer
import pandas as pd

LOOKBACK = 8
BITS = 16

# load model
csv  = 'data/ebpf_live.csv'
pt   = 'model/ebpf_live_memformer.pt'
df   = pd.read_csv(csv)
vocab_size = df['delta_id'].nunique()
model = MemFormer(vocab_size)
model.load_state_dict(torch.load(pt, weights_only=True))
model.eval()
print(f"Model loaded — vocab {vocab_size}")

# build vocab mapping from training data
from collections import Counter
raw_deltas = df['raw_delta'].values
delta_ids  = df['delta_id'].values
id2delta = {}
for rid, did in zip(raw_deltas, delta_ids):
    id2delta[int(did)] = int(rid)

# eBPF program
prog = r"""
#include <uapi/linux/ptrace.h>
struct event_t { u64 pc; u64 addr; u32 pid; };
BPF_PERF_OUTPUT(events);
int trace_page_fault(struct pt_regs *ctx, unsigned long address, unsigned int flags) {
    struct event_t e = {};
    e.pid  = bpf_get_current_pid_tgid() >> 32;
    e.pc   = PT_REGS_IP(ctx);
    e.addr = address;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}
"""

class Event(ctypes.Structure):
    _fields_ = [("pc", ctypes.c_uint64),
                ("addr", ctypes.c_uint64),
                ("pid",  ctypes.c_uint32)]

libc = ctypes.CDLL("libc.so.6", use_errno=True)
MADV_WILLNEED = 3
PAGE_SIZE = 4096

delta_window = deque(maxlen=LOOKBACK)
pc_window    = deque(maxlen=LOOKBACK)
last_addr    = None
prefetch_count = 0
access_count   = 0

def handle_event(cpu, data, size):
    global last_addr, prefetch_count, access_count
    e = b["events"].event(data)
    addr = e.addr & 0xFFFFFFFF
    pc   = e.pc   & 0xFFFF
    access_count += 1

    if last_addr is not None:
        delta = addr - last_addr
        # find closest delta_id in vocab
        closest = min(id2delta.keys(),
                      key=lambda k: abs(id2delta[k] - delta))
        delta_window.append(closest)
        pc_window.append(pc)

        if len(delta_window) == LOOKBACK:
            with torch.no_grad():
                x_d = torch.tensor([list(delta_window)], dtype=torch.long)
                x_p = torch.tensor([list(pc_window)],    dtype=torch.long)
                logits = model(x_d, x_p)[0]
                bits = (logits > 0).long().tolist()
                pred_id = sum(bits[b_] * (2**b_) for b_ in range(BITS))

            pred_delta = id2delta.get(pred_id, 0)
            prefetch_addr = (addr + pred_delta) & ~(PAGE_SIZE - 1)

            if prefetch_addr > 0:
                libc.madvise(
                    ctypes.c_void_p(prefetch_addr),
                    ctypes.c_size_t(PAGE_SIZE),
                    ctypes.c_int(MADV_WILLNEED)
                )
                prefetch_count += 1

    last_addr = addr

    if access_count % 1000 == 0:
        print(f"Accesses: {access_count} | Prefetches issued: {prefetch_count}", flush=True)
    if access_count >= 10000:
        print(f"\nDone. Issued {prefetch_count} prefetch hints from {access_count} accesses.")
        print(f"Prefetch rate: {prefetch_count/access_count*100:.1f}%")
        exit(0)

b = BPF(text=prog)
b.attach_kprobe(event="handle_mm_fault", fn_name="trace_page_fault")
print("MemFormer prefetcher running — intercepting kernel page faults...")
b["events"].open_perf_buffer(handle_event, page_cnt=64)
while True:
    b.perf_buffer_poll()
