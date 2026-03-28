import torch, os, sys
sys.path.insert(0, 'model')
from memformer import MemFormer
import pandas as pd

csv = sys.argv[1]
pt  = sys.argv[2]

df = pd.read_csv(csv)
vocab_size = df['delta_id'].nunique()

model = MemFormer(vocab_size)
model.load_state_dict(torch.load(pt, weights_only=True))
model.eval()

# only quantize Linear layers — Embedding quantization broken in torch 2.11
quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

q_path = pt.replace('.pt', '_int8.pt')
torch.save(quantized.state_dict(), q_path)

orig  = os.path.getsize(pt)
quant = os.path.getsize(q_path)
print(f"Original:    {orig/1024:.1f} KB")
print(f"Quantized:   {quant/1024:.1f} KB")
print(f"Compression: {orig/quant:.2f}x")
