import torch, sys, os
sys.path.insert(0, 'model')
from memformer import MemFormer
import pandas as pd

csv = sys.argv[1]
pt  = sys.argv[2]
df  = pd.read_csv(csv)
vocab_size = df['delta_id'].nunique()

model = MemFormer(vocab_size)
model.load_state_dict(torch.load(pt, weights_only=True))
model.eval()

dummy_delta = torch.zeros(1, 8, dtype=torch.long)
dummy_pc    = torch.zeros(1, 8, dtype=torch.long)

out_path = pt.replace('.pt', '.onnx')

torch.onnx.export(
    model,
    (dummy_delta, dummy_pc),
    out_path,
    input_names=['delta_seq', 'pc_seq'],
    output_names=['logits'],
    opset_version=14
)

print(f"Exported: {out_path}")
print(f"ONNX size: {os.path.getsize(out_path)/1024:.1f} KB")
