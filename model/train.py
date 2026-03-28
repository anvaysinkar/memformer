import torch, sys, os
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MemDataset
from memformer import MemFormer
import pandas as pd

BITS = 16
EPOCHS = 5
BATCH = 256
LR = 1e-3

def bits_to_id(bits_tensor):
    bits = (bits_tensor > 0.5).long()
    ids = torch.zeros(bits.shape[0], dtype=torch.long)
    for b in range(BITS):
        ids += bits[:, b] * (2 ** b)
    return ids

if __name__ == "__main__":
    csv = sys.argv[1]
    df = pd.read_csv(csv)
    vocab_size = df['delta_id'].nunique()
    print(f"Vocab size: {vocab_size}")

    train_ds = MemDataset(csv, vocab_size, 'train')
    test_ds  = MemDataset(csv, vocab_size, 'test')
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, num_workers=0)

    model = MemFormer(vocab_size)
    print(f"Model params: {model.count_params():,}")

    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_d, x_p, target_bin, _ in train_dl:
            opt.zero_grad()
            out  = model(x_d, x_p)
            loss = loss_fn(out, target_bin)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_dl):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_d, x_p, target_bin, target_id in test_dl:
            out     = model(x_d, x_p)
            pred_id = bits_to_id(torch.sigmoid(out))
            correct += (pred_id == target_id).sum().item()
            total   += len(target_id)
    acc = correct / total * 100
    print(f"Test accuracy: {acc:.2f}%")

    name = os.path.basename(csv).replace('.csv','')
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), f"model/{name}_memformer.pt")
    print(f"Saved model/'{name}_memformer.pt'")
