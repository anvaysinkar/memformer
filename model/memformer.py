import torch
import torch.nn as nn

BITS = 16

class MemFormer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 pc_vocab=65536, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.delta_embed = nn.Embedding(vocab_size + 1, d_model // 2)
        self.pc_embed    = nn.Embedding(pc_vocab, d_model // 2)
        self.pos_enc     = nn.Embedding(64, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, BITS)

    def forward(self, x_delta, x_pc):
        B, T = x_delta.shape
        d_emb = self.delta_embed(x_delta)
        p_emb = self.pc_embed(x_pc)
        x = torch.cat([d_emb, p_emb], dim=-1)
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_enc(pos)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
