import copy
import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

WEEK_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(WEEK_DIR)
DATA_PATH = os.path.join(STAGE_DIR, "data", "four-books.json")
MODEL_PATH = os.path.join(STAGE_DIR, "data", "transformer.pth")

CONTEXT = 128        # how many characters the model can look back on
D_MODEL = 128        # width of every vector inside the model
N_HEADS = 4          # attention heads; each head works in D_MODEL / N_HEADS = 32 dims
N_LAYERS = 4         # how many blocks are stacked
DROPOUT = 0.1
BATCH_SIZE = 64
STEPS = 3000         # upper limit; training stops earlier once validation loss stops improving
LEARNING_RATE = 3e-4
EVAL_EVERY = 200
PATIENCE = 4         # stop after this many evaluations without a better validation loss
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_text():
    chapters = json.load(open(DATA_PATH, encoding="utf-8"))
    return "\n".join(p for c in chapters for p in c["paragraphs"])   # "\n" separates paragraphs


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)      # one Linear makes q, k, v for all heads at once
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT, CONTEXT))  # 1 = may look, 0 = future (hidden)
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, length, _ = x.shape
        head_dim = D_MODEL // N_HEADS
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)                        # each (batch, length, D_MODEL)
        # split the last dim into heads, then move heads forward: (batch, heads, length, head_dim)
        q = q.view(batch, length, N_HEADS, head_dim).transpose(1, 2)
        k = k.view(batch, length, N_HEADS, head_dim).transpose(1, 2)
        v = v.view(batch, length, N_HEADS, head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / math.sqrt(head_dim)               # (batch, heads, length, length)
        scores = scores.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
        weights = F.softmax(scores, dim=3)                                  # each row sums to 1
        weights = self.dropout(weights)
        mixed = weights @ v                                                 # weighted average of values
        mixed = mixed.transpose(1, 2).reshape(batch, length, D_MODEL)      # join the heads back together
        return self.out(mixed)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = Attention()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))   # tokens talk to each other
        x = x + self.mlp(self.norm2(x))         # each token thinks on its own
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)                # 沒有位置向量,模型分不出「子曰」和「曰子」
        self.pos_emb = nn.Embedding(CONTEXT, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, ids):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.dropout(self.tok_emb(ids) + self.pos_emb(positions))    # (batch, length, D_MODEL)
        x = self.blocks(x)
        return self.head(self.norm(x))                                    # logits (batch, length, vocab)


def get_batch(ids):
    starts = torch.randint(len(ids) - CONTEXT - 1, (BATCH_SIZE,))
    x = torch.stack([ids[s:s + CONTEXT] for s in starts])
    y = torch.stack([ids[s + 1:s + CONTEXT + 1] for s in starts])        # same window, one step ahead
    return x.to(DEVICE), y.to(DEVICE)


def loss_on(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))   # all positions at once


def fixed_windows(ids):
    # every non-overlapping window of the validation text, so val loss is comparable between evaluations
    count = (len(ids) - 1) // CONTEXT
    x = torch.stack([ids[i * CONTEXT:(i + 1) * CONTEXT] for i in range(count)])
    y = torch.stack([ids[i * CONTEXT + 1:(i + 1) * CONTEXT + 1] for i in range(count)])
    return x.to(DEVICE), y.to(DEVICE)


def main():
    text = load_text()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text])
    split = int(len(ids) * 0.95)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"{len(text)} characters, vocab {len(vocab)}, train {len(train_ids)} / val {len(val_ids)}")

    torch.manual_seed(0)
    model = Transformer(len(vocab)).to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, training on {DEVICE}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    val_x, val_y = fixed_windows(val_ids)

    best_val, best_state, bad_checks = float("inf"), None, 0
    for step in range(1, STEPS + 1):
        loss = loss_on(model, *get_batch(train_ids))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0:
            with torch.inference_mode():
                model.eval()
                val_loss = loss_on(model, val_x, val_y).item()
                model.train()
            print(f"step {step}/{STEPS}  train loss {loss.item():.3f}  val loss {val_loss:.3f}")
            if val_loss < best_val:                       # keep the weights that generalise best
                best_val, best_state, bad_checks = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad_checks += 1
                if bad_checks == PATIENCE:                 # the model is memorising the text now
                    print(f"stopping: no improvement for {PATIENCE} evaluations")
                    break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"best val loss {best_val:.3f}")


if __name__ == "__main__":
    main()
