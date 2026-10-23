import torch
from torch.nn import functional as F

from transformer import CONTEXT, DEVICE, MODEL_PATH, Transformer, load_text

PROMPTS = ["子曰：", "孟子曰：", "君子", "大學之道", "天命"]
MAX_CHARS = 80       # safety cap; generation normally stops at the first 。？！
STOP_CHARS = "。？！"
OPEN_QUOTES, CLOSE_QUOTES = "「『“‘", "」』”’"   # a sentence inside a quote is not finished until the quote closes
TEMPERATURE = 0.8
TOP_K = 20


@torch.inference_mode()
def generate(model, itos, stoi, prompt):
    model.eval()
    ids = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE)
    open_quotes = sum(c in OPEN_QUOTES for c in prompt) - sum(c in CLOSE_QUOTES for c in prompt)
    for _ in range(MAX_CHARS):
        logits = model(ids[:, -CONTEXT:])[0, -1] / TEMPERATURE          # last position only
        top = torch.topk(logits, TOP_K)                                   # keep the TOP_K most likely characters
        probs = F.softmax(top.values, dim=0)
        next_id = top.indices[torch.multinomial(probs, 1)]
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        char = itos[next_id.item()]
        open_quotes += (char in OPEN_QUOTES) - (char in CLOSE_QUOTES)
        if open_quotes <= 0 and (char in STOP_CHARS or char in CLOSE_QUOTES):   # one complete sentence, quotes closed
            break
    return "".join(itos[i] for i in ids[0].tolist()).replace("\n", " ")


text = load_text()
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

model = Transformer(len(vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"loaded {MODEL_PATH} (vocab {len(vocab)}), device {DEVICE}")

while True:
    prompt = input("【提示詞】").strip()
    if prompt in ("q", "exit"):
        break
    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        print(f"這些字不在四書裡,模型沒看過:{''.join(unknown)}\n")
        continue
    for p in ([prompt] if prompt else PROMPTS):
        print(f"【生成】{generate(model, itos, stoi, p)}")
    print()
