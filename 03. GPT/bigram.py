# INSTALLING LIBRARIES: 
import torch
import torch.nn as nn
import torch.nn.functional as F


# HYPERPARAMETERS:
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023)

# DOWNLOADING THE DATASET:
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


# READING THE DATASET AND INSPECTING:
with open("./input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# GETTING ALL THE UNIQUE CHARACTERS:
chars = sorted(list(set(text)))
vocab_size = len(chars)

# CHARACTER LEVEL ENCODING:
s2i = { ch:i for i, ch in enumerate(chars) }
i2s = { i:ch for i, ch in enumerate(chars) }
encode = lambda x: [s2i[c] for c in x]
decode = lambda y: ''.join([i2s[i] for i in y])


# ENCODING THE ENTIRE DATASET:
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# INITIALIZING BATCHES OF DATA:
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# IMPLEMENTATION OF BIGRAM LANGUAGE MODEL:
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    # idx and targets are both (B, T) tensors.
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



# IMPLEMENTATION OF THE MODEL:
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# OPTIMIZER OBJECT:
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# GENERAL TRAINING LOOP:
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
    
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# GENERATING TEXT:
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))